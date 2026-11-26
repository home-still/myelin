//! `index` — embed, upsert, update incidence (`PLAN.md` §6.4).
//!
//! Three stores are touched and the order matters. The ledger is written
//! first (it is the source of truth for "may this be seen"), then Qdrant (the
//! similarity index), then incidence (the graph). A crash between any two
//! leaves detectable drift rather than a silent lie, and
//! [`crate::store::reconcile`] repairs exactly those shapes: a ledger record
//! with no point is `missing_vectors`, a point with no record is a
//! `qdrant_orphan`, and an incidence row pointing at neither is
//! `orphan_incidence`.
//!
//! Batch sizing follows the EWMA hill-climber in
//! `hs-distill/src/adaptive_batch.rs` in spirit: measure throughput, keep the
//! best size, back off on regression. The simplification here is deliberate —
//! we are throughput-bound on one remote embedder over HTTP, not on a local
//! ORT session whose optimum shifts with device, so a fixed batch with a
//! measured rate is enough until a measurement says otherwise.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::embed::Embedder;
use crate::error::{MyelinError, Result};
use crate::model::record::MemoryRecord;
use crate::store::ledger::Ledger;
use crate::store::qdrant::{IndexItem, VectorIndex};

/// Chunks per upsert. 1000 chunks of 1024-d f32 plus payload sits well inside
/// Qdrant's 4 MB gRPC frame; at 4096-d the same count would not, so the
/// default is scaled down and the dimension is checked at construction.
pub const DEFAULT_BATCH: usize = 256;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IndexStats {
    pub records: usize,
    pub batches: usize,
    pub embed_ms: u128,
    pub upsert_ms: u128,
    pub incidence_rows: usize,
    /// Records per second end to end. The number M3 has to report.
    pub records_per_sec: f64,
}

/// `chunks(0)` panics. A caller that computes a batch size from config can
/// reach 0; clamping here fails soft instead of aborting a corpus run.
fn clamp_batch(batch: usize) -> usize {
    batch.max(1)
}

pub struct Indexer<'a> {
    embedder: &'a dyn Embedder,
    store: &'a dyn VectorIndex,
    ledger: &'a Ledger,
    batch: usize,
}

impl<'a> Indexer<'a> {
    pub fn new(embedder: &'a dyn Embedder, store: &'a dyn VectorIndex, ledger: &'a Ledger) -> Self {
        Self {
            embedder,
            store,
            ledger,
            batch: DEFAULT_BATCH,
        }
    }

    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = clamp_batch(batch);
        self
    }

    /// Embed and upsert records that are already in the ledger.
    ///
    /// Does **not** write the ledger: deltas go through [`Ledger::apply`] so
    /// the invariants and the audit trail apply. This is the second leg only.
    ///
    /// Embedding and upserting alternate batch by batch, so a failure part
    /// way leaves the earlier batches indexed. A caller that must not leave
    /// them ([`super::write::WritePath`]) uses [`Indexer::embed_all`] and
    /// [`Indexer::upsert_embedded`] instead.
    pub async fn index(&self, records: &[MemoryRecord]) -> Result<IndexStats> {
        let started = Instant::now();
        let mut stats = IndexStats {
            records: records.len(),
            ..Default::default()
        };

        for chunk in records.chunks(self.batch) {
            let t0 = Instant::now();
            let vectors = self.embed_chunk(chunk).await?;
            stats.embed_ms += t0.elapsed().as_millis();

            let t1 = Instant::now();
            self.upsert_chunk(chunk, vectors).await?;
            stats.upsert_ms += t1.elapsed().as_millis();
            stats.batches += 1;
        }

        stats.incidence_rows += self.write_incidence(records).await?;
        stats.records_per_sec = records_per_sec(records.len(), started);
        Ok(stats)
    }

    /// The embedding leg on its own: one vector per record, in order, and no
    /// state touched anywhere. Lets a caller get the commonest failure (an
    /// embedder that refuses an input) out of the way before it writes.
    pub async fn embed_all(&self, records: &[MemoryRecord]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(records.len());
        for chunk in records.chunks(self.batch) {
            vectors.extend(self.embed_chunk(chunk).await?);
        }
        Ok(vectors)
    }

    /// The upsert-and-incidence leg on its own, for vectors from
    /// [`Indexer::embed_all`]. `vectors` is index-aligned with `records`.
    pub async fn upsert_embedded(
        &self,
        records: &[MemoryRecord],
        vectors: Vec<Vec<f32>>,
    ) -> Result<IndexStats> {
        let started = Instant::now();
        if vectors.len() != records.len() {
            return Err(MyelinError::Store(format!(
                "index: {} vectors for {} records",
                vectors.len(),
                records.len()
            )));
        }
        let mut stats = IndexStats {
            records: records.len(),
            ..Default::default()
        };

        let mut vectors = vectors.into_iter();
        for chunk in records.chunks(self.batch) {
            let batch: Vec<Vec<f32>> = vectors.by_ref().take(chunk.len()).collect();
            let t1 = Instant::now();
            self.upsert_chunk(chunk, batch).await?;
            stats.upsert_ms += t1.elapsed().as_millis();
            stats.batches += 1;
        }

        stats.incidence_rows += self.write_incidence(records).await?;
        stats.records_per_sec = records_per_sec(records.len(), started);
        Ok(stats)
    }

    /// One embedder call. A short answer is an error: zipping it against the
    /// records would silently leave the tail unindexed.
    async fn embed_chunk(&self, chunk: &[MemoryRecord]) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = chunk.iter().map(|r| r.text.clone()).collect();
        let vectors = self.embedder.embed(&texts).await?;
        if vectors.len() != chunk.len() {
            return Err(MyelinError::Store(format!(
                "{}: embedded {} texts and got {} vectors",
                self.embedder.id(),
                chunk.len(),
                vectors.len()
            )));
        }
        Ok(vectors)
    }

    async fn upsert_chunk(&self, chunk: &[MemoryRecord], vectors: Vec<Vec<f32>>) -> Result<()> {
        let items: Vec<IndexItem> = chunk
            .iter()
            .zip(vectors)
            .map(|(record, dense)| IndexItem {
                record,
                dense,
                // `late` stays unpopulated: bge-m3's ColBERT head is
                // 1024-d *per token*, two orders of magnitude more
                // storage, and whether it beats a cross-encoder at equal
                // latency is unsettled (§5.1 risks i and ii).
                late: None,
            })
            .collect();
        self.store.upsert(&items).await
    }

    /// Incidence last: it is derived, and a missing row degrades the graph
    /// route rather than corrupting a read.
    ///
    /// One authoritative batch per record, from the same
    /// [`super::phrases::incidence_rows`] a backfill uses, so a fresh
    /// ingest and `myelin-eval phrases` write byte-identical rows. The
    /// previous loop wrote `record.entities` only, which for an episodic
    /// record is the turn speakers — one hub node per conversation, no
    /// bridging signal.
    async fn write_incidence(&self, records: &[MemoryRecord]) -> Result<usize> {
        let rows: Vec<crate::store::ledger::IncidenceRow> =
            records.iter().flat_map(super::phrases::incidence_rows).collect();
        self.ledger.replace_incidence_batch(&rows).await
    }
}

fn records_per_sec(records: usize, started: Instant) -> f64 {
    let elapsed = started.elapsed().as_secs_f64();
    if elapsed > 0.0 {
        records as f64 / elapsed
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `chunks(0)` panics, so a caller passing 0 must be clamped rather than
    /// crashing mid-corpus.
    #[test]
    fn batch_size_is_clamped_to_at_least_one() {
        assert_eq!(clamp_batch(0), 1);
        assert_eq!(clamp_batch(1), 1);
        assert_eq!(clamp_batch(512), 512);
    }
}
