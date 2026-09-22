//! Rebuild a Qdrant collection from the ledger.
//!
//! # Why this exists
//!
//! The ledger is the system of record; the vector index is a **derived
//! cache**. That was always the design, but nothing enforced it, and the
//! difference only became visible when all five `myelin_*` collections were
//! deleted through the Qdrant dashboard while a benchmark was running.
//!
//! The ledgers were untouched — 162,181 live LongMemEval_S records, integrity
//! `ok` — and yet there was no way to put them back. `build` is not that way:
//! its resume guard is the `unit_complete` **audit event**, which lives in the
//! ledger, so re-running it against a surviving ledger skips every unit and
//! yields an empty collection. Deleting the ledger to force a real rebuild
//! would re-run hours of LLM extraction and mint *different* records, because
//! extraction is not deterministic — silently invalidating comparability with
//! every measurement ever published.
//!
//! So: embed-only, from the rows that are already there. No reader, no
//! extraction, no new records, ids preserved exactly. Historical runs stay
//! comparable because the records they were measured against are the same
//! records.
//!
//! # What it is not
//!
//! Not a repair for a *divergent* index — that is
//! [`myelin_core::store::reconcile`], which reports drift. This assumes the
//! collection is missing or stale and rewrites it wholesale.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::Utc;
use myelin_core::config::MyelinConfig;
use myelin_core::embed::remote::RemoteEmbedder;
use myelin_core::pipeline::index::Indexer;
use myelin_core::store::{ledger::Ledger, qdrant::QdrantStore};

/// How many records are read from SQLite per page.
///
/// Independent of the embedder's own batch size: this bounds how much of the
/// ledger is resident at once, and `Indexer` re-chunks within it.
const PAGE: i64 = 512;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReindexReport {
    /// Live records the ledger offered.
    pub records: usize,
    /// Records embedded and upserted.
    pub indexed: usize,
    /// What the ledger says the index should hold, read before the walk.
    pub expected: usize,
}

impl ReindexReport {
    /// Every live record reached the index.
    ///
    /// The check that matters: a rebuild that silently indexed nothing —
    /// which is exactly what `build` does against a surviving ledger — must
    /// not be reported as success.
    pub fn is_complete(&self) -> bool {
        self.indexed == self.expected && self.records == self.expected
    }
}

/// Embed every live record in `ledger_path` into `collection`.
///
/// `limit` stops early, for a throughput probe before committing to a long
/// embedder window.
pub async fn reindex(
    cfg: &MyelinConfig,
    ledger_path: &Path,
    collection: &str,
    limit: Option<usize>,
) -> Result<ReindexReport> {
    anyhow::ensure!(
        collection.starts_with("myelin_"),
        "refusing to write to {collection}: the production collections on `big` are off limits"
    );

    let embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)
        .context("embedder client")?;

    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    store
        .ensure_collection(cfg.embed.dim, false)
        .await
        .context("ensure collection")?;

    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;

    // `now` is fixed for the whole walk. Reading it per page would let a
    // record expire mid-rebuild and leave the index short by one, with
    // nothing to show why.
    let now = Utc::now();
    let expected = ledger.count_live(now).await.context("count live")? as usize;
    let target = limit.unwrap_or(expected);

    eprintln!(
        "reindex {} -> {collection}: {expected} live records{}",
        ledger_path.display(),
        limit.map_or(String::new(), |n| format!(" (stopping after {n})"))
    );

    let indexer = Indexer::new(&embedder, &store, &ledger);
    let mut report = ReindexReport {
        expected,
        ..Default::default()
    };
    let started = Instant::now();
    let mut after = 0i64;

    loop {
        let page = ledger
            .live_records_page(now, after, PAGE)
            .await
            .context("read ledger page")?;
        if page.is_empty() {
            break;
        }

        after = page.last().map(|(rid, _)| *rid).unwrap_or(after);
        let mut records: Vec<_> = page.into_iter().map(|(_, r)| r).collect();
        if report.records + records.len() > target {
            records.truncate(target - report.records);
        }
        report.records += records.len();

        let stats = indexer.index(&records).await.context("embed and upsert")?;
        report.indexed += stats.records;

        let elapsed = started.elapsed().as_secs_f64();
        let rate = report.indexed as f64 / elapsed.max(f64::EPSILON);
        eprintln!(
            "  {:>7}/{target} indexed  {rate:.0} rec/s  eta {:.0}s",
            report.indexed,
            (target.saturating_sub(report.indexed)) as f64 / rate.max(f64::EPSILON)
        );

        if report.records >= target {
            break;
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rebuild_that_indexed_nothing_is_not_complete() {
        // The failure this whole module exists to prevent: `build` against a
        // surviving ledger skips every unit, exits 0, and leaves an empty
        // collection behind.
        let empty = ReindexReport {
            records: 0,
            indexed: 0,
            expected: 162_181,
        };
        assert!(!empty.is_complete());
    }

    #[test]
    fn a_short_rebuild_is_not_complete() {
        // Losing pages silently is the other way to get a wrong index.
        let short = ReindexReport {
            records: 162_181,
            indexed: 162_000,
            expected: 162_181,
        };
        assert!(!short.is_complete());
    }

    #[test]
    fn indexing_every_live_record_is_complete() {
        let whole = ReindexReport {
            records: 4_875,
            indexed: 4_875,
            expected: 4_875,
        };
        assert!(whole.is_complete());
    }
}
