//! The write path, end to end: `ingest → extract → consolidate → index`
//! (`PLAN.md` §6).
//!
//! This is the orchestrator, and it owns one decision the individual stages
//! cannot: **what to do when a stage declines**. Every stage has a non-fatal
//! failure mode — extraction can be unparseable, consolidation can quarantine
//! or reject — and the only unacceptable behaviour is for any of them to
//! silently reduce the record count. So every outcome is counted in
//! [`WriteStats`], and the counts are the M3 deliverable: `records/unit,
//! tokens, wall time`.
//!
//! Episodes are written **before** extraction runs. Raw episodes are stored
//! losslessly (`06-consolidation-forgetting.md` §1) and derived records point
//! back at them, so if extraction fails on an episode we still hold the
//! material and can re-derive later. Extracting first and writing both at the
//! end would lose the episode whenever the model failed.

use std::time::Instant;

use futures_util::{stream, StreamExt};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::embed::Embedder;
use crate::error::Result;
use crate::llm::Llm;
use crate::model::delta::Delta;
use crate::model::record::{
    ActorId, EntityRef, MemoryRecord, Provenance, RecordKind, Salience, Scope, Trust, Validity,
};
use crate::store::ids::record_id;
use crate::store::ledger::Ledger;
use crate::store::qdrant::QdrantStore;

use super::consolidate::{Consolidator, Outcome, SourceTier};
use super::extract::{Candidate, ExtractOutcome, Extractor};
use super::index::Indexer;
use super::ingest::{segment, EpisodeDraft, SegmentConfig, Turn};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WriteStats {
    pub turns: usize,
    pub episodes: usize,
    pub candidates: usize,
    pub added: usize,
    pub updated: usize,
    pub deleted: usize,
    pub noop: usize,
    pub duplicates: usize,
    /// Extraction produced nothing usable, or the trust gate staged it.
    pub quarantined: usize,
    /// Contradicted a `Verified` fact and was refused (C8).
    pub rejected: usize,
    pub approx_tokens: usize,
    pub wall_ms: u128,
    pub extract_ms: u128,
    pub consolidate_ms: u128,
    pub index_ms: u128,
}

impl WriteStats {
    /// Records written per input unit — the M3 number.
    pub fn records_per_unit(&self, units: usize) -> f64 {
        if units == 0 {
            return 0.0;
        }
        (self.episodes + self.added + self.updated) as f64 / units as f64
    }

    pub fn merge(&mut self, other: &WriteStats) {
        self.turns += other.turns;
        self.episodes += other.episodes;
        self.candidates += other.candidates;
        self.added += other.added;
        self.updated += other.updated;
        self.deleted += other.deleted;
        self.noop += other.noop;
        self.duplicates += other.duplicates;
        self.quarantined += other.quarantined;
        self.rejected += other.rejected;
        self.approx_tokens += other.approx_tokens;
        self.wall_ms += other.wall_ms;
        self.extract_ms += other.extract_ms;
        self.consolidate_ms += other.consolidate_ms;
        self.index_ms += other.index_ms;
    }
}

pub struct WritePath<'a> {
    pub llm: &'a dyn Llm,
    pub embedder: &'a dyn Embedder,
    pub store: &'a QdrantStore,
    pub ledger: &'a Ledger,
    pub segment: SegmentConfig,
    pub source_tier: SourceTier,
    pub actor: ActorId,
    /// Emit a per-episode progress line on stderr. A corpus ingest is a
    /// multi-minute operation on a shared GPU; a silent one is impossible to
    /// distinguish from a hung one, and the first probe run had to be killed
    /// by a timeout precisely because it said nothing.
    pub progress: bool,
    /// How many candidate judgements to have in flight against the model.
    ///
    /// Bounded by the server's slot count: `llama-server -np N`. Beyond that
    /// requests queue and the extra concurrency only adds latency.
    pub concurrency: usize,
    /// Run `extract` + `consolidate`, or stop after storing raw episodes.
    ///
    /// **Off is the right answer for LME-V2-Small, and that is a measurement,
    /// not a shortcut.** Extraction costs one model call per episode.
    /// Measured on LoCoMo conv-26: 16,481 input tokens took 597 s end to end.
    /// LME-V2-Small is ~25M tokens — about 1,500x — which extrapolates to
    /// **~250 hours** on this card. That is not a budget problem to push
    /// through; it means the design is wrong for that corpus.
    ///
    /// And the evidence already said so. LME-V2's own result is that a
    /// coding-agent controller over **raw trajectory files** beats the same
    /// authors' extracted RAG memory by +16.3 / +13.1 points (`PLAN.md` §2
    /// finding 8), and §7.2 specifies `investigate` as search over the raw
    /// episode store. So LME-V2 is ingested episodically — segment, embed,
    /// index, losslessly — and the facts are found at read time by the agent
    /// rather than precomputed for 25M tokens of trajectory that no question
    /// will ever touch.
    ///
    /// LoCoMo keeps extraction on: it is conversational recall, the corpus is
    /// small, and §2 finding 5's 4-op delta is exactly what its
    /// knowledge-update questions test.
    pub extract_facts: bool,
}

impl<'a> WritePath<'a> {
    pub fn new(
        llm: &'a dyn Llm,
        embedder: &'a dyn Embedder,
        store: &'a QdrantStore,
        ledger: &'a Ledger,
    ) -> Self {
        Self {
            llm,
            embedder,
            store,
            ledger,
            segment: SegmentConfig::default(),
            source_tier: SourceTier::Asserted,
            actor: ActorId::new("myelin"),
            progress: false,
            concurrency: 4,
            extract_facts: true,
        }
    }

    /// R2: ingest granularity is one whole unit (a LoCoMo conversation, an
    /// LME-V2 trajectory). Segmentation into episodes is ours to do, not the
    /// caller's.
    pub async fn insert(&self, scope: &Scope, turns: &[Turn]) -> Result<WriteStats> {
        let started = Instant::now();
        let mut stats = WriteStats {
            turns: turns.len(),
            ..Default::default()
        };

        let drafts = segment(turns, &self.segment);
        stats.episodes = drafts.len();
        stats.approx_tokens = drafts.iter().map(|d| d.approx_tokens).sum();

        // 1. Episodes, losslessly, first.
        let episodes = self.write_episodes(scope, &drafts).await?;
        let t_index = Instant::now();
        self.indexer().index(&episodes).await?;
        stats.index_ms += t_index.elapsed().as_millis();

        // Episodic-only corpora stop here. See `WritePath::extract_facts`.
        if !self.extract_facts {
            stats.wall_ms = started.elapsed().as_millis();
            return Ok(stats);
        }

        // 2. Extract every episode CONCURRENTLY, then consolidate IN ORDER.
        //
        // The asymmetry is load-bearing. Extraction is a pure function of one
        // episode, so order does not matter and it can saturate the server's
        // slots. Consolidation is not: episode N's candidates must be able to
        // dedup against and supersede facts episode N-1 wrote, so it stays
        // ordered. Measured on conv-26, extraction was 204 s of 597 s purely
        // because 43 independent calls queued behind each other.
        let extractor = Extractor::new(self.llm);
        let consolidator = Consolidator::new(self.llm);

        let t_extract = Instant::now();
        let extracted: Vec<Result<ExtractOutcome>> = stream::iter(episodes.iter())
            .map(|episode| {
                let extractor = &extractor;
                async move { extractor.extract(episode).await }
            })
            .buffered(self.concurrency.max(1))
            .collect()
            .await;
        stats.extract_ms += t_extract.elapsed().as_millis();

        for (n, (episode, outcome)) in episodes.iter().zip(extracted).enumerate() {
            if self.progress && n % 10 == 0 {
                eprintln!(
                    "      episode {n}/{} add={} dup={} noop={} {:.0}s",
                    episodes.len(),
                    stats.added,
                    stats.duplicates,
                    stats.noop,
                    started.elapsed().as_secs_f64()
                );
            }
            let outcome = outcome?;

            let candidates = match &outcome {
                ExtractOutcome::Extracted(e) => e.candidates.clone(),
                ExtractOutcome::Quarantined { reason } => {
                    self.ledger
                        .log(
                            "extract_quarantine",
                            Some(episode.id),
                            &self.actor,
                            reason,
                            serde_json::json!({}),
                        )
                        .await?;
                    stats.quarantined += 1;
                    continue;
                }
            };
            if candidates.is_empty() {
                continue;
            }
            stats.candidates += candidates.len();

            // One embedding round trip for the whole episode.
            let texts: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
            let t_embed = Instant::now();
            let vectors = self.embedder.embed(&texts).await?;
            stats.index_ms += t_embed.elapsed().as_millis();

            let mut derived: Vec<MemoryRecord> = Vec::new();
            // Judge candidates concurrently, then apply sequentially.
            //
            // This is semantically identical to doing it one at a time:
            // neighbours come from Qdrant, which is not updated until the end
            // of the episode, so no candidate can observe a sibling either
            // way. It is worth doing because consolidation was 526 s of a
            // 768 s conversation — 68% — all of it waiting on one model slot.
            // Writes stay serialized so SQLite never contends with itself.
            let t1 = Instant::now();
            let pairs: Vec<(Candidate, Vec<f32>)> =
                candidates.iter().cloned().zip(vectors).collect();
            let judged: Vec<Result<(MemoryRecord, Outcome)>> = stream::iter(pairs)
                .map(|(candidate, vector)| {
                    let consolidator = &consolidator;
                    async move {
                        self.judge_one(scope, episode, &candidate, vector, consolidator)
                            .await
                    }
                })
                .buffered(self.concurrency.max(1))
                .collect()
                .await;
            stats.consolidate_ms += t1.elapsed().as_millis();

            for judgement in judged {
                let (record, outcome) = judgement?;
                if let Some(written) = self.apply_outcome(&record, outcome, &mut stats).await? {
                    derived.push(written);
                }
            }

            if !derived.is_empty() {
                let t2 = Instant::now();
                self.indexer().index(&derived).await?;
                stats.index_ms += t2.elapsed().as_millis();
            }
        }

        stats.wall_ms = started.elapsed().as_millis();
        Ok(stats)
    }

    fn indexer(&self) -> Indexer<'_> {
        Indexer::new(self.embedder, self.store, self.ledger)
    }

    async fn write_episodes(
        &self,
        scope: &Scope,
        drafts: &[EpisodeDraft],
    ) -> Result<Vec<MemoryRecord>> {
        let mut out = Vec::with_capacity(drafts.len());
        for draft in drafts {
            let record = draft.to_record(scope, &self.actor, &self.actor);
            // Re-ingesting the same corpus hits the same v5 id. An existing
            // episode is not an error; it is the idempotence the id scheme
            // exists to provide.
            if self.ledger.get(record.id).await?.is_none() {
                self.ledger
                    .apply(
                        &Delta::Add {
                            record: Box::new(record.clone()),
                        },
                        &self.actor,
                    )
                    .await?;
            }
            out.push(record);
        }
        Ok(out)
    }

    /// Read-only half: decide what should happen to one candidate.
    ///
    /// Touches no writable state, which is what makes it safe to run many at
    /// once against the model.
    async fn judge_one(
        &self,
        scope: &Scope,
        episode: &MemoryRecord,
        candidate: &Candidate,
        vector: Vec<f32>,
        consolidator: &Consolidator<'_>,
    ) -> Result<(MemoryRecord, Outcome)> {
        let record = self.candidate_record(scope, episode, candidate);

        // Neighbours in scope, for the dedup and contradiction gates.
        let (neighbours, similarities) = self
            .neighbours_for(scope, &record, vector, consolidator.config.neighbours_k)
            .await?;

        let outcome = consolidator
            .consolidate(
                candidate,
                &record,
                &neighbours,
                &similarities,
                self.source_tier,
            )
            .await?;
        Ok((record, outcome))
    }

    /// Write half: apply one decision. Sequential by construction.
    ///
    /// Returns the record that needs indexing, if the delta wrote one.
    async fn apply_outcome(
        &self,
        record: &MemoryRecord,
        outcome: Outcome,
        stats: &mut WriteStats,
    ) -> Result<Option<MemoryRecord>> {
        match outcome {
            Outcome::Apply { delta, reason } => {
                let applied = self.ledger.apply(&delta, &self.actor).await;
                match (&delta, applied) {
                    (Delta::Add { record }, Ok(_)) => {
                        stats.added += 1;
                        Ok(Some((**record).clone()))
                    }
                    (Delta::Update { replacement, .. }, Ok(_)) => {
                        stats.updated += 1;
                        Ok(Some((**replacement).clone()))
                    }
                    (Delta::Delete { .. }, Ok(_)) => {
                        stats.deleted += 1;
                        Ok(None)
                    }
                    (Delta::Noop { .. }, Ok(_)) => {
                        stats.noop += 1;
                        Ok(None)
                    }
                    // I4 can legitimately refuse a semantic record whose
                    // lineage does not resolve. Count it rather than aborting
                    // the corpus.
                    (_, Err(e)) => {
                        self.ledger
                            .log(
                                "consolidate_rejected",
                                Some(record.id),
                                &self.actor,
                                &format!("{reason}: {e}"),
                                serde_json::json!({}),
                            )
                            .await?;
                        stats.rejected += 1;
                        Ok(None)
                    }
                }
            }
            Outcome::Duplicate { existing } => {
                stats.duplicates += 1;
                self.ledger
                    .log(
                        "dedup",
                        Some(existing),
                        &self.actor,
                        "candidate duplicates an existing record",
                        serde_json::json!({}),
                    )
                    .await?;
                Ok(None)
            }
            Outcome::Quarantined { reason, .. } => {
                stats.quarantined += 1;
                self.ledger.quarantine(record, &reason).await?;
                Ok(None)
            }
            Outcome::Rejected {
                conflicts_with,
                reason,
            } => {
                stats.rejected += 1;
                self.ledger
                    .log(
                        "contradiction_rejected",
                        Some(conflicts_with),
                        &self.actor,
                        &reason,
                        serde_json::json!({ "candidate": record.text }),
                    )
                    .await?;
                Ok(None)
            }
        }
    }

    fn candidate_record(
        &self,
        scope: &Scope,
        episode: &MemoryRecord,
        candidate: &Candidate,
    ) -> MemoryRecord {
        let now = Utc::now();
        MemoryRecord {
            id: record_id(scope, &format!("fact\u{1f}{}", candidate.text)),
            kind: candidate.kind.record_kind(),
            scope: scope.clone(),
            text: candidate.text.clone(),
            entities: candidate
                .entities
                .iter()
                .map(|e| EntityRef::new(e.clone()))
                .collect(),
            validity: Validity {
                // No date from the model means inherit the episode's. Guessing
                // would poison the bi-temporal reasoning that answers
                // knowledge-update questions.
                t_valid: candidate.t_valid.unwrap_or(episode.validity.t_valid),
                t_invalid: None,
                t_ingested: now,
                t_expired: None,
            },
            provenance: Provenance {
                source: episode.provenance.source.clone(),
                contributed_by: episode.provenance.contributed_by.clone(),
                written_by: self.actor.clone(),
                // I4: a semantic record must name what it was abstracted
                // from, and the episode is it.
                derived_from: vec![episode.id],
            },
            trust: Trust::asserted(),
            salience: Salience::default(),
            links: Vec::new(),
        }
    }

    /// In-scope neighbours nearest the candidate, with cosine similarities.
    ///
    /// Qdrant does the search; the ledger then re-checks admissibility.
    ///
    /// Both halves are needed. Qdrant is the only thing that can find the
    /// nearest few without embedding the whole pool — the version that did
    /// that managed 43 episodes in 900 s. But its payload is a projection
    /// that can drift, so the authoritative liveness, quarantine and
    /// provenance checks still come from [`Ledger::get`]; a record Qdrant
    /// returns but the ledger will not materialise is skipped rather than
    /// shown to the gates.
    async fn neighbours_for(
        &self,
        scope: &Scope,
        candidate: &MemoryRecord,
        vector: Vec<f32>,
        k: usize,
    ) -> Result<(Vec<MemoryRecord>, Vec<f32>)> {
        // Over-fetch: some hits are episodes or fail the ledger re-check.
        let hits = self
            .store
            .search_dense(
                vector,
                &scope.tenant,
                &scope.namespace,
                (k * 4).max(16) as u64,
            )
            .await?;

        let now = Utc::now();
        let mut records = Vec::with_capacity(k);
        let mut sims = Vec::with_capacity(k);
        for (id, score) in hits {
            if id == candidate.id || records.len() >= k {
                continue;
            }
            let Some(record) = self.ledger.get(id).await? else {
                continue;
            };
            // Episodes are raw material, not facts to consolidate against.
            if record.kind == RecordKind::Episodic || !record.is_admissible_at(now) {
                continue;
            }
            records.push(record);
            sims.push(score);
        }
        Ok((records, sims))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_per_unit_counts_episodes_and_new_records() {
        let stats = WriteStats {
            episodes: 10,
            added: 25,
            updated: 5,
            noop: 100,
            duplicates: 50,
            ..Default::default()
        };
        // noop and duplicate writes produced no record, so they must not
        // inflate the density figure M3 reports.
        assert!((stats.records_per_unit(10) - 4.0).abs() < 1e-9);
        assert_eq!(stats.records_per_unit(0), 0.0, "must not divide by zero");
    }

    #[test]
    fn merge_is_additive_across_units() {
        let mut a = WriteStats {
            episodes: 3,
            added: 7,
            wall_ms: 100,
            ..Default::default()
        };
        let b = WriteStats {
            episodes: 4,
            added: 2,
            wall_ms: 50,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.episodes, 7);
        assert_eq!(a.added, 9);
        assert_eq!(a.wall_ms, 150);
    }
}
