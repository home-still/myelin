//! `recall` — the fast read path (`PLAN.md` §7.1).
//!
//! ```text
//! scope-filter  → payload predicates, applied BEFORE ranking
//! retrieve      → dense + lex, one round trip, two ranked lists
//! fuse          → RRF in Rust at k = 60 (Qdrant's own is k = 1, §5.2)
//! rerank        → cross-encoder over the fused head        (§2 finding 2)
//! compose       → budgeted, bookended, deduped, top-k      (§7.3)
//! ```
//!
//! Scope predicates run first because post-filtering spends the probe budget
//! on inadmissible memories: ShardMemo measures **+2.9 / +3.1 F1** at S=20/80
//! for masking before routing rather than after (§2 finding 3). Here that is
//! free — the predicates are Qdrant payload indexes created in
//! [`crate::store::qdrant::QdrantStore::ensure_collection`].
//!
//! [`Channels`] exists so the M4 ablation runs the *same code path* for
//! BM25-only, dense-only and hybrid. A baseline measured by different code is
//! not a baseline.

use serde::{Deserialize, Serialize};

use crate::embed::Embedder;
use crate::error::Result;
use crate::model::evidence::EvidenceSet;
use crate::model::query::Recall;
use crate::model::record::RecordKind;
use crate::rerank::Reranker;
use crate::store::ledger::Ledger;
use crate::store::qdrant::QdrantStore;

use super::compose::{compose, ComposeConfig, Ranked};
use super::fuse::{rrf, RankedList, DEFAULT_RRF_K};

/// Which retrieval channels participate. The ablation axis of M4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channels {
    /// Dense only — the arm MemPro says costs 2.36 points to remove.
    Dense,
    /// BM25 only — the arm MemPro says costs **12.68** points to remove,
    /// which is why it is a first-class channel and not a fallback.
    Lex,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrieveConfig {
    /// Per-channel candidate depth before fusion.
    pub prefetch_limit: u64,
    /// Fusion constant. 60 is Cormack's; 1 is what Qdrant does server-side.
    pub rrf_k: f32,
    pub channels: Channels,
    /// How many fused candidates the reranker sees. Reranking is the
    /// highest-leverage stage (MS MARCO MRR@10 18.7 → 36.5, §2 finding 2), so
    /// it gets a deeper pool than `compose` will finally emit.
    pub rerank_depth: usize,
    pub compose: ComposeConfig,
}

impl Default for RetrieveConfig {
    fn default() -> Self {
        Self {
            prefetch_limit: 50,
            rrf_k: DEFAULT_RRF_K,
            channels: Channels::Hybrid,
            rerank_depth: 25,
            compose: ComposeConfig::default(),
        }
    }
}

/// What a single `recall` did, for the latency and ablation tables.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RecallTrace {
    pub dense_hits: usize,
    pub lex_hits: usize,
    pub fused: usize,
    pub reranked: usize,
    pub admitted: usize,
    pub embed_ms: u128,
    pub search_ms: u128,
    pub rerank_ms: u128,
    pub total_ms: u128,
}

pub struct Retriever<'a> {
    pub embedder: &'a dyn Embedder,
    pub store: &'a QdrantStore,
    pub ledger: &'a Ledger,
    pub reranker: Option<&'a dyn Reranker>,
    pub config: RetrieveConfig,
}

impl<'a> Retriever<'a> {
    pub fn new(embedder: &'a dyn Embedder, store: &'a QdrantStore, ledger: &'a Ledger) -> Self {
        Self {
            embedder,
            store,
            ledger,
            reranker: None,
            config: RetrieveConfig::default(),
        }
    }

    pub fn with_config(mut self, config: RetrieveConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_reranker(mut self, reranker: &'a dyn Reranker) -> Self {
        self.reranker = Some(reranker);
        self
    }

    pub async fn recall(&self, query: &Recall) -> Result<(EvidenceSet, RecallTrace)> {
        let started = std::time::Instant::now();
        let mut trace = RecallTrace::default();

        // Dense is needed for the dense channel and for compose's
        // near-duplicate suppression; skip it entirely for the lex-only arm
        // so that arm's latency is honest.
        let t0 = std::time::Instant::now();
        let dense = if self.config.channels == Channels::Lex {
            Vec::new()
        } else {
            self.embedder
                .embed(std::slice::from_ref(&query.text))
                .await?
                .pop()
                .unwrap_or_default()
        };
        trace.embed_ms = t0.elapsed().as_millis();

        let kinds: Vec<&str> = query
            .kinds
            .as_ref()
            .map(|ks| ks.iter().map(|k| k.as_str()).collect())
            .unwrap_or_default();

        let t1 = std::time::Instant::now();
        let lists = self
            .store
            .hybrid_search(
                dense.clone(),
                &query.text,
                &query.scope.tenant,
                query.scope.namespace.as_deref(),
                &kinds,
                self.config.prefetch_limit,
            )
            .await?;
        trace.search_ms = t1.elapsed().as_millis();
        trace.dense_hits = lists.dense.len();
        trace.lex_hits = lists.lex.len();

        // Fuse the selected channels. A single-channel arm still goes through
        // `rrf` so the ablation compares ranking, not plumbing.
        let mut ranked_lists = Vec::new();
        if matches!(self.config.channels, Channels::Dense | Channels::Hybrid) {
            ranked_lists.push(RankedList::new(
                "dense",
                lists.dense.iter().map(|h| h.id).collect(),
            ));
        }
        if matches!(self.config.channels, Channels::Lex | Channels::Hybrid) {
            ranked_lists.push(RankedList::new(
                "lex",
                lists.lex.iter().map(|h| h.id).collect(),
            ));
        }
        let fused = rrf(&ranked_lists, self.config.rrf_k);
        trace.fused = fused.len();

        // Text comes from the payload, so the head of the list costs no
        // SQLite round trip.
        let mut text_by_id = std::collections::HashMap::new();
        for hit in lists.dense.iter().chain(lists.lex.iter()) {
            text_by_id.entry(hit.id).or_insert_with(|| hit.text.clone());
        }

        let head: Vec<(uuid::Uuid, f32)> =
            fused.into_iter().take(self.config.rerank_depth).collect();

        // Rerank before the ledger check: reranking is the expensive stage
        // and there is no point paying it for records the ledger will refuse.
        // But the ledger check is cheap and local, so it runs first.
        let now = chrono::Utc::now();
        let mut admissible: Vec<(uuid::Uuid, f32, String)> = Vec::new();
        for (id, score) in head {
            let Some(record) = self.ledger.get(id).await? else {
                // In Qdrant, absent from the ledger: drift, not evidence.
                continue;
            };
            // I2 is enforced by `Ledger::get` failing to materialise a record
            // with no source; I3 and validity are re-checked here because the
            // Qdrant payload is a projection that can lag.
            if record.kind == RecordKind::Episodic && query.kinds.is_none() {
                // Episodes are raw material; `investigate` reads them
                // explicitly, `recall` composes facts.
                continue;
            }
            if !record.is_admissible_at(now) {
                continue;
            }
            let text = text_by_id
                .get(&id)
                .cloned()
                .unwrap_or_else(|| record.text.clone());
            admissible.push((id, score, text));
        }
        trace.admitted = admissible.len();

        let t2 = std::time::Instant::now();
        if let Some(reranker) = self.reranker {
            let docs: Vec<String> = admissible.iter().map(|(_, _, t)| t.clone()).collect();
            let scores = reranker.rerank(&query.text, &docs).await?;
            for (slot, score) in admissible.iter_mut().zip(scores) {
                slot.1 = score;
            }
            admissible.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            trace.reranked = admissible.len();
        }
        trace.rerank_ms = t2.elapsed().as_millis();

        let mut ranked = Vec::with_capacity(admissible.len());
        for (id, score, _) in admissible.into_iter().take(self.config.compose.k * 3) {
            if let Some(record) = self.ledger.get(id).await? {
                ranked.push(Ranked {
                    record,
                    score,
                    vector: None,
                });
            }
        }

        let mut set = compose(ranked, &self.config.compose);
        trace.total_ms = started.elapsed().as_millis();
        set.tokens = set
            .items
            .iter()
            .map(|i| super::ingest::approx_tokens(&i.value))
            .sum();
        Ok((set, trace))
    }
}
