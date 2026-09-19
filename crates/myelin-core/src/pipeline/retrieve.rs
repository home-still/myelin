//! `recall` — the fast read path (`PLAN.md` §7.1).
//!
//! ```text
//! scope-filter  → payload predicates, applied BEFORE ranking
//! retrieve      → dense + lex, one round trip, two ranked lists
//! graph         → PPR over phrase↔record incidence, a third list (§5.4)
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
use crate::rerank::Reranker;
use crate::store::graph::{GraphIndex, DEFAULT_DAMPING, DEFAULT_ITERATIONS};
use crate::store::ledger::Ledger;
use crate::store::qdrant::QdrantStore;

use super::compose::{compose, ComposeConfig, Ranked};
use super::fuse::{rrf, RankedList, DEFAULT_RRF_K};
use super::phrases::phrases;

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
    /// Cross-encoder score below which the whole evidence set is withheld.
    ///
    /// **This is an abstention gate, and it exists because abstention is
    /// where LongMemEval-V2 is won or lost.** 72 of the 240 `web` questions
    /// are `-abs`: the haystack genuinely does not contain the answer and
    /// the correct response is to say so. Measured at k=6 we score 16.7% on
    /// them, because `recall` always returns its best six records however
    /// bad they are, and a reader handed six plausible-looking page
    /// fragments answers from them.
    ///
    /// **Calibrated, and it does not work. Default `None`, and that is the
    /// measurement talking.**
    ///
    /// The idea was that the signal is already computed and thrown away:
    /// bge-reranker-v2-m3 separated an answer-bearing document from two
    /// distractors by +2.88 against −11.04 on a toy query, so a threshold
    /// on the top score should cost nothing and buy abstention.
    ///
    /// Calibrated over 120 real LME-V2 questions, the two populations
    /// overlap almost entirely:
    ///
    /// | | n | min | p25 | median | max |
    /// |---|---|---|---|---|---|
    /// | answerable | 86 | −2.72 | −0.03 | **0.76** | 4.13 |
    /// | abstention | 34 | −4.11 | −0.54 | **0.47** | 2.43 |
    ///
    /// Every threshold trades one error for the other at a loss. Expected
    /// full-set accuracy on `web` (168 answerable, 72 abstention, measured
    /// 0.429 / 0.222):
    ///
    /// ```text
    /// no gate                        36.7%
    /// tau = 0   keep .74  hold .35   37.1%
    /// tau = 1   keep .45  hold .62   34.6%
    /// tau = 2   keep .16  hold .85   31.3%
    /// ```
    ///
    /// The best value buys +0.4 points, inside the noise on 240 questions.
    ///
    /// The reason is worth keeping: **a cross-encoder scores relevance, not
    /// answer-containment.** An LME-V2 abstention question asks something
    /// plausible about an environment the haystack really does describe, so
    /// topically relevant pages score high and the question is still
    /// unanswerable. Sufficiency is a different predicate from relevance
    /// and no threshold on the latter can approximate it — that needs the
    /// reflect step in `investigate`.
    ///
    /// Kept as a knob rather than deleted because `top_score` is a useful
    /// diagnostic and a different reranker or corpus could separate. Two
    /// further reasons it stays off by default:
    ///
    /// 1. every M4 number was measured without it and must stay
    ///    comparable, and
    /// 2. these are **cross-encoder logits**. Applied to RRF scores they
    ///    would be meaningless, so the gate is ignored unless a reranker is
    ///    configured rather than silently comparing against the wrong
    ///    scale.
    pub tau_abstain: Option<f32>,
    /// Fuse a personalized-PageRank ranking over the phrase↔record incidence
    /// graph as a third channel (`PLAN.md` §7.1 `route?`, §5.4).
    ///
    /// Off by default until a measurement says otherwise, the discipline
    /// [`RetrieveConfig::tau_abstain`] already got. The verdict lives in
    /// `docs/measurements/m12-graph-route.md`.
    pub graph: bool,
    /// PPR candidate depth before fusion. Equal to `prefetch_limit` so every
    /// fused channel contributes the same depth — RRF compares ranks, and a
    /// channel allowed a longer list gets more mass for free.
    pub graph_limit: usize,
    /// `PLAN.md` §5.4 records damping as `[UNVERIFIED]` in the corpus and a
    /// parameter to tune, not a constant.
    pub graph_damping: f32,
    pub graph_iterations: usize,
    pub compose: ComposeConfig,
}

impl Default for RetrieveConfig {
    fn default() -> Self {
        Self {
            prefetch_limit: 50,
            rrf_k: DEFAULT_RRF_K,
            channels: Channels::Hybrid,
            rerank_depth: 25,
            tau_abstain: None,
            graph: false,
            graph_limit: 50,
            graph_damping: DEFAULT_DAMPING,
            graph_iterations: DEFAULT_ITERATIONS,
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
    /// Best cross-encoder score seen, when a reranker ran. The calibration
    /// input for [`RetrieveConfig::tau_abstain`], and reported so a run can
    /// show the score distribution it actually saw rather than asserting a
    /// threshold was reasonable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_score: Option<f32>,
    /// The gate fired and the evidence set was withheld.
    #[serde(default)]
    pub abstained: bool,
    /// Query phrases [`crate::pipeline::phrases::phrases`] extracted, the
    /// number of records PPR contributed, and what the channel cost — so the
    /// mechanism's reach and price are reportable rather than asserted. All
    /// zero when the switch is off.
    #[serde(default)]
    pub graph_seeds: usize,
    #[serde(default)]
    pub graph_hits: usize,
    #[serde(default)]
    pub graph_ms: u128,
}

pub struct Retriever<'a> {
    pub embedder: &'a dyn Embedder,
    pub store: &'a QdrantStore,
    pub ledger: &'a Ledger,
    pub reranker: Option<&'a dyn Reranker>,
    /// Present only when the caller wired one; the switch alone is inert.
    pub graph: Option<&'a GraphIndex>,
    pub config: RetrieveConfig,
}

impl<'a> Retriever<'a> {
    pub fn new(embedder: &'a dyn Embedder, store: &'a QdrantStore, ledger: &'a Ledger) -> Self {
        Self {
            embedder,
            store,
            ledger,
            reranker: None,
            graph: None,
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

    pub fn with_graph(mut self, graph: &'a GraphIndex) -> Self {
        self.graph = Some(graph);
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

        // The graph channel (`PLAN.md` §7.1 `route?`, §5.4). It joins
        // `ranked_lists` *before* `rrf` consumes them, so all three channels
        // go through one fusion and every later stage is untouched: a
        // PPR-surfaced record still has to pass `is_admissible_at` (I3,
        // validity) and still competes in the cross-encoder rerank.
        //
        // With the switch off, or with no `GraphIndex` wired, this block is a
        // no-op and the code path is byte-identical to the two-channel one.
        if self.config.graph {
            if let Some(index) = self.graph {
                let t = std::time::Instant::now();
                let seeds = phrases(&query.text);
                trace.graph_seeds = seeds.len();
                // No seeds: PPR would return only background reset mass,
                // which is a uniform ranking and pure noise in RRF. Skip the
                // channel rather than fuse noise into it.
                if !seeds.is_empty() {
                    // Scoped by tenant, never by namespace: C12 forbids a
                    // cross-tenant read and the E4 isolation test attacks
                    // exactly this.
                    let graph = index
                        .tenant(
                            self.ledger,
                            &query.scope.tenant,
                            query.scope.namespace.as_deref(),
                        )
                        .await?;
                    let ids: Vec<uuid::Uuid> = graph
                        .personalized_pagerank(
                            &seeds,
                            self.config.graph_damping,
                            self.config.graph_iterations,
                        )
                        .into_iter()
                        .take(self.config.graph_limit)
                        .map(|(id, _)| id)
                        .collect();
                    trace.graph_hits = ids.len();
                    ranked_lists.push(RankedList::new("ppr", ids));
                }
                trace.graph_ms = t.elapsed().as_millis();
            }
        }
        let fused = rrf(&ranked_lists, self.config.rrf_k);
        trace.fused = fused.len();

        // Text comes from the payload, so the head of the list costs no
        // SQLite round trip.
        let mut text_by_id = std::collections::HashMap::new();
        for hit in lists.dense.iter().chain(lists.lex.iter()) {
            text_by_id.entry(hit.id).or_insert_with(|| hit.text.clone());
        }

        // `rerank_depth` is a floor on how deep the reranker looks, never a
        // ceiling on what the caller asked for.
        //
        // It used to be a bare `.take(self.config.rerank_depth)`, which
        // silently capped `k` at 25. An LME-V2 operating-point sweep at
        // k=60 therefore measured k=25 with a different tie-break and
        // reported the difference as a finding — both runs came back with
        // the same ~9,600-token evidence set, which is what gave it away. A
        // config constant must not quietly overrule a query parameter (R4).
        let depth = self.config.rerank_depth.max(query.budget.k);
        let head: Vec<(uuid::Uuid, f32)> = fused.into_iter().take(depth).collect();

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
            //
            // Kind is NOT filtered here. `PLAN.md` §7.1 lists `kind?` as an
            // optional scope predicate, and when the caller supplies one it
            // is already a Qdrant payload filter applied *before* ranking —
            // which is the whole point of scope-before-routing (§2 finding
            // 3, +2.9/+3.1 F1). An earlier draft dropped episodic records
            // here whenever `kinds` was absent, on the theory that episodes
            // are raw material for `investigate`. That was wrong twice: it
            // post-filtered, spending the probe budget on rows it then threw
            // away, and it made `recall` return nothing at all on an
            // episodic-only corpus — which is exactly what LME-V2 is.
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
            trace.top_score = admissible.first().map(|(_, s, _)| *s);

            // The abstention gate. Only here, inside the reranker branch:
            // `tau_abstain` is a cross-encoder logit and comparing it to an
            // RRF score would be a category error, so a configured
            // threshold is ignored rather than misapplied when no reranker
            // is present.
            if let (Some(tau), Some(top)) = (self.config.tau_abstain, trace.top_score) {
                if top < tau {
                    trace.abstained = true;
                    admissible.clear();
                }
            }
        }
        trace.rerank_ms = t2.elapsed().as_millis();

        // R4: `k` and the token budget are QUERY-time parameters against one
        // identical store, so the request wins over the configured default.
        // `RetrieveConfig::compose` supplies everything the query does not
        // name (near-duplicate threshold, bookending). Without this the MCP
        // `recall` tool silently returned 6 items for `k: 3`, which is how
        // this was found.
        //
        // `timeline` is decided here for the same reason and cannot be decided
        // anywhere else: `compose` never sees the question, and the dated
        // index is only wanted for a question that asks for an elapsed time
        // or for the order of two events. Configured off means off; the
        // question shape only ever narrows it.
        let compose_cfg = ComposeConfig {
            k: query.budget.k,
            max_tokens: query.budget.tokens,
            timeline: self.config.compose.timeline
                && crate::time::is_interval_question(&query.text),
            ..self.config.compose.clone()
        };

        // 3x the emitted count: `compose` drops near-duplicates and
        // budget-busting items, so it needs slack to reach k.
        let mut ranked = Vec::with_capacity(admissible.len());
        for (id, score, _) in admissible.into_iter().take(compose_cfg.k * 3) {
            if let Some(record) = self.ledger.get(id).await? {
                ranked.push(Ranked {
                    record,
                    score,
                    vector: None,
                });
            }
        }

        let mut set = compose(ranked, &compose_cfg);
        trace.total_ms = started.elapsed().as_millis();
        set.tokens = set
            .items
            .iter()
            .map(|i| super::ingest::approx_tokens(&i.value))
            .sum();
        Ok((set, trace))
    }
}
