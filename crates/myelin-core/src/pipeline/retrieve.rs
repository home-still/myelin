//! `recall` — the fast read path (`PLAN.md` §7.1).
//!
//! ```text
//! scope-filter  → payload predicates, applied BEFORE ranking
//! retrieve      → dense + lex, one round trip, two ranked lists
//! graph         → PPR over phrase↔record incidence, a third list (§5.4)
//! fuse          → RRF in Rust at the configured k (Qdrant's own is 1, §5.2)
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
use crate::error::{MyelinError, Result};
use crate::llm::Llm;
use crate::model::evidence::EvidenceSet;
use crate::model::query::Recall;
use crate::model::record::MemoryRecord;
use crate::rerank::Reranker;
use crate::store::graph::{GraphIndex, DEFAULT_DAMPING, DEFAULT_ITERATIONS};
use crate::store::ledger::Ledger;
use crate::store::qdrant::QdrantStore;

use super::compose::{compose, ComposeConfig, Ranked};
use super::fuse::{rrf, RankedList, DEFAULT_RRF_K};
use super::phrases::phrases;
use super::decompose::Decomposer;
use super::select::Selector;
use super::turn_windows;

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
    /// Multiplier on `k` for the reranker's candidate head, so the
    /// cross-encoder *selects* rather than merely reorders.
    ///
    /// The effective depth is `max(rerank_depth, k * rerank_factor)`. At the
    /// shipped operating point (`rerank_depth` 25, `k` 25) a factor of 1
    /// makes those equal, and equality is the degenerate case: the reranker
    /// is handed exactly the set that will be emitted, so it can reorder it
    /// but can never exclude anything, and every fused candidate past rank
    /// `k` is discarded on RRF rank alone without the cross-encoder ever
    /// scoring it. `rerank_depth`'s own doc comment above promises "a deeper
    /// pool than `compose` will finally emit"; with a factor of 1 that
    /// promise is false.
    ///
    /// This is why M37 measured raising `prefetch_limit` from 50 to 400 as a
    /// **null at every k** (+0.6 / +0.0 / +0.8 points of recall at k =
    /// 25/50/100): the extra candidates were fused and then truncated away
    /// at the `take(depth)` below before the reranker saw them. Widening the
    /// first stage cannot pay while the second stage is degenerate.
    ///
    /// **Measured, and it is a null. Default 1, and that is the measurement
    /// talking.**
    ///
    /// M37 ran `factor = 4` (head 100, i.e. the entire fused list at the
    /// default prefetch) against `factor = 1` over the 255 answerable,
    /// string-checkable LME-V2 rows: answer-string recall **57.65% → 58.43%,
    /// +0.78 points, 95% CI [−3.14, +4.71]**, 14 rows gained and 12 lost.
    /// The pre-registered bar was +3.0 with a CI excluding zero, so the
    /// default stays 1.
    ///
    /// The null is informative rather than disappointing, because the deep
    /// arm also emitted **42% more evidence** (mean 16.8 → 23.9 items;
    /// `factor = 1` could not even fill the requested k = 25 once ledger
    /// admissibility had thinned a 25-candidate head) and still did not move
    /// recall. A cross-encoder handed four times the candidates returns
    /// essentially the same set.
    ///
    /// Read with the other two widening nulls in the same milestone —
    /// `prefetch_limit` 50 → 400, and k 25 → 100 buying only +9.1 — the
    /// conclusion is that widening is exhausted and the *representation* is
    /// the bottleneck: 98.5% of the LME-V2 index is raw AXTree page dumps,
    /// and entity extraction yields the single token `page` for all 37,731
    /// of them. `docs/measurements/m37-rerank-head.md`.
    ///
    /// Kept, not deleted: it repairs a real degeneracy (`depth == k` means
    /// the reranker cannot exclude anything) and is the knob any future
    /// arm on a sharper representation will need.
    pub rerank_factor: usize,
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
    /// Ask the model which of the reranked candidates jointly answer the
    /// question, and put those first (M21,
    /// [`crate::pipeline::select::Selector`]).
    ///
    /// **This can never become a `recall` default.** `PLAN.md` §7.1 specifies
    /// this path as "target p95 < 100 ms, **no LLM in the loop**", and a
    /// model call per query violates that by construction whatever the switch
    /// measures. It exists as a ceiling probe against
    /// [`ComposeConfig::mmr_lambda`]: MMR diversifies by vector distance,
    /// which is a proxy for "covers something else" rather than for "covers
    /// what the question needs", and this says whether MMR's shortfall is the
    /// heuristic or something deeper. If it wins, its home is `investigate`.
    ///
    /// Inert without [`Retriever::with_llm`] — the switch alone does nothing,
    /// the same contract [`RetrieveConfig::graph`] has with
    /// [`Retriever::with_graph`].
    pub select_sufficient: bool,
    /// Split the question into sub-queries and retrieve for each, fusing
    /// them into the same RRF call as the original (M24). `None` — the
    /// default, and every run before M24 — is the single-query path.
    ///
    /// The value is the cap on sub-queries, bounded by
    /// [`crate::pipeline::decompose::MAX_SUBQUERIES`]. Cost is one model
    /// call plus one embedding and one `hybrid_search` per sub-query; the
    /// embeddings go in a single batched call, the searches do not.
    ///
    /// **The original question is always retrieved too**, so a sub-query set
    /// that misses the point can only add candidates and never lose them.
    /// Nothing is de-duplicated across sub-query results: M21 measured that
    /// co-evidence for one question resembles *itself* 1.60× more than the
    /// rest of the set, so a redundancy penalty here would be aimed exactly
    /// at the records multi-hop needs.
    ///
    /// Cannot ship on for `recall` whatever it measures — `PLAN.md` §7.1
    /// forbids an LLM in that loop — so its home if it wins is
    /// `investigate`. Inert without [`Retriever::with_llm`].
    pub decompose: Option<usize>,
    /// M66: emit episodes as turn windows, ranked turn by turn
    /// ([`crate::pipeline::turn_windows`]). The value is the radius, the
    /// neighbours kept on each side of a selected turn. `None`, the default,
    /// emits whole episodes.
    ///
    /// The turns are scored by the same cross-encoder as the pool, so it
    /// refuses to run without one. It also refuses to run with
    /// [`RetrieveConfig::select_sufficient`], because the global turn ranking
    /// would silently discard the selector's order.
    pub turn_windows: Option<usize>,
    /// M74: show the sufficiency selector each candidate's best-matching
    /// line instead of its first 400 characters
    /// ([`crate::pipeline::turn_windows::focus_views`]), in `recall`'s
    /// selection and in `investigate`'s. The selector's choice is the only
    /// thing it changes. Needs the cross-encoder, and refuses to run
    /// without one. Default off pending its arm
    /// (`docs/measurements/m74-selector-focus.md`).
    pub select_focus: bool,
    pub compose: ComposeConfig,
}

impl Default for RetrieveConfig {
    fn default() -> Self {
        Self {
            prefetch_limit: 50,
            rrf_k: DEFAULT_RRF_K,
            channels: Channels::Hybrid,
            rerank_depth: 25,
            rerank_factor: 1,
            tau_abstain: None,
            graph: false,
            graph_limit: 50,
            graph_damping: DEFAULT_DAMPING,
            graph_iterations: DEFAULT_ITERATIONS,
            select_sufficient: false,
            decompose: None,
            turn_windows: None,
            select_focus: false,
            compose: ComposeConfig::default(),
        }
    }
}

/// How many fused candidates the reranker scores, given the config floor,
/// the `k` multiplier and the query's `k`.
///
/// A free function so the arithmetic is testable without a store, an
/// embedder, a reranker and a live ledger — the same reason
/// [`super::investigate`] extracted its gate. This one earned it: the
/// degenerate case `depth == k` shipped for the whole M25–M36 sequence and
/// was only found by measuring a *different* knob and getting a null.
///
/// Two rules, in order:
///
/// 1. `rerank_depth` is a **floor**, never a ceiling on the caller's `k` —
///    a config constant must not overrule a query parameter (R4).
/// 2. `k * factor` is the selection head. `factor` of 0 is meaningless and
///    is read as 1 rather than collapsing the head to nothing.
///
/// Saturating throughout: `k` is caller-supplied and `factor` is config, and
/// `usize` overflow here would silently truncate the head to a few records.
pub fn rerank_head_depth(rerank_depth: usize, rerank_factor: usize, k: usize) -> usize {
    rerank_depth.max(k.saturating_mul(rerank_factor.max(1)))
}

/// What a single `recall` did, for the latency and ablation tables.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RecallTrace {
    pub dense_hits: usize,
    pub lex_hits: usize,
    pub fused: usize,
    pub reranked: usize,
    pub admitted: usize,
    /// Fused candidates actually handed to the reranker, i.e. the
    /// `max(rerank_depth, k * rerank_factor)` head. Reported because
    /// `reranked == admitted == k` is the signature of a degenerate second
    /// stage, and M37 spent an afternoon inferring that from a null instead
    /// of reading it off a trace.
    #[serde(default)]
    pub rerank_depth: usize,
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
    /// How many candidates the sufficiency selector kept, and what the model
    /// call cost. Zero when [`RetrieveConfig::select_sufficient`] is off or
    /// no [`Llm`] was wired — which is the check that catches an inert
    /// switch before a whole arm is measured against nothing.
    #[serde(default)]
    pub selected: usize,
    #[serde(default)]
    pub select_ms: u128,
    /// Why the selector fell back to rank order, when it did.
    ///
    /// Recorded because a fallback is indistinguishable from a selection
    /// that agreed with rank order, so without it an inert selector reads
    /// as a clean null. Measured: a 100-candidate prompt over real
    /// LongMemEval records is 8,298 tokens against a reader serving 8,192
    /// per slot, and every call 400s
    /// ([`crate::pipeline::select::Degradation::CallFailed`]).
    ///
    /// Carries the *cause* rather than a bool because the two causes need
    /// opposite responses: M32 measured 11 fallbacks in 298 queries against
    /// a healthy server, all of them
    /// [`crate::pipeline::select::Degradation::ModelDeclined`], and a gate
    /// that cannot tell them apart refuses that run.
    #[serde(default)]
    pub select_degraded: crate::pipeline::select::Degradation,
    /// Sub-queries the decomposer produced and what the call cost. Zero
    /// when [`RetrieveConfig::decompose`] is off, when no [`Llm`] was
    /// wired, or when the model judged the question to need only itself —
    /// which is a real answer and not a failure, so `decompose_ms` is the
    /// field that distinguishes "did not run" from "ran and said one".
    #[serde(default)]
    pub subqueries: usize,
    #[serde(default)]
    pub decompose_ms: u128,
    /// Turns the cross-encoder scored for [`RetrieveConfig::turn_windows`]
    /// (M66), and what that cost. Zero when the switch is off.
    #[serde(default)]
    pub turns_scored: usize,
    #[serde(default)]
    pub turn_windows_ms: u128,
    /// The reranked pool in rank order, `(id, text)`, before `compose`
    /// truncates to `k`.
    ///
    /// In-process only (`serde(skip)`): it is the input to the offline
    /// instruments that need retrieval's *ceiling* rather than its
    /// emission, and persisting it would add megabytes to every artifact
    /// for a diagnostic no artifact reader consumes.
    ///
    /// Carries the text as well as the id because the two corpora decide
    /// "did we retrieve the gold" differently: LoCoMo resolves record ids
    /// to `dia_id` turns through I4 lineage, LongMemEval_S matches the
    /// `has_answer` turn's text prefix against the record. An id-only pool
    /// can be scored on the first corpus and not the second. The clone is
    /// the head of a list that is about to be truncated anyway — ~12 KB
    /// against an embedding call and a 25-document rerank.
    #[serde(skip)]
    pub pool: Vec<(uuid::Uuid, String)>,
    /// Which of `compose`'s two limits bit (M28), mirrored from
    /// [`EvidenceSet`]. `dropped_for_tokens > 0` means the *token budget*
    /// truncated the evidence; `k_bound` with zero drops means `k` did.
    /// Every width measurement M25-M27 made ran without recording this and
    /// attributed the whole loss to `k`.
    #[serde(default)]
    pub dropped_for_tokens: usize,
    #[serde(default)]
    pub k_bound: bool,
}

pub struct Retriever<'a> {
    pub embedder: &'a dyn Embedder,
    pub store: &'a QdrantStore,
    pub ledger: &'a Ledger,
    pub reranker: Option<&'a dyn Reranker>,
    /// Present only when the caller wired one; the switch alone is inert.
    pub graph: Option<&'a GraphIndex>,
    /// Present only when the caller wired one;
    /// [`RetrieveConfig::select_sufficient`] alone is inert.
    pub llm: Option<&'a dyn Llm>,
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
            llm: None,
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

    pub fn with_llm(mut self, llm: &'a dyn Llm) -> Self {
        self.llm = Some(llm);
        self
    }

    pub async fn recall(&self, query: &Recall) -> Result<(EvidenceSet, RecallTrace)> {
        let started = std::time::Instant::now();
        let mut trace = RecallTrace::default();

        // Decomposition (M24) runs first: its sub-queries need embedding
        // alongside the original, and one batched `embed` call is the
        // difference between N round trips and one.
        //
        // Inert without an `Llm`, exactly as `select_sufficient` and
        // `graph` are inert without their collaborators — and, like them,
        // a switch that is on with nothing wired is reported in the trace
        // (`subqueries` stays 0 with `decompose_ms` > 0 only when the call
        // actually ran) rather than failing silently.
        let mut subqueries: Vec<String> = Vec::new();
        if let (Some(max), Some(llm)) = (self.config.decompose, self.llm) {
            let td = std::time::Instant::now();
            subqueries = Decomposer::new(llm).decompose(&query.text, max).await?;
            trace.decompose_ms = td.elapsed().as_millis();
            trace.subqueries = subqueries.len();
        }

        // Dense is needed for the dense channel and for compose's
        // near-duplicate suppression; skip it entirely for the lex-only arm
        // so that arm's latency is honest.
        //
        // The original question is always index 0, so the single-query path
        // is byte-identical when `subqueries` is empty.
        let t0 = std::time::Instant::now();
        let texts: Vec<String> = std::iter::once(query.text.clone())
            .chain(subqueries.iter().cloned())
            .collect();
        let dense_vectors: Vec<Vec<f32>> = if self.config.channels == Channels::Lex {
            vec![Vec::new(); texts.len()]
        } else {
            let mut got = self.embedder.embed(&texts).await?;
            got.resize(texts.len(), Vec::new());
            got
        };
        trace.embed_ms = t0.elapsed().as_millis();

        let kinds: Vec<&str> = query
            .kinds
            .as_ref()
            .map(|ks| ks.iter().map(|k| k.as_str()).collect())
            .unwrap_or_default();

        // Near-duplicate suppression is a cosine, so it needs the stored
        // vectors back from Qdrant. ~1024 floats per hit is not free, so ask
        // only when `compose` will actually use them; with the threshold at
        // zero `compose` is exact-text-only and `Ranked::vector` stays `None`.
        let want_vectors = self.config.compose.tau_near_dup > 0.0;

        let scope = crate::store::qdrant::SearchScope {
            tenant: &query.scope.tenant,
            namespace: query.scope.namespace.as_deref(),
            agent: query.scope.agent.as_deref(),
            session: query.scope.session.as_deref(),
        };

        let t1 = std::time::Instant::now();
        let mut all_lists = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            all_lists.push(
                self.store
                    .hybrid_search(
                        dense_vectors[i].clone(),
                        text,
                        scope,
                        &kinds,
                        self.config.prefetch_limit,
                        want_vectors,
                    )
                    .await?,
            );
        }
        trace.search_ms = t1.elapsed().as_millis();
        // The original question's own hit counts, so the trace stays
        // comparable across the switch rather than reporting a total that
        // silently grew with the sub-query count.
        let lists = &all_lists[0];
        trace.dense_hits = lists.dense.len();
        trace.lex_hits = lists.lex.len();

        // Fuse the selected channels. A single-channel arm still goes through
        // `rrf` so the ablation compares ranking, not plumbing.
        //
        // Each sub-query contributes its own pair of lists to the *same*
        // fusion (M24), so a record that answers two sub-questions
        // accumulates reciprocal rank from both and rises — which is the
        // multi-hop shape — while every stage below this sees one ordinary
        // fused list and is untouched.
        let mut ranked_lists = Vec::new();
        for l in &all_lists {
            if matches!(self.config.channels, Channels::Dense | Channels::Hybrid) {
                ranked_lists.push(RankedList::new(
                    "dense",
                    l.dense.iter().map(|h| h.id).collect(),
                ));
            }
            if matches!(self.config.channels, Channels::Lex | Channels::Hybrid) {
                ranked_lists.push(RankedList::new("lex", l.lex.iter().map(|h| h.id).collect()));
            }
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
        // SQLite round trip. Vectors ride along the same way when
        // `want_vectors` asked for them; only the dense channel carries one.
        //
        // Across *every* sub-query's hits: a record surfaced only by a
        // sub-query still has to be materialisable, and falling back to the
        // ledger's `record.text` for it would silently cost a round trip
        // per decomposed candidate.
        let mut text_by_id = std::collections::HashMap::new();
        let mut vector_by_id: std::collections::HashMap<uuid::Uuid, Vec<f32>> =
            std::collections::HashMap::new();
        for hit in all_lists.iter().flat_map(|l| l.dense.iter().chain(l.lex.iter())) {
            text_by_id.entry(hit.id).or_insert_with(|| hit.text.clone());
            if let Some(v) = hit.vector.as_ref() {
                vector_by_id.entry(hit.id).or_insert_with(|| v.clone());
            }
        }

        let depth = rerank_head_depth(
            self.config.rerank_depth,
            self.config.rerank_factor,
            query.budget.k,
        );
        trace.rerank_depth = depth;
        let head: Vec<(uuid::Uuid, f32)> = fused.into_iter().take(depth).collect();

        // Rerank before the ledger check: reranking is the expensive stage
        // and there is no point paying it for records the ledger will refuse.
        // But the ledger check is cheap and local, so it runs first.
        let now = chrono::Utc::now();
        let mut admissible: Vec<(uuid::Uuid, f32, String)> = Vec::new();
        let mut record_by_id: std::collections::HashMap<uuid::Uuid, MemoryRecord> =
            std::collections::HashMap::new();
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
            // C12: the scope predicates are Qdrant payload filters, and a
            // payload is a projection that can lag or regress. Re-check them
            // against the materialised record so a leaked id cannot survive
            // fusion — this is a point lookup on a record already in hand,
            // not a second listing query.
            if !query.scope.admits(&record.scope) {
                continue;
            }
            let text = text_by_id
                .get(&id)
                .cloned()
                .unwrap_or_else(|| record.text.clone());
            admissible.push((id, score, text));
            if self.config.turn_windows.is_some() {
                // M66 splits the record's turns, so it keeps the record it
                // already read rather than reading it again.
                record_by_id.insert(id, record);
            }
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

        // The sufficiency selector (M21). After the rerank sort, because it
        // reorders the reranked pool rather than replacing it, and before
        // the `k * 3` materialisation below, because the whole point is to
        // reach past the head of that window. `compose` is untouched — it
        // still takes the head of whatever order it is handed.
        if self.config.select_sufficient && !admissible.is_empty() {
            if let Some(llm) = self.llm {
                let t3 = std::time::Instant::now();
                let docs: Vec<String> = admissible.iter().map(|(_, _, t)| t.clone()).collect();
                let docs = if self.config.select_focus {
                    let reranker = self.reranker.ok_or_else(|| {
                        MyelinError::Config("select_focus needs the cross-encoder; none is wired".into())
                    })?;
                    crate::pipeline::turn_windows::focus_views(reranker, &query.text, &docs).await?
                } else {
                    docs
                };
                let keep = Selector::new(llm)
                    .select(&query.text, &docs, query.budget.k)
                    .await?;
                // Stable partition: the kept ids move to the front in the
                // model's order, everything else keeps its reranked order
                // behind them. Nothing is dropped, so a selector that picks
                // badly costs rank positions and never evidence.
                let mut slots: Vec<Option<(uuid::Uuid, f32, String)>> =
                    admissible.into_iter().map(Some).collect();
                let mut front = Vec::with_capacity(slots.len());
                for &i in &keep.keep {
                    if let Some(slot) = slots[i].take() {
                        front.push(slot);
                    }
                }
                front.extend(slots.into_iter().flatten());
                admissible = front;
                trace.selected = keep.keep.len();
                trace.select_degraded = keep.degraded;
                trace.select_ms = t3.elapsed().as_millis();
            }
        }

        // The reranked pool's ids, in rank order, for the offline
        // instruments that need the *ceiling* and not just the emission.
        //
        // Every "is this loss retrieval's or truncation's" argument since M9
        // has been settled by running two `k` values and differencing them.
        // M21 measured 0.662 emitted against 0.852 pool on LongMemEval
        // temporal that way — one finding, two runs. With the pool in hand
        // both numbers come from one pass, and a null result stays
        // interpretable: gold absent from the pool is retrieval's problem,
        // gold present but unemitted is `compose`'s.
        //
        // `skip`, not `skip_serializing_if`: 25 uuids per question is ~1.4 MB
        // on a LoCoMo run, for a diagnostic every consumer computes in
        // process. Persisting it would grow every artifact on disk to carry
        // something no artifact reader uses.
        trace.pool = admissible
            .iter()
            .map(|(id, _, text)| (*id, text.clone()))
            .collect();

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
            // The question's own day, for the anchored timeline (M46).
            as_of: query.as_of,
            ..self.config.compose.clone()
        };

        // 3x the emitted count: `compose` drops near-duplicates and
        // budget-busting items, so it needs slack to reach k.
        let slots = compose_cfg.k * 3;
        let mut ranked = Vec::with_capacity(admissible.len());
        if let Some(radius) = self.config.turn_windows {
            let Some(reranker) = self.reranker else {
                return Err(MyelinError::Config(
                    "turn windows rank turns with the cross-encoder, and no reranker is wired"
                        .into(),
                ));
            };
            if self.config.select_sufficient {
                return Err(MyelinError::Config(
                    "turn windows rank the pool turn by turn, which would discard the \
                     sufficiency selector's order; run one or the other"
                        .into(),
                ));
            }
            let t = std::time::Instant::now();
            let pool: Vec<turn_windows::PoolItem> = admissible
                .into_iter()
                .filter_map(|(id, score, _)| {
                    record_by_id
                        .remove(&id)
                        .map(|record| turn_windows::PoolItem { id, score, record })
                })
                .collect();
            let (windowed, stats) =
                turn_windows::select(reranker, &query.text, pool, compose_cfg.k, slots, radius)
                    .await?;
            trace.turns_scored = stats.turns_scored;
            trace.turn_windows_ms = t.elapsed().as_millis();
            for w in windowed {
                ranked.push(Ranked {
                    record: w.record,
                    score: w.score,
                    vector: vector_by_id.remove(&w.id),
                    window: w.window,
                });
            }
        } else {
            for (id, score, _) in admissible.into_iter().take(slots) {
                if let Some(record) = self.ledger.get(id).await? {
                    ranked.push(Ranked {
                        record,
                        score,
                        // `remove`, not `get`: each id reaches `compose` once,
                        // so the vector can be moved rather than cloned.
                        vector: vector_by_id.remove(&id),
                        window: None,
                    });
                }
            }
        }


        let mut set = compose(ranked,&compose_cfg);
        trace.dropped_for_tokens = set.dropped_for_tokens;
        trace.k_bound = set.k_bound;
        trace.total_ms = started.elapsed().as_millis();
        set.tokens = set
            .items
            .iter()
            .map(|i| super::ingest::approx_tokens(&i.value))
            .sum();
        Ok((set, trace))
    }
}

#[cfg(test)]
mod tests {
    use super::{rerank_head_depth, RetrieveConfig};

    /// The shipped operating point, pinned as the thing that was wrong.
    ///
    /// `rerank_depth` 25 and `k` 25 make the head exactly `k`, so the
    /// cross-encoder is handed the set it will emit and cannot drop a single
    /// record. This asserts the arithmetic, not the wisdom: the default is
    /// still `factor = 1` until an arm says otherwise, and if someone flips
    /// that default this test is the one that should make them justify it.
    #[test]
    fn factor_one_at_the_default_k_makes_the_reranker_a_reordering() {
        let cfg = RetrieveConfig::default();
        assert_eq!(cfg.rerank_factor, 1, "default must stay off until measured");
        let depth = rerank_head_depth(cfg.rerank_depth, cfg.rerank_factor, 25);
        assert_eq!(depth, 25, "head equals k, so nothing can be excluded");
    }

    /// The property that makes the second stage a *selector*: strictly more
    /// candidates scored than emitted.
    #[test]
    fn a_factor_above_one_gives_the_reranker_more_than_it_emits() {
        for factor in [2usize, 4, 8] {
            for k in [6usize, 25, 60] {
                let depth = rerank_head_depth(25, factor, k);
                assert!(
                    depth > k,
                    "factor {factor} at k {k} produced depth {depth}; a head \
                     no deeper than k cannot select"
                );
                assert_eq!(depth, (k * factor).max(25));
            }
        }
    }

    /// R4: a config constant must never overrule a query parameter. This is
    /// the regression the floor was introduced for — a bare `take(depth)`
    /// once capped `k` at 25 and an operating-point sweep at k=60 silently
    /// measured k=25.
    #[test]
    fn the_configured_depth_is_a_floor_and_never_caps_k() {
        assert_eq!(rerank_head_depth(25, 1, 3), 25, "floor applies below k");
        assert_eq!(rerank_head_depth(25, 1, 60), 60, "k is never capped");
        assert_eq!(rerank_head_depth(200, 1, 60), 200, "deeper floor still wins");
    }

    /// A zero factor is a misconfiguration, not an instruction to rerank
    /// nothing: collapsing the head to zero would return an empty evidence
    /// set for every query.
    #[test]
    fn a_zero_factor_is_read_as_one_rather_than_emptying_the_head() {
        assert_eq!(rerank_head_depth(25, 0, 60), 60);
        assert_eq!(rerank_head_depth(1, 0, 7), 7);
    }

    /// `k` is caller-supplied and `factor` is config; multiplying them must
    /// not wrap a `usize` and truncate the head to a handful of records.
    #[test]
    fn an_absurd_k_and_factor_saturate_instead_of_wrapping() {
        let depth = rerank_head_depth(25, usize::MAX, usize::MAX);
        assert_eq!(depth, usize::MAX, "overflow must saturate high, not low");
    }
}
