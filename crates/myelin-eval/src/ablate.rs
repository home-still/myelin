//! Retrieval ablations (`EVALUATION.md` §8 rows 1, 2 and RRF; `PLAN.md` M4).
//!
//! **This measures retrieval, not answers.** No reader, no judge, no
//! temperature. A question's gold `evidence` in LoCoMo is a list of `dia_id`
//! turn pointers, so "did we retrieve the right thing" is decidable by exact
//! set arithmetic. Putting an LLM in this loop would add ~2 s/question and a
//! confound, and would measure the reader's tolerance for bad evidence rather
//! than the retriever's accuracy.
//!
//! The coverage map is reconstructed by **re-running the production
//! segmenter**, not by matching text. [`crate::build::turns_for`] plus
//! [`segment`] plus [`record_id`] reproduce the exact episode ids and the
//! exact turn list behind each one, because the id is a v5 hash of the
//! episode's natural key. Substring matching would have been ambiguous
//! ("Speaker: Yes" recurs) and this is not.
//!
//! Every arm is the same [`Retriever`] with different [`RetrieveConfig`]
//! fields. That is the point of `Channels` existing at all: a baseline
//! measured by different code is not a baseline.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use async_trait::async_trait;
use myelin_core::config::MyelinConfig;
use myelin_core::embed::{remote::RemoteEmbedder, Embedder};
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::pipeline::ingest::{segment, SegmentConfig};
use myelin_core::pipeline::investigate::{InvestigateConfig, Investigator};
use myelin_core::pipeline::retrieve::{Channels, RetrieveConfig, Retriever};
use myelin_core::rerank::{cross::CrossEncoder, Reranker};
use myelin_core::store::graph::GraphIndex;
use myelin_core::store::ids::record_id;
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::QdrantStore;
use uuid::Uuid;

use crate::build::turns_for;
use crate::datasets::locomo::{self, LocomoConversation};

/// Question-level gold: which turns must be retrieved to answer it.
struct Question {
    tenant: String,
    text: String,
    category: u8,
    gold: HashSet<String>,
}

/// `record id → the turns it can testify to`.
///
/// Episodic records cover their own turn range. Derived records inherit the
/// union of their ancestors' coverage — which is precisely what I4's
/// `derived_from` lineage exists for, so no extra provenance field is needed
/// to make grounding checkable.
struct Coverage(HashMap<Uuid, HashSet<String>>);

impl Coverage {
    async fn build(conversations: &[LocomoConversation], ledger: &Ledger) -> Result<Self> {
        let mut covers: HashMap<Uuid, HashSet<String>> = HashMap::new();
        let cfg = SegmentConfig::default();

        for conv in conversations {
            let scope = myelin_core::model::record::Scope::new(
                format!("locomo/{}", conv.sample_id),
                "myelin",
                "locomo",
            );
            for draft in segment(&turns_for(conv), &cfg) {
                let id = record_id(&scope, &draft.natural_key());
                covers.insert(
                    id,
                    draft.turns.iter().map(|t| t.source.doc.clone()).collect(),
                );
            }
        }

        // Derived records: resolve lineage once, breadth-first, rather than
        // per-query. A semantic record's ancestors are always episodes here,
        // but the walk is general so a future consolidation-of-consolidations
        // does not silently lose coverage.
        let records = ledger
            .records_in_namespace("locomo")
            .await
            .context("load records for coverage")?;
        let parents: HashMap<Uuid, Vec<Uuid>> = records
            .iter()
            .map(|r| (r.id, r.provenance.derived_from.clone()))
            .collect();

        for id in parents.keys() {
            if covers.contains_key(id) {
                continue;
            }
            let mut acc = HashSet::new();
            let mut stack = parents.get(id).cloned().unwrap_or_default();
            let mut seen = HashSet::new();
            while let Some(p) = stack.pop() {
                if !seen.insert(p) {
                    continue;
                }
                if let Some(c) = covers.get(&p) {
                    acc.extend(c.iter().cloned());
                } else if let Some(gp) = parents.get(&p) {
                    stack.extend(gp.iter().copied());
                }
            }
            covers.insert(*id, acc);
        }

        Ok(Self(covers))
    }

    fn of(&self, id: &Uuid) -> Option<&HashSet<String>> {
        self.0.get(id)
    }
}

/// Memoising wrapper so six arms embed each question once, not six times.
///
/// Worth the 30 lines: 1,540 comparable questions × 6 arms × ~45 ms is 7
/// minutes of GPU time spent recomputing identical vectors on a card that is
/// shared with a live voice assistant.
struct CachingEmbedder<'a> {
    inner: &'a dyn Embedder,
    cache: Mutex<HashMap<String, Vec<f32>>>,
}

impl<'a> CachingEmbedder<'a> {
    fn new(inner: &'a dyn Embedder) -> Self {
        Self {
            inner,
            cache: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl Embedder for CachingEmbedder<'_> {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn dim(&self) -> u64 {
        self.inner.dim()
    }

    async fn embed(&self, texts: &[String]) -> myelin_core::error::Result<Vec<Vec<f32>>> {
        let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        let mut misses: Vec<String> = Vec::new();
        {
            let cache = self.cache.lock().expect("embed cache poisoned");
            for t in texts {
                match cache.get(t) {
                    Some(v) => out.push(Some(v.clone())),
                    None => {
                        out.push(None);
                        misses.push(t.clone());
                    }
                }
            }
        }
        if !misses.is_empty() {
            let fresh = self.inner.embed(&misses).await?;
            let mut cache = self.cache.lock().expect("embed cache poisoned");
            for (t, v) in misses.iter().zip(fresh) {
                cache.insert(t.clone(), v);
            }
        }
        let cache = self.cache.lock().expect("embed cache poisoned");
        Ok(out
            .into_iter()
            .zip(texts)
            .map(|(hit, t)| hit.unwrap_or_else(|| cache[t].clone()))
            .collect())
    }
}

/// The dev split is the first `units` conversations; the holdout is
/// everything after them.
///
/// Split by conversation and never by question: questions from one
/// conversation share a memory, so a question-level split would leak every
/// tuning decision across the boundary. The holdout exists so that a
/// default changed on the strength of the dev table can be *confirmed*
/// rather than merely asserted — which matters here, because the M4 result
/// changed `rrf_k`.
fn select_split(path: &Path, units: usize, holdout: bool) -> Result<Vec<LocomoConversation>> {
    let conversations = locomo::load(path)?;
    let split: Vec<_> = if holdout {
        conversations.into_iter().skip(units).collect()
    } else {
        conversations.into_iter().take(units).collect()
    };
    anyhow::ensure!(
        !split.is_empty(),
        "empty split: units={units} holdout={holdout}"
    );
    Ok(split)
}

/// Questions with gold evidence, in dataset order.
///
/// Category 5 is LoCoMo's adversarial split: it carries no evidence to
/// retrieve, so scoring retrieval on it would be meaningless. It is an
/// abstention test and belongs to G3, not here.
fn collect_questions(split: &[LocomoConversation]) -> Vec<Question> {
    let mut questions = Vec::new();
    for conv in split {
        let tenant = format!("locomo/{}", conv.sample_id);
        for qa in &conv.qa {
            if qa.evidence.is_empty() {
                continue;
            }
            questions.push(Question {
                tenant: tenant.clone(),
                text: qa.question.clone(),
                category: qa.category,
                gold: qa.evidence.iter().cloned().collect(),
            });
        }
    }
    questions
}

pub struct Arm {
    pub name: &'static str,
    pub config: RetrieveConfig,
    pub rerank: bool,
    pub graph: bool,
}

fn arms(k: usize) -> Vec<Arm> {
    let base = |channels, rrf_k, graph| RetrieveConfig {
        channels,
        rrf_k,
        graph,
        compose: myelin_core::pipeline::compose::ComposeConfig {
            k,
            ..Default::default()
        },
        ..Default::default()
    };
    vec![
        Arm {
            name: "dense_only",
            config: base(Channels::Dense, 60.0, false),
            rerank: false,
            graph: false,
        },
        Arm {
            name: "bm25_only",
            config: base(Channels::Lex, 60.0, false),
            rerank: false,
            graph: false,
        },
        Arm {
            name: "hybrid_k60",
            config: base(Channels::Hybrid, 60.0, false),
            rerank: false,
            graph: false,
        },
        Arm {
            name: "hybrid_k1",
            config: base(Channels::Hybrid, 1.0, false),
            rerank: false,
            graph: false,
        },
        Arm {
            name: "hybrid_rerank",
            config: base(Channels::Hybrid, 60.0, false),
            rerank: true,
            graph: false,
        },
        // `EVALUATION.md` §8 row 7. Each pairs against the arm above it that
        // differs in exactly one switch — `graph_k1` against `hybrid_k1`,
        // `graph_rerank` against `hybrid_rerank` — so the only difference the
        // table reports is the PPR channel.
        Arm {
            name: "graph_k1",
            config: base(Channels::Hybrid, 1.0, true),
            rerank: false,
            graph: true,
        },
        Arm {
            name: "graph_rerank",
            config: base(Channels::Hybrid, 60.0, true),
            rerank: true,
            graph: true,
        },
    ]
}

#[derive(Debug, Clone, Default)]
pub struct ArmResult {
    pub arm: String,
    pub questions: usize,
    /// Mean fraction of a question's gold turns that some returned item covers.
    pub recall: f64,
    /// Fraction of questions where at least one gold turn was covered.
    pub any: f64,
    /// Mean reciprocal rank of the first item covering any gold turn.
    pub mrr: f64,
    /// Mean returned items — the denominator of precision, and the thing
    /// `EVALUATION.md` §8 row 4 says must not grow for free.
    pub items: f64,
    pub p50_ms: u128,
    pub p90_ms: u128,
    /// Whether this arm pays the query-embedding cost. The latency columns
    /// exclude it (the cache is warmed before any arm is timed), so the
    /// end-to-end number for a dense-using arm is `p50_ms + embed_p50`.
    pub uses_dense: bool,
    pub by_category: Vec<(u8, usize, f64)>,
}

/// One ablation run: the arms, plus the query-embedding cost every dense arm
/// must be charged for.
#[derive(Debug, Clone, Default)]
pub struct AblationRun {
    pub arms: Vec<ArmResult>,
    pub embed_p50_ms: u128,
    pub embed_p90_ms: u128,
}

fn percentile(sorted: &[u128], q: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

/// Run the M4 ablation over a LoCoMo split.
///
/// `units` is the dev split size. The split is by conversation, not by
/// question: questions from the same conversation share a memory, so splitting
/// by question would leak every tuning decision across the boundary.
#[allow(clippy::too_many_arguments)]
pub async fn ablate_locomo(
    path: &Path,
    collection: &str,
    ledger_path: &Path,
    units: usize,
    k: usize,
    limit: Option<usize>,
    holdout: bool,
) -> Result<AblationRun> {
    let cfg = MyelinConfig::load().context("load myelin config")?;

    let split = select_split(path, units, holdout)?;

    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;
    let coverage = Coverage::build(&split, &ledger).await?;

    let mut questions = collect_questions(&split);
    if let Some(n) = limit {
        questions.truncate(n);
    }

    let base_embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)
        .context("embedder client")?;
    let embedder = CachingEmbedder::new(&base_embedder);

    // Warm the cache before any arm is timed, and time the warming.
    //
    // Without this the first arm pays for every later arm's embeddings and
    // the latency column compares cold against hot. With it, every arm is
    // measured on the same footing -- retrieval only -- and the query-embed
    // cost is reported once, as its own measured number, to be added back to
    // whichever arms actually use the dense channel.
    let mut embed_latencies = Vec::with_capacity(questions.len());
    for q in &questions {
        let t = std::time::Instant::now();
        embedder
            .embed(std::slice::from_ref(&q.text))
            .await
            .context("warm query embedding")?;
        embed_latencies.push(t.elapsed().as_millis());
    }
    embed_latencies.sort_unstable();
    let embed_p50 = percentile(&embed_latencies, 0.50);
    let embed_p90 = percentile(&embed_latencies, 0.90);
    println!(
        "  query embedding ({}): p50 {embed_p50}ms  p90 {embed_p90}ms over {} unique questions",
        cfg.embed.model,
        questions.len()
    );

    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;

    let reranker = CrossEncoder::new(&cfg.rerank.url, &cfg.rerank.model).ok();

    // Warm the cross-encoder too, for a reason the embedder cache does not
    // cover: llama.cpp's reranker breaks score ties differently on the very
    // first request after a cold start than on every request after it. That
    // is the M12/M14 non-reproducibility — two otherwise byte-identical
    // runs disagreeing on one evidence slot — and it lands entirely on
    // whichever reranked arm happens to run first. One throwaway pair,
    // outside every timed region, removes it.
    if let Some(r) = reranker.as_ref() {
        if let Err(e) = r
            .rerank("warm up the cross-encoder", &["warm up the cross-encoder".to_string()])
            .await
        {
            eprintln!("  reranker warm-up failed ({e}); first reranked arm may tie-break differently");
        }
    }

    // One index for the whole run: the cache is per tenant, and LoCoMo's
    // questions arrive grouped by conversation, so the graph arms pay one
    // SQLite read per conversation rather than one per question.
    let graph_index = GraphIndex::new();

    let mut results = Vec::new();
    for arm in arms(k) {
        if arm.rerank && reranker.is_none() {
            eprintln!("  {:<14} skipped: no reranker configured", arm.name);
            continue;
        }
        let mut retriever =
            Retriever::new(&embedder, &store, &ledger).with_config(arm.config.clone());
        if arm.rerank {
            retriever = retriever.with_reranker(reranker.as_ref().unwrap() as &dyn Reranker);
        }
        if arm.graph {
            retriever = retriever.with_graph(&graph_index);
        }

        let mut recall_sum = 0.0;
        let mut any_sum = 0.0;
        let mut mrr_sum = 0.0;
        let mut items_sum = 0.0;
        let mut latencies: Vec<u128> = Vec::with_capacity(questions.len());
        let mut per_cat: HashMap<u8, (usize, f64)> = HashMap::new();

        for q in &questions {
            let query = Recall {
                scope: ScopeFilter::tenant(&q.tenant).with_namespace("locomo"),
                text: q.text.clone(),
                budget: Budget {
                    k,
                    ..Default::default()
                },
                mode: Mode::Recall,
                kinds: None,
                as_of: None,
            };
            let (evidence, trace) = retriever
                .recall(&query)
                .await
                .with_context(|| format!("{}: recall {:?}", arm.name, q.text))?;

            let mut hit: HashSet<&String> = HashSet::new();
            let mut first_rank = None;
            for (rank, item) in evidence.items.iter().enumerate() {
                let Some(c) = coverage.of(&item.record_id) else {
                    continue;
                };
                let overlap: Vec<&String> = q.gold.iter().filter(|g| c.contains(*g)).collect();
                if !overlap.is_empty() && first_rank.is_none() {
                    first_rank = Some(rank + 1);
                }
                hit.extend(overlap);
            }

            let recall = hit.len() as f64 / q.gold.len() as f64;
            recall_sum += recall;
            any_sum += f64::from(u8::from(!hit.is_empty()));
            mrr_sum += first_rank.map_or(0.0, |r| 1.0 / r as f64);
            items_sum += evidence.items.len() as f64;
            latencies.push(trace.total_ms);

            let e = per_cat.entry(q.category).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += recall;
        }

        latencies.sort_unstable();
        let n = questions.len() as f64;
        let mut by_category: Vec<(u8, usize, f64)> = per_cat
            .into_iter()
            .map(|(c, (n, s))| (c, n, s / n as f64))
            .collect();
        by_category.sort_unstable_by_key(|r| r.0);

        let result = ArmResult {
            arm: arm.name.to_string(),
            questions: questions.len(),
            recall: recall_sum / n,
            any: any_sum / n,
            mrr: mrr_sum / n,
            items: items_sum / n,
            p50_ms: percentile(&latencies, 0.50),
            p90_ms: percentile(&latencies, 0.90),
            uses_dense: arm.config.channels != Channels::Lex,
            by_category,
        };
        println!(
            "  {:<14} recall@{k} {:.4}  any {:.4}  mrr {:.4}  items {:.2}  p50 {}ms  p90 {}ms",
            result.arm, result.recall, result.any, result.mrr, result.items, result.p50_ms,
            result.p90_ms
        );
        results.push(result);
    }

    Ok(AblationRun {
        arms: results,
        embed_p50_ms: embed_p50,
        embed_p90_ms: embed_p90,
    })
}

/// One cell of the width grid.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WidthPoint {
    pub prefetch_limit: u64,
    pub rerank_depth: usize,
    pub k: usize,
    /// Did this cell run the sufficiency selector (M21,
    /// `RetrieveConfig::select_sufficient`)?
    ///
    /// The selector stable-partitions the reranked pool — nothing is
    /// dropped — so it cannot change `pool_recall` and can only move
    /// `recall`. That makes it a direct read on how much of
    /// [`WidthPoint::truncation_loss`] is recoverable by choosing better,
    /// which is the quantity M25 and M26 localised and nobody has measured
    /// on a full population.
    pub select: bool,
    /// Gold-turn recall of the **emitted** evidence — what the reader sees.
    pub recall: f64,
    /// Share of questions whose emitted evidence holds **every** gold turn.
    ///
    /// `recall` averages over turns, which hides how rarely a multi-turn
    /// question gets all it needs: on `m19_locomo_full` multi-hop held
    /// every gold turn on 22% of rows, and scored 86% there against 36%
    /// with none held (the 2026-09-24 bottleneck review). L3 sizes `k`
    /// on this, not on the average.
    #[serde(default)]
    pub all: f64,
    /// Share of questions whose emitted evidence holds every gold turn
    /// **verbatim**: the turn's own text, as its episode renders it, appears
    /// in some emitted item (M66).
    ///
    /// `all` counts a record as holding a turn through I4 lineage, which is
    /// right for whole records and wrong for a turn window: a windowed
    /// episode still has the gold turn's lineage when the window cut the
    /// turn out. This is an independent text check. It does not credit a
    /// fact that paraphrases the turn, so read it beside `all`, not instead
    /// of it.
    #[serde(default)]
    pub turn_all: f64,
    /// Mean evidence tokens per question, as `compose` charged them. M66
    /// raises `k` and holds this flat.
    #[serde(default)]
    pub evidence_tokens: f64,
    /// Gold-turn recall of the **reranked pool**, before `compose`
    /// truncates to `k`. The ceiling `recall` is measured against.
    pub pool_recall: f64,
    pub any: f64,
    /// Records in the reranked pool, averaged. Shows when `prefetch_limit`
    /// stops binding because the store has nothing more to give.
    pub pool: f64,
    pub p50_ms: u128,
    pub p90_ms: u128,
    /// Fraction of queries on which the selector fell back to rank order.
    ///
    /// A selecting cell with a high value measured a mis-sized server, not
    /// a mechanism: `Selector::select` degrades to the unmodified order on
    /// any non-empty failure, and that is indistinguishable in the output
    /// from a selection that agreed with rank order. `print_width` refuses
    /// to report such a cell as a result.
    pub select_degraded: f64,
    /// The token budget this cell composed under (M28).
    ///
    /// `compose` truncates on `k` **or** on `max_tokens`, and every width
    /// cell M25–M27 ran used `Budget::default()`'s 2,048 without recording
    /// which one bound. Measured offline over the 162,181 live
    /// LongMemEval_S records, the mean costs **380 tokens** — so six of
    /// them is 2,282 and the *budget* binds before `k` does. On LoCoMo's
    /// 56-token mean it does not. Two limits, two different fixes.
    pub budget_tokens: usize,
    /// Mean candidates per query that had a free slot and were refused by
    /// the token budget. Zero means `k` is the only limit that bit.
    pub dropped_for_tokens: f64,
    /// Mean 1-based rank, within the reranked pool, of the records
    /// `compose` actually emitted (M29).
    ///
    /// **Separates two explanations of the same recall number.** A tight
    /// budget makes `compose` skip oversized candidates and keep scanning,
    /// so the emitted set is drawn from *deeper* in the ranked list. If a
    /// tight budget wins because of that deeper reach, this rises with
    /// `dropped_for_tokens`; if it wins because short records are simply
    /// better evidence, it does not and `mean_emitted_tokens` falls alone.
    ///
    /// **Only informative when `select` is off.** `RecallTrace::pool` is
    /// captured *after* the sufficiency selector's stable partition, so on
    /// a selecting cell this measures position in the selector's output,
    /// not in the reranker's — and `compose` then takes its head, pinning
    /// the value at `mean(1..k)` (3.5 at k = 6) whatever the selector did.
    /// M31 measured exactly that: 3.5 at both `(8192, off)` and
    /// `(8192, on)`. `mean_emitted_tokens` has no such defect and stays
    /// readable on every cell.
    pub mean_emitted_rank: f64,
    /// Mean `approx_tokens` of the emitted records, the other half of that
    /// separation.
    pub mean_emitted_tokens: f64,
}

impl WidthPoint {
    /// Gold that retrieval put in front of `compose` and `compose` then
    /// dropped. The half of the loss a better *selector* could recover.
    pub fn truncation_loss(&self) -> f64 {
        self.pool_recall - self.recall
    }

    /// Gold that never entered the reranked pool at all. The half of the
    /// loss only wider or better *retrieval* can recover.
    pub fn retrieval_loss(&self) -> f64 {
        1.0 - self.pool_recall
    }
}

/// M25 — the width grid: does `prefetch_limit` × `rerank_depth` bind?
///
/// # Why this is the knob left
///
/// Six retrieval mechanisms have been measured since M12 and six were
/// nulls, but **not one of them varied width**: every arm in [`arms`] runs
/// at `RetrieveConfig::default()`'s `prefetch_limit = 50` and
/// `rerank_depth = 25`. M21 and M22 both declared width out of scope, and
/// M24 then measured the consequence directly — decomposition grew the
/// fused pool 61 → 62 while `admitted` stayed **25 → 25**, because
/// `rerank_depth` binds. A mechanism that widens the candidate pool cannot
/// pay off through a window that does not widen with it.
///
/// # What the two recalls separate
///
/// `recall` is the emitted evidence; `pool_recall` is the reranked pool
/// before `compose` truncates. The gap between them is **truncation loss**
/// — gold retrieval found and `compose` dropped, recoverable by a better
/// selector. The gap from `pool_recall` to 1.0 is **retrieval loss** —
/// gold that never entered the pool, recoverable only by wider or better
/// retrieval. Every "is this retrieval's or the reader's" argument since M9
/// has needed that split; M21 got it by differencing two runs at different
/// `k`, and this gets it from one pass.
///
/// # No reader, and that is the point
///
/// Like [`ablate_locomo`], this scores against LoCoMo's own `dia_id` gold
/// by exact set arithmetic through the production segmenter. It needs an
/// embedder, a reranker and Qdrant — never a generation model. On a card
/// shared with a household voice assistant that is the difference between a
/// measurement and a deferral.
///
/// # The rule, fixed before the first cell ran
///
/// `RetrieveConfig::default()`'s `(50, 25)` changes **only if** some cell
/// improves emitted `recall@k` by **≥ 0.02 absolute** over that cell *and*
/// costs less than **2×** its p50 latency. A cell that raises `pool_recall`
/// without raising `recall` changes nothing about the defaults: it has
/// moved the loss from retrieval to truncation, which is a finding about
/// where to aim next and not a reason to pay for a wider pool the emitter
/// cannot use.
///
/// # Two corpora, two gold annotations
///
/// `corpus` selects both the questions and the [`GoldSource`]. LoCoMo
/// resolves record ids to `dia_id` turns through I4 lineage; LongMemEval_S
/// matches the `has_answer` turn's text prefix. The rest of the sweep —
/// the grid, the two recalls, the rule — is identical, which is the point:
/// M25 measured LoCoMo and found retrieval nearly solved there, and the
/// only way to know whether that generalises is to run the same instrument
/// on the corpus M21 says disagrees.
#[allow(clippy::too_many_arguments)]
pub async fn width_sweep(
    corpus: &str,
    path: &Path,
    collection: &str,
    ledger_path: &Path,
    units: usize,
    k: usize,
    grid: &[(u64, usize, bool, usize)],
    limit: Option<usize>,
    holdout: bool,
    dedupe_lineage: bool,
    turn_windows: Option<usize>,
) -> Result<Vec<WidthPoint>> {
    let cfg = MyelinConfig::load().context("load myelin config")?;
    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;

    // Each gold turn's text as its episode renders it, for `turn_all`. On
    // LoCoMo it comes from the same `turns_for` the build ingested, so a
    // turn is found exactly when its whole line was emitted. Keyed by
    // (tenant, dia_id): LoCoMo's ids restart in every conversation (`D1:3`
    // exists in all ten), so the id alone names ten different turns.
    // LongMemEval_S gold is already turn text.
    let mut turn_text: HashMap<(String, String), String> = HashMap::new();
    let (mut questions, source) = match corpus {
        "locomo" => {
            let split = select_split(path, units, holdout)?;
            for conv in &split {
                let tenant = format!("locomo/{}", conv.sample_id);
                for t in crate::build::turns_for(conv) {
                    turn_text.insert(
                        (tenant.clone(), t.source.doc.clone()),
                        format!("{}: {}", t.speaker, t.text),
                    );
                }
            }
            let coverage = Coverage::build(&split, &ledger).await?;
            (collect_questions(&split), GoldSource::Lineage(coverage))
        }
        "longmemeval_s" => (longmemeval_questions(path)?, GoldSource::TextPrefix),
        other => anyhow::bail!(
            "width needs a per-turn gold annotation, which only `locomo` (dia_id) and \
             `longmemeval_s` (has_answer) carry; {other:?} has none"
        ),
    };
    if let Some(n) = limit {
        questions.truncate(n);
    }

    let base_embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)
        .context("embedder client")?;
    let embedder = CachingEmbedder::new(&base_embedder);
    // Warm the cache outside every timed cell, for the reason
    // `ablate_locomo` does it: otherwise the first cell pays every later
    // cell's embeddings and the latency column compares cold to hot.
    for q in &questions {
        embedder
            .embed(std::slice::from_ref(&q.text))
            .await
            .context("warm query embedding")?;
    }

    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    let reranker = CrossEncoder::new(&cfg.rerank.url, &cfg.rerank.model)
        .context("width is a reranker question; a run without one measures nothing")?;
    // The cross-encoder ties differently on its very first request after a
    // cold start (M12/M14). One throwaway pair, outside every timed cell.
    if let Err(e) = reranker
        .rerank("warm up the cross-encoder", &["warm up the cross-encoder".to_string()])
        .await
    {
        eprintln!("  reranker warm-up failed ({e}); the first cell may tie-break differently");
    }
    // Wired unconditionally, exactly as `bench` wires it: the switch alone
    // is inert without a client, and wiring the client only under the
    // switch is the failure class M12, M14 and M20 each lost a run to.
    let llm = myelin_core::llm::openai::OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model)
        .context("selector client")?;

    let mut out = Vec::new();
    for &(prefetch_limit, rerank_depth, select, budget_tokens) in grid {
        let retriever = Retriever::new(&embedder, &store, &ledger)
            .with_reranker(&reranker as &dyn Reranker)
            .with_llm(&llm)
            .with_config(RetrieveConfig {
                select_sufficient: select,
                prefetch_limit,
                rerank_depth,
                // Every fused channel contributes the same depth, which is
                // what `graph_limit` documents; kept in step so a wider
                // prefetch is not silently one-channel-wide.
                graph_limit: prefetch_limit as usize,
                compose: myelin_core::pipeline::compose::ComposeConfig {
                    k,
                    max_tokens: budget_tokens,
                    dedupe_lineage,
                    ..Default::default()
                },
                turn_windows,
                ..Default::default()
            });

        let mut recall_sum = 0.0;
        let mut all_sum = 0.0;
        let mut turn_all_sum = 0.0;
        let mut evidence_tokens_sum = 0.0;
        let mut pool_recall_sum = 0.0;
        let mut any_sum = 0.0;
        let mut pool_sum = 0.0;
        let mut degraded = 0.0;
        let mut dropped = 0.0;
        let mut rank_sum = 0.0;
        let mut rank_n = 0.0;
        let mut emitted_tokens = 0.0;
        let mut emitted_n = 0.0;
        let mut lat: Vec<u128> = Vec::with_capacity(questions.len());

        for q in &questions {
            let query = Recall {
                scope: ScopeFilter::tenant(&q.tenant).with_namespace(corpus),
                text: q.text.clone(),
                budget: Budget { k, tokens: budget_tokens, ..Default::default() },
                mode: Mode::Recall,
                kinds: None,
                as_of: None,
            };
            let (evidence, trace) = retriever
                .recall(&query)
                .await
                .with_context(|| format!("({prefetch_limit},{rerank_depth}): recall {:?}", q.text))?;

            // Both recalls go through one function over the same shape, so
            // their difference is a quantity and not a comparison of two
            // implementations.
            let emitted: Vec<(Uuid, String)> = evidence
                .items
                .iter()
                .map(|i| (i.record_id, i.value.clone()))
                .collect();
            let held = gold_recall(&source, q, &emitted);
            recall_sum += held;
            all_sum += f64::from(u8::from(held >= 1.0 - f64::EPSILON));
            // A gold id with no turn text (two LoCoMo ids name no turn) is
            // never held, in every cell alike.
            let verbatim = q.gold.iter().all(|g| {
                let text = match corpus {
                    "locomo" => turn_text.get(&(q.tenant.clone(), g.clone())),
                    _ => Some(g),
                };
                text.is_some_and(|t| evidence.items.iter().any(|i| i.value.contains(t.as_str())))
            });
            turn_all_sum += f64::from(u8::from(verbatim));
            evidence_tokens_sum += evidence.tokens as f64;
            let pool = gold_recall(&source, q, &trace.pool);
            pool_recall_sum += pool;
            any_sum += f64::from(u8::from(pool > 0.0));
            pool_sum += trace.pool.len() as f64;
            // Only a FAILED call is the M27 failure class. A model that
            // answered and named nothing usable is a real answer at a low
            // steady rate (M32: 11/298 on a healthy server), and gating on
            // the union of the two refuses healthy cells.
            degraded += f64::from(u8::from(
                trace.select_degraded == myelin_core::pipeline::select::Degradation::CallFailed,
            ));
            dropped += trace.dropped_for_tokens as f64;
            // Rank of each emitted record inside the reranked pool, and
            // its size. `timeline`/`profile` items carry a nil record id
            // and are not in the pool, so they are skipped rather than
            // counted as rank 0.
            for item in &evidence.items {
                if let Some(pos) = trace.pool.iter().position(|(id, _)| *id == item.record_id) {
                    rank_sum += (pos + 1) as f64;
                    rank_n += 1.0;
                    emitted_tokens +=
                        myelin_core::pipeline::ingest::approx_tokens(&trace.pool[pos].1) as f64;
                    emitted_n += 1.0;
                }
            }
            lat.push(trace.total_ms);
        }

        lat.sort_unstable();
        let n = questions.len() as f64;
        let point = WidthPoint {
            prefetch_limit,
            rerank_depth,
            k,
            select,
            recall: recall_sum / n,
            all: all_sum / n,
            turn_all: turn_all_sum / n,
            evidence_tokens: evidence_tokens_sum / n,
            pool_recall: pool_recall_sum / n,
            any: any_sum / n,
            pool: pool_sum / n,
            p50_ms: percentile(&lat, 0.50),
            p90_ms: percentile(&lat, 0.90),
            select_degraded: degraded / n,
            budget_tokens,
            dropped_for_tokens: dropped / n,
            mean_emitted_rank: if rank_n > 0.0 { rank_sum / rank_n } else { 0.0 },
            mean_emitted_tokens: if emitted_n > 0.0 { emitted_tokens / emitted_n } else { 0.0 },
        };
        println!(
            "  prefetch {:<4} depth {:<4} select {:<5} budget {:<6} recall@{k} {:.4}  all {:.4}  \
             turn-all {:.4}  ev-tok {:.0}  pool {:.4}  (trunc {:.4} / miss {:.4})  tok-drops {:.2}  \
             rank {:.1}  rec-tok {:.0}  p50 {}ms",
            point.prefetch_limit,
            point.rerank_depth,
            point.select,
            point.budget_tokens,
            point.recall,
            point.all,
            point.turn_all,
            point.evidence_tokens,
            point.pool_recall,
            point.truncation_loss(),
            point.retrieval_loss(),
            point.dropped_for_tokens,
            point.mean_emitted_rank,
            point.mean_emitted_tokens,
            point.p50_ms
        );
        out.push(point);
    }
    Ok(out)
}

/// LongMemEval_S questions with their `has_answer` gold turns.
///
/// Scope mirrors `bench`'s exactly — tenant `lme_s/{question_id}`,
/// namespace `longmemeval_s` — because a sweep that retrieved from a
/// different scope than the benchmark would measure a different store.
///
/// Questions whose gold turns are all shorter than the matcher's floor are
/// dropped rather than scored 0: `coverage::is_found` refuses a unit under
/// 30 characters because short turns ("Thanks!") occur in almost any
/// evidence set, and counting those questions as misses would report the
/// corpus's filler as retrieval failure.
fn longmemeval_questions(dataset: &Path) -> Result<Vec<Question>> {
    let items = crate::datasets::longmemeval::load(dataset).context("load longmemeval_s")?;
    let mut out = Vec::new();
    let mut dropped = 0usize;
    for item in &items {
        let gold: HashSet<String> = item
            .haystack_sessions
            .iter()
            .flatten()
            .filter(|t| t.has_answer == Some(true))
            .map(|t| t.content.clone())
            .filter(|c| c.trim().chars().count() > 30)
            .collect();
        if gold.is_empty() {
            dropped += 1;
            continue;
        }
        out.push(Question {
            tenant: format!("lme_s/{}", item.question_id),
            text: item.question.clone(),
            category: crate::bench::question_type_code(&item.question_type),
            gold,
        });
    }
    eprintln!(
        "  longmemeval_s: {} questions with matchable gold turns, {dropped} dropped \
         (no has_answer turn over the matcher's 30-character floor)",
        out.len()
    );
    Ok(out)
}

/// How a corpus decides whether a record carries a question's gold.
///
/// The two benchmarks annotate differently and neither annotation converts
/// into the other, so the sweep carries both rather than picking one and
/// declaring the other out of scope.
enum GoldSource {
    /// LoCoMo cites evidence turns by `dia_id`, and I4 lineage resolves a
    /// record to every turn it can testify to. Exact set arithmetic: a
    /// record either covers the cited turn or it does not.
    Lineage(Coverage),
    /// LongMemEval_S flags the answer-bearing turn with `has_answer` and
    /// there is no id to resolve, so a gold unit counts when its 80-char
    /// prefix appears in a record. Reuses `coverage::is_found` verbatim —
    /// a second matcher here would make this instrument's numbers
    /// incomparable with every coverage number since M21.
    TextPrefix,
}

/// Fraction of a question's gold units these records cover.
///
/// Shared by the emitted set and the pool so the two recalls in a
/// [`WidthPoint`] are the same arithmetic over different inputs — the only
/// way their difference is a quantity rather than a comparison of two
/// implementations.
fn gold_recall(source: &GoldSource, q: &Question, records: &[(Uuid, String)]) -> f64 {
    if q.gold.is_empty() {
        return 0.0;
    }
    let covered = match source {
        GoldSource::Lineage(coverage) => {
            let mut hit: HashSet<&String> = HashSet::new();
            for (id, _) in records {
                if let Some(c) = coverage.of(id) {
                    hit.extend(q.gold.iter().filter(|g| c.contains(*g)));
                }
            }
            hit.len()
        }
        GoldSource::TextPrefix => {
            let texts: Vec<String> = records.iter().map(|(_, t)| t.clone()).collect();
            q.gold
                .iter()
                .filter(|g| crate::coverage::is_found(g, &texts))
                .count()
        }
    };
    covered as f64 / q.gold.len() as f64
}

/// The default grid: the shipped cell first, then each axis alone, then
/// both.
///
/// The shipped cell runs **first** so a partial sweep still carries its own
/// baseline — M22's lesson that an arm without a same-code base is not a
/// measurement, applied to a run that may be cut short by a GPU window
/// closing.
pub const WIDTH_GRID: [(u64, usize, bool, usize); 6] = [
    (50, 25, false, 2048),
    (50, 50, false, 2048),
    (100, 25, false, 2048),
    (100, 50, false, 2048),
    (200, 50, false, 2048),
    (200, 100, false, 2048),
];

/// The selection grid (M27): the two width extremes, each with and without
/// the sufficiency selector.
///
/// Four cells and not the full six-cell cross, because the question is not
/// "which width" — M25 and M26 answered that twice — but **how much of
/// `truncation_loss` a selector recovers, and whether it recovers more
/// where there is more to recover.** The shipped cell has the least
/// truncation loss on both corpora and `(200, 100)` the most, so the two
/// extremes bracket the effect; the middle cells would cost an hour each
/// to interpolate a line between two points.
///
/// Shipped-and-unselected runs first, for [`WIDTH_GRID`]'s reason: a sweep
/// cut short by a closing GPU window still carries its own baseline.
///
/// # The diagnostic rule, fixed before the first cell ran
///
/// [`width_verdict`]'s defaults rule still applies and still governs
/// `RetrieveConfig`, but for a selector cell its latency clause is
/// **dispositive by construction**: a model call per query cannot come in
/// under 2x a pure-retrieval p50, so a selecting cell can never be reported
/// as `Change`. That is correct — `PLAN.md` 7.1 forbids an LLM in `recall`
/// whatever this measures — and it means the VERDICT line is not the
/// finding here. The finding is the recall delta, read against this:
///
/// - **Selection recovers the truncation loss** if emitted `recall@k` rises
///   by >= 0.02 absolute at the shipped cell with `pool_recall` unchanged
///   (it must be: the selector stable-partitions, it never drops a
///   candidate). Recovering it does NOT flip a default — M21 measured this
///   same mechanism at exactly +0.0 judged inside `investigate` — it opens
///   a branch that only a bench arm can close.
/// - **It fails to recover it** if the rise is under 0.02, and then the
///   selection branch is closed on evidence rather than opinion, leaving
///   granularity the only remaining lever on the loss M25 and M26
///   localised.
/// - **The interaction is the second question.** M26 found the
///   cross-encoder's precision at the top 6 *degrades* as its pool widens
///   (emitted recall fell monotonically 0.8251 -> 0.7791 while pool recall
///   rose to 0.9976). If selection is what recovers that, the gain at
///   `(200, 100)` must exceed the gain at `(50, 25)`. If it does not, the
///   wide pool is not merely unhelpful but unrecoverable, and
///   `prefetch_limit`/`rerank_depth` should never be revisited again.
pub const SELECT_GRID: [(u64, usize, bool, usize); 4] = [
    (50, 25, false, 2048),
    (50, 25, true, 2048),
    (200, 100, false, 2048),
    (200, 100, true, 2048),
];

/// The budget grid (M28): the shipped width, swept over `max_tokens`.
///
/// `compose` truncates on `k` **or** on `max_tokens`, and M25-M27 all ran
/// at `Budget::default()`'s 2,048 without recording which bit. Measured
/// offline over 162,181 live LongMemEval_S records the mean costs 380
/// tokens, so six of them is 2,282 and the budget binds first; on LoCoMo's
/// 56-token mean it does not. This separates the two limits on the one
/// axis nobody has ever varied.
///
/// The shipped budget runs first, for [`WIDTH_GRID`]'s reason.
/// LoCoMo's shipped retrieval, one cell: the default width at the 4,096-token
/// budget `bench` composes under. For sweeping `k` and compose switches at
/// fixed retrieval (L3, M65).
pub const SHIPPED_GRID: [(u64, usize, bool, usize); 1] = [(50, 25, false, 4096)];

pub const BUDGET_GRID: [(u64, usize, bool, usize); 4] = [
    (50, 25, false, 2048),
    (50, 25, false, 4096),
    (50, 25, false, 8192),
    (50, 25, false, 16384),
];

/// The interaction grid (M31): the shipped width, crossed over the token
/// budget and the selector.
///
/// M26, M27 and M29 measured three mechanisms and found one fact under all
/// of them — **the cross-encoder's top-6 ordering is the bottleneck**.
/// Widening the pool costs −0.046, the selector reorders past the top-6
/// for +0.0585, and a tight budget skips past it for +0.0608. The last two
/// are nearly the same size, which is the reason to suspect they are the
/// same effect and cannot both own it.
///
/// A 2×2 is the smallest design that separates a main effect from an
/// interaction, and this is the only question on the board where two
/// shipped-off mechanisms are each credited with the same ~0.06.
///
/// **Pre-registered before the first cell, in `PLAN.md` §15:** if they are
/// the same escape mechanism, `select` at 8,192 recovers most of M29's
/// 0.0608 while `select` at 2,048 adds much less than its solo +0.0585 —
/// the interaction term is **negative and at least half the smaller main
/// effect**. If the interaction is ≈ 0 they are independent levers and the
/// combination is the new operating point.
///
/// Shipped-and-plain first, for [`WIDTH_GRID`]'s reason.
pub const INTERACTION_GRID: [(u64, usize, bool, usize); 4] = [
    (50, 25, false, 2048),
    (50, 25, true, 2048),
    (50, 25, false, 8192),
    (50, 25, true, 8192),
];

/// Absolute emitted-recall gain a cell must clear to change the defaults.
pub const RECALL_MARGIN: f64 = 0.02;
/// Fraction of degraded selector calls above which a cell is not a result.
///
/// Not zero: one refused request in five hundred is noise, and failing a
/// 90-minute sweep on it would be its own kind of unreliability. Low
/// enough that a systematically mis-sized server — which degrades on
/// *every* query, as a 100-candidate prompt does against an 8,192-token
/// slot — can never pass.
pub const MAX_DEGRADED: f64 = 0.02;
/// Multiple of the shipped cell's p50 latency a winner may cost.
pub const LATENCY_FACTOR: f64 = 2.0;

/// What the pre-registered rule says about a finished grid.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum WidthVerdict {
    /// No cell ran at `RetrieveConfig::default()`, so nothing can be
    /// compared to the shipped configuration and the rule is not evaluable.
    /// M22's lesson: an arm without a same-code base is not a measurement.
    NoBaseline { prefetch_limit: u64, rerank_depth: usize },
    /// A cell cleared both thresholds.
    Change {
        prefetch_limit: u64,
        rerank_depth: usize,
        from: f64,
        to: f64,
    },
    /// Nothing cleared them. `best` is the highest emitted recall on the
    /// grid, which may still be the shipped cell.
    Keep { base: f64, best: f64 },
    /// A selecting cell fell back to rank order on more than
    /// [`MAX_DEGRADED`] of its queries, so the grid measured a mis-sized
    /// server rather than a mechanism.
    ///
    /// Reported instead of a recall comparison because the comparison
    /// would be arithmetically fine and semantically empty: the selecting
    /// cell's evidence set IS the unselected cell's, so it reads as a
    /// clean null for a mechanism that never ran.
    Degraded {
        prefetch_limit: u64,
        rerank_depth: usize,
        rate: f64,
    },
}

/// Apply the rule [`width_sweep`]'s doc fixed before the first cell ran.
///
/// A function and not a branch inside the printer: a threshold a human
/// applies after seeing the table is not a pre-registration, and a rule
/// that cannot be unit-tested is a rule nobody can check was applied.
///
/// The shipped cell is found **by its values**, never by its position, so
/// reordering [`WIDTH_GRID`] cannot silently change what the comparison is
/// against — and `select` must be **off** for it, because
/// `RetrieveConfig::select_sufficient` ships off. Without that clause a
/// selector cell at the shipped width becomes its own baseline and every
/// arm is measured against the arm, which is the defect M21 added
/// `Ours::arm` to prevent and M23 found unfixed on the LME-V2 path.
pub fn width_verdict(points: &[WidthPoint]) -> WidthVerdict {
    // Before anything else: did the selector actually run where it was
    // asked to? A degraded cell's recall is real arithmetic over an
    // evidence set the mechanism never touched.
    if let Some(bad) = points
        .iter()
        .find(|p| p.select && p.select_degraded > MAX_DEGRADED)
    {
        return WidthVerdict::Degraded {
            prefetch_limit: bad.prefetch_limit,
            rerank_depth: bad.rerank_depth,
            rate: bad.select_degraded,
        };
    }
    let defaults = RetrieveConfig::default();
    let Some(base) = points.iter().find(|p| {
        p.prefetch_limit == defaults.prefetch_limit
            && p.rerank_depth == defaults.rerank_depth
            && p.select == defaults.select_sufficient
    }) else {
        return WidthVerdict::NoBaseline {
            prefetch_limit: defaults.prefetch_limit,
            rerank_depth: defaults.rerank_depth,
        };
    };
    let budget = base.p50_ms as f64 * LATENCY_FACTOR;
    let winner = points
        .iter()
        .filter(|p| !std::ptr::eq(*p, base))
        .filter(|p| p.recall - base.recall >= RECALL_MARGIN)
        .filter(|p| (p.p50_ms as f64) < budget)
        .max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap_or(std::cmp::Ordering::Equal));
    match winner {
        Some(w) => WidthVerdict::Change {
            prefetch_limit: w.prefetch_limit,
            rerank_depth: w.rerank_depth,
            from: base.recall,
            to: w.recall,
        },
        None => WidthVerdict::Keep {
            base: base.recall,
            best: points
                .iter()
                .map(|p| p.recall)
                .fold(f64::NEG_INFINITY, f64::max),
        },
    }
}

/// Render the width grid and print the verdict [`width_verdict`] computed.
pub fn print_width(points: &[WidthPoint], k: usize, corpus: &str) {
    println!("\n=== width grid — prefetch x rerank_depth, k={k}, corpus {corpus} ===");
    println!(
        "\n{:>8} {:>6} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10} {:>7} {:>8} {:>8}",
        "prefetch", "depth", "select", "budget", "recall", "pool", "trunc", "miss",
        "tok-drops", "rank", "rec-tok", "p50ms"
    );
    for p in points {
        println!(
            "{:>8} {:>6} {:>7} {:>7} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.2} {:>7.1} {:>8.0} {:>8}",
            p.prefetch_limit,
            p.rerank_depth,
            p.select,
            p.budget_tokens,
            p.recall,
            p.pool_recall,
            p.truncation_loss(),
            p.retrieval_loss(),
            p.dropped_for_tokens,
            p.mean_emitted_rank,
            p.mean_emitted_tokens,
            p.p50_ms
        );
    }
    println!(
        "\nrecall = gold-turn recall of the EMITTED evidence (what a reader would see).\n\
         pool   = the same arithmetic over the reranked pool, before compose truncates to k.\n\
         trunc  = pool - recall: gold retrieval found and compose dropped. A selector's to win.\n\
         miss   = 1 - pool: gold that never entered the pool. Only retrieval can win it.\n\
         tok-drops = candidates per query that had a free slot and were refused by the token\n\
         budget. Non-zero means `max_tokens` bound, NOT k, and the fix is compression.\n\
         {} Gold is decided by {}.",
        if points.iter().any(|p| p.select) {
            "Selector cells cost ONE model call per query; no cell generates an answer and\n             no cell is judged."
        } else {
            "No reader and no judge at all."
        },
        match corpus {
            "longmemeval_s" => "the has_answer turn's 80-char text prefix",
            _ => "LoCoMo's dia_id turns through I4 lineage",
        }
    );

    println!(
        "\nrule (fixed before the sweep): change the default only for >= +{RECALL_MARGIN:.2} \
         emitted recall at < {LATENCY_FACTOR:.0}x the shipped cell's p50."
    );
    match width_verdict(points) {
        WidthVerdict::NoBaseline { prefetch_limit, rerank_depth } => println!(
            "VERDICT not evaluable: no cell ran at the shipped ({prefetch_limit}, \
             {rerank_depth}), so nothing can be compared to it and the defaults do not move."
        ),
        WidthVerdict::Change { prefetch_limit, rerank_depth, from, to } => println!(
            "VERDICT change defaults to prefetch={prefetch_limit} depth={rerank_depth}: \
             recall {from:.4} -> {to:.4} (+{:.4})",
            to - from
        ),
        WidthVerdict::Degraded { prefetch_limit, rerank_depth, rate } => println!(
            "VERDICT not a result: the selector at prefetch={prefetch_limit} \
             depth={rerank_depth} fell back to rank order on {:.1}% of queries. \
             That cell measured a mis-sized server, not a mechanism — raise the \
             reader's per-slot context and re-run it.",
            rate * 100.0
        ),
        WidthVerdict::Keep { base, best } => println!(
            "VERDICT defaults stay. Best cell on the grid is {best:.4} against the shipped \
             {base:.4} (+{:.4}), which does not clear +{RECALL_MARGIN:.2}.",
            best - base
        ),
    }

    let defaults = RetrieveConfig::default();
    if let Some(base) = points.iter().find(|p| {
        p.prefetch_limit == defaults.prefetch_limit && p.rerank_depth == defaults.rerank_depth
    }) {
        println!(
            "\nwhere the loss lives at the shipped cell: {:.1}% of gold is truncated by \
             compose, {:.1}% never reaches the pool.",
            base.truncation_loss() * 100.0,
            base.retrieval_loss() * 100.0
        );
    }
}

/// Persist a finished grid, so the numbers survive the terminal that
/// printed them.
///
/// `standing` does not read this: width is a retrieval diagnostic and not a
/// published metric. It exists because "a run artifact that does not record
/// what produced it is not reproducible" applies to retrieval measurements
/// too, and because the first sweep took 72 minutes of a contended card.
pub fn write_width(points: &[WidthPoint], out: &Path) -> Result<()> {
    std::fs::create_dir_all(out).with_context(|| format!("mkdir {}", out.display()))?;
    let path = out.join("width.json");
    let body = serde_json::json!({
        "schema": 1,
        "grid": points,
        "verdict": width_verdict(points),
        "rule": {
            "recall_margin": RECALL_MARGIN,
            "latency_factor": LATENCY_FACTOR,
        },
    });
    let mut text = serde_json::to_string_pretty(&body)?;
    text.push('\n');
    std::fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
    println!("\nwrote {}", path.display());
    Ok(())
}

/// Render the §8 table. Deltas are against `hybrid_k60`, the configured
/// default, so each row answers "what does changing this cost?".
pub fn print_table(run: &AblationRun, k: usize) {
    let results = &run.arms;
    let baseline = results.iter().find(|r| r.arm == "hybrid_k60");
    println!("\n=== M4 ablation: retrieval recall@{k} on the LoCoMo dev split ===");
    println!(
        "{:<15} {:>8} {:>8} {:>8} {:>7} {:>8} {:>8} {:>8} {:>9}",
        "arm", "recall", "Δ", "any", "mrr", "items", "p50ms", "p90ms", "e2e p50"
    );
    for r in results {
        let delta = baseline.map_or(0.0, |b| r.recall - b.recall);
        let e2e = r.p50_ms + if r.uses_dense { run.embed_p50_ms } else { 0 };
        println!(
            "{:<15} {:>8.4} {:>+8.4} {:>8.4} {:>7.4} {:>8.2} {:>8} {:>8} {:>9}",
            r.arm, r.recall, delta, r.any, r.mrr, r.items, r.p50_ms, r.p90_ms, e2e
        );
    }
    println!(
        "latency columns exclude query embedding (p50 {}ms, p90 {}ms); `e2e p50` adds it back \
         for the arms that use the dense channel.",
        run.embed_p50_ms, run.embed_p90_ms
    );
    if let Some(cats) = results.first().map(|r| &r.by_category) {
        println!("\nper-category recall (category: n)");
        for (cat, n, _) in cats {
            print!("  cat{cat} (n={n}):");
            for r in results {
                let v = r
                    .by_category
                    .iter()
                    .find(|c| c.0 == *cat)
                    .map_or(0.0, |c| c.2);
                print!(" {}={:.3}", r.arm, v);
            }
            println!();
        }
    }
}

/// One point on the marginal-step-value curve (`PLAN.md` M7).
#[derive(Debug, Clone)]
pub struct StepPoint {
    pub max_steps: usize,
    pub recall: f64,
    pub any: f64,
    /// Steps actually taken. The loop stops early on sufficiency or on a
    /// barren search, so this is well below `max_steps` at the top of the
    /// sweep — which is the interesting part: budget granted is not budget
    /// spent.
    pub mean_steps: f64,
    pub mean_pool: f64,
    pub barren_fraction: f64,
    pub conflict_fraction: f64,
    pub p50_ms: u128,
    pub p90_ms: u128,
}

/// M7 — accuracy and cost as a function of the step budget.
///
/// The exit criterion asks for the *marginal* value of a step, so the table
/// reports the delta against `max_steps = 1` (which is `recall` with an
/// extra reflect call) rather than absolute numbers alone. A step that buys
/// no recall and costs two seconds is a step the operating point should not
/// pay for.
///
/// Scored by the same deterministic evidence-coverage metric as the
/// ablation: no reader, no judge. Adding a reader here would measure the
/// reader's tolerance for extra context, not the loop's ability to find it.
#[allow(clippy::too_many_arguments)]
pub async fn investigate_curve(
    path: &Path,
    collection: &str,
    ledger_path: &Path,
    units: usize,
    k: usize,
    steps: &[usize],
    limit: Option<usize>,
    holdout: bool,
) -> Result<Vec<StepPoint>> {
    let cfg = MyelinConfig::load().context("load myelin config")?;
    let split = select_split(path, units, holdout)?;

    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;
    let coverage = Coverage::build(&split, &ledger).await?;
    let mut questions = collect_questions(&split);
    if let Some(n) = limit {
        questions.truncate(n);
    }

    let base_embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)
        .context("embedder client")?;
    let embedder = CachingEmbedder::new(&base_embedder);
    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    let reranker = CrossEncoder::new(&cfg.rerank.url, &cfg.rerank.model).ok();
    let llm = myelin_core::llm::openai::OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model)
        .context("controller client")?;

    let mut retriever = Retriever::new(&embedder, &store, &ledger);
    if let Some(r) = &reranker {
        retriever = retriever.with_reranker(r as &dyn Reranker);
    }

    let mut out = Vec::new();
    for &max_steps in steps {
        let mut recall_sum = 0.0;
        let mut any_sum = 0.0;
        let mut steps_sum = 0.0;
        let mut pool_sum = 0.0;
        let mut barren = 0.0;
        let mut conflicts = 0.0;
        let mut lat: Vec<u128> = Vec::with_capacity(questions.len());

        for q in &questions {
            let query = Recall {
                scope: ScopeFilter::tenant(&q.tenant).with_namespace("locomo"),
                text: q.text.clone(),
                budget: Budget {
                    k,
                    tokens: 2048,
                    max_steps,
                },
                mode: Mode::Investigate,
                kinds: None,
                as_of: None,
            };
            let (evidence, trace) = Investigator::new(&llm, &retriever)
                .with_config(InvestigateConfig {
                    max_steps: max_steps.max(1),
                    ..Default::default()
                })
                .investigate(&query)
                .await
                .with_context(|| format!("investigate {:?}", q.text))?;

            let mut hit: HashSet<&String> = HashSet::new();
            for item in &evidence.items {
                if let Some(c) = coverage.of(&item.record_id) {
                    hit.extend(q.gold.iter().filter(|g| c.contains(*g)));
                }
            }
            recall_sum += hit.len() as f64 / q.gold.len() as f64;
            any_sum += f64::from(u8::from(!hit.is_empty()));
            steps_sum += trace.steps as f64;
            pool_sum += trace.pool as f64;
            barren += trace.barren_steps as f64;
            conflicts += f64::from(u8::from(trace.conflicts_seen > 0));
            lat.push(trace.total_ms);
        }

        lat.sort_unstable();
        let n = questions.len() as f64;
        let point = StepPoint {
            max_steps,
            recall: recall_sum / n,
            any: any_sum / n,
            mean_steps: steps_sum / n,
            mean_pool: pool_sum / n,
            barren_fraction: if steps_sum > 0.0 { barren / steps_sum } else { 0.0 },
            conflict_fraction: conflicts / n,
            p50_ms: percentile(&lat, 0.50),
            p90_ms: percentile(&lat, 0.90),
        };
        println!(
            "  max_steps={:<2} recall {:.4}  any {:.4}  steps {:.2}  pool {:.1}  p50 {}ms  p90 {}ms",
            point.max_steps,
            point.recall,
            point.any,
            point.mean_steps,
            point.mean_pool,
            point.p50_ms,
            point.p90_ms
        );
        out.push(point);
    }
    Ok(out)
}

pub fn print_step_curve(points: &[StepPoint], k: usize) {
    println!("\n=== M7 marginal step value, recall@{k} on the LoCoMo dev split ===");
    println!(
        "{:<10} {:>8} {:>9} {:>8} {:>7} {:>7} {:>8} {:>8} {:>8}",
        "max_steps", "recall", "Δ vs 1", "any", "steps", "pool", "barren", "p50ms", "p90ms"
    );
    let base = points.first().map(|p| p.recall).unwrap_or(0.0);
    for p in points {
        println!(
            "{:<10} {:>8.4} {:>+9.4} {:>8.4} {:>7.2} {:>7.1} {:>7.0}% {:>8} {:>8}",
            p.max_steps,
            p.recall,
            p.recall - base,
            p.any,
            p.mean_steps,
            p.mean_pool,
            p.barren_fraction * 100.0,
            p.p50_ms,
            p.p90_ms
        );
    }
    // The marginal number M7 actually asks for: recall bought per extra
    // second of p50 latency, against the 1-step point.
    println!("\nmarginal value (recall gained per extra second of p50 latency, vs max_steps=1)");
    let base_ms = points.first().map(|p| p.p50_ms).unwrap_or(0);
    for p in points.iter().skip(1) {
        let dt = (p.p50_ms.saturating_sub(base_ms)) as f64 / 1000.0;
        let dr = p.recall - base;
        println!(
            "  {:<2} -> {:+.4} recall for {:+.2}s  =  {}",
            p.max_steps,
            dr,
            dt,
            if dt > 0.0 {
                format!("{:+.4} recall/s", dr / dt)
            } else {
                "no extra latency".to_string()
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(prefetch: u64, depth: usize, recall: f64, pool: f64, p50: u128) -> WidthPoint {
        WidthPoint {
            prefetch_limit: prefetch,
            rerank_depth: depth,
            k: 6,
            select: false,
            recall,
            all: 0.0,
            turn_all: 0.0,
            evidence_tokens: 0.0,
            pool_recall: pool,
            any: 1.0,
            pool: depth as f64,
            p50_ms: p50,
            p90_ms: p50 * 2,
            select_degraded: 0.0,
            budget_tokens: Budget::default().tokens,
            dropped_for_tokens: 0.0,
            mean_emitted_rank: 0.0,
            mean_emitted_tokens: 0.0,
        }
    }

    /// The shipped cell of the grid, whatever `RetrieveConfig::default()`
    /// currently is — so the tests track the defaults rather than pinning a
    /// copy of them that can silently diverge.
    fn shipped(recall: f64, pool: f64, p50: u128) -> WidthPoint {
        let d = RetrieveConfig::default();
        cell(d.prefetch_limit, d.rerank_depth, recall, pool, p50)
    }

    /// The loss decomposition has to be exhaustive, or "where does the loss
    /// live" is answered by a number that does not add up.
    #[test]
    fn the_two_losses_and_the_recall_partition_the_gold() {
        let p = shipped(0.40, 0.75, 100);
        assert!((p.recall + p.truncation_loss() + p.retrieval_loss() - 1.0).abs() < 1e-12);
        assert!((p.truncation_loss() - 0.35).abs() < 1e-12);
        assert!((p.retrieval_loss() - 0.25).abs() < 1e-12);
    }

    /// A cell that clears both thresholds wins, and the verdict names it.
    #[test]
    fn a_cell_that_clears_both_thresholds_changes_the_defaults() {
        let grid = vec![
            shipped(0.40, 0.70, 100),
            cell(200, 100, 0.40 + RECALL_MARGIN, 0.90, 150),
        ];
        assert_eq!(
            width_verdict(&grid),
            WidthVerdict::Change {
                prefetch_limit: 200,
                rerank_depth: 100,
                from: 0.40,
                to: 0.40 + RECALL_MARGIN,
            }
        );
    }

    /// **The margin is a floor, not a hint.** A cell one ulp short of it
    /// loses, because "nearly cleared a pre-registered bar" is how a
    /// measured null becomes a shipped default.
    #[test]
    fn a_cell_just_under_the_margin_does_not() {
        let grid = vec![
            shipped(0.40, 0.70, 100),
            cell(200, 100, 0.40 + RECALL_MARGIN - 0.0001, 0.90, 150),
        ];
        assert!(matches!(width_verdict(&grid), WidthVerdict::Keep { .. }));
    }

    /// Latency is a real constraint and not a footnote: a cell that buys
    /// recall at more than the budget is refused however well it scores.
    #[test]
    fn a_cell_over_the_latency_budget_is_refused() {
        let grid = vec![
            shipped(0.40, 0.70, 100),
            cell(200, 100, 0.90, 0.95, (100.0 * LATENCY_FACTOR) as u128),
        ];
        assert!(
            matches!(width_verdict(&grid), WidthVerdict::Keep { best, .. } if (best - 0.90).abs() < 1e-12),
            "the cell is still reported as the best on the grid, just not adopted"
        );
    }

    /// **Pool recall is not the quantity the rule is about.** A cell that
    /// retrieves far more gold but emits no more of it has moved the loss
    /// from retrieval to truncation — a finding about where to aim next,
    /// not a reason to pay for a wider pool the emitter cannot use.
    #[test]
    fn a_cell_that_only_raises_pool_recall_changes_nothing() {
        let grid = vec![
            shipped(0.40, 0.55, 100),
            cell(200, 100, 0.40, 0.99, 120),
        ];
        assert!(matches!(width_verdict(&grid), WidthVerdict::Keep { .. }));
    }

    /// The baseline is found by its values. Reordering the grid must not
    /// change what the rule compares against — M22's defect was exactly a
    /// selection rule picking the wrong row.
    #[test]
    fn the_baseline_is_found_by_value_not_by_position() {
        let winner = cell(200, 100, 0.40 + RECALL_MARGIN, 0.90, 150);
        let base = shipped(0.40, 0.70, 100);
        let forward = width_verdict(&[base.clone(), winner.clone()]);
        let reversed = width_verdict(&[winner, base]);
        assert_eq!(forward, reversed);
        assert!(matches!(forward, WidthVerdict::Change { .. }));
    }

    /// A grid with no shipped cell is not evaluable, and must say so rather
    /// than silently comparing two arms to each other.
    #[test]
    fn a_grid_without_the_shipped_cell_is_not_evaluable() {
        let d = RetrieveConfig::default();
        let grid = vec![cell(200, 100, 0.90, 0.95, 120)];
        assert_eq!(
            width_verdict(&grid),
            WidthVerdict::NoBaseline {
                prefetch_limit: d.prefetch_limit,
                rerank_depth: d.rerank_depth,
            }
        );
    }

    /// The shipped grid must contain the shipped cell, or every sweep it
    /// drives reports `NoBaseline`.
    #[test]
    fn the_default_grid_contains_the_shipped_cell() {
        let d = RetrieveConfig::default();
        assert!(
            WIDTH_GRID.contains(&(d.prefetch_limit, d.rerank_depth, d.select_sufficient, Budget::default().tokens)),
            "WIDTH_GRID {WIDTH_GRID:?} must include the shipped ({}, {})",
            d.prefetch_limit,
            d.rerank_depth
        );
        assert_eq!(
            WIDTH_GRID[0],
            (d.prefetch_limit, d.rerank_depth, d.select_sufficient, Budget::default().tokens),
            "and it must run first, so a sweep cut short by a closing GPU window \
             still carries its own baseline"
        );
    }

    /// **A selector cell is never the baseline.** `select_sufficient`
    /// ships off, so a selecting cell at the shipped width is an arm; let
    /// it match and every arm is measured against the arm, which is
    /// exactly the defect M21 added `Ours::arm` for and M23 found still
    /// unfixed on the LME-V2 path.
    #[test]
    fn a_selecting_cell_at_the_shipped_width_is_not_the_baseline() {
        let d = RetrieveConfig::default();
        let mut selecting = shipped(0.95, 0.99, 100);
        selecting.select = true;
        // Only the selecting cell exists at the shipped width: there is no
        // baseline at all, and the rule must say so rather than compare
        // the arm to itself.
        assert_eq!(
            width_verdict(&[selecting.clone()]),
            WidthVerdict::NoBaseline {
                prefetch_limit: d.prefetch_limit,
                rerank_depth: d.rerank_depth,
            }
        );
        // With a real baseline present, the selector is judged against it.
        let verdict = width_verdict(&[shipped(0.80, 0.99, 100), selecting]);
        assert_eq!(
            verdict,
            WidthVerdict::Change {
                prefetch_limit: d.prefetch_limit,
                rerank_depth: d.rerank_depth,
                from: 0.80,
                to: 0.95,
            }
        );
    }

    /// The selection grid must carry the shipped-and-unselected cell, or
    /// every sweep it drives reports `NoBaseline`.
    #[test]
    fn the_select_grid_contains_the_shipped_unselected_cell() {
        let d = RetrieveConfig::default();
        assert_eq!(
            SELECT_GRID[0],
            (d.prefetch_limit, d.rerank_depth, d.select_sufficient, Budget::default().tokens)
        );
        // And it pairs every width it visits, or the selector's effect is
        // confounded with the width's.
        let mut widths: Vec<(u64, usize)> =
            SELECT_GRID.iter().map(|(p, d, _, _)| (*p, *d)).collect();
        widths.sort_unstable();
        widths.dedup();
        for w in widths {
            for on in [false, true] {
                assert!(
                    SELECT_GRID.contains(&(w.0, w.1, on, Budget::default().tokens)),
                    "width {w:?} is missing its select={on} twin"
                );
            }
        }
    }

    /// **A degraded selector cell is not a result.** This is the exact
    /// shape that nearly shipped: the selecting cell's recall equals the
    /// unselected cell's, which reads as a clean null for a mechanism that
    /// never ran, because every request was refused for exceeding the
    /// reader's per-slot context.
    #[test]
    fn a_degraded_selector_cell_is_refused_rather_than_reported() {
        let d = RetrieveConfig::default();
        let base = shipped(0.8251, 0.9665, 410);
        let mut selecting = shipped(0.8251, 0.9665, 1385);
        selecting.select = true;
        selecting.select_degraded = 1.0;
        assert_eq!(
            width_verdict(&[base, selecting]),
            WidthVerdict::Degraded {
                prefetch_limit: d.prefetch_limit,
                rerank_depth: d.rerank_depth,
                rate: 1.0,
            },
            "identical recall plus a total fallback rate is a mis-sized server, not a null"
        );
    }

    /// A trickle of fallbacks is noise, not a broken run: failing a
    /// 90-minute sweep on one refused request would be its own kind of
    /// unreliability.
    #[test]
    fn a_trickle_of_fallbacks_still_reports_a_result() {
        let base = shipped(0.8251, 0.9665, 410);
        let mut selecting = shipped(0.8836, 0.9665, 1385);
        selecting.select = true;
        selecting.select_degraded = MAX_DEGRADED;
        let verdict = width_verdict(&[base, selecting]);
        assert!(
            !matches!(verdict, WidthVerdict::Degraded { .. }),
            "at exactly the threshold the cell is still a result: {verdict:?}"
        );
        // And it is `Keep`, not `Change`: the selector's model call puts it
        // over the 2x latency budget by construction, which is the
        // pre-registered reason the VERDICT line is not the finding for a
        // selecting cell. The recall delta in the table is.
        assert!(matches!(verdict, WidthVerdict::Keep { .. }), "{verdict:?}");
    }

    /// Every grid must contain the shipped cell first, or the sweep it
    /// drives reports `NoBaseline` and an hour of GPU buys nothing.
    ///
    /// Covers all four by construction rather than one at a time: a fifth
    /// grid added without its baseline fails here instead of at the end of
    /// a run.
    #[test]
    fn every_grid_leads_with_the_shipped_cell() {
        let d = RetrieveConfig::default();
        let shipped = (
            d.prefetch_limit,
            d.rerank_depth,
            d.select_sufficient,
            Budget::default().tokens,
        );
        for (name, grid) in [
            ("WIDTH_GRID", &WIDTH_GRID[..]),
            ("SELECT_GRID", &SELECT_GRID[..]),
            ("BUDGET_GRID", &BUDGET_GRID[..]),
            ("INTERACTION_GRID", &INTERACTION_GRID[..]),
        ] {
            assert_eq!(grid[0], shipped, "{name} must lead with the shipped cell");
        }
    }

    /// The interaction grid must be a complete 2x2, or it measures a main
    /// effect and calls it an interaction.
    #[test]
    fn the_interaction_grid_is_a_complete_two_by_two() {
        let budgets: std::collections::BTreeSet<usize> =
            INTERACTION_GRID.iter().map(|c| c.3).collect();
        assert_eq!(budgets.len(), 2, "two budgets: {budgets:?}");
        // One width throughout, or budget and width are confounded.
        let widths: std::collections::BTreeSet<(u64, usize)> =
            INTERACTION_GRID.iter().map(|c| (c.0, c.1)).collect();
        assert_eq!(widths.len(), 1, "one width: {widths:?}");
        for b in budgets {
            for on in [false, true] {
                assert!(
                    INTERACTION_GRID
                        .iter()
                        .any(|c| c.3 == b && c.2 == on),
                    "missing cell budget={b} select={on}"
                );
            }
        }
    }

    /// LongMemEval_S is scored by text prefix, and the two recalls in a
    /// `WidthPoint` must be the same arithmetic over different inputs or
    /// their difference is not a quantity.
    #[test]
    fn the_text_prefix_source_scores_both_inputs_identically() {
        let gold = "Dana finally adopted the rescue dog she had been visiting for months now";
        let q = Question {
            tenant: "lme_s/q1".into(),
            text: "which dog".into(),
            category: 1,
            gold: HashSet::from([gold.to_string()]),
        };
        let carrying = vec![(Uuid::nil(), format!("[2026-09-20] {gold} and named her Biscuit"))];
        let not = vec![(Uuid::nil(), "Ravi relocated to Lisbon".to_string())];
        assert_eq!(gold_recall(&GoldSource::TextPrefix, &q, &carrying), 1.0);
        assert_eq!(gold_recall(&GoldSource::TextPrefix, &q, &not), 0.0);
        // The date stamp `compose` prepends must not defeat the match:
        // every emitted value carries one and the pool's values do not.
        assert_eq!(
            gold_recall(&GoldSource::TextPrefix, &q, &carrying),
            gold_recall(
                &GoldSource::TextPrefix,
                &q,
                &[(Uuid::nil(), format!("{gold} and named her Biscuit"))]
            ),
            "a stamped record and a bare one must score the same"
        );
    }

    /// A question with no gold scores zero rather than dividing by zero.
    #[test]
    fn a_question_with_no_gold_is_zero_not_nan() {
        let q = Question {
            tenant: "t".into(),
            text: "q".into(),
            category: 1,
            gold: HashSet::new(),
        };
        let r = gold_recall(&GoldSource::TextPrefix, &q, &[(Uuid::nil(), "anything".into())]);
        assert_eq!(r, 0.0);
        assert!(r.is_finite());
    }

    /// `rerank_depth` may never exceed `prefetch_limit`: the reranked pool
    /// cannot be wider than the prefetch that fills it, and a grid cell
    /// that asks for it measures the prefetch instead of the depth.
    #[test]
    fn no_grid_cell_asks_for_a_pool_wider_than_its_prefetch() {
        for (prefetch, depth, _, _) in WIDTH_GRID
            .iter()
            .chain(SELECT_GRID.iter())
            .chain(BUDGET_GRID.iter())
            .copied()
        {
            assert!(
                depth as u64 <= prefetch,
                "({prefetch}, {depth}) reranks deeper than it prefetches"
            );
        }
    }
}
