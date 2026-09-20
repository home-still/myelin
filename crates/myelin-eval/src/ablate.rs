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
    /// Gold-turn recall of the **emitted** evidence — what the reader sees.
    pub recall: f64,
    /// Gold-turn recall of the **reranked pool**, before `compose`
    /// truncates to `k`. The ceiling `recall` is measured against.
    pub pool_recall: f64,
    pub any: f64,
    /// Records in the reranked pool, averaged. Shows when `prefetch_limit`
    /// stops binding because the store has nothing more to give.
    pub pool: f64,
    pub p50_ms: u128,
    pub p90_ms: u128,
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
#[allow(clippy::too_many_arguments)]
pub async fn width_sweep(
    path: &Path,
    collection: &str,
    ledger_path: &Path,
    units: usize,
    k: usize,
    grid: &[(u64, usize)],
    limit: Option<usize>,
    holdout: bool,
) -> Result<Vec<WidthPoint>> {
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

    let mut out = Vec::new();
    for &(prefetch_limit, rerank_depth) in grid {
        let retriever = Retriever::new(&embedder, &store, &ledger)
            .with_reranker(&reranker as &dyn Reranker)
            .with_config(RetrieveConfig {
                prefetch_limit,
                rerank_depth,
                // Every fused channel contributes the same depth, which is
                // what `graph_limit` documents; kept in step so a wider
                // prefetch is not silently one-channel-wide.
                graph_limit: prefetch_limit as usize,
                compose: myelin_core::pipeline::compose::ComposeConfig {
                    k,
                    ..Default::default()
                },
                ..Default::default()
            });

        let mut recall_sum = 0.0;
        let mut pool_recall_sum = 0.0;
        let mut any_sum = 0.0;
        let mut pool_sum = 0.0;
        let mut lat: Vec<u128> = Vec::with_capacity(questions.len());

        for q in &questions {
            let query = Recall {
                scope: ScopeFilter::tenant(&q.tenant).with_namespace("locomo"),
                text: q.text.clone(),
                budget: Budget { k, ..Default::default() },
                mode: Mode::Recall,
                kinds: None,
            };
            let (evidence, trace) = retriever
                .recall(&query)
                .await
                .with_context(|| format!("({prefetch_limit},{rerank_depth}): recall {:?}", q.text))?;

            recall_sum += gold_recall(&coverage, q, evidence.items.iter().map(|i| i.record_id));
            let pool = gold_recall(&coverage, q, trace.pool_ids.iter().copied());
            pool_recall_sum += pool;
            any_sum += f64::from(u8::from(pool > 0.0));
            pool_sum += trace.pool_ids.len() as f64;
            lat.push(trace.total_ms);
        }

        lat.sort_unstable();
        let n = questions.len() as f64;
        let point = WidthPoint {
            prefetch_limit,
            rerank_depth,
            k,
            recall: recall_sum / n,
            pool_recall: pool_recall_sum / n,
            any: any_sum / n,
            pool: pool_sum / n,
            p50_ms: percentile(&lat, 0.50),
            p90_ms: percentile(&lat, 0.90),
        };
        println!(
            "  prefetch {:<4} depth {:<4} recall@{k} {:.4}  pool {:.4}  (trunc {:.4} / miss {:.4})  \
             |pool| {:.1}  p50 {}ms",
            point.prefetch_limit,
            point.rerank_depth,
            point.recall,
            point.pool_recall,
            point.truncation_loss(),
            point.retrieval_loss(),
            point.pool,
            point.p50_ms
        );
        out.push(point);
    }
    Ok(out)
}

/// Fraction of a question's gold turns covered by these records.
///
/// Shared by the emitted set and the pool so the two recalls in a
/// [`WidthPoint`] are the same arithmetic over different inputs — the only
/// way their difference is a quantity rather than a comparison of two
/// implementations.
fn gold_recall(
    coverage: &Coverage,
    q: &Question,
    records: impl Iterator<Item = Uuid>,
) -> f64 {
    if q.gold.is_empty() {
        return 0.0;
    }
    let mut hit: HashSet<&String> = HashSet::new();
    for id in records {
        if let Some(c) = coverage.of(&id) {
            hit.extend(q.gold.iter().filter(|g| c.contains(*g)));
        }
    }
    hit.len() as f64 / q.gold.len() as f64
}

/// The default grid: the shipped cell first, then each axis alone, then
/// both.
///
/// The shipped cell runs **first** so a partial sweep still carries its own
/// baseline — M22's lesson that an arm without a same-code base is not a
/// measurement, applied to a run that may be cut short by a GPU window
/// closing.
pub const WIDTH_GRID: [(u64, usize); 6] = [
    (50, 25),
    (50, 50),
    (100, 25),
    (100, 50),
    (200, 50),
    (200, 100),
];

/// Absolute emitted-recall gain a cell must clear to change the defaults.
pub const RECALL_MARGIN: f64 = 0.02;
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
}

/// Apply the rule [`width_sweep`]'s doc fixed before the first cell ran.
///
/// A function and not a branch inside the printer: a threshold a human
/// applies after seeing the table is not a pre-registration, and a rule
/// that cannot be unit-tested is a rule nobody can check was applied.
///
/// The shipped cell is found **by its values**, never by its position, so
/// reordering [`WIDTH_GRID`] cannot silently change what the comparison is
/// against.
pub fn width_verdict(points: &[WidthPoint]) -> WidthVerdict {
    let defaults = RetrieveConfig::default();
    let Some(base) = points.iter().find(|p| {
        p.prefetch_limit == defaults.prefetch_limit && p.rerank_depth == defaults.rerank_depth
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
pub fn print_width(points: &[WidthPoint], k: usize) {
    println!("\n=== M25 width grid — prefetch x rerank_depth, k={k} ===");
    println!(
        "\n{:>8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "prefetch", "depth", "recall", "pool", "trunc", "miss", "|pool|", "p50ms"
    );
    for p in points {
        println!(
            "{:>8} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>8.1} {:>8}",
            p.prefetch_limit,
            p.rerank_depth,
            p.recall,
            p.pool_recall,
            p.truncation_loss(),
            p.retrieval_loss(),
            p.pool,
            p.p50_ms
        );
    }
    println!(
        "\nrecall = gold-turn recall of the EMITTED evidence (what a reader would see).\n\
         pool   = the same arithmetic over the reranked pool, before compose truncates to k.\n\
         trunc  = pool - recall: gold retrieval found and compose dropped. A selector's to win.\n\
         miss   = 1 - pool: gold that never entered the pool. Only retrieval can win it.\n\
         No reader and no judge: LoCoMo's dia_id gold is decided by set arithmetic."
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
            recall,
            pool_recall: pool,
            any: 1.0,
            pool: depth as f64,
            p50_ms: p50,
            p90_ms: p50 * 2,
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
            WIDTH_GRID.contains(&(d.prefetch_limit, d.rerank_depth)),
            "WIDTH_GRID {WIDTH_GRID:?} must include the shipped ({}, {})",
            d.prefetch_limit,
            d.rerank_depth
        );
        assert_eq!(
            WIDTH_GRID[0],
            (d.prefetch_limit, d.rerank_depth),
            "and it must run first, so a sweep cut short by a closing GPU window \
             still carries its own baseline"
        );
    }

    /// `rerank_depth` may never exceed `prefetch_limit`: the reranked pool
    /// cannot be wider than the prefetch that fills it, and a grid cell
    /// that asks for it measures the prefetch instead of the depth.
    #[test]
    fn no_grid_cell_asks_for_a_pool_wider_than_its_prefetch() {
        for (prefetch, depth) in WIDTH_GRID {
            assert!(
                depth as u64 <= prefetch,
                "({prefetch}, {depth}) reranks deeper than it prefetches"
            );
        }
    }
}
