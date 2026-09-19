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
