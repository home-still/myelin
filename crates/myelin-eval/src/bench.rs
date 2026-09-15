//! LoCoMo end-to-end answer accuracy.
//!
//! [`crate::ablate`] measures *retrieval* — did the gold evidence come back.
//! This measures the thing a user experiences: the reader's answer, scored
//! against LoCoMo's gold answer. The two can disagree, and the gap between
//! them is the reader's contribution.
//!
//! # Scoring is deterministic, and that is the point
//!
//! `PLAN.md` §1 G2 wants conversational accuracy with confidence intervals.
//! An LLM judge would make every number depend on a judge model we do not
//! have pinned (see `docs/measurements/m6-g1-breakeven.md` — the
//! LongMemEval-V2 packager hard-requires `gpt-5.2` and we score with a local
//! Qwen3.5-9B, which is exactly why those runs are not leaderboard-
//! comparable). Deterministic scoring has no such dependency: the same run
//! scores identically on any machine, forever.
//!
//! The metric is SQuAD-style normalised token F1, the same family LoCoMo's
//! own evaluation uses. Normalisation is spelled out in [`normalize`] rather
//! than described, because every reimplementation of "SQuAD normalisation"
//! differs slightly and the difference moves the number by a point or two.
//!
//! **These are our numbers under our documented scorer, not official LoCoMo
//! leaderboard numbers.** No LoCoMo harness is vendored here, so nothing
//! claims protocol identity with the published table.
//!
//! # Category 5 is scored as abstention, not F1
//!
//! LoCoMo category 5 is adversarial: the question cannot be answered from the
//! conversation. Token F1 against an absent gold answer is meaningless, so
//! those items are scored as a binary — did the reader decline. This mirrors
//! LongMemEval-V2's abstention split and lets the two corpora be read the
//! same way, which matters because
//! `docs/measurements/m5-reference-baselines.md` found abstention to be this
//! reader's dominant failure mode.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use myelin_core::config::MyelinConfig;
use myelin_core::embed::remote::RemoteEmbedder;
use myelin_core::llm::{CompletionRequest, Llm, Message};
use myelin_core::llm::openai::OpenAiLlm;
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::pipeline::investigate::Investigator;
use myelin_core::pipeline::retrieve::Retriever;
use myelin_core::rerank::cross::CrossEncoder;
use myelin_core::rerank::Reranker;
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::QdrantStore;
use serde::Serialize;

use crate::datasets::{locomo, longmemeval};

/// Instruction given to the reader for every question.
///
/// The abstention clause is explicit because
/// `docs/measurements/m5-reference-baselines.md` measured this reader
/// fabricating answers on 97.2% of unanswerable questions when given *no
/// evidence at all*. Leaving abstention implicit measures the prompt's
/// omission rather than the memory system.
const READER_SYSTEM: &str = "You answer questions using only the supplied memories. \
Answer in as few words as possible — a name, a date, a short phrase. \
Do not explain. Do not restate the question. \
If the memories do not contain the answer, reply exactly: I don't know.";

/// One scored question, written to `per_question.jsonl`.
///
/// Field names match what `adapters/paired_ci.py` reads (`question_id`,
/// `score`, `is_abstention_problem`) so LoCoMo runs and LongMemEval-V2 runs
/// go through one confidence-interval tool instead of two.
#[derive(Debug, Clone, Serialize)]
pub struct ScoredQuestion {
    pub question_id: String,
    pub tenant: String,
    pub category: u8,
    pub question_text: String,
    pub answer_gold: String,
    pub response_raw: String,
    pub score: f64,
    pub exact_match: f64,
    pub is_abstention_problem: bool,
    pub retrieved_items: usize,
    pub memory_query_duration_seconds: f64,
}

/// Aggregate over one bench run.
#[derive(Debug, Clone, Serialize)]
pub struct BenchRun {
    pub corpus: String,
    pub collection: String,
    pub mode: String,
    pub k: usize,
    pub max_steps: usize,
    pub questions: usize,
    /// Mean token F1 over non-adversarial items (categories 1–4).
    pub f1_answerable: f64,
    /// Mean exact match over non-adversarial items.
    pub em_answerable: f64,
    /// Fraction of category-5 items the reader correctly declined.
    pub abstention_accuracy: f64,
    pub by_category: Vec<CategoryScore>,
    pub query_p50_seconds: f64,
    pub query_avg_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CategoryScore {
    pub category: u8,
    pub count: usize,
    pub mean_score: f64,
}

/// SQuAD-style normalisation, written out rather than referenced.
///
/// Lowercase, drop articles, strip punctuation, collapse whitespace. The
/// article list is exactly `a`/`an`/`the`; punctuation is anything
/// `char::is_ascii_punctuation` accepts. Changing any of this changes every
/// number in the run, so it is pinned here and tested.
pub fn normalize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the"))
        .map(str::to_string)
        .collect()
}

/// Token-level F1 between a prediction and a gold answer.
///
/// Multiset intersection, not set: a prediction that repeats a gold token
/// twice should not earn credit twice, and `HashSet` would silently allow it.
pub fn token_f1(prediction: &str, gold: &str) -> f64 {
    let pred = normalize(prediction);
    let gold = normalize(gold);
    if pred.is_empty() || gold.is_empty() {
        // Both empty is a match; one empty is not. Mirrors SQuAD.
        return f64::from(u8::from(pred.is_empty() == gold.is_empty()));
    }
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for t in &gold {
        *counts.entry(t.as_str()).or_insert(0) += 1;
    }
    let mut overlap = 0i64;
    for t in &pred {
        let e = counts.entry(t.as_str()).or_insert(0);
        if *e > 0 {
            *e -= 1;
            overlap += 1;
        }
    }
    if overlap == 0 {
        return 0.0;
    }
    let precision = overlap as f64 / pred.len() as f64;
    let recall = overlap as f64 / gold.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

/// Did the reader decline to answer?
///
/// Deliberately narrow. A loose match (any sentence containing "not") would
/// count "the memories do not say when, but it was Tuesday" as an
/// abstention, which is a confident wrong answer wearing a hedge. The reader
/// is instructed to emit an exact string; this accepts that string and a
/// small set of near-misses observed in practice.
pub fn is_abstention(response: &str) -> bool {
    let n = normalize(response).join(" ");
    n.is_empty()
        || n == "i dont know"
        || n == "i don t know"
        || n == "unknown"
        || n == "no information"
        || n.starts_with("i dont know")
        || n.starts_with("i don t know")
        || n.starts_with("i cannot determine")
        || n.starts_with("i can t determine")
        || n.starts_with("there is no information")
        || n.starts_with("no information")
}

/// Flatten LoCoMo's `answer` field to a string.
///
/// It is `Option<Value>` because category-5 items may omit it and some items
/// carry a number rather than a string; `Value::to_string` would wrap strings
/// in quotes and poison the token overlap.
fn gold_answer(value: Option<&serde_json::Value>) -> String {
    match value {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx]
}

/// Run LoCoMo end-to-end and score every question.
#[allow(clippy::too_many_arguments)]
pub async fn bench_locomo(
    path: &Path,
    collection: &str,
    ledger_path: &Path,
    k: usize,
    mode: Mode,
    max_steps: usize,
    limit: Option<usize>,
    out_dir: &Path,
) -> Result<BenchRun> {
    let cfg = MyelinConfig::load().context("load myelin config")?;
    let conversations = locomo::load(path).context("load locomo")?;

    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("reader client")?;
    let embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)
        .context("embedder client")?;
    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;
    let reranker = CrossEncoder::new(&cfg.rerank.url, &cfg.rerank.model).ok();

    let mut retriever = Retriever::new(&embedder, &store, &ledger);
    if let Some(r) = reranker.as_ref() {
        retriever = retriever.with_reranker(r as &dyn Reranker);
    }

    let mut scored: Vec<ScoredQuestion> = Vec::new();
    let mut latencies: Vec<f64> = Vec::new();

    'outer: for conv in &conversations {
        let tenant = format!("locomo/{}", conv.sample_id);
        for (i, qa) in conv.qa.iter().enumerate() {
            if let Some(n) = limit {
                if scored.len() >= n {
                    break 'outer;
                }
            }
            let gold = gold_answer(qa.answer.as_ref());
            let adversarial = qa.category == 5;

            let query = Recall {
                scope: ScopeFilter::tenant(&tenant).with_namespace("locomo"),
                text: qa.question.clone(),
                budget: Budget {
                    k,
                    tokens: 4096,
                    max_steps,
                },
                mode,
                kinds: None,
            };

            let started = std::time::Instant::now();
            let evidence = match mode {
                Mode::Investigate => {
                    Investigator::new(&llm, &retriever)
                        .investigate(&query)
                        .await
                        .with_context(|| format!("investigate {tenant}#{i}"))?
                        .0
                }
                Mode::Recall => {
                    retriever
                        .recall(&query)
                        .await
                        .with_context(|| format!("recall {tenant}#{i}"))?
                        .0
                }
            };
            let elapsed = started.elapsed().as_secs_f64();
            latencies.push(elapsed);

            let context = evidence
                .items
                .iter()
                .enumerate()
                .map(|(n, it)| format!("[{n}] {}", it.value))
                .collect::<Vec<_>>()
                .join("\n");
            let response = llm
                .complete(
                    &CompletionRequest::new(vec![
                        Message::system(READER_SYSTEM),
                        Message::user(format!(
                            "<memories>\n{context}\n</memories>\n<question>\n{}\n</question>",
                            qa.question
                        )),
                    ])
                    .with_max_tokens(160),
                )
                .await
                .with_context(|| format!("reader {tenant}#{i}"))?
                .text;

            let declined = is_abstention(&response);
            let (score, exact) = if adversarial {
                let s = f64::from(u8::from(declined));
                (s, s)
            } else if declined {
                // An abstention on an answerable question earns nothing, and
                // must not accidentally score via token overlap with a gold
                // answer that happens to contain "know".
                (0.0, 0.0)
            } else {
                let f1 = token_f1(&response, &gold);
                let em = f64::from(u8::from(normalize(&response) == normalize(&gold)));
                (f1, em)
            };

            scored.push(ScoredQuestion {
                question_id: format!("{}#{i}", conv.sample_id),
                tenant: tenant.clone(),
                category: qa.category,
                question_text: qa.question.clone(),
                answer_gold: gold,
                response_raw: response,
                score,
                exact_match: exact,
                is_abstention_problem: adversarial,
                retrieved_items: evidence.items.len(),
                memory_query_duration_seconds: elapsed,
            });
        }
    }

    finish_run("locomo", collection, mode, k, max_steps, scored, latencies, out_dir)
}

/// Run LongMemEval_S end-to-end against a memory that `build` already wrote.
///
/// Scored with the same deterministic token-F1 as LoCoMo. LongMemEval's own
/// protocol uses a GPT-4o judge with type-specific prompts, which we do not
/// have; `docs/measurements/m9-judge-panel.md` measures our local judge at
/// kappa 0.8813 against a frontier model and slightly *harsher*, so a
/// judge-free metric is the more conservative choice here and it is
/// reproducible forever. **These are not protocol-identical LongMemEval_S
/// numbers** and are not comparable to the published table.
///
/// Every question is scoped to its own tenant, matching the 500 independent
/// memories `build_longmemeval_s` writes. Reading across tenants would answer
/// from other questions' haystacks.
#[allow(clippy::too_many_arguments)]
pub async fn bench_longmemeval_s(
    dataset: &Path,
    collection: &str,
    ledger_path: &Path,
    k: usize,
    mode: Mode,
    max_steps: usize,
    limit: Option<usize>,
    out_dir: &Path,
) -> Result<BenchRun> {
    let cfg = MyelinConfig::load().context("load myelin config")?;
    let mut items = longmemeval::load(dataset).context("load longmemeval_s")?;
    if let Some(n) = limit {
        items.truncate(n);
    }

    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("reader client")?;
    let embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)
        .context("embedder client")?;
    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;
    let reranker = CrossEncoder::new(&cfg.rerank.url, &cfg.rerank.model).ok();

    let mut retriever = Retriever::new(&embedder, &store, &ledger);
    if let Some(r) = reranker.as_ref() {
        retriever = retriever.with_reranker(r as &dyn Reranker);
    }

    let mut scored: Vec<ScoredQuestion> = Vec::new();
    let mut latencies: Vec<f64> = Vec::new();

    for item in &items {
        let adversarial = item.is_abstention();
        let gold = item.answer_text();
        let query = Recall {
            scope: ScopeFilter::tenant(format!("lme_s/{}", item.question_id))
                .with_namespace("longmemeval_s"),
            text: item.question.clone(),
            budget: Budget {
                k,
                tokens: 4096,
                max_steps,
            },
            mode,
            kinds: None,
        };

        let started = std::time::Instant::now();
        let evidence = match mode {
            Mode::Investigate => {
                Investigator::new(&llm, &retriever)
                    .investigate(&query)
                    .await
                    .with_context(|| format!("investigate {}", item.question_id))?
                    .0
            }
            Mode::Recall => {
                retriever
                    .recall(&query)
                    .await
                    .with_context(|| format!("recall {}", item.question_id))?
                    .0
            }
        };
        let elapsed = started.elapsed().as_secs_f64();
        latencies.push(elapsed);

        let context = evidence
            .items
            .iter()
            .enumerate()
            .map(|(n, it)| format!("[{n}] {}", it.value))
            .collect::<Vec<_>>()
            .join("\n");
        let response = llm
            .complete(
                &CompletionRequest::new(vec![
                    Message::system(READER_SYSTEM),
                    Message::user(format!(
                        "<memories>\n{context}\n</memories>\n<today>\n{}\n</today>\n<question>\n{}\n</question>",
                        item.question_date, item.question
                    )),
                ])
                .with_max_tokens(160),
            )
            .await
            .with_context(|| format!("reader {}", item.question_id))?
            .text;

        let declined = is_abstention(&response);
        let (score, exact) = if adversarial {
            let s = f64::from(u8::from(declined));
            (s, s)
        } else if declined {
            (0.0, 0.0)
        } else {
            (
                token_f1(&response, &gold),
                f64::from(u8::from(normalize(&response) == normalize(&gold))),
            )
        };

        scored.push(ScoredQuestion {
            question_id: item.question_id.clone(),
            tenant: format!("lme_s/{}", item.question_id),
            category: question_type_code(&item.question_type),
            question_text: item.question.clone(),
            answer_gold: gold,
            response_raw: response,
            score,
            exact_match: exact,
            is_abstention_problem: adversarial,
            retrieved_items: evidence.items.len(),
            memory_query_duration_seconds: elapsed,
        });
    }

    finish_run("longmemeval_s", collection, mode, k, max_steps, scored, latencies, out_dir)
}

/// LongMemEval names its question types; `ScoredQuestion::category` is numeric
/// so both corpora share one row shape and one CI tool.
fn question_type_code(t: &str) -> u8 {
    match t {
        "single-session-user" => 1,
        "single-session-assistant" => 2,
        "single-session-preference" => 3,
        "multi-session" => 4,
        "temporal-reasoning" => 5,
        "knowledge-update" => 6,
        _ => 0,
    }
}


/// Aggregate, print nothing, write `per_question.jsonl` and
/// `aggregated_metrics.json`. Shared by both corpora so a metric fixed for one
/// is fixed for both, and so `adapters/paired_ci.py` reads one row shape.
#[allow(clippy::too_many_arguments)]
fn finish_run(
    corpus: &str,
    collection: &str,
    mode: Mode,
    k: usize,
    max_steps: usize,
    scored: Vec<ScoredQuestion>,
    latencies: Vec<f64>,
    out_dir: &Path,
) -> Result<BenchRun> {
    let answerable: Vec<&ScoredQuestion> =
        scored.iter().filter(|s| !s.is_abstention_problem).collect();
    let adversarial: Vec<&ScoredQuestion> =
        scored.iter().filter(|s| s.is_abstention_problem).collect();

    let mean = |xs: &[&ScoredQuestion], f: fn(&ScoredQuestion) -> f64| {
        if xs.is_empty() {
            0.0
        } else {
            xs.iter().map(|s| f(s)).sum::<f64>() / xs.len() as f64
        }
    };

    let mut per_cat: HashMap<u8, (usize, f64)> = HashMap::new();
    for s in &scored {
        let e = per_cat.entry(s.category).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += s.score;
    }
    let mut by_category: Vec<CategoryScore> = per_cat
        .into_iter()
        .map(|(category, (count, total))| CategoryScore {
            category,
            count,
            mean_score: total / count as f64,
        })
        .collect();
    by_category.sort_by_key(|c| c.category);

    let mut sorted = latencies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };

    let run = BenchRun {
        corpus: corpus.to_string(),
        collection: collection.to_string(),
        mode: match mode {
            Mode::Investigate => "investigate".into(),
            Mode::Recall => "recall".into(),
        },
        k,
        max_steps,
        questions: scored.len(),
        f1_answerable: mean(&answerable, |s| s.score),
        em_answerable: mean(&answerable, |s| s.exact_match),
        abstention_accuracy: mean(&adversarial, |s| s.score),
        by_category,
        query_p50_seconds: percentile(&sorted, 0.50),
        query_avg_seconds: avg,
    };

    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create {}", out_dir.display()))?;
    let mut lines = String::new();
    for s in &scored {
        lines.push_str(&serde_json::to_string(s)?);
        lines.push('\n');
    }
    std::fs::write(out_dir.join("per_question.jsonl"), lines)
        .context("write per_question.jsonl")?;
    std::fs::write(
        out_dir.join("aggregated_metrics.json"),
        serde_json::to_string_pretty(&run)?,
    )
    .context("write aggregated_metrics.json")?;

    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_drops_articles_and_punctuation() {
        assert_eq!(normalize("The Quick, brown fox!"), vec!["quick", "brown", "fox"]);
        // "a" as an article disappears; "a" inside a word does not.
        assert_eq!(normalize("a cat"), vec!["cat"]);
        assert_eq!(normalize("apple"), vec!["apple"]);
    }

    #[test]
    fn token_f1_is_multiset_not_set() {
        // Repeating a gold token must not earn credit twice. With set
        // semantics this scores 1.0; with multiset semantics precision is
        // 1/2 and recall 1/1, giving F1 = 2/3.
        let f1 = token_f1("paris paris", "paris");
        assert!((f1 - 2.0 / 3.0).abs() < 1e-9, "got {f1}");
    }

    #[test]
    fn token_f1_partial_overlap() {
        // "in" is not an article, so gold has 3 tokens: born, in, paris.
        let f1 = token_f1("born in Paris", "born in London");
        assert!((f1 - 2.0 / 3.0).abs() < 1e-9, "got {f1}");
        assert_eq!(token_f1("Paris", "paris"), 1.0);
        assert_eq!(token_f1("London", "Paris"), 0.0);
    }

    #[test]
    fn abstention_detector_rejects_hedged_answers() {
        assert!(is_abstention("I don't know"));
        assert!(is_abstention("I don't know."));
        assert!(is_abstention("unknown"));
        // A confident wrong answer wearing a hedge is NOT an abstention.
        assert!(!is_abstention(
            "The memories do not say when, but it was Tuesday"
        ));
        assert!(!is_abstention("Paris"));
        assert!(!is_abstention("He does not know her name"));
    }

    #[test]
    fn gold_answer_unwraps_strings_without_quoting() {
        use serde_json::json;
        assert_eq!(gold_answer(Some(&json!("Paris"))), "Paris");
        assert_eq!(gold_answer(Some(&json!(7))), "7");
        assert_eq!(gold_answer(None), "");
        assert_eq!(gold_answer(Some(&serde_json::Value::Null)), "");
    }
}