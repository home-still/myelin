//! Grade a finished run's answers with the local reader, to validate a
//! deterministic scorer against something other than itself.
//!
//! # This is not a metric, and the distinction is the whole point
//!
//! [`crate::bench`] refuses an LLM judge as a *metric* because the number
//! would then depend on a model we do not have pinned. Using one to measure
//! whether a deterministic scorer agrees with a human-legible notion of
//! correctness is a different job: the judge never enters a reported accuracy
//! number, its verdicts are cached to disk, and
//! `docs/measurements/m14-temporal-scorer.md` dumps the disagreements verbatim
//! so a reader can audit the judge instead of trusting it.
//!
//! # Self-grading, and why it is the conservative direction here
//!
//! The judge is the same `qwen3.5-9b` that produced the answers.
//! `docs/measurements/m9-judge-panel.md` measured this model as *harsher* than
//! `gemini-3.1-flash-lite` — 25.6% vs 27.9% marked correct, Fleiss κ 0.8813 —
//! so a scorer that agrees with it is not being flattered by a lenient grader.
//!
//! # Resume, and the model-mismatch guard
//!
//! Verdicts are cached in `<run>/judge_verdicts.json` and a re-run judges only
//! the ids the cache lacks. A cache written by a *different* model is a hard
//! error naming both: silently mixing two judges' verdicts into one κ is the
//! failure mode `adapters/judge_panel.py` already guards against.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use myelin_core::config::MyelinConfig;
use myelin_core::llm::openai::OpenAiLlm;
use myelin_core::llm::{CompletionRequest, Llm, Message};
use serde::{Deserialize, Serialize};

use crate::bench::{is_abstention, ScoredQuestion};

/// `<run>/judge_verdicts.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeFile {
    pub model: String,
    /// `question_id` -> 1 correct, 0 incorrect.
    pub verdicts: BTreeMap<String, u8>,
}

/// The grading rubric, verbatim and pinned.
///
/// The date clauses are explicit because they are the thing in dispute: a
/// grader left to its own devices credits `2023-06-27` against `The week
/// before 27 June 2023`, which is exactly the partial credit token F1 was
/// giving and the reason this milestone exists.
const JUDGE_SYSTEM: &str = "You are a strict grader. You are given a question, a reference answer, and a model answer.
The model answer is correct when it conveys the same fact as the reference answer.
A date is correct when it names the same day, or a range of days that the reference date falls inside.
A date that is off by one or more days is incorrect.
A date coarser than the reference is incorrect: naming a month when the reference names a day is incorrect.
A length of time is correct when it names the same length.
Ignore wording, ordering, and extra words that make no additional claim.
The answer is incorrect if it makes any claim that contradicts the reference answer.
Reply with exactly one character, 1 for correct or 0 for incorrect, and nothing else.";

/// How a `judge` invocation ended, for the one-line summary.
#[derive(Debug, Clone, Copy)]
pub struct JudgeStats {
    pub judged: usize,
    pub cached: usize,
    pub correct: usize,
}

/// Grade a finished run's answerable, answered rows.
///
/// Adversarial items are skipped: both scorers grade them by
/// [`is_abstention`], so nothing about them is in dispute. Declined answers
/// are skipped for the same reason — they are 0.0 under both scorers by the
/// shared rule.
pub async fn judge_run(
    run: &Path,
    category: Option<u8>,
    limit: Option<usize>,
) -> Result<(JudgeFile, JudgeStats)> {
    let rows_path = run.join("per_question.jsonl");
    let text = std::fs::read_to_string(&rows_path)
        .with_context(|| format!("read {}", rows_path.display()))?;
    let mut docket: Vec<ScoredQuestion> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: ScoredQuestion = serde_json::from_str(line).with_context(|| {
            format!(
                "{} is not a `bench` run directory (its per_question.jsonl has no \
                 `exact_match`/`tenant`); the vendored LME-V2 harness writes a \
                 different row shape",
                run.display()
            )
        })?;
        if row.is_abstention_problem || is_abstention(&row.response_raw) {
            continue;
        }
        if category.is_some_and(|c| c != row.category) {
            continue;
        }
        docket.push(row);
        if limit.is_some_and(|n| docket.len() >= n) {
            break;
        }
    }

    let cfg = MyelinConfig::load().context("load myelin config")?;
    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("judge client")?;
    let model = llm.id().to_string();

    let cache_path = run.join("judge_verdicts.json");
    let mut verdicts: BTreeMap<String, u8> = BTreeMap::new();
    if cache_path.exists() {
        let existing: JudgeFile = serde_json::from_str(
            &std::fs::read_to_string(&cache_path)
                .with_context(|| format!("read {}", cache_path.display()))?,
        )
        .with_context(|| format!("parse {}", cache_path.display()))?;
        anyhow::ensure!(
            existing.model == model,
            "{} was written by judge {:?} but the configured judge is {:?}; \
             mixing two judges' verdicts into one agreement number is not a \
             measurement — delete the file or point the config back at {:?}",
            cache_path.display(),
            existing.model,
            model,
            existing.model
        );
        verdicts = existing.verdicts;
    }

    let cached = docket
        .iter()
        .filter(|r| verdicts.contains_key(&r.question_id))
        .count();
    let mut judged = 0usize;
    for row in &docket {
        if verdicts.contains_key(&row.question_id) {
            continue;
        }
        let user = format!(
            "<question>\n{}\n</question>\n<reference>\n{}\n</reference>\n<model_answer>\n{}\n</model_answer>",
            row.question_text, row.answer_gold, row.response_raw
        );
        let reply = llm
            .complete(
                &CompletionRequest::new(vec![
                    Message::system(JUDGE_SYSTEM),
                    Message::user(user),
                ])
                .with_max_tokens(4),
            )
            .await
            .with_context(|| format!("judge {}", row.question_id))?
            .text;
        // R7 discipline: an unparseable judgement is an error, never coerced
        // into a verdict. A silent 0 would look exactly like a wrong answer.
        let verdict = reply
            .chars()
            .find_map(|c| match c {
                '0' => Some(0u8),
                '1' => Some(1u8),
                _ => None,
            })
            .with_context(|| {
                format!(
                    "judge {} replied {reply:?}, which contains no 0 or 1",
                    row.question_id
                )
            })?;
        verdicts.insert(row.question_id.clone(), verdict);
        judged += 1;
    }

    let file = JudgeFile { model, verdicts };
    std::fs::write(&cache_path, serde_json::to_string_pretty(&file)?)
        .with_context(|| format!("write {}", cache_path.display()))?;

    let correct = docket
        .iter()
        .filter(|r| file.verdicts.get(&r.question_id) == Some(&1))
        .count();
    Ok((
        file,
        JudgeStats {
            judged,
            cached,
            correct,
        },
    ))
}
