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
    /// The answer each verdict was given for (M42).
    ///
    /// A verdict is a function of the **answer**, not of the row id. Keying
    /// the cache on `question_id` alone is sound within one run, where the
    /// answer cannot change, and unsound across runs — which is exactly where
    /// a paired arm lives.
    ///
    /// M42 measured the cost. Its arm left 469 of 500 responses byte-identical
    /// to the base, and re-judging them flipped **2**, moving the control off
    /// zero by −0.43 points. The mechanism under test was worth +1.80, so a
    /// quarter of the headline was the grader disagreeing with itself. Without
    /// this map there is no way to tell those apart.
    ///
    /// Absent on caches written before M42; those are trusted in place,
    /// because a verdict file inside a run directory was written for that
    /// run's answers.
    #[serde(default)]
    pub answers: BTreeMap<String, String>,
}

/// Whether a cached or seeded verdict may stand for a row, given the answer it
/// was recorded for (`cached`) and the answer the row holds now if the row is
/// still judged (`now`; `None` for a decline or an abstention item).
fn reusable(cached: Option<&str>, now: Option<&str>) -> bool {
    match (cached, now) {
        (Some(cached), Some(now)) => cached == now,
        // No recorded answer: a pre-M42 own cache, trusted in place for a row
        // that is still judged.
        (None, Some(_)) => true,
        (_, None) => false,
    }
}

/// The verdict `judge` gave for the answer `row` holds now, if any.
///
/// A verdict recorded for a different answer is no verdict: it graded
/// something this run no longer says. A file written before M42 has no
/// `answers` map and is trusted in place, as `judge_run` trusts it.
pub fn verdict_for(judge: &JudgeFile, row: &ScoredQuestion) -> Option<u8> {
    let verdict = *judge.verdicts.get(&row.question_id)?;
    match judge.answers.get(&row.question_id) {
        Some(answer) if *answer != row.response_raw => None,
        _ => Some(verdict),
    }
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
    seed: Option<&Path>,
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
    let mut answers: BTreeMap<String, String> = BTreeMap::new();
    for path in seed.iter().map(|s| s.join("judge_verdicts.json")).chain([
        cache_path.clone(),
    ]) {
        if !path.exists() {
            continue;
        }
        let existing: JudgeFile = serde_json::from_str(
            &std::fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse {}", path.display()))?;
        anyhow::ensure!(
            existing.model == model,
            "{} was written by judge {:?} but the configured judge is {:?}; \
             mixing two judges' verdicts into one agreement number is not a \
             measurement — delete the file or point the config back at {:?}",
            path.display(),
            existing.model,
            model,
            existing.model
        );
        // A seeded verdict is only reusable if it was given for the answer
        // this run actually holds. Own cache without an `answers` map predates
        // M42 and is trusted in place; a *seed* without one is refused, because
        // there is nothing to check it against.
        let is_seed = path != cache_path;
        anyhow::ensure!(
            !is_seed || !existing.answers.is_empty(),
            "{} has no `answers` map, so its verdicts cannot be matched to \
             this run's responses; re-judge that run before seeding from it",
            path.display()
        );
        for (id, verdict) in existing.verdicts {
            match existing.answers.get(&id) {
                Some(answer) => {
                    verdicts.insert(id.clone(), verdict);
                    answers.insert(id, answer.clone());
                }
                None if !is_seed => {
                    verdicts.insert(id, verdict);
                }
                None => {}
            }
        }
    }
    // Keep a verdict only for a row this run still sends to the judge, and
    // only if it was given for the answer that row holds now.
    //
    // A row outside the docket is a decline or an abstention item, and never
    // has a verdict. Keeping one there is not harmless: every scorer looked
    // the verdict up before the decline rule. So a seed's "correct" for an
    // answer this run replaced with "I don't know." was counted as correct.
    // Found 2026-09-25 (M70): 16 seeded runs since M43 carried such verdicts,
    // M57 among them with 21 (its 83.40 was 79.20).
    let current: BTreeMap<&str, &str> = docket
        .iter()
        .map(|r| (r.question_id.as_str(), r.response_raw.as_str()))
        .collect();
    verdicts.retain(|id, _| {
        reusable(answers.get(id).map(String::as_str), current.get(id.as_str()).copied())
    });
    answers.retain(|id, _| verdicts.contains_key(id));

    let cached = docket
        .iter()
        .filter(|r| verdicts.contains_key(&r.question_id))
        .count();
    let mut judged = 0usize;
    for row in &docket {
        if verdicts.contains_key(&row.question_id) {
            answers.insert(row.question_id.clone(), row.response_raw.clone());
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
        answers.insert(row.question_id.clone(), row.response_raw.clone());
        judged += 1;
    }

    let file = JudgeFile {
        model,
        verdicts,
        answers,
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn file(pairs: &[(&str, u8, &str)]) -> JudgeFile {
        JudgeFile {
            model: "qwen3.5-9b".into(),
            verdicts: pairs.iter().map(|(i, v, _)| (i.to_string(), *v)).collect(),
            answers: pairs
                .iter()
                .map(|(i, _, a)| (i.to_string(), a.to_string()))
                .collect(),
        }
    }

    /// The map a seeded verdict is matched on must survive a round trip, or
    /// every seeded run silently falls back to re-judging.
    #[test]
    fn answers_round_trip_through_the_verdict_file() {
        let f = file(&[("q1", 1, "Instant Pot"), ("q2", 0, "I don't know.")]);
        let text = serde_json::to_string(&f).expect("serialise");
        let back: JudgeFile = serde_json::from_str(&text).expect("parse");
        assert_eq!(back.answers.get("q1").map(String::as_str), Some("Instant Pot"));
        assert_eq!(back.verdicts.get("q2"), Some(&0));
    }

    /// A cache written before M42 has no `answers` map and must still load —
    /// every judged run on disk predates this field.
    #[test]
    fn a_pre_m42_verdict_file_still_loads() {
        let legacy = r#"{"model":"qwen3.5-9b","verdicts":{"q1":1}}"#;
        let back: JudgeFile = serde_json::from_str(legacy).expect("legacy cache must load");
        assert_eq!(back.verdicts.get("q1"), Some(&1));
        assert!(
            back.answers.is_empty(),
            "an absent map must read as empty, not fail"
        );
    }

    /// The retention rule, which is the whole mechanism: a verdict survives
    /// only while the answer it was given for is still the answer.
    #[test]
    fn a_verdict_is_dropped_when_its_answer_changed() {
        let seeded = file(&[
            ("unchanged", 1, "Instant Pot"),
            ("changed", 0, "I don't know."),
        ]);
        // What the arm now holds: one row was rewritten by the second pass.
        let current: BTreeMap<&str, &str> = [
            ("unchanged", "Instant Pot"),
            ("changed", "fixing the fence"),
        ]
        .into_iter()
        .collect();

        let mut verdicts = seeded.verdicts.clone();
        verdicts.retain(
            |id, _| match (seeded.answers.get(id), current.get(id.as_str())) {
                (Some(cached), Some(now)) => cached == now,
                _ => true,
            },
        );

        assert_eq!(verdicts.get("unchanged"), Some(&1), "reused, never re-graded");
        assert!(
            !verdicts.contains_key("changed"),
            "a rewritten answer must be judged afresh, or the arm grades the \
             base's response"
        );
    }

    /// M70's finding: a seed's verdict must not survive on a row this run
    /// declined. It is not in the docket, so it can never be re-checked.
    #[test]
    fn a_verdict_survives_only_for_the_answer_it_graded_on_a_row_still_judged() {
        assert!(reusable(Some("Instant Pot"), Some("Instant Pot")));
        assert!(!reusable(Some("Instant Pot"), Some("a slow cooker")));
        assert!(!reusable(Some("Instant Pot"), None), "the row now declines");
        assert!(reusable(None, Some("Instant Pot")), "pre-M42 own cache, row still judged");
        assert!(!reusable(None, None));
    }

    #[test]
    fn verdict_for_refuses_a_verdict_recorded_for_another_answer() {
        let judge = file(&[("q", 1, "Instant Pot")]);
        let row = |answer: &str| -> ScoredQuestion {
            serde_json::from_value(serde_json::json!({
                "question_id": "q", "tenant": "t", "category": 1,
                "question_text": "?", "answer_gold": "g", "response_raw": answer,
                "score": 0.0, "exact_match": 0.0, "is_abstention_problem": false,
                "retrieved_items": 6, "memory_query_duration_seconds": 0.1
            }))
            .expect("row")
        };
        assert_eq!(verdict_for(&judge, &row("Instant Pot")), Some(1));
        assert_eq!(verdict_for(&judge, &row("I don't know.")), None);
    }

}
