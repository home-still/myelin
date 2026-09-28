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
    /// The rubric each verdict was graded under (2026-09-28, M77's finding).
    ///
    /// A verdict is a function of the rubric as well as the answer. Absent on
    /// caches written before this field; every one of those verdicts was
    /// graded by the fact rubric, and reads as [`Rubric::Fact`].
    #[serde(default)]
    pub rubrics: BTreeMap<String, Rubric>,
}

/// Which grading rubric a row is judged under.
///
/// One question fits almost every row: does the answer convey the reference's
/// fact? A LongMemEval_S `single-session-preference` row has no fact. Its
/// reference is a rubric ("The user would prefer responses that …"), and M77
/// measured what grading it as a fact costs: on 90 seed-rows the fact rubric
/// and LongMemEval's own grader disagreed 15 to 6, the 15 being answers built
/// on the user's stated preferences (`docs/measurements/m77-advice-profile-clause.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rubric {
    Fact,
    Preference,
}

/// LongMemEval_S's tenant prefix, as `build.rs` writes it. Category codes are
/// per corpus (LoCoMo also has a category 3), so the corpus is checked first.
const LME_S_TENANT_PREFIX: &str = "lme_s/";

/// The judge's reply is one character; a little room for a stray token.
const JUDGE_REPLY_TOKENS: u32 = 4;

/// The rubric `row` is graded under.
fn rubric_for(row: &ScoredQuestion) -> Rubric {
    let preference = crate::bench::question_type_code("single-session-preference");
    if row.tenant.starts_with(LME_S_TENANT_PREFIX) && row.category == preference {
        Rubric::Preference
    } else {
        Rubric::Fact
    }
}

/// The one request that grades `row`.
///
/// The fact path is byte-for-byte the request every verdict before
/// 2026-09-28 was given, so no other row's grading moves.
fn judge_request(row: &ScoredQuestion) -> CompletionRequest {
    let (system, reference_tag) = match rubric_for(row) {
        Rubric::Fact => (JUDGE_SYSTEM, "reference"),
        Rubric::Preference => (JUDGE_PREFERENCE_SYSTEM, "rubric"),
    };
    let user = format!(
        "<question>\n{}\n</question>\n<{reference_tag}>\n{}\n</{reference_tag}>\n<model_answer>\n{}\n</model_answer>",
        row.question_text, row.answer_gold, row.response_raw
    );
    CompletionRequest::new(vec![Message::system(system), Message::user(user)])
        .with_max_tokens(JUDGE_REPLY_TOKENS)
}

/// Keep a cached or seeded verdict only for a row this run still sends to the
/// judge, only if it was given for the answer that row holds now, and only if
/// it was graded under the rubric that row is graded under now.
fn keep_reusable(
    verdicts: &mut BTreeMap<String, u8>,
    answers: &mut BTreeMap<String, String>,
    rubrics: &mut BTreeMap<String, Rubric>,
    current: &BTreeMap<&str, (&str, Rubric)>,
) {
    verdicts.retain(|id, _| {
        let now = current.get(id.as_str());
        let graded_under = rubrics.get(id).copied().unwrap_or(Rubric::Fact);
        reusable(answers.get(id).map(String::as_str), now.map(|(answer, _)| *answer))
            && now.is_some_and(|(_, rubric)| *rubric == graded_under)
    });
    answers.retain(|id, _| verdicts.contains_key(id));
    rubrics.retain(|id, _| verdicts.contains_key(id));
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

/// The rubric for a LongMemEval_S preference row (2026-09-28).
///
/// Its criterion is LongMemEval's own preference template, the one its
/// grader gives this type: "The model does not need to reflect all the points
/// in the rubric. The response is correct as long as it recalls and utilizes
/// the user's personal information correctly." (Wu et al. 2024, arXiv
/// 2410.10813; `xiaowu0162/LongMemEval@9e0b455:src/evaluation/evaluate_qa.py`,
/// `get_anscheck_prompt`, reproduced byte for byte in
/// `adapters/judge_matched.py`). It keeps the strict grader's voice and reply
/// format, and says outright what the template implies: a generic answer
/// recalls nothing.
const JUDGE_PREFERENCE_SYSTEM: &str = "You are a strict grader. You are given a question, a rubric for the desired personalized response, and a model answer.
The model answer is correct when it recalls and utilizes the user's personal information correctly, as the rubric describes.
The model answer does not need to reflect every point in the rubric.
A generic answer that uses none of the user's personal information is incorrect.
The answer is incorrect if it makes any claim that contradicts the user's personal information in the rubric.
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
    let mut rubrics: BTreeMap<String, Rubric> = BTreeMap::new();
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
            let rubric = existing.rubrics.get(&id).copied().unwrap_or(Rubric::Fact);
            match existing.answers.get(&id) {
                Some(answer) => {
                    verdicts.insert(id.clone(), verdict);
                    answers.insert(id.clone(), answer.clone());
                    rubrics.insert(id, rubric);
                }
                None if !is_seed => {
                    verdicts.insert(id.clone(), verdict);
                    rubrics.insert(id, rubric);
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
    //
    // And a verdict graded under another rubric is no verdict (2026-09-28): a
    // preference row judged as a fact before then is judged again.
    let current: BTreeMap<&str, (&str, Rubric)> = docket
        .iter()
        .map(|r| (r.question_id.as_str(), (r.response_raw.as_str(), rubric_for(r))))
        .collect();
    keep_reusable(&mut verdicts, &mut answers, &mut rubrics, &current);

    let cached = docket
        .iter()
        .filter(|r| verdicts.contains_key(&r.question_id))
        .count();
    let mut judged = 0usize;
    for row in &docket {
        if verdicts.contains_key(&row.question_id) {
            answers.insert(row.question_id.clone(), row.response_raw.clone());
            rubrics.insert(row.question_id.clone(), rubric_for(row));
            continue;
        }
        let reply = llm
            .complete(&judge_request(row))
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
        rubrics.insert(row.question_id.clone(), rubric_for(row));
        judged += 1;
    }

    let file = JudgeFile {
        model,
        verdicts,
        answers,
        rubrics,
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
            rubrics: BTreeMap::new(),
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

    fn scored(tenant: &str, category: u8, answer: &str) -> ScoredQuestion {
        serde_json::from_value(serde_json::json!({
            "question_id": "q", "tenant": tenant, "category": category,
            "question_text": "Any tips for the music store?",
            "answer_gold": "The user would prefer responses that compare a Stratocaster and a Les Paul.",
            "response_raw": answer,
            "score": 0.0, "exact_match": 0.0, "is_abstention_problem": false,
            "retrieved_items": 6, "memory_query_duration_seconds": 0.1
        }))
        .expect("row")
    }

    /// M77's finding: a LongMemEval_S preference row is graded against its
    /// rubric, and every other row, LoCoMo's category 3 included, keeps the
    /// fact request byte for byte.
    #[test]
    fn a_preference_row_is_graded_against_its_rubric_and_no_other_row_moves() {
        let pref = judge_request(&scored("lme_s/q", 3, "Compare the necks."));
        assert_eq!(pref.messages[0].content, JUDGE_PREFERENCE_SYSTEM);
        assert!(pref.messages[1].content.contains("<rubric>\nThe user would prefer"));

        for (tenant, category) in [("lme_s/q", 1), ("lme_s/q", 5), ("locomo/conv-26", 3)] {
            let row = scored(tenant, category, "Compare the necks.");
            let req = judge_request(&row);
            assert_eq!(rubric_for(&row), Rubric::Fact, "{tenant} {category}");
            assert_eq!(req.messages[0].content, JUDGE_SYSTEM);
            assert_eq!(
                req.messages[1].content,
                format!(
                    "<question>\n{}\n</question>\n<reference>\n{}\n</reference>\n<model_answer>\n{}\n</model_answer>",
                    row.question_text, row.answer_gold, row.response_raw
                ),
                "the fact request every earlier verdict was given"
            );
            assert_eq!(req.max_tokens, Some(JUDGE_REPLY_TOKENS));
        }
    }

    /// A verdict graded as a fact is not reused for a row now graded against
    /// its rubric, even for the same answer; a verdict graded under the row's
    /// own rubric is.
    #[test]
    fn a_verdict_is_reused_only_under_the_rubric_it_was_graded_by() {
        let mut verdicts: BTreeMap<String, u8> =
            [("pref_old", 0), ("pref_new", 1), ("fact", 1)].map(|(i, v)| (i.to_string(), v)).into();
        let mut answers: BTreeMap<String, String> =
            [("pref_old", "a"), ("pref_new", "b"), ("fact", "c")]
                .map(|(i, a)| (i.to_string(), a.to_string()))
                .into();
        // `pref_old` predates the field; `fact` predates it too.
        let mut rubrics: BTreeMap<String, Rubric> =
            [("pref_new".to_string(), Rubric::Preference)].into();
        let current: BTreeMap<&str, (&str, Rubric)> = [
            ("pref_old", ("a", Rubric::Preference)),
            ("pref_new", ("b", Rubric::Preference)),
            ("fact", ("c", Rubric::Fact)),
        ]
        .into();

        keep_reusable(&mut verdicts, &mut answers, &mut rubrics, &current);

        assert!(!verdicts.contains_key("pref_old"), "graded as a fact: judge again");
        assert_eq!(verdicts.get("pref_new"), Some(&1));
        assert_eq!(verdicts.get("fact"), Some(&1), "an old fact verdict still stands");
        assert!(!answers.contains_key("pref_old"));
    }

    /// Every verdict file on disk predates the rubric map and must still load,
    /// reading as graded by the fact rubric.
    #[test]
    fn a_verdict_file_without_rubrics_loads_as_graded_by_the_fact_rubric() {
        let back: JudgeFile = serde_json::from_str(
            r#"{"model":"qwen3.5-9b","verdicts":{"q1":1},"answers":{"q1":"x"}}"#,
        )
        .expect("load");
        assert!(back.rubrics.is_empty());
        let written = JudgeFile {
            rubrics: [("q1".to_string(), Rubric::Preference)].into(),
            ..back
        };
        let text = serde_json::to_string(&written).expect("serialise");
        assert!(text.contains(r#""rubrics":{"q1":"preference"}"#), "{text}");
    }

}
