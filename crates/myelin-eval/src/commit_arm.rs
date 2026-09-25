//! M42's arm, computed over a finished run instead of re-running one.
//!
//! `commit_answer` is a pure function of the first response: a row that did
//! not decline comes back byte-identical, without a model call. So the whole
//! arm is determined by the base run's rows plus a second call on the ~20%
//! that declined.
//!
//! Running it this way is not a shortcut, it is the **better measurement**.
//! M40 established the control that makes an arm readable: rows where the
//! mechanism did not fire must move by exactly zero. Re-running the pipeline
//! end-to-end obtains that control empirically and only approximately — the
//! reader is resampled, the reranker breaks ties differently after a cold
//! start (M-series precedent), and any drift is then attributed to the
//! mechanism. Taking the first pass from the base run makes the control exact
//! **by construction**: non-firing rows are the same bytes, so every point of
//! difference is the second pass and nothing else.
//!
//! It also removes a hardware confound that is otherwise unavoidable here. The
//! base was produced with the reader wholly on the GPU; the card no longer has
//! room for that, so a freshly re-run base would differ in backend as well as
//! in mechanism. Only the second pass — which has no counterpart in the base —
//! runs on the current configuration.
//!
//! The end-to-end wiring of `bench --commit-answer` is verified separately by
//! a small `--limit` run. This module measures the mechanism; that run proves
//! the switch reaches it.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use myelin_core::config::MyelinConfig;
use myelin_core::llm::openai::OpenAiLlm;

use crate::bench::{
    commit_answer, commit_consensus, commit_grounded, is_abstention, Consensus, ScoredQuestion,
};
use crate::datasets::longmemeval;

/// What the arm did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CommitArmReport {
    pub rows: usize,
    /// Rows whose first response declined, so the second pass ran.
    pub fired: usize,
    /// Rows where the second pass produced an answer.
    pub committed: usize,
    /// Rows left byte-identical to the base — the control.
    pub untouched: usize,
}

impl CommitArmReport {
    /// Every row is either untouched or committed, never both and never
    /// neither. A row that is *neither* would mean a response changed without
    /// the mechanism claiming it, which would silently break the control.
    pub fn is_consistent(&self) -> bool {
        self.untouched + self.committed == self.rows && self.committed <= self.fired
    }
}

/// Which second pass the arm applies to a declining row.
#[derive(Debug, Clone, Copy)]
pub enum Pass {
    /// M42: one greedy call, `{answer, evidence_absent}`.
    Greedy,
    /// M45: seeded samples clustered by meaning; commit the majority.
    Consensus(Consensus),
    /// M61: cite the memories that state the answer about the named entity,
    /// then answer from those alone (`bench::commit_grounded`).
    Grounded,
}

/// The corpora whose first-pass prompt this replays. The prompt must be the
/// base's byte for byte, or the untouched rows are no longer an exact control:
/// LongMemEval_S showed the reader `<today>`, and LoCoMo (without
/// `--question-date`) did not.
enum Prompt {
    /// `<today>` per question, from the LongMemEval_S dataset.
    WithToday(HashMap<String, String>),
    NoToday,
}

/// Apply the second pass `pass` to every declining row of `base`, writing
/// `out`.
///
/// `None` is M42's arm byte for byte: one greedy call, no sampling keys.
/// `Some` draws `samples` seeded answers per declining row, clusters them
/// by meaning in one structured call, and commits the majority only at or
/// above `agree`; every sample and the agreement land on the row so the
/// threshold can be re-applied offline against a calibration set.
pub async fn run(
    cfg: &MyelinConfig,
    base: &Path,
    dataset: &Path,
    out: &Path,
    pass: Pass,
) -> Result<CommitArmReport> {
    anyhow::ensure!(
        base != out,
        "refusing to write into the source run {}; an arm must not overwrite the base it is paired against",
        base.display()
    );
    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("reader client")?;

    let base_metrics: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(base.join("aggregated_metrics.json"))
            .with_context(|| format!("read {}/aggregated_metrics.json", base.display()))?,
    )
    .context("parse the base's aggregated_metrics.json")?;
    // The replay shows the base's own system prompt, clauses and all, rebuilt
    // through the function the base was read with (M71: the shipped
    // LongMemEval_S run carries the premise clause).
    let system = crate::bench::reader_system_of_run(&base_metrics);
    // `question_date` is not carried on a scored row, and LongMemEval_S's
    // prompt includes it, so the second pass must be shown the same `<today>`
    // the first one saw. LoCoMo's first pass showed none.
    let prompt = match base_metrics.get("corpus").and_then(serde_json::Value::as_str) {
        Some("longmemeval_s") => Prompt::WithToday(
            longmemeval::load(dataset)
                .context("load corpus")?
                .into_iter()
                .map(|q| (q.question_id, q.question_date))
                .collect(),
        ),
        Some("locomo") => {
            anyhow::ensure!(
                base_metrics.get("question_date").and_then(serde_json::Value::as_bool) != Some(true),
                "the LoCoMo base showed `<today>` (--question-date); commit-arm replays LoCoMo without it"
            );
            Prompt::NoToday
        }
        other => anyhow::bail!("commit-arm replays LongMemEval_S and LoCoMo runs; the base's corpus is {other:?}"),
    };

    let rows_path = base.join("per_question.jsonl");
    let text = std::fs::read_to_string(&rows_path)
        .with_context(|| format!("read {}", rows_path.display()))?;

    let mut report = CommitArmReport::default();
    let mut out_rows: Vec<ScoredQuestion> = Vec::new();

    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut row: ScoredQuestion =
            serde_json::from_str(line).with_context(|| format!("parse a row of {rows_path:?}"))?;
        report.rows += 1;

        if !is_abstention(&row.response_raw) {
            report.untouched += 1;
            out_rows.push(row);
            continue;
        }

        let context = row
            .evidence
            .iter()
            .enumerate()
            .map(|(n, it)| format!("[{n}] {it}"))
            .collect::<Vec<_>>()
            .join("\n");
        let user = match &prompt {
            Prompt::WithToday(dates) => {
                let today = dates.get(&row.question_id).map(String::as_str).unwrap_or("");
                format!(
                    "<memories>\n{context}\n</memories>\n<today>\n{today}\n</today>\n<question>\n{}\n</question>",
                    row.question_text
                )
            }
            Prompt::NoToday => format!(
                "<memories>\n{context}\n</memories>\n<question>\n{}\n</question>",
                row.question_text
            ),
        };

        let first = std::mem::take(&mut row.response_raw);
        let (response, fired, committed) = match pass {
            Pass::Greedy => {
                let (response, outcome) = commit_answer(&llm, &system, &user, first).await;
                (response, outcome.fired, outcome.committed)
            }
            Pass::Grounded => {
                let (response, outcome) =
                    commit_grounded(&llm, &system, &user, first, row.evidence.len()).await;
                (response, outcome.fired, outcome.committed)
            }
            Pass::Consensus(c) => {
                let (response, outcome) =
                    commit_consensus(&llm, &system, &user, &row.question_text, first, c)
                        .await;
                row.commit_samples = Some(outcome.samples.clone());
                row.commit_agreement = Some(outcome.agreement);
                (response, outcome.fired, outcome.committed)
            }
        };
        report.fired += usize::from(fired);
        report.committed += usize::from(committed);
        if !committed {
            report.untouched += 1;
        }
        if report.fired % 10 == 0 {
            eprintln!(
                "  {} declines seen, {} committed",
                report.fired, report.committed
            );
        }

        row.response_raw = response;
        out_rows.push(row);
    }

    write_arm(base, out, &out_rows, pass)?;
    Ok(report)
}

/// Write the arm as an ordinary run directory, so `judge` and `rescore` read
/// it exactly as they read a benched one.
///
/// The metrics file is the base's with `commit_answer` set, which is what
/// `standing::Ours::arm` and `rescore_run`'s `RunSpec` reconstruction read. A
/// run that did not record its own switches cannot be paired later.
fn write_arm(
    base: &Path,
    out: &Path,
    rows: &[ScoredQuestion],
    pass: Pass,
) -> Result<()> {
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;

    let lines: Vec<String> = rows
        .iter()
        .map(|r| serde_json::to_string(r).context("serialise row"))
        .collect::<Result<_>>()?;
    std::fs::write(out.join("per_question.jsonl"), lines.join("\n") + "\n")
        .context("write per_question.jsonl")?;

    let metrics_path = base.join("aggregated_metrics.json");
    let mut metrics: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&metrics_path)
            .with_context(|| format!("read {}", metrics_path.display()))?,
    )
    .context("parse aggregated_metrics.json")?;
    if let Some(obj) = metrics.as_object_mut() {
        obj.insert("commit_answer".into(), serde_json::Value::Bool(true));
        obj.insert(
            "derived_from".into(),
            serde_json::Value::String(base.display().to_string()),
        );
        // M45's parameters, or nothing: an artifact without them is M42's
        // arm, and `standing` reads either as an arm of the defaults.
        match pass {
            Pass::Greedy => {}
            Pass::Consensus(c) => {
                obj.insert("commit_samples".into(), serde_json::json!(c.samples));
                obj.insert("commit_seed".into(), serde_json::json!(c.seed));
                obj.insert("commit_agree".into(), serde_json::json!(c.agree));
            }
            Pass::Grounded => {
                obj.insert("commit_grounded".into(), serde_json::Value::Bool(true));
            }
        }
    }
    std::fs::write(
        out.join("aggregated_metrics.json"),
        serde_json::to_string_pretty(&metrics).context("serialise metrics")?,
    )
    .context("write aggregated_metrics.json")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_is_either_untouched_or_committed() {
        assert!(CommitArmReport {
            rows: 500,
            fired: 90,
            committed: 60,
            untouched: 440,
        }
        .is_consistent());
    }

    /// A row that is neither untouched nor committed means a response moved
    /// without the mechanism claiming it, which would silently destroy the
    /// control the whole arm rests on.
    #[test]
    fn an_unaccounted_row_is_inconsistent() {
        assert!(!CommitArmReport {
            rows: 500,
            fired: 90,
            committed: 60,
            untouched: 439,
        }
        .is_consistent());
    }

    #[test]
    fn more_commits_than_declines_is_impossible() {
        assert!(!CommitArmReport {
            rows: 10,
            fired: 2,
            committed: 3,
            untouched: 7,
        }
        .is_consistent());
    }
}
