//! `myelin-eval coverage` — did the reader ever see the gold evidence? (M21).
//!
//! # Why this exists as an artifact rather than a script
//!
//! Every milestone since M9 has argued about whether a loss is retrieval's or
//! the reader's, and every such argument has been settled by an ad-hoc script
//! that no longer exists. Both benchmark corpora annotate the answer-bearing
//! material themselves — LongMemEval_S flags the evidence turn with
//! `has_answer`, LoCoMo cites the evidence turns by `dia_id` — so gold
//! coverage of the composed evidence is computable **offline, deterministically
//! and with no model call**. Making it a subcommand makes retrieval recall a
//! reported column beside the judged column instead of an inference.
//!
//! # The finding it was built for
//!
//! `RetrieveConfig::rerank_depth` is 25 and `retrieve.rs` sets
//! `depth = rerank_depth.max(query.budget.k)`, so a `k = 6` run and a `k = 25`
//! run over the same store see an *identical* 25-candidate reranked pool.
//! Measured over LongMemEval_S `temporal-reasoning`, mean gold-turn recall of
//! the **composed** set is 0.662 at k=6 and 0.852 at k=25 — the same pool. The
//! evidence is discarded by `compose`'s top-k truncation, not missed by
//! retrieval.
//!
//! # This is not [`crate::evidence_audit`]
//!
//! That module reads the **vendored LME-V2 harness's** `HarnessRow` and asks an
//! LLM whether the evidence was *sufficient* — a judged, model-graded verdict
//! on a corpus with no per-turn annotation. This one reads a
//! [`crate::bench::ScoredQuestion`] row and computes gold *coverage* from the
//! dataset's own labels. One is a judgement, the other is arithmetic; they
//! answer the same question on two corpora that admit different instruments.
//! `coverage` is therefore undefined for LME-V2 and hard-errors on it rather
//! than reporting an empty table.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::bench::ScoredQuestion;
use crate::datasets::{locomo, longmemeval};

/// A gold unit's text must be longer than this to be matchable.
///
/// Short turns (`"Thanks!"`, `"Sounds good"`) occur verbatim in almost any
/// evidence set, so counting them as found would report coverage of the
/// corpus's filler rather than of its evidence.
const MIN_GOLD_CHARS: usize = 30;

/// How much of a gold unit must appear in an evidence value for it to count.
///
/// A prefix and not the whole text: `compose` emits the record, and a record
/// is a consolidated form of the turn rather than the turn verbatim, so a
/// full-text equality test would report ~0 everywhere. 80 characters is long
/// enough to be unique in a 500-session haystack and short enough to survive
/// the write path's trailing edits.
const GOLD_PREFIX_CHARS: usize = 80;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageRow {
    pub question_id: String,
    pub category: u8,
    pub gold_units: usize,
    pub found: usize,
    pub recall: f64,
    /// `"all"` | `"partial"` | `"none"`
    pub bucket: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryCoverage {
    pub category: u8,
    pub annotated: usize,
    pub mean_recall: f64,
    pub all: usize,
    pub partial: usize,
    pub none: usize,
    /// Mean scored value within each bucket — the column that says whether
    /// coverage is the binding constraint. `0.0` when the bucket is empty;
    /// read it against the count beside it.
    pub mean_score_all: f64,
    pub mean_score_partial: f64,
    pub mean_score_none: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub run: String,
    pub corpus: String,
    /// Rows with at least one annotated gold unit.
    pub annotated: usize,
    /// Rows the dataset does not annotate (LoCoMo adversarial, LongMemEval
    /// `_abs`).
    pub unannotated: usize,
    pub mean_recall: f64,
    /// LoCoMo evidence ids that resolve to no turn. Excluded from every
    /// denominator and reported rather than silently dropped: a loader change
    /// that stopped resolving `dia_id` would otherwise look like perfect
    /// coverage of nothing.
    pub dangling_evidence: usize,
    pub by_category: Vec<CategoryCoverage>,
    pub rows: Vec<CoverageRow>,
}

/// Strip a leading `[YYYY-MM-DD]` stamp and the whitespace after it.
///
/// `ComposeConfig::stamp_valid_time` is on by default, so every evidence value
/// begins with one; leaving it in place would not break the substring test
/// (the stamp is a prefix of the value, not of the gold) but stripping keeps
/// the matched text comparable to the raw corpus turn.
///
/// Hand-rolled rather than `regex`: this is the whole of `^\[\d{4}-\d{2}-\d{2}\]\s*`
/// and the workspace has no regex dependency to spend on it.
fn strip_stamp(value: &str) -> &str {
    let b = value.as_bytes();
    if b.len() < 12 || b[0] != b'[' || b[11] != b']' {
        return value;
    }
    let shaped = |i: usize, d: bool| {
        if d {
            b[i].is_ascii_digit()
        } else {
            b[i] == b'-'
        }
    };
    let ok = (1..=4).all(|i| shaped(i, true))
        && shaped(5, false)
        && shaped(6, true)
        && shaped(7, true)
        && shaped(8, false)
        && shaped(9, true)
        && shaped(10, true);
    if !ok {
        return value;
    }
    value[12..].trim_start()
}

/// Does this gold unit appear in the composed evidence?
fn is_found(gold: &str, evidence: &[String]) -> bool {
    let c = gold.trim();
    if c.chars().count() <= MIN_GOLD_CHARS {
        return false;
    }
    // Char-safe: a byte slice at 80 panics on this corpus, which is full of
    // emoji and smart quotes.
    let prefix = c
        .char_indices()
        .nth(GOLD_PREFIX_CHARS)
        .map_or(c, |(i, _)| &c[..i]);
    evidence.iter().any(|e| strip_stamp(e).contains(prefix))
}

/// The gold units for one run row, by corpus.
enum Golds {
    /// Resolved gold unit texts.
    Units(Vec<String>),
}

/// Read `<run>/per_question.jsonl`.
fn read_rows(run_dir: &Path) -> Result<Vec<ScoredQuestion>> {
    let path = run_dir.join("per_question.jsonl");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: ScoredQuestion = serde_json::from_str(line).with_context(|| {
            format!(
                "{} is not a `bench` run directory (its per_question.jsonl has no \
                 `exact_match`/`tenant`); the vendored LME-V2 harness writes a \
                 different row shape",
                run_dir.display()
            )
        })?;
        rows.push(row);
    }
    // `ScoredQuestion::evidence` is `#[serde(default)]` and was only added in
    // M19, so every run before it parses cleanly with an empty evidence list
    // on every row — and would report 0.000 recall, which is indistinguishable
    // from a run that retrieved nothing. An instrument that cannot tell those
    // apart is worse than no instrument, so say which one this is.
    anyhow::ensure!(
        rows.is_empty() || rows.iter().any(|r| !r.evidence.is_empty()),
        "{} records no composed evidence on any of its {} rows: it predates \
         `ScoredQuestion::evidence` (M19), so what the reader was shown is not \
         on disk and coverage is not computable from it. Re-run the arm.",
        path.display(),
        rows.len()
    );
    Ok(rows)
}

/// Which corpus produced this run.
fn read_corpus(run_dir: &Path) -> Result<String> {
    let path = run_dir.join("aggregated_metrics.json");
    let metrics: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))?;
    let corpus = metrics
        .get("corpus")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    match corpus.as_str() {
        "longmemeval_s" | "locomo" => Ok(corpus),
        other => bail!(
            "coverage is not defined for corpus {other:?} (field `corpus` in {}): \
             it needs a per-turn gold annotation, which only `longmemeval_s` \
             (`has_answer`) and `locomo` (`qa[].evidence`) carry. LME-V2 has none — \
             use `myelin-eval evidence-audit` there.",
            path.display()
        ),
    }
}

/// Gold units per `question_id`, plus the dangling-evidence count.
fn golds_longmemeval(dataset: &Path, rows: &[ScoredQuestion]) -> Result<HashMap<String, Golds>> {
    let items = longmemeval::load(dataset).context("load longmemeval_s")?;
    let by_id: HashMap<&str, &longmemeval::LongMemEvalItem> = items
        .iter()
        .map(|it| (it.question_id.as_str(), it))
        .collect();
    let mut out = HashMap::with_capacity(rows.len());
    for row in rows {
        let item = by_id.get(row.question_id.as_str()).with_context(|| {
            format!(
                "run row {} has no item in {}; the run and the dataset disagree",
                row.question_id,
                dataset.display()
            )
        })?;
        let units: Vec<String> = item
            .haystack_sessions
            .iter()
            .flatten()
            .filter(|t| t.has_answer == Some(true))
            .map(|t| t.content.clone())
            .collect();
        out.insert(row.question_id.clone(), Golds::Units(units));
    }
    Ok(out)
}

/// LoCoMo: `question_id` is `{sample_id}#{i}` over the **unfiltered** `qa`
/// list (`bench.rs`), so the index is a direct subscript.
fn golds_locomo(
    dataset: &Path,
    rows: &[ScoredQuestion],
) -> Result<(HashMap<String, Golds>, usize)> {
    let conversations = locomo::load(dataset).context("load locomo")?;
    let mut by_sample: HashMap<&str, (&locomo::LocomoConversation, HashMap<&str, &str>)> =
        HashMap::new();
    for conv in &conversations {
        let mut turns: HashMap<&str, &str> = HashMap::new();
        for session in &conv.sessions {
            for turn in &session.turns {
                turns.entry(turn.dia_id.as_str()).or_insert(&turn.text);
            }
        }
        by_sample.insert(conv.sample_id.as_str(), (conv, turns));
    }

    let mut dangling = 0usize;
    let mut out = HashMap::with_capacity(rows.len());
    for row in rows {
        let (sample, index) = row.question_id.rsplit_once('#').with_context(|| {
            format!(
                "LoCoMo run row {} is not `{{sample_id}}#{{index}}`",
                row.question_id
            )
        })?;
        let index: usize = index.parse().with_context(|| {
            format!("LoCoMo run row {} has a non-numeric index", row.question_id)
        })?;
        let (conv, turns) = by_sample.get(sample).with_context(|| {
            format!(
                "run row {} names sample {sample}, absent from {}",
                row.question_id,
                dataset.display()
            )
        })?;
        let qa = conv.qa.get(index).with_context(|| {
            format!(
                "run row {} indexes qa[{index}] but sample {sample} has {} items",
                row.question_id,
                conv.qa.len()
            )
        })?;
        let mut units = Vec::with_capacity(qa.evidence.len());
        for id in &qa.evidence {
            match turns.get(id.as_str()) {
                Some(text) => units.push((*text).to_string()),
                // Counted, never silently dropped: LoCoMo cites a handful of
                // ids that its own sessions do not contain, and a denominator
                // that quietly shrank would overstate recall.
                None => dangling += 1,
            }
        }
        out.insert(row.question_id.clone(), Golds::Units(units));
    }
    Ok((out, dangling))
}

/// Default dataset path for a corpus.
fn default_dataset(corpus: &str) -> PathBuf {
    match corpus {
        "locomo" => PathBuf::from("data/locomo10.json"),
        _ => PathBuf::from("data/longmemeval_s.json"),
    }
}

pub fn run(run_dir: &Path, dataset: Option<&Path>) -> Result<CoverageReport> {
    let corpus = read_corpus(run_dir)?;
    let rows = read_rows(run_dir)?;
    let dataset_path = dataset
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_dataset(&corpus));

    let (golds, dangling_evidence) = if corpus == "locomo" {
        golds_locomo(&dataset_path, &rows)?
    } else {
        (golds_longmemeval(&dataset_path, &rows)?, 0)
    };

    let mut out_rows = Vec::with_capacity(rows.len());
    let mut unannotated = 0usize;
    for row in &rows {
        let Golds::Units(units) = golds
            .get(&row.question_id)
            .expect("every row got a gold entry above");
        if units.is_empty() {
            unannotated += 1;
            continue;
        }
        let found = units.iter().filter(|u| is_found(u, &row.evidence)).count();
        let bucket = if found == units.len() {
            "all"
        } else if found == 0 {
            "none"
        } else {
            "partial"
        };
        out_rows.push(CoverageRow {
            question_id: row.question_id.clone(),
            category: row.category,
            gold_units: units.len(),
            found,
            recall: found as f64 / units.len() as f64,
            bucket: bucket.to_string(),
            score: row.score,
        });
    }

    let annotated = out_rows.len();
    let mean_recall = mean(out_rows.iter().map(|r| r.recall));

    let mut categories: Vec<u8> = out_rows.iter().map(|r| r.category).collect();
    categories.sort_unstable();
    categories.dedup();
    let by_category = categories
        .into_iter()
        .map(|category| {
            let rows: Vec<&CoverageRow> =
                out_rows.iter().filter(|r| r.category == category).collect();
            let bucket = |name: &str| -> Vec<&&CoverageRow> {
                rows.iter().filter(|r| r.bucket == name).collect()
            };
            let (all, partial, none) = (bucket("all"), bucket("partial"), bucket("none"));
            CategoryCoverage {
                category,
                annotated: rows.len(),
                mean_recall: mean(rows.iter().map(|r| r.recall)),
                all: all.len(),
                partial: partial.len(),
                none: none.len(),
                mean_score_all: mean(all.iter().map(|r| r.score)),
                mean_score_partial: mean(partial.iter().map(|r| r.score)),
                mean_score_none: mean(none.iter().map(|r| r.score)),
            }
        })
        .collect();

    let report = CoverageReport {
        run: run_dir.display().to_string(),
        corpus,
        annotated,
        unannotated,
        mean_recall,
        dangling_evidence,
        by_category,
        rows: out_rows,
    };

    let path = run_dir.join("coverage.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(report)
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut n = 0usize;
    let mut total = 0.0;
    for v in values {
        n += 1;
        total += v;
    }
    if n == 0 {
        0.0
    } else {
        total / n as f64
    }
}

pub fn print_table(report: &CoverageReport) {
    println!("\ncoverage — {} ({})", report.run, report.corpus);
    println!(
        "  annotated {} | unannotated {} | mean gold recall {:.3}{}",
        report.annotated,
        report.unannotated,
        report.mean_recall,
        if report.corpus == "locomo" {
            format!(" | dangling evidence {}", report.dangling_evidence)
        } else {
            String::new()
        }
    );
    println!(
        "\n  {:<4} {:>5} {:>7}  {:>5} {:>7}  {:>7} {:>7}  {:>5} {:>7}",
        "cat", "n", "recall", "all", "score", "partial", "score", "none", "score"
    );
    let cell = |count: usize, value: f64| {
        if count == 0 {
            "      —".to_string()
        } else {
            format!("{value:>7.3}")
        }
    };
    for c in &report.by_category {
        println!(
            "  {:<4} {:>5} {:>7.3}  {:>5} {}  {:>7} {}  {:>5} {}",
            c.category,
            c.annotated,
            c.mean_recall,
            c.all,
            cell(c.all, c.mean_score_all),
            c.partial,
            cell(c.partial, c.mean_score_partial),
            c.none,
            cell(c.none, c.mean_score_none),
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Tests. The fixture tests are hermetic and pin the matcher's rules; the
// `artifacts` test pins the three measured M19 recall figures and is gated
// because `per_question.jsonl` and `/data/` are both gitignored (see
// `.gitignore`) — they exist on the measurement host, not in a checkout.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_date_stamp_is_stripped_and_a_malformed_one_is_not() {
        assert_eq!(strip_stamp("[2023-05-20] hello"), "hello");
        // Two spaces, per `\s*`.
        assert_eq!(strip_stamp("[2023-05-20]   hello"), "hello");
        // Not a stamp: left intact rather than eating 12 characters of text.
        assert_eq!(strip_stamp("[2023-5-20] hello"), "[2023-5-20] hello");
        assert_eq!(strip_stamp("no stamp here"), "no stamp here");
        assert_eq!(strip_stamp("[short]"), "[short]");
    }

    /// The floor is the whole reason a "coverage" number means anything: a
    /// four-word turn occurs in every evidence set ever composed.
    #[test]
    fn a_short_gold_unit_never_counts_as_found() {
        let evidence = vec!["[2023-05-20] Thanks! That helps a lot.".to_string()];
        assert!(!is_found("Thanks!", &evidence));
        // 31 characters clears the floor and is found in the same value.
        let long = "Thanks! That helps a lot, truly";
        assert_eq!(long.chars().count(), 31);
        assert!(is_found(long, &[format!("[2023-05-20] {long} — and more")]));
    }

    /// Only the first 80 characters have to survive: `compose` emits a
    /// consolidated record, not the raw turn.
    #[test]
    fn only_the_gold_prefix_has_to_appear() {
        let gold = "I finally replaced the area rug in the living room last Tuesday \
                    and the cat has already claimed it as her own private territory.";
        let truncated: String = gold.chars().take(95).collect();
        assert!(is_found(gold, &[format!("[2023-07-20] {truncated}")]));
        // 79 characters of the gold is one short of the prefix and must miss.
        let too_short: String = gold.chars().take(79).collect();
        assert!(!is_found(gold, &[format!("[2023-07-20] {too_short}")]));
    }

    /// A byte slice at offset 80 panics mid-codepoint on this corpus.
    #[test]
    fn a_multibyte_gold_unit_does_not_panic() {
        let gold = "🎧".repeat(100);
        assert!(!is_found(&gold, &["nothing".to_string()]));
        assert!(is_found(&gold, &[format!("[2023-01-01] {gold}")]));
    }
}

/// Pins the instrument against the three M19 runs it was calibrated on.
///
/// Gated: `runs/**/per_question.jsonl` and `/data/` are gitignored, so these
/// artifacts are present on the measurement host and absent from a clean
/// checkout. Run with `cargo test -p myelin-eval --features artifacts`.
#[cfg(all(test, feature = "artifacts"))]
mod artifact_tests {
    use super::*;

    /// `cargo test` runs with the **crate** as the working directory, and
    /// the artifacts and datasets both live at the workspace root.
    fn workspace(rel: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(rel)
    }

    fn recall_of(run: &str) -> CoverageReport {
        let dir = workspace(run);
        super::run(&dir, Some(&workspace(dataset_for(run))))
            .unwrap_or_else(|e| panic!("coverage over {}: {e}", dir.display()))
    }

    fn dataset_for(run: &str) -> &'static str {
        if run.contains("locomo") {
            "data/locomo10.json"
        } else {
            "data/longmemeval_s.json"
        }
    }

    /// The figures every M21 conclusion is read against. A matcher refactor
    /// that moves one of them has changed what "coverage" means.
    #[test]
    fn the_m19_temporal_runs_reproduce_their_measured_recall() {
        let base = recall_of("runs/m19_lme_t_base");
        assert_eq!(base.annotated, 132);
        assert!(
            (base.mean_recall - 0.662).abs() < 0.002,
            "base recall {}",
            base.mean_recall
        );

        let investigate = recall_of("runs/m19_lme_t_armWinv");
        assert_eq!(investigate.annotated, 132);
        assert!(
            (investigate.mean_recall - 0.653).abs() < 0.002,
            "investigate recall {}",
            investigate.mean_recall
        );

        let wide = recall_of("runs/m19_lme_t_armW25");
        assert_eq!(wide.annotated, 132);
        assert!(
            (wide.mean_recall - 0.852).abs() < 0.002,
            "k=25 recall {}",
            wide.mean_recall
        );
        // 106 of 132 questions have their complete gold evidence inside the
        // same 25-item pool the k=6 arm ranked over.
        let all: usize = wide.by_category.iter().map(|c| c.all).sum();
        assert_eq!(all, 106);
    }

    /// LoCoMo must parse through the other loader, against the run that
    /// actually recorded its composed evidence — `runs/locomo_recall` is the
    /// M9 baseline and predates the field.
    #[test]
    fn a_locomo_run_resolves_its_evidence_ids() {
        let report = recall_of("runs/m19_locomo_full");
        assert_eq!(report.corpus, "locomo");
        assert!(report.annotated > 0, "no annotated LoCoMo rows");
        assert!(
            report.mean_recall > 0.0,
            "LoCoMo gold turns never matched the composed evidence"
        );
    }

    /// The guard that stops the instrument lying. Every run before M19 has
    /// an empty `evidence` on every row, and 0.000 recall from that is
    /// indistinguishable from a run that retrieved nothing.
    #[test]
    fn a_pre_m19_run_is_refused_rather_than_scored_at_zero() {
        let err = super::run(&workspace("runs/locomo_recall"), None)
            .expect_err("a run with no recorded evidence must not report a recall");
        let message = format!("{err}");
        assert!(
            message.contains("predates"),
            "the error must name the cause: {message}"
        );
    }
}
