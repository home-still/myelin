//! `myelin-eval evidence-audit` — which side of the pipeline loses an
//! answerable LongMemEval-V2 question (M16).
//!
//! # The question this answers, and why it is not another parameter attempt
//!
//! `docs/measurements/m6-g1-breakeven.md` measured LME-V2-Small at 36.7%
//! overall / 42.9% answerable against a break-even bar of 51.0, and
//! `docs/measurements/m6-abstention-gate.md` closed the abstention half: all
//! three memory-side levers were measured and rejected. Four parameter
//! attempts have been spent guessing at the remaining gap. This module does
//! not make a fifth. It splits the *wrong answerable* rows by a single fact
//! that is already on disk — whether the evidence the reader was shown
//! contained the reference answer:
//!
//! - *insufficient + wrong* is retrieval's loss: the answer was never put in
//!   front of the reader, so a width/depth arm can still move it.
//! - *sufficient + wrong* is the reader's loss: the answer was there and the
//!   pinned `qwen3.5-9b` missed it, which no memory-side knob repairs.
//!
//! The share of wrong answerable rows that are *sufficient* selects M17's
//! direction under the rule fixed in
//! `docs/measurements/m16-evidence-sufficiency.md` before the measurement ran.
//!
//! # This reads the vendored harness's rows, not `bench`'s
//!
//! [`crate::judge`] deliberately hard-errors on a vendored-harness directory
//! ("is not a `bench` run directory"). This module is the mirror: it reads
//! [`HarnessRow`], the 26-field row `adapters/run_myelin.py` writes, and
//! hard-errors on a [`crate::bench::ScoredQuestion`] line. Pointing either at
//! the other's artifacts must fail loudly rather than produce an empty report.
//!
//! # The audit never re-scores
//!
//! `correct` is `row.score >= 0.5`, the harness's own verdict. A second scorer
//! here would make the 2x2 a comparison of two graders instead of a split of
//! one run's losses. The judge grades *evidence*; [`crate::judge`] grades
//! *answers*; neither enters a reported accuracy number.
//!
//! # Self-judging, and the two things that bound it
//!
//! The sufficiency judge is the same `qwen3.5-9b` that produced the answers —
//! no `gpt-5.2` key exists. Two checks are reported rather than asserted:
//! a deterministic gold-token-recall proxy on the `norm_phrase_set_match*`
//! families, where a gold string is literal text, and the
//! *insufficient + correct* cell, which bounds the judge's false-negative rate
//! from the direction that matters (evidence it called insufficient that
//! demonstrably sufficed). `docs/measurements/m9-judge-panel.md` measured this
//! model as the *harsher* grader at Fleiss κ 0.8813.
//!
//! Reader only: no store, no Qdrant, no embedder, no reranker.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, Result};
use myelin_core::config::MyelinConfig;
use myelin_core::llm::openai::OpenAiLlm;
use myelin_core::llm::{complete_json, CompletionRequest, Llm, Message};
use serde::{Deserialize, Serialize};

use crate::attack_live::wilson;
use crate::bench::{gold_answer, normalize};

/// One row of a vendored LongMemEval-V2 harness `per_question.jsonl`.
/// Only the fields the audit reads; the harness writes 26.
#[derive(Debug, Clone, Deserialize)]
pub struct HarnessRow {
    pub question_id: String,
    pub question_text: String,
    /// Raw type, e.g. `dynamic-environment`, `procedure-abs`.
    pub question_type: String,
    /// Mapped bucket: `static` | `dynamic` | `procedure` | `gotchas`, plus the
    /// `*-abs` variants, which are all abstention items and never judged.
    pub category: String,
    pub is_abstention_problem: bool,
    /// e.g. `mc_choice_match|require_non_empty=true`. The family before the
    /// first `|` decides whether the deterministic proxy means anything.
    pub eval_function: String,
    /// String, bool or list depending on the item — flattened by
    /// [`HarnessRow::gold`], which leaves a plain string unquoted.
    pub answer_gold: serde_json::Value,
    pub response_raw: String,
    /// The harness's own verdict: 1.0 or 0.0.
    pub score: f64,
    pub memory_context: Vec<ContextItem>,
    pub memory_context_token_count: usize,
    /// A screenshot the *question* carries (29 LME-V2-Small items do). The
    /// judge is text-only, so these are counted and reported, never silently
    /// folded into the rate.
    #[serde(default)]
    pub question_image: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContextItem {
    #[serde(rename = "type")]
    pub kind: String,
    pub value: String,
}

impl HarnessRow {
    /// The harness's verdict, never recomputed.
    pub fn correct(&self) -> bool {
        self.score >= 0.5
    }

    /// The evidence the reader was shown, text items only.
    pub fn evidence(&self) -> String {
        self.memory_context
            .iter()
            .filter(|i| i.kind == "text")
            .map(|i| i.value.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Context items the audit could not show a text-only judge.
    pub fn dropped_items(&self) -> usize {
        self.memory_context.iter().filter(|i| i.kind != "text").count()
    }

    /// `answer_gold` flattened to the string the judge and the proxy see.
    pub fn gold(&self) -> String {
        gold_answer(Some(&self.answer_gold))
    }

    /// The `eval_function` family, before the first `|`.
    pub fn eval_family(&self) -> &str {
        self.eval_function
            .split('|')
            .next()
            .unwrap_or(&self.eval_function)
    }
}

/// Which side of the pipeline lost the question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sufficiency {
    /// The evidence contains what the reference answer states.
    Sufficient,
    /// It does not. Retrieval never put the answer in front of the reader.
    Insufficient,
}

impl Sufficiency {
    /// Row index into [`AuditReport::cells`].
    const fn idx(self) -> usize {
        match self {
            Sufficiency::Insufficient => INSUFF,
            Sufficiency::Sufficient => SUFF,
        }
    }
}

/// `cells[INSUFF][..]` — retrieval never supplied the answer.
const INSUFF: usize = 0;
/// `cells[SUFF][..]` — the answer was in the evidence.
const SUFF: usize = 1;

/// `<run>/evidence_audit.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditFile {
    pub model: String,
    /// `question_id` -> the judge's verdict.
    pub verdicts: BTreeMap<String, Sufficiency>,
}

#[derive(Debug, Clone)]
pub struct AuditReport {
    pub run: String,
    pub questions: usize,
    pub answerable: usize,
    /// `[sufficiency][correct]` counts over answerable rows:
    /// `[Insufficient][0]`, `[Insufficient][1]`, `[Sufficient][0]`, `[Sufficient][1]`.
    pub cells: [[usize; 2]; 2],
    pub per_category: Vec<CategoryCells>,
    /// Deterministic gold-token recall, reported only for the
    /// `norm_phrase_set_match*` families where a gold string is literal text.
    pub proxy: ProxyAgreement,
    /// Up to 5 verbatim examples from each diagnostic cell.
    pub examples: Vec<AuditExample>,
    pub abstention_correct: usize,
    pub abstention_total: usize,
    pub judged: usize,
    pub cached: usize,
    /// Non-text `memory_context` items the text-only judge could not see.
    pub context_items_dropped: usize,
    /// Answerable rows whose *question* carries a screenshot.
    pub questions_with_image: usize,
}

impl AuditReport {
    /// Wrong answerable rows whose evidence sufficed — the numerator of `S`.
    pub fn sufficient_but_wrong(&self) -> usize {
        self.cells[SUFF][0]
    }

    /// Answerable rows judged, i.e. the four cells' sum. Less than
    /// [`Self::answerable`] only under `--limit` or an interrupted pass.
    pub fn scored(&self) -> usize {
        self.cells.iter().flatten().sum()
    }

    /// Answerable rows the harness scored wrong, among the judged.
    pub fn wrong(&self) -> usize {
        self.cells[INSUFF][0] + self.cells[SUFF][0]
    }

    /// Answerable rows the harness scored right, among the judged.
    pub fn right(&self) -> usize {
        self.cells[INSUFF][1] + self.cells[SUFF][1]
    }

    /// `S` — the share of wrong answerable rows whose evidence sufficed.
    /// The decision rule in `m16-evidence-sufficiency.md` reads this.
    pub fn s(&self) -> f64 {
        let wrong = self.wrong();
        if wrong == 0 {
            return 0.0;
        }
        self.cells[SUFF][0] as f64 / wrong as f64
    }

    /// Accuracy if the **reader** were perfect on the evidence it already
    /// gets: every *sufficient + wrong* row flips, the abstention column is
    /// the measured one, and *insufficient + wrong* stays wrong because no
    /// reader repairs evidence that does not contain the answer.
    ///
    /// This is the ceiling the rule's reader-limited branch is judged on: if
    /// it is below 51.0, perfecting the reader over today's evidence cannot
    /// clear the bar, whatever `S` says.
    pub fn reader_fix_ceiling(&self) -> f64 {
        self.over_questions(self.right() + self.sufficient_but_wrong() + self.abstention_correct)
    }

    /// Accuracy if **retrieval** put the answer in front of the reader on
    /// every *insufficient* row and the reader then answered at its measured
    /// [`Self::p_correct_given_sufficient`] rate. The retrieval-limited
    /// branch's ceiling, and the number that says whether the branch has
    /// enough headroom to be worth running.
    pub fn retrieval_fix_ceiling(&self) -> f64 {
        let gained = self.p_correct_given_sufficient() * self.cells[INSUFF][0] as f64;
        if self.questions == 0 {
            return 0.0;
        }
        ((self.right() + self.abstention_correct) as f64 + gained) / self.questions as f64
    }

    /// The same counterfactual with a perfect reader on the repaired rows —
    /// the arithmetic upper bound, stated so the realistic figure above is
    /// not mistaken for one.
    pub fn retrieval_fix_ceiling_perfect(&self) -> f64 {
        self.over_questions(self.right() + self.cells[INSUFF][0] + self.abstention_correct)
    }

    /// The run's measured accuracy over the same denominator, for the line
    /// the ceilings are compared against.
    pub fn measured(&self) -> f64 {
        self.over_questions(self.right() + self.abstention_correct)
    }

    /// P(correct | sufficient) — the reader's hit rate when the answer is in
    /// front of it.
    pub fn p_correct_given_sufficient(&self) -> f64 {
        let n = self.cells[SUFF][0] + self.cells[SUFF][1];
        if n == 0 {
            return 0.0;
        }
        self.cells[SUFF][1] as f64 / n as f64
    }

    /// P(correct | insufficient) — how often the row was scored right anyway,
    /// from pretraining, a lucky multiple-choice guess, or a judge error.
    pub fn p_correct_given_insufficient(&self) -> f64 {
        let n = self.cells[INSUFF][0] + self.cells[INSUFF][1];
        if n == 0 {
            return 0.0;
        }
        self.cells[INSUFF][1] as f64 / n as f64
    }

    /// How much the judge's label moves the harness's own verdict, in points.
    /// This is the judge-validity number: a label that does not predict
    /// correctness is not measuring sufficiency, whatever its rate looks like.
    pub fn discrimination(&self) -> f64 {
        self.p_correct_given_sufficient() - self.p_correct_given_insufficient()
    }

    /// Worst-case `S` if **every** *insufficient + correct* row is a judge
    /// false negative and the same false-negative rate holds on the wrong
    /// rows. Reported because the pre-registered 20%-of-answerable trigger on
    /// that cell fires on both LME-V2-Small domains.
    pub fn s_worst_case(&self) -> f64 {
        let correct = self.right();
        let wrong = self.wrong();
        if correct == 0 || wrong == 0 {
            return 0.0;
        }
        let fn_rate = self.cells[INSUFF][1] as f64 / correct as f64;
        (self.cells[SUFF][0] as f64 + fn_rate * self.cells[INSUFF][0] as f64) / wrong as f64
    }

    fn over_questions(&self, n: usize) -> f64 {
        if self.questions == 0 {
            return 0.0;
        }
        n as f64 / self.questions as f64
    }
}

#[derive(Debug, Clone)]
pub struct CategoryCells {
    pub category: String,
    pub cells: [[usize; 2]; 2],
}

/// The deterministic proxy against the judge, with the *direction* of every
/// disagreement — which is the informative part. Gold-token recall over
/// ~9,000 tokens of accessibility tree over-calls sufficiency (a four-token
/// gold scatters across an unrelated page), so it is a trustworthy witness
/// only in one direction: where the gold's tokens are *absent*, the evidence
/// cannot contain the answer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProxyAgreement {
    pub n: usize,
    pub agree: usize,
    /// Proxy says sufficient, judge says insufficient.
    pub only_proxy: usize,
    /// Proxy says insufficient, judge says sufficient.
    pub only_judge: usize,
    /// Rows where the proxy found no gold tokens at all, and how many of
    /// those the judge also called insufficient — the direction where the
    /// deterministic signal is decisive.
    pub proxy_insufficient: usize,
    pub proxy_insufficient_confirmed: usize,
}

impl ProxyAgreement {
    pub fn rate(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.agree as f64 / self.n as f64
    }

    /// Agreement restricted to the direction the proxy can actually witness.
    pub fn decisive_rate(&self) -> f64 {
        if self.proxy_insufficient == 0 {
            return 0.0;
        }
        self.proxy_insufficient_confirmed as f64 / self.proxy_insufficient as f64
    }
}

#[derive(Debug, Clone)]
pub struct AuditExample {
    pub question_id: String,
    pub sufficiency: Sufficiency,
    pub correct: bool,
    pub question_text: String,
    pub gold: String,
    pub response: String,
    /// Evidence, truncated to 1,200 characters for the printed dump.
    pub evidence_head: String,
}

/// The cell arithmetic, over rows and a verdict map. No `Llm`, no `Path`, no
/// cache file: every number the measurement doc reports is this function.
#[derive(Debug, Clone, Default)]
pub struct Tally {
    pub answerable: usize,
    pub cells: [[usize; 2]; 2],
    pub per_category: Vec<CategoryCells>,
    pub abstention_correct: usize,
    pub abstention_total: usize,
}

/// The rubric, pinned. Rev 2 — **the judgement clauses are byte-identical to
/// rev 1; only the output channel changed.**
///
/// Every judgement clause is load-bearing. "not the reference answer" stops
/// the judge grading the gold; "your own knowledge of the world is not
/// evidence" stops it marking sufficient on a fact it simply knows, which is
/// exactly the failure the *insufficient + correct* cell measures in the
/// reader.
///
/// Rev 1 ended `Reply with exactly one character, 1 or 0, and nothing else.`
/// and relied on the model obeying it. On row 29 of the `web` run
/// (`23aecb38`) it replied `"The evidence provided describes"` instead, which
/// [`audit`] correctly refused to coerce into a verdict — and would have
/// refused on every prose reply after it. The fix is the mechanism this
/// codebase already uses where "a free-form answer is unusable"
/// ([`CompletionRequest::json_schema`], as `extract`, `consolidate` and
/// `pipeline::adjudicate` do): constrained decoding against
/// [`verdict_schema`], so format compliance is enforced by the decoder rather
/// than hoped for. The 28 rev-1 verdicts were discarded; every label reported
/// in `m16-evidence-sufficiency.md` comes from this prompt and this decoder.
const AUDIT_SYSTEM: &str = "You are auditing a retrieval system, not a reader.
You are given a question, the reference answer, and the evidence a reader was shown.
Decide only this: does the evidence contain the information needed to produce the reference answer?
Answer true if it does, false if it does not.
Judge the evidence, not the reference answer, and not whether the reference answer is correct.
Use only what the evidence states. Your own knowledge of the world is not evidence.
Reply with JSON only: {\"sufficient\": true} or {\"sufficient\": false}.";

/// The judge's whole output surface. `additionalProperties: false` because a
/// verdict with a commentary field attached is a verdict the decoder was not
/// constraining.
fn verdict_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "sufficient": { "type": "boolean" } },
        "required": ["sufficient"],
        "additionalProperties": false
    })
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct SufficiencyVerdict {
    sufficient: bool,
}

/// Gold-token recall at or above this is the proxy's `Sufficient`.
const PROXY_THRESHOLD: f64 = 0.8;
/// Characters of evidence kept per printed example.
const EVIDENCE_HEAD: usize = 1200;
/// Examples dumped per diagnostic cell.
const EXAMPLES_PER_CELL: usize = 5;

/// Fraction of the gold's tokens that appear in the evidence, as a multiset
/// intersection — the same overlap loop [`crate::bench::token_f1`] uses, so a
/// gold token repeated twice needs two occurrences in the evidence.
///
/// Recall, not F1: the evidence is thousands of tokens and the gold is a
/// handful, so precision is meaningless here.
pub fn gold_token_recall(gold: &str, evidence: &str) -> f64 {
    let gold = normalize(gold);
    if gold.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for t in &gold {
        *counts.entry(t.as_str()).or_insert(0) += 1;
    }
    let mut overlap = 0i64;
    for t in normalize(evidence) {
        if let Some(e) = counts.get_mut(t.as_str()) {
            if *e > 0 {
                *e -= 1;
                overlap += 1;
            }
        }
    }
    overlap as f64 / gold.len() as f64
}

/// Split every row into the 2x2, plus the abstention column the ceiling needs.
///
/// Abstention rows are counted and never judged: `m6-abstention-gate.md`
/// measured and rejected all three memory-side abstention levers, so
/// re-litigating them here would spend the GPU window on a closed question.
/// Answerable rows with no verdict (under `--limit`, or an interrupted pass)
/// land in no cell, which is why [`AuditReport::scored`] is reported beside
/// [`AuditReport::answerable`].
pub fn tally(rows: &[HarnessRow], verdicts: &BTreeMap<String, Sufficiency>) -> Tally {
    let mut out = Tally::default();
    let mut by_category: BTreeMap<String, [[usize; 2]; 2]> = BTreeMap::new();
    for row in rows {
        if row.is_abstention_problem {
            out.abstention_total += 1;
            out.abstention_correct += usize::from(row.correct());
            continue;
        }
        out.answerable += 1;
        let Some(verdict) = verdicts.get(&row.question_id) else {
            continue;
        };
        let (i, j) = (verdict.idx(), usize::from(row.correct()));
        out.cells[i][j] += 1;
        by_category.entry(row.category.clone()).or_default()[i][j] += 1;
    }
    out.per_category = by_category
        .into_iter()
        .map(|(category, cells)| CategoryCells { category, cells })
        .collect();
    out
}

/// How often the deterministic proxy agrees with the judge, over the rows
/// where a gold string is literal text.
///
/// `mc_choice_match` golds are `false`, `A`, `B` — token containment against
/// 9,000 tokens of evidence is meaningless there, so those rows are excluded
/// rather than counted as agreement. This makes the judge auditable; it does
/// not replace it.
pub fn proxy_agreement(
    rows: &[HarnessRow],
    verdicts: &BTreeMap<String, Sufficiency>,
) -> ProxyAgreement {
    let mut out = ProxyAgreement::default();
    for row in rows {
        if row.is_abstention_problem || !row.eval_family().starts_with("norm_phrase_set_match") {
            continue;
        }
        let Some(&verdict) = verdicts.get(&row.question_id) else {
            continue;
        };
        let proxy = if gold_token_recall(&row.gold(), &row.evidence()) >= PROXY_THRESHOLD {
            Sufficiency::Sufficient
        } else {
            Sufficiency::Insufficient
        };
        out.n += 1;
        out.agree += usize::from(proxy == verdict);
        match (proxy, verdict) {
            (Sufficiency::Sufficient, Sufficiency::Insufficient) => out.only_proxy += 1,
            (Sufficiency::Insufficient, Sufficiency::Sufficient) => out.only_judge += 1,
            _ => {}
        }
        if proxy == Sufficiency::Insufficient {
            out.proxy_insufficient += 1;
            out.proxy_insufficient_confirmed +=
                usize::from(verdict == Sufficiency::Insufficient);
        }
    }
    out
}

/// Up to [`EXAMPLES_PER_CELL`] verbatim rows from each *diagnostic* cell:
/// retrieval's loss, the reader's loss, and the judge's false-negative bound.
/// *sufficient + correct* is the system working and is not dumped.
fn collect_examples(
    rows: &[HarnessRow],
    verdicts: &BTreeMap<String, Sufficiency>,
) -> Vec<AuditExample> {
    const DIAGNOSTIC: [(Sufficiency, bool); 3] = [
        (Sufficiency::Insufficient, false),
        (Sufficiency::Sufficient, false),
        (Sufficiency::Insufficient, true),
    ];
    let mut out = Vec::new();
    for (want_suff, want_correct) in DIAGNOSTIC {
        let mut taken = 0usize;
        for row in rows {
            if taken >= EXAMPLES_PER_CELL {
                break;
            }
            if row.is_abstention_problem {
                continue;
            }
            let Some(&verdict) = verdicts.get(&row.question_id) else {
                continue;
            };
            if verdict != want_suff || row.correct() != want_correct {
                continue;
            }
            let evidence = row.evidence();
            out.push(AuditExample {
                question_id: row.question_id.clone(),
                sufficiency: verdict,
                correct: row.correct(),
                question_text: row.question_text.clone(),
                gold: row.gold(),
                response: row.response_raw.clone(),
                evidence_head: head(&evidence, EVIDENCE_HEAD),
            });
            taken += 1;
        }
    }
    out
}

/// First `n` characters, on a character boundary, with a marker when cut.
fn head(text: &str, n: usize) -> String {
    match text.char_indices().nth(n) {
        None => text.to_string(),
        Some((byte, _)) => format!("{}… [{} chars total]", &text[..byte], text.chars().count()),
    }
}

/// Judge every answerable row of a vendored-harness run directory.
///
/// `limit` truncates the docket to the first N answerable rows in file order —
/// [`crate::judge::judge_run`]'s semantics exactly — so a cost probe is
/// idempotent: a second call with the same limit judges nothing and reports
/// the same cells. Verdicts are flushed to `<run>/evidence_audit.json` after
/// every call, so an interrupted 323-row pass resumes instead of restarting.
pub async fn audit(run: &Path, limit: Option<usize>) -> Result<AuditReport> {
    let rows_path = run.join("per_question.jsonl");
    let text = std::fs::read_to_string(&rows_path)
        .with_context(|| format!("read {}", rows_path.display()))?;
    let rows = parse_rows(&text, &rows_path)?;

    let docket: Vec<&HarnessRow> = rows
        .iter()
        .filter(|r| !r.is_abstention_problem)
        .take(limit.unwrap_or(usize::MAX))
        .collect();

    let cfg = MyelinConfig::load().context("load myelin config")?;
    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("sufficiency judge client")?;
    let model = llm.id().to_string();

    let cache_path = run.join("evidence_audit.json");
    let mut verdicts: BTreeMap<String, Sufficiency> = BTreeMap::new();
    if cache_path.exists() {
        let existing: AuditFile = serde_json::from_str(
            &std::fs::read_to_string(&cache_path)
                .with_context(|| format!("read {}", cache_path.display()))?,
        )
        .with_context(|| format!("parse {}", cache_path.display()))?;
        anyhow::ensure!(
            existing.model == model,
            "{} was written by judge {:?} but the configured judge is {:?}; \
             mixing two judges' sufficiency verdicts into one 2x2 is not a \
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
        let verdict = judge_one(&llm, row).await?;
        verdicts.insert(row.question_id.clone(), verdict);
        judged += 1;
        // Flushed per row, not at the end: a 323-row pass that dies on row
        // 300 must not throw away 299 GPU-seconds of verdicts.
        write_cache(&cache_path, &model, &verdicts)?;
    }
    if judged == 0 {
        write_cache(&cache_path, &model, &verdicts)?;
    }

    let t = tally(&rows, &verdicts);
    Ok(AuditReport {
        run: run.display().to_string(),
        questions: rows.len(),
        answerable: t.answerable,
        cells: t.cells,
        per_category: t.per_category,
        proxy: proxy_agreement(&rows, &verdicts),
        examples: collect_examples(&rows, &verdicts),
        abstention_correct: t.abstention_correct,
        abstention_total: t.abstention_total,
        judged,
        cached,
        context_items_dropped: rows.iter().map(HarnessRow::dropped_items).sum(),
        questions_with_image: rows
            .iter()
            .filter(|r| !r.is_abstention_problem && r.question_image.is_some())
            .count(),
    })
}

/// Deserialise a harness `per_question.jsonl`, naming the mistake on failure.
///
/// The mirror of [`crate::judge`]'s guard: a `bench` run's rows carry
/// `category` as a number and no `eval_function`, so they fail here. An empty
/// audit from the wrong directory would look like a finished measurement.
fn parse_rows(text: &str, path: &Path) -> Result<Vec<HarnessRow>> {
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: HarnessRow = serde_json::from_str(line).with_context(|| {
            format!(
                "{} is not a vendored LongMemEval-V2 harness run: its \
                 per_question.jsonl has no `eval_function`/`memory_context` \
                 row shape. `bench` writes a different row (`exact_match`, \
                 numeric `category`) — use `myelin-eval judge` for those",
                path.display()
            )
        })?;
        rows.push(row);
    }
    anyhow::ensure!(
        !rows.is_empty(),
        "{} is empty: nothing to audit",
        path.display()
    );
    Ok(rows)
}

/// One sufficiency verdict, under constrained decoding.
///
/// A reply that does not parse as [`verdict_schema`] is an error, never
/// coerced: a defaulted `insufficient` would be indistinguishable from
/// retrieval's loss and would move `S` in the direction that licenses more
/// tuning. Takes `&dyn Llm` so the no-coercion rule is testable without a
/// server.
async fn judge_one(llm: &dyn Llm, row: &HarnessRow) -> Result<Sufficiency> {
    let user = format!(
        "<question>\n{}\n</question>\n<reference_answer>\n{}\n</reference_answer>\n<evidence>\n{}\n</evidence>",
        row.question_text,
        row.gold(),
        row.evidence()
    );
    // 16: `{"sufficient": false}` is ~8 tokens; the cap exists so a model
    // that ignores the grammar fails fast instead of narrating for 20k.
    let req = CompletionRequest::new(vec![Message::system(AUDIT_SYSTEM), Message::user(user)])
        .with_schema(verdict_schema())
        .with_max_tokens(16);
    let verdict: SufficiencyVerdict = complete_json(llm, &req)
        .await
        .with_context(|| format!("judge evidence for {}", row.question_id))?;
    Ok(if verdict.sufficient {
        Sufficiency::Sufficient
    } else {
        Sufficiency::Insufficient
    })
}

fn write_cache(
    path: &Path,
    model: &str,
    verdicts: &BTreeMap<String, Sufficiency>,
) -> Result<()> {
    let file = AuditFile {
        model: model.to_string(),
        verdicts: verdicts.clone(),
    };
    std::fs::write(path, serde_json::to_string_pretty(&file)?)
        .with_context(|| format!("write {}", path.display()))
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 {
        return 0.0;
    }
    n as f64 * 100.0 / d as f64
}

/// The 2x2, its intervals, the ceiling, the proxy, and the labels in full.
///
/// The verbatim dump is not decoration: the rule in
/// `m16-evidence-sufficiency.md` is applied by a human reading whether the
/// judge's labels are right, so the labels are part of the deliverable.
pub fn print(report: &AuditReport) {
    println!("\n=== M16 — evidence sufficiency audit: {} ===", report.run);
    println!(
        "questions {:>18}\n\
         answerable {:>17}  (judged {}, cached {}, scored {})\n\
         abstention {:>17}  ({} correct, unchanged from the run)\n\
         context items dropped {:>6}  (non-text; the judge is text-only)\n\
         answerable with image {:>6}  (question-side screenshot the judge cannot see)",
        report.questions,
        report.answerable,
        report.judged,
        report.cached,
        report.scored(),
        report.abstention_total,
        report.abstention_correct,
        report.context_items_dropped,
        report.questions_with_image,
    );
    if report.scored() < report.answerable {
        println!(
            "\nPARTIAL: {} of {} answerable rows carry a verdict. Every cell, S and the\n\
             ceiling below cover only those rows.",
            report.scored(),
            report.answerable
        );
    }

    println!("\n--- the 2x2 over answerable rows ---");
    println!("                      wrong   correct    total");
    for (label, i) in [("insufficient", INSUFF), ("sufficient", SUFF)] {
        let row = report.cells[i];
        println!(
            "{label:<14} {:>10} {:>9} {:>8}",
            row[0],
            row[1],
            row[0] + row[1]
        );
    }
    println!(
        "{:<14} {:>10} {:>9} {:>8}",
        "total",
        report.wrong(),
        report.right(),
        report.scored()
    );

    let suff_n = report.cells[SUFF][0] + report.cells[SUFF][1];
    let insuff_n = report.cells[INSUFF][0] + report.cells[INSUFF][1];
    let (s_lo, s_hi) = wilson(report.sufficient_but_wrong(), report.wrong());
    let (r_lo, r_hi) = wilson(report.cells[SUFF][1], suff_n);
    let (i_lo, i_hi) = wilson(report.cells[INSUFF][1], insuff_n);
    println!(
        "\nS = P(sufficient | wrong)      {:>6.1}%  [{:.1}, {:.1}]  ({}/{})\n\
         rule: S >= 60% reader-limited, S <= 40% retrieval-limited, else both\n\
         S worst case, if every insufficient+correct row is a judge false\n\
         negative and the same rate holds on the wrong rows: {:.1}%\n\
         \n\
         P(correct | sufficient)        {:>6.1}%  [{:.1}, {:.1}]  ({}/{})\n\
         P(correct | insufficient)      {:>6.1}%  [{:.1}, {:.1}]  ({}/{})\n\
         discrimination                 {:>+6.1} points — the judge-validity number\n\
         insufficient+correct is {:.1}% of answerable (>20% trips the\n\
         pre-registered unreliability clause and the verdict must say so)",
        report.s() * 100.0,
        s_lo * 100.0,
        s_hi * 100.0,
        report.sufficient_but_wrong(),
        report.wrong(),
        report.s_worst_case() * 100.0,
        report.p_correct_given_sufficient() * 100.0,
        r_lo * 100.0,
        r_hi * 100.0,
        report.cells[SUFF][1],
        suff_n,
        report.p_correct_given_insufficient() * 100.0,
        i_lo * 100.0,
        i_hi * 100.0,
        report.cells[INSUFF][1],
        insuff_n,
        report.discrimination() * 100.0,
        pct(report.cells[INSUFF][1], report.answerable),
    );

    println!("\n--- the same 2x2 per category ---");
    println!("category            insuf/wrong  insuf/right  suf/wrong  suf/right        S");
    for c in &report.per_category {
        let wrong = c.cells[INSUFF][0] + c.cells[SUFF][0];
        println!(
            "{:<18} {:>11} {:>12} {:>10} {:>10} {:>7.1}%",
            c.category,
            c.cells[INSUFF][0],
            c.cells[INSUFF][1],
            c.cells[SUFF][0],
            c.cells[SUFF][1],
            pct(c.cells[SUFF][0], wrong),
        );
    }

    println!("\n--- the two counterfactual ceilings, against the 51.0 bar ---");
    println!(
        "measured overall          ({} right + {} abstention-correct) / {}          = {:.1}%\n\
         reader-fix ceiling        (+ {} sufficient-but-wrong, perfect reader)        = {:.1}%\n\
         retrieval-fix ceiling     (+ {} insufficient-wrong at P(correct|suf) {:.1}%) = {:.1}%\n\
         retrieval-fix, perfect    (+ all {} insufficient-wrong answered)             = {:.1}%\n\
         break-even bar (LAFS, average query <= 26.9 s)                              =   51.0%",
        report.right(),
        report.abstention_correct,
        report.questions,
        report.measured() * 100.0,
        report.sufficient_but_wrong(),
        report.reader_fix_ceiling() * 100.0,
        report.cells[INSUFF][0],
        report.p_correct_given_sufficient() * 100.0,
        report.retrieval_fix_ceiling() * 100.0,
        report.cells[INSUFF][0],
        report.retrieval_fix_ceiling_perfect() * 100.0,
    );

    println!("\n--- deterministic proxy (norm_phrase_set_match* only) ---");
    println!(
        "gold-token recall >= {:.1} vs the judge: {}/{} agree ({:.1}%)\n\
         proxy sufficient, judge insufficient: {:>4}  (token scatter over ~9k tokens)\n\
         proxy insufficient, judge sufficient: {:>4}\n\
         the decisive direction — gold tokens absent, so the evidence cannot\n\
         contain the answer: judge agrees {}/{} ({:.1}%)\n\
         excluded: mc_choice_match golds (`false`, `A`) and the llm_*_checker\n\
         families, where token containment against ~9k evidence tokens says nothing.",
        PROXY_THRESHOLD,
        report.proxy.agree,
        report.proxy.n,
        report.proxy.rate() * 100.0,
        report.proxy.only_proxy,
        report.proxy.only_judge,
        report.proxy.proxy_insufficient_confirmed,
        report.proxy.proxy_insufficient,
        report.proxy.decisive_rate() * 100.0,
    );

    println!("\n--- verbatim examples, up to {EXAMPLES_PER_CELL} per diagnostic cell ---");
    if report.examples.is_empty() {
        println!("none: no diagnostic cell has a judged row.");
    }
    for e in &report.examples {
        println!(
            "\n[{}] {:?} + {}\nQ: {}\nGOLD: {}\nANSWER: {}\nEVIDENCE: {}",
            e.question_id,
            e.sufficiency,
            if e.correct { "correct" } else { "wrong" },
            e.question_text,
            e.gold,
            e.response,
            e.evidence_head,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, category: &str, score: f64, abstention: bool) -> HarnessRow {
        HarnessRow {
            question_id: id.to_string(),
            question_text: format!("question {id}"),
            question_type: category.to_string(),
            category: category.to_string(),
            is_abstention_problem: abstention,
            eval_function: "norm_phrase_set_match|require_non_empty=true".to_string(),
            answer_gold: serde_json::Value::String("gold".to_string()),
            response_raw: "answer".to_string(),
            score,
            memory_context: vec![ContextItem {
                kind: "text".to_string(),
                value: "evidence".to_string(),
            }],
            memory_context_token_count: 1,
            question_image: None,
        }
    }

    /// A verdict is read from the constrained field, and a reply that is not
    /// a verdict is an error. Both halves matter: an inverted mapping would
    /// flip the milestone's conclusion, and a coerced verdict would be
    /// indistinguishable from a real one.
    #[tokio::test]
    async fn a_verdict_is_read_from_json_and_prose_is_never_coerced() {
        struct Says(&'static str);
        #[async_trait::async_trait]
        impl Llm for Says {
            fn id(&self) -> &str {
                "says"
            }
            async fn raw_complete(
                &self,
                _r: &CompletionRequest,
            ) -> myelin_core::error::Result<myelin_core::llm::Completion> {
                Ok(myelin_core::llm::Completion {
                    text: self.0.to_string(),
                    tool_calls: vec![],
                    finish_reason: None,
                    usage: myelin_core::llm::Usage::default(),
                })
            }
        }
        let r = row("q", "static", 0.0, false);
        assert_eq!(
            judge_one(&Says(r#"{"sufficient": true}"#), &r).await.unwrap(),
            Sufficiency::Sufficient
        );
        assert_eq!(
            judge_one(&Says(r#"{"sufficient": false}"#), &r)
                .await
                .unwrap(),
            Sufficiency::Insufficient
        );
        let err = judge_one(&Says("The evidence provided describes"), &r)
            .await
            .expect_err("prose must not become a verdict")
            .to_string();
        assert!(err.contains("judge evidence for q"), "{err}");
    }

    /// The arithmetic every number in `m16-evidence-sufficiency.md` rests on.
    #[test]
    fn cells_split_on_score_and_sufficiency() {
        let rows = vec![
            row("iw", "static", 0.0, false),
            row("ir", "static", 1.0, false),
            row("sw", "dynamic", 0.0, false),
            row("sr", "dynamic", 1.0, false),
            row("abs", "static-abs", 1.0, true),
            row("abs2", "static-abs", 0.0, true),
        ];
        let verdicts: BTreeMap<String, Sufficiency> = [
            ("iw", Sufficiency::Insufficient),
            ("ir", Sufficiency::Insufficient),
            ("sw", Sufficiency::Sufficient),
            ("sr", Sufficiency::Sufficient),
            // A verdict for an abstention row must not be countable: the
            // audit never judges them, and a stale cache entry must not
            // smuggle one into the 2x2.
            ("abs", Sufficiency::Sufficient),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

        let t = tally(&rows, &verdicts);
        assert_eq!(t.cells, [[1, 1], [1, 1]], "one row per cell");
        assert_eq!(t.answerable, 4);
        assert_eq!(
            (t.abstention_total, t.abstention_correct),
            (2, 1),
            "abstention rows are counted for the ceiling, never celled"
        );
        assert_eq!(
            t.per_category.len(),
            2,
            "static-abs contributes no category row: {:?}",
            t.per_category
        );
        let statics = t
            .per_category
            .iter()
            .find(|c| c.category == "static")
            .expect("static category");
        assert_eq!(statics.cells, [[1, 1], [0, 0]]);

        // An answerable row with no verdict lands in no cell, so a --limit
        // probe's cells sum to the number judged, not the docket size.
        let t = tally(&rows, &BTreeMap::new());
        assert_eq!(t.cells, [[0, 0], [0, 0]]);
        assert_eq!(t.answerable, 4);
    }

    #[test]
    fn gold_token_recall_ignores_case_and_punctuation() {
        let gold = r#"["Actions, Change Status"]"#;
        assert_eq!(
            gold_token_recall(gold, "click actions then Change Status, twice"),
            1.0
        );
        assert_eq!(gold_token_recall(gold, "nothing relevant here"), 0.0);
        // Multiset, not set: one occurrence cannot satisfy two gold tokens.
        assert_eq!(gold_token_recall("status status", "status"), 0.5);
    }

    #[test]
    fn a_bench_run_directory_is_rejected() {
        let bench_row = serde_json::json!({
            "question_id": "q1",
            "tenant": "t",
            "category": 2,
            "question_text": "when?",
            "answer_gold": "yesterday",
            "response_raw": "yesterday",
            "score": 1.0,
            "exact_match": 1.0,
            "is_abstention_problem": false,
            "retrieved_items": 6,
            "memory_query_duration_seconds": 1.0
        })
        .to_string();
        let err = parse_rows(&bench_row, Path::new("runs/x/per_question.jsonl"))
            .expect_err("a bench row must not deserialise as a harness row")
            .to_string();
        assert!(err.contains("per_question.jsonl"), "{err}");
        assert!(err.contains("vendored LongMemEval-V2 harness run"), "{err}");
    }

    #[test]
    fn evidence_joins_text_items_and_counts_the_rest() {
        let mut r = row("q", "static", 1.0, false);
        r.memory_context = vec![
            ContextItem {
                kind: "text".to_string(),
                value: "one".to_string(),
            },
            ContextItem {
                kind: "image_url".to_string(),
                value: "data:...".to_string(),
            },
            ContextItem {
                kind: "text".to_string(),
                value: "two".to_string(),
            },
        ];
        assert_eq!(r.evidence(), "one\n\ntwo");
        assert_eq!(r.dropped_items(), 1);
    }

    /// The two ceilings answer different questions and must not be confused:
    /// one credits a perfect reader over today's evidence, the other credits
    /// perfect retrieval read at the measured rate. The verdict in
    /// `m16-evidence-sufficiency.md` compares both against 51.0.
    #[test]
    fn the_two_ceilings_credit_opposite_sides_of_the_pipeline() {
        let report = AuditReport {
            run: "runs/x".to_string(),
            questions: 100,
            answerable: 60,
            // 20 insufficient+wrong, 5 insufficient+right, 15 sufficient+wrong,
            // 20 sufficient+right.
            cells: [[20, 5], [15, 20]],
            per_category: Vec::new(),
            proxy: ProxyAgreement::default(),
            examples: Vec::new(),
            abstention_correct: 10,
            abstention_total: 40,
            judged: 60,
            cached: 0,
            context_items_dropped: 0,
            questions_with_image: 0,
        };
        // (25 right + 10 abstention) / 100.
        assert!((report.measured() - 0.35).abs() < 1e-9);
        // + the 15 sufficient-but-wrong rows a perfect reader would get.
        assert!(
            (report.reader_fix_ceiling() - 0.50).abs() < 1e-9,
            "{}",
            report.reader_fix_ceiling()
        );
        // + the 20 insufficient-wrong rows, read at 20/35 = 57.1%.
        assert!(
            (report.retrieval_fix_ceiling() - (35.0 + 20.0 * 20.0 / 35.0) / 100.0).abs() < 1e-9,
            "{}",
            report.retrieval_fix_ceiling()
        );
        assert!((report.retrieval_fix_ceiling_perfect() - 0.55).abs() < 1e-9);
        // S = 15 / 35.
        assert!((report.s() - 15.0 / 35.0).abs() < 1e-9);
        assert!((report.p_correct_given_sufficient() - 20.0 / 35.0).abs() < 1e-9);
        assert!((report.p_correct_given_insufficient() - 5.0 / 25.0).abs() < 1e-9);
        // Worst case: the judge called 5 of 25 correct rows insufficient, so
        // 20% of the 20 insufficient-wrong rows move: (15 + 4) / 35.
        assert!(
            (report.s_worst_case() - 19.0 / 35.0).abs() < 1e-9,
            "{}",
            report.s_worst_case()
        );
    }
}
