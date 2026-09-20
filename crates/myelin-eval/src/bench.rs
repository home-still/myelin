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
//! # Two deterministic scorers, both on every row
//!
//! [`Scorer::TokenF1`] is SQuAD-style normalised token F1, the same family
//! LoCoMo's own evaluation uses, and the metric M3..M13 reported.
//! Normalisation is spelled out in [`normalize`] rather than described,
//! because every reimplementation of "SQuAD normalisation" differs slightly
//! and the difference moves the number by a point or two.
//!
//! [`Scorer::Temporal`] resolves a temporal gold answer to a closed interval
//! of days and scores containment ([`crate::temporal`]); it keeps token F1
//! wherever the gold answer is not temporal. It is the **LoCoMo default**
//! since M14, which measured it agreeing with a reader-only judge 96.7% of
//! the time against token F1's 84.9% on the 272-item temporal stratum, and
//! found token F1 had been awarding 0.50–0.75 to answers naming the anchor
//! date instead of the offset asked for
//! (`docs/measurements/m14-temporal-scorer.md`). LongMemEval_S keeps token
//! F1: the grammar resolves only 26 of its 470 answerable golds.
//!
//! **Both columns are written on every row, forever** —
//! `ScoredQuestion::{score_token_f1, score_temporal}` — because a metric
//! change that erases the old metric makes every historical number
//! unreadable. `ScoredQuestion::score` carries whichever one the run's
//! `Scorer` selected, and [`rescore_run`] re-derives both from the persisted
//! text without a GPU.
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
use myelin_core::llm::openai::OpenAiLlm;
use myelin_core::llm::{CompletionRequest, Llm, Message};
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::pipeline::investigate::Investigator;
use myelin_core::pipeline::retrieve::{RetrieveConfig, Retriever};
use myelin_core::rerank::cross::CrossEncoder;
use myelin_core::rerank::Reranker;
use myelin_core::store::graph::GraphIndex;
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::QdrantStore;
use serde::{Deserialize, Serialize};

use crate::build::parse_session_time;
use crate::datasets::{locomo, longmemeval};
use crate::temporal;

/// Instruction given to the reader for every question.
///
/// The abstention clause is explicit because
/// `docs/measurements/m5-reference-baselines.md` measured this reader
/// fabricating answers on 97.2% of unanswerable questions when given *no
/// evidence at all*. Leaving abstention implicit measures the prompt's
/// omission rather than the memory system.
///
/// The date clause was M19's arm B, the control for
/// `ComposeConfig::resolve_relative`: the reader is *already* shown
/// `[YYYY-MM-DD]` on every item — `stamp_valid_time` has defaulted on since
/// M13 — and still answered "Last Tuesday" to 103 of LoCoMo's 321 temporal
/// questions. Naming the operation in the prompt is worth a paired
/// **+14.3 points (95% CI [+10.2, +18.7])** on that stratum alone, and
/// **+5.2 ([+2.5, +8.1])** on top of the resolved annotation, so it ships as
/// part of the prompt rather than as a switch. Both mechanisms are needed:
/// the annotation is worth +28.4 ([+23.4, +33.7]) on top of the clause.
/// `docs/measurements/m19-temporal-resolution.md`.
const READER_SYSTEM: &str = "You answer questions using only the supplied memories. \
Answer in as few words as possible — a name, a date, a short phrase. \
Do not explain. Do not restate the question. \
If the memories do not contain the answer, reply exactly: I don't know. \
Each memory is prefixed in brackets with the date it was recorded. \
When the question asks when something happened, resolve relative expressions such as \"last Tuesday\" or \
\"two weeks ago\" against that bracketed date and answer with an absolute date.";

/// M20 arm B, the control for `ComposeConfig::profile`.
///
/// Appended to [`READER_SYSTEM`] rather than folded into it, because M19's
/// precedent is that a reader clause is measured as its own arm before it
/// ships: the compose-side annotation was +37.6 and the reader-side clause
/// +14.3 on the same stratum, with both marginals significant, so neither
/// arm may be assumed to subsume the other.
///
/// It contradicts two lines of `READER_SYSTEM` on purpose. "Answer in as few
/// words as possible" and "If the memories do not contain the answer, reply
/// exactly: I don't know" are exactly what a preference question triggers —
/// there is no literal answer to *"suggest some accessories"* in any memory,
/// only dispositions from which one is stated.
const READER_PREFERENCE_CLAUSE: &str = " \
When the question asks for a recommendation or a suggestion, or asks what the user \
would like, answer with the preferences the user themselves stated in the memories — \
name the brands, topics and constraints they stated — rather than generic options. \
A memory beginning [profile] states what the user is known to prefer; treat it as \
established about the user. Do not reply I don't know when the memories state a \
relevant preference.";

/// One scored question, written to `per_question.jsonl`.
///
/// Field names match what `adapters/paired_ci.py` reads (`question_id`,
/// `score`, `is_abstention_problem`) so LoCoMo runs and LongMemEval-V2 runs
/// go through one confidence-interval tool instead of two.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredQuestion {
    pub question_id: String,
    pub tenant: String,
    pub category: u8,
    pub question_text: String,
    pub answer_gold: String,
    pub response_raw: String,
    pub score: f64,
    pub exact_match: f64,
    /// Both scorers, on every row, so a run is comparable under either
    /// without being re-read. A metric change that erased the old metric
    /// would make every historical number unreadable.
    #[serde(default)]
    pub score_token_f1: f64,
    #[serde(default)]
    pub score_temporal: f64,
    /// Which grammar matched the gold answer: `interval`, `duration`, or
    /// `none`. `none` means `score_temporal == score_token_f1` by definition.
    #[serde(default)]
    pub temporal_kind: String,
    pub is_abstention_problem: bool,
    pub retrieved_items: usize,
    /// What the reader was actually shown: every
    /// [`myelin_core::model::EvidenceItem::value`] in emitted order, labels
    /// and date stamps included.
    ///
    /// Always on, and not a flag. Nothing on disk recorded this before M19,
    /// so every claim of the form "retrieval found the record and the reader
    /// failed to use it" was an inference; with it, `V2` in
    /// `docs/measurements/m19-temporal-resolution.md` is an arithmetic check
    /// against `data/locomo10.json`. `#[serde(default)]` so the 28 historical
    /// run artifacts `standing` reads still parse.
    #[serde(default)]
    pub evidence: Vec<String>,
    pub memory_query_duration_seconds: f64,
}

/// Every run written before M23 composed to this budget; a run artifact
/// without the key is declaring it was one of those.
fn default_budget_tokens() -> usize {
    4096
}

/// Aggregate over one bench run.
///
/// `Deserialize` as well as `Serialize`: `myelin-eval standing` reads these
/// artifacts back off disk, and the five switch fields carry
/// `#[serde(default)]` because runs written before M13 genuinely lack those
/// keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRun {
    pub corpus: String,
    pub collection: String,
    pub mode: String,
    pub k: usize,
    pub max_steps: usize,
    /// Which mechanisms produced this run. A run artifact that does not
    /// record that is not reproducible.
    #[serde(default)]
    pub graph: bool,
    #[serde(default)]
    pub chronological: bool,
    #[serde(default)]
    pub question_date: bool,
    /// M19's mechanisms, mirroring `ComposeConfig::resolve_relative` and
    /// `ComposeConfig::timeline`. **Both ship on**, and both are recorded on
    /// every run so a future flip is visible in the artifact rather than only
    /// in git. Arm B is not here: the date clause became part of
    /// `READER_SYSTEM`, so every run after M19 carries it and a per-run field
    /// would only ever say `true`.
    #[serde(default)]
    pub resolve_dates: bool,
    #[serde(default)]
    pub timeline: bool,
    /// M20's two arms, mirroring `ComposeConfig::profile` and
    /// `READER_PREFERENCE_CLAUSE`. Both are recorded per run, because unlike
    /// M19's date clause neither has shipped into the prompt: an arm that
    /// did not write down which of the two it carried is unreadable.
    #[serde(default)]
    pub profile: bool,
    #[serde(default)]
    pub profile_clause: bool,
    /// M21's two arms, mirroring `ComposeConfig::mmr_lambda` and
    /// `RetrieveConfig::select_sufficient`. Recorded per run for the reason
    /// M20's pair is: neither has shipped into a default, so the artifact is
    /// the only record of which mechanism produced its rows — and
    /// `rescore_run` reads them back so a rescored artifact does not forget.
    #[serde(default)]
    pub mmr: Option<f32>,
    #[serde(default)]
    pub select_sufficient: bool,
    /// M23's width/budget triple, mirroring `Budget::tokens` and
    /// `RetrieveConfig::{prefetch_limit, rerank_depth}`. Recorded for the
    /// reason M21's pair is: `standing` fingerprints LME-V2 harness runs on
    /// exactly these keys, and a bench run that does not name them is
    /// unreadable next to one that does. `None`/4096 on every run written
    /// before M23.
    #[serde(default = "default_budget_tokens")]
    pub budget_tokens: usize,
    #[serde(default)]
    pub prefetch_limit: Option<u64>,
    #[serde(default)]
    pub rerank_depth: Option<usize>,
    /// M23's read-path arms, mirroring `InvestigateConfig::rerank_pool`,
    /// `::premise_analysis`, `::typed_probes` and
    /// `ComposeConfig::untrusted_max`. All four ship off, so — like M20's
    /// and M21's pairs — the artifact is the only record of which one
    /// produced these rows. Absent on every run written before M23.
    #[serde(default)]
    pub rerank_pool: bool,
    #[serde(default)]
    pub premise: bool,
    #[serde(default)]
    pub typed_probes: bool,
    #[serde(default)]
    pub untrusted_max: Option<usize>,
    /// M24's sub-query decomposition cap, mirroring
    /// `RetrieveConfig::decompose`. Ships off; absent on every run before
    /// M24.
    #[serde(default)]
    pub decompose: Option<usize>,
    /// Which category codes were scored. Empty means every one of them,
    /// which is what every run before M19 did.
    #[serde(default)]
    pub categories: Vec<u8>,
    /// Which column `score` carries, and where the row scores came from.
    /// `rescored_from` is `None` for a live bench run. Empty on a pre-M14
    /// artifact, which is token F1 by definition.
    #[serde(default)]
    pub scorer: String,
    #[serde(default)]
    pub rescored_from: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryScore {
    pub category: u8,
    pub count: usize,
    pub mean_score: f64,
}

/// The switch set for one bench run.
///
/// A struct and not seven more positional parameters: [`bench_locomo`]
/// already carried twelve, and [`RunSpec`] exists in this file for exactly
/// this reason. `Default` is the all-off arm, which is
/// `RetrieveConfig::default()` field for field and therefore the path the M9
/// and M12 baselines exercised.
#[derive(Debug, Clone, Default)]
pub struct BenchSwitches {
    /// Fuse the PPR channel over the phrase↔record graph (M12).
    pub graph: bool,
    /// Emit evidence oldest-first instead of `bookend`'s interleave (M13).
    pub chronological: bool,
    /// Give LoCoMo's reader a `<today>` reference date (M13).
    pub question_date: bool,
    /// Compose the `[profile]` block — M20 arm A, `ComposeConfig::profile`.
    pub profile: bool,
    /// Append [`READER_PREFERENCE_CLAUSE`] to the reader prompt — M20 arm B.
    ///
    /// Two independent switches because M20 measures A and B alone and
    /// together; one combined flag cannot produce the marginals.
    pub profile_clause: bool,
    /// Select the evidence for joint coverage instead of independent rank —
    /// M21 arm A, `ComposeConfig::mmr_lambda`.
    pub mmr: Option<f32>,
    /// Ask the model which candidates jointly answer the question — M21 arm
    /// B, `RetrieveConfig::select_sufficient`. A ceiling probe: `PLAN.md`
    /// §7.1 forbids an LLM in the `recall` loop, so this can never become a
    /// `recall` default whatever it measures.
    pub select_sufficient: bool,
    /// Rerank the accumulated pool against the original question once,
    /// after the last step — M23 A2, `InvestigateConfig::rerank_pool`.
    /// `investigate`-only and inert without a reranker.
    pub rerank_pool: bool,
    /// Replace the bare insufficiency statement with a premise analysis —
    /// M23 A3, `InvestigateConfig::premise_analysis`.
    ///
    /// Implies `abstain_on_insufficient`, exactly as the MCP server does:
    /// the analysis rewrites the statement the gate emits, so the switch
    /// alone is inert and an inert switch is the failure mode M12, M14 and
    /// M20 each lost a run to.
    pub premise: bool,
    /// Split the question into sub-queries and retrieve for each — M24,
    /// `RetrieveConfig::decompose`. The value is the cap on sub-queries.
    ///
    /// On `investigate` this decomposes *every* probe, so the cost is one
    /// model call per step rather than one per query. That is the honest
    /// composition of the two mechanisms and it is left alone: special-
    /// casing the first step would make the arm measure something the MCP
    /// path does not do.
    pub decompose: Option<usize>,
    /// Let the reflect gate aim its next probe at a record kind — M23 D2,
    /// `InvestigateConfig::typed_probes`. Meaningless until a store carries
    /// the typed pools `build --pools` mints.
    pub typed_probes: bool,
    /// Cap untrusted occupancy in the composed set — M23 B1,
    /// `ComposeConfig::untrusted_max`.
    ///
    /// Reachable from `bench` and not only from `attack --live` because a
    /// defence measured for attack success and never for utility is half a
    /// measurement: M15 paid for its adjudicator's false-positive column
    /// over 550 real episodes, and a quota that quietly drops real evidence
    /// on LoCoMo would be invisible from the attack harness alone.
    pub untrusted_max: Option<usize>,
    /// Score only these category codes. Empty means all of them.
    ///
    /// A stratum arm has to be runnable over 321 or 127 questions rather than
    /// 1,986 or 500, or M19 does not fit in one GPU window. Filtering happens
    /// **before** retrieval, so a skipped question costs nothing.
    pub categories: Vec<u8>,
}

impl BenchSwitches {
    /// Is this question in the scored stratum?
    fn wants(&self, category: u8) -> bool {
        self.categories.is_empty() || self.categories.contains(&category)
    }
}

/// Which column [`ScoredQuestion::score`] carries.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
pub enum Scorer {
    /// SQuAD-style token F1 — the M3..M13 metric.
    #[value(name = "token-f1")]
    TokenF1,
    /// Date-aware: [`temporal::temporal_score`] where the gold answer names a
    /// time, token F1 everywhere else.
    Temporal,
    /// Read verdicts written by `myelin-eval judge` off disk. Rescore-only:
    /// a live `bench` has no verdicts yet.
    ///
    /// It exists because the deterministic scorers are both wrong for
    /// LongMemEval_S's temporal stratum: only 4 of its 127 golds parse as
    /// durations (the order questions' golds are event names), and token F1
    /// credits "Three weeks" against "Two weeks" at 0.5 — the exact failure
    /// M14 existed to remove. `adapters/paired_ci.py` pairs on the `score`
    /// field, so a judged comparison needs the verdicts *in* that field.
    #[value(name = "judge")]
    Judge,
}

impl Scorer {
    /// Written into `BenchRun::scorer` and used as a run-directory suffix, so
    /// a directory name stays a function of the switch set.
    pub fn slug(self) -> &'static str {
        match self {
            Scorer::TokenF1 => "token_f1",
            Scorer::Temporal => "temporal",
            Scorer::Judge => "judge",
        }
    }
}

/// Both scorers' verdicts on one answered question.
///
/// A struct and not a tuple: four of the five fields are `f64` and a
/// positional return would let a caller swap `token_f1` for `temporal`
/// silently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scores {
    /// The reported column, per the run's [`Scorer`].
    pub score: f64,
    pub exact: f64,
    pub token_f1: f64,
    pub temporal: f64,
    pub kind: &'static str,
}

/// Score one answered question under both scorers.
///
/// The abstention rules run first and are shared: an adversarial item scores
/// 1.0 iff the reader declined, and a decline on an answerable item earns
/// nothing — never token overlap with a gold answer that happens to contain
/// "know". Both corpora and `rescore` go through here so the rule cannot
/// drift between them.
pub fn score_one(response: &str, gold: &str, adversarial: bool, scorer: Scorer) -> Scores {
    let declined = is_abstention(response);
    if adversarial {
        // Both columns carry the same value, so no consumer of
        // `score_temporal` sees a surprise on the adversarial stratum.
        let s = f64::from(u8::from(declined));
        return Scores {
            score: s,
            exact: s,
            token_f1: s,
            temporal: s,
            kind: "none",
        };
    }
    if declined {
        return Scores {
            score: 0.0,
            exact: 0.0,
            token_f1: 0.0,
            temporal: 0.0,
            kind: "none",
        };
    }
    let f1 = token_f1(response, gold);
    let em_f1 = f64::from(u8::from(normalize(response) == normalize(gold)));
    let (temporal, kind) = match temporal::temporal_score(response, gold) {
        None => (f1, "none"),
        Some((t, k)) => (t, k.as_str()),
    };
    let (score, exact) = match scorer {
        Scorer::TokenF1 => (f1, em_f1),
        // A temporal item's exact match is "named the right time", which is
        // what `score == 1.0` already means.
        Scorer::Temporal => (
            temporal,
            if kind == "none" {
                em_f1
            } else {
                f64::from(u8::from(temporal == 1.0))
            },
        ),
        // `judge` names no deterministic column: [`rescore_run`] fills it
        // from `<run>/judge_verdicts.json` and `bench_cmd` refuses the
        // scorer before any GPU work. Reaching here means a deterministic
        // function was asked for a judged score, so it reports the
        // deterministic one it has rather than inventing a verdict.
        Scorer::Judge => (f1, em_f1),
    };
    Scores {
        score,
        exact,
        token_f1: f1,
        temporal,
        kind,
    }
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

/// Flatten LoCoMo's `answer` field — or an LME-V2 harness row's
/// `answer_gold` — to a string.
///
/// It is `Option<Value>` because category-5 items may omit it and some items
/// carry a number rather than a string; `Value::to_string` would wrap strings
/// in quotes and poison the token overlap.
pub(crate) fn gold_answer(value: Option<&serde_json::Value>) -> String {
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

/// Appends one JSON line to `per_question.jsonl` per scored question, and
/// keeps a copy for the run-level aggregation.
///
/// The rows used to live only in a `Vec` until `finish_run`, which meant any
/// error before that point destroyed the whole run: M17 lost 54 minutes of
/// generations to an HTTP 400 raised in the *scoring* stage, after every
/// answer had already been produced. The file is opened when the run
/// directory is created and flushed after every row, so a run that dies —
/// or is interrupted — keeps every question it finished.
struct RowSink {
    file: std::io::BufWriter<std::fs::File>,
    rows: Vec<ScoredQuestion>,
}

impl RowSink {
    /// Creates (and truncates) `per_question.jsonl` under `out_dir`.
    fn create(out_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("create {}", out_dir.display()))?;
        let path = out_dir.join("per_question.jsonl");
        let file =
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        Ok(Self {
            file: std::io::BufWriter::new(file),
            rows: Vec::new(),
        })
    }

    /// Writes the row, flushes it, then keeps it. Flushing per row is the
    /// whole point: a buffered line that never reaches the disk is exactly
    /// the loss this type exists to prevent, and one `write` per reader call
    /// is free next to the call itself.
    fn push(&mut self, row: ScoredQuestion) -> Result<()> {
        use std::io::Write;
        serde_json::to_writer(&mut self.file, &row)?;
        self.file.write_all(b"\n")?;
        self.file.flush().context("flush per_question.jsonl")?;
        self.rows.push(row);
        Ok(())
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    fn into_rows(self) -> Vec<ScoredQuestion> {
        self.rows
    }
}

/// Run LoCoMo end-to-end and score every question.
#[allow(clippy::too_many_arguments)]
pub async fn bench_locomo(
    path: &Path,
    collection: &str,
    ledger_path: &Path,
    k: usize,
    budget_tokens: usize,
    prefetch_limit: Option<u64>,
    rerank_depth: Option<usize>,
    mode: Mode,
    max_steps: usize,
    limit: Option<usize>,
    switches: &BenchSwitches,
    scorer: Scorer,
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

    let graph_index = GraphIndex::new();
    // One config, always passed. With every switch off this is
    // `RetrieveConfig::default()` field for field, so the baseline path stays
    // the one the M9 and M12 runs exercised.
    //
    // The switch and the index are separate: `RetrieveConfig::graph` defaults
    // false, so wiring an index alone would be inert.
    //
    // `Investigator` wraps `&Retriever`, so `--mode investigate --graph`
    // composes with no extra wiring.
    let mut retriever = Retriever::new(&embedder, &store, &ledger).with_config(RetrieveConfig {
        graph: switches.graph,
        select_sufficient: switches.select_sufficient,
        decompose: switches.decompose,
        // `None` means the measured default — the same contract every other
        // override on this config uses, and what the width arms (M23) pass
        // values for.
        prefetch_limit: prefetch_limit.unwrap_or(RetrieveConfig::default().prefetch_limit),
        rerank_depth: rerank_depth.unwrap_or(RetrieveConfig::default().rerank_depth),
        // `resolve_relative` is NOT overridden: it ships on, and a bench run
        // that silently disabled the shipped mechanism because a flag
        // defaulted false would measure a configuration nobody runs. Same
        // treatment `stamp_valid_time` has had since M13.
        compose: myelin_core::pipeline::compose::ComposeConfig {
            chronological: switches.chronological,
            profile: switches.profile,
            mmr_lambda: switches.mmr,
            untrusted_max: switches.untrusted_max,
            ..Default::default()
        },
        ..Default::default()
    });
    if let Some(r) = reranker.as_ref() {
        retriever = retriever.with_reranker(r as &dyn Reranker);
    }
    if switches.graph {
        retriever = retriever.with_graph(&graph_index);
    }
    // Unconditional, like `with_reranker`: `RetrieveConfig::select_sufficient`
    // defaults false, so a wired client is inert until the switch is on — and
    // wiring it only under the switch is the exact class of failure M12, M14
    // and M20 each lost a run to.
    retriever = retriever.with_llm(&llm);

    // `InvestigateConfig::select_sufficient` ships **on**, but a bench that
    // read the shipped default could only ever produce one of the two arms —
    // and M21 has to measure the default it is about to set. So the loop's
    // switch follows `--select-sufficient` here, exactly as the `recall`
    // path's does, and `BenchSwitches::default()` stays the all-off arm.
    let investigate_cfg = myelin_core::pipeline::investigate::InvestigateConfig {
        select_sufficient: switches.select_sufficient,
        rerank_pool: switches.rerank_pool,
        premise_analysis: switches.premise,
        // The same implication the MCP server applies: the analysis
        // rewrites what the gate emits, so `--premise` without the gate
        // measures nothing.
        abstain_on_insufficient: switches.premise,
        typed_probes: switches.typed_probes,
        ..Default::default()
    };

    // Opened before the first reader call so an interrupted run keeps every
    // question it finished (see `RowSink`).
    let mut scored = RowSink::create(out_dir)?;
    let mut latencies: Vec<f64> = Vec::new();
    // Arm B rides on `READER_SYSTEM` rather than replacing it: the arm is the
    // clause, and swapping the whole prompt would confound it with the
    // abstention and date instructions every prior run carried.
    let system = if switches.profile_clause {
        format!("{READER_SYSTEM}{READER_PREFERENCE_CLAUSE}")
    } else {
        READER_SYSTEM.to_string()
    };
    let system = system.as_str();

    'outer: for conv in &conversations {
        let tenant = format!("locomo/{}", conv.sample_id);
        // LoCoMo has no per-question date; it asks from the position of the
        // end of the conversation, so the last session's date is the reader's
        // `<today>`. A conversation whose session dates all fail to parse
        // gets the unmodified two-block prompt: an invented date is worse
        // than none.
        //
        // Measured (M13, `docs/measurements/m13-temporal-axis.md`): off by
        // default. It reaches 335/1,986 answers and shifts category 2 from
        // relative to absolute dates as intended (38.9% → 44.2% of answers
        // carry a date), but buys +0.2 points there (95% CI [−1.8, +2.2]),
        // because only 22.7% of that stratum's gold answers are a plain
        // absolute date and a quarter are relative expressions *anchored* to
        // one (`The sunday before 25 May 2023`). What it does buy is
        // abstention: +3.1 points on the 446 adversarial items, CI
        // [+0.7, +5.6] — a reader that knows the date can tell the memories
        // do not cover the period asked about.
        let today = conv
            .sessions
            .iter()
            .filter_map(|s| s.date_time.as_deref().and_then(parse_session_time))
            .max();
        for (i, qa) in conv.qa.iter().enumerate() {
            // Before retrieval, so a stratum arm costs nothing for the
            // questions it skips. `question_id` keeps the *unfiltered* index
            // `i`, so `paired_ci.py` pairs a stratum run against a full run.
            if !switches.wants(qa.category) {
                continue;
            }
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
                    tokens: budget_tokens,
                    max_steps,
                },
                mode,
                kinds: None,
            };

            let started = std::time::Instant::now();
            let evidence = match mode {
                Mode::Investigate => {
                    Investigator::new(&llm, &retriever)
                        .with_config(investigate_cfg)
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
            let user = match today.filter(|_| switches.question_date) {
                // The same `<today>` tag and the same ISO format the
                // LongMemEval_S prompt already uses, so the two corpora do
                // not present the date two ways.
                Some(t) => format!(
                    "<memories>\n{context}\n</memories>\n<today>\n{}\n</today>\n<question>\n{}\n</question>",
                    t.format("%Y-%m-%d"),
                    qa.question
                ),
                None => format!(
                    "<memories>\n{context}\n</memories>\n<question>\n{}\n</question>",
                    qa.question
                ),
            };
            let response = llm
                .complete(
                    &CompletionRequest::new(vec![Message::system(system), Message::user(user)])
                        .with_max_tokens(160),
                )
                .await
                .with_context(|| format!("reader {tenant}#{i}"))?
                .text;

            let s = score_one(&response, &gold, adversarial, scorer);

            scored.push(ScoredQuestion {
                question_id: format!("{}#{i}", conv.sample_id),
                tenant: tenant.clone(),
                category: qa.category,
                question_text: qa.question.clone(),
                answer_gold: gold,
                response_raw: response,
                score: s.score,
                exact_match: s.exact,
                score_token_f1: s.token_f1,
                score_temporal: s.temporal,
                temporal_kind: s.kind.to_string(),
                is_abstention_problem: adversarial,
                retrieved_items: evidence.items.len(),
                evidence: evidence.items.iter().map(|i| i.value.clone()).collect(),
                memory_query_duration_seconds: elapsed,
            })?;
        }
    }

    finish_run(
        &RunSpec {
            corpus: "locomo".into(),
            collection: collection.to_string(),
            mode,
            k,
            max_steps,
            budget_tokens,
            prefetch_limit,
            rerank_depth,
            switches: switches.clone(),
            scorer,
            rescored_from: None,
        },
        scored.into_rows(),
        latencies,
        out_dir,
    )
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
    budget_tokens: usize,
    prefetch_limit: Option<u64>,
    rerank_depth: Option<usize>,
    mode: Mode,
    max_steps: usize,
    limit: Option<usize>,
    switches: &BenchSwitches,
    scorer: Scorer,
    out_dir: &Path,
) -> Result<BenchRun> {
    let cfg = MyelinConfig::load().context("load myelin config")?;
    let mut items = longmemeval::load(dataset).context("load longmemeval_s")?;
    // Stratum before `--limit`: truncating the 500 to N and *then* filtering
    // would leave a handful of rows for a 127-question stratum.
    items.retain(|it| switches.wants(question_type_code(&it.question_type)));
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

    let graph_index = GraphIndex::new();
    // See `bench_locomo`: one config, always passed, so the all-off arm is
    // `RetrieveConfig::default()` field for field. The width pair follows
    // `None`-means-default, the contract every other override here uses.
    let mut retriever = Retriever::new(&embedder, &store, &ledger).with_config(RetrieveConfig {
        graph: switches.graph,
        select_sufficient: switches.select_sufficient,
        decompose: switches.decompose,
        prefetch_limit: prefetch_limit.unwrap_or(RetrieveConfig::default().prefetch_limit),
        rerank_depth: rerank_depth.unwrap_or(RetrieveConfig::default().rerank_depth),
        compose: myelin_core::pipeline::compose::ComposeConfig {
            chronological: switches.chronological,
            profile: switches.profile,
            mmr_lambda: switches.mmr,
            untrusted_max: switches.untrusted_max,
            ..Default::default()
        },
        ..Default::default()
    });
    if let Some(r) = reranker.as_ref() {
        retriever = retriever.with_reranker(r as &dyn Reranker);
    }
    if switches.graph {
        retriever = retriever.with_graph(&graph_index);
    }
    // See `bench_locomo`: wired unconditionally so the switch cannot be inert.
    retriever = retriever.with_llm(&llm);

    // See `bench_locomo`: switch-driven so both investigate arms exist.
    let investigate_cfg = myelin_core::pipeline::investigate::InvestigateConfig {
        select_sufficient: switches.select_sufficient,
        rerank_pool: switches.rerank_pool,
        premise_analysis: switches.premise,
        abstain_on_insufficient: switches.premise,
        typed_probes: switches.typed_probes,
        ..Default::default()
    };

    // Opened before the first reader call so an interrupted run keeps every
    // question it finished (see `RowSink`).
    let mut scored = RowSink::create(out_dir)?;
    let mut latencies: Vec<f64> = Vec::new();
    // See `bench_locomo`: the clause is appended, not substituted.
    let system = if switches.profile_clause {
        format!("{READER_SYSTEM}{READER_PREFERENCE_CLAUSE}")
    } else {
        READER_SYSTEM.to_string()
    };
    let system = system.as_str();

    for item in &items {
        let adversarial = item.is_abstention();
        let gold = item.answer_text();
        let query = Recall {
            scope: ScopeFilter::tenant(format!("lme_s/{}", item.question_id))
                .with_namespace("longmemeval_s"),
            text: item.question.clone(),
            budget: Budget {
                k,
                tokens: budget_tokens,
                max_steps,
            },
            mode,
            kinds: None,
        };

        let started = std::time::Instant::now();
        let evidence = match mode {
            Mode::Investigate => {
                Investigator::new(&llm, &retriever)
                    .with_config(investigate_cfg)
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
                    Message::system(system),
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

        let s = score_one(&response, &gold, adversarial, scorer);

        scored.push(ScoredQuestion {
            question_id: item.question_id.clone(),
            tenant: format!("lme_s/{}", item.question_id),
            category: question_type_code(&item.question_type),
            question_text: item.question.clone(),
            answer_gold: gold,
            response_raw: response,
            score: s.score,
            exact_match: s.exact,
            score_token_f1: s.token_f1,
            score_temporal: s.temporal,
            temporal_kind: s.kind.to_string(),
            is_abstention_problem: adversarial,
            retrieved_items: evidence.items.len(),
            evidence: evidence.items.iter().map(|i| i.value.clone()).collect(),
            memory_query_duration_seconds: elapsed,
        })?;
    }

    finish_run(
        &RunSpec {
            corpus: "longmemeval_s".into(),
            collection: collection.to_string(),
            mode,
            k,
            max_steps,
            budget_tokens,
            prefetch_limit,
            rerank_depth,
            switches: switches.clone(),
            scorer,
            rescored_from: None,
        },
        scored.into_rows(),
        latencies,
        out_dir,
    )
}

/// LongMemEval names its question types; `ScoredQuestion::category` is numeric
/// so both corpora share one row shape and one CI tool.
pub(crate) fn question_type_code(t: &str) -> u8 {
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

/// Everything a run artifact must name about how it was produced.
///
/// A struct and not eleven positional parameters: a run artifact that does not
/// record its own provenance is not reproducible, and the list only grows.
pub struct RunSpec {
    pub corpus: String,
    pub collection: String,
    pub mode: Mode,
    pub k: usize,
    pub max_steps: usize,
    /// The composed-evidence budget the loop paid for: `Budget::tokens`.
    /// 4096 unless a width arm widened it.
    pub budget_tokens: usize,
    /// Candidate-pool width the reranker was fed from. `None` means
    /// `RetrieveConfig::default()`'s 50.
    pub prefetch_limit: Option<u64>,
    /// Reranked-pool depth. `None` means `RetrieveConfig::default()`'s 25.
    pub rerank_depth: Option<usize>,
    /// Which mechanisms were on. Carried whole rather than field by field:
    /// this list has grown at every milestone since M12.
    pub switches: BenchSwitches,
    pub scorer: Scorer,
    pub rescored_from: Option<String>,
}

/// Aggregate, print nothing, write `per_question.jsonl` and
/// `aggregated_metrics.json`. Shared by both corpora so a metric fixed for one
/// is fixed for both, and so `adapters/paired_ci.py` reads one row shape.
fn finish_run(
    spec: &RunSpec,
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
        corpus: spec.corpus.clone(),
        collection: spec.collection.clone(),
        mode: match spec.mode {
            Mode::Investigate => "investigate".into(),
            Mode::Recall => "recall".into(),
        },
        k: spec.k,
        max_steps: spec.max_steps,
        budget_tokens: spec.budget_tokens,
        prefetch_limit: spec.prefetch_limit,
        rerank_depth: spec.rerank_depth,
        graph: spec.switches.graph,
        chronological: spec.switches.chronological,
        question_date: spec.switches.question_date,
        // Read off the shipped defaults rather than switches: `bench` no
        // longer overrides either, so this is what the run actually used.
        resolve_dates: myelin_core::pipeline::compose::ComposeConfig::default().resolve_relative,
        timeline: myelin_core::pipeline::compose::ComposeConfig::default().timeline,
        // Read off the switches, not the defaults: neither M20 arm has
        // shipped into a default, so the run artifact is the only record of
        // which one produced it.
        profile: spec.switches.profile,
        profile_clause: spec.switches.profile_clause,
        // Same rule for M21's pair: both default off, so only the artifact
        // says which produced these rows.
        mmr: spec.switches.mmr,
        select_sufficient: spec.switches.select_sufficient,
        // M23's four, same rule again: all ship off, so the artifact is the
        // only record of which produced these rows — and `standing` reads
        // them back to decide whether the run is an arm.
        rerank_pool: spec.switches.rerank_pool,
        premise: spec.switches.premise,
        typed_probes: spec.switches.typed_probes,
        untrusted_max: spec.switches.untrusted_max,
        decompose: spec.switches.decompose,
        categories: spec.switches.categories.clone(),
        scorer: spec.scorer.slug().to_string(),
        rescored_from: spec.rescored_from.clone(),
        questions: scored.len(),
        f1_answerable: mean(&answerable, |s| s.score),
        em_answerable: mean(&answerable, |s| s.exact_match),
        abstention_accuracy: mean(&adversarial, |s| s.score),
        by_category,
        query_p50_seconds: percentile(&sorted, 0.50),
        query_avg_seconds: avg,
    };

    // `per_question.jsonl` is already on disk: `RowSink` wrote and flushed
    // each row as it was scored. Only the aggregate is written here, so an
    // error anywhere above still leaves every finished question.
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    std::fs::write(
        out_dir.join("aggregated_metrics.json"),
        serde_json::to_string_pretty(&run)?,
    )
    .context("write aggregated_metrics.json")?;

    Ok(run)
}

/// Recompute scores for a finished bench run from its own rows.
///
/// `response_raw` and `answer_gold` are persisted, so nothing has to be
/// re-generated: a scorer change can be applied to every historical run
/// without a GPU or a reader call. The old `score` is **not** carried
/// forward — recomputing both columns from the raw text is what makes an old
/// run and a new run comparable, and it also re-verifies that token F1
/// reproduces the historical value.
pub fn rescore_run(source: &Path, out_dir: &Path, scorer: Scorer) -> Result<BenchRun> {
    anyhow::ensure!(
        !out_dir.starts_with(source),
        "refusing to write into the source run {}; rescoring must not overwrite the artifact it reads",
        source.display()
    );
    let rows_path = source.join("per_question.jsonl");
    let text = std::fs::read_to_string(&rows_path)
        .with_context(|| format!("read {}", rows_path.display()))?;

    // `--scorer judge` reports what the judge said, so the verdicts have to
    // be on disk before a single row is scored: failing half way through
    // would leave a directory whose `scorer: "judge"` is a claim nothing
    // backs.
    let verdicts = match scorer {
        Scorer::Judge => Some(read_judge_file(source)?),
        _ => None,
    };

    // Truncates and rewrites rather than appending: every row is re-derived
    // from the source run, so a partial file from an earlier attempt must not
    // survive under the new rows.
    let mut scored = RowSink::create(out_dir)?;
    let mut latencies: Vec<f64> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        // `runs/` also holds directories written by the vendored LME-V2
        // harness whose rows share this file name but not this schema. Say so,
        // rather than surfacing a raw serde message about a missing field.
        let mut row: ScoredQuestion = serde_json::from_str(line).with_context(|| {
            format!(
                "{} is not a `bench` run directory (its per_question.jsonl has no \
                 `exact_match`/`tenant`); the vendored LME-V2 harness writes a \
                 different row shape",
                source.display()
            )
        })?;
        // The deterministic columns are recomputed under token F1 whatever
        // the reported scorer is, so `score_token_f1` and `score_temporal`
        // stay readable on a judged artifact too.
        let s = score_one(
            &row.response_raw,
            &row.answer_gold,
            row.is_abstention_problem,
            if scorer == Scorer::Judge {
                Scorer::TokenF1
            } else {
                scorer
            },
        );
        row.score = match &verdicts {
            Some(judge) => judged_score(judge, &row)?,
            None => s.score,
        };
        row.exact_match = match &verdicts {
            // A judged verdict is already 0/1: "exactly right" and "right"
            // are the same claim, so reporting a separate exact match would
            // be a second, unbacked number.
            Some(_) => row.score,
            None => s.exact,
        };
        row.score_token_f1 = s.token_f1;
        row.score_temporal = s.temporal;
        row.temporal_kind = s.kind.to_string();
        latencies.push(row.memory_query_duration_seconds);
        scored.push(row)?;
    }

    let metrics_path = source.join("aggregated_metrics.json");
    let metrics: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&metrics_path)
            .with_context(|| format!("read {}", metrics_path.display()))?,
    )
    .with_context(|| format!("parse {}", metrics_path.display()))?;
    // Absent or null reads as `false`/`0`, which is how the pre-M12 runs read:
    // they predate the switches and were produced with all of them off.
    let flag = |key: &str| metrics.get(key).and_then(serde_json::Value::as_bool) == Some(true);
    let count = |key: &str| {
        metrics
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0)
    };
    let spec = RunSpec {
        corpus: metrics
            .get("corpus")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
        collection: metrics
            .get("collection")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
        mode: match metrics.get("mode").and_then(serde_json::Value::as_str) {
            Some("investigate") => Mode::Investigate,
            _ => Mode::Recall,
        },
        k: count("k"),
        max_steps: count("max_steps"),
        // 4096, not 0, on a pre-M23 artifact: `count` defaults missing keys
        // to zero, and a zero budget is a configuration that never ran.
        budget_tokens: metrics
            .get("budget_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(default_budget_tokens()),
        prefetch_limit: metrics
            .get("prefetch_limit")
            .and_then(serde_json::Value::as_u64),
        rerank_depth: metrics
            .get("rerank_depth")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok()),
        // Read back rather than defaulted: a rescored artifact that forgot
        // which mechanisms produced its rows would break every later
        // comparison against the run it came from.
        switches: BenchSwitches {
            graph: flag("graph"),
            chronological: flag("chronological"),
            question_date: flag("question_date"),
            profile: flag("profile"),
            profile_clause: flag("profile_clause"),
            mmr: metrics
                .get("mmr")
                .and_then(serde_json::Value::as_f64)
                .map(|v| v as f32),
            select_sufficient: flag("select_sufficient"),
            rerank_pool: flag("rerank_pool"),
            premise: flag("premise"),
            typed_probes: flag("typed_probes"),
            untrusted_max: metrics
                .get("untrusted_max")
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| usize::try_from(n).ok()),
            decompose: metrics
                .get("decompose")
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| usize::try_from(n).ok()),
            categories: metrics
                .get("categories")
                .and_then(serde_json::Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(serde_json::Value::as_u64)
                        .filter_map(|n| u8::try_from(n).ok())
                        .collect()
                })
                .unwrap_or_default(),
        },
        scorer,
        rescored_from: Some(source.display().to_string()),
    };
    // Row order is preserved, so `paired_ci.py`'s id intersection pairs a
    // rescored run against its source or another rescored run unchanged.
    finish_run(&spec, scored.into_rows(), latencies, out_dir)
}

/// Load `<run>/judge_verdicts.json`, naming the command that writes it.
fn read_judge_file(run: &Path) -> Result<crate::judge::JudgeFile> {
    let path = run.join("judge_verdicts.json");
    let text = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "{} has no judge_verdicts.json; run `myelin-eval judge --run {}` first",
            run.display(),
            run.display()
        )
    })?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// One row's judged score.
///
/// A missing verdict is **not** a zero. [`crate::judge`] never sends an
/// adversarial item or a declined answer to the judge, and both of those are
/// scored by the same deterministic rule `score_one` uses. A missing verdict
/// on an *answered* answerable row means the judge never saw it, and scoring
/// it as wrong would report a number the judge did not produce — so it is a
/// hard error naming the id, matching `standing::judged`'s
/// `IncompleteArtifact` rule.
fn judged_score(judge: &crate::judge::JudgeFile, row: &ScoredQuestion) -> Result<f64> {
    let declined = is_abstention(&row.response_raw);
    if row.is_abstention_problem {
        return Ok(f64::from(u8::from(declined)));
    }
    match judge.verdicts.get(&row.question_id) {
        Some(v) => Ok(f64::from(u8::from(*v == 1))),
        None if declined => Ok(0.0),
        None => anyhow::bail!(
            "question {} was answered but has no verdict from judge {}; \
             re-run `myelin-eval judge` over the whole run before rescoring",
            row.question_id,
            judge.model
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_drops_articles_and_punctuation() {
        assert_eq!(
            normalize("The Quick, brown fox!"),
            vec!["quick", "brown", "fox"]
        );
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

    /// A run directory with three rows and one verdict, for the judged
    /// rescore path.
    fn fixture_run(dir: &Path, verdicts: &[(&str, u8)]) {
        std::fs::create_dir_all(dir).unwrap();
        let row = |id: &str, response: &str, adversarial: bool| {
            serde_json::json!({
                "question_id": id,
                "tenant": "locomo/t",
                "category": 2,
                "question_text": "when?",
                "answer_gold": "June 2023",
                "response_raw": response,
                "score": 0.0,
                "exact_match": 0.0,
                "is_abstention_problem": adversarial,
                "retrieved_items": 6,
                "memory_query_duration_seconds": 1.0,
            })
            .to_string()
        };
        let rows = [
            row("answered", "2023-06-15", false),
            row("declined", "I don't know", false),
            row("adversarial", "I don't know", true),
        ];
        std::fs::write(dir.join("per_question.jsonl"), rows.join("\n")).unwrap();
        std::fs::write(
            dir.join("aggregated_metrics.json"),
            serde_json::json!({"corpus": "locomo", "collection": "c", "mode": "recall", "k": 6})
                .to_string(),
        )
        .unwrap();
        let map: std::collections::BTreeMap<String, u8> = verdicts
            .iter()
            .map(|(id, v)| ((*id).to_string(), *v))
            .collect();
        std::fs::write(
            dir.join("judge_verdicts.json"),
            serde_json::to_string(&crate::judge::JudgeFile {
                model: "test-judge".into(),
                verdicts: map,
            })
            .unwrap(),
        )
        .unwrap();
    }

    /// The judged column reports what the judge said, and the two rows the
    /// judge never sees are scored by the shared decline rule instead of
    /// being dropped.
    #[test]
    fn the_judge_scorer_reads_verdicts_and_applies_the_decline_rule() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("run");
        fixture_run(&src, &[("answered", 1)]);
        let run = rescore_run(&src, &tmp.path().join("out"), Scorer::Judge).unwrap();
        assert_eq!(run.scorer, "judge");

        let rows: Vec<ScoredQuestion> =
            std::fs::read_to_string(tmp.path().join("out/per_question.jsonl"))
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
        let score = |id: &str| rows.iter().find(|r| r.question_id == id).unwrap().score;
        assert_eq!(score("answered"), 1.0, "verdict 1 is a point");
        assert_eq!(score("declined"), 0.0, "a decline on an answerable item");
        assert_eq!(score("adversarial"), 1.0, "declining an unanswerable one");
        // The deterministic columns survive a judged rescore, so the artifact
        // stays readable under either metric.
        let answered = rows.iter().find(|r| r.question_id == "answered").unwrap();
        assert_eq!(answered.score_temporal, 1.0);
        assert_eq!(answered.temporal_kind, "interval");
    }

    /// A row is on disk as soon as it is scored, not when the run ends.
    ///
    /// This is the whole point of `RowSink`: M17 lost 54 minutes of
    /// generations because every row lived in a `Vec` until `finish_run`,
    /// and the run died in the scoring stage. Delete the `flush` and this
    /// test fails; delete the streaming and it fails harder.
    #[test]
    fn a_scored_row_reaches_the_file_before_the_run_finishes() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out");
        let mut sink = RowSink::create(&out).unwrap();
        let path = out.join("per_question.jsonl");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "",
            "the file must exist and be empty before the first row"
        );

        let row = ScoredQuestion {
            question_id: "q1".into(),
            tenant: "t".into(),
            category: 1,
            question_text: "when?".into(),
            answer_gold: "June 2023".into(),
            response_raw: "2023-06-15".into(),
            score: 1.0,
            exact_match: 0.0,
            score_token_f1: 1.0,
            score_temporal: 1.0,
            temporal_kind: "interval".into(),
            is_abstention_problem: false,
            retrieved_items: 6,
            evidence: vec!["e".into()],
            memory_query_duration_seconds: 1.0,
        };
        sink.push(row.clone()).unwrap();

        // Read it back while the sink is still open and the run unfinished.
        let text = std::fs::read_to_string(&path).unwrap();
        let back: ScoredQuestion = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(back.question_id, "q1");
        assert_eq!(sink.len(), 1, "the row is kept for the aggregate too");

        // A second `create` on the same directory truncates: `rescore_run`
        // re-derives every row and must not append under a previous attempt.
        let again = RowSink::create(&out).unwrap();
        assert_eq!(again.len(), 0);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
    }

    /// An answered row the judge never saw is a hard error, not a zero:
    /// scoring it wrong would report a number the judge did not produce.
    #[test]
    fn an_answered_row_with_no_verdict_is_an_error_naming_the_question() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("run");
        fixture_run(&src, &[]);
        let err = rescore_run(&src, &tmp.path().join("out"), Scorer::Judge).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("answered"), "{msg}");
        assert!(msg.contains("no verdict"), "{msg}");
    }

    /// A missing verdicts file names the command that writes it.
    #[test]
    fn a_missing_verdicts_file_names_the_judge_command() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("run");
        fixture_run(&src, &[("answered", 1)]);
        std::fs::remove_file(src.join("judge_verdicts.json")).unwrap();
        let err = rescore_run(&src, &tmp.path().join("out"), Scorer::Judge).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("myelin-eval judge --run"), "{msg}");
    }
}
