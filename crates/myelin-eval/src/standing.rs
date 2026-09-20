//! `myelin-eval standing` — where we actually stand against the published
//! numbers, computed from artifacts instead of asserted in prose (M18).
//!
//! # The failure this replaces
//!
//! Until this module existed, "SOTA" lived in two disconnected places: a
//! hand-maintained markdown table of published numbers (`PLAN.md` §11.5) and
//! twenty-six run directories holding ours. Nothing joined them, so every
//! comparison was a human re-reading both sides and writing a sentence. The
//! sentence was also inadmissible: LoCoMo and LongMemEval_S papers report
//! LLM-judge scores and our columns were token F1, which is a different
//! quantity — a gap computed across those two is meaningless in an unknown
//! direction.
//!
//! So this joins [`Registry`] (one JSON row per published claim, each citing a
//! locally converted paper) against [`Ours`] (one metric per artifact on disk)
//! and emits a [`Verdict`] per row. A row we cannot compare says so and names
//! why; a row missing our side names the command that would produce it. The
//! output is a file, not a paragraph.
//!
//! # Comparability is the product, not the gap
//!
//! The rules are evaluated in a fixed order (first match wins) because the
//! reasons are not symmetric:
//!
//! 1. an unciteable source (`vendor`, `abstract_only`, `paywalled`) can never
//!    be beaten *or* matched — there is no protocol to reproduce. A
//!    `gate: true` row like this still lands in `gated_failures`, so a vendor
//!    blog post cannot satisfy a gate by being unverifiable.
//! 2. no artifact on our side is a missing measurement, not a loss.
//! 3. a different population (`n`) is `PLAN.md` §1.1's landmine: LoCoMo has
//!    1,986 questions and every paper reporting "1,540" silently dropped the
//!    adversarial category. Comparing across that is how a system claims a
//!    number it did not earn.
//! 4. a different judge class, then 5. a different backbone class, are
//!    caveats rather than blockers: the gap is real but a reader must know a
//!    local 9B judge produced one side of it.
//!
//! `claim_allowed` is deliberately the narrowest of these: strictly
//! [`Verdict::Comparable`] and strictly ahead. Everything else is context.
//!
//! # LAFS is never computed here
//!
//! The leaderboard's integral lives in
//! `vendor/longmemeval-v2/leaderboard/compute_lafs.py`. This module shells out
//! to `adapters/lafs_point.py`, and a failure to do so becomes
//! [`Verdict::MissingArtifact`] rather than a Rust reimplementation that could
//! drift from the official tool by a rounding rule.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::attack_live::AttackRun;
use crate::bench::{is_abstention, BenchRun, ScoredQuestion};
use crate::judge::JudgeFile;

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// `docs/sota/registry.json` — every published number we compare against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registry {
    pub schema: u32,
    pub rows: Vec<RegistryRow>,
}

/// One published claim.
///
/// Hand-authored, and every field is load-bearing: `unit` and `direction`
/// decide the sign of the gap, `n` decides whether the populations are the
/// same one, and `judge_class`/`backbone_class` decide whether the comparison
/// needs a caveat. `source` is what makes the row auditable — a DOI, the
/// local converted stem, and the verbatim sentence the value came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryRow {
    pub id: String,
    pub metric: String,
    pub benchmark: String,
    pub system: String,
    pub value: f64,
    pub unit: Unit,
    pub direction: Direction,
    pub n: usize,
    pub backbone: String,
    pub backbone_class: Class,
    pub judge_class: JudgeClass,
    pub provenance: Provenance,
    pub gate: bool,
    /// Whether the gate wants `ours >= theirs` or strictly `ours > theirs`.
    ///
    /// Four of `PLAN.md` §1's five gates are thresholds where equality passes
    /// (LongMemEval_S ≥ 80.80, ASR ≤ 10%). The LAFS gate is not: a gain has to
    /// be strictly positive for the frontier to have moved, and a gain of
    /// exactly 0.0 is the measured signature of a dominated point. Encoding
    /// that difference here keeps it out of the comparison code.
    #[serde(default)]
    pub bar: Bar,
    /// A protocol difference a reader must see but the comparison does not
    /// encode — e.g. MemPro tests the 90% of each dataset it did not sample
    /// for training. Rendered in the report; never compared.
    #[serde(default)]
    pub caveat: Option<String>,
    /// Whether `n` means the same population on both sides.
    ///
    /// `true` (the default) on every benchmark row, because there the
    /// populations are drawn from one released question set and a differing
    /// `n` is `PLAN.md` §1.1's landmine. `false` only where the two numbers
    /// are rates over *each side's own* attack set — MINJA's 3 datasets,
    /// the EHR paper's 50 indication prompts, our 40 attacks — where equal
    /// `n` was never possible and demanding it would make G3 permanently
    /// unverifiable. A row that sets it MUST name both populations in
    /// `caveat`.
    #[serde(default = "yes")]
    pub population_comparable: bool,
    pub source: Source,
}

/// Where a registry row's number came from, precisely enough to re-check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub doi: String,
    /// The local `home-still` stem, so the quote can be re-read offline.
    pub stem: String,
    pub locator: String,
    /// Line range in the converted markdown.
    pub lines: String,
    pub quote: String,
}

/// The registry's wire names are fixed by `docs/sota/registry.json`'s schema,
/// not derived: `pct_0_100` and `fraction_0_1` do not round-trip through
/// `rename_all`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Unit {
    #[serde(rename = "pct_0_100")]
    PctZeroHundred,
    #[serde(rename = "fraction_0_1")]
    FractionZeroOne,
    #[serde(rename = "seconds")]
    Seconds,
}

impl Unit {
    fn slug(self) -> &'static str {
        match self {
            Unit::PctZeroHundred => "pct_0_100",
            Unit::FractionZeroOne => "fraction_0_1",
            Unit::Seconds => "seconds",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    HigherIsBetter,
    LowerIsBetter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bar {
    /// `ours` may tie the published number.
    #[default]
    AtLeast,
    /// `ours` must strictly beat it.
    GreaterThan,
}

/// Answer-model class. Never compared across classes without saying so:
/// `PLAN.md` §13 lists that exact comparison as a threat to G2's validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    FrontierApi,
    OpenWeights,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeClass {
    FrontierApi,
    OpenWeightsLocal,
    /// A deterministic scorer: token F1, the date-aware scorer, exact match.
    Deterministic,
    /// No grader at all — an attack-success rate or a latency.
    #[serde(rename = "none")]
    NoJudge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    Paper,
    Leaderboard,
    /// A vendor blog: not reproducible from any paper.
    Vendor,
    /// Quoted from an abstract because no converted full text exists.
    AbstractOnly,
    Paywalled,
    /// A bar this project set, with the literature row it was derived from as
    /// its `source`. `PLAN.md` §11.5's "MINJA ASR ≤ 10%" is ours; the 6.67%
    /// pre-populated ASR it was rounded up from is the EHR paper's. Encoding
    /// it as `paper` would attribute our threshold to someone else.
    ProjectGate,
}

/// `serde(default)` for [`RegistryRow::population_comparable`].
fn yes() -> bool {
    true
}

impl Provenance {
    /// Can a comparison against this row mean anything at all?
    fn citeable(self) -> bool {
        matches!(
            self,
            Provenance::Paper | Provenance::Leaderboard | Provenance::ProjectGate
        )
    }
}

// ---------------------------------------------------------------------------
// Our side
// ---------------------------------------------------------------------------

/// One metric we can compute from artifacts on disk.
#[derive(Debug, Clone, Serialize)]
pub struct Ours {
    pub metric: String,
    pub value: f64,
    pub unit: Unit,
    pub n: usize,
    pub judge_class: JudgeClass,
    pub backbone_class: Class,
    pub run: PathBuf,
    /// How it was computed, for the report's provenance column.
    pub detail: String,
    /// Set when the artifact exists but does not cover its own population —
    /// an answered row with no judge verdict, say. The value is still
    /// computed so a reader can see what the partial artifact says, but the
    /// row is never claimable.
    pub incomplete: Option<String>,
    /// The run carries a mechanism switch that does **not** ship on, so it
    /// measures an arm rather than the system.
    ///
    /// Before M21 every full-population run was a shipped-default run and
    /// this could not matter. M21 added `runs/m21_full_sel`, a 500-question
    /// arm that scores 60.40 against the default configuration's 56.60 —
    /// and the selection rule below is value-ordered, so the standing table
    /// published the arm. "Where we stand" is what the defaults do.
    #[serde(default)]
    pub arm: bool,
    /// The run's recorded switch set disagrees with **today's** shipped
    /// defaults, so it measures a configuration this code no longer
    /// produces. `None` is a run that today's defaults could have written.
    ///
    /// [`Ours::arm`] and this field answer different questions and neither
    /// implies the other. `arm` is "this run turned something on that ships
    /// off" — a deliberate measurement of today's code. This is "this run
    /// predates, or postdates, a change to what ships" — a measurement of
    /// code that is gone.
    ///
    /// M22 is why it exists. `standing` published **39.91** on
    /// `lme_v2_small.overall_full_set.combined` from `runs/myelin_inv2_web_small`,
    /// a pre-M19 artifact, for six milestones after M19 changed what ships;
    /// re-measuring the *same configuration* on the same store gave
    /// **36.59**. Nothing caught it because the stale run is not an arm —
    /// it was the shipped configuration, of a system that no longer exists.
    /// The drift is computable from the artifacts themselves, so it is
    /// computed rather than remembered.
    #[serde(default)]
    pub unrecorded: Vec<&'static str>,
    /// Every run that supplied this metric, so the report shows what was not
    /// selected.
    pub candidates: Vec<Candidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub run: PathBuf,
    pub value: f64,
}

// ---------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    Comparable,
    CaveatJudge {
        ours: JudgeClass,
        theirs: JudgeClass,
    },
    CaveatBackbone {
        ours: Class,
        theirs: Class,
    },
    NotComparableSubset {
        ours: usize,
        theirs: usize,
    },
    NotComparableSource {
        provenance: Provenance,
    },
    IncompleteArtifact {
        detail: String,
    },
    /// The artifact does not record the operating point it ran at, so what
    /// it measures cannot be reproduced or checked against today's
    /// defaults.
    ///
    /// Not a defect in the data and not a partial artifact: the run
    /// happened and its number is real. What is missing is the evidence
    /// that the number describes *this* system.
    StaleConfig {
        detail: String,
    },
    MissingArtifact {
        command: String,
    },
    /// The registry row's unit and ours cannot be converted into each other.
    ///
    /// A defect in the extractor table or the registry, not in the data — but
    /// one malformed row must not abort the whole report, which is what the
    /// `panic!` in `converted` used to do.
    UnitMismatch {
        detail: String,
    },
}

impl Verdict {
    /// Is a gap between these two numbers a quantity at all?
    fn quantified(&self) -> bool {
        matches!(
            self,
            Verdict::Comparable | Verdict::CaveatJudge { .. } | Verdict::CaveatBackbone { .. }
        )
    }

    fn slug(&self) -> String {
        match self {
            Verdict::Comparable => "comparable".into(),
            Verdict::CaveatJudge { ours, theirs } => {
                format!(
                    "caveat-judge({} vs {})",
                    judge_slug(*ours),
                    judge_slug(*theirs)
                )
            }
            Verdict::CaveatBackbone { ours, theirs } => {
                format!(
                    "caveat-backbone({} vs {})",
                    class_slug(*ours),
                    class_slug(*theirs)
                )
            }
            Verdict::NotComparableSubset { ours, theirs } => {
                format!("not-comparable(n {ours} vs {theirs})")
            }
            Verdict::NotComparableSource { provenance } => {
                format!("not-comparable({})", prov_slug(*provenance))
            }
            Verdict::IncompleteArtifact { detail } => format!("incomplete({detail})"),
            Verdict::StaleConfig { detail } => format!("stale-config({detail})"),
            Verdict::UnitMismatch { detail } => format!("unit-mismatch({detail})"),
            Verdict::MissingArtifact { .. } => "missing-artifact".into(),
        }
    }
}

fn judge_slug(j: JudgeClass) -> &'static str {
    match j {
        JudgeClass::FrontierApi => "frontier_api",
        JudgeClass::OpenWeightsLocal => "open_weights_local",
        JudgeClass::Deterministic => "deterministic",
        JudgeClass::NoJudge => "none",
    }
}

fn class_slug(c: Class) -> &'static str {
    match c {
        Class::FrontierApi => "frontier_api",
        Class::OpenWeights => "open_weights",
    }
}

fn prov_slug(p: Provenance) -> &'static str {
    match p {
        Provenance::Paper => "paper",
        Provenance::Leaderboard => "leaderboard",
        Provenance::Vendor => "vendor",
        Provenance::AbstractOnly => "abstract_only",
        Provenance::Paywalled => "paywalled",
        Provenance::ProjectGate => "project_gate",
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StandingRow {
    pub registry_id: String,
    pub metric: String,
    pub system: String,
    pub theirs: f64,
    pub ours: Option<f64>,
    /// The population our value was computed over. The whole point of the
    /// subset rule, so a consumer of `standing.json` never has to go back to
    /// the run directory to find it.
    pub ours_n: Option<usize>,
    /// Positive always means "we are ahead", whichever way the metric points.
    pub gap: Option<f64>,
    pub verdict: Verdict,
    pub claim_allowed: bool,
    pub gate: bool,
    pub run: Option<PathBuf>,
    pub source_doi: String,
    /// `RegistryRow::caveat`, carried through for the report.
    pub caveat: Option<String>,
    /// `Ours::detail`, so the table's `ours` column is auditable.
    pub detail: Option<String>,
    pub candidates: Vec<Candidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StandingReport {
    pub commit: String,
    pub generated_at: DateTime<Utc>,
    pub rows: Vec<StandingRow>,
    pub gated_failures: Vec<String>,
    /// Every metric we extracted, matched or not, in metric order — the
    /// provenance half of the report. A metric no registry row claims is
    /// still reported: it is what we measured that nobody published, and
    /// dropping it would make the artifact look narrower than the evidence.
    pub ours: Vec<Ours>,
}

// ---------------------------------------------------------------------------
// The metric set
// ---------------------------------------------------------------------------

/// One metric `standing` knows how to extract.
///
/// The direction belongs to the metric, not to a claim about it: an ASR is
/// lower-is-better whoever reports it. A registry row that disagrees is a
/// hard error, because the sign of every gap depends on this.
struct MetricDef {
    id: &'static str,
    direction: Direction,
    /// Printed when no artifact supplies the metric.
    command: &'static str,
    /// The population the metric id advertises, when it advertises one.
    ///
    /// `longmemeval_s.judge_score.n500` promises 500 rows; nothing enforced
    /// it, so a `--limit 50` run produced a row labelled `.n500` carrying
    /// n=50 and a number 450 questions short. The suffix is a contract, and
    /// this is where it is checked.
    nominal_n: Option<usize>,
}

const METRICS: &[MetricDef] = &[
    MetricDef {
        id: "locomo.judge_score.n1540",
        direction: Direction::HigherIsBetter,
        command: "myelin-eval judge --run runs/locomo_recall",
        nominal_n: Some(1540),
    },
    MetricDef {
        id: "locomo.token_f1.n1540",
        direction: Direction::HigherIsBetter,
        command: "myelin-eval bench --corpus locomo --out runs/locomo_recall",
        nominal_n: Some(1540),
    },
    MetricDef {
        id: "locomo.temporal.n1540",
        direction: Direction::HigherIsBetter,
        command: "myelin-eval rescore --run runs/locomo_recall --scorer temporal",
        nominal_n: Some(1540),
    },
    MetricDef {
        id: "locomo.abstention_accuracy.n446",
        direction: Direction::HigherIsBetter,
        command: "myelin-eval bench --corpus locomo --out runs/locomo_recall",
        nominal_n: Some(446),
    },
    MetricDef {
        id: "longmemeval_s.judge_score.n500",
        direction: Direction::HigherIsBetter,
        command: "myelin-eval judge --run runs/lme_s_recall",
        nominal_n: Some(500),
    },
    MetricDef {
        id: "longmemeval_s.token_f1.n500",
        direction: Direction::HigherIsBetter,
        command: "myelin-eval bench --corpus longmemeval-s --out runs/lme_s_recall",
        nominal_n: Some(500),
    },
    MetricDef {
        id: "dmr.accuracy.n500",
        direction: Direction::HigherIsBetter,
        // Deliberately unbuilt: `PLAN.md` §11.5 marks DMR saturated (Zep
        // reports 98.2%), so no runner exists and the row is expected to read
        // `missing-artifact` forever. It stays in the registry because
        // dropping a published number we have not matched would make the
        // table look more complete than the evidence.
        command: "no DMR runner exists (PLAN.md §11.2: DMR is saturated and out of scope)",
        nominal_n: Some(500),
    },
    MetricDef {
        id: "lme_v2_small.overall_full_set.web",
        direction: Direction::HigherIsBetter,
        command: "adapters/run_myelin.py --domain web (LME-V2 harness, tier small)",
        nominal_n: None,
    },
    MetricDef {
        id: "lme_v2_small.overall_full_set.enterprise",
        direction: Direction::HigherIsBetter,
        command: "adapters/run_myelin.py --domain enterprise (LME-V2 harness, tier small)",
        nominal_n: None,
    },
    MetricDef {
        id: "lme_v2_small.overall_full_set.combined",
        direction: Direction::HigherIsBetter,
        command: "adapters/run_myelin.py for BOTH domains at one memory config",
        nominal_n: None,
    },
    MetricDef {
        id: "lme_v2_small.memory_query_avg_seconds.web",
        direction: Direction::LowerIsBetter,
        command: "adapters/run_myelin.py --domain web (LME-V2 harness, tier small)",
        nominal_n: None,
    },
    MetricDef {
        id: "lme_v2_small.memory_query_avg_seconds.enterprise",
        direction: Direction::LowerIsBetter,
        command: "adapters/run_myelin.py --domain enterprise (LME-V2 harness, tier small)",
        nominal_n: None,
    },
    MetricDef {
        id: "lme_v2_small.lafs_gain.small",
        direction: Direction::HigherIsBetter,
        command: "<python> crates/myelin-eval/adapters/lafs_point.py <<< {\"tier\":\"small\",\"points\":[…]}",
        nominal_n: None,
    },
    MetricDef {
        id: "minja.asr.k6_prepopulated",
        direction: Direction::LowerIsBetter,
        command: "myelin-eval attack --live --ledger-dir data --out runs/attack_live_m18",
        nominal_n: None,
    },
    MetricDef {
        id: "minja.asr.k6_prepopulated_defended",
        direction: Direction::LowerIsBetter,
        command: "myelin-eval attack --live --ledger-dir data --out runs/attack_live_m18",
        nominal_n: None,
    },
    MetricDef {
        id: "minja.injection_success.k6_prepopulated",
        direction: Direction::LowerIsBetter,
        command: "myelin-eval attack --live --ledger-dir data --out runs/attack_live_m18",
        nominal_n: None,
    },
];

fn metric_def(id: &str) -> Option<&'static MetricDef> {
    METRICS.iter().find(|m| m.id == id)
}

/// Which way is better for this metric.
///
/// The one piece of [`METRICS`] the ratchet needs, exposed rather than
/// duplicated: a second copy of the direction table is how a lower-is-
/// better metric like `minja.asr.*` ends up graded upside down in one
/// command and not the other. Unknown ids fall back to higher-is-better,
/// matching [`collect`]'s own default for a metric with no definition.
pub fn metric_direction(id: &str) -> Direction {
    metric_def(id)
        .map(|d| d.direction)
        .unwrap_or(Direction::HigherIsBetter)
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Parse and validate the registry.
///
/// Validation is deliberately fatal rather than per-row: an unknown metric id
/// or a direction that disagrees with the metric's own means the table and
/// the extractor have diverged, and a standing report computed across that
/// divergence would be wrong in a direction nobody can see.
pub fn load_registry(path: &Path) -> Result<Registry> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let reg: Registry =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    anyhow::ensure!(
        reg.schema == 1,
        "{} declares schema {} but this build reads schema 1",
        path.display(),
        reg.schema
    );
    let mut ids = BTreeSet::new();
    for row in &reg.rows {
        anyhow::ensure!(
            ids.insert(row.id.clone()),
            "registry row id {:?} appears twice",
            row.id
        );
        let def = metric_def(&row.metric).with_context(|| {
            format!(
                "registry row {:?} names metric {:?}, which `standing` cannot extract; \
                 known metrics: {}",
                row.id,
                row.metric,
                METRICS.iter().map(|m| m.id).collect::<Vec<_>>().join(", ")
            )
        })?;
        anyhow::ensure!(
            def.direction == row.direction,
            "registry row {:?} says {:?} but metric {:?} is {:?}",
            row.id,
            row.direction,
            row.metric,
            def.direction
        );
        anyhow::ensure!(
            row.n > 0,
            "registry row {:?} has n = 0: a claim over no questions is not a claim",
            row.id
        );
        anyhow::ensure!(
            !row.source.quote.trim().is_empty() || !row.provenance.citeable(),
            "registry row {:?} is {:?} but carries no quote",
            row.id,
            prov_slug(row.provenance)
        );
    }
    Ok(reg)
}

/// A vendored LongMemEval-V2 harness run, before domains are paired.
struct HarnessRun {
    dir: PathBuf,
    domain: String,
    acc: f64,
    count: usize,
    avg_seconds: f64,
    judge_class: JudgeClass,
    /// The keys that make two domain runs one operating point.
    fingerprint: String,
    /// Keys in [`PAIR_KEYS`] the artifact does not record at all, so this
    /// run's operating point is not fully recoverable from disk.
    ///
    /// Distinct from a key recorded as `null`: `tau_abstain` is null in
    /// every artifact and means "not set", where an absent `dated` means
    /// the harness that wrote the run had never heard of the switch.
    /// `runs/myelin_inv2_web_small` — the artifact `standing` quoted 39.91
    /// from until M23 — records neither `select`, `dated`,
    /// `prefetch_limit` nor `rerank_depth`.
    unrecorded: Vec<&'static str>,
    /// Does this run's operating point differ from the server's defaults?
    /// A pair is an arm when either half is.
    arm: bool,
}

/// Which `memory_params` make two domain runs one operating point.
///
/// `tau_abstain` is not in the list on purpose: it is `null` in every run and
/// present in some artifacts only because a later harness version wrote the
/// key. Comparing it would split a pair over a schema change.
///
/// `select` and `dated` (M22) ARE in the list, for the opposite reason: they
/// change what the server does per query, so a selection-enabled web run that
/// paired with a plain enterprise run would publish a combined accuracy for
/// an operating point that never ran. Every run written before M22 lacks both
/// keys and therefore reads `null` for each, so their existing pairings are
/// unchanged.
///
/// `pool_rerank` and `premise` (M23 Phase A) join them for the same reason,
/// with the same backward compatibility: runs written before M23 lack both
/// and read `null`, and the M23 arms write both as explicit booleans, so the
/// M23 arms pair only with each other and the M22 bases pair only with each
/// other.
///
/// `decompose` (M24) joins them for the same reason: it changes the
/// candidate pool per query, so a decomposed web run must not cross-pair
/// with a plain enterprise one.
const PAIR_KEYS: [&str; 12] = [
    "mode",
    "k",
    "budget_tokens",
    "max_steps",
    "prefetch_limit",
    "rerank_depth",
    "select",
    "dated",
    "pool_rerank",
    "premise",
    "typed_probes",
    "decompose",
];

/// Walk `runs`, extract every metric any artifact supports, and keep the best
/// per metric.
///
/// `python` runs `adapters/lafs_point.py`; nothing else here needs an
/// interpreter, a GPU, or a network.
pub fn collect(runs: &Path, python: &str) -> Result<BTreeMap<String, Ours>> {
    let mut found: BTreeMap<String, Vec<Ours>> = BTreeMap::new();
    let mut harness: Vec<HarnessRun> = Vec::new();

    let mut dirs = Vec::new();
    walk(runs, 0, &mut dirs)?;
    dirs.sort();

    for dir in &dirs {
        let agg = dir.join("aggregated_metrics.json");
        if agg.exists() {
            let text =
                std::fs::read_to_string(&agg).with_context(|| format!("read {}", agg.display()))?;
            let value: Value =
                serde_json::from_str(&text).with_context(|| format!("parse {}", agg.display()))?;
            if value.get("corpus").is_some() {
                for o in bench_metrics(dir, &text)? {
                    found.entry(o.metric.clone()).or_default().push(o);
                }
            } else if value.get("overall").is_some() {
                let (run, metrics) = harness_metrics(dir, &value)?;
                for o in metrics {
                    found.entry(o.metric.clone()).or_default().push(o);
                }
                harness.push(run);
            } else {
                eprintln!(
                    "warning: {} has neither `corpus` nor `overall`; not a run artifact",
                    agg.display()
                );
            }
        }
        let attack = dir.join("attack_live.json");
        if attack.exists() {
            for o in attack_metrics(dir, &attack)? {
                found.entry(o.metric.clone()).or_default().push(o);
            }
        }
    }

    for o in pair_metrics(&harness, python) {
        found.entry(o.metric.clone()).or_default().push(o);
    }

    let mut out = BTreeMap::new();
    for (metric, mut candidates) in found {
        let direction = metric_def(&metric)
            .map(|d| d.direction)
            .unwrap_or(Direction::HigherIsBetter);
        // Complete first, then the shipped configuration, then the largest
        // population, then best by direction.
        //
        // Population before value on purpose: `runs/lme_s_recall_probe` is a
        // 114-row smoke run and scores higher than the 470-row real one, so a
        // value-first rule would select the probe and the join would then
        // report "not comparable" for a benchmark we have a comparable
        // artifact for. Ties inside one population are broken by the metric's
        // own direction, and by path so the report is deterministic.
        //
        // **Defaults before population, and before value.** `standing`
        // answers "where do we stand", and we stand where the defaults put
        // us. M21 is where this started to matter: `runs/m21_full_sel` is a
        // complete 500-question artifact scoring 60.40 against the default
        // configuration's 56.60, and it measures a switch that ships **off**
        // — so a value-ordered rule published a number no default produces.
        // An arm is still listed in `candidates`, and is still selected when
        // it is the only artifact for a metric.
        candidates.sort_by(|a, b| prefer(a, b, direction));
        let listing: Vec<Candidate> = candidates
            .iter()
            .map(|c| Candidate {
                run: c.run.clone(),
                value: c.value,
            })
            .collect();
        let mut best = candidates.remove(0);
        best.candidates = listing;
        out.insert(metric, best);
    }
    Ok(out)
}

/// Which of two artifacts for the same metric the report should quote.
///
/// Complete before partial, **current configuration before stale**,
/// **shipped configuration before arm**, larger population before smaller,
/// then the metric's own direction, then path so the report is
/// deterministic.
///
/// Stale outranks arm on purpose: an arm measures today's code with a
/// switch flipped, which is at least a configuration this build can
/// reproduce. A stale artifact measures code that is gone, and "where do we
/// stand" cannot be answered by it at all.
fn prefer(a: &Ours, b: &Ours, direction: Direction) -> std::cmp::Ordering {
    let (x, y) = match direction {
        Direction::HigherIsBetter => (b.value, a.value),
        Direction::LowerIsBetter => (a.value, b.value),
    };
    a.incomplete
        .is_some()
        .cmp(&b.incomplete.is_some())
        // Fewer unrecorded keys first: zero is a fully recoverable
        // operating point, and among artifacts that are all partly
        // unrecoverable the closest one to today's code is the least bad
        // answer to "where do we stand". On the LME-V2 pair this is the
        // difference between quoting `myelin_inv2_web_small` (pre-M19,
        // seven keys missing, 39.91) and `m22_base_*` (three, 36.59) —
        // and 36.59 is the number M22 actually measured the defaults at.
        .then_with(|| a.unrecorded.len().cmp(&b.unrecorded.len()))
        .then_with(|| a.arm.cmp(&b.arm))
        .then_with(|| b.n.cmp(&a.n))
        .then_with(|| x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal))
        .then_with(|| a.run.cmp(&b.run))
}

/// Depth-capped directory walk. `runs/rescored/<run>` is two levels down, and
/// nothing legitimate is deeper.
fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) -> Result<()> {
    if depth > 2 || !dir.is_dir() {
        return Ok(());
    }
    out.push(dir.to_path_buf());
    let entries = std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            walk(&entry.path(), depth + 1, out)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Extractors — `bench`
// ---------------------------------------------------------------------------

/// Every metric a `bench` run directory supports.
///
/// Subsets are computed from `per_question.jsonl`, never from a pre-aggregated
/// field, so "the 1,540" has one definition. `BenchRun` supplies the run's
/// switches and its aggregate score; the rows supply the population.
fn bench_metrics(dir: &Path, agg_text: &str) -> Result<Vec<Ours>> {
    let run: BenchRun = serde_json::from_str(agg_text)
        .with_context(|| format!("parse {} as a bench run", dir.display()))?;
    let rows = read_rows(dir)?;
    let verdicts = read_verdicts(dir)?;
    let mut out = Vec::new();

    // Every myelin run is served by the local open-weights reader
    // (`ops/big/serve-models.sh`); the bench artifact does not record the
    // model, so this is stated rather than read.
    let backbone = Class::OpenWeights;
    let scorer = if run.scorer.is_empty() {
        "token_f1"
    } else {
        run.scorer.as_str()
    };

    // Does this run measure the system, or an arm of it? Every switch below
    // ships **off**, so any of them being set makes the run a measurement of
    // something the defaults do not do. `resolve_dates` and `timeline` are
    // deliberately absent: both ship on since M19, and a run that carries
    // them is the shipped configuration.
    let arm = run.graph
        || run.chronological
        || run.question_date
        || run.profile
        || run.profile_clause
        || run.mmr.is_some()
        || run.select_sufficient
        // M23's four, all shipping off pending their measurement.
        || run.rerank_pool
        || run.premise
        || run.typed_probes
        || run.untrusted_max.is_some()
        // M24's sub-query decomposition.
        || run.decompose.is_some();

    // Does this run record its own operating point? Every key below defines
    // part of what the system does per query today. An artifact that does
    // not carry one of them was written by a harness that predates it, and
    // the run's behaviour on that axis cannot be recovered from the
    // artifact — so the run cannot be quoted as "where we stand" even when
    // its number is the best one on disk.
    //
    // This is the M22 defect made mechanical, and it deliberately fires on
    // absence rather than on a value: the point is not that the old run was
    // configured differently, it is that nobody can check.
    let unrecorded = unrecorded_bench_keys(agg_text)?;

    match run.corpus.as_str() {
        "locomo" => {
            let stratum: Vec<&ScoredQuestion> = rows
                .iter()
                .filter(|r| (1..=4).contains(&r.category))
                .collect();
            let abstention = rows.iter().filter(|r| r.is_abstention_problem).count();

            if scorer == "token_f1" {
                out.push(Ours {
                    metric: "locomo.token_f1.n1540".into(),
                    value: run.f1_answerable * 100.0,
                    unit: Unit::PctZeroHundred,
                    n: stratum.len(),
                    judge_class: JudgeClass::Deterministic,
                    backbone_class: backbone,
                    run: dir.to_path_buf(),
                    detail: format!(
                        "aggregated_metrics.json f1_answerable over categories 1-4, mode={} k={}",
                        run.mode, run.k
                    ),
                    incomplete: None,
                    arm: false,
                    unrecorded: Vec::new(),
                    candidates: Vec::new(),
                });
            }
            if scorer == "temporal" {
                out.push(Ours {
                    metric: "locomo.temporal.n1540".into(),
                    value: run.f1_answerable * 100.0,
                    unit: Unit::PctZeroHundred,
                    n: stratum.len(),
                    judge_class: JudgeClass::Deterministic,
                    backbone_class: backbone,
                    run: dir.to_path_buf(),
                    detail: format!(
                        "date-aware scorer, f1_answerable over categories 1-4, rescored_from={}",
                        run.rescored_from
                            .clone()
                            .unwrap_or_else(|| "live run".into())
                    ),
                    incomplete: None,
                    arm: false,
                    unrecorded: Vec::new(),
                    candidates: Vec::new(),
                });
            }
            out.push(Ours {
                metric: "locomo.abstention_accuracy.n446".into(),
                value: run.abstention_accuracy * 100.0,
                unit: Unit::PctZeroHundred,
                n: abstention,
                judge_class: JudgeClass::Deterministic,
                backbone_class: backbone,
                run: dir.to_path_buf(),
                detail: "aggregated_metrics.json abstention_accuracy over category 5".into(),
                incomplete: None,
                arm: false,
                unrecorded: Vec::new(),
                candidates: Vec::new(),
            });
            if let Some(judge) = &verdicts {
                out.push(judged(
                    "locomo.judge_score.n1540",
                    dir,
                    &stratum,
                    judge,
                    backbone,
                ));
            }
        }
        "longmemeval_s" => {
            let answerable: Vec<&ScoredQuestion> =
                rows.iter().filter(|r| !r.is_abstention_problem).collect();
            if scorer == "token_f1" {
                out.push(Ours {
                    metric: "longmemeval_s.token_f1.n500".into(),
                    value: run.f1_answerable * 100.0,
                    unit: Unit::PctZeroHundred,
                    n: answerable.len(),
                    judge_class: JudgeClass::Deterministic,
                    backbone_class: backbone,
                    run: dir.to_path_buf(),
                    detail: format!(
                        "aggregated_metrics.json f1_answerable over the {} answerable rows, mode={} k={}",
                        answerable.len(),
                        run.mode,
                        run.k
                    ),
                    incomplete: None,
                    arm: false,
                    unrecorded: Vec::new(),
                    candidates: Vec::new(),
                });
            }
            if let Some(judge) = &verdicts {
                // The whole 500, not the 470 answerable ones. MemPro's
                // LongMemEval `Avg.` reproduces exactly as the micro-average
                // over the official type counts (133 temporal, 133
                // multi-session, 78 knowledge-update, 70 single-session-user,
                // 56 single-session-assistant, 30 preference = 500), and the
                // 30 `_abs` items sit *inside* those counts — so a comparable
                // number has to carry them. They are graded by the same
                // deterministic decline rule `bench` uses; only the answerable
                // rows go to the judge.
                let all: Vec<&ScoredQuestion> = rows.iter().collect();
                out.push(judged(
                    "longmemeval_s.judge_score.n500",
                    dir,
                    &all,
                    judge,
                    backbone,
                ));
            }
        }
        other => eprintln!(
            "warning: {} is corpus {other:?}, which has no metrics",
            dir.display()
        ),
    }
    // A stratum run (`bench --categories 2`) has zero rows outside its own
    // stratum, and a metric with an empty population is not a measurement.
    // Without this, a LoCoMo temporal-only arm publishes
    // `locomo.abstention_accuracy.n446` as 0.00 — a number no reader
    // produced — and it counts as *complete*, which is the half of the
    // selection rule that runs before "prefer the larger n".
    out.retain(|o| o.n > 0);
    // One flag per run, stamped on every metric it produced, rather than
    // threaded through each literal and `judged`'s signature.
    for o in &mut out {
        o.arm = arm;
        o.unrecorded.clone_from(&unrecorded);
    }
    Ok(out)
}

/// Keys a `bench` artifact must carry for its operating point to be
/// recoverable from disk, in the order the report names them.
///
/// Only switches that change what the system does **per query** belong
/// here. `graph`, `chronological`, `question_date`, `profile`,
/// `profile_clause`, `mmr` and `select_sufficient` are deliberately absent
/// for the opposite reason to [`Ours::arm`]'s list: they all ship off, so an
/// artifact that omits one was doing what today's default does, and its
/// number is still this system's number.
///
/// `resolve_dates` and `timeline` are the two that matter, and they are why
/// this exists. Both ship **on** (M19), both are written from
/// `ComposeConfig::default()` on every run since, and every artifact older
/// than M19 omits them — which is exactly the population whose numbers
/// `standing` was still quoting six milestones later.
const BENCH_OPERATING_POINT: [&str; 2] = ["resolve_dates", "timeline"];

/// Which of [`BENCH_OPERATING_POINT`] this artifact does not record.
///
/// Reads the raw JSON rather than [`BenchRun`] on purpose: `serde(default)`
/// is what makes the historical artifacts parse at all, and it erases the
/// distinction between "recorded as false" and "written before the key
/// existed" — which is the entire signal.
fn unrecorded_bench_keys(agg_text: &str) -> Result<Vec<&'static str>> {
    let raw: Value = serde_json::from_str(agg_text).context("parse bench artifact as json")?;
    Ok(BENCH_OPERATING_POINT
        .iter()
        .filter(|k| raw.get(**k).is_none())
        .copied()
        .collect())
}

/// Render a drift list as the sentence the report prints, or `None` when
/// the artifact records everything today's code has.
///
/// One function for both artifact shapes so the two paths can never drift
/// apart in how they say the same thing.
fn stale_note(unrecorded: &[&'static str]) -> Option<String> {
    (!unrecorded.is_empty()).then(|| {
        format!(
            "artifact does not record {}; re-measure on this code before quoting it",
            unrecorded.join(", ")
        )
    })
}

/// Every operating-point key either half of a domain pair failed to record,
/// de-duplicated and in [`PAIR_KEYS`] order so the note is stable.
fn union_unrecorded(a: &HarnessRun, b: &HarnessRun) -> Vec<&'static str> {
    PAIR_KEYS
        .iter()
        .filter(|k| a.unrecorded.contains(k) || b.unrecorded.contains(k))
        .copied()
        .collect()
}

/// The judged column over one stratum.
///
/// A missing verdict is not a zero. [`crate::judge`] skips rows whose answer
/// is a decline, and those *are* zeros under every scorer — but a missing
/// verdict on an answered row means the judge never saw it, and scoring that
/// as wrong would report a number the judge did not produce. The first case
/// counts as covered; the second makes the metric
/// [`Verdict::IncompleteArtifact`].
///
/// An abstention item in the stratum is never judged: it is correct iff the
/// reader declined, which is [`crate::bench::score_one`]'s rule and the
/// benchmarks' own. LoCoMo's stratum excludes them (every paper reporting
/// "1,540" dropped the adversarial category); LongMemEval_S's includes them.
fn judged(
    metric: &str,
    dir: &Path,
    stratum: &[&ScoredQuestion],
    judge: &JudgeFile,
    backbone: Class,
) -> Ours {
    let mut correct = 0usize;
    let mut covered = 0usize;
    let mut declined = 0usize;
    for row in stratum {
        if row.is_abstention_problem {
            covered += 1;
            if is_abstention(&row.response_raw) {
                correct += 1;
            }
            continue;
        }
        match judge.verdicts.get(&row.question_id) {
            Some(v) => {
                covered += 1;
                if *v == 1 {
                    correct += 1;
                }
            }
            None if is_abstention(&row.response_raw) => {
                covered += 1;
                declined += 1;
            }
            None => {}
        }
    }
    let n = stratum.len();
    let incomplete = (covered < n).then(|| format!("{covered}/{n} judged"));
    Ours {
        metric: metric.into(),
        // The denominator is the whole stratum, so a decline costs what it
        // costs. On the full corpora that is LoCoMo's 1,540 (`PLAN.md` §1.1)
        // and LongMemEval_S's answerable rows; a short run reports its own n
        // and is rejected by the population rule rather than scaled up.
        value: if n == 0 {
            0.0
        } else {
            correct as f64 / n as f64 * 100.0
        },
        unit: Unit::PctZeroHundred,
        n,
        judge_class: JudgeClass::OpenWeightsLocal,
        backbone_class: backbone,
        run: dir.to_path_buf(),
        detail: format!(
            "judge {} : {correct} of {n} correct ({declined} declines scored 0)",
            judge.model
        ),
        incomplete,
        arm: false,
        unrecorded: Vec::new(),
        candidates: Vec::new(),
    }
}

fn read_rows(dir: &Path) -> Result<Vec<ScoredQuestion>> {
    let path = dir.join("per_question.jsonl");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        rows.push(
            serde_json::from_str::<ScoredQuestion>(line)
                .with_context(|| format!("parse a row of {}", path.display()))?,
        );
    }
    Ok(rows)
}

fn read_verdicts(dir: &Path) -> Result<Option<JudgeFile>> {
    let path = dir.join("judge_verdicts.json");
    if !path.exists() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    Ok(Some(
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?,
    ))
}

// ---------------------------------------------------------------------------
// Extractors — the vendored LongMemEval-V2 harness
// ---------------------------------------------------------------------------

fn harness_metrics(dir: &Path, agg: &Value) -> Result<(HarnessRun, Vec<Ours>)> {
    let args = read_json(&dir.join("run_args.json"))?;
    let domain = args
        .get("domain")
        .and_then(Value::as_str)
        .with_context(|| format!("{}/run_args.json has no `domain`", dir.display()))?
        .to_string();
    let evaluator = args
        .get("evaluator_model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    // `gpt-5.2` is the leaderboard's evaluator. Anything else is our local
    // judge, which is the caveat on every LME-V2 comparison we make.
    let judge_class = if evaluator == "gpt-5.2" {
        JudgeClass::FrontierApi
    } else {
        JudgeClass::OpenWeightsLocal
    };
    let overall = agg.get("overall").context("no `overall`")?;
    let acc = overall
        .get("overall_full_set")
        .and_then(Value::as_f64)
        .context("overall.overall_full_set")?
        * 100.0;
    let count = overall
        .get("count_all_questions")
        .and_then(Value::as_u64)
        .context("overall.count_all_questions")? as usize;
    let avg_seconds = agg
        .get("memory_query")
        .and_then(|m| m.get("avg_seconds"))
        .and_then(Value::as_f64)
        .context("memory_query.avg_seconds")?;

    let fingerprint = fingerprint(dir)?;
    let mut out = Vec::new();
    // A run with no memory config is a control, not a measurement of this
    // system: `runs/baseline_no_retrieval_web_small` answers from no evidence
    // at all, and its `memory_query.avg_seconds` is 7e-7 — the fastest
    // "retrieval" on disk and a meaningless winner for a latency claim.
    let control = fingerprint.starts_with("unpairable:");
    if control {
        eprintln!(
            "note: {} has no runtime_inputs/memory_config.json; treated as a control, \
             not a measurement",
            dir.display()
        );
    }
    let unrecorded = unrecorded_pair_keys(dir)?;
    let arm = harness_arm(dir)?;
    if !control && (domain == "web" || domain == "enterprise") {
        out.push(Ours {
            metric: format!("lme_v2_small.overall_full_set.{domain}"),
            value: acc,
            unit: Unit::PctZeroHundred,
            n: count,
            judge_class,
            backbone_class: Class::OpenWeights,
            run: dir.to_path_buf(),
            detail: format!(
                "overall.overall_full_set x 100 over {count} questions, evaluator {}",
                if evaluator.is_empty() {
                    "(unset)"
                } else {
                    &evaluator
                }
            ),
            incomplete: None,
            arm,
            unrecorded: unrecorded.clone(),
            candidates: Vec::new(),
        });
        out.push(Ours {
            metric: format!("lme_v2_small.memory_query_avg_seconds.{domain}"),
            value: avg_seconds,
            unit: Unit::Seconds,
            n: count,
            judge_class: JudgeClass::NoJudge,
            backbone_class: Class::OpenWeights,
            run: dir.to_path_buf(),
            detail: format!("memory_query.avg_seconds over {count} questions"),
            incomplete: None,
            arm,
            unrecorded: unrecorded.clone(),
            candidates: Vec::new(),
        });
    }
    Ok((
        HarnessRun {
            dir: dir.to_path_buf(),
            domain,
            acc,
            count,
            avg_seconds,
            judge_class,
            fingerprint,
            unrecorded,
            arm,
        },
        out,
    ))
}

/// The operating point a harness run was produced at, as a stable string.
///
/// A run with no `runtime_inputs/memory_config.json` gets an unpairable
/// fingerprint rather than an error: `runs/baseline_no_retrieval_web_small`
/// is the no-memory control and has no memory config to record. Pairing it
/// with an enterprise run would invent an operating point that never existed.
fn fingerprint(dir: &Path) -> Result<String> {
    let path = dir.join("runtime_inputs/memory_config.json");
    if !path.exists() {
        return Ok(format!("unpairable:{}", dir.display()));
    }
    let cfg = read_json(&path)?;
    let params = cfg.get("memory_params").unwrap_or(&Value::Null);
    Ok(PAIR_KEYS
        .iter()
        .map(|k| {
            let v = params.get(*k).unwrap_or(&Value::Null);
            format!("{k}={v}")
        })
        .collect::<Vec<_>>()
        .join(" "))
}

/// Which [`PAIR_KEYS`] this harness artifact does not record at all.
///
/// A control with no memory config records nothing, but it is already
/// excluded from every metric by its unpairable fingerprint, so it reports
/// no drift rather than all of it.
///
/// `Value::Null` is **not** absence here. `tau_abstain` is explicitly null
/// in every artifact that has it, and an explicit null is a recorded
/// decision; a missing key is a harness that could not have made one.
fn unrecorded_pair_keys(dir: &Path) -> Result<Vec<&'static str>> {
    let path = dir.join("runtime_inputs/memory_config.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let cfg = read_json(&path)?;
    let Some(params) = cfg.get("memory_params") else {
        return Ok(Vec::new());
    };
    Ok(PAIR_KEYS
        .iter()
        .filter(|k| params.get(**k).is_none())
        .copied()
        .collect())
}

/// The boolean switches in [`PAIR_KEYS`], and what the MCP server does when
/// the caller says nothing about them.
///
/// `dated` is the one that is **on**: `apply_operating_point` suppresses the
/// date machinery only for an explicit `false`, because "a corpus is dated
/// until someone says it is not". The other four are M21–M23 mechanisms
/// that all ship off pending a measurement.
const PAIR_SWITCH_DEFAULTS: [(&str, bool); 5] = [
    ("select", false),
    ("dated", true),
    ("pool_rerank", false),
    ("premise", false),
    ("typed_probes", false),
];

/// Does this harness artifact record an operating point the server's
/// defaults do not produce?
///
/// The LME-V2 path had no answer to this at all before M23 — every
/// `Ours` it built was `arm: false` — which is [`Ours::arm`]'s own
/// motivating defect, unfixed on the benchmark G1 is scored on. Measured
/// consequence: `runs/m22_nodate_web` carries `dated: false`, a switch M22
/// measured as a **null** and therefore left off, and it was published as
/// `lme_v2_small.overall_full_set.combined` at 39.02 ahead of the shipped
/// configuration's 36.59.
///
/// A recorded `null` is not an arm: it is the caller declining to override,
/// which is what the default already is. Only an explicit value that
/// differs counts.
fn harness_arm(dir: &Path) -> Result<bool> {
    let path = dir.join("runtime_inputs/memory_config.json");
    if !path.exists() {
        return Ok(false);
    }
    let cfg = read_json(&path)?;
    let Some(params) = cfg.get("memory_params") else {
        return Ok(false);
    };
    let switched = PAIR_SWITCH_DEFAULTS
        .iter()
        .any(|(key, shipped)| params.get(key).and_then(Value::as_bool) == Some(!shipped));
    // Width is an arm whenever it is stated: `RetrieveConfig::default()`
    // supplies both, so any recorded number is an override of it.
    let widened = ["prefetch_limit", "rerank_depth"]
        .iter()
        .any(|key| params.get(key).is_some_and(|v| !v.is_null()));
    Ok(switched || widened)
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// LongMemEval-V2 tier-small's population, per domain.
///
/// Not a convention: `docs/sota/registry.json` states the AgentRunbook rows
/// against the tier's own 451 questions (web 240 + enterprise 211), and any
/// other population is a different measurement wearing the same metric id.
const LME_V2_SMALL_WEB: usize = 240;
const LME_V2_SMALL_ENTERPRISE: usize = 211;

/// The two metrics that need a web+enterprise pair: the tier-small combined
/// accuracy, and the LAFS gain the leaderboard scores.
///
/// A domain on its own is 240 or 211 of the tier's 451 questions, so neither
/// is comparable to a published LME-V2-Small number. Pairing is by operating
/// point, never by directory name.
fn pair_metrics(harness: &[HarnessRun], python: &str) -> Vec<Ours> {
    let mut pairs: Vec<(&HarnessRun, &HarnessRun)> = Vec::new();
    for web in harness.iter().filter(|h| h.domain == "web") {
        // A `--limit` run is a pilot, not a submission. Without this check a
        // 40-question web pilot pairs with a 211-question enterprise arm —
        // the fingerprints match, because `--limit` is not an operating point
        // — and publishes a "combined" accuracy over 251 questions against a
        // bar defined on 451. Measured: M22's three pilots did exactly that
        // and entered the candidate list, where the only thing standing
        // between one of them and the published number was its value.
        if web.count != LME_V2_SMALL_WEB {
            continue;
        }
        for ent in harness.iter().filter(|h| h.domain == "enterprise") {
            if ent.count != LME_V2_SMALL_ENTERPRISE {
                continue;
            }
            if web.fingerprint == ent.fingerprint {
                pairs.push((web, ent));
            }
        }
    }
    let mut out = Vec::new();
    let mut points = Vec::new();
    for (web, ent) in &pairs {
        let n = web.count + ent.count;
        if n == 0 {
            continue;
        }
        let acc = (web.acc * web.count as f64 + ent.acc * ent.count as f64) / n as f64;
        let latency =
            (web.avg_seconds * web.count as f64 + ent.avg_seconds * ent.count as f64) / n as f64;
        let name = format!("{}+{}", stem(&web.dir), stem(&ent.dir));
        out.push(Ours {
            metric: "lme_v2_small.overall_full_set.combined".into(),
            value: acc,
            unit: Unit::PctZeroHundred,
            n,
            judge_class: if web.judge_class == ent.judge_class {
                web.judge_class
            } else {
                // Two evaluators over one number is not one number.
                JudgeClass::OpenWeightsLocal
            },
            backbone_class: Class::OpenWeights,
            run: web.dir.clone(),
            detail: format!(
                "question-weighted mean of {} ({:.2} over {}) and {} ({:.2} over {}); {}",
                stem(&web.dir),
                web.acc,
                web.count,
                stem(&ent.dir),
                ent.acc,
                ent.count,
                web.fingerprint
            ),
            incomplete: None,
            arm: web.arm || ent.arm,
            // A pair is only as recoverable as its least-recorded half.
            unrecorded: union_unrecorded(web, ent),
            candidates: Vec::new(),
        });
        points.push((name, acc, latency, n, web.dir.clone()));
    }

    if points.is_empty() {
        return out;
    }
    let request = serde_json::json!({
        "tier": "small",
        "points": points
            .iter()
            .map(|(name, acc, latency, _, _)| serde_json::json!({
                "name": name, "acc": acc, "latency": latency
            }))
            .collect::<Vec<_>>(),
    });
    // The LAFS point is computed over every pair, so it inherits the drift
    // of every pair that fed it: one unrecoverable operating point in the
    // submission set makes the frontier it was scored against unreadable.
    let lafs_unrecorded: Vec<&'static str> = PAIR_KEYS
        .iter()
        .filter(|k| {
            pairs
                .iter()
                .any(|(web, ent)| web.unrecorded.contains(k) || ent.unrecorded.contains(k))
        })
        .copied()
        .collect();
    let n = points.iter().map(|p| p.3).max().unwrap_or(0);
    let run = points[0].4.clone();
    match lafs_gain(python, &request) {
        Ok(gain) => out.push(Ours {
            metric: "lme_v2_small.lafs_gain.small".into(),
            value: gain,
            unit: Unit::PctZeroHundred,
            n,
            judge_class: JudgeClass::OpenWeightsLocal,
            backbone_class: Class::OpenWeights,
            run,
            detail: format!(
                "adapters/lafs_point.py over {} submission point(s): {}",
                points.len(),
                points
                    .iter()
                    .map(|(name, acc, latency, _, _)| format!("{name} {acc:.2}@{latency:.2}s"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            incomplete: None,
            // The LAFS point is a frontier over every submitted pair, so
            // one arm in the set makes the whole point an arm's.
            arm: pairs.iter().any(|(web, ent)| web.arm || ent.arm),
            unrecorded: lafs_unrecorded,
            candidates: Vec::new(),
        }),
        Err(err) => eprintln!(
            "warning: LAFS gain unavailable ({}); the metric is reported as a missing artifact",
            err.trim()
        ),
    }
    out
}

fn stem(dir: &Path) -> String {
    dir.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| dir.display().to_string())
}

/// Shell out to the official LAFS tool. Never reimplemented in Rust.
fn lafs_gain(python: &str, request: &Value) -> std::result::Result<f64, String> {
    let script = Path::new("crates/myelin-eval/adapters/lafs_point.py");
    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn {python} {}: {e}", script.display()))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| "no stdin".to_string())?
        .write_all(request.to_string().as_bytes())
        .map_err(|e| format!("write request: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("wait: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).to_string());
    }
    let reply: Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("parse reply: {e}: {}", String::from_utf8_lossy(&out.stdout)))?;
    reply
        .get("lafs_gain")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("reply has no lafs_gain: {reply}"))
}

// ---------------------------------------------------------------------------
// Extractors — the live attack sweep
// ---------------------------------------------------------------------------

/// G3's three numbers, from the artifact `attack --live --out` writes.
///
/// The selector is `(prepopulated, tier, adjudicated)` rather than the
/// condition's printed name, so a label change cannot silently re-point a
/// gate at a different condition.
fn attack_metrics(dir: &Path, path: &Path) -> Result<Vec<Ours>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let run: AttackRun =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let mut out = Vec::new();
    for (metric, adjudicated, injection) in [
        ("minja.asr.k6_prepopulated", false, false),
        ("minja.asr.k6_prepopulated_defended", true, false),
        ("minja.injection_success.k6_prepopulated", false, true),
    ] {
        // Operating-point rule, fixed before the M23 sweep ran: the reported
        // condition is the CHEAPEST quota that meets the gate — `None` first
        // (adjudicator alone), then `Some(3)`, then `Some(2)`, the order of
        // decreasing retained utility for untrusted-but-benign evidence. A
        // condition whose ASR at k=6 is over 10% cannot be the operating
        // point; if none qualifies, the least restrictive is reported and
        // the gate fails on its real number.
        let matching: Vec<&crate::attack_live::Condition> = run
            .conditions
            .iter()
            .filter(|c| {
                c.prepopulated
                    && c.adjudicated == adjudicated
                    && matches!(
                        c.tier,
                        myelin_core::pipeline::consolidate::SourceTier::Untrusted
                    )
            })
            .collect();
        let quota_rank = |q: Option<usize>| match q {
            None => 0usize,
            Some(3) => 1,
            Some(_) => 2,
        };
        let cond = matching
            .iter()
            .copied()
            .filter(|c| {
                c.asr
                    .iter()
                    .find(|(k, _)| *k == 6)
                    .map(|(_, h)| c.rate(*h))
                    .is_some_and(|r| r <= 0.10)
            })
            .min_by_key(|c| quota_rank(c.untrusted_max))
            .or_else(|| {
                matching
                    .iter()
                    .copied()
                    .min_by_key(|c| quota_rank(c.untrusted_max))
            });
        let Some(cond) = cond else { continue };
        let hits = if injection {
            cond.injection
                .iter()
                .find(|(k, _)| *k == 6)
                .map(|(_, h)| *h)
        } else {
            cond.asr.iter().find(|(k, _)| *k == 6).map(|(_, h)| *h)
        };
        let Some(hits) = hits else { continue };
        out.push(Ours {
            metric: metric.into(),
            value: cond.rate(hits) * 100.0,
            unit: Unit::PctZeroHundred,
            n: cond.attempted,
            judge_class: JudgeClass::NoJudge,
            backbone_class: Class::OpenWeights,
            run: dir.to_path_buf(),
            detail: format!(
                "condition {:?}: {hits}/{} at k=6 over {} cohorts, {} injections admitted",
                cond.name, cond.attempted, cond.cohorts, cond.injected
            ),
            incomplete: None,
            arm: false,
            unrecorded: Vec::new(),
            candidates: Vec::new(),
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Join the registry against our artifacts.
///
/// Pure: no I/O, so the rule order is testable against fixtures.
pub fn compare(reg: &Registry, ours: &BTreeMap<String, Ours>) -> StandingReport {
    let mut rows = Vec::new();
    let mut gated_failures = Vec::new();
    let mut matched: BTreeSet<&str> = BTreeSet::new();

    for row in &reg.rows {
        let mine = ours.get(&row.metric);
        if mine.is_some() {
            matched.insert(row.metric.as_str());
        }
        let mut verdict = classify(row, mine);
        // One unconvertible unit pair degrades exactly its own row. It used
        // to `panic!` inside `converted`, which threw away the whole report
        // — including every row that was fine — for a defect in one registry
        // or extractor entry.
        let mut gap_value: Option<f64> = None;
        if let (true, Some(m)) = (verdict.quantified(), mine) {
            match gap(row, m) {
                Ok(g) => gap_value = Some(g),
                Err(detail) => verdict = Verdict::UnitMismatch { detail },
            }
        }
        let gap = gap_value;
        let ours_converted = match mine {
            Some(m) => converted(m, row.unit).ok(),
            None => None,
        };
        let claim_allowed = verdict == Verdict::Comparable && gap.is_some_and(|g| g > 0.0);
        if row.gate {
            // `Comparable`, not `quantified()`: a caveated comparison is a
            // number whose protocol differs from the paper's, and a gate is
            // a claim that we beat the paper. `claim_allowed` directly above
            // already draws that line; a gate must not be laxer than the
            // claim it licenses.
            let ok = verdict == Verdict::Comparable
                && gap.is_some_and(|g| match row.bar {
                    Bar::AtLeast => g >= 0.0,
                    Bar::GreaterThan => g > 0.0,
                });
            if !ok {
                gated_failures.push(gate_failure(row, &verdict, gap, ours_converted));
            }
        }
        rows.push(StandingRow {
            registry_id: row.id.clone(),
            metric: row.metric.clone(),
            system: row.system.clone(),
            theirs: row.value,
            ours: ours_converted,
            ours_n: mine.map(|m| m.n),
            gap,
            verdict,
            claim_allowed,
            gate: row.gate,
            run: mine.map(|m| m.run.clone()),
            source_doi: row.source.doi.clone(),
            caveat: row.caveat.clone(),
            detail: mine.map(|m| m.detail.clone()),
            candidates: mine.map(|m| m.candidates.clone()).unwrap_or_default(),
        });
    }

    StandingReport {
        // A standing report with no commit is not reproducible, so a failure
        // to resolve one is reported rather than folded into a plausible
        // looking string. It happens for real: on macOS a background Xcode
        // update resets the licence agreement and every `git` call exits 69.
        commit: crate::manifest::commit_id().unwrap_or_else(|e| {
            eprintln!(
                "warning: cannot resolve the commit ({e:#}); this report cannot be tied to a tree"
            );
            "unknown".into()
        }),
        generated_at: Utc::now(),
        rows,
        gated_failures,
        ours: ours.values().cloned().collect(),
    }
}

fn classify(row: &RegistryRow, mine: Option<&Ours>) -> Verdict {
    if !row.provenance.citeable() {
        return Verdict::NotComparableSource {
            provenance: row.provenance,
        };
    }
    let Some(mine) = mine else {
        return Verdict::MissingArtifact {
            command: metric_def(&row.metric)
                .map(|d| d.command.to_string())
                .unwrap_or_else(|| format!("(no extractor for {})", row.metric)),
        };
    };
    if let Some(detail) = &mine.incomplete {
        return Verdict::IncompleteArtifact {
            detail: detail.clone(),
        };
    }
    // Before the population check, and before the judge and backbone
    // caveats: those describe how our number compares to theirs, and this
    // says we do not know what our number is a measurement *of*. A gap
    // computed across an unrecoverable operating point is a number with no
    // referent, which is the failure mode M22 found after six milestones.
    if let Some(detail) = stale_note(&mine.unrecorded) {
        return Verdict::StaleConfig { detail };
    }
    if row.population_comparable && mine.n != row.n {
        return Verdict::NotComparableSubset {
            ours: mine.n,
            theirs: row.n,
        };
    }
    // The metric id's own promise, checked after the paper's population so
    // the sharper message wins when both apply. It is the only check at all
    // when `population_comparable` is false: `longmemeval_s.judge_score.n500`
    // hardcodes 500 while `n` is whatever the run produced, so a `--limit 50`
    // artifact would otherwise be published under a `.n500` label with no
    // trace of the missing 450.
    if let Some(nominal) = metric_def(&row.metric).and_then(|d| d.nominal_n) {
        if mine.n != nominal {
            return Verdict::IncompleteArtifact {
                detail: format!(
                    "metric id advertises n={nominal}, artifact has n={}",
                    mine.n
                ),
            };
        }
    }
    if mine.judge_class != row.judge_class {
        return Verdict::CaveatJudge {
            ours: mine.judge_class,
            theirs: row.judge_class,
        };
    }
    if mine.backbone_class != row.backbone_class {
        return Verdict::CaveatBackbone {
            ours: mine.backbone_class,
            theirs: row.backbone_class,
        };
    }
    Verdict::Comparable
}

/// Our value in the registry row's unit.
///
/// Percent and fraction convert both ways, because a registry row records the
/// number *as the paper printed it* — HiGMem prints 0.78 and MemPro prints
/// 84.93 for quantities on the same scale, and normalising either one on the
/// way in would put a number in the table that appears in no paper. Any other
/// pairing (seconds against percent) is a category error in the extractor
/// table rather than a data error, so it panics rather than scaling something
/// uninterpretable.
fn converted(mine: &Ours, unit: Unit) -> Result<f64, String> {
    match (mine.unit, unit) {
        (a, b) if a == b => Ok(mine.value),
        (Unit::FractionZeroOne, Unit::PctZeroHundred) => Ok(mine.value * 100.0),
        (Unit::PctZeroHundred, Unit::FractionZeroOne) => Ok(mine.value / 100.0),
        (a, b) => Err(format!(
            "metric {}: cannot compare {} against {}",
            mine.metric,
            a.slug(),
            b.slug()
        )),
    }
}

/// Positive means we are ahead, whichever way the metric points.
fn gap(row: &RegistryRow, mine: &Ours) -> Result<f64, String> {
    let ours = converted(mine, row.unit)?;
    Ok(match row.direction {
        Direction::HigherIsBetter => ours - row.value,
        Direction::LowerIsBetter => row.value - ours,
    })
}

fn gate_failure(
    row: &RegistryRow,
    verdict: &Verdict,
    gap: Option<f64>,
    ours: Option<f64>,
) -> String {
    let why = match verdict {
        Verdict::NotComparableSource { provenance } => format!(
            "unverifiable source ({}) — a gate cannot be satisfied by a number with no protocol",
            prov_slug(*provenance)
        ),
        Verdict::MissingArtifact { command } => format!("no artifact; run `{command}`"),
        Verdict::IncompleteArtifact { detail } => format!("artifact incomplete ({detail})"),
        Verdict::StaleConfig { detail } => {
            format!("artifact predates today's operating point ({detail})")
        }
        Verdict::NotComparableSubset { ours, theirs } => {
            format!("different population (ours n={ours}, theirs n={theirs})")
        }
        _ => match (gap, ours) {
            (Some(g), Some(_)) if g == 0.0 && row.bar == Bar::GreaterThan => format!(
                "tied at {:.2}; this gate needs a strict improvement, and a tie is what a \
                 dominated submission scores",
                row.value
            ),
            (Some(g), Some(o)) => format!(
                "behind by {:.2} (ours {o:.2}, gate needs {} {:.2})",
                -g,
                match row.bar {
                    Bar::AtLeast => ">=",
                    Bar::GreaterThan => ">",
                },
                row.value
            ),
            _ => "no gap computed".into(),
        },
    };
    format!("{} [{}] {}", row.id, row.metric, why)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

pub fn render_markdown(report: &StandingReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# `myelin-eval standing` — commit `{}`, generated {}\n\n",
        report.commit,
        report.generated_at.format("%Y-%m-%dT%H:%M:%SZ")
    ));
    s.push_str("| metric | system | theirs | ours | gap | verdict | claim | run |\n");
    s.push_str("|---|---|---|---|---|---|---|---|\n");
    for row in &report.rows {
        s.push_str(&format!(
            "| {} | {}{} | {:.2} | {} | {} | {} | {} | {} |\n",
            row.metric,
            row.system,
            if row.gate { " **(gate)**" } else { "" },
            row.theirs,
            row.ours
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "—".into()),
            row.gap
                .map(|v| format!("{v:+.2}"))
                .unwrap_or_else(|| "—".into()),
            row.verdict.slug(),
            if row.claim_allowed { "yes" } else { "no" },
            row.run
                .as_ref()
                .map(|p| format!("`{}`", p.display()))
                .unwrap_or_else(|| "—".into()),
        ));
    }

    if !report.gated_failures.is_empty() {
        s.push_str(&format!(
            "\n## Unsupported gates ({})\n\n",
            report.gated_failures.len()
        ));
        for f in &report.gated_failures {
            s.push_str(&format!("- {f}\n"));
        }
    }

    let caveats: Vec<&StandingRow> = report.rows.iter().filter(|r| r.caveat.is_some()).collect();
    if !caveats.is_empty() {
        s.push_str("\n## Protocol caveats\n\n");
        for row in caveats {
            s.push_str(&format!(
                "- `{}` — {}\n",
                row.registry_id,
                row.caveat.as_deref().unwrap_or("")
            ));
        }
    }

    s.push_str("\n## How our side was computed\n\n");
    s.push_str(
        "Values are in the extractor's own unit, which is the one the run \
         artifact carries; the comparison table above converts to each \
         registry row's unit. `claimed by` names the registry rows a metric \
         is compared against — `none` is something we measured that nobody \
         published.\n\n",
    );
    s.push_str(
        "| metric | value | n | claimed by | detail | candidates |\n|---|---|---|---|---|---|\n",
    );
    for o in &report.ours {
        let claimants: Vec<&str> = report
            .rows
            .iter()
            .filter(|r| r.metric == o.metric)
            .map(|r| r.registry_id.as_str())
            .collect();
        s.push_str(&format!(
            "| {} | {} | {} | {} | {}{} | {} |\n",
            o.metric,
            format_value(o),
            o.n,
            if claimants.is_empty() {
                "none".to_string()
            } else {
                claimants.len().to_string()
            },
            o.detail,
            o.incomplete
                .as_deref()
                .map(|d| format!(" — INCOMPLETE: {d}"))
                .unwrap_or_default(),
            candidate_list(&o.candidates),
        ));
    }
    s
}

/// Seconds keep three decimals; scores two. A latency printed to two
/// decimals hides the difference between 1.83 s and 1.834 s, which is the
/// whole content of a latency claim.
fn format_value(o: &Ours) -> String {
    match o.unit {
        Unit::Seconds => format!("{:.3} s", o.value),
        Unit::FractionZeroOne => format!("{:.4}", o.value),
        Unit::PctZeroHundred => format!("{:.2}", o.value),
    }
}

fn candidate_list(candidates: &[Candidate]) -> String {
    candidates
        .iter()
        .map(|c| format!("`{}` {:.2}", stem(&c.run), c.value))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The stdout table. Same columns as the markdown, sized for a terminal.
pub fn print(report: &StandingReport) {
    println!(
        "standing  commit {}  {} rows  {} unsupported gate(s)",
        report.commit,
        report.rows.len(),
        report.gated_failures.len()
    );
    println!(
        "  {:<44} {:<16} {:>8} {:>8} {:>8}  verdict",
        "metric", "system", "theirs", "ours", "gap"
    );
    for row in &report.rows {
        println!(
            "  {:<44} {:<16} {:>8.2} {:>8} {:>8}  {}{}",
            row.metric,
            truncate(&row.system, 16),
            row.theirs,
            row.ours
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "-".into()),
            row.gap
                .map(|v| format!("{v:+.2}"))
                .unwrap_or_else(|| "-".into()),
            row.verdict.slug(),
            if row.gate { "  [GATE]" } else { "" },
        );
    }
    if !report.gated_failures.is_empty() {
        println!("\nunsupported gates:");
        for f in &report.gated_failures {
            println!("  - {f}");
        }
    }
    let unclaimed: Vec<&Ours> = report
        .ours
        .iter()
        .filter(|o| !report.rows.iter().any(|r| r.metric == o.metric))
        .collect();
    if !unclaimed.is_empty() {
        println!("\nmeasured, nobody published:");
        for o in unclaimed {
            println!(
                "  {:<44} {:>10} n={:<5} {}",
                o.metric,
                format_value(o),
                o.n,
                o.detail
            );
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n - 1).chain(std::iter::once('…')).collect()
    }
}

/// `standing`'s whole command: load, collect, compare, write, and fail the
/// build when a gate is unsupported.
pub fn run(
    registry: &Path,
    runs: &Path,
    out: &Path,
    gate: bool,
    python: &str,
) -> Result<StandingReport> {
    let reg = load_registry(registry)?;
    let ours = collect(runs, python)?;
    let report = compare(&reg, &ours);
    std::fs::create_dir_all(out).with_context(|| format!("mkdir {}", out.display()))?;
    let json_path = out.join("standing.json");
    let mut json = serde_json::to_string_pretty(&report)?;
    json.push('\n');
    std::fs::write(&json_path, json).with_context(|| format!("write {}", json_path.display()))?;
    let md_path = out.join("standing.md");
    std::fs::write(&md_path, render_markdown(&report))
        .with_context(|| format!("write {}", md_path.display()))?;
    print(&report);
    println!("\nwrote {} and {}", json_path.display(), md_path.display());
    if gate && !report.gated_failures.is_empty() {
        bail!(
            "{} gate row(s) unsupported; see {}",
            report.gated_failures.len(),
            md_path.display()
        );
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> Source {
        Source {
            doi: "10.48550/arXiv.2606.00619".into(),
            stem: "10.48550_arxiv.2606.00619".into(),
            locator: "Table 1".into(),
            lines: "L190".into(),
            quote: "MemPro-15 … 84.93".into(),
        }
    }

    fn row(metric: &str, value: f64, n: usize) -> RegistryRow {
        RegistryRow {
            id: format!("test.{metric}"),
            metric: metric.into(),
            benchmark: "locomo".into(),
            system: "Test".into(),
            value,
            unit: Unit::PctZeroHundred,
            direction: Direction::HigherIsBetter,
            n,
            backbone: "gpt-4o-mini".into(),
            backbone_class: Class::FrontierApi,
            judge_class: JudgeClass::FrontierApi,
            provenance: Provenance::Paper,
            gate: false,
            bar: Bar::AtLeast,
            caveat: None,
            population_comparable: true,
            source: source(),
        }
    }

    fn mine(metric: &str, value: f64, n: usize) -> Ours {
        Ours {
            metric: metric.into(),
            value,
            unit: Unit::PctZeroHundred,
            n,
            judge_class: JudgeClass::FrontierApi,
            backbone_class: Class::FrontierApi,
            run: PathBuf::from("runs/test"),
            detail: "fixture".into(),
            incomplete: None,
            arm: false,
            unrecorded: Vec::new(),
            candidates: Vec::new(),
        }
    }

    fn one(reg: RegistryRow, ours: Vec<Ours>) -> StandingRow {
        let map: BTreeMap<String, Ours> = ours.into_iter().map(|o| (o.metric.clone(), o)).collect();
        let report = compare(
            &Registry {
                schema: 1,
                rows: vec![reg],
            },
            &map,
        );
        report.rows.into_iter().next().unwrap()
    }

    /// `standing` answers "where do we stand", and we stand where the
    /// defaults put us. M21 produced `runs/m21_full_sel`, a complete
    /// 500-question artifact scoring 60.40 against the default
    /// configuration's 56.60 — measuring a switch that ships **off**. A
    /// value-ordered rule published the arm.
    #[test]
    fn a_higher_scoring_arm_never_displaces_the_shipped_configuration() {
        let shipped = |value: f64| Ours {
            run: PathBuf::from("runs/m21_full_base"),
            ..mine("longmemeval_s.judge_score.n500", value, 500)
        };
        let arm = |value: f64| Ours {
            run: PathBuf::from("runs/m21_full_sel"),
            arm: true,
            unrecorded: Vec::new(),
            ..mine("longmemeval_s.judge_score.n500", value, 500)
        };
        let mut rows = [arm(60.40), shipped(56.60)];
        rows.sort_by(|a, b| prefer(a, b, Direction::HigherIsBetter));
        assert_eq!(rows[0].run, PathBuf::from("runs/m21_full_base"));
        assert_eq!(rows[0].value, 56.60);

        // …even when the arm covers a larger population, because a number
        // the defaults do not produce is not where we stand.
        let mut wider = [
            Ours {
                n: 500,
                ..arm(60.40)
            },
            Ours {
                n: 470,
                ..shipped(56.60)
            },
        ];
        wider.sort_by(|a, b| prefer(a, b, Direction::HigherIsBetter));
        assert_eq!(wider[0].run, PathBuf::from("runs/m21_full_base"));

        // But an incomplete default never displaces a complete arm: a
        // partial artifact is not a configuration, it is a broken run.
        let mut partial = [
            arm(60.40),
            Ours {
                incomplete: Some("300/500 judged".into()),
                ..shipped(56.60)
            },
        ];
        partial.sort_by(|a, b| prefer(a, b, Direction::HigherIsBetter));
        assert_eq!(partial[0].run, PathBuf::from("runs/m21_full_sel"));
    }

    #[test]
    fn a_vendor_number_can_never_satisfy_a_gate() {
        let mut reg = row("locomo.judge_score.n1540", 92.5, 1540);
        reg.provenance = Provenance::Vendor;
        reg.gate = true;
        let map: BTreeMap<String, Ours> = [(
            "locomo.judge_score.n1540".to_string(),
            mine("locomo.judge_score.n1540", 99.0, 1540),
        )]
        .into();
        let report = compare(
            &Registry {
                schema: 1,
                rows: vec![reg],
            },
            &map,
        );
        assert_eq!(
            report.rows[0].verdict,
            Verdict::NotComparableSource {
                provenance: Provenance::Vendor
            }
        );
        // Even though our fixture number is higher, the gate stays unsupported.
        assert!(!report.rows[0].claim_allowed);
        assert_eq!(report.gated_failures.len(), 1);
        assert!(
            report.gated_failures[0].contains("unverifiable source"),
            "{:?}",
            report.gated_failures
        );
    }

    #[test]
    fn the_locomo_1986_landmine_is_not_a_gap() {
        let got = one(
            row("locomo.judge_score.n1540", 84.93, 1540),
            vec![mine("locomo.judge_score.n1540", 70.0, 1986)],
        );
        assert_eq!(
            got.verdict,
            Verdict::NotComparableSubset {
                ours: 1986,
                theirs: 1540
            }
        );
        assert_eq!(got.gap, None);
    }

    #[test]
    fn a_fraction_converts_and_lower_is_better_flips_the_sign() {
        let mut reg = row("locomo.token_f1.n1540", 40.0, 1540);
        reg.judge_class = JudgeClass::Deterministic;
        let mut ours = mine("locomo.token_f1.n1540", 0.5307, 1540);
        ours.unit = Unit::FractionZeroOne;
        ours.judge_class = JudgeClass::Deterministic;
        let got = one(reg, vec![ours]);
        assert!((got.ours.unwrap() - 53.07).abs() < 1e-9, "{:?}", got.ours);
        assert!((got.gap.unwrap() - 13.07).abs() < 1e-9, "{:?}", got.gap);
        assert!(got.claim_allowed);

        // Same numbers, lower-is-better: ahead becomes behind.
        let mut reg = row("minja.asr.k6_prepopulated", 76.8, 40);
        reg.direction = Direction::LowerIsBetter;
        reg.judge_class = JudgeClass::NoJudge;
        let mut ours = mine("minja.asr.k6_prepopulated", 77.5, 40);
        ours.judge_class = JudgeClass::NoJudge;
        let got = one(reg, vec![ours]);
        assert!((got.gap.unwrap() + 0.7).abs() < 1e-9, "{:?}", got.gap);
        assert!(!got.claim_allowed);
    }

    #[test]
    fn a_strict_bar_rejects_a_tie_and_an_inclusive_one_accepts_it() {
        let mut reg = row("lme_v2_small.lafs_gain.small", 0.0, 451);
        reg.gate = true;
        reg.bar = Bar::GreaterThan;
        reg.judge_class = JudgeClass::OpenWeightsLocal;
        reg.backbone_class = Class::OpenWeights;
        reg.provenance = Provenance::Leaderboard;
        let mut ours = mine("lme_v2_small.lafs_gain.small", 0.0, 451);
        ours.judge_class = JudgeClass::OpenWeightsLocal;
        ours.backbone_class = Class::OpenWeights;

        let map: BTreeMap<String, Ours> = [(ours.metric.clone(), ours.clone())].into();
        let strict = compare(
            &Registry {
                schema: 1,
                rows: vec![reg.clone()],
            },
            &map,
        );
        assert_eq!(
            strict.gated_failures.len(),
            1,
            "{:?}",
            strict.gated_failures
        );

        reg.bar = Bar::AtLeast;
        let inclusive = compare(
            &Registry {
                schema: 1,
                rows: vec![reg],
            },
            &map,
        );
        assert!(
            inclusive.gated_failures.is_empty(),
            "{:?}",
            inclusive.gated_failures
        );
    }

    /// The decline rule, on a real `bench` artifact shape.
    fn locomo_fixture(dir: &Path, judged_all: bool, decline_the_gap: bool) {
        std::fs::create_dir_all(dir).unwrap();
        let mut rows = String::new();
        let mut verdicts = BTreeMap::new();
        for i in 0..1540 {
            let last = i == 1539;
            let response = if last && decline_the_gap {
                "I don't know."
            } else {
                "an answer"
            };
            rows.push_str(&format!(
                "{}\n",
                serde_json::json!({
                    "question_id": format!("q{i}"),
                    "tenant": "t",
                    "category": 1 + (i % 4) as u8,
                    "question_text": "q",
                    "answer_gold": "a",
                    "response_raw": response,
                    "score": 1.0,
                    "exact_match": 1.0,
                    "is_abstention_problem": false,
                    "retrieved_items": 6,
                    "memory_query_duration_seconds": 0.1
                })
            ));
            if !last || judged_all {
                verdicts.insert(format!("q{i}"), 1u8);
            }
        }
        std::fs::write(dir.join("per_question.jsonl"), rows).unwrap();
        std::fs::write(
            dir.join("judge_verdicts.json"),
            serde_json::to_string(&JudgeFile {
                model: "qwen3.5-9b".into(),
                verdicts,
            })
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("aggregated_metrics.json"),
            serde_json::to_string(&serde_json::json!({
                "corpus": "locomo",
                "collection": "myelin_locomo",
                "mode": "recall",
                "k": 6,
                "max_steps": 2,
                "questions": 1540,
                "f1_answerable": 0.53,
                "em_answerable": 0.28,
                "abstention_accuracy": 0.70,
                "by_category": [],
                "query_p50_seconds": 0.25,
                "query_avg_seconds": 0.27
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn an_unjudged_answered_row_is_incomplete_but_a_decline_is_a_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        let missing = runs.join("locomo_missing");
        locomo_fixture(&missing, false, false);
        let ours = collect(&runs, "/nonexistent/python").unwrap();
        let judged = &ours["locomo.judge_score.n1540"];
        assert_eq!(
            judged.incomplete.as_deref(),
            Some("1539/1540 judged"),
            "{:?}",
            judged
        );
        let got = one(
            row("locomo.judge_score.n1540", 84.93, 1540),
            vec![judged.clone()],
        );
        assert_eq!(
            got.verdict,
            Verdict::IncompleteArtifact {
                detail: "1539/1540 judged".into()
            }
        );
        assert_eq!(got.gap, None);

        // Same hole, but the unjudged row declined: `judge` skips declines by
        // design, so coverage is complete and the value counts it as wrong.
        let declined = runs.join("locomo_declined");
        locomo_fixture(&declined, false, true);
        let ours = collect(&runs, "/nonexistent/python").unwrap();
        let judged = &ours["locomo.judge_score.n1540"];
        assert_eq!(judged.incomplete, None, "{judged:?}");
        assert!(
            (judged.value - 1539.0 / 1540.0 * 100.0).abs() < 1e-9,
            "{judged:?}"
        );
        assert_eq!(judged.candidates.len(), 2, "both fixtures are candidates");
    }

    #[test]
    fn the_two_artifact_shapes_are_discriminated_by_their_own_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        locomo_fixture(&runs.join("bench_shape"), true, false);

        let harness = runs.join("harness_shape");
        std::fs::create_dir_all(harness.join("runtime_inputs")).unwrap();
        std::fs::write(
            harness.join("aggregated_metrics.json"),
            serde_json::json!({
                "overall": {"overall_full_set": 0.45, "count_all_questions": 240},
                "memory_query": {"avg_seconds": 11.06}
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            harness.join("run_args.json"),
            serde_json::json!({"domain": "web", "evaluator_model": "Qwen/Qwen3.5-9B"}).to_string(),
        )
        .unwrap();
        std::fs::write(
            harness.join("runtime_inputs/memory_config.json"),
            serde_json::json!({"memory_params": {"k": 25, "mode": "investigate"}}).to_string(),
        )
        .unwrap();

        // A directory with neither key is skipped, not misread.
        let junk = runs.join("junk");
        std::fs::create_dir_all(&junk).unwrap();
        std::fs::write(junk.join("aggregated_metrics.json"), "{\"nope\": 1}").unwrap();

        let ours = collect(&runs, "/nonexistent/python").unwrap();
        assert_eq!(ours["locomo.judge_score.n1540"].n, 1540);
        assert_eq!(ours["lme_v2_small.overall_full_set.web"].n, 240);
        assert!(
            (ours["lme_v2_small.overall_full_set.web"].value - 45.0).abs() < 1e-9,
            "{:?}",
            ours["lme_v2_small.overall_full_set.web"]
        );
        assert_eq!(
            ours["lme_v2_small.overall_full_set.web"].judge_class,
            JudgeClass::OpenWeightsLocal
        );
        // One domain alone supplies no combined metric and no LAFS point.
        assert!(!ours.contains_key("lme_v2_small.overall_full_set.combined"));
        assert!(!ours.contains_key("lme_v2_small.lafs_gain.small"));
    }

    #[test]
    fn an_unknown_metric_id_fails_the_registry_rather_than_being_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("registry.json");
        let mut reg = Registry {
            schema: 1,
            rows: vec![row("locomo.judge_score.n1540", 84.93, 1540)],
        };
        reg.rows[0].metric = "locomo.vibes".into();
        std::fs::write(&path, serde_json::to_string(&reg).unwrap()).unwrap();
        let err = load_registry(&path).unwrap_err().to_string();
        assert!(err.contains("locomo.vibes"), "{err}");
    }

    /// A stratum arm must not publish a metric whose population it never
    /// touched: `abstention_accuracy` over zero adversarial rows is 0.00 by
    /// arithmetic and a lie by measurement.
    #[test]
    fn a_stratum_run_publishes_no_empty_population_metric() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        let dir = runs.join("locomo_recall_rdates_cat2_temporal");
        std::fs::create_dir_all(&dir).unwrap();
        let mut rows = String::new();
        for i in 0..321 {
            rows.push_str(&format!(
                "{}\n",
                serde_json::json!({
                    "question_id": format!("q{i}"),
                    "tenant": "t",
                    "category": 2,
                    "question_text": "when?",
                    "answer_gold": "June 2023",
                    "response_raw": "2023-06-15",
                    "score": 1.0,
                    "exact_match": 1.0,
                    "is_abstention_problem": false,
                    "retrieved_items": 6,
                    "memory_query_duration_seconds": 0.1
                })
            ));
        }
        std::fs::write(dir.join("per_question.jsonl"), rows).unwrap();
        std::fs::write(
            dir.join("aggregated_metrics.json"),
            serde_json::json!({
                "corpus": "locomo",
                "collection": "myelin_locomo",
                "mode": "recall",
                "k": 6,
                "max_steps": 2,
                "scorer": "temporal",
                "resolve_dates": true,
                "categories": [2],
                "questions": 321,
                "f1_answerable": 0.2534,
                "em_answerable": 0.22,
                "abstention_accuracy": 0.0,
                "by_category": [],
                "query_p50_seconds": 0.25,
                "query_avg_seconds": 0.27
            })
            .to_string(),
        )
        .unwrap();

        let ours = collect(&runs, "/nonexistent/python").unwrap();
        assert!(
            !ours.contains_key("locomo.abstention_accuracy.n446"),
            "a temporal-only arm touched no adversarial row"
        );
        // The stratum's own number is published with its real `n`, which is
        // what the population rule then rejects as not-comparable rather
        // than letting 321 rows displace the 1,540-row baseline.
        assert_eq!(ours["locomo.temporal.n1540"].n, 321);
    }

    /// An operating point is the set of `memory_params` that change what the
    /// server does per query, and a pair must share all of them. `select` and
    /// `dated` are query-time switches (M22); `pool_rerank`, `premise`, and
    /// `typed_probes` are the M23 ones — without any of them in `PAIR_KEYS` a
    /// probe-tagged or pool-reranking web arm cross-pairs with its plain twin
    /// and `standing` publishes a combined accuracy for a point that never ran:
    /// two arms would yield four pairings instead of two.
    #[test]
    fn a_selection_run_does_not_pair_with_a_plain_one() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        let arm = |name: &str, domain: &str, count: usize, acc: f64, select: bool| {
            let dir = runs.join(name);
            std::fs::create_dir_all(dir.join("runtime_inputs")).unwrap();
            std::fs::write(
                dir.join("aggregated_metrics.json"),
                serde_json::json!({
                    "overall": {"overall_full_set": acc, "count_all_questions": count},
                    "memory_query": {"avg_seconds": 11.0}
                })
                .to_string(),
            )
            .unwrap();
            std::fs::write(
                dir.join("run_args.json"),
                serde_json::json!({"domain": domain, "evaluator_model": "Qwen/Qwen3.5-9B"})
                    .to_string(),
            )
            .unwrap();
            std::fs::write(
                dir.join("runtime_inputs/memory_config.json"),
                serde_json::json!({"memory_params": {
                    "mode": "investigate", "k": 25, "select": select, "dated": false
                }})
                .to_string(),
            )
            .unwrap();
        };
        arm("m22_base_web", "web", 240, 0.40, false);
        arm("m22_base_ent", "enterprise", 211, 0.40, false);
        arm("m22_sel_web", "web", 240, 0.55, true);
        arm("m22_sel_ent", "enterprise", 211, 0.55, true);

        let ours = collect(&runs, "/nonexistent/python").unwrap();
        let combined = &ours["lme_v2_small.overall_full_set.combined"];
        assert_eq!(
            combined.candidates.len(),
            2,
            "two arms are two pairs; four means the switch does not split \
             operating points: {:?}",
            combined
                .candidates
                .iter()
                .map(|c| (c.run.display().to_string(), c.value))
                .collect::<Vec<_>>()
        );
        assert_eq!(combined.n, 451, "a pair is the tier's full population");
        // The winner is the selection arm, and its detail names only its own
        // two directories — never a plain run.
        assert!(
            combined.detail.contains("m22_sel_web") && combined.detail.contains("m22_sel_ent"),
            "{}",
            combined.detail
        );
        assert!((combined.value - 55.0).abs() < 1e-9, "{combined:?}");
    }

    /// Same contract, M23 Phase A: a `pool_rerank` web arm must not cross-pair
    /// with a plain enterprise run, and an arm that omits the key entirely
    /// (pre-M23 harness) pairs only with another such run.
    #[test]
    fn a_pool_rerank_run_does_not_pair_with_a_plain_one() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        let arm = |name: &str, domain: &str, count: usize, acc: f64, rerank: serde_json::Value| {
            let dir = runs.join(name);
            std::fs::create_dir_all(dir.join("runtime_inputs")).unwrap();
            std::fs::write(
                dir.join("aggregated_metrics.json"),
                serde_json::json!({
                    "overall": {"overall_full_set": acc, "count_all_questions": count},
                    "memory_query": {"avg_seconds": 11.0}
                })
                .to_string(),
            )
            .unwrap();
            std::fs::write(
                dir.join("run_args.json"),
                serde_json::json!({"domain": domain, "evaluator_model": "Qwen/Qwen3.5-9B"})
                    .to_string(),
            )
            .unwrap();
            std::fs::write(
                dir.join("runtime_inputs/memory_config.json"),
                serde_json::json!({"memory_params": {
                    "mode": "investigate", "k": 25, "select": false, "dated": false,
                    "pool_rerank": rerank
                }})
                .to_string(),
            )
            .unwrap();
        };
        arm("m23_base_web", "web", 240, 0.39, serde_json::Value::Null);
        arm(
            "m23_base_ent",
            "enterprise",
            211,
            0.39,
            serde_json::Value::Null,
        );
        arm("m23_rr_web", "web", 240, 0.50, serde_json::json!(true));
        arm(
            "m23_rr_ent",
            "enterprise",
            211,
            0.50,
            serde_json::json!(true),
        );

        let ours = collect(&runs, "/nonexistent/python").unwrap();
        let combined = &ours["lme_v2_small.overall_full_set.combined"];
        assert_eq!(
            combined.candidates.len(),
            2,
            "explicit true must not pair with absent: {:?}",
            combined
                .candidates
                .iter()
                .map(|c| c.run.display().to_string())
                .collect::<Vec<_>>()
        );
        assert!((combined.value - 50.0).abs() < 1e-9, "{combined:?}");
    }

    /// Every artifact written before M22 lacks both keys, so both read `null`
    /// and the pairings those runs already have must not move.
    #[test]
    fn runs_predating_the_new_keys_still_pair_with_each_other() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        for (name, domain, count) in [
            ("myelin_inv2_web_small", "web", 240usize),
            ("myelin_inv2_enterprise_small", "enterprise", 211usize),
        ] {
            let dir = runs.join(name);
            std::fs::create_dir_all(dir.join("runtime_inputs")).unwrap();
            std::fs::write(
                dir.join("aggregated_metrics.json"),
                serde_json::json!({
                    "overall": {"overall_full_set": 0.3991, "count_all_questions": count},
                    "memory_query": {"avg_seconds": 11.5}
                })
                .to_string(),
            )
            .unwrap();
            std::fs::write(
                dir.join("run_args.json"),
                serde_json::json!({"domain": domain, "evaluator_model": "Qwen/Qwen3.5-9B"})
                    .to_string(),
            )
            .unwrap();
            std::fs::write(
                dir.join("runtime_inputs/memory_config.json"),
                serde_json::json!({"memory_params": {"mode": "investigate", "k": 25}}).to_string(),
            )
            .unwrap();
        }
        let ours = collect(&runs, "/nonexistent/python").unwrap();
        let combined = &ours["lme_v2_small.overall_full_set.combined"];
        assert_eq!(combined.n, 451);
        assert!((combined.value - 39.91).abs() < 1e-9, "{combined:?}");
    }

    /// A `--limit` pilot is not a submission. Its fingerprint is identical to
    /// the full arm's — `--limit` is not an operating point — so without a
    /// population guard a 40-question web pilot pairs with a 211-question
    /// enterprise arm and publishes a "combined" accuracy over 251 questions
    /// against a bar defined on 451. M22's three pilots did exactly that.
    #[test]
    fn a_limit_pilot_never_pairs_into_a_tier_population() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        let arm = |name: &str, domain: &str, count: usize, acc: f64| {
            let dir = runs.join(name);
            std::fs::create_dir_all(dir.join("runtime_inputs")).unwrap();
            std::fs::write(
                dir.join("aggregated_metrics.json"),
                serde_json::json!({
                    "overall": {"overall_full_set": acc, "count_all_questions": count},
                    "memory_query": {"avg_seconds": 11.0}
                })
                .to_string(),
            )
            .unwrap();
            std::fs::write(
                dir.join("run_args.json"),
                serde_json::json!({"domain": domain, "evaluator_model": "Qwen/Qwen3.5-9B"})
                    .to_string(),
            )
            .unwrap();
            std::fs::write(
                dir.join("runtime_inputs/memory_config.json"),
                serde_json::json!({"memory_params": {"mode": "investigate", "k": 25}}).to_string(),
            )
            .unwrap();
        };
        // One real pair, plus a pilot that scores far higher than either.
        arm("full_web", "web", 240, 0.40);
        arm("full_ent", "enterprise", 211, 0.40);
        arm("pilot_web", "web", 40, 0.95);

        let ours = collect(&runs, "/nonexistent/python").unwrap();
        let combined = &ours["lme_v2_small.overall_full_set.combined"];
        assert_eq!(combined.n, 451, "only the tier's own population may pair");
        assert!(
            (combined.value - 40.0).abs() < 1e-9,
            "the 0.95 pilot must not win: {combined:?}"
        );
        assert_eq!(
            combined.candidates.len(),
            1,
            "one pair, not two: {:?}",
            combined
                .candidates
                .iter()
                .map(|c| (c.run.display().to_string(), c.value))
                .collect::<Vec<_>>()
        );
    }

    // ---- config drift: the M22 defect, pinned (M23) ----

    /// Build a minimal LME-V2 harness run directory.
    ///
    /// `params` is spliced in verbatim so a test can express "this key is
    /// absent" — which is the whole signal — rather than only "this key is
    /// false".
    fn harness_run(runs: &Path, name: &str, domain: &str, count: usize, acc: f64, params: Value) {
        let dir = runs.join(name);
        std::fs::create_dir_all(dir.join("runtime_inputs")).unwrap();
        std::fs::write(
            dir.join("aggregated_metrics.json"),
            serde_json::json!({
                "overall": {"overall_full_set": acc, "count_all_questions": count},
                "memory_query": {"avg_seconds": 11.0}
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("run_args.json"),
            serde_json::json!({"domain": domain, "evaluator_model": "Qwen/Qwen3.5-9B"}).to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("runtime_inputs/memory_config.json"),
            serde_json::json!({ "memory_params": params }).to_string(),
        )
        .unwrap();
    }

    /// The full operating point, as today's adapter records it.
    fn full_params() -> Value {
        serde_json::json!({
            "mode": "investigate", "k": 25, "budget_tokens": 10000, "max_steps": 2,
            "prefetch_limit": null, "rerank_depth": null,
            "select": false, "dated": true,
            "pool_rerank": false, "premise": false, "typed_probes": false,
            "decompose": null
        })
    }

    /// **The M22 defect.** An old artifact that records none of the
    /// operating-point keys scores *higher* than a fresh one, and must
    /// still lose — because nobody can check what produced it.
    ///
    /// Before M23 this selection was made on value alone among non-arms,
    /// and `standing` published 39.91 from a pre-M19 run for six
    /// milestones while the shipped defaults were worth 36.59.
    #[test]
    fn a_run_that_does_not_record_its_operating_point_never_displaces_one_that_does() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        // The old, better-scoring, unrecoverable pair.
        let old = serde_json::json!({"mode": "investigate", "k": 25, "max_steps": 2});
        harness_run(&runs, "old_web", "web", 240, 0.60, old.clone());
        harness_run(&runs, "old_ent", "enterprise", 211, 0.60, old);
        // Today's pair, scoring worse.
        harness_run(&runs, "new_web", "web", 240, 0.40, full_params());
        harness_run(&runs, "new_ent", "enterprise", 211, 0.40, full_params());

        let ours = collect(&runs, "/nonexistent/python").unwrap();
        let combined = &ours["lme_v2_small.overall_full_set.combined"];
        assert!(
            combined.detail.contains("new_web"),
            "the recoverable pair must win even at 40 against 60: {}",
            combined.detail
        );
        assert!((combined.value - 40.0).abs() < 1e-9, "{combined:?}");
        assert!(
            combined.unrecorded.is_empty(),
            "the winning pair records everything: {:?}",
            combined.unrecorded
        );
    }

    /// And when every artifact is partly unrecoverable, the least
    /// unrecoverable one wins — not the highest-scoring one.
    #[test]
    fn among_unrecoverable_artifacts_the_closest_to_today_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        let ancient = serde_json::json!({"mode": "investigate", "k": 25, "max_steps": 2});
        harness_run(&runs, "ancient_web", "web", 240, 0.60, ancient.clone());
        harness_run(&runs, "ancient_ent", "enterprise", 211, 0.60, ancient);
        // Missing only the three newest keys.
        let recent = serde_json::json!({
            "mode": "investigate", "k": 25, "budget_tokens": 10000, "max_steps": 2,
            "prefetch_limit": null, "rerank_depth": null, "select": false, "dated": true
        });
        harness_run(&runs, "recent_web", "web", 240, 0.40, recent.clone());
        harness_run(&runs, "recent_ent", "enterprise", 211, 0.40, recent);

        let ours = collect(&runs, "/nonexistent/python").unwrap();
        let combined = &ours["lme_v2_small.overall_full_set.combined"];
        assert!(
            combined.detail.contains("recent_web"),
            "three missing keys beats seven: {}",
            combined.detail
        );
        assert_eq!(
            combined.unrecorded,
            vec!["pool_rerank", "premise", "typed_probes", "decompose"]
        );
    }

    /// **The arm defect on the LME-V2 path.** `dated` ships **on**, so a
    /// run carrying `dated: false` measures an arm — and M22 measured that
    /// arm as a null. It scored 39.02 against the shipped configuration's
    /// 36.59 and was published, because `harness_metrics` hardcoded
    /// `arm: false` and had done since the path was written.
    #[test]
    fn an_lme_v2_arm_never_displaces_the_shipped_configuration() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        let undated = serde_json::json!({
            "mode": "investigate", "k": 25, "budget_tokens": 10000, "max_steps": 2,
            "prefetch_limit": null, "rerank_depth": null, "select": false, "dated": false,
            "pool_rerank": false, "premise": false, "typed_probes": false,
            "decompose": null
        });
        harness_run(&runs, "nodate_web", "web", 240, 0.39, undated.clone());
        harness_run(&runs, "nodate_ent", "enterprise", 211, 0.39, undated);
        harness_run(&runs, "base_web", "web", 240, 0.3659, full_params());
        harness_run(&runs, "base_ent", "enterprise", 211, 0.3659, full_params());

        let ours = collect(&runs, "/nonexistent/python").unwrap();
        let combined = &ours["lme_v2_small.overall_full_set.combined"];
        assert!(
            combined.detail.contains("base_web"),
            "the shipped configuration is where we stand, even at 36.59 against 39.00: {}",
            combined.detail
        );
        assert!(!combined.arm, "{combined:?}");
        assert_eq!(combined.candidates.len(), 2, "both pairs stay listed");
    }

    /// A stated width is an override of `RetrieveConfig::default()`, so it
    /// is an arm too — the M23 width sweep must not publish itself.
    #[test]
    fn a_stated_width_is_an_arm() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        let mut wide = full_params();
        wide["prefetch_limit"] = serde_json::json!(200);
        harness_run(&runs, "wide_web", "web", 240, 0.50, wide.clone());
        harness_run(&runs, "wide_ent", "enterprise", 211, 0.50, wide);
        harness_run(&runs, "base_web", "web", 240, 0.40, full_params());
        harness_run(&runs, "base_ent", "enterprise", 211, 0.40, full_params());

        let ours = collect(&runs, "/nonexistent/python").unwrap();
        let combined = &ours["lme_v2_small.overall_full_set.combined"];
        assert!(combined.detail.contains("base_web"), "{}", combined.detail);
    }

    /// The bench path has its own era signal: `resolve_dates` and
    /// `timeline` ship on since M19 and are written from the live defaults
    /// on every run since, so an artifact that omits them is pre-M19.
    #[test]
    fn a_bench_artifact_without_the_m19_keys_is_stale() {
        let pre_m19 = serde_json::json!({
            "corpus": "locomo", "collection": "c", "mode": "recall", "k": 6,
            "max_steps": 0, "questions": 10, "f1_answerable": 0.5,
            "em_answerable": 0.1, "abstention_accuracy": 0.7, "by_category": [],
            "query_p50_seconds": 1.0, "query_avg_seconds": 1.0
        })
        .to_string();
        assert_eq!(
            unrecorded_bench_keys(&pre_m19).unwrap(),
            vec!["resolve_dates", "timeline"]
        );

        let mut post = serde_json::from_str::<Value>(&pre_m19).unwrap();
        post["resolve_dates"] = Value::Bool(true);
        post["timeline"] = Value::Bool(true);
        assert!(unrecorded_bench_keys(&post.to_string()).unwrap().is_empty());
    }

    /// A stale row is not comparable to anybody: the verdict fires before
    /// the judge and backbone caveats, so the report says *why* rather than
    /// printing a gap against a number with no referent.
    #[test]
    fn a_stale_row_is_never_claimable() {
        let r = row("locomo.judge_score.n1540", 84.93, 1540);
        let mut stale = mine("locomo.judge_score.n1540", 69.87, 1540);
        stale.unrecorded = vec!["resolve_dates"];
        match classify(&r, Some(&stale)) {
            Verdict::StaleConfig { detail } => {
                assert!(detail.contains("resolve_dates"), "{detail}")
            }
            other => panic!("{other:?}"),
        }
        assert!(!classify(&r, Some(&stale)).quantified());
    }

    /// A bench artifact carrying an M23 switch measures an arm, and an arm
    /// never becomes "where we stand" however well it scores.
    ///
    /// This is [`Ours::arm`]'s original defect — M21's `runs/m21_full_sel`
    /// at 60.40 displacing the shipped 56.60 — re-asserted for the four
    /// switches M23 adds, because the OR chain that detects it is
    /// hand-maintained and a switch left out of it is silently publishable.
    #[test]
    fn a_bench_run_carrying_an_m23_switch_is_an_arm() {
        let base = serde_json::json!({
            "corpus": "locomo", "collection": "c", "mode": "investigate", "k": 6,
            "max_steps": 2, "scorer": "temporal",
            "resolve_dates": true, "timeline": true,
            "questions": 1, "f1_answerable": 0.5, "em_answerable": 0.1,
            "abstention_accuracy": 0.0, "by_category": [],
            "query_p50_seconds": 0.2, "query_avg_seconds": 0.2
        });
        for switch in [
            serde_json::json!({"rerank_pool": true}),
            serde_json::json!({"premise": true}),
            serde_json::json!({"typed_probes": true}),
            serde_json::json!({"untrusted_max": 2}),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join("runs/arm");
            std::fs::create_dir_all(&dir).unwrap();
            let mut agg = base.clone();
            let (key, value) = switch.as_object().unwrap().iter().next().unwrap();
            agg[key] = value.clone();
            std::fs::write(
                dir.join("per_question.jsonl"),
                serde_json::json!({
                    "question_id": "q", "tenant": "t", "category": 2,
                    "question_text": "when?", "answer_gold": "g",
                    "response_raw": "g", "score": 1.0, "exact_match": 1.0,
                    "is_abstention_problem": false, "retrieved_items": 6,
                    "memory_query_duration_seconds": 0.1
                })
                .to_string(),
            )
            .unwrap();
            std::fs::write(dir.join("aggregated_metrics.json"), agg.to_string()).unwrap();

            let ours = collect(&tmp.path().join("runs"), "/nonexistent/python").unwrap();
            assert!(
                ours["locomo.temporal.n1540"].arm,
                "{key} ships off, so a run carrying it is an arm"
            );
            assert!(
                ours["locomo.temporal.n1540"].unrecorded.is_empty(),
                "{key}: recording the M19 keys means the artifact is not stale"
            );
        }
    }
}
