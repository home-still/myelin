//! The progression ratchet — a pinned floor under every metric we publish.
//!
//! `standing` answers "how do we compare to the literature". This answers a
//! question nothing else did: **did we get worse than we already were?**
//!
//! # Why it exists
//!
//! M22 re-measured the shipped defaults on LME-V2-Small and got **36.59**
//! where `standing` had been publishing **39.91** since M16 — a 3.3-point
//! regression that survived six milestones because no arm re-ran the base
//! and no instrument compared us to ourselves. `standing`'s registry is a
//! table of *other people's* numbers; a system can lose three points against
//! its own past while every row in that table stays exactly as red as it was.
//!
//! The ratchet closes that. Every metric worth publishing is pinned in
//! `docs/sota/progression.json`, and a metric that moves the wrong way fails
//! the command.
//!
//! # What may be pinned
//!
//! Only a **quotable** row: complete, not an arm, and carrying no config
//! drift. Pinning an arm is the trap this instrument exists to avoid — a
//! selection arm pinned as the floor makes every honest default run look
//! like a regression, and the fix then looks like "turn the arm on", which
//! is how a measured null becomes a shipped default by accident.
//!
//! # Direction is not cosmetic
//!
//! `minja.asr.*` is lower-is-better and `locomo.judge_score.*` is
//! higher-is-better. A ratchet that compares with `<` in both directions
//! silently inverts on the security metrics — the ones where a regression
//! matters most — so every comparison goes through the metric's own
//! [`Direction`].

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::standing::{collect, metric_direction, Direction, Ours};

/// A pinned floor for one metric.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Pin {
    pub value: f64,
    /// The population the pin was measured over. A later run with a smaller
    /// population is not a comparison, so the ratchet reports it rather than
    /// grading it.
    pub n: usize,
    /// The artifact the pin came from, for the reader who wants to re-derive
    /// it.
    pub run: String,
    /// When it was pinned, so a stale floor is visible as one.
    pub recorded: String,
}

/// The checked-in floor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub schema: u32,
    /// Metric id → pin. `BTreeMap` so the file is byte-stable under
    /// `--update` and a diff shows only what moved.
    pub pinned: BTreeMap<String, Pin>,
}

impl Default for Baseline {
    fn default() -> Self {
        Self {
            schema: 1,
            pinned: BTreeMap::new(),
        }
    }
}

/// What happened to one metric between the pin and today.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Movement {
    /// Better than the pin, by the metric's own direction.
    Improved { from: f64, to: f64 },
    /// Equal to the pin, or inside `tolerance`.
    Held { value: f64 },
    /// Worse than the pin by more than `tolerance`. The only status that
    /// fails the command by default.
    Regressed { from: f64, to: f64, by: f64 },
    /// Pinned, but today's best artifact for it is not quotable — an arm, a
    /// partial artifact, or one that does not record its operating point.
    ///
    /// Not a regression: the number may be fine. It is the *evidence* that
    /// is gone, which is the state M22 was in for six milestones, so it is
    /// reported loudly and fails under `--strict`.
    Unverifiable { pinned: f64, why: String },
    /// Quotable today and not pinned. Nothing to compare against; `--update`
    /// adopts it.
    New { value: f64 },
}

impl Movement {
    /// Does this movement fail the command?
    pub fn fails(&self, strict: bool) -> bool {
        match self {
            Movement::Regressed { .. } => true,
            Movement::Unverifiable { .. } => strict,
            _ => false,
        }
    }
}

/// One metric's verdict.
#[derive(Debug, Clone, Serialize)]
pub struct Row {
    pub metric: String,
    pub movement: Movement,
    /// The artifact today's value came from, when there is one.
    pub run: Option<String>,
    pub n: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RatchetReport {
    pub rows: Vec<Row>,
    pub failures: usize,
}

/// Why this row cannot be pinned, or `None` when it can.
///
/// The three disqualifiers are exactly the three `standing` already
/// computes, read here rather than re-derived so the two commands can never
/// disagree about what "our number" means.
fn unquotable(o: &Ours) -> Option<String> {
    if let Some(detail) = &o.incomplete {
        return Some(format!("artifact incomplete ({detail})"));
    }
    if !o.unrecorded.is_empty() {
        return Some(format!(
            "artifact does not record {}",
            o.unrecorded.join(", ")
        ));
    }
    if o.arm {
        return Some(format!(
            "{} measures an arm, not the shipped configuration",
            o.run.display()
        ));
    }
    None
}

/// Is `now` worse than `pinned` by more than `tolerance`, in the metric's
/// own direction? Returns the regression size when it is.
fn regression(metric: &str, pinned: f64, now: f64, tolerance: f64) -> Option<f64> {
    let worse_by = match metric_direction(metric) {
        Direction::HigherIsBetter => pinned - now,
        Direction::LowerIsBetter => now - pinned,
    };
    (worse_by > tolerance).then_some(worse_by)
}

/// Compare today's artifacts against the pinned floor.
pub fn compare(
    baseline: &Baseline,
    ours: &BTreeMap<String, Ours>,
    tolerance: f64,
) -> RatchetReport {
    let mut rows = Vec::new();

    for (metric, o) in ours {
        let quotable = unquotable(o);
        let pin = baseline.pinned.get(metric);
        let movement = match (pin, quotable) {
            (Some(pin), Some(why)) => Movement::Unverifiable {
                pinned: pin.value,
                why,
            },
            (Some(pin), None) => match regression(metric, pin.value, o.value, tolerance) {
                Some(by) => Movement::Regressed {
                    from: pin.value,
                    to: o.value,
                    by,
                },
                None if o.value == pin.value => Movement::Held { value: o.value },
                None => {
                    // Inside tolerance but not equal is still "held": the
                    // pin did not move and neither did the claim.
                    let improved = match metric_direction(metric) {
                        Direction::HigherIsBetter => o.value > pin.value,
                        Direction::LowerIsBetter => o.value < pin.value,
                    };
                    if improved {
                        Movement::Improved {
                            from: pin.value,
                            to: o.value,
                        }
                    } else {
                        Movement::Held { value: o.value }
                    }
                }
            },
            (None, None) => Movement::New { value: o.value },
            // Not pinned and not quotable: nothing to say, and saying it
            // would bury the rows that matter.
            (None, Some(_)) => continue,
        };
        rows.push(Row {
            metric: metric.clone(),
            movement,
            run: Some(o.run.display().to_string()),
            n: Some(o.n),
        });
    }

    // A pinned metric with no artifact at all today. Deleting a run
    // directory must not silently retire its floor.
    for (metric, pin) in &baseline.pinned {
        if !ours.contains_key(metric) {
            rows.push(Row {
                metric: metric.clone(),
                movement: Movement::Unverifiable {
                    pinned: pin.value,
                    why: format!(
                        "no artifact on disk supports this metric (pinned from {})",
                        pin.run
                    ),
                },
                run: None,
                n: None,
            });
        }
    }

    rows.sort_by(|a, b| a.metric.cmp(&b.metric));
    RatchetReport { failures: 0, rows }
}

/// Raise the floor to today's quotable values. Never lowers one.
///
/// The ratchet only ever moves in the improving direction, which is what
/// makes it a ratchet: accepting a regression has to be a deliberate edit to
/// the checked-in file, visible in review, and not a flag someone reached
/// for to make the build green.
pub fn raise(baseline: &mut Baseline, ours: &BTreeMap<String, Ours>, today: &str) -> Vec<String> {
    let mut moved = Vec::new();
    for (metric, o) in ours {
        if unquotable(o).is_some() {
            continue;
        }
        let better = match baseline.pinned.get(metric) {
            None => true,
            Some(pin) => match metric_direction(metric) {
                Direction::HigherIsBetter => o.value > pin.value,
                Direction::LowerIsBetter => o.value < pin.value,
            },
        };
        if better {
            let from = baseline.pinned.get(metric).map(|p| p.value);
            baseline.pinned.insert(
                metric.clone(),
                Pin {
                    value: o.value,
                    n: o.n,
                    run: o.run.display().to_string(),
                    recorded: today.to_string(),
                },
            );
            moved.push(match from {
                Some(f) => format!("{metric}: {f:.2} -> {:.2}", o.value),
                None => format!("{metric}: pinned at {:.2}", o.value),
            });
        }
    }
    moved
}

pub fn load(path: &Path) -> Result<Baseline> {
    if !path.exists() {
        return Ok(Baseline::default());
    }
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let baseline: Baseline =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    anyhow::ensure!(
        baseline.schema == 1,
        "{} declares schema {} but this build reads schema 1",
        path.display(),
        baseline.schema
    );
    Ok(baseline)
}

pub fn save(path: &Path, baseline: &Baseline) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    let mut text = serde_json::to_string_pretty(baseline)?;
    text.push('\n');
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

pub fn print(report: &RatchetReport, strict: bool) {
    println!("\n=== myelin-eval ratchet — our numbers against our own floor ===\n");
    println!("{:<46} {:>9} {:>9}  status", "metric", "pinned", "now");
    for row in &report.rows {
        let (pinned, now, status) = match &row.movement {
            Movement::Improved { from, to } => (
                format!("{from:.2}"),
                format!("{to:.2}"),
                "IMPROVED".to_string(),
            ),
            Movement::Held { value } => (
                format!("{value:.2}"),
                format!("{value:.2}"),
                "held".to_string(),
            ),
            Movement::Regressed { from, to, by } => (
                format!("{from:.2}"),
                format!("{to:.2}"),
                format!("REGRESSED by {by:.2}"),
            ),
            Movement::Unverifiable { pinned, why } => (
                format!("{pinned:.2}"),
                "—".to_string(),
                format!("unverifiable: {why}"),
            ),
            Movement::New { value } => ("—".to_string(), format!("{value:.2}"), "new".to_string()),
        };
        println!("{:<46} {pinned:>9} {now:>9}  {status}", row.metric);
    }
    let regressed = report
        .rows
        .iter()
        .filter(|r| matches!(r.movement, Movement::Regressed { .. }))
        .count();
    let unverifiable = report
        .rows
        .iter()
        .filter(|r| matches!(r.movement, Movement::Unverifiable { .. }))
        .count();
    println!(
        "\n{regressed} regressed, {unverifiable} unverifiable{}",
        if strict {
            " (both fail under --strict)"
        } else {
            " (only regressions fail; --strict also fails the unverifiable)"
        }
    );
    if unverifiable > 0 {
        println!(
            "an unverifiable row is not a bad number — it is a number whose operating point \n\
             the artifact no longer pins. Re-measure the base on this code and it clears."
        );
    }
}

/// `ratchet`'s whole command.
pub fn run(
    runs: &Path,
    baseline_path: &Path,
    python: &str,
    update: bool,
    strict: bool,
    tolerance: f64,
) -> Result<RatchetReport> {
    let mut baseline = load(baseline_path)?;
    let ours = collect(runs, python)?;

    if update {
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let moved = raise(&mut baseline, &ours, &today);
        save(baseline_path, &baseline)?;
        if moved.is_empty() {
            println!("floor unchanged: nothing quotable beat its pin");
        } else {
            println!("raised {} pin(s):", moved.len());
            for m in &moved {
                println!("  {m}");
            }
        }
    }

    let mut report = compare(&baseline, &ours, tolerance);
    report.failures = report
        .rows
        .iter()
        .filter(|r| r.movement.fails(strict))
        .count();
    print(&report, strict);

    if report.failures > 0 {
        bail!(
            "{} metric(s) below the pinned floor in {}",
            report.failures,
            baseline_path.display()
        );
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standing::{Class, JudgeClass, Unit};
    use std::path::PathBuf;

    fn ours(metric: &str, value: f64) -> Ours {
        Ours {
            metric: metric.into(),
            value,
            unit: Unit::PctZeroHundred,
            n: 1540,
            judge_class: JudgeClass::OpenWeightsLocal,
            backbone_class: Class::OpenWeights,
            run: PathBuf::from("runs/today"),
            detail: "test".into(),
            incomplete: None,
            arm: false,
            unrecorded: Vec::new(),
            candidates: Vec::new(),
        }
    }

    fn pinned(metric: &str, value: f64) -> Baseline {
        let mut b = Baseline::default();
        b.pinned.insert(
            metric.into(),
            Pin {
                value,
                n: 1540,
                run: "runs/yesterday".into(),
                recorded: "2026-09-01".into(),
            },
        );
        b
    }

    fn map(o: Ours) -> BTreeMap<String, Ours> {
        BTreeMap::from([(o.metric.clone(), o)])
    }

    /// The whole point: a metric that moves the wrong way fails, and the
    /// report says by how much.
    #[test]
    fn a_drop_below_the_floor_is_a_failure() {
        let base = pinned("locomo.judge_score.n1540", 69.87);
        let report = compare(&base, &map(ours("locomo.judge_score.n1540", 66.50)), 0.0);
        assert_eq!(
            report.rows[0].movement,
            Movement::Regressed {
                from: 69.87,
                to: 66.50,
                by: 69.87 - 66.50
            }
        );
        assert!(report.rows[0].movement.fails(false));
    }

    /// Progress is not a failure, and the floor is not raised by comparing.
    #[test]
    fn a_rise_above_the_floor_passes() {
        let base = pinned("locomo.judge_score.n1540", 69.87);
        let report = compare(&base, &map(ours("locomo.judge_score.n1540", 72.10)), 0.0);
        assert_eq!(
            report.rows[0].movement,
            Movement::Improved {
                from: 69.87,
                to: 72.10
            }
        );
        assert!(!report.rows[0].movement.fails(true));
    }

    /// **Direction is read from the metric, not assumed.** ASR is
    /// lower-is-better: 15% against a 12.5% floor is a regression, and the
    /// same arithmetic on a judge score would be a win. This is the test
    /// that would catch a ratchet quietly inverted on the security metrics.
    #[test]
    fn a_lower_is_better_metric_is_not_graded_upside_down() {
        let metric = "minja.asr.k6_prepopulated_defended";
        let base = pinned(metric, 0.125);
        let worse = compare(&base, &map(ours(metric, 0.15)), 0.0);
        assert!(
            matches!(worse.rows[0].movement, Movement::Regressed { .. }),
            "a higher ASR is worse: {:?}",
            worse.rows[0].movement
        );
        let better = compare(&base, &map(ours(metric, 0.08)), 0.0);
        assert!(
            matches!(better.rows[0].movement, Movement::Improved { .. }),
            "a lower ASR is better: {:?}",
            better.rows[0].movement
        );
    }

    /// An arm is never graded against the floor. Without this, M21's
    /// `runs/m21_full_sel` (60.40, a switch that ships off) would pin a
    /// floor the shipped configuration's 56.60 can never clear, and every
    /// honest run afterwards reads as a regression.
    #[test]
    fn an_arm_is_unverifiable_rather_than_a_comparison() {
        let base = pinned("longmemeval_s.judge_score.n500", 56.60);
        let mut arm = ours("longmemeval_s.judge_score.n500", 60.40);
        arm.arm = true;
        arm.n = 500;
        let report = compare(&base, &map(arm), 0.0);
        assert!(
            matches!(report.rows[0].movement, Movement::Unverifiable { .. }),
            "{:?}",
            report.rows[0].movement
        );
        assert!(!report.rows[0].movement.fails(false));
        assert!(report.rows[0].movement.fails(true));
    }

    /// The M22 shape exactly: the number is *better*, and it is still not a
    /// comparison, because the artifact does not record what produced it.
    #[test]
    fn a_run_that_does_not_record_its_operating_point_is_unverifiable() {
        let base = pinned("lme_v2_small.overall_full_set.combined", 36.59);
        let mut stale = ours("lme_v2_small.overall_full_set.combined", 39.91);
        stale.unrecorded = vec!["select", "dated"];
        let report = compare(&base, &map(stale), 0.0);
        match &report.rows[0].movement {
            Movement::Unverifiable { pinned, why } => {
                assert_eq!(*pinned, 36.59);
                assert!(why.contains("select"), "{why}");
            }
            other => {
                panic!("a higher but unrecoverable number must not read as progress: {other:?}")
            }
        }
    }

    /// Deleting the run directory must not retire the floor silently.
    #[test]
    fn a_pinned_metric_with_no_artifact_is_reported() {
        let base = pinned("locomo.judge_score.n1540", 69.87);
        let report = compare(&base, &BTreeMap::new(), 0.0);
        assert_eq!(report.rows.len(), 1);
        assert!(matches!(
            report.rows[0].movement,
            Movement::Unverifiable { .. }
        ));
        assert_eq!(report.rows[0].run, None);
    }

    /// `raise` is a ratchet: it adopts an improvement and refuses a
    /// regression, so `--update` can never be the thing that made the build
    /// green.
    #[test]
    fn raising_the_floor_only_ever_moves_it_up() {
        let mut base = pinned("locomo.judge_score.n1540", 69.87);
        let moved = raise(
            &mut base,
            &map(ours("locomo.judge_score.n1540", 66.00)),
            "2026-09-20",
        );
        assert!(moved.is_empty(), "a worse number must not become the floor");
        assert_eq!(base.pinned["locomo.judge_score.n1540"].value, 69.87);

        let moved = raise(
            &mut base,
            &map(ours("locomo.judge_score.n1540", 71.00)),
            "2026-09-20",
        );
        assert_eq!(moved.len(), 1);
        assert_eq!(base.pinned["locomo.judge_score.n1540"].value, 71.00);
        assert_eq!(base.pinned["locomo.judge_score.n1540"].run, "runs/today");
    }

    /// And it refuses to pin an arm at all, in either direction.
    #[test]
    fn raising_the_floor_never_pins_an_arm() {
        let mut base = Baseline::default();
        let mut arm = ours("locomo.judge_score.n1540", 99.0);
        arm.arm = true;
        assert!(raise(&mut base, &map(arm), "2026-09-20").is_empty());
        assert!(base.pinned.is_empty());
    }

    /// Tolerance absorbs noise without absorbing a real drop.
    #[test]
    fn tolerance_is_applied_in_the_metrics_own_direction() {
        let base = pinned("locomo.judge_score.n1540", 69.87);
        let inside = compare(&base, &map(ours("locomo.judge_score.n1540", 69.50)), 0.5);
        assert!(matches!(inside.rows[0].movement, Movement::Held { .. }));
        let outside = compare(&base, &map(ours("locomo.judge_score.n1540", 69.00)), 0.5);
        assert!(matches!(
            outside.rows[0].movement,
            Movement::Regressed { .. }
        ));
    }

    #[test]
    fn a_baseline_round_trips_through_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("progression.json");
        let base = pinned("locomo.judge_score.n1540", 69.87);
        save(&path, &base).unwrap();
        let back = load(&path).unwrap();
        assert_eq!(back.pinned, base.pinned);
        assert_eq!(back.schema, 1);
    }

    #[test]
    fn a_missing_baseline_is_an_empty_floor_not_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let base = load(&tmp.path().join("absent.json")).unwrap();
        assert!(base.pinned.is_empty());
    }
}
