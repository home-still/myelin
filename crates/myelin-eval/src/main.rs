//! `myelin-eval` — the evaluation harness (`PLAN.md` §3.3).
//!
//! The subcommand surface exists now so later milestones add bodies rather than
//! argument surfaces.  Only `fetch` is implemented (M3); the remaining six
//! arms stay as `not implemented (milestone M5+)` placeholders.


use std::path::Path;

use anyhow::Context;
use clap::{Parser, Subcommand};

use myelin_core::model::query::Mode;
use myelin_eval::build::build_locomo;
use myelin_eval::datasets::{self, locomo, longmemeval};

/// myelin-eval — the agentic-memory evaluation harness
#[derive(Parser)]
#[command(name = "myelin-eval")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

/// The corpora `PLAN.md` §9.1 names. Each pins its own default collection
/// and ledger so that a `--corpus` switch cannot quietly append one corpus's
/// records to another's memory.
/// Corpora `bench` can score, each carrying the paths `build` wrote.
///
/// Defaults live on the enum rather than on the flags so `--corpus
/// longmemeval-s` alone is correct; a default collection of `myelin_locomo`
/// silently benching the wrong store is the failure this prevents.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum BenchCorpus {
    Locomo,
    #[value(name = "longmemeval-s")]
    LongmemevalS,
}

struct BenchDefaults {
    dataset: &'static str,
    collection: &'static str,
    ledger: &'static str,
    slug: &'static str,
    /// Which scorer this corpus reports by default.
    ///
    /// Per corpus and not one global flag, because the evidence that settled
    /// it is per corpus: `docs/measurements/m14-temporal-scorer.md` validated
    /// the date-aware scorer against a judge on **LoCoMo category 2** and
    /// found it agreeing 96.7% against token F1's 84.9% (+11.8 points, 95% CI
    /// [+7.7, +15.8]). LongMemEval_S has no such docket, and its grammar
    /// coverage is 26 of 470 answerable golds — 11 of 127 even in its own
    /// temporal-reasoning stratum — so there is nothing there to flip on.
    scorer: myelin_eval::bench::Scorer,
}

impl BenchCorpus {
    fn defaults(self) -> BenchDefaults {
        match self {
            Self::Locomo => BenchDefaults {
                dataset: "data/locomo10.json",
                collection: "myelin_locomo",
                ledger: "data/locomo.ledger",
                slug: "locomo",
                scorer: myelin_eval::bench::Scorer::Temporal,
            },
            Self::LongmemevalS => BenchDefaults {
                dataset: "data/longmemeval_s.json",
                collection: "myelin_longmemeval_s",
                ledger: "data/longmemeval_s.ledger",
                slug: "lme_s",
                scorer: myelin_eval::bench::Scorer::TokenF1,
            },
        }
    }
}

fn mode_slug(mode: Mode) -> &'static str {
    match mode {
        Mode::Recall => "recall",
        Mode::Investigate => "investigate",
    }
}

/// Where a `bench` run lands when `--out` is not given.
///
/// Every switch gets its own suffix, in a fixed order, so a directory name is
/// a function of the switch set and no arm can clobber another — least of all
/// the M9 baselines in `runs/locomo_recall` and `runs/lme_s_recall` that every
/// paired-CI comparison is against.
///
/// The scorer suffix is keyed on the scorer's *identity*, not on whether it is
/// the default: M14 flipped LoCoMo's default to `temporal`, and that must not
/// silently redirect a `--scorer token-f1` run onto the M9 baseline path.
fn bench_out_dir(
    slug: &str,
    mode: Mode,
    switches: &myelin_eval::bench::BenchSwitches,
    scorer: myelin_eval::bench::Scorer,
) -> String {
    let cats = if switches.categories.is_empty() {
        String::new()
    } else {
        let codes: Vec<String> = switches.categories.iter().map(u8::to_string).collect();
        format!("_cat{}", codes.join(""))
    };
    format!(
        "runs/{slug}_{}{}{}{}{cats}{}",
        mode_slug(mode),
        if switches.graph { "_graph" } else { "" },
        if switches.chronological { "_chrono" } else { "" },
        if switches.question_date { "_qdate" } else { "" },
        match scorer {
            myelin_eval::bench::Scorer::TokenF1 => "",
            myelin_eval::bench::Scorer::Temporal => "_temporal",
            myelin_eval::bench::Scorer::Judge => "_judge",
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use myelin_eval::bench::{BenchSwitches, Scorer};

    #[test]
    fn only_token_f1_with_every_switch_off_reaches_the_m9_baseline_path() {
        // `runs/locomo_recall` and `runs/lme_s_recall` are the committed M9
        // artifacts every paired CI is measured against. Exactly one switch
        // set may name them, and it is the one that produced them.
        let off = BenchSwitches::default();
        assert_eq!(
            bench_out_dir("locomo", Mode::Recall, &off, Scorer::TokenF1),
            "runs/locomo_recall"
        );
        assert_eq!(
            bench_out_dir("lme_s", Mode::Recall, &off, Scorer::TokenF1),
            "runs/lme_s_recall"
        );
        // The post-M14 LoCoMo default must not be one of them.
        assert_eq!(
            bench_out_dir("locomo", Mode::Recall, &off, Scorer::Temporal),
            "runs/locomo_recall_temporal"
        );
    }

    #[test]
    fn switch_suffixes_keep_their_fixed_order() {
        assert_eq!(
            bench_out_dir(
                "locomo",
                Mode::Recall,
                &BenchSwitches {
                    graph: true,
                    chronological: true,
                    question_date: true,
                    ..Default::default()
                },
                Scorer::Temporal
            ),
            "runs/locomo_recall_graph_chrono_qdate_temporal"
        );
        assert_eq!(
            bench_out_dir(
                "locomo",
                Mode::Investigate,
                &BenchSwitches {
                    chronological: true,
                    ..Default::default()
                },
                Scorer::TokenF1
            ),
            "runs/locomo_investigate_chrono"
        );
    }

    /// A stratum arm never names the full-set path it is measured against.
    #[test]
    fn a_stratum_arm_cannot_clobber_the_full_set_path() {
        let arm = |categories: &[u8]| {
            bench_out_dir(
                "locomo",
                Mode::Recall,
                &BenchSwitches {
                    categories: categories.to_vec(),
                    ..Default::default()
                },
                Scorer::Temporal,
            )
        };
        assert_eq!(arm(&[2]), "runs/locomo_recall_cat2_temporal");
        assert_eq!(arm(&[2, 5]), "runs/locomo_recall_cat25_temporal");
        // No stratum: the 1,540-question path the baseline owns.
        assert_eq!(arm(&[]), "runs/locomo_recall_temporal");
    }

    #[test]
    fn the_scorer_default_is_per_corpus() {
        // M14's judge docket was LoCoMo category 2 and nothing else; the
        // grammar reaches 26 of LongMemEval_S's 470 answerable golds, so the
        // flip is LoCoMo-only.
        assert_eq!(BenchCorpus::Locomo.defaults().scorer, Scorer::Temporal);
        assert_eq!(
            BenchCorpus::LongmemevalS.defaults().scorer,
            Scorer::TokenF1
        );
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Corpus {
    Locomo,
    LmeV2Small,
    LmeV2Medium,
    #[value(name = "longmemeval-s")]
    LongmemevalS,
}

impl Corpus {
    fn slug(self) -> &'static str {
        match self {
            Corpus::Locomo => "locomo",
            Corpus::LmeV2Small => "lme_v2_small",
            Corpus::LmeV2Medium => "lme_v2_medium",
            Corpus::LongmemevalS => "longmemeval_s",
        }
    }

    fn collection(self) -> String {
        format!("myelin_{}", self.slug())
    }

    fn ledger(self) -> String {
        format!("data/{}.ledger", self.slug())
    }
}

#[derive(Subcommand)]
enum Command {
    /// Download and checksum-pin the benchmark datasets
    Fetch,
    /// Build a memory from a dataset into a backend
    Build {
        /// Which corpus. `locomo` extracts facts; `lme-v2-small` and
        /// `lme-v2-medium` ingest episodically — see
        /// `WritePath::extract_facts` for the ~250 GPU-hour measurement
        /// behind that split.
        #[arg(long, value_enum, default_value_t = Corpus::Locomo)]
        corpus: Corpus,
        /// Qdrant collection to build into. Must be myelin_*-prefixed: the
        /// nine production collections on `big` are off limits.
        #[arg(long)]
        collection: Option<String>,
        /// SQLite ledger path.
        #[arg(long)]
        ledger: Option<String>,
        /// Ingest only the first N units (per domain, for LME-V2), for a
        /// throughput probe before committing to a long GPU window.
        #[arg(long)]
        limit: Option<usize>,
        /// Directory holding the LME-V2 release files.
        #[arg(long, default_value = "/tmp/lmev2")]
        lmev2_dir: String,
        /// Repair any ledger/vector drift found at the end of the build
        /// instead of failing.
        #[arg(long)]
        repair: bool,
    },
    /// Populate the phrase↔record incidence graph over an already-built
    /// ledger. Pure SQLite: no Qdrant, no GPU, no model.
    Phrases {
        #[arg(long, value_enum, default_value_t = Corpus::Locomo)]
        corpus: Corpus,
        /// Defaults to data/<slug>.ledger.
        #[arg(long)]
        ledger: Option<String>,
        /// Records per transaction.
        #[arg(long, default_value_t = 5000)]
        batch: usize,
        /// Stop after N records, for a smoke run.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Score LoCoMo end-to-end: retrieve, read, and grade the answer with
    /// a deterministic scorer (no LLM judge). See `bench.rs`.
    Bench {
        /// Which corpus to score.
        #[arg(long, value_enum, default_value_t = BenchCorpus::Locomo)]
        corpus: BenchCorpus,
        /// Defaults follow --corpus when left unset.
        #[arg(long)]
        dataset: Option<String>,
        #[arg(long)]
        collection: Option<String>,
        #[arg(long)]
        ledger: Option<String>,
        #[arg(long, default_value_t = 6)]
        k: usize,
        /// `recall` is the fast path; `investigate` runs the agentic loop.
        #[arg(long, default_value = "recall")]
        mode: String,
        /// `investigate` only. Default matches `InvestigateConfig`.
        #[arg(long, default_value_t = 2)]
        max_steps: usize,
        /// Stop after N questions. Omit for all 1,986.
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        out: Option<String>,
        /// Fuse the PPR channel over the phrase↔record graph (`--corpus`'s
        /// ledger must have been through `myelin-eval phrases`).
        #[arg(long)]
        graph: bool,
        /// Emit evidence oldest-first instead of `bookend`'s relevance
        /// interleave.
        #[arg(long)]
        chronological: bool,
        /// Give the reader a `<today>` reference date. LongMemEval_S always
        /// carries one; this adds LoCoMo's last session date.
        #[arg(long)]
        question_date: bool,
        /// Score only these category codes, for a stratum arm. LoCoMo: 1
        /// multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.
        /// LongMemEval_S: 1 ss-user, 2 ss-assistant, 3 ss-preference,
        /// 4 multi-session, 5 temporal-reasoning, 6 knowledge-update.
        #[arg(long, value_delimiter = ',')]
        categories: Option<Vec<u8>>,
        /// Which scorer `score` reports. Both columns are always written on
        /// every row, so a run stays readable under either. Defaults follow
        /// `--corpus`.
        #[arg(long, value_enum)]
        scorer: Option<myelin_eval::bench::Scorer>,
    },
    /// Run the MINJA-style poisoning attack suite (EVALUATION.md §7)
    Attack {
        /// Also run E1/E2, which need a live store and a GPU: they build
        /// two scratch collections through the real write path.
        #[arg(long)]
        live: bool,
        /// Where scratch ledgers go. Deleted when the run finishes.
        #[arg(long, default_value = "data")]
        ledger_dir: String,
        /// Where to serialise the `--live` sweep: writes
        /// `<dir>/attack_live.json`, which is the artifact `standing` reads
        /// for G3. Without it a 54-minute run leaves nothing on disk but
        /// stdout.
        #[arg(long)]
        out: Option<String>,
    },
    /// Run the injection adjudicator over LoCoMo's real episodes and report
    /// the false-positive rate (M15). Reader only — no store.
    AdjudicateProbe {
        #[arg(long, default_value = "data/locomo10.json")]
        dataset: String,
        /// Stop after N episodes, for a cost probe before the full pass.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Run the M4 retrieval ablation against an already-built memory
    Ablate {
        #[arg(long, default_value = "data/locomo10.json")]
        dataset: String,
        #[arg(long, default_value = "myelin_locomo")]
        collection: String,
        #[arg(long, default_value = "data/locomo.ledger")]
        ledger: String,
        /// Dev-split size in *conversations*. Splitting by question would
        /// leak: questions from one conversation share one memory.
        #[arg(long, default_value_t = 5)]
        units: usize,
        /// Evidence-set size. `EVALUATION.md` §8 row 4 sweeps this.
        #[arg(long, default_value_t = 6)]
        k: usize,
        /// Cap questions per arm, for a quick smoke of the harness itself.
        #[arg(long)]
        limit: Option<usize>,
        /// Score the held-out conversations (everything after --units)
        /// instead of the dev split. Use once, to confirm a decision the
        /// dev table already made.
        #[arg(long)]
        holdout: bool,
        /// Instead of the channel ablation, sweep `investigate`'s step
        /// budget and report the marginal value of each extra step (M7).
        #[arg(long, value_delimiter = ',')]
        steps: Option<Vec<usize>>,
    },
    /// Re-score a finished bench run under a different scorer. Pure CPU:
    /// `response_raw` and `answer_gold` are on disk, so no reader call and no
    /// GPU are needed to apply a scorer change to every historical run.
    Rescore {
        /// A run directory written by `bench`.
        #[arg(long)]
        run: String,
        #[arg(long, value_enum, default_value_t = myelin_eval::bench::Scorer::Temporal)]
        scorer: myelin_eval::bench::Scorer,
        /// Defaults to runs/rescored/<source-basename>_<scorer>.
        #[arg(long)]
        out: Option<String>,
    },
    /// Grade a finished run's answers with the local reader (M14 scorer
    /// validation). Needs the reader only — no embedder, reranker, or store.
    Judge {
        #[arg(long)]
        run: String,
        /// Restrict to one category. LoCoMo 2 is the temporal stratum.
        #[arg(long)]
        category: Option<u8>,
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Audit a vendored LongMemEval-V2 run: for every answerable question the
    /// harness scored wrong, was the answer in the evidence? (M16). Reader only.
    EvidenceAudit {
        /// A run directory written by `adapters/run_myelin.py`.
        #[arg(long)]
        run: String,
        /// Judge at most the first N answerable rows, for a cost probe.
        /// Verdicts cache per row, so an unlimited re-run judges only what
        /// is missing and an interrupted pass resumes.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Join docs/sota/registry.json against the run artifacts and report where
    /// we stand, with a comparability verdict per row
    Standing {
        #[arg(long, default_value = "docs/sota/registry.json")]
        registry: String,
        #[arg(long, default_value = "runs")]
        runs: String,
        #[arg(long, default_value = "runs/standing")]
        out: String,
        /// Exit non-zero when a `gate: true` row is unsupported or not beaten
        #[arg(long)]
        gate: bool,
        #[arg(long, default_value = ".venv/bin/python")]
        python: String,
    },
    /// Package a leaderboard submission
    Package,
}

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Command::Fetch => "fetch",
            Command::Build { .. } => "build",
            Command::Phrases { .. } => "phrases",
            Command::Bench { .. } => "bench",
            Command::Attack { .. } => "attack",
            Command::AdjudicateProbe { .. } => "adjudicate-probe",
            Command::Ablate { .. } => "ablate",
            Command::Rescore { .. } => "rescore",
            Command::Judge { .. } => "judge",
            Command::EvidenceAudit { .. } => "evidence-audit",
            Command::Standing { .. } => "standing",
            Command::Package => "package",
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Fetch => fetch().await,
        Command::Build {
            corpus,
            ref collection,
            ref ledger,
            limit,
            ref lmev2_dir,
            repair,
        } => {
            build_cmd(
                corpus,
                collection.as_deref(),
                ledger.as_deref(),
                limit,
                lmev2_dir,
                repair,
            )
            .await
        }
        Command::Phrases {
            corpus,
            ref ledger,
            batch,
            limit,
        } => {
            let path = ledger.clone().unwrap_or_else(|| corpus.ledger());
            eprintln!(
                "backfilling incidence for namespace {} from {path}",
                corpus.slug()
            );
            let stats = myelin_eval::phrases::backfill_phrases(
                Path::new(&path),
                corpus.slug(),
                batch,
                limit,
            )
            .await?;
            println!(
                "phrases {}: {} records, {} edges, {} distinct phrases",
                corpus.slug(),
                stats.records,
                stats.edges,
                stats.distinct_phrases
            );
            Ok(())
        }
        Command::Attack {
            live,
            ref ledger_dir,
            ref out,
        } => {
            let (e3, e5) = myelin_eval::attack::run_offline()?;
            myelin_eval::attack::print_gate_report(&e3, &e5);
            anyhow::ensure!(
                e3.catch_rate() >= 0.90,
                "E3 gate: catch rate {:.1}% is below the 90% floor",
                e3.catch_rate() * 100.0
            );
            anyhow::ensure!(
                e5.admitted == 0,
                "E5 gate: {} poisoned records admitted at first-party trust",
                e5.admitted
            );
            anyhow::ensure!(
                live || out.is_none(),
                "--out serialises the --live sweep; the offline E3/E5 gates have \
                 no per-condition artifact. Pass --live."
            );
            if live {
                let run =
                    myelin_eval::attack_live::run(Path::new(ledger_dir), &[3, 6, 10]).await?;
                myelin_eval::attack_live::print(&run);
                if let Some(dir) = out {
                    let dir = Path::new(dir);
                    std::fs::create_dir_all(dir)
                        .with_context(|| format!("create {}", dir.display()))?;
                    let path = dir.join("attack_live.json");
                    std::fs::write(&path, serde_json::to_string_pretty(&run)?)
                        .with_context(|| format!("write {}", path.display()))?;
                    println!("\nwrote {}", path.display());
                }
            } else {
                println!("\nE1/E2 skipped (pass --live; they need a GPU and a live store).");
            }
            Ok(())
        }
        Command::AdjudicateProbe { ref dataset, limit } => {
            adjudicate_probe_cmd(dataset, limit).await
        }
        Command::Ablate {
            ref dataset,
            ref collection,
            ref ledger,
            units,
            k,
            limit,
            ref steps,
            holdout,
        } => {
            ablate_cmd(
                dataset,
                collection,
                ledger,
                units,
                k,
                limit,
                steps.as_deref(),
                holdout,
            )
            .await
        }
        Command::Bench {
            corpus,
            ref dataset,
            ref collection,
            ref ledger,
            k,
            ref mode,
            max_steps,
            limit,
            ref out,
            graph,
            chronological,
            question_date,
            ref categories,
            scorer,
        } => {
            bench_cmd(
                corpus,
                dataset.as_deref(),
                collection.as_deref(),
                ledger.as_deref(),
                k,
                mode,
                max_steps,
                limit,
                out.as_deref(),
                myelin_eval::bench::BenchSwitches {
                    graph,
                    chronological,
                    question_date,
                    categories: categories.clone().unwrap_or_default(),
                },
                scorer,
            )
            .await
        }
        Command::Rescore {
            ref run,
            scorer,
            ref out,
        } => rescore_cmd(run, scorer, out.as_deref()),
        Command::Judge {
            ref run,
            category,
            limit,
        } => judge_cmd(run, category, limit).await,
        Command::EvidenceAudit { ref run, limit } => evidence_audit_cmd(run, limit).await,
        Command::Standing {
            ref registry,
            ref runs,
            ref out,
            gate,
            ref python,
        } => myelin_eval::standing::run(
            Path::new(registry),
            Path::new(runs),
            Path::new(out),
            gate,
            python,
        )
        .map(|_| ()),
        rest => {
            println!("{}: not implemented (milestone M5+)", rest.name());
            Ok(())
        }
    }
}

/// Fetch all pinned datasets into `data/` and print a verification table.
///
/// Each file is checksummed against its pinned SHA-256; a mismatch is a hard
/// error rather than a silent downgrade (`PLAN.md` §1.1 — reproducibility
/// requires a pinned corpus).
async fn fetch() -> anyhow::Result<()> {
    let data_dir = Path::new("data");

    let path = datasets::fetch_pinned(&datasets::LOCOMO, data_dir).await?;
    let digest = datasets::sha256_file(&path)?;

    // LongMemEval_S is 278 MB and only needed for G2's second number, but it
    // is fetched here rather than on demand so one command produces the whole
    // pinned corpus set and a checksum mismatch surfaces before a GPU window
    // is spent on it.
    let lme_s = datasets::fetch_pinned(&datasets::LONGMEMEVAL_S, data_dir).await?;
    let lme_s_items = longmemeval::load(&lme_s)?;
    println!(
        "longmemeval_s   {:>6} questions  sha256 {}",
        lme_s_items.len(),
        datasets::sha256_file(&lme_s)?
    );

    let conversations = locomo::load(&path)?;
    let counts = locomo::qa_counts(&conversations);

    println!("file:          {}", datasets::LOCOMO.name);
    println!("url:           {}", datasets::LOCOMO.url);
    println!("bytes:         {}", datasets::LOCOMO.bytes);
    println!("sha256:        {} (verified)", digest);
    println!("conversations: {}", conversations.len());

    println!("qa counts:");
    for (cat, n) in &counts.by_category {
        println!("  category {}: {}", cat, n);
    }
    println!("  total:        {}", counts.total);
    println!("  comparable:   {}  (category 5 dropped)", counts.comparable_1540);
    println!("  adversarial:  {}  (category 5)", counts.adversarial);

    Ok(())
}
/// M3: drive LoCoMo through the write path and report records/unit, tokens
/// and wall time.
async fn build_cmd(
    corpus: Corpus,
    collection: Option<&str>,
    ledger: Option<&str>,
    limit: Option<usize>,
    lmev2_dir: &str,
    repair: bool,
) -> anyhow::Result<()> {
    let collection = collection.map_or_else(|| corpus.collection(), str::to_string);
    let ledger = ledger.map_or_else(|| corpus.ledger(), str::to_string);
    anyhow::ensure!(
        collection.starts_with("myelin_"),
        "refusing to build into {collection:?}: collections must be myelin_*-prefixed \
         so a typo cannot touch the production collections on big"
    );

    let report = match corpus {
        Corpus::Locomo => {
            let data = Path::new("data/locomo10.json");
            anyhow::ensure!(
                data.exists(),
                "missing {}; run `myelin-eval fetch` first",
                data.display()
            );
            eprintln!("ingesting LoCoMo -> collection {collection}, ledger {ledger}");
            build_locomo(data, &collection, Path::new(&ledger), limit, repair).await?
        }
        Corpus::LongmemevalS => {
            let data = Path::new("data/longmemeval_s.json");
            anyhow::ensure!(
                data.exists(),
                "missing {}; run `myelin-eval fetch` first",
                data.display()
            );
            eprintln!("ingesting LongMemEval_S -> collection {collection}, ledger {ledger}");
            myelin_eval::build::build_longmemeval_s(
                data,
                &collection,
                Path::new(&ledger),
                limit,
                repair,
            )
            .await?
        }
        Corpus::LmeV2Small | Corpus::LmeV2Medium => {
            let dir = Path::new(lmev2_dir);
            let trajectories = dir.join("trajectories.jsonl");
            let questions = dir.join("questions.jsonl");
            let haystack = dir.join("haystacks").join(format!("{}.json", corpus.slug()));
            for p in [&trajectories, &questions, &haystack] {
                anyhow::ensure!(p.exists(), "missing {}", p.display());
            }
            eprintln!(
                "ingesting {} -> collection {collection}, ledger {ledger}",
                corpus.slug()
            );
            myelin_eval::build::build_lmev2(
                &trajectories,
                &haystack,
                &questions,
                corpus.slug(),
                &collection,
                Path::new(&ledger),
                limit,
                repair,
            )
            .await?
        }
    };

    let units = report.per_unit.len();
    report.print(units);
    Ok(())
}

/// Run the M4 ablation table against a memory that `build` already wrote.
///
/// Deliberately separate from `build`: rebuilding a memory costs GPU-hours and
/// the ablation costs minutes, so they must be independently runnable.
#[allow(clippy::too_many_arguments)]
async fn ablate_cmd(
    dataset: &str,
    collection: &str,
    ledger: &str,
    units: usize,
    k: usize,
    limit: Option<usize>,
    steps: Option<&[usize]>,
    holdout: bool,
) -> anyhow::Result<()> {
    if let Some(steps) = steps {
        let points = myelin_eval::ablate::investigate_curve(
            Path::new(dataset),
            collection,
            Path::new(ledger),
            units,
            k,
            steps,
            limit,
            holdout,
        )
        .await?;
        myelin_eval::ablate::print_step_curve(&points, k);
        return Ok(());
    }
    let run = myelin_eval::ablate::ablate_locomo(
        Path::new(dataset),
        collection,
        Path::new(ledger),
        units,
        k,
        limit,
        holdout,
    )
    .await?;
    myelin_eval::ablate::print_table(&run, k);
    Ok(())
}

/// Score LoCoMo end-to-end against a memory that `build` already wrote.
///
/// Separate from `ablate` for the same reason `ablate` is separate from
/// `build`: retrieval quality and answer quality are different questions and
/// answering the second costs a reader call per question.
#[allow(clippy::too_many_arguments)]
async fn bench_cmd(
    corpus: BenchCorpus,
    dataset: Option<&str>,
    collection: Option<&str>,
    ledger: Option<&str>,
    k: usize,
    mode: &str,
    max_steps: usize,
    limit: Option<usize>,
    out: Option<&str>,
    switches: myelin_eval::bench::BenchSwitches,
    scorer: Option<myelin_eval::bench::Scorer>,
) -> anyhow::Result<()> {
    let mode = match mode {
        "recall" => Mode::Recall,
        "investigate" => Mode::Investigate,
        other => anyhow::bail!("--mode must be recall or investigate, got {other:?}"),
    };
    anyhow::ensure!(
        !(switches.question_date && corpus == BenchCorpus::LongmemevalS),
        "--question-date is LoCoMo-only; the LongMemEval_S prompt already carries <today>"
    );
    // Before any GPU work: a judged run needs verdicts, and a live `bench`
    // has none. Discovering that after 50 minutes of reader calls would
    // throw the run away.
    anyhow::ensure!(
        scorer != Some(myelin_eval::bench::Scorer::Judge),
        "--scorer judge is rescore-only: run bench, then judge --run <out>, \
         then rescore --run <out> --scorer judge"
    );
    let d = corpus.defaults();
    let dataset = dataset.unwrap_or(d.dataset);
    let collection = collection.unwrap_or(d.collection);
    let ledger = ledger.unwrap_or(d.ledger);
    let scorer = scorer.unwrap_or(d.scorer);
    let owned_out = out.map_or_else(
        || bench_out_dir(d.slug, mode, &switches, scorer),
        str::to_string,
    );
    let out = owned_out.as_str();

    let run = match corpus {
        BenchCorpus::Locomo => {
            myelin_eval::bench::bench_locomo(
                Path::new(dataset),
                collection,
                Path::new(ledger),
                k,
                mode,
                max_steps,
                limit,
                &switches,
                scorer,
                Path::new(out),
            )
            .await?
        }
        BenchCorpus::LongmemevalS => {
            myelin_eval::bench::bench_longmemeval_s(
                Path::new(dataset),
                collection,
                Path::new(ledger),
                k,
                mode,
                max_steps,
                limit,
                &switches,
                scorer,
                Path::new(out),
            )
            .await?
        }
    };

    println!();
    println!(
        "  {} {} k={} over {} questions",
        run.corpus, run.mode, run.k, run.questions
    );
    // Named by the scorer the run reports, because `score` carries whichever
    // column `--scorer` selected and a fixed "token F1" label would lie.
    let score_label = match run.scorer.as_str() {
        "temporal" => "temporal (answerable)",
        _ => "token F1 (answerable)",
    };
    println!("    {score_label:<23} {:.4}", run.f1_answerable);
    println!("    exact match             {:.4}", run.em_answerable);
    // Named by what marks an item unanswerable in each corpus, not by
    // LoCoMo's category number: LongMemEval_S uses an `_abs` id suffix and
    // printing "category 5" there pointed at temporal-reasoning instead.
    let abs_label = match run.corpus.as_str() {
        "locomo" => "abstention (category 5)",
        _ => "abstention (_abs items)",
    };
    println!("    {abs_label:<23} {:.4}", run.abstention_accuracy);
    println!(
        "    query latency           p50 {:.2}s  avg {:.2}s",
        run.query_p50_seconds, run.query_avg_seconds
    );
    println!();
    println!("    {:<10}{:>7}{:>12}", "category", "n", "mean");
    for c in &run.by_category {
        println!(
            "    {:<10}{:>7}{:>12.4}",
            c.category, c.count, c.mean_score
        );
    }
    println!();
    println!("  wrote {out}/per_question.jsonl and {out}/aggregated_metrics.json");
    Ok(())
}

/// Re-score a finished bench run. Pure CPU, and it never writes into the
/// source: a rescored directory lives under `runs/rescored/` so provenance is
/// visible from the path and cannot collide with a live `bench` directory.
fn rescore_cmd(
    run_dir: &str,
    scorer: myelin_eval::bench::Scorer,
    out: Option<&str>,
) -> anyhow::Result<()> {
    let source = Path::new(run_dir);
    let basename = source
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .with_context(|| format!("--run {run_dir} has no directory name"))?;
    let owned_out = out.map_or_else(
        || format!("runs/rescored/{basename}_{}", scorer.slug()),
        str::to_string,
    );
    let out = owned_out.as_str();
    let run = myelin_eval::bench::rescore_run(source, Path::new(out), scorer)?;

    println!();
    println!(
        "  {} {} rescored under {} over {} questions",
        run.corpus, run.mode, run.scorer, run.questions
    );
    println!("    mean score (answerable) {:.4}", run.f1_answerable);
    println!("    exact match             {:.4}", run.em_answerable);
    println!("    abstention              {:.4}", run.abstention_accuracy);
    println!();
    println!("    {:<10}{:>7}{:>12}", "category", "n", "mean");
    for c in &run.by_category {
        println!("    {:<10}{:>7}{:>12.4}", c.category, c.count, c.mean_score);
    }
    println!();
    println!("  wrote {out}/per_question.jsonl and {out}/aggregated_metrics.json");
    Ok(())
}

/// Grade a finished run's answers with the local reader.
///
/// Reader-only, so the GPU window this needs is a fraction of a bench run's:
/// no embedder, no reranker, no store, no ledger.
async fn judge_cmd(run: &str, category: Option<u8>, limit: Option<usize>) -> anyhow::Result<()> {
    let dir = Path::new(run);
    let (file, stats) = myelin_eval::judge::judge_run(dir, category, limit).await?;
    let total = stats.judged + stats.cached;
    println!();
    println!("  judge {} over {run}", file.model);
    println!(
        "    {} judged, {} from cache, {} of {total} marked correct ({:.1}%)",
        stats.judged,
        stats.cached,
        stats.correct,
        if total == 0 {
            0.0
        } else {
            100.0 * stats.correct as f64 / total as f64
        }
    );
    println!("  wrote {run}/judge_verdicts.json");
    Ok(())
}

/// M16: which side of the pipeline loses an answerable question.
///
/// Reader-only, like `judge`: no embedder, no store, no ledger. The rule in
/// `docs/measurements/m16-evidence-sufficiency.md` is applied to the split,
/// so the verbatim labels are printed beside the rate.
async fn evidence_audit_cmd(run: &str, limit: Option<usize>) -> anyhow::Result<()> {
    let report = myelin_eval::evidence_audit::audit(Path::new(run), limit).await?;
    myelin_eval::evidence_audit::print(&report);
    Ok(())
}

/// M15: the injection gate's false-positive cost on real corpus text.
///
/// Reader-only, like `judge`: no embedder, no store, no ledger. The rule in
/// `docs/measurements/m15-injection-adjudication.md` is applied to this
/// number, so it prints the flagged text in full rather than a rate alone.
async fn adjudicate_probe_cmd(dataset: &str, limit: Option<usize>) -> anyhow::Result<()> {
    let report = myelin_eval::adjudicate_probe::probe(Path::new(dataset), limit).await?;
    myelin_eval::adjudicate_probe::print(&report);
    Ok(())
}
