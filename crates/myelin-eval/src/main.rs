//! `myelin-eval` — the evaluation harness (`PLAN.md` §3.3).
//!
//! The subcommand surface exists now so later milestones add bodies rather than
//! argument surfaces.  Only `fetch` is implemented (M3); the remaining six
//! arms stay as `not implemented (milestone M5+)` placeholders.


use std::path::Path;

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
}

impl BenchCorpus {
    fn defaults(self) -> BenchDefaults {
        match self {
            Self::Locomo => BenchDefaults {
                dataset: "data/locomo10.json",
                collection: "myelin_locomo",
                ledger: "data/locomo.ledger",
                slug: "locomo",
            },
            Self::LongmemevalS => BenchDefaults {
                dataset: "data/longmemeval_s.json",
                collection: "myelin_longmemeval_s",
                ledger: "data/longmemeval_s.ledger",
                slug: "lme_s",
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
    /// Render comparison tables and the Pareto/LAFS report
    Report,
    /// Package a leaderboard submission
    Package,
}

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Command::Fetch => "fetch",
            Command::Build { .. } => "build",
            Command::Bench { .. } => "bench",
            Command::Attack { .. } => "attack",
            Command::Ablate { .. } => "ablate",
            Command::Report => "report",
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
        Command::Attack { live, ref ledger_dir } => {
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
            if live {
                let run =
                    myelin_eval::attack_live::run(Path::new(ledger_dir), &[3, 6, 10]).await?;
                myelin_eval::attack_live::print(&run);
            } else {
                println!("\nE1/E2 skipped (pass --live; they need a GPU and a live store).");
            }
            Ok(())
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
            )
            .await
        }
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
) -> anyhow::Result<()> {
    let mode = match mode {
        "recall" => Mode::Recall,
        "investigate" => Mode::Investigate,
        other => anyhow::bail!("--mode must be recall or investigate, got {other:?}"),
    };
    let d = corpus.defaults();
    let dataset = dataset.unwrap_or(d.dataset);
    let collection = collection.unwrap_or(d.collection);
    let ledger = ledger.unwrap_or(d.ledger);
    let owned_out = out.map_or_else(
        || format!("runs/{}_{}", d.slug, mode_slug(mode)),
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
    println!(
        "    token F1 (answerable)   {:.4}",
        run.f1_answerable
    );
    println!("    exact match             {:.4}", run.em_answerable);
    println!(
        "    abstention (category 5) {:.4}",
        run.abstention_accuracy
    );
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
