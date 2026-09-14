//! `myelin-eval` — the evaluation harness (`PLAN.md` §3.3).
//!
//! The subcommand surface exists now so later milestones add bodies rather than
//! argument surfaces.  Only `fetch` is implemented (M3); the remaining six
//! arms stay as `not implemented (milestone M5+)` placeholders.


use std::path::Path;

use clap::{Parser, Subcommand};

use myelin_eval::build::build_locomo;
use myelin_eval::datasets::{self, locomo};

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
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Corpus {
    Locomo,
    LmeV2Small,
    LmeV2Medium,
}

impl Corpus {
    fn slug(self) -> &'static str {
        match self {
            Corpus::Locomo => "locomo",
            Corpus::LmeV2Small => "lme_v2_small",
            Corpus::LmeV2Medium => "lme_v2_medium",
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
    },
    /// Run the accuracy/latency benchmark suite
    Bench,
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
            Command::Bench => "bench",
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
        } => {
            build_cmd(
                corpus,
                collection.as_deref(),
                ledger.as_deref(),
                limit,
                lmev2_dir,
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
        } => ablate_cmd(dataset, collection, ledger, units, k, limit).await,
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
            build_locomo(data, &collection, Path::new(&ledger), limit).await?
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
async fn ablate_cmd(
    dataset: &str,
    collection: &str,
    ledger: &str,
    units: usize,
    k: usize,
    limit: Option<usize>,
) -> anyhow::Result<()> {
    let run = myelin_eval::ablate::ablate_locomo(
        Path::new(dataset),
        collection,
        Path::new(ledger),
        units,
        k,
        limit,
    )
    .await?;
    myelin_eval::ablate::print_table(&run, k);
    Ok(())
}
