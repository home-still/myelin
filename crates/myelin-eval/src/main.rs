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

#[derive(Subcommand)]
enum Command {
    /// Download and checksum-pin the benchmark datasets
    Fetch,
    /// Build a memory from a dataset into a backend
    Build {
        /// Qdrant collection to build into. Must be myelin_*-prefixed: the
        /// nine production collections on `big` are off limits.
        #[arg(long, default_value = "myelin_locomo")]
        collection: String,
        /// SQLite ledger path.
        #[arg(long, default_value = "data/locomo.ledger")]
        ledger: String,
        /// Ingest only the first N conversations, for a throughput probe
        /// before committing to a long GPU window.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Run the accuracy/latency benchmark suite
    Bench,
    /// Run the MINJA-style poisoning attack suite
    Attack,
    /// Run the ablation matrix
    Ablate,
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
            Command::Attack => "attack",
            Command::Ablate => "ablate",
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
            ref collection,
            ref ledger,
            limit,
        } => build_cmd(collection, ledger, limit).await,
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
async fn build_cmd(collection: &str, ledger: &str, limit: Option<usize>) -> anyhow::Result<()> {
    anyhow::ensure!(
        collection.starts_with("myelin_"),
        "refusing to build into {collection:?}: collections must be myelin_*-prefixed \
         so a typo cannot touch the production collections on big"
    );
    let data = Path::new("data/locomo10.json");
    anyhow::ensure!(
        data.exists(),
        "missing {}; run `myelin-eval fetch` first",
        data.display()
    );

    eprintln!("ingesting LoCoMo -> collection {collection}, ledger {ledger}");
    let report = build_locomo(data, collection, Path::new(ledger), limit).await?;
    let units = report.per_unit.len();
    report.print(units);
    Ok(())
}
