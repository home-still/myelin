//! `myelin-eval` — the evaluation harness (`PLAN.md` §3.3).
//!
//! The subcommand surface exists now so later milestones add bodies rather than
//! argument surfaces. Nothing here is implemented yet.

use clap::{Parser, Subcommand};

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
    Build,
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
            Command::Build => "build",
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
    println!("{}: not implemented (milestone M5+)", args.command.name());
    Ok(())
}
