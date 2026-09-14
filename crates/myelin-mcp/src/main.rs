//! `myelin-mcp` — the MCP surface over `myelin-core`.
//!
//! Transport selection only, for now: stdio by default, streamable HTTP behind
//! `--serve <ADDR>`, exactly as `hs-mcp` does (`docs/research/09-house-conventions.md` §8).
//! Tool registration is M10; this binary deliberately contains no memory logic
//! (`PLAN.md` §3.2).

use clap::Parser;

/// myelin-mcp — MCP server over the myelin memory backend
#[derive(Parser)]
#[command(name = "myelin-mcp")]
struct Args {
    /// Run as a streamable-HTTP server on this address (default: stdio mode)
    /// Example: --serve 127.0.0.1:7446
    #[arg(long)]
    serve: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = myelin_core::config::MyelinConfig::load()?;

    match args.serve.as_deref() {
        Some(addr) => println!("myelin-mcp: transport=streamable-http addr={addr}"),
        None => println!("myelin-mcp: transport=stdio"),
    }
    println!(
        "myelin-mcp: qdrant={} collection={}",
        cfg.qdrant.url, cfg.qdrant.collection
    );
    println!("myelin-mcp: no tools registered (milestone M10)");

    Ok(())
}
