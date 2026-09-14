//! `myelin-mcp` — the MCP surface over `myelin-core` (`PLAN.md` §3.2).
//!
//! stdio by default, streamable HTTP behind `--serve`, selected by clap
//! exactly as `hs-mcp` does (`docs/research/09-house-conventions.md` §8).
//! The binary contains no memory logic; see `server.rs`.

use std::sync::Arc;

use clap::Parser;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::transport::stdio;
use rmcp::ServiceExt;

mod server;

use server::{Backend, MyelinServer};

/// myelin-mcp — MCP server over the myelin memory backend
#[derive(Parser)]
#[command(name = "myelin-mcp")]
struct Args {
    /// Run as a streamable-HTTP server on this address (default: stdio mode).
    /// Example: --serve 127.0.0.1:7446
    #[arg(long)]
    serve: Option<String>,

    /// Qdrant collection to read. Overrides the configured default so one
    /// build can serve LoCoMo and another LME-V2 without editing config.
    #[arg(long)]
    collection: Option<String>,

    /// SQLite ledger path. Must match the collection: the ledger is the
    /// admissibility authority and a mismatched pair silently returns nothing.
    #[arg(long)]
    ledger: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut cfg = myelin_core::config::MyelinConfig::load()?;
    if let Some(ledger) = args.ledger {
        cfg.ledger = ledger;
    }
    let collection = args
        .collection
        .unwrap_or_else(|| cfg.qdrant.collection.clone());

    let backend = Arc::new(Backend::open(&cfg, &collection).await?);

    match args.serve.as_deref() {
        Some(addr) => {
            // `with_json_response(true)`: the benchmark adapter is a plain
            // request/response client with no use for SSE streaming, and a
            // single JSON body is far easier to debug from Python.
            let config = StreamableHttpServerConfig::default().with_json_response(true);
            let service: StreamableHttpService<MyelinServer, LocalSessionManager> =
                StreamableHttpService::new(
                    {
                        let backend = backend.clone();
                        move || Ok(MyelinServer::new(backend.clone()))
                    },
                    Default::default(),
                    config,
                );
            let router = axum::Router::new().nest_service("/mcp", service);
            let listener = tokio::net::TcpListener::bind(addr).await?;
            // Printed before serving and flushed: a supervisor waits on this
            // line to know the port is accepting.
            println!(
                "myelin-mcp: transport=streamable-http addr={} collection={collection} ledger={}",
                listener.local_addr()?,
                cfg.ledger
            );
            axum::serve(listener, router).await?;
        }
        None => {
            // stdio carries the protocol, so diagnostics must go to stderr.
            eprintln!("myelin-mcp: transport=stdio collection={collection}");
            let service = MyelinServer::new(backend).serve(stdio()).await?;
            service.waiting().await?;
        }
    }

    Ok(())
}
