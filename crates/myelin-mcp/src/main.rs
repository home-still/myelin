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

/// The bearer token `--serve` requires on a non-loopback bind.
const TOKEN_ENV: &str = "MYELIN_MCP_TOKEN";

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

    /// Candidates fetched per channel before fusion. Default 50
    /// (`RetrieveConfig::prefetch_limit`). A `k` near the default makes the
    /// reranker an ordering of the whole pool rather than a selection from
    /// it, so a width arm must raise this with `k`.
    #[arg(long)]
    prefetch_limit: Option<u64>,

    /// Minimum candidates the reranker scores. Default 25
    /// (`RetrieveConfig::rerank_depth`); the effective depth is
    /// `max(rerank_depth, k)`.
    #[arg(long)]
    rerank_depth: Option<usize>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // stdout is the stdio transport, so the subscriber writes to stderr in
    // both modes. Without this the `tracing::error!` in `server::mcp_err` —
    // the only place the full, un-redacted core error survives — goes
    // nowhere. `from_default_env()` alone would be that same hole with extra
    // steps: an unset `RUST_LOG` yields an empty filter, which disables every
    // event.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let mut cfg = myelin_core::config::MyelinConfig::load()?;
    if let Some(ledger) = args.ledger {
        cfg.ledger = ledger;
    }
    let collection = args
        .collection
        .unwrap_or_else(|| cfg.qdrant.collection.clone());

    let backend = Arc::new(
        Backend::open(&cfg, &collection, args.prefetch_limit, args.rerank_depth).await?,
    );
    check_pairing(&backend, &cfg.ledger, &collection).await?;
    // The pool size actually used, on stderr in both transports: an arm that
    // did not widen the candidate pool is a configuration bug, and this is
    // the line that proves it either way from the run's own log.
    eprintln!(
        "retrieval: prefetch={} rerank_depth={}",
        backend.config.prefetch_limit, backend.config.rerank_depth
    );

    match args.serve.as_deref() {
        Some(addr) => {
            let listener = tokio::net::TcpListener::bind(addr).await?;
            let bound = listener.local_addr()?;
            let token = std::env::var(TOKEN_ENV).ok().filter(|t| !t.is_empty());

            // An anonymous bind is only defensible when the kernel already
            // restricts who can reach it. `forget` with `mode: "hard"` and
            // `confirm: true` erases a record and its whole descendant
            // closure, and there is no other authentication anywhere in this
            // binary.
            if !bound.ip().is_loopback() && token.is_none() {
                anyhow::bail!(
                    "--serve {bound} is not loopback; set {TOKEN_ENV} to a bearer token \
                     (every tool, including hard forget, is otherwise anonymous)"
                );
            }

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
            let mut router = axum::Router::new().nest_service("/mcp", service);
            if let Some(token) = token.clone() {
                let expected: Arc<str> = Arc::from(format!("Bearer {token}"));
                router = router.layer(axum::middleware::from_fn(
                    move |req: axum::extract::Request, next: axum::middleware::Next| {
                        let expected = expected.clone();
                        async move {
                            let presented = req
                                .headers()
                                .get(axum::http::header::AUTHORIZATION)
                                .and_then(|v| v.to_str().ok())
                                .unwrap_or_default();
                            if constant_time_eq(presented.as_bytes(), expected.as_bytes()) {
                                next.run(req).await
                            } else {
                                use axum::response::IntoResponse;
                                axum::http::StatusCode::UNAUTHORIZED.into_response()
                            }
                        }
                    },
                ));
            }
            // Printed before serving and flushed: a supervisor waits on this
            // line to know the port is accepting.
            println!(
                "myelin-mcp: transport=streamable-http addr={bound} collection={collection} \
                 ledger={} auth={}",
                cfg.ledger,
                if token.is_some() { "bearer" } else { "none" }
            );
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    tokio::signal::ctrl_c().await.ok();
                })
                .await?;
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

/// Compare two byte strings without an early exit.
///
/// `==` on `String` returns as soon as it finds a mismatching byte, which
/// leaks the shared prefix length to anyone who can time the response.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    // The length itself is not secret — the token's is fixed by whoever set
    // the env var — but the comparison below must still not short-circuit.
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Refuse to serve a ledger and a collection that plainly do not belong
/// together.
///
/// The pair is unenforced everywhere else and a mismatch is silent: every
/// recall fuses ids from one store and then fails to materialise them from
/// the other, so the server answers "no evidence" forever and looks healthy
/// doing it. Empty-and-empty is a fresh store and legal.
async fn check_pairing(backend: &Backend, ledger_path: &str, collection: &str) -> anyhow::Result<()> {
    let records = backend.ledger.count_live(chrono::Utc::now()).await? as f64;
    let points = match backend.store.count().await {
        Ok(n) => n as f64,
        // A collection that does not exist yet is not a mismatch; the write
        // path creates it.
        Err(e) => {
            tracing::warn!(error = %e, collection, "could not count points; skipping pairing check");
            return Ok(());
        }
    };
    if records == 0.0 && points == 0.0 {
        return Ok(());
    }
    let denom = records.max(points);
    if (records - points).abs() / denom > 0.01 {
        anyhow::bail!(
            "ledger and collection are not a pair: {ledger_path} holds {records:.0} live records \
             but collection {collection} holds {points:.0} points (>1% apart); \
             every recall would fuse ids the ledger cannot materialise"
        );
    }
    Ok(())
}
