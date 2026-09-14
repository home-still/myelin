//! `myelin-core` — the agentic-memory backend.
//!
//! Library only, no binary: the MCP surface (`myelin-mcp`) and the evaluation
//! harness (`myelin-eval`) are both projections of this crate. See `PLAN.md` §3.1.

pub mod config;
pub mod error;
pub mod store;

pub use error::{MyelinError, Result};
