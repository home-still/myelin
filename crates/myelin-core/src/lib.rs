//! `myelin-core` — the agentic-memory backend.
//!
//! Library only, no binary: the MCP surface (`myelin-mcp`) and the evaluation
//! harness (`myelin-eval`) are both projections of this crate. See `PLAN.md` §3.1.

pub mod config;
pub mod embed;
pub mod error;
pub mod llm;
pub mod model;
pub mod pipeline;
pub mod rerank;
pub mod store;

pub use error::{MyelinError, Result};
pub use model::{
    AppliedDelta, Delta, EvidenceItem, EvidenceSet, MemoryRecord, Mode, Recall, RecordKind, Scope,
    ScopeFilter, TrustTier,
};
pub use store::{Ledger, QdrantStore};
