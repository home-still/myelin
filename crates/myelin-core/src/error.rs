//! One error type for the crate, one variant per failure domain. No `anyhow`
//! in the library (`PLAN.md` §3.1) — callers get a type they can match on.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MyelinError {
    #[error("qdrant: {0}")]
    Qdrant(#[from] qdrant_client::QdrantError),

    #[error("config: {0}")]
    Config(String),

    #[error("store: {0}")]
    Store(String),

    /// llama-swap answers `200` with a zero-byte body when the upstream model
    /// fails to load (`docs/research/00-verified-environment.md` §7.2). An empty
    /// completion is a failure, never a null answer — `PLAN.md` §4.1 R7.
    #[error("llm returned an empty completion from {model}")]
    EmptyCompletion { model: String },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T, E = MyelinError> = std::result::Result<T, E>;
