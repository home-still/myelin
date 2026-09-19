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

    /// Distinct from [`MyelinError::EmptyCompletion`] on purpose. An empty
    /// body means the model never ran; an empty *content* with
    /// `finish_reason = "length"` means it ran and spent the whole budget
    /// thinking. Measured on Qwen3.5-9B: a two-line episode consumed all 512
    /// tokens as `reasoning_content` and returned `content: ""`. Reporting
    /// that as a load failure sends you to `nvidia-smi` instead of to the
    /// token budget.
    #[error("{model} spent its entire {max_tokens:?}-token budget on reasoning and returned no content; disable thinking or raise max_tokens")]
    BudgetExhausted {
        model: String,
        max_tokens: Option<u32>,
    },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}

impl MyelinError {
    /// A stable slug naming the failure domain and nothing else.
    ///
    /// `Display` on the [`MyelinError::Qdrant`] and [`MyelinError::Io`]
    /// variants forwards the inner error verbatim, which can carry a URL, a
    /// filesystem path or a host name. Any surface that answers an untrusted
    /// caller reports this instead and logs the full error locally.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Qdrant(_) => "qdrant",
            Self::Config(_) => "config",
            Self::Store(_) => "store",
            Self::EmptyCompletion { .. } => "llm-empty-completion",
            Self::BudgetExhausted { .. } => "llm-budget-exhausted",
            Self::Io(_) => "io",
            Self::Serde(_) => "serde",
        }
    }
}

pub type Result<T, E = MyelinError> = std::result::Result<T, E>;
