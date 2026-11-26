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

    /// A store definition the registry refuses: a malformed name, or a name
    /// or tenant already bound elsewhere. The message names only what the
    /// caller sent and the binding it collides with, so it is safe to return.
    #[error("store definition: {0}")]
    StoreDefinition(String),

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

    /// A write failed in one of its legs and left nothing behind: the ledger
    /// and the vector index hold what they held before the call.
    ///
    /// `cause` is the failing leg's own message (the embedder's HTTP status
    /// and body, the index's gRPC status), so a caller can tell an
    /// over-long input from a dead index. It is what an MCP caller reads.
    #[error("write failed in the {leg} leg, nothing was kept: {cause}")]
    WriteFailed { leg: WriteLeg, cause: String },

    /// A write failed in one of its legs and state was, or may have been,
    /// left behind. `left_behind` says exactly what: the rollback itself
    /// failed, or the leg is one that does not roll back.
    #[error("write failed in the {leg} leg: {cause}; {left_behind}")]
    WriteNotRolledBack {
        leg: WriteLeg,
        cause: String,
        left_behind: String,
    },
}

/// The leg of a write that failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteLeg {
    /// Embedding the text. Runs before anything is written.
    Embed,
    /// Writing ledger rows.
    Ledger,
    /// Upserting points, or the incidence rows derived with them.
    Index,
}

impl WriteLeg {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Ledger => "ledger",
            Self::Index => "index",
        }
    }
}

impl std::fmt::Display for WriteLeg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl MyelinError {
    /// A write that failed in `leg` and was rolled back.
    pub fn write_failed(leg: WriteLeg, cause: &MyelinError) -> Self {
        Self::WriteFailed {
            leg,
            cause: cause.to_string(),
        }
    }

    /// True for the two variants a write failure surfaces as: their text
    /// names the failing leg and its cause, and a caller is meant to read it.
    pub fn is_write_failure(&self) -> bool {
        matches!(self, Self::WriteFailed { .. } | Self::WriteNotRolledBack { .. })
    }

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
            Self::StoreDefinition(_) => "store-definition",
            Self::EmptyCompletion { .. } => "llm-empty-completion",
            Self::BudgetExhausted { .. } => "llm-budget-exhausted",
            Self::Io(_) => "io",
            Self::Serde(_) => "serde",
            Self::WriteFailed { .. } => "write-failed",
            Self::WriteNotRolledBack { .. } => "write-not-rolled-back",
        }
    }
}

pub type Result<T, E = MyelinError> = std::result::Result<T, E>;
