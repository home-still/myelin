//! The 4-op write delta (`PLAN.md` §2 finding 5; Mem0 `10.48550/arxiv.2504.19413`,
//! Memory-R1 `10.48550/arXiv.2508.19828`).
//!
//! This is the **only** way the store mutates. `DELETE` is what makes temporal
//! consistency work — systems without it fail knowledge-update questions.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::record::MemoryRecord;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Delta {
    /// Insert a new record.
    Add { record: Box<MemoryRecord> },
    /// Write `replacement` as a *new* record and set `target`'s `t_invalid`.
    /// Never an in-place edit (I1).
    Update {
        target: Uuid,
        replacement: Box<MemoryRecord>,
        reason: String,
    },
    /// Set `target`'s `t_invalid`. The row and its ledger history remain.
    Delete { target: Uuid, reason: String },
    /// Explicitly decided to do nothing. Recorded, because "we considered this
    /// and declined" is audit-relevant (C10).
    Noop {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target: Option<Uuid>,
        reason: String,
    },
}

impl Delta {
    pub fn op(&self) -> &'static str {
        match self {
            Delta::Add { .. } => "add",
            Delta::Update { .. } => "update",
            Delta::Delete { .. } => "delete",
            Delta::Noop { .. } => "noop",
        }
    }

    /// The record this delta is about, for audit indexing.
    pub fn subject(&self) -> Option<Uuid> {
        match self {
            Delta::Add { record } => Some(record.id),
            Delta::Update { target, .. } | Delta::Delete { target, .. } => Some(*target),
            Delta::Noop { target, .. } => *target,
        }
    }

    pub fn reason(&self) -> &str {
        match self {
            Delta::Add { .. } => "add",
            Delta::Update { reason, .. }
            | Delta::Delete { reason, .. }
            | Delta::Noop { reason, .. } => reason,
        }
    }
}

/// What `Ledger::apply` actually did. Returned so the caller can audit and so
/// the indexer knows which records need vectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppliedDelta {
    pub op: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<Uuid>,
    /// Records newly written; these need indexing.
    #[serde(default)]
    pub written: Vec<Uuid>,
    /// Records whose `t_invalid` was set; these leave the hot index.
    #[serde(default)]
    pub invalidated: Vec<Uuid>,
}
