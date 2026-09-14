//! `EvidenceSet` — what `compose()` returns and the MCP contract type
//! (`PLAN.md` §7.3).
//!
//! **R1 is a wire-format constraint, not a preference.** The LongMemEval-V2
//! `Memory` base class requires `query()` to return
//! `list[{type: "text"|"image", value: str}]`. So [`EvidenceSet::to_wire`]
//! emits exactly that — no bespoke envelope, no extra keys. Everything we need
//! and the benchmark does not (record ids, provenance, scores) stays on
//! [`EvidenceItem`] and never reaches the wire.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::record::SourceRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Text,
    Image,
}

/// One piece of evidence, with the provenance that makes I2 checkable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub kind: EvidenceKind,
    pub value: String,
    pub record_id: Uuid,
    pub source: SourceRef,
    pub score: f32,
}

/// The exact JSON object shape R1 mandates. `type` is a Rust keyword-adjacent
/// name, hence the rename.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireItem {
    #[serde(rename = "type")]
    pub kind: EvidenceKind,
    pub value: String,
}

/// Budgeted, ordered, provenance-carrying. Not a blob.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSet {
    pub items: Vec<EvidenceItem>,
    /// Token count of the composed evidence, for the budget accounting in §7.3.
    pub tokens: usize,
    /// `investigate` only: the search/read/reflect trace (§7.2).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trace: Vec<TraceStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceStep {
    pub step: usize,
    pub action: String,
    pub query: String,
    pub hits: usize,
}

impl EvidenceSet {
    /// The R1 wire form. This is what the benchmark adapter returns verbatim.
    pub fn to_wire(&self) -> Vec<WireItem> {
        self.items
            .iter()
            .map(|i| WireItem {
                kind: i.kind,
                value: i.value.clone(),
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
