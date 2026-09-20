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

use super::record::{SourceRef, TrustTier};

/// `JsonSchema` is derived here and on [`WireItem`] so the MCP tool schema is
/// generated from the *same* struct the benchmark contract is defined by. A
/// hand-mirrored copy in `myelin-mcp` would be a second place for R1 to drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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
    /// The record's trust tier, carried so a consumer can discount it.
    ///
    /// Measured necessity: E1 injected paraphrased poison that the pattern
    /// gate misses, and the reader repeated the attacker's payload in
    /// **80-100%** of answers. The store knew all along — poison lands at
    /// `Untrusted` (score 0.30) and first-party memory at `Verified` (0.90)
    /// — and the read path was throwing that away.
    pub trust: TrustTier,
}

/// The exact JSON object shape R1 mandates. `type` is a Rust keyword-adjacent
/// name, hence the rename.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
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
    /// Candidates `compose` had in hand, had room for under `k`, and
    /// skipped because they did not fit `max_tokens` (M28).
    ///
    /// **Which of the two truncation limits actually bit.** `compose`
    /// stops on `k` *or* on the token budget, and every width measurement
    /// M25–M27 made ran at `Budget::default()` — `k = 6`, `tokens = 2048`
    /// — without recording which one bound. Measured offline over the
    /// 162,181 live LongMemEval_S records, mean cost is **380 tokens**, so
    /// six of them is 2,282 and the *budget* binds before `k` does; on
    /// LoCoMo's 56-token mean it does not. A "truncation loss" that is
    /// really a token-budget loss has a different fix, so the two are
    /// counted apart.
    ///
    /// Non-zero also flags a confound: the selection loop `continue`s
    /// rather than breaking, so a bound budget silently prefers *shorter*
    /// records, and a mechanism that promotes short ones can win for a
    /// reason that is not relevance.
    #[serde(default)]
    pub dropped_for_tokens: usize,
    /// `compose` filled every one of `k` slots and still had candidates
    /// left. With `dropped_for_tokens == 0` this is a clean `k` bind.
    #[serde(default)]
    pub k_bound: bool,
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
