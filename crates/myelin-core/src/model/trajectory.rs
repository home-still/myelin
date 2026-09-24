//! Agent histories, stored state by state (M62).
//!
//! The episodic records keep an agent trajectory as merged search chunks:
//! consecutive step turns share an episode of up to ~512 tokens and
//! provenance names only the first turn's source, so one state cannot be read
//! back exactly. That serves "find where X happened". It cannot serve "open
//! state 7", which is what a file-reading controller does. AgentRunbook-C, the
//! LongMemEval-V2 authors' own memory (`10.48550/arXiv.2605.12493` §4.2),
//! keeps every trajectory as a file and lets the controller open spans of
//! it. This is myelin's structured copy for the same job, written beside the
//! episodic records and read only by the trajectory tools
//! (`docs/measurements/m62-native-trajectory-tools.md`).

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{MyelinError, Result};
use crate::model::record::Scope;

/// What a trajectory set out to do and how it ended, without its states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrajectoryHeader {
    /// The trajectory's own id, unique within its tenant and namespace.
    pub id: String,
    pub scope: Scope,
    /// The trajectory's goal episode record. Lineage for every evidence item
    /// cut from this trajectory, and its anchor: forgetting that record
    /// (C11 `hard_delete`) forgets the structured copy with it (I5).
    pub record_id: Uuid,
    pub goal: String,
    pub environment: String,
    pub start_url: String,
    /// `success` or `failure`, as recorded.
    pub outcome: String,
}

/// One state: the page the agent saw and the step that led to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrajectoryState {
    pub state_index: u32,
    pub step: u32,
    pub url: String,
    /// `None` on the initial state, which has a page and no preceding action.
    pub action: Option<String>,
    pub thought: Option<String>,
    pub accessibility_tree: String,
}

/// One state's step without its page: what a trajectory summary lists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrajectoryStep {
    pub state_index: u32,
    pub step: u32,
    pub url: String,
    pub action: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentTrajectory {
    pub header: TrajectoryHeader,
    pub states: Vec<TrajectoryState>,
}

impl AgentTrajectory {
    /// Refuse a trajectory the tools could not read back in order: no states,
    /// or state indices that are not strictly increasing. A span is a range
    /// of state indices, so a repeated or out-of-order index would make
    /// "states 3-7" mean two different things.
    pub fn validate(&self) -> Result<()> {
        let id = &self.header.id;
        if id.trim().is_empty() {
            return Err(MyelinError::Store("trajectory with an empty id".into()));
        }
        if self.header.scope.session.is_some() {
            return Err(MyelinError::Store(format!(
                "trajectory {id}: a stored trajectory is not session-scoped"
            )));
        }
        if self.states.is_empty() {
            return Err(MyelinError::Store(format!("trajectory {id} has no states")));
        }
        for pair in self.states.windows(2) {
            if pair[1].state_index <= pair[0].state_index {
                return Err(MyelinError::Store(format!(
                    "trajectory {id}: state index {} follows {}; indices must strictly increase",
                    pair[1].state_index, pair[0].state_index
                )));
            }
        }
        Ok(())
    }
}
