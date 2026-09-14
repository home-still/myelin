//! Read-side request types (`PLAN.md` §7).
//!
//! **R4:** `recall` vs `investigate`, `k`, `rrf_k` and step budgets are all
//! *query-time* parameters against one identical store. Any design where
//! `investigate` needs its own index cannot produce a multi-operating-point
//! leaderboard submission — which, per the LAFS arithmetic, is where the score
//! comes from.

use serde::{Deserialize, Serialize};

use super::record::RecordKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Fast path, no LLM in the loop (§7.1).
    Recall,
    /// Agentic search → read → reflect → search again (§7.2).
    Investigate,
}

/// Scope predicates applied *before* ranking (ShardMemo scope-before-routing,
/// §2 finding 3). `tenant` is mandatory: there is no read path that can span
/// tenants (C12).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeFilter {
    pub tenant: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

impl ScopeFilter {
    pub fn tenant(tenant: impl Into<String>) -> Self {
        Self {
            tenant: tenant.into(),
            agent: None,
            session: None,
            namespace: None,
        }
    }

    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.agent = Some(agent.into());
        self
    }

    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session = Some(session.into());
        self
    }
}

/// Evidence-set bloat destroys precision: HiGMem retrieves 8.09 vs 99.84
/// turns/query at P@K 0.1909 vs 0.0101 (§2 finding 4). `k` is small on purpose,
/// and it is a *security* parameter too — MINJA ASR climbs 6% → 20% → 38% as
/// k goes 3 → 5 → 10 (§2 finding 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    pub k: usize,
    pub tokens: usize,
    /// `investigate` only: iteration cap.
    pub max_steps: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            k: 6,
            tokens: 2048,
            max_steps: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recall {
    pub scope: ScopeFilter,
    pub text: String,
    #[serde(default)]
    pub budget: Budget,
    pub mode: Mode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kinds: Option<Vec<RecordKind>>,
}

impl Recall {
    pub fn fast(scope: ScopeFilter, text: impl Into<String>) -> Self {
        Self {
            scope,
            text: text.into(),
            budget: Budget::default(),
            mode: Mode::Recall,
            kinds: None,
        }
    }
}
