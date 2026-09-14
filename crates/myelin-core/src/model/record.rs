//! `MemoryRecord` and its parts (`PLAN.md` §4). One record type, four kinds,
//! explicit time and explicit trust.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{MyelinError, Result};

/// Four kinds, one table. `Working` is scratch state; `Episodic` is never
/// evicted from the ledger (§8), only from the hot index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind {
    Episodic,
    Semantic,
    Procedural,
    Working,
}

impl RecordKind {
    pub const ALL: [RecordKind; 4] = [
        RecordKind::Episodic,
        RecordKind::Semantic,
        RecordKind::Procedural,
        RecordKind::Working,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            RecordKind::Episodic => "episodic",
            RecordKind::Semantic => "semantic",
            RecordKind::Procedural => "procedural",
            RecordKind::Working => "working",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "episodic" => Ok(RecordKind::Episodic),
            "semantic" => Ok(RecordKind::Semantic),
            "procedural" => Ok(RecordKind::Procedural),
            "working" => Ok(RecordKind::Working),
            other => Err(MyelinError::Store(format!("unknown record kind {other:?}"))),
        }
    }
}

/// The scope-filter key. `tenant` is mandatory on every read path — per-tenant
/// isolation is the primary defense against MINJA-style poisoning (C12, §9),
/// because isolation removes the shared-bank premise entirely.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Scope {
    pub tenant: String,
    pub agent: String,
    pub session: Option<String>,
    pub namespace: String,
}

impl Scope {
    pub fn new(
        tenant: impl Into<String>,
        agent: impl Into<String>,
        namespace: impl Into<String>,
    ) -> Self {
        Self {
            tenant: tenant.into(),
            agent: agent.into(),
            session: None,
            namespace: namespace.into(),
        }
    }

    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session = Some(session.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActorId(pub String);

impl ActorId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Where a record came from: a doc/chunk/turn id plus an optional line or turn
/// range. Non-optional in Rust — a record without a source cannot be
/// constructed. I2 is enforced at the SQLite boundary instead (see
/// [`crate::store::ledger`]), because that is the only place a null can appear:
/// external import, manual edit, or drift.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    pub doc: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
}

impl SourceRef {
    pub fn doc(doc: impl Into<String>) -> Self {
        Self {
            doc: doc.into(),
            span: None,
        }
    }

    pub fn span(doc: impl Into<String>, start: u32, end: u32) -> Self {
        Self {
            doc: doc.into(),
            span: Some(Span { start, end }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

/// Immutable source pointer (C1, §9). `derived_from` is the consolidation
/// lineage that makes "show me why you believe that" answerable (I4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub source: SourceRef,
    pub contributed_by: ActorId,
    pub written_by: ActorId,
    #[serde(default)]
    pub derived_from: Vec<Uuid>,
}

/// Bi-temporal validity (Zep/Graphiti, `10.48550/arxiv.2501.13956`).
///
/// `t_valid` is when the fact became true in the world; `t_ingested` is when we
/// learned it. `t_invalid` is set by UPDATE/DELETE and never overwritten (I1);
/// `t_expired` is when we stopped believing our own record (§8 TTL).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Validity {
    pub t_valid: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_invalid: Option<DateTime<Utc>>,
    pub t_ingested: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_expired: Option<DateTime<Utc>>,
}

impl Validity {
    /// Valid now and not retracted or expired.
    pub fn is_live_at(&self, now: DateTime<Utc>) -> bool {
        self.t_valid <= now
            && self.t_invalid.is_none_or(|t| t > now)
            && self.t_expired.is_none_or(|t| t > now)
    }
}

/// C4. `Quarantined` is invisible to every read path except the explicit review
/// tool (I3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustTier {
    Verified,
    Asserted,
    Untrusted,
    Quarantined,
}

impl TrustTier {
    pub fn as_str(self) -> &'static str {
        match self {
            TrustTier::Verified => "verified",
            TrustTier::Asserted => "asserted",
            TrustTier::Untrusted => "untrusted",
            TrustTier::Quarantined => "quarantined",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "verified" => Ok(TrustTier::Verified),
            "asserted" => Ok(TrustTier::Asserted),
            "untrusted" => Ok(TrustTier::Untrusted),
            "quarantined" => Ok(TrustTier::Quarantined),
            other => Err(MyelinError::Store(format!("unknown trust tier {other:?}"))),
        }
    }

    /// A confidence score is not a safety filter — a Gemini guard agent accepted
    /// 54 malicious entries at trust = 1.0 (`05-security-governance.md` §3), so
    /// admissibility keys off the *tier*, never the score.
    pub fn is_readable(self) -> bool {
        !matches!(self, TrustTier::Quarantined)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trust {
    pub tier: TrustTier,
    pub score: f32,
    #[serde(default)]
    pub checks: Vec<String>,
}

impl Trust {
    pub fn asserted() -> Self {
        Self {
            tier: TrustTier::Asserted,
            score: 0.5,
            checks: Vec::new(),
        }
    }
}

/// Decay state (§8). `strength` is Ebbinghaus `S`, strengthened on access.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Salience {
    pub importance: f32,
    pub access_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_access: Option<DateTime<Utc>>,
    pub strength: f32,
}

impl Default for Salience {
    fn default() -> Self {
        Self {
            importance: 0.5,
            access_count: 0,
            last_access: None,
            strength: 1.0,
        }
    }
}

/// A graph seed: a phrase node in the bipartite incidence structure (§5.4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityRef {
    pub phrase: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

impl EntityRef {
    pub fn new(phrase: impl Into<String>) -> Self {
        Self {
            phrase: phrase.into(),
            kind: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    /// Written by UPDATE: the new record supersedes the old one.
    Supersedes,
    Contradicts,
    /// Consolidation lineage; the edge form of `Provenance::derived_from`.
    DerivedFrom,
    CoEpisode,
}

impl LinkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            LinkKind::Supersedes => "supersedes",
            LinkKind::Contradicts => "contradicts",
            LinkKind::DerivedFrom => "derived_from",
            LinkKind::CoEpisode => "co_episode",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "supersedes" => Ok(LinkKind::Supersedes),
            "contradicts" => Ok(LinkKind::Contradicts),
            "derived_from" => Ok(LinkKind::DerivedFrom),
            "co_episode" => Ok(LinkKind::CoEpisode),
            other => Err(MyelinError::Store(format!("unknown link kind {other:?}"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Link {
    pub relation: LinkKind,
    pub target: Uuid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// v5 over (namespace, natural key) — idempotent re-ingest. See
    /// [`crate::store::ids`].
    pub id: Uuid,
    pub kind: RecordKind,
    pub scope: Scope,
    /// The only thing a model ever reads.
    pub text: String,
    #[serde(default)]
    pub entities: Vec<EntityRef>,
    pub validity: Validity,
    pub provenance: Provenance,
    pub trust: Trust,
    #[serde(default)]
    pub salience: Salience,
    #[serde(default)]
    pub links: Vec<Link>,
}

impl MemoryRecord {
    /// Admissible to a read path: live, readable tier. Provenance is guaranteed
    /// by the type; the null case is caught at the SQLite boundary (I2).
    pub fn is_admissible_at(&self, now: DateTime<Utc>) -> bool {
        self.trust.tier.is_readable() && self.validity.is_live_at(now)
    }

    /// I4 applies to `Semantic` records only: an abstracted fact must say what
    /// it was abstracted from.
    pub fn requires_lineage(&self) -> bool {
        matches!(self.kind, RecordKind::Semantic)
    }
}
