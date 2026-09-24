//! The data model (`PLAN.md` §4). One record type, four kinds, explicit time
//! and explicit trust; one 4-op delta that is the only way the store mutates.

pub mod delta;
pub mod evidence;
pub mod query;
pub mod record;
pub mod trajectory;

pub use delta::{AppliedDelta, Delta};
pub use evidence::{EvidenceItem, EvidenceKind, EvidenceSet, TraceStep, WireItem};
pub use query::{Budget, Mode, Recall, ScopeFilter};
pub use record::{
    ActorId, EntityRef, Link, LinkKind, MemoryRecord, Provenance, RecordKind, Salience, Scope,
    SourceRef, Span, Trust, TrustTier, Validity,
};
pub use trajectory::{AgentTrajectory, TrajectoryHeader, TrajectoryState, TrajectoryStep};
