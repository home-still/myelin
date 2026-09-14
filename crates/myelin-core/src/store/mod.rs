//! Persistence (`PLAN.md` §5).
//!
//! Qdrant holds the vectors and answers "which records are similar"; SQLite
//! holds truth, time, permission and lineage and answers "which records may be
//! seen and are still believed". Neither is authoritative alone, which is why
//! [`reconcile`] is a first-class operation rather than a maintenance script.

pub mod export;
#[cfg(feature = "graph")]
pub mod graph;
pub mod ids;
pub mod ledger;
pub mod qdrant;
pub mod reconcile;

pub use export::{export_namespace, import_namespace, ExportBundle, MemoryConfigJson};
pub use ledger::Ledger;
pub use qdrant::QdrantStore;
pub use reconcile::{reconcile, DriftReport};
