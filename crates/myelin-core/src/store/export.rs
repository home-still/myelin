//! Namespace export / import (`PLAN.md` §3.1, R3).
//!
//! **R3 is an ABI requirement, not a convenience.** LongMemEval-V2's
//! `reconcile_loaded_memory_config` *requires* the requested config to equal
//! the saved config when loading a prebuilt artifact. So a built memory must
//! round-trip byte-faithfully, and the emitted config must be a pure function
//! of the build inputs — no timestamps, no host names, no run ids.
//!
//! Byte-faithfulness is tested by
//! `export → canonical_json → import → export → canonical_json`, asserting the
//! two strings are identical. Every query that feeds the bundle carries an
//! `ORDER BY`, because determinism here is the whole property.

use serde::{Deserialize, Serialize};

use crate::error::{MyelinError, Result};
use crate::model::record::MemoryRecord;
use crate::store::ledger::{AclEdge, Event, IncidenceRow, Ledger, LinkRow, QuarantineRow};

/// Bumped whenever the on-disk bundle shape changes. A mismatch is a hard
/// error: silently importing an older bundle is how a "byte-faithful" claim
/// quietly becomes false.
pub const BUNDLE_VERSION: u32 = 1;

/// The pinned build configuration. Equality is checked on import (R3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryConfigJson {
    pub schema_version: u32,
    /// [`crate::embed::Embedder::id`] — changing the embedder invalidates the
    /// built memory.
    pub embedder: String,
    pub dense_dim: u32,
    pub distance: String,
    pub fusion: FusionConfig,
    pub late_interaction: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionConfig {
    pub kind: String,
    /// Cormack's canonical constant. Qdrant's server-side RRF uses k = 1
    /// (measured, §5.2), which is why fusion is ours and this is a parameter.
    pub k: u32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            kind: "rrf".into(),
            k: 60,
        }
    }
}

impl MemoryConfigJson {
    pub fn new(embedder: impl Into<String>, dense_dim: u32) -> Self {
        Self {
            schema_version: BUNDLE_VERSION,
            embedder: embedder.into(),
            dense_dim,
            distance: "cosine".into(),
            fusion: FusionConfig::default(),
            late_interaction: false,
        }
    }
}

/// Everything needed to rebuild a namespace's ledger state exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportBundle {
    pub version: u32,
    pub namespace: String,
    pub config: MemoryConfigJson,
    pub records: Vec<MemoryRecord>,
    pub links: Vec<LinkRow>,
    pub incidence: Vec<IncidenceRow>,
    pub quarantine: Vec<QuarantineRow>,
    pub acl_ua: Vec<AclEdge>,
    pub acl_ar: Vec<AclEdge>,
    pub events: Vec<Event>,
}

/// Deterministic serialization. `serde_json` preserves struct field order and
/// every collection is already sorted by its query, so this is stable across
/// processes and machines.
pub fn canonical_json(bundle: &ExportBundle) -> Result<String> {
    Ok(serde_json::to_string_pretty(bundle)?)
}

pub async fn export_namespace(
    ledger: &Ledger,
    namespace: &str,
    config: &MemoryConfigJson,
) -> Result<ExportBundle> {
    let records = ledger.records_in_namespace(namespace).await?;
    let ids: std::collections::HashSet<_> = records.iter().map(|r| r.id).collect();

    let links = ledger.links(Some(namespace)).await?;
    let incidence = ledger.incidence(Some(namespace)).await?;

    let mut quarantine: Vec<QuarantineRow> = ledger
        .review_quarantine(i64::MAX)
        .await?
        .into_iter()
        .filter(|q| q.record.scope.namespace == namespace)
        .collect();
    quarantine.sort_by_key(|q| q.id);

    let (acl_ua, acl_ar) = ledger.acl_edges().await?;

    let events: Vec<Event> = ledger
        .events(None)
        .await?
        .into_iter()
        .filter(|e| e.record_id.is_some_and(|id| ids.contains(&id)))
        .collect();

    Ok(ExportBundle {
        version: BUNDLE_VERSION,
        namespace: namespace.to_string(),
        config: config.clone(),
        records,
        links,
        incidence,
        quarantine,
        acl_ua,
        acl_ar,
        events,
    })
}

/// Load a bundle into a ledger.
///
/// Fails if `requested` differs from the bundle's saved config — this is the
/// Rust side of `reconcile_loaded_memory_config` (R3). It also fails on a
/// version mismatch.
///
/// Records are inserted directly rather than through `apply`, because a
/// re-import must reproduce the *stored* state — including already-invalidated
/// records and quarantined material — not replay a write path that would
/// re-derive new timestamps.
pub async fn import_namespace(
    ledger: &Ledger,
    bundle: &ExportBundle,
    requested: &MemoryConfigJson,
) -> Result<()> {
    if bundle.version != BUNDLE_VERSION {
        return Err(MyelinError::Store(format!(
            "R3: bundle version {} != supported {BUNDLE_VERSION}",
            bundle.version
        )));
    }
    if &bundle.config != requested {
        return Err(MyelinError::Store(format!(
            "R3: requested memory config does not equal the saved config\n  saved:     {:?}\n  requested: {:?}",
            bundle.config, requested
        )));
    }
    ledger.import_bundle(bundle).await
}
