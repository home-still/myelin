//! Ledger ↔ Qdrant drift repair (`PLAN.md` §5.3, C9).
//!
//! SSGM Theorem 1 bounds dual-store drift at `O(N·ε_step)` where **`N` is the
//! reconciliation *interval*** — steps between reconciliations, not the total
//! horizon. Drift therefore depends on the cadence we choose, which makes
//! reconciliation a tunable drift budget rather than a cleanup chore. This is
//! the same drift-repair pattern `hs catalog repair` already uses.
//!
//! Six directions, each independently detected and independently repairable:
//!
//! | direction | meaning | repair |
//! |---|---|---|
//! | `qdrant_orphans` | point in Qdrant, no ledger record | delete the point |
//! | `missing_vectors` | ledger record, no Qdrant point | re-embed and upsert |
//! | `payload_drift` | Qdrant payload disagrees with the ledger | overwrite payload |
//! | `stale_points` | point for a record that is no longer live | delete the point |
//! | `dangling_links` | link whose endpoint is gone | delete the link |
//! | `orphan_incidence` | incidence row for a missing record | delete the row |
//! | `missing_provenance` | ledger row with NULL `prov_source` | quarantine it |
//!
//! Only `missing_vectors` needs an [`Embedder`]; the rest repair with no model
//! in the loop, so a reconcile can run on a workstation with no GPU claim.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::embed::Embedder;
use crate::error::Result;
use crate::model::record::{ActorId, LinkKind};
use crate::store::ledger::Ledger;
use crate::store::qdrant::{IndexItem, QdrantStore};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DriftReport {
    pub qdrant_orphans: Vec<Uuid>,
    pub missing_vectors: Vec<Uuid>,
    pub payload_drift: Vec<Uuid>,
    pub stale_points: Vec<Uuid>,
    pub dangling_links: Vec<(Uuid, Uuid, LinkKind)>,
    pub orphan_incidence: Vec<(String, Uuid)>,
    pub missing_provenance: Vec<Uuid>,
    /// True when the repairs in this report were applied, not just detected.
    pub repaired: bool,
}

impl DriftReport {
    pub fn total(&self) -> usize {
        self.qdrant_orphans.len()
            + self.missing_vectors.len()
            + self.payload_drift.len()
            + self.stale_points.len()
            + self.dangling_links.len()
            + self.orphan_incidence.len()
            + self.missing_provenance.len()
    }

    pub fn is_clean(&self) -> bool {
        self.total() == 0
    }
}

/// Detect drift, and repair it when `apply` is set.
///
/// `embedder` is only consulted for `missing_vectors`. Pass `None` to run a
/// model-free reconcile; those records are then reported but left unrepaired.
pub async fn reconcile(
    ledger: &Ledger,
    store: &QdrantStore,
    namespace: &str,
    embedder: Option<&dyn Embedder>,
    apply: bool,
) -> Result<DriftReport> {
    let now = Utc::now();
    let mut report = DriftReport::default();

    let records = ledger.records_in_namespace(namespace).await?;
    let points = store.scroll_namespace(namespace).await?;

    let by_id: std::collections::HashMap<Uuid, _> =
        records.iter().map(|r| (r.id, r)).collect();
    let point_ids: std::collections::HashMap<Uuid, _> = points.into_iter().collect();

    // Direction 1: points with no ledger record.
    for id in point_ids.keys() {
        if !by_id.contains_key(id) {
            report.qdrant_orphans.push(*id);
        }
    }
    report.qdrant_orphans.sort_unstable();

    // Directions 2-4: ledger records vs their points.
    for record in &records {
        let live = record.is_admissible_at(now);
        match point_ids.get(&record.id) {
            None => {
                if live {
                    report.missing_vectors.push(record.id);
                }
            }
            Some(snapshot) => {
                if !live {
                    // A retracted or quarantined record must not stay in the
                    // hot index; leaving it there is how I3 leaks.
                    report.stale_points.push(record.id);
                } else {
                    // Every field the snapshot carries is compared. The list
                    // used to be hand-maintained and omitted `t_valid` —
                    // which is the one field whose corruption M19 shipped:
                    // 162,181 records stamped with the build date, invisible
                    // to this check for six milestones. `text` and
                    // `entities` stay out of the snapshot because they are
                    // large and because drift in either changes the
                    // content-derived record id, which direction 1 catches.
                    let drifted = snapshot.tenant != record.scope.tenant
                        || snapshot.agent != record.scope.agent
                        || snapshot.session != record.scope.session
                        || snapshot.namespace != record.scope.namespace
                        || snapshot.kind != record.kind.as_str()
                        || snapshot.trust_tier != record.trust.tier.as_str()
                        || snapshot.t_valid != record.validity.t_valid.timestamp()
                        || snapshot.t_invalid
                            != record.validity.t_invalid.map(|t| t.timestamp());
                    if drifted {
                        report.payload_drift.push(record.id);
                    }
                }
            }
        }
    }

    report.dangling_links = ledger
        .dangling_links()
        .await?
        .into_iter()
        .map(|l| (l.src, l.dst, l.relation))
        .collect();
    report.orphan_incidence = ledger
        .orphan_incidence()
        .await?
        .into_iter()
        .map(|i| (i.phrase, i.record_id))
        .collect();
    report.missing_provenance = ledger.records_missing_provenance().await?;

    if !apply {
        return Ok(report);
    }

    // ── Repair ──────────────────────────────────────────────────

    if !report.qdrant_orphans.is_empty() {
        store.delete_points(&report.qdrant_orphans).await?;
    }
    if !report.stale_points.is_empty() {
        store.delete_points(&report.stale_points).await?;
    }
    for id in &report.payload_drift {
        if let Some(record) = by_id.get(id) {
            store.sync_payload(record).await?;
        }
    }
    if let Some(embedder) = embedder {
        let pending: Vec<_> = report
            .missing_vectors
            .iter()
            .filter_map(|id| by_id.get(id).copied())
            .collect();
        if !pending.is_empty() {
            let texts: Vec<String> = pending.iter().map(|r| r.text.clone()).collect();
            let vectors = embedder.embed(&texts).await?;
            let items: Vec<IndexItem> = pending
                .iter()
                .zip(vectors)
                .map(|(record, dense)| IndexItem {
                    record,
                    dense,
                    late: None,
                })
                .collect();
            store.upsert(&items).await?;
        }
    }
    for (src, dst, relation) in &report.dangling_links {
        ledger.delete_link(*src, *dst, *relation).await?;
    }
    for (phrase, record_id) in &report.orphan_incidence {
        ledger.delete_incidence(phrase, *record_id).await?;
    }
    for id in &report.missing_provenance {
        // Cannot invent a source, so make it unreadable instead. I2 already
        // hides it from `visible()`; this makes the state explicit and audited.
        ledger
            .quarantine_existing(*id, "reconcile: record has no provenance source (I2)")
            .await?;
    }

    ledger
        .log(
            "reconcile",
            None,
            &ActorId::new("system"),
            "drift repair",
            serde_json::to_value(&report)?,
        )
        .await?;

    report.repaired = true;
    Ok(report)
}
