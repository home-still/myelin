//! `reconcile` repairs an artificially drifted store — the third part of the
//! M1 exit criterion.
//!
//! Requires the live Qdrant on `big`; gated behind the `integration` feature
//! like the capability tests. Uses a uniquely-named `myelin_test_*` scratch
//! collection and deletes it at the end; the production collections are never
//! touched.
//!
//! The drift injected here is not hypothetical. Every direction corresponds to
//! a failure mode the `hs` pipeline has actually produced: points surviving a
//! deleted catalog row, catalog rows whose vectors never landed, payload flags
//! that stopped matching the projection, and link/incidence rows pointing at
//! records that are gone.

#![cfg(feature = "integration")]

use async_trait::async_trait;
use chrono::{Duration, Utc};
use myelin_core::config::{MyelinConfig, QdrantConfig};
use myelin_core::embed::Embedder;
use myelin_core::error::Result;
use myelin_core::model::delta::Delta;
use myelin_core::model::record::{
    ActorId, EntityRef, LinkKind, MemoryRecord, Provenance, RecordKind, Salience, Scope, SourceRef,
    Trust, Validity,
};
use myelin_core::store::ids::record_id;
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::{IndexItem, QdrantStore};
use myelin_core::store::reconcile::reconcile;
use uuid::Uuid;

const NS: &str = "ns-reconcile";
const DIM: u64 = 4;

/// Deterministic stand-in for bge-m3. The reconciler's job is to notice a
/// missing vector and put *a* correct one back; which model produced it is
/// M3's concern, and a fake keeps this test free of GPU tenancy.
struct FakeEmbedder;

#[async_trait]
impl Embedder for FakeEmbedder {
    fn dim(&self) -> u64 {
        DIM
    }
    fn id(&self) -> &str {
        "fake-4d"
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; DIM as usize];
                for (i, b) in t.bytes().enumerate() {
                    v[i % DIM as usize] += b as f32;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0);
                v.iter().map(|x| x / norm).collect()
            })
            .collect())
    }
}

fn qdrant_config() -> QdrantConfig {
    MyelinConfig::load().expect("load MyelinConfig").qdrant
}

/// Deletes its scratch collection on drop — including when the test panics.
///
/// The happy path still deletes explicitly, because that proves cleanup works.
/// This is the backstop for the failure path: a failing assertion must not
/// leave a collection behind on a Qdrant that also holds production data.
/// `Drop` cannot await, so the delete runs on a throwaway runtime.
struct ScratchGuard {
    name: String,
}

impl ScratchGuard {
    fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        let name = std::mem::take(&mut self.name);
        let _ = std::thread::spawn(move || {
            let Ok(rt) = tokio::runtime::Runtime::new() else {
                return;
            };
            rt.block_on(async {
                if let Ok(store) = QdrantStore::with_collection(&qdrant_config(), &name) {
                    let _ = store.drop_collection().await;
                }
            });
        })
        .join();
    }
}

fn scope() -> Scope {
    Scope::new("tenant-r", "agent-1", NS)
}

fn record(key: &str, text: &str) -> MemoryRecord {
    let sc = scope();
    let now = Utc::now();
    MemoryRecord {
        id: record_id(&sc, key),
        kind: RecordKind::Episodic,
        scope: sc,
        text: text.to_string(),
        entities: vec![EntityRef::new("alpha")],
        validity: Validity {
            t_valid: now - Duration::hours(1),
            t_invalid: None,
            t_ingested: now - Duration::hours(1),
            t_expired: None,
        },
        provenance: Provenance {
            source: SourceRef::doc("doc-1"),
            contributed_by: ActorId::new("user-1"),
            written_by: ActorId::new("agent-1"),
            derived_from: Vec::new(),
        },
        trust: Trust::asserted(),
        salience: Salience::default(),
        links: Vec::new(),
    }
}

#[tokio::test]
async fn reconcile_detects_and_repairs_every_drift_direction() {
    let collection = format!("myelin_test_reconcile_{}", Uuid::new_v4().simple());
    let _guard = ScratchGuard::new(&collection);
    let store = QdrantStore::with_collection(&qdrant_config(), &collection).expect("store");
    store.ensure_collection(DIM, false).await.expect("create");

    let ledger = Ledger::open_memory().await.unwrap();
    let actor = ActorId::new("test");
    let embedder = FakeEmbedder;

    // ── A clean, fully indexed store ────────────────────────────
    let mut records = Vec::new();
    for i in 0..5 {
        let r = record(&format!("rec-{i}"), &format!("memory number {i}"));
        ledger
            .apply(
                &Delta::Add {
                    record: Box::new(r.clone()),
                },
                &actor,
            )
            .await
            .unwrap();
        records.push(r);
    }
    let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
    let vectors = embedder.embed(&texts).await.unwrap();
    let items: Vec<IndexItem> = records
        .iter()
        .zip(vectors)
        .map(|(record, dense)| IndexItem {
            record,
            dense,
            late: None,
        })
        .collect();
    store.upsert(&items).await.unwrap();

    let clean = reconcile(&ledger, &store, NS, Some(&embedder), false)
        .await
        .unwrap();
    assert!(
        clean.is_clean(),
        "a freshly built store must reconcile clean, got {clean:?}"
    );

    // ── Inject one instance of every drift direction ────────────

    // 1. missing vector: the point vanished, the record did not.
    store.delete_points(&[records[0].id]).await.unwrap();

    // 2. qdrant orphan: a point with no ledger record at all.
    let orphan_id = Uuid::new_v4();
    {
        use qdrant_client::qdrant::{NamedVectors, PointStruct, UpsertPointsBuilder, Vector};
        use qdrant_client::Payload;
        let mut payload: std::collections::HashMap<String, qdrant_client::qdrant::Value> = std::collections::HashMap::new();
        payload.insert("namespace".to_string(), NS.into());
        payload.insert("tenant".to_string(), "tenant-r".into());
        store
            .client()
            .upsert_points(
                UpsertPointsBuilder::new(
                    store.collection(),
                    vec![PointStruct::new(
                        orphan_id.to_string(),
                        NamedVectors::default()
                            .add_vector("dense", Vector::new_dense(vec![0.5f32; DIM as usize])),
                        Payload::from(payload),
                    )],
                )
                .wait(true),
            )
            .await
            .expect("insert orphan point");
    }

    // 3. payload drift: the index says this record is verified; it is not.
    {
        use qdrant_client::qdrant::{PointId, PointsIdsList, SetPayloadPointsBuilder};
        use qdrant_client::Payload;
        let mut payload: std::collections::HashMap<String, qdrant_client::qdrant::Value> = std::collections::HashMap::new();
        payload.insert("trust_tier".to_string(), "verified".into());
        store
            .client()
            .set_payload(
                SetPayloadPointsBuilder::new(store.collection(), Payload::from(payload))
                    .points_selector(PointsIdsList {
                        ids: vec![PointId::from(records[1].id.to_string())],
                    })
                    .wait(true),
            )
            .await
            .expect("drift payload");
    }

    // 4. stale point: retracted in the ledger, still in the hot index.
    ledger
        .apply(
            &Delta::Delete {
                target: records[2].id,
                reason: "drift test".into(),
            },
            &actor,
        )
        .await
        .unwrap();

    // 5. dangling link and 6. orphan incidence: endpoints that do not exist.
    let ghost = Uuid::new_v4();
    sqlx::query("INSERT INTO link (src, dst, relation, at) VALUES (?,?,?,?)")
        .bind(records[3].id.to_string())
        .bind(ghost.to_string())
        .bind("derived_from")
        .bind(Utc::now().to_rfc3339())
        .execute(ledger.pool())
        .await
        .unwrap();
    ledger.set_incidence("ghost-phrase", ghost, 1.0).await.unwrap();

    // 7. a record with no provenance source, as a corrupt import would leave.
    let unsourced = record("unsourced", "no source");
    sqlx::query(
        "INSERT INTO record (
            id, kind, tenant, agent, session, namespace, text, entities,
            t_valid, t_invalid, t_ingested, t_expired,
            trust_tier, trust_score, trust_checks,
            prov_source, prov_contributed_by, prov_written_by, prov_derived_from, salience
         ) VALUES (?,'episodic','tenant-r','agent-1',NULL,?,?,'[]',?,NULL,?,NULL,
                   'asserted',0.5,'[]',NULL,'u','a','[]','{}')",
    )
    .bind(unsourced.id.to_string())
    .bind(NS)
    .bind(&unsourced.text)
    .bind(unsourced.validity.t_valid.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true))
    .bind(unsourced.validity.t_ingested.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true))
    .execute(ledger.pool())
    .await
    .unwrap();

    // ── Detect ──────────────────────────────────────────────────

    let found = reconcile(&ledger, &store, NS, Some(&embedder), false)
        .await
        .unwrap();
    assert!(!found.repaired, "dry run must not claim repair");
    assert_eq!(found.missing_vectors, vec![records[0].id], "direction 1");
    assert_eq!(found.qdrant_orphans, vec![orphan_id], "direction 2");
    assert_eq!(found.payload_drift, vec![records[1].id], "direction 3");
    assert_eq!(found.stale_points, vec![records[2].id], "direction 4");
    assert_eq!(
        found.dangling_links,
        vec![(records[3].id, ghost, LinkKind::DerivedFrom)],
        "direction 5"
    );
    assert_eq!(
        found.orphan_incidence,
        vec![("ghost-phrase".to_string(), ghost)],
        "direction 6"
    );
    assert_eq!(found.missing_provenance, vec![unsourced.id], "direction 7");
    assert_eq!(found.total(), 7);

    // A dry run must change nothing.
    let again = reconcile(&ledger, &store, NS, Some(&embedder), false)
        .await
        .unwrap();
    assert_eq!(again.total(), 7, "dry run mutated the store");

    // ── Repair ──────────────────────────────────────────────────

    let repaired = reconcile(&ledger, &store, NS, Some(&embedder), true)
        .await
        .unwrap();
    assert!(repaired.repaired);
    assert_eq!(repaired.total(), 7, "repair pass must report what it fixed");

    let after = reconcile(&ledger, &store, NS, Some(&embedder), true)
        .await
        .unwrap();
    assert!(
        after.is_clean(),
        "store still drifted after repair: {after:?}"
    );

    // ── The repairs are the right repairs ───────────────────────

    assert!(
        store.get_payload(records[0].id).await.unwrap().is_some(),
        "missing vector was not re-indexed"
    );
    assert!(
        store.get_payload(orphan_id).await.unwrap().is_none(),
        "orphan point was not deleted"
    );
    assert_eq!(
        store
            .get_payload(records[1].id)
            .await
            .unwrap()
            .expect("record 1 point")
            .trust_tier,
        "asserted",
        "payload drift was not corrected"
    );
    assert!(
        store.get_payload(records[2].id).await.unwrap().is_none(),
        "retracted record is still in the hot index"
    );
    assert!(ledger.dangling_links().await.unwrap().is_empty());
    assert!(ledger.orphan_incidence().await.unwrap().is_empty());

    // The unsourced record could not be given a source, so it was quarantined
    // — still invisible (I2 and now I3), but explicitly so, and audited.
    let tier: String = sqlx::query_scalar("SELECT trust_tier FROM record WHERE id = ?")
        .bind(unsourced.id.to_string())
        .fetch_one(ledger.pool())
        .await
        .unwrap();
    assert_eq!(tier, "quarantined", "unsourced record was not quarantined");
    let visible = ledger
        .visible(
            &myelin_core::model::query::ScopeFilter::tenant("tenant-r").with_namespace(NS),
            Utc::now(),
            100,
        )
        .await
        .unwrap();
    assert!(
        !visible.iter().any(|r| r.id == unsourced.id),
        "unsourced record reached a read path"
    );
    let events = ledger.events(Some(unsourced.id)).await.unwrap();
    assert!(
        events.iter().any(|e| e.kind == "quarantine"),
        "quarantining an unsourced record must be audited (C10)"
    );

    store.drop_collection().await.expect("drop scratch collection");
}

/// A `t_valid` that drifts from the ledger's is drift, and the reconciler
/// must say so.
///
/// `payload_drift` compared a hand-maintained field list that omitted
/// `t_valid` — the one field whose corruption M19 actually shipped: 162,181
/// records stamped with the build date instead of the conversation's, which
/// survived six milestones because no check looked at it. The drift
/// detector's field list must be the payload's, not a transcription of it.
#[tokio::test]
async fn payload_drift_catches_a_mutated_t_valid() {
    let collection = format!("myelin_test_tvalid_{}", Uuid::new_v4().simple());
    let _guard = ScratchGuard::new(&collection);
    let store = QdrantStore::with_collection(&qdrant_config(), &collection).expect("store");
    store.ensure_collection(DIM, false).await.expect("create");

    let ledger = Ledger::open_memory().await.unwrap();
    let actor = ActorId::new("test");
    let embedder = FakeEmbedder;

    let r = record("tv-0", "the user moved to Berlin in May");
    ledger
        .apply(
            &Delta::Add {
                record: Box::new(r.clone()),
            },
            &actor,
        )
        .await
        .unwrap();
    let dense = embedder.embed(&[r.text.clone()]).await.unwrap().remove(0);
    store
        .upsert(&[IndexItem {
            record: &r,
            dense,
            late: None,
        }])
        .await
        .unwrap();

    let clean = reconcile(&ledger, &store, NS, Some(&embedder), false)
        .await
        .unwrap();
    assert!(clean.is_clean(), "fresh store must reconcile clean: {clean:?}");

    // Stamp the point with a different valid time and nothing else. Every
    // other payload field still matches the ledger exactly.
    {
        use qdrant_client::qdrant::{PointId, PointsIdsList, SetPayloadPointsBuilder};
        use qdrant_client::Payload;
        let mut payload: std::collections::HashMap<String, qdrant_client::qdrant::Value> =
            std::collections::HashMap::new();
        payload.insert(
            "t_valid".to_string(),
            (r.validity.t_valid.timestamp() + 86_400).into(),
        );
        store
            .client()
            .set_payload(
                SetPayloadPointsBuilder::new(store.collection(), Payload::from(payload))
                    .points_selector(PointsIdsList {
                        ids: vec![PointId::from(r.id.to_string())],
                    })
                    .wait(true),
            )
            .await
            .expect("drift t_valid");
    }

    let found = reconcile(&ledger, &store, NS, Some(&embedder), false)
        .await
        .unwrap();
    assert_eq!(
        found.payload_drift,
        vec![r.id],
        "a t_valid that disagrees with the ledger must be reported as drift"
    );

    // And repair puts it back, because `sync_payload` writes the whole
    // projection.
    let repaired = reconcile(&ledger, &store, NS, Some(&embedder), true)
        .await
        .unwrap();
    assert!(repaired.repaired);
    let after = reconcile(&ledger, &store, NS, Some(&embedder), false)
        .await
        .unwrap();
    assert!(after.is_clean(), "repair left drift behind: {after:?}");

    store.drop_collection().await.expect("drop scratch collection");
}

/// A point written *before* `t_valid` entered the payload carries `t_valid: None`
/// in the snapshot. That is a migration gap, *not* content drift: every invariant
/// still matches the ledger, so `total()`/`is_clean()` stay clean, and only the
/// new `missing_t_valid` count flags it. It must be repairable via `sync_payload`
/// — a `set_payload` merge that adds the field back — without re-embedding. This
/// is exactly the pre-migration corpus `myelin_locomo` carries (M19), and the
/// reason a plain reconcile must not error out on it.
#[tokio::test]
async fn pre_migration_points_migrate_t_valid_without_reembedding() {
    let collection = format!("myelin_test_tvalid_mig_{}", Uuid::new_v4().simple());
    let _guard = ScratchGuard::new(&collection);
    let store = QdrantStore::with_collection(&qdrant_config(), &collection).expect("store");
    store.ensure_collection(DIM, false).await.expect("create");

    let ledger = Ledger::open_memory().await.unwrap();
    let actor = ActorId::new("test");
    let embedder = FakeEmbedder;

    let rec = record("mig-0", "pre-migration memory");
    ledger
        .apply(&Delta::Add { record: Box::new(rec.clone()) }, &actor)
        .await
        .unwrap();

    let dense = vec![0.5f32; DIM as usize];

    // Full-replace the point with a payload that omits `t_valid`, to simulate a
    // pre-migration point (written before the field entered `payload_of`).
    // `Payload` is a newtype over `HashMap<String, Value>` with no `.remove`
    // method. Extract the inner map, drop the field, and let the block below
    // rebuild it via `Payload::from` — this simulates a pre-migration point
    // written before `t_valid` entered `payload_of`.
    let mut payload: std::collections::HashMap<String, qdrant_client::qdrant::Value> =
        QdrantStore::payload_of(&rec).into();
    payload.remove("t_valid");
    {
        use qdrant_client::qdrant::{NamedVectors, PointStruct, UpsertPointsBuilder, Vector};
        use qdrant_client::Payload;
        store
            .client()
            .upsert_points(
                UpsertPointsBuilder::new(
                    store.collection(),
                    vec![PointStruct::new(
                        rec.id.to_string(),
                        NamedVectors::default()
                            .add_vector("dense", Vector::new_dense(dense.clone())),
                        Payload::from(payload),
                    )],
                )
                    .wait(true),
            )
            .await
            .expect("pre-migration point upsert");
    }

    // A missing field is a migration gap, not content drift: every invariant
    // still matches the ledger, so the invariants reconcile clean.
    let missing = reconcile(&ledger, &store, NS, Some(&embedder), false)
        .await
        .unwrap();
    assert_eq!(
        missing.missing_t_valid,
        1,
        "exactly one pre-migration point must be flagged: {missing:?}"
    );
    assert!(
        missing.payload_drift.is_empty(),
        "a missing t_valid is a migration gap, not content drift: {missing:?}"
    );
    assert_eq!(
        missing.total(),
        0,
        "`total()` must exclude pre-migration gaps: {missing:?}"
    );
    assert!(
        missing.is_clean(),
        "a pre-migration store must reconcile clean on the invariants: {missing:?}"
    );

    // --repair adds the field back from the ledger, without re-embedding.
    let migrated = reconcile(&ledger, &store, NS, Some(&embedder), true)
        .await
        .unwrap();
    assert!(migrated.repaired, "the repair must run: {migrated:?}");

    // Re-reconcile (fresh detection) must be clean: the point's t_valid now
    // equals the ledger's and it is no longer a migration gap.
    let after = reconcile(&ledger, &store, NS, Some(&embedder), false)
        .await
        .unwrap();
    assert_eq!(
        after.missing_t_valid,
        0,
        "the repair must clear the pre-migration gap: {after:?}"
    );
    assert!(
        after.is_clean(),
        "the store must be clean after the t_valid migration: {after:?}"
    );

    store.drop_collection().await.expect("drop scratch collection");
}
