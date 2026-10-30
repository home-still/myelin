//! Caller metadata (`MemoryRecord::meta`) in the ledger, and the two queries
//! that exist for consumers who store documents rather than conversations:
//! the per-namespace inventory and the ids of a scope.
//!
//! Hermetic: SQLite only.

use chrono::{Duration, Utc};
use myelin_core::model::delta::Delta;
use myelin_core::model::record::{
    ActorId, MemoryRecord, Provenance, RecordKind, Salience, Scope, SourceRef, Trust, Validity,
};
use myelin_core::store::ids::record_id;
use myelin_core::store::ledger::Ledger;

fn record(tenant: &str, namespace: &str, key: &str, meta: Option<serde_json::Value>) -> MemoryRecord {
    let sc = Scope::new(tenant, "agent-1", namespace);
    let now = Utc::now();
    MemoryRecord {
        id: record_id(&sc, key),
        kind: RecordKind::Episodic,
        scope: sc,
        text: format!("document: {key}"),
        entities: Vec::new(),
        validity: Validity {
            t_valid: now - Duration::hours(1),
            t_invalid: None,
            t_ingested: now - Duration::hours(1),
            t_expired: None,
        },
        provenance: Provenance {
            source: SourceRef::doc(key),
            contributed_by: ActorId::new("user-1"),
            written_by: ActorId::new("agent-1"),
            derived_from: Vec::new(),
        },
        trust: Trust::asserted(),
        salience: Salience::default(),
        links: Vec::new(),
        meta: meta.map(|v| v.as_object().cloned().expect("meta is an object")),
    }
}

async fn add(ledger: &Ledger, r: MemoryRecord) -> uuid::Uuid {
    let id = r.id;
    ledger
        .apply(&Delta::Add { record: Box::new(r) }, &ActorId::new("test"))
        .await
        .unwrap();
    id
}

#[tokio::test]
async fn meta_round_trips_and_its_absence_stays_none() {
    let ledger = Ledger::open_memory().await.unwrap();
    let meta = serde_json::json!({"year": 2021, "topics": ["a", "b"], "category": "x", "n": null});
    let with = add(&ledger, record("t", "d1", "with", Some(meta.clone()))).await;
    let without = add(&ledger, record("t", "d1", "without", None)).await;

    let got = ledger.get(with).await.unwrap().unwrap();
    assert_eq!(got.meta, meta.as_object().cloned());
    assert_eq!(ledger.get(without).await.unwrap().unwrap().meta, None);
}

/// A ledger created before `meta` existed has a `record` table without the
/// column, and `CREATE TABLE IF NOT EXISTS` leaves it that way. Opening it
/// must add the column, keep every row and read the old rows as `meta: None`.
#[tokio::test]
async fn an_old_schema_ledger_gains_the_meta_column_on_open() {
    use sqlx::sqlite::SqliteConnectOptions;
    use sqlx::{Connection, SqliteConnection};

    // The shipped schema minus the `meta` column is the pre-`meta` schema.
    let old_schema = include_str!("../src/store/schema.sql")
        .replace(",\n    meta                TEXT\n", "\n");
    assert!(!old_schema.contains("meta                TEXT"), "precondition");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.ledger");
    let old_id = uuid::Uuid::new_v4();
    {
        let mut conn = SqliteConnection::connect_with(
            &SqliteConnectOptions::new().filename(&path).create_if_missing(true),
        )
        .await
        .unwrap();
        sqlx::raw_sql(sqlx::AssertSqlSafe(old_schema)).execute(&mut conn).await.unwrap();
        sqlx::query(
            "INSERT INTO record (
                id, kind, tenant, agent, session, namespace, text, entities,
                t_valid, t_invalid, t_ingested, t_expired,
                trust_tier, trust_score, trust_checks,
                prov_source, prov_contributed_by, prov_written_by, prov_derived_from,
                salience
             ) VALUES (?, 'episodic', 't', 'a', NULL, 'd1', 'old text', '[]',
                '2020-01-01T00:00:00.000000000Z', NULL, '2020-01-01T00:00:00.000000000Z', NULL,
                'asserted', 0.5, '[]', '{\"doc\":\"old\"}', 'u', 'a', '[]',
                '{\"importance\":0.5,\"access_count\":0,\"strength\":1.0}')",
        )
        .bind(old_id.to_string())
        .execute(&mut conn)
        .await
        .unwrap();
        let cols: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info('record') WHERE name = 'meta'",
        )
        .fetch_one(&mut conn)
        .await
        .unwrap();
        assert_eq!(cols, 0, "precondition: the old table has no meta column");
        conn.close().await.unwrap();
    }

    let ledger = Ledger::open(&path).await.unwrap();
    let old = ledger.get(old_id).await.unwrap().expect("old row survives");
    assert_eq!(old.text, "old text");
    assert_eq!(old.meta, None);

    // And the column is usable: a new record writes and reads its meta.
    let new_id = add(
        &ledger,
        record("t", "d1", "new", Some(serde_json::json!({"year": 2024}))),
    )
    .await;
    assert_eq!(
        ledger.get(new_id).await.unwrap().unwrap().meta,
        serde_json::json!({"year": 2024}).as_object().cloned()
    );

    // Opening again, with the column already present, changes nothing.
    ledger.pool().close().await;
    let again = Ledger::open(&path).await.unwrap();
    assert_eq!(again.count().await.unwrap(), 2);
}

#[tokio::test]
async fn a_meta_that_is_not_an_object_is_an_error_not_a_default() {
    let ledger = Ledger::open_memory().await.unwrap();
    let id = add(&ledger, record("t", "d1", "bad", None)).await;
    sqlx::query("UPDATE record SET meta = '[1,2]' WHERE id = ?")
        .bind(id.to_string())
        .execute(ledger.pool())
        .await
        .unwrap();
    let err = ledger.get(id).await.expect_err("array meta must be refused");
    assert!(err.to_string().contains("meta is not a JSON object"), "{err}");
}

#[tokio::test]
async fn namespace_inventory_counts_only_live_records() {
    let ledger = Ledger::open_memory().await.unwrap();
    add(&ledger, record("t", "d1", "a", None)).await;
    add(&ledger, record("t", "d1", "b", None)).await;
    let forgotten = add(&ledger, record("t", "d1", "c", None)).await;
    let only_gone = add(&ledger, record("t", "d2", "d", None)).await;
    add(&ledger, record("t", "d3", "e", None)).await;
    add(&ledger, record("other", "d1", "f", None)).await;
    for id in [forgotten, only_gone] {
        ledger
            .apply(
                &Delta::Delete {
                    target: id,
                    reason: "test".into(),
                },
                &ActorId::new("test"),
            )
            .await
            .unwrap();
    }

    let now = Utc::now();
    let all = ledger.namespace_inventory("t", None, now).await.unwrap();
    assert_eq!(
        all,
        vec![("d1".to_string(), 2), ("d3".to_string(), 1)],
        "soft-forgotten records are not counted, an emptied namespace is not listed, \
         another tenant's records are not seen"
    );
    let one = ledger.namespace_inventory("t", Some("d1"), now).await.unwrap();
    assert_eq!(one, vec![("d1".to_string(), 2)]);
    assert!(ledger
        .namespace_inventory("t", Some("d2"), now)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn ids_in_scope_is_tenant_and_namespace_scoped_and_includes_forgotten() {
    let ledger = Ledger::open_memory().await.unwrap();
    let a = add(&ledger, record("t", "d1", "a", None)).await;
    let b = add(&ledger, record("t", "d1", "b", None)).await;
    add(&ledger, record("t", "d2", "c", None)).await;
    add(&ledger, record("other", "d1", "d", None)).await;
    ledger
        .apply(
            &Delta::Delete {
                target: b,
                reason: "test".into(),
            },
            &ActorId::new("test"),
        )
        .await
        .unwrap();

    let mut want = vec![a, b];
    want.sort();
    assert_eq!(ledger.ids_in_scope("t", "d1").await.unwrap(), want);
    assert!(ledger.ids_in_scope("t", "missing").await.unwrap().is_empty());
}
