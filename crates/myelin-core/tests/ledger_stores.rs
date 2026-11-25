//! The store registry in the ledger: what `define_store` accepts, what it
//! refuses, and that a file ledger created before the table existed gains it.
//!
//! Hermetic: SQLite only.

use myelin_core::model::store::StoreDef;
use myelin_core::store::ledger::Ledger;
use myelin_core::MyelinError;

fn def(name: &str, tenant: &str) -> StoreDef {
    StoreDef {
        name: name.into(),
        tenant: tenant.into(),
        description: format!("what {name} holds"),
        default_read: false,
        agent_writes: false,
        dated: false,
    }
}

fn refusal(result: myelin_core::Result<()>) -> String {
    match result {
        Err(MyelinError::StoreDefinition(msg)) => msg,
        other => panic!("expected a refused store definition, got {other:?}"),
    }
}

#[tokio::test]
async fn a_defined_store_reads_back_and_stores_are_ordered_by_name() {
    let ledger = Ledger::open_memory().await.expect("ledger");
    assert!(ledger.stores().await.expect("stores").is_empty());
    assert_eq!(ledger.store("academic").await.expect("store"), None);

    let memory = StoreDef {
        agent_writes: true,
        dated: true,
        ..def("memory", "user_memory")
    };
    let academic = StoreDef {
        default_read: true,
        ..def("academic", "academic_papers")
    };
    ledger.define_store(&memory).await.expect("define memory");
    ledger
        .define_store(&academic)
        .await
        .expect("define academic");

    assert_eq!(
        ledger.stores().await.expect("stores"),
        vec![academic.clone(), memory.clone()]
    );
    assert_eq!(ledger.store("memory").await.expect("store"), Some(memory));
    assert_eq!(
        ledger.store("academic").await.expect("store"),
        Some(academic)
    );
}

/// Restating a store changes what may change — the description and the three
/// flags — and nothing else, however many times it runs.
#[tokio::test]
async fn restating_a_store_updates_its_description_and_flags() {
    let ledger = Ledger::open_memory().await.expect("ledger");
    ledger
        .define_store(&def("web", "web_pages"))
        .await
        .expect("define");

    let restated = StoreDef {
        description: "pages captured from the web".into(),
        default_read: true,
        agent_writes: true,
        dated: true,
        ..def("web", "web_pages")
    };
    for _ in 0..2 {
        ledger.define_store(&restated).await.expect("restate");
    }
    assert_eq!(ledger.stores().await.expect("stores"), vec![restated]);
}

/// A name never moves to another tenant and a tenant never gains a second
/// name; both refusals leave the registry as it was.
#[tokio::test]
async fn a_name_or_a_tenant_bound_elsewhere_is_refused() {
    let ledger = Ledger::open_memory().await.expect("ledger");
    let original = def("personal", "personal_docs");
    ledger.define_store(&original).await.expect("define");

    let moved = refusal(ledger.define_store(&def("personal", "other_tenant")).await);
    assert!(
        moved.contains("personal_docs") && moved.contains("other_tenant"),
        "{moved}"
    );

    let aliased = refusal(ledger.define_store(&def("private", "personal_docs")).await);
    assert!(
        aliased.contains("personal") && aliased.contains("private"),
        "{aliased}"
    );

    assert_eq!(ledger.stores().await.expect("stores"), vec![original]);
}

#[tokio::test]
async fn a_malformed_definition_is_refused_before_it_is_stored() {
    let ledger = Ledger::open_memory().await.expect("ledger");
    for name in ["", "Academic", "9lives", "a-b", &"a".repeat(33)] {
        let msg = refusal(ledger.define_store(&def(name, "t")).await);
        assert!(msg.contains("must match"), "{name:?}: {msg}");
    }
    let msg = refusal(ledger.define_store(&def("ok", "  ")).await);
    assert!(msg.contains("needs a tenant"), "{msg}");
    let msg = refusal(
        ledger
            .define_store(&StoreDef {
                description: String::new(),
                ..def("ok", "t")
            })
            .await,
    );
    assert!(msg.contains("needs a description"), "{msg}");
    assert!(ledger.stores().await.expect("stores").is_empty());
}

/// The deployed ledgers were created before the `store` table. Opening one
/// must add the table and leave its records alone.
#[tokio::test]
async fn a_ledger_from_before_stores_gains_the_table_on_open() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("old.ledger");
    {
        let ledger = Ledger::open(&path).await.expect("create");
        sqlx_drop_store(&ledger).await;
    }
    let ledger = Ledger::open(&path).await.expect("reopen");
    ledger
        .define_store(&def("academic", "academic_papers"))
        .await
        .expect("define on a migrated ledger");
    assert_eq!(ledger.stores().await.expect("stores").len(), 1);
}

/// Put the file back to the pre-`store` schema.
async fn sqlx_drop_store(ledger: &Ledger) {
    sqlx::query("DROP TABLE store")
        .execute(ledger.pool())
        .await
        .expect("drop store table");
}
