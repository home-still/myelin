//! Caller metadata is filterable in the pre-ranking scope, and a scope can be
//! erased except for named records.
//!
//! These guard behaviour a consumer that stores documents relies on: a year
//! range or a topic must narrow the *candidate set* (a post-filter over a
//! ranked top-k would return fewer than k and could return none), and
//! replacing a document means deleting its old generation while keeping the
//! new one.
//!
//! ```text
//! MYELIN_QDRANT__URL=http://127.0.0.1:6434 \
//!     cargo test -p myelin-core --features integration --test meta_filter
//! ```
//!
//! Scratch collections only (`myelin_test_*`), removed on drop.

#![cfg(feature = "integration")]

mod common;

use std::collections::{BTreeSet, HashSet};

use common::{commit, episode, scratch_store, HashEmbedder};
use myelin_core::model::query::{Budget, MetaCondition, Mode, Recall, ScopeFilter};
use myelin_core::model::record::{ActorId, MemoryRecord, Scope};
use myelin_core::pipeline::forget::forget_scope;
use myelin_core::pipeline::retrieve::Retriever;
use myelin_core::store::ledger::Ledger;
use qdrant_client::qdrant::FieldType;
use uuid::Uuid;

const TENANT: &str = "t/meta";
const NS: &str = "doc-1";

fn with_meta(mut r: MemoryRecord, meta: serde_json::Value) -> MemoryRecord {
    r.meta = meta.as_object().cloned();
    r
}

/// Three records of one tenant: years 2019 / 2021 / 2023, categories a / b / a,
/// all sharing the word `paper` so every one is a candidate for any query.
fn corpus() -> Vec<MemoryRecord> {
    let scope = Scope::new(TENANT, "agent", NS);
    vec![
        with_meta(
            episode(&scope, "y2019", "paper about thyroid antibodies"),
            serde_json::json!({"year": 2019, "category": "a", "topics": ["thyroid"]}),
        ),
        with_meta(
            episode(&scope, "y2021", "paper about thyroid imaging"),
            serde_json::json!({"year": 2021, "category": "b", "topics": ["thyroid", "imaging"]}),
        ),
        with_meta(
            episode(&scope, "y2023", "paper about thyroid surgery"),
            serde_json::json!({"year": 2023, "category": "a", "topics": ["surgery"]}),
        ),
    ]
}

fn ask(filter: Vec<MetaCondition>) -> Recall {
    Recall {
        scope: ScopeFilter::tenant(TENANT).with_namespace(NS),
        text: "paper about thyroid".to_string(),
        budget: Budget {
            k: 10,
            tokens: 8192,
            max_steps: 1,
        },
        mode: Mode::Recall,
        kinds: None,
        as_of: None,
        meta_filter: filter,
    }
}

async fn recalled(
    retriever: &Retriever<'_>,
    filter: Vec<MetaCondition>,
) -> BTreeSet<Uuid> {
    let (set, _) = retriever.recall(&ask(filter)).await.expect("recall");
    set.items.iter().map(|i| i.record_id).collect()
}

#[tokio::test]
async fn meta_conditions_narrow_the_candidate_set_before_ranking() {
    let (store, _guard) = scratch_store("meta_filter").await;
    store.ensure_meta_index("year", FieldType::Integer).await.unwrap();
    store
        .ensure_meta_index("category", FieldType::Keyword)
        .await
        .unwrap();
    store.ensure_meta_index("topics", FieldType::Keyword).await.unwrap();
    // Idempotent: a second call with the same type is not an error.
    store.ensure_meta_index("year", FieldType::Integer).await.unwrap();

    let ledger = Ledger::open_memory().await.unwrap();
    let embedder = HashEmbedder;
    let records = corpus();
    commit(&ledger, &store, &embedder, &records).await;
    let id = |i: usize| records[i].id;
    let retriever = Retriever::new(&embedder, &store, &ledger);

    assert_eq!(
        recalled(&retriever, vec![]).await,
        BTreeSet::from([id(0), id(1), id(2)]),
        "precondition: with no filter all three are candidates"
    );

    let range = |gte: Option<i64>, lt: Option<i64>| MetaCondition::Range {
        key: "year".into(),
        gt: None,
        gte,
        lt,
        lte: None,
    };
    assert_eq!(
        recalled(&retriever, vec![range(Some(2021), None)]).await,
        BTreeSet::from([id(1), id(2)]),
        "year >= 2021"
    );
    assert_eq!(
        recalled(&retriever, vec![range(None, Some(2021))]).await,
        BTreeSet::from([id(0)]),
        "year < 2021"
    );

    let eq = |key: &str, value: &str| MetaCondition::Eq {
        key: key.into(),
        value: value.into(),
    };
    assert_eq!(
        recalled(&retriever, vec![eq("category", "a")]).await,
        BTreeSet::from([id(0), id(2)]),
        "category == a"
    );
    // An `eq` on an array field matches when the array contains the value.
    assert_eq!(
        recalled(&retriever, vec![eq("topics", "thyroid")]).await,
        BTreeSet::from([id(0), id(1)]),
        "topics contains thyroid"
    );
    // Conditions are ANDed.
    assert_eq!(
        recalled(
            &retriever,
            vec![eq("category", "a"), range(Some(2021), None)]
        )
        .await,
        BTreeSet::from([id(2)]),
        "category == a AND year >= 2021"
    );
    assert!(
        recalled(&retriever, vec![eq("category", "zzz")]).await.is_empty(),
        "a condition nothing satisfies returns nothing, not the unfiltered set"
    );
}

#[tokio::test]
async fn a_meta_filter_with_the_graph_channel_is_refused() {
    use myelin_core::pipeline::retrieve::RetrieveConfig;

    let (store, _guard) = scratch_store("meta_graph").await;
    let ledger = Ledger::open_memory().await.unwrap();
    let embedder = HashEmbedder;
    let retriever = Retriever::new(&embedder, &store, &ledger).with_config(RetrieveConfig {
        graph: true,
        ..RetrieveConfig::default()
    });
    let err = retriever
        .recall(&ask(vec![MetaCondition::Eq {
            key: "category".into(),
            value: "a".into(),
        }]))
        .await
        .expect_err("graph + meta filter");
    assert!(err.to_string().contains("graph channel"), "{err}");
}

#[tokio::test]
async fn forget_scope_with_keep_leaves_exactly_the_kept_record() {
    let (store, _guard) = scratch_store("forget_scope").await;
    let ledger = Ledger::open_memory().await.unwrap();
    let embedder = HashEmbedder;
    let mut records = corpus();
    // A record of another namespace and one of another tenant must survive.
    let other_ns = episode(&Scope::new(TENANT, "agent", "doc-2"), "other-ns", "paper elsewhere");
    let other_tenant = episode(&Scope::new("t/other", "agent", NS), "other-t", "paper elsewhere");
    records.push(other_ns.clone());
    records.push(other_tenant.clone());
    commit(&ledger, &store, &embedder, &records).await;

    let keep = records[1].id;
    let gone = forget_scope(
        &ledger,
        &store,
        TENANT,
        NS,
        &HashSet::from([keep]),
        &ActorId::new("test"),
        "replace document",
    )
    .await
    .unwrap();
    let mut want = vec![records[0].id, records[2].id];
    want.sort();
    assert_eq!(gone, want, "everything in the scope except `keep` is erased");

    // The ledger and the vector index agree: exactly the kept record is
    // visible in the scope, and neighbouring scopes are untouched.
    let retriever = Retriever::new(&embedder, &store, &ledger);
    assert_eq!(
        recalled(&retriever, vec![]).await,
        BTreeSet::from([keep]),
        "only the kept record is recalled from the scope"
    );
    let now = chrono::Utc::now();
    assert_eq!(
        ledger.namespace_inventory(TENANT, None, now).await.unwrap(),
        vec![("doc-1".to_string(), 1), ("doc-2".to_string(), 1)]
    );
    assert_eq!(
        ledger.namespace_inventory("t/other", None, now).await.unwrap(),
        vec![(NS.to_string(), 1)]
    );
    assert_eq!(store.count().await.unwrap(), 3, "keep + two other-scope points");

    // Erasing again with nothing to erase is a no-op, not an error.
    let again = forget_scope(
        &ledger,
        &store,
        TENANT,
        NS,
        &HashSet::from([keep]),
        &ActorId::new("test"),
        "replace document",
    )
    .await
    .unwrap();
    assert!(again.is_empty());
}
