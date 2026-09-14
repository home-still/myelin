//! E4 tenant isolation and E6 unlearning (`EVALUATION.md` §7).
//!
//! These two are invariant tests, not benchmarks: §7 says they run on every
//! commit, and **a single cross-tenant leak fails G3 outright.**
//!
//! They live here rather than in `invariants.rs` because they must exercise
//! the *read path*, not the ledger. `invariants.rs` already proves I5's
//! cascade in SQLite (`i5_hard_delete_removes_exactly_the_descendant_closure`).
//! What that cannot show is whether the deleted material is still sitting in
//! Qdrant answering queries — which is exactly the question E6 asks, and the
//! failure mode a ledger-only test is blind to.
//!
//! ```text
//! cargo test -p myelin-core --features integration --test attack_isolation
//! ```

#![cfg(feature = "integration")]

mod common;

use common::{commit, episode, scratch_store, semantic, HashEmbedder};
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::model::record::{ActorId, Scope};
use myelin_core::pipeline::retrieve::Retriever;
use myelin_core::embed::Embedder;
use myelin_core::store::ledger::Ledger;

fn ask(tenant: &str, namespace: Option<&str>, text: &str) -> Recall {
    let mut scope = ScopeFilter::tenant(tenant);
    scope.namespace = namespace.map(str::to_string);
    Recall {
        scope,
        text: text.to_string(),
        budget: Budget {
            k: 20,
            tokens: 8192,
            max_steps: 1,
        },
        mode: Mode::Recall,
        kinds: None,
    }
}

/// E4 — cross-tenant read attempts on every read path. Zero leaks.
///
/// The query is deliberately chosen to be the *best possible* match for the
/// other tenant's records and a poor one for its own: if the scope filter
/// were dropped, ranking alone would put the foreign records on top, so a
/// leak cannot hide behind a weak query.
#[tokio::test]
async fn e4_a_tenant_cannot_read_another_tenants_memory() {
    let (store, _guard) = scratch_store("e4").await;
    let ledger = Ledger::open_memory().await.expect("ledger");
    let embedder = HashEmbedder;

    let alice = Scope::new("t/alice", "myelin", "shared");
    let mallory = Scope::new("t/mallory", "myelin", "shared");

    let alice_ep = episode(&alice, "a-ep", "Alice: I am planting tomatoes this spring.");
    let mallory_ep = episode(
        &mallory,
        "m-ep",
        "Mallory: the vault passphrase is CORRECT-HORSE-BATTERY.",
    );
    let secrets: Vec<_> = (0..6)
        .map(|i| {
            semantic(
                &mallory,
                &format!("m-sec-{i}"),
                &format!("The vault passphrase is CORRECT-HORSE-BATTERY, copy {i}."),
                vec![mallory_ep.id],
            )
        })
        .collect();
    let mut all = vec![alice_ep.clone(), mallory_ep.clone()];
    all.extend(secrets.iter().cloned());
    all.push(semantic(
        &alice,
        "a-sem",
        "Alice plants tomatoes in spring.",
        vec![alice_ep.id],
    ));
    commit(&ledger, &store, &embedder, &all).await;

    let retriever = Retriever::new(&embedder, &store, &ledger);

    // Path 1: `recall`, querying Mallory's secret from Alice's tenant.
    let (evidence, trace) = retriever
        .recall(&ask("t/alice", Some("shared"), "vault passphrase"))
        .await
        .expect("recall");
    for item in &evidence.items {
        assert!(
            !item.value.contains("CORRECT-HORSE-BATTERY"),
            "E4 LEAK via recall: {:?}",
            item.value
        );
    }
    // Alice owns exactly 2 records of the 9 in the collection. If the scope
    // predicate were applied AFTER ranking, both channels would have probed
    // the whole collection and returned up to 50 hits each. Bounding the
    // channel hit counts by her own record count is what distinguishes
    // scope-BEFORE-routing from a post-filter that merely hides the leak
    // (ShardMemo, +2.9/+3.1 F1, §2 finding 3).
    assert!(
        trace.dense_hits <= 2 && trace.lex_hits <= 2,
        "scope filtering is not happening in Qdrant: dense={} lex={} for a \
         tenant that owns 2 of 9 records",
        trace.dense_hits,
        trace.lex_hits
    );

    // Path 2: the ledger's own scope query, which `recall` re-checks against.
    let visible = ledger
        .visible(
            &ScopeFilter::tenant("t/alice").with_namespace("shared"),
            chrono::Utc::now(),
            100,
        )
        .await
        .expect("visible");
    assert!(
        visible.iter().all(|r| r.scope.tenant == "t/alice"),
        "E4 LEAK via ledger::visible"
    );

    // Path 3: the store, unfiltered by the ledger — proves the isolation is
    // enforced in the Qdrant payload filter and not only in Rust afterwards.
    let query_vec = embedder
        .embed(&["vault passphrase".to_string()])
        .await
        .expect("embed")
        .pop()
        .unwrap();
    let lists = store
        .hybrid_search(
            query_vec,
            "vault passphrase",
            "t/alice",
            Some("shared"),
            &[],
            50,
        )
        .await
        .expect("hybrid_search");
    for hit in lists.dense.iter().chain(lists.lex.iter()) {
        assert!(
            !hit.text.contains("CORRECT-HORSE-BATTERY"),
            "E4 LEAK via store::hybrid_search: {:?}",
            hit.text
        );
    }

    // Control: the same query from Mallory's own tenant DOES find them.
    // Without this the test would pass against a store that returns nothing
    // to anybody.
    let (mine, _) = retriever
        .recall(&ask("t/mallory", Some("shared"), "vault passphrase"))
        .await
        .expect("recall");
    assert!(
        mine.items
            .iter()
            .any(|i| i.value.contains("CORRECT-HORSE-BATTERY")),
        "the control failed: Mallory cannot read her own memory, so the \
         isolation assertions above prove nothing"
    );
}

/// E6 — unlearning. Delete a source and it is gone from every read path.
///
/// The ledger cascade is already proven by `invariants.rs`. What this adds
/// is the half a ledger test cannot see: the vectors. A record erased from
/// SQLite but left in Qdrant still answers queries, and `recall` would serve
/// it from the payload without ever consulting the ledger for its text.
#[tokio::test]
async fn e6_unlearning_removes_descendants_from_the_vector_store_too() {
    let (store, _guard) = scratch_store("e6").await;
    let ledger = Ledger::open_memory().await.expect("ledger");
    let embedder = HashEmbedder;
    let scope = Scope::new("t/unlearn", "myelin", "ns");

    let source = episode(
        &scope,
        "ep",
        "Dana: my national insurance number is QQ123456C.",
    );
    let derived: Vec<_> = (0..4)
        .map(|i| {
            semantic(
                &scope,
                &format!("sem-{i}"),
                &format!("Dana's national insurance number is QQ123456C (fact {i})."),
                vec![source.id],
            )
        })
        .collect();
    let bystander_ep = episode(&scope, "keep-ep", "Dana: I like hiking in the Peak District.");
    let bystander = semantic(
        &scope,
        "keep-sem",
        "Dana likes hiking in the Peak District.",
        vec![bystander_ep.id],
    );

    let mut all = vec![source.clone(), bystander_ep.clone(), bystander.clone()];
    all.extend(derived.iter().cloned());
    commit(&ledger, &store, &embedder, &all).await;

    let retriever = Retriever::new(&embedder, &store, &ledger);
    let (before, _) = retriever
        .recall(&ask("t/unlearn", Some("ns"), "national insurance number"))
        .await
        .expect("recall");
    assert!(
        before.items.iter().any(|i| i.value.contains("QQ123456C")),
        "the secret must be retrievable before deletion, or the test proves nothing"
    );

    let erased = ledger
        .hard_delete(source.id, &ActorId::new("test"), "E6 unlearn")
        .await
        .expect("hard_delete");
    assert_eq!(
        erased.len(),
        5,
        "expected the episode and its four descendants"
    );
    store.delete_points(&erased).await.expect("delete_points");

    let (after, _) = retriever
        .recall(&ask("t/unlearn", Some("ns"), "national insurance number"))
        .await
        .expect("recall");
    for item in &after.items {
        assert!(
            !item.value.contains("QQ123456C"),
            "E6 FAIL: unlearned material still reachable via recall: {:?}",
            item.value
        );
    }

    // And directly in the store, bypassing the ledger's admissibility check
    // entirely — the ledger can no longer hide a stale point, because the
    // record it would look up is gone.
    let query_vec = embedder
        .embed(&["national insurance number".to_string()])
        .await
        .expect("embed")
        .pop()
        .unwrap();
    let lists = store
        .hybrid_search(
            query_vec,
            "national insurance number",
            "t/unlearn",
            Some("ns"),
            &[],
            50,
        )
        .await
        .expect("hybrid_search");
    for hit in lists.dense.iter().chain(lists.lex.iter()) {
        assert!(
            !hit.text.contains("QQ123456C"),
            "E6 FAIL: unlearned vector still in Qdrant: {:?}",
            hit.text
        );
    }

    // The bystander survives. An over-eager cascade is as much a bug as a
    // leak: unlearning that deletes unrelated memories is not unlearning.
    let (kept, _) = retriever
        .recall(&ask("t/unlearn", Some("ns"), "hiking Peak District"))
        .await
        .expect("recall");
    assert!(
        kept.items.iter().any(|i| i.value.contains("Peak District")),
        "E6 FAIL: deletion took unrelated records with it"
    );
}
