//! `recall`'s graph channel, end to end against a live store.
//!
//! `tests/invariants.rs::bridged_record_outranks_an_unrelated_one` proves PPR
//! ranks a bridged record above an unrelated one. This proves the *route*:
//! that the PPR list joins `dense`/`lex` inside the one `rrf` call and that a
//! record BM25 cannot reach at all ends up in the composed evidence set.
//!
//! Both arms run `Channels::Lex`. BM25-only is deterministic and
//! interpretable, and it isolates the switch under test: the answer record
//! shares no term with the question, so Qdrant's sparse channel cannot return
//! it for a structural reason rather than a ranking accident. The
//! `HashEmbedder` never runs on this path — `recall` skips query embedding
//! entirely for `Channels::Lex` — so no arbitrary hash geometry can decide
//! the outcome.
//!
//! ```text
//! cargo test -p myelin-core --features integration --test graph_route
//! ```

#![cfg(feature = "integration")]

mod common;

use common::{commit, episode, scratch_store, HashEmbedder};
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::model::record::Scope;
use myelin_core::pipeline::phrases::incidence_rows;
use myelin_core::pipeline::retrieve::{Channels, RetrieveConfig, Retriever};
use myelin_core::store::graph::GraphIndex;
use myelin_core::store::ledger::Ledger;

const QUESTION: &str = "Which company does Caroline work for?";

#[tokio::test]
async fn the_graph_channel_surfaces_a_record_bm25_cannot_reach() {
    let (store, _guard) = scratch_store("graph_route").await;
    let ledger = Ledger::open_memory().await.expect("ledger");
    let embedder = HashEmbedder;
    let scope = Scope::new("t/graph", "myelin", "graph");

    // The query names Caroline; the answer names Acme Robotics; only the
    // bridge names both. Two hops: caroline -> bridge -> berlin -> answer.
    let records = vec![
        episode(&scope, "bridge", "Caroline moved to Berlin last spring."),
        episode(&scope, "answer", "The Berlin office is run by Acme Robotics."),
        episode(&scope, "unrelated", "Dana adopted a rescue dog called Pepper."),
    ];
    commit(&ledger, &store, &embedder, &records).await;

    // `commit` writes the ledger and Qdrant only — it never touches
    // `incidence` — so the graph is written here, through the same function
    // the ingest path and the backfill use.
    for r in &records {
        ledger
            .replace_incidence_batch(&incidence_rows(r))
            .await
            .expect("write incidence");
    }
    let answer = records[1].id;

    let query = Recall {
        scope: ScopeFilter::tenant("t/graph").with_namespace("graph"),
        text: QUESTION.to_string(),
        budget: Budget::default(),
        mode: Mode::Recall,
        kinds: None,
    };
    let lex_only = RetrieveConfig {
        channels: Channels::Lex,
        ..Default::default()
    };
    let with_graph = RetrieveConfig {
        graph: true,
        ..lex_only.clone()
    };
    let index = GraphIndex::new();

    let (off, off_trace) = Retriever::new(&embedder, &store, &ledger)
        .with_config(lex_only)
        .recall(&query)
        .await
        .expect("recall, graph off");
    let (on, on_trace) = Retriever::new(&embedder, &store, &ledger)
        .with_config(with_graph)
        .with_graph(&index)
        .recall(&query)
        .await
        .expect("recall, graph on");

    assert!(
        !off.items.iter().any(|i| i.record_id == answer),
        "BM25 alone returned the answer record, so this test cannot attribute \
         anything to the graph channel: {:?}",
        off.items.iter().map(|i| &i.value).collect::<Vec<_>>()
    );
    assert!(
        on.items.iter().any(|i| i.record_id == answer),
        "the graph channel did not surface the bridged record; trace {on_trace:?}"
    );
    assert_eq!(
        (off_trace.graph_seeds, off_trace.graph_hits),
        (0, 0),
        "the switch is off, so the channel must not have run"
    );
    assert_eq!(
        on_trace.graph_seeds, 1,
        "the query must seed exactly `caroline`; trace {on_trace:?}"
    );
}
