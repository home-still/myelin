//! M50c: the events block is appended after the base evidence and never
//! touches it (`myelin_core::pipeline::events_block`).
//!
//! Runs against a scratch Qdrant collection like the other integration tests;
//! the events index is its own collection, as in the dual-index design.

mod common;

use common::{commit, episode, scratch_store, semantic, HashEmbedder};
use myelin_core::model::evidence::EvidenceKind;
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::model::record::{Scope, SourceRef};
use myelin_core::pipeline::events_block::{
    events_retrieve_config, EventsBlock, EVENTS_HEADER, EVENTS_MECHANISM,
};
use myelin_core::pipeline::retrieve::Retriever;
use myelin_core::store::ledger::Ledger;

const TENANT: &str = "t/events";
const NS: &str = "events";

fn ask(text: &str) -> Recall {
    Recall {
        scope: ScopeFilter::tenant(TENANT).with_namespace(NS),
        text: text.to_string(),
        budget: Budget {
            k: 6,
            tokens: 2048,
            max_steps: 1,
        },
        mode: Mode::Recall,
        kinds: None,
        as_of: None,
    }
}

#[tokio::test]
async fn the_block_appends_after_the_base_and_leaves_it_untouched() {
    let embedder = HashEmbedder;
    let scope = Scope::new(TENANT, "myelin", NS);

    // The base: turns in their own collection.
    let (turn_store, _g1) = scratch_store("m50c-turns").await;
    let turn_ledger = Ledger::open_memory().await.expect("ledger");
    let turns: Vec<_> = (0..8)
        .map(|i| episode(&scope, &format!("turn-{i}"), &format!("Caroline: session {i}, we talked about the support group")))
        .collect();
    commit(&turn_ledger, &turn_store, &embedder, &turns).await;
    let base = Retriever::new(&embedder, &turn_store, &turn_ledger);

    // The events index: events only, in a second collection, with their
    // source turns in its ledger for lineage.
    let (event_store, _g2) = scratch_store("m50c-events").await;
    let event_ledger = Ledger::open_memory().await.expect("ledger");
    let parent = episode(&scope, "turn-0", "Caroline: session 0, we talked about the support group");
    // The parent turn goes to the ledger only, as in production, where the
    // events ledger is a copy of the base ledger and the new collection
    // indexes nothing but events.
    event_ledger
        .apply(
            &myelin_core::model::delta::Delta::Add { record: Box::new(parent.clone()) },
            &myelin_core::model::record::ActorId::new("test"),
        )
        .await
        .expect("parent turn");
    let events: Vec<_> = (0..5)
        .map(|i| {
            semantic(
                &scope,
                &format!("ev-{i}"),
                &format!("On 2023-05-0{} Caroline went to support group meeting {i}", i + 1),
                vec![parent.id],
            )
        })
        .collect();
    commit(&event_ledger, &event_store, &embedder, &events).await;
    let event_retriever = Retriever::new(&embedder, &event_store, &event_ledger)
        .with_config(events_retrieve_config());

    let query = ask("When did Caroline go to the support group?");
    let (mut with_block, _) = base.recall(&query).await.expect("base recall");
    let (without_block, _) = base.recall(&query).await.expect("base recall again");
    let base_len = with_block.items.len();
    let base_tokens = with_block.tokens;

    let block = EventsBlock::new(&event_retriever);
    let appended = block.append(&query, &mut with_block).await.expect("events");

    assert_eq!(appended, block.m, "the top m events, not the whole index");
    assert_eq!(
        with_block.items[..base_len],
        without_block.items[..],
        "the base evidence is byte-identical with the block on"
    );
    let header = &with_block.items[base_len];
    assert_eq!(header.value, EVENTS_HEADER);
    assert!(header.record_id.is_nil());
    assert_eq!(header.source, SourceRef::doc(EVENTS_MECHANISM));
    let appended_items = &with_block.items[base_len + 1..];
    assert_eq!(appended_items.len(), block.m);
    let event_ids: Vec<_> = events.iter().map(|e| e.id).collect();
    assert!(appended_items.iter().all(|i| event_ids.contains(&i.record_id)), "only events, from the events index");
    assert!(appended_items.iter().all(|i| i.kind == EvidenceKind::Text));
    assert!(
        !appended_items.iter().any(|i| i.value.starts_with("[timeline]")),
        "no second timeline view in the block"
    );
    assert!(with_block.tokens > base_tokens, "the block's tokens are counted");
}

#[tokio::test]
async fn nothing_found_appends_nothing() {
    let embedder = HashEmbedder;
    let (event_store, _g) = scratch_store("m50c-empty").await;
    let event_ledger = Ledger::open_memory().await.expect("ledger");
    let event_retriever = Retriever::new(&embedder, &event_store, &event_ledger)
        .with_config(events_retrieve_config());
    let mut set = myelin_core::model::evidence::EvidenceSet::default();
    let appended = EventsBlock::new(&event_retriever)
        .append(&ask("anything"), &mut set)
        .await
        .expect("events");
    assert_eq!(appended, 0);
    assert!(set.items.is_empty(), "no header without events");
}
