//! `recall` honours the *query's* budget, not the retriever's defaults.
//!
//! **R4 is the reason this is a test and not a comment.** `PLAN.md` §7 R4:
//! "`recall` vs `investigate`, `k`, `rrf_k` and step budgets are all
//! *query-time* parameters against one identical store." A leaderboard
//! submission has to present several operating points from one built memory,
//! so a `k` that is fixed at construction time cannot produce one.
//!
//! It is also a regression test. The MCP `recall` tool returned **6** items
//! for `k: 3` because [`Retriever::recall`] composed with
//! `self.config.compose` and never looked at `query.budget`. Nothing failed;
//! it just quietly ignored the caller. `k` is a security parameter too —
//! MINJA ASR climbs 6% → 20% → 38% as k goes 3 → 5 → 10 (`PLAN.md` §2
//! finding 9) — so silently returning double is not a cosmetic bug.
//!
//! ```text
//! cargo test -p myelin-core --features integration --test recall_budget
//! ```

#![cfg(feature = "integration")]

mod common;

use common::{commit, episode, scratch_store, semantic, HashEmbedder};
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::model::record::Scope;
use myelin_core::pipeline::retrieve::{rerank_head_depth, Retriever};
use myelin_core::store::ledger::Ledger;

fn ask(k: usize, tokens: usize, text: &str) -> Recall {
    Recall {
        scope: ScopeFilter::tenant("t/budget").with_namespace("budget"),
        text: text.to_string(),
        budget: Budget {
            k,
            tokens,
            max_steps: 1,
        },
        mode: Mode::Recall,
        kinds: None,
    }
}

#[tokio::test]
async fn recall_returns_exactly_the_k_the_query_asked_for() {
    let (store, _guard) = scratch_store("budget").await;
    let ledger = Ledger::open_memory().await.expect("ledger");
    let embedder = HashEmbedder;
    let scope = Scope::new("t/budget", "myelin", "budget");

    // Twelve near-synonyms so neither `k` under test is reachable by
    // accident: with only three records, "returns 3 for k=3" would also pass
    // a build that ignores `k` entirely.
    let parent = episode(&scope, "ep", "Dana: I adopted a rescue dog.");
    let mut all = vec![parent.clone()];
    all.extend((0..12).map(|i| {
        semantic(
            &scope,
            &format!("sem-{i}"),
            &format!("Dana adopted a rescue dog in city number {i}"),
            vec![parent.id],
        )
    }));
    commit(&ledger, &store, &embedder, &all).await;

    let retriever = Retriever::new(&embedder, &store, &ledger);
    for k in [1usize, 3, 9] {
        let (set, _) = retriever
            .recall(&ask(k, 4096, "Which dog did Dana adopt?"))
            .await
            .expect("recall");
        assert_eq!(
            set.len(),
            k,
            "recall({k}) returned {} items; `k` is a query-time parameter (R4) \
             and a security parameter (MINJA ASR 6% -> 38% as k goes 3 -> 10)",
            set.len()
        );
    }
}

#[tokio::test]
async fn a_tight_token_budget_truncates_below_k() {
    let (store, _guard) = scratch_store("tokens").await;
    let ledger = Ledger::open_memory().await.expect("ledger");
    let embedder = HashEmbedder;
    let scope = Scope::new("t/budget", "myelin", "budget");

    let parent = episode(&scope, "ep", "Dana: I adopted a rescue dog.");
    let mut all = vec![parent.clone()];
    all.extend((0..6).map(|i| {
        semantic(
            &scope,
            &format!("long-{i}"),
            &format!("Record {i}: {}", "long filler sentence. ".repeat(20)),
            vec![parent.id],
        )
    }));
    commit(&ledger, &store, &embedder, &all).await;

    // Each record is ~100 tokens, so 150 admits one and the second would
    // bust the ceiling.
    let (set, _) = Retriever::new(&embedder, &store, &ledger)
        .recall(&ask(6, 150, "filler"))
        .await
        .expect("recall");

    assert!(
        set.len() < 6,
        "a 150-token budget returned all 6 records ({} tokens); the token \
         ceiling is not being applied",
        set.tokens
    );
    assert!(
        !set.is_empty(),
        "the top item must be admitted even when it alone busts the budget — \
         returning nothing is worse than a slight overrun"
    );
}

/// Widening the reranker's head must change *which* records are emitted,
/// never *how many*.
///
/// The distinction is the whole safety argument for `rerank_factor`.
/// Shuster et al. (2021, EMNLP Findings, "Retrieval Augmentation Reduces
/// Hallucination in Conversation") measured that feeding a reader more
/// documents raises hallucination: "increasing the number of documents for
/// these models yields higher levels of hallucination." So the fix for a
/// degenerate second stage has to buy recall on the *selection* side while
/// leaving the reader's context width exactly where it was. If a future
/// change lets `rerank_factor` leak into the emitted count, this fails.
#[tokio::test]
async fn a_deeper_rerank_head_does_not_widen_what_the_reader_sees() {
    use myelin_core::pipeline::retrieve::RetrieveConfig;

    let (store, _guard) = scratch_store("factor").await;
    let ledger = Ledger::open_memory().await.expect("ledger");
    let embedder = HashEmbedder;
    let scope = Scope::new("t/budget", "myelin", "budget");

    let parent = episode(&scope, "ep", "Dana: I adopted a rescue dog.");
    let mut all = vec![parent.clone()];
    all.extend((0..24).map(|i| {
        semantic(
            &scope,
            &format!("sem-{i}"),
            &format!("Dana adopted a rescue dog in city number {i}"),
            vec![parent.id],
        )
    }));
    commit(&ledger, &store, &embedder, &all).await;

    for factor in [1usize, 4] {
        let retriever = Retriever::new(&embedder, &store, &ledger).with_config(RetrieveConfig {
            rerank_factor: factor,
            ..RetrieveConfig::default()
        });
        let (set, trace) = retriever
            .recall(&ask(5, 4096, "Which dog did Dana adopt?"))
            .await
            .expect("recall");
        assert_eq!(
            set.len(),
            5,
            "factor {factor} emitted {} items for k=5; the head depth must \
             not reach the reader's context width",
            set.len()
        );
        assert_eq!(
            trace.rerank_depth,
            rerank_head_depth(25, factor, 5),
            "the trace must report the head actually used, so an arm can \
             verify the knob at the wire instead of inferring it"
        );
    }
}
