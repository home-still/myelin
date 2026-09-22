//! Decomposition reaches records one query cannot (M24).
//!
//! The unit tests in `pipeline::decompose` cover the decomposer's contract —
//! what it returns for a given model response. They cannot cover the thing
//! the mechanism actually exists for, which is a property of the *retrieval*:
//! that a record only a sub-question describes ends up in the evidence set.
//!
//! That is the whole claim. M16 and M22 both measure this system as
//! retrieval-limited — a perfect reader over today's evidence reaches 41.5
//! against a 51.0 break-even, a perfect retrieval reaches 76.9 — so every
//! mechanism that reorders one pool is bounded by what that pool contains.
//! This test asserts the pool changed, not that a ranking did.
//!
//! ```text
//! cargo test -p myelin-core --features integration --test decompose_route
//! ```

#![cfg(feature = "integration")]

mod common;

use async_trait::async_trait;
use common::{commit, episode, scratch_store, semantic, HashEmbedder};
use myelin_core::error::Result;
use myelin_core::llm::{Completion, CompletionRequest, Llm, Usage};
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::model::record::Scope;
use myelin_core::pipeline::retrieve::{RetrieveConfig, Retriever};
use myelin_core::store::ledger::Ledger;

/// A decomposer that always returns the same two sub-queries.
///
/// The mechanism under test is the retrieval, not the model's judgement, so
/// the split is fixed: a live model would make this a test of prompt luck.
struct FixedSplit(Vec<String>);

#[async_trait]
impl Llm for FixedSplit {
    fn id(&self) -> &str {
        "fixed-split"
    }
    async fn raw_complete(&self, _req: &CompletionRequest) -> Result<Completion> {
        Ok(Completion {
            reasoning: None,
            text: serde_json::json!({ "queries": self.0 }).to_string(),
            tool_calls: vec![],
            finish_reason: None,
            usage: Usage::default(),
        })
    }
}

fn ask(k: usize, text: &str) -> Recall {
    Recall {
        scope: ScopeFilter::tenant("t/decompose").with_namespace("decompose"),
        text: text.to_string(),
        budget: Budget {
            k,
            tokens: 4096,
            max_steps: 1,
        },
        mode: Mode::Recall,
        kinds: None,
        as_of: None,
    }
}

/// Two facts, each described in vocabulary the question does not contain,
/// plus eight distractors written in the question's own words.
///
/// `HashEmbedder` is lexical, so a record sharing no tokens with the
/// question is genuinely unreachable from it. That is what makes each
/// sub-query a *different retrieval* rather than a re-ranking of the same
/// pool — and it is why the entity names are kept out of the question:
/// leave "Dana" in it and the first hop is reachable anyway, which would
/// make the test pass for the wrong reason.
async fn fixture() -> (
    myelin_core::store::qdrant::QdrantStore,
    common::ScratchGuard,
    Ledger,
) {
    let (store, guard) = scratch_store("decompose").await;
    let ledger = Ledger::open_memory().await.expect("ledger");
    let embedder = HashEmbedder;
    let scope = Scope::new("t/decompose", "myelin", "decompose");

    let parent = episode(&scope, "ep", "A conversation covering several updates.");
    let mut all = vec![parent.clone()];
    all.push(semantic(
        &scope,
        "hop-a",
        "Dana adopted a rescue dog named Biscuit",
        vec![parent.id],
    ));
    all.push(semantic(
        &scope,
        "hop-b",
        "Ravi relocated to Lisbon for a new position",
        vec![parent.id],
    ));
    // Sixty distractors in the question's vocabulary. Sixty and not eight
    // because `RetrieveConfig::prefetch_limit` is 50: under that, every
    // query retrieves the whole store and "the pool grew" is unobservable
    // for the arithmetic reason that it cannot. Over it, the single-query
    // path genuinely cannot see the hops.
    all.extend((0..60).map(|i| {
        semantic(
            &scope,
            &format!("noise-{i}"),
            &format!("Which household milestone landed first this year, item {i}"),
            vec![parent.id],
        )
    }));
    commit(&ledger, &store, &embedder, &all).await;
    (store, guard, ledger)
}

const QUESTION: &str = "Which household milestone landed first this year";

fn texts(set: &myelin_core::model::evidence::EvidenceSet) -> Vec<String> {
    set.items.iter().map(|i| i.value.clone()).collect()
}

/// With decomposition off, the second hop is out of reach: nothing in the
/// question's own vocabulary retrieves "relocated to Lisbon".
///
/// This is the control, and it has to hold for the next test to mean
/// anything — if the baseline already found both hops the mechanism would
/// be measuring nothing.
#[tokio::test]
async fn the_single_query_path_misses_the_second_hop() {
    let (store, _guard, ledger) = fixture().await;
    let embedder = HashEmbedder;
    let llm = FixedSplit(vec![]);

    let retriever = Retriever::new(&embedder, &store, &ledger).with_llm(&llm);
    let (set, trace) = retriever.recall(&ask(8, QUESTION)).await.expect("recall");

    assert_eq!(trace.subqueries, 0, "the switch is off");
    assert_eq!(trace.decompose_ms, 0, "and the model was never called");
    assert!(
        !texts(&set).iter().any(|t| t.contains("Lisbon")),
        "the control must not already have the second hop: {:?}",
        texts(&set)
    );
}

/// With decomposition on, a sub-query in the *record's* vocabulary pulls it
/// into the same fused pool, and it survives into the evidence set.
#[tokio::test]
async fn a_subquery_reaches_a_record_the_question_cannot() {
    let (store, _guard, ledger) = fixture().await;
    let embedder = HashEmbedder;
    let llm = FixedSplit(vec![
        "Dana rescue dog Biscuit".into(),
        "Ravi relocated Lisbon new position".into(),
    ]);

    let retriever = Retriever::new(&embedder, &store, &ledger)
        .with_llm(&llm)
        .with_config(RetrieveConfig {
            decompose: Some(4),
            ..Default::default()
        });
    let (set, trace) = retriever.recall(&ask(8, QUESTION)).await.expect("recall");

    assert_eq!(trace.subqueries, 2, "both sub-queries survived cleaning");
    let got = texts(&set);
    assert!(
        got.iter().any(|t| t.contains("Lisbon")),
        "the sub-query's record must reach the evidence set: {got:?}"
    );
    assert!(
        got.iter().any(|t| t.contains("Biscuit")),
        "and so must the first hop's: {got:?}"
    );
}

/// The switch alone is inert. Without [`Retriever::with_llm`] there is no
/// decomposer, and a run that quietly measured nothing is the failure class
/// M12, M14 and M20 each lost a run to.
#[tokio::test]
async fn the_switch_without_an_llm_is_inert_and_says_so() {
    let (store, _guard, ledger) = fixture().await;
    let embedder = HashEmbedder;

    let retriever = Retriever::new(&embedder, &store, &ledger).with_config(RetrieveConfig {
        decompose: Some(4),
        ..Default::default()
    });
    let (set, trace) = retriever.recall(&ask(8, QUESTION)).await.expect("recall");

    assert_eq!(trace.subqueries, 0);
    assert_eq!(trace.decompose_ms, 0);
    assert!(!set.items.is_empty(), "and it is still an ordinary recall");
}

/// The *pool* is a union and only ever grows. Nothing is de-duplicated
/// across sub-query results — M21 measured that co-evidence for one
/// question resembles itself 1.60× more than the rest of the set, so a
/// redundancy filter here would be aimed exactly at the records multi-hop
/// needs.
///
/// Asserted on `trace.fused` and not on the emitted set, and the
/// distinction is the point: `k` bounds what `compose` emits, so a wider
/// pool legitimately changes *which* records win the top-k. Claiming the
/// emitted set is monotone would be claiming decomposition cannot reorder
/// anything, which is the opposite of what it is for.
#[tokio::test]
async fn the_fused_pool_only_ever_grows() {
    let (store, _guard, ledger) = fixture().await;
    let embedder = HashEmbedder;

    let plain = Retriever::new(&embedder, &store, &ledger);
    let (_, before) = plain.recall(&ask(10, QUESTION)).await.expect("recall");

    let llm = FixedSplit(vec!["Ravi relocated Lisbon new position".into()]);
    let wide = Retriever::new(&embedder, &store, &ledger)
        .with_llm(&llm)
        .with_config(RetrieveConfig {
            decompose: Some(4),
            ..Default::default()
        });
    let (_, after) = wide.recall(&ask(10, QUESTION)).await.expect("recall");

    assert_eq!(after.subqueries, 1);
    assert!(
        after.fused > before.fused,
        "one sub-query in a vocabulary the question does not share must add \
         candidates: {} -> {}",
        before.fused,
        after.fused
    );
}

