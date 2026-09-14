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

use async_trait::async_trait;
use myelin_core::config::{MyelinConfig, QdrantConfig};
use myelin_core::embed::Embedder;
use myelin_core::model::delta::Delta;
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::model::record::{
    ActorId, MemoryRecord, Provenance, RecordKind, Salience, Scope, SourceRef, Trust, Validity,
};
use myelin_core::pipeline::retrieve::Retriever;
use myelin_core::store::ids::record_id;
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::{IndexItem, QdrantStore};
use qdrant_client::qdrant::DeleteCollectionBuilder;
use uuid::Uuid;

const DIM: u64 = 8;

fn qdrant_config() -> QdrantConfig {
    MyelinConfig::load().expect("load MyelinConfig").qdrant
}

/// Deterministic, dependency-free vectors.
///
/// A real embedder would make this test measure the embedder. What is under
/// test is whether the caller's `k` survives the pipeline, and for that the
/// only property the vectors need is to be stable and distinct.
struct HashEmbedder;

#[async_trait]
impl Embedder for HashEmbedder {
    fn dim(&self) -> u64 {
        DIM
    }

    fn id(&self) -> &str {
        "test-hash-embedder"
    }

    async fn embed(&self, texts: &[String]) -> myelin_core::error::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; DIM as usize];
                for (i, b) in t.bytes().enumerate() {
                    v[i % DIM as usize] += f32::from(b) / 255.0;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                v.iter().map(|x| x / norm).collect()
            })
            .collect())
    }
}

struct ScratchGuard(String);

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        let name = std::mem::take(&mut self.0);
        let _ = std::thread::spawn(move || {
            let Ok(rt) = tokio::runtime::Runtime::new() else {
                return;
            };
            rt.block_on(async {
                if let Ok(c) = myelin_core::store::qdrant::client(&qdrant_config()) {
                    let _ = c.delete_collection(DeleteCollectionBuilder::new(&name)).await;
                }
            });
        })
        .join();
    }
}

/// The episode every semantic record in these tests derives from.
///
/// Not a formality: `Ledger::apply` rejects a semantic record whose ancestor
/// is absent (I4), which is how the first draft of this test failed. A
/// dangling `derived_from` is exactly the corruption I4 exists to prevent.
fn episode(scope: &Scope) -> MemoryRecord {
    let actor = ActorId::new("test");
    MemoryRecord {
        id: record_id(scope, "episode"),
        kind: RecordKind::Episodic,
        scope: scope.clone(),
        text: "Dana: I adopted a rescue dog.".to_string(),
        entities: Vec::new(),
        validity: Validity {
            t_valid: chrono::Utc::now(),
            t_invalid: None,
            t_ingested: chrono::Utc::now(),
            t_expired: None,
        },
        provenance: Provenance {
            source: SourceRef::doc("test"),
            contributed_by: actor.clone(),
            written_by: actor,
            derived_from: Vec::new(),
        },
        trust: Trust::asserted(),
        salience: Salience::default(),
        links: Vec::new(),
    }
}

fn semantic(scope: &Scope, text: &str, parent: Uuid) -> MemoryRecord {
    let actor = ActorId::new("test");
    MemoryRecord {
        id: record_id(scope, text),
        kind: RecordKind::Semantic,
        scope: scope.clone(),
        text: text.to_string(),
        entities: Vec::new(),
        validity: Validity {
            t_valid: chrono::Utc::now(),
            t_invalid: None,
            t_ingested: chrono::Utc::now(),
            t_expired: None,
        },
        provenance: Provenance {
            source: SourceRef::doc("test"),
            contributed_by: actor.clone(),
            written_by: actor,
            derived_from: vec![parent],
        },
        trust: Trust::asserted(),
        salience: Salience::default(),
        links: Vec::new(),
    }
}

#[tokio::test]
async fn recall_returns_exactly_the_k_the_query_asked_for() {
    let collection = format!("myelin_test_budget_{}", Uuid::new_v4().simple());
    let _guard = ScratchGuard(collection.clone());

    let mut cfg = qdrant_config();
    cfg.collection = collection.clone();
    let store = QdrantStore::new(&cfg).expect("qdrant store");
    store
        .ensure_collection(DIM, false)
        .await
        .expect("create scratch collection");

    let ledger = Ledger::open_memory().await.expect("ledger");
    let scope = Scope::new("t/budget", "myelin", "budget");
    let actor = ActorId::new("test");
    let embedder = HashEmbedder;

    // Twelve near-synonyms so that neither `k` under test is reachable by
    // accident: with only 3 records, "returns 3 for k=3" would also pass a
    // build that ignores `k` entirely.
    let parent = episode(&scope);
    ledger
        .apply(
            &Delta::Add {
                record: Box::new(parent.clone()),
            },
            &actor,
        )
        .await
        .expect("apply episode");

    let texts: Vec<String> = (0..12)
        .map(|i| format!("Dana adopted a rescue dog in city number {i}"))
        .collect();
    let vectors = embedder.embed(&texts).await.expect("embed");

    // One batched upsert, not one per record: a per-record round trip with
    // `wait(true)` times out against a Qdrant that is concurrently indexing.
    let mut records = Vec::with_capacity(texts.len());
    for text in &texts {
        let record = semantic(&scope, text, parent.id);
        ledger
            .apply(
                &Delta::Add {
                    record: Box::new(record.clone()),
                },
                &actor,
            )
            .await
            .expect("apply add");
        records.push(record);
    }
    let items: Vec<IndexItem<'_>> = records
        .iter()
        .zip(&vectors)
        .map(|(record, dense)| IndexItem {
            record,
            dense: dense.clone(),
            late: None,
        })
        .collect();
    store.upsert(&items).await.expect("upsert");

    let ask = |k: usize| Recall {
        scope: ScopeFilter::tenant("t/budget").with_namespace("budget"),
        text: "Which dog did Dana adopt?".to_string(),
        budget: Budget {
            k,
            tokens: 4096,
            max_steps: 1,
        },
        mode: Mode::Recall,
        kinds: None,
    };

    let retriever = Retriever::new(&embedder, &store, &ledger);
    for k in [1usize, 3, 9] {
        let (set, _) = retriever.recall(&ask(k)).await.expect("recall");
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
    let collection = format!("myelin_test_tokens_{}", Uuid::new_v4().simple());
    let _guard = ScratchGuard(collection.clone());

    let mut cfg = qdrant_config();
    cfg.collection = collection.clone();
    let store = QdrantStore::new(&cfg).expect("qdrant store");
    store
        .ensure_collection(DIM, false)
        .await
        .expect("create scratch collection");

    let ledger = Ledger::open_memory().await.expect("ledger");
    let scope = Scope::new("t/tokens", "myelin", "tokens");
    let actor = ActorId::new("test");
    let embedder = HashEmbedder;

    let parent = episode(&scope);
    ledger
        .apply(
            &Delta::Add {
                record: Box::new(parent.clone()),
            },
            &actor,
        )
        .await
        .expect("apply episode");

    let texts: Vec<String> = (0..6)
        .map(|i| format!("Record {i}: {}", "long filler sentence. ".repeat(20)))
        .collect();
    let vectors = embedder.embed(&texts).await.expect("embed");
    // One batched upsert, not one per record: a per-record round trip with
    // `wait(true)` times out against a Qdrant that is concurrently indexing.
    let mut records = Vec::with_capacity(texts.len());
    for text in &texts {
        let record = semantic(&scope, text, parent.id);
        ledger
            .apply(
                &Delta::Add {
                    record: Box::new(record.clone()),
                },
                &actor,
            )
            .await
            .expect("apply add");
        records.push(record);
    }
    let items: Vec<IndexItem<'_>> = records
        .iter()
        .zip(&vectors)
        .map(|(record, dense)| IndexItem {
            record,
            dense: dense.clone(),
            late: None,
        })
        .collect();
    store.upsert(&items).await.expect("upsert");

    let (set, _) = Retriever::new(&embedder, &store, &ledger)
        .recall(&Recall {
            scope: ScopeFilter::tenant("t/tokens").with_namespace("tokens"),
            text: "filler".to_string(),
            // Each record is ~100 tokens, so 150 admits one and the second
            // would bust the ceiling.
            budget: Budget {
                k: 6,
                tokens: 150,
                max_steps: 1,
            },
            mode: Mode::Recall,
            kinds: None,
        })
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
