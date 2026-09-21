//! Shared scaffolding for the integration tests.
//!
//! Every test here owns a uniquely-named `myelin_test_*` scratch collection
//! and deletes it on drop, including on panic. The nine production
//! collections on `big` are never touched.

#![allow(dead_code)]

use async_trait::async_trait;
use myelin_core::config::{MyelinConfig, QdrantConfig};
use myelin_core::embed::Embedder;
use myelin_core::model::delta::Delta;
use myelin_core::model::record::{
    ActorId, MemoryRecord, Provenance, RecordKind, Salience, Scope, SourceRef, Trust, Validity,
};
use myelin_core::store::ids::record_id;
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::{IndexItem, QdrantStore};
use qdrant_client::qdrant::DeleteCollectionBuilder;
use uuid::Uuid;

pub const DIM: u64 = 8;

pub fn qdrant_config() -> QdrantConfig {
    MyelinConfig::load().expect("load MyelinConfig").qdrant
}

/// Deterministic, dependency-free vectors.
///
/// A real embedder would make these tests measure the embedder. What is
/// under test is scoping, budgets and deletion; the only property the
/// vectors need is to be stable and **distinct**.
///
/// Distinctness is why this is an FNV-1a hash per dimension rather than the
/// obvious sum of bytes modulo `DIM`. That version was not distinct at all:
/// any two English sentences of similar length produced vectors at
/// cosine > 0.999, because summing bytes throws away order. It went unnoticed
/// while `compose`'s cosine near-duplicate suppression was dead (every
/// `Ranked::vector` was `None`); the moment `recall` started returning real
/// vectors, a 12-record fixture composed down to one item. Hashing the whole
/// text once per dimension makes two texts that differ anywhere
/// near-orthogonal, which is what a fake embedder owes its tests.
pub struct HashEmbedder;

#[async_trait]
impl Embedder for HashEmbedder {
    fn dim(&self) -> u64 {
        DIM
    }

    fn id(&self) -> &str {
        "test-hash-embedder"
    }

    async fn embed(&self, texts: &[String]) -> myelin_core::error::Result<Vec<Vec<f32>>> {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

        /// MurmurHash3's finalizer. FNV-1a alone is not enough here: its
        /// last step only stirs the low bits, so twelve texts differing in
        /// their final character still agree in the top 16 — and the top 16
        /// are what becomes a coordinate. Without the avalanche the fixture
        /// vectors come back at cosine 1.0000 and `compose` dedups a
        /// thirteen-record corpus down to three.
        fn avalanche(mut h: u64) -> u64 {
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            h ^ (h >> 33)
        }

        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; DIM as usize];
                for (d, slot) in v.iter_mut().enumerate() {
                    let mut h = FNV_OFFSET ^ (d as u64 + 1);
                    h = h.wrapping_mul(FNV_PRIME);
                    for b in t.bytes() {
                        h ^= u64::from(b);
                        h = h.wrapping_mul(FNV_PRIME);
                    }
                    // Centre on zero so the vectors span the sphere instead
                    // of crowding the positive orthant, where everything is
                    // similar to everything.
                    *slot = (avalanche(h) >> 48) as f32 / f32::from(u16::MAX) - 0.5;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                v.iter().map(|x| x / norm).collect()
            })
            .collect())
    }
}

/// Deletes its scratch collection on drop, including when the test panics.
///
/// Without it every failing assertion leaks a collection onto a Qdrant that
/// also holds production data, and the leak check becomes permanently dirty.
/// `Drop` cannot await, so the delete runs on a throwaway runtime.
pub struct ScratchGuard(pub String);

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

/// A scratch collection plus its guard. Name is `myelin_test_<label>_<uuid>`.
pub async fn scratch_store(label: &str) -> (QdrantStore, ScratchGuard) {
    let name = format!("myelin_test_{label}_{}", Uuid::new_v4().simple());
    let guard = ScratchGuard(name.clone());
    let mut cfg = qdrant_config();
    cfg.collection = name;
    let store = QdrantStore::new(&cfg).expect("qdrant store");
    store
        .ensure_collection(DIM, false)
        .await
        .expect("create scratch collection");
    (store, guard)
}

pub fn episode(scope: &Scope, key: &str, text: &str) -> MemoryRecord {
    base(scope, key, RecordKind::Episodic, text, Vec::new())
}

pub fn semantic(scope: &Scope, key: &str, text: &str, parents: Vec<Uuid>) -> MemoryRecord {
    base(scope, key, RecordKind::Semantic, text, parents)
}

fn base(
    scope: &Scope,
    key: &str,
    kind: RecordKind,
    text: &str,
    derived_from: Vec<Uuid>,
) -> MemoryRecord {
    let actor = ActorId::new("test");
    let now = chrono::Utc::now();
    MemoryRecord {
        id: record_id(scope, key),
        kind,
        scope: scope.clone(),
        text: text.to_string(),
        entities: Vec::new(),
        validity: Validity {
            t_valid: now,
            t_invalid: None,
            t_ingested: now,
            t_expired: None,
        },
        provenance: Provenance {
            source: SourceRef::doc("test"),
            contributed_by: actor.clone(),
            written_by: actor,
            derived_from,
        },
        trust: Trust::asserted(),
        salience: Salience::default(),
        links: Vec::new(),
    }
}

/// Write records to both stores, batched.
///
/// One upsert, not one per record: a per-record round trip with `wait(true)`
/// times out against a Qdrant that is concurrently indexing.
pub async fn commit(
    ledger: &Ledger,
    store: &QdrantStore,
    embedder: &HashEmbedder,
    records: &[MemoryRecord],
) {
    let actor = ActorId::new("test");
    for r in records {
        ledger
            .apply(
                &Delta::Add {
                    record: Box::new(r.clone()),
                },
                &actor,
            )
            .await
            .unwrap_or_else(|e| panic!("apply {}: {e}", r.id));
    }
    let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
    let vectors = embedder.embed(&texts).await.expect("embed");
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
}
