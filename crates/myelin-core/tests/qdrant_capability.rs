//! Capability tests against a live Qdrant 1.19.1.
//!
//! These pin the four findings that the storage design in `PLAN.md` §5.1 rests
//! on, measured first in `docs/probes/qdrant_capability.rs` and written up in
//! `docs/research/00-verified-environment.md` §3. They are assertions, not
//! probes: if the server's behaviour changes, the build fails loudly here
//! rather than silently degrading retrieval quality later.
//!
//! Gated behind the `integration` feature so `cargo test --workspace` stays
//! hermetic:
//!
//! ```text
//! cargo test -p myelin-core --features integration
//! ```
//!
//! Endpoint comes from [`MyelinConfig::load`], so `MYELIN_QDRANT__URL` points
//! the whole file at a different instance. Each test owns a uniquely-named
//! `myelin_test_*` scratch collection and deletes it at the end — the nine
//! production collections on `big` are never touched.

#![cfg(feature = "integration")]

use myelin_core::config::{MyelinConfig, QdrantConfig};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeleteCollectionBuilder, Distance, Fusion, Modifier,
    MultiVectorComparator, MultiVectorConfigBuilder, NamedVectors, PointStruct,
    PrefetchQueryBuilder, Query, QueryPointsBuilder, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, UpdateStatus, UpsertPointsBuilder, Vector, VectorInput,
    VectorParamsBuilder, VectorsConfigBuilder,
};
use qdrant_client::{Payload, Qdrant};
use uuid::Uuid;

/// Scores are compared at f32 resolution; the probe's observations are exact to
/// about seven significant digits.
const EPS: f32 = 1e-5;

fn qdrant_config() -> QdrantConfig {
    MyelinConfig::load()
        .expect("load MyelinConfig")
        .qdrant
}

/// `myelin_test_<fn>_<uuid>` — unique per run so tests are order- and
/// parallelism-independent, and prefixed so a leak from a failed assertion is
/// unmistakably ours.
fn scratch_name(test: &str) -> String {
    format!("myelin_test_{test}_{}", Uuid::new_v4().simple())
}

/// A collection carrying all three retrieval channels: named dense `dense`,
/// named multivector `late` (max_sim), named sparse `lex`.
async fn scratch(test: &str) -> (Qdrant, String) {
    let client = myelin_core::store::qdrant::client(&qdrant_config()).expect("build qdrant client");
    let name = scratch_name(test);

    let mut vectors = VectorsConfigBuilder::default();
    vectors.add_named_vector_params("dense", VectorParamsBuilder::new(4, Distance::Cosine));
    vectors.add_named_vector_params(
        "late",
        VectorParamsBuilder::new(4, Distance::Cosine)
            .multivector_config(MultiVectorConfigBuilder::new(MultiVectorComparator::MaxSim)),
    );
    let mut sparse = SparseVectorsConfigBuilder::default();
    sparse.add_named_vector_params("lex", SparseVectorParamsBuilder::default());

    client
        .create_collection(
            CreateCollectionBuilder::new(&name)
                .vectors_config(vectors)
                .sparse_vectors_config(sparse),
        )
        .await
        .expect("create scratch collection with dense + multivector + sparse");

    (client, name)
}

/// The two probe points: every point carries all three channels at once.
fn three_channel_points() -> Vec<PointStruct> {
    vec![
        PointStruct::new(
            1u64,
            NamedVectors::default()
                .add_vector("dense", Vector::new_dense(vec![1.0, 0.0, 0.0, 0.0]))
                .add_vector(
                    "late",
                    Vector::new_multi(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]),
                )
                .add_vector("lex", Vector::new_sparse(vec![7u32, 42], vec![0.9, 0.4])),
            Payload::new(),
        ),
        PointStruct::new(
            2u64,
            NamedVectors::default()
                .add_vector("dense", Vector::new_dense(vec![0.0, 1.0, 0.0, 0.0]))
                .add_vector("late", Vector::new_multi(vec![vec![0.0, 0.0, 1.0, 0.0]]))
                .add_vector("lex", Vector::new_sparse(vec![42u32, 99], vec![0.7, 0.5])),
            Payload::new(),
        ),
    ]
}

async fn upsert(client: &Qdrant, collection: &str, points: Vec<PointStruct>) {
    let response = client
        .upsert_points(UpsertPointsBuilder::new(collection, points).wait(true))
        .await
        .expect("upsert points");
    let status = response.result.expect("upsert returned a result").status;
    assert_eq!(
        status,
        UpdateStatus::Completed as i32,
        "upsert did not complete: status {status}"
    );
}

async fn drop_collection(client: &Qdrant, collection: &str) {
    let deleted = client
        .delete_collection(DeleteCollectionBuilder::new(collection))
        .await
        .expect("delete scratch collection");
    assert!(deleted.result, "scratch collection {collection} was not deleted");
}

/// §3.1 — one collection holds dense + multivector + sparse. If this breaks,
/// hybrid retrieval needs a second store and the whole storage design in
/// `PLAN.md` §5.1 changes.
#[tokio::test]
async fn three_channels_in_one_collection() {
    let (client, name) = scratch("three_channels").await;
    upsert(&client, &name, three_channel_points()).await;
    drop_collection(&client, &name).await;
}

/// §3.2 — Qdrant's server-side RRF uses **k = 1**, not Cormack's k = 60.
///
/// With a single prefetch list the fused score of the rank-r document is
/// `1/(1 + r)` for 1-based r. Cormack/Clarke/Buettcher's k = 60 would give the
/// near-flat series 0.0164, 0.0161, 0.0159, … This test is how we find out if
/// the constant ever changes; until then, `PLAN.md` §5 fuses client-side to get
/// k = 60 semantics.
#[tokio::test]
async fn server_side_rrf_uses_k_equals_one() {
    let (client, name) = scratch("rrf_k").await;

    // Dense vectors fan out from the query so the prefetch order is strict.
    let points: Vec<PointStruct> = (0..4u64)
        .map(|i| {
            let t = 0.1 * (i as f32);
            PointStruct::new(
                100 + i,
                NamedVectors::default()
                    .add_vector("dense", Vector::new_dense(vec![1.0 - t, t, 0.0, 0.0])),
                Payload::new(),
            )
        })
        .collect();
    upsert(&client, &name, points).await;

    let response = client
        .query(
            QueryPointsBuilder::new(&name)
                .add_prefetch(
                    PrefetchQueryBuilder::default()
                        .query(Query::new_nearest(vec![1.0f32, 0.0, 0.0, 0.0]))
                        .using("dense")
                        .limit(10u64),
                )
                .query(Query::new_fusion(Fusion::Rrf))
                .limit(10u64),
        )
        .await
        .expect("rrf fusion query");

    let scores: Vec<f32> = response.result.iter().map(|p| p.score).collect();
    assert_eq!(scores.len(), 4, "expected all four points back, got {scores:?}");

    for (i, score) in scores.iter().enumerate() {
        let rank = (i + 1) as f32;
        let expected = 1.0 / (1.0 + rank); // k = 1
        assert!(
            (score - expected).abs() < EPS,
            "rank {rank}: RRF score {score} != 1/(1 + rank) = {expected}; \
             Qdrant's RRF constant is no longer k = 1 (all scores: {scores:?})"
        );
    }

    drop_collection(&client, &name).await;
}

/// §3.3 — server-side late-interaction (max_sim) rerank of a dense prefetch
/// works over gRPC. The finding is that the call succeeds at all: the identical
/// query over REST fails with `422 internal.query.indices: must be unique`,
/// which is why `myelin_core::store::qdrant` is gRPC-only.
#[tokio::test]
async fn server_side_maxsim_rerank_works() {
    let (client, name) = scratch("maxsim").await;
    upsert(&client, &name, three_channel_points()).await;

    let response = client
        .query(
            QueryPointsBuilder::new(&name)
                .add_prefetch(
                    PrefetchQueryBuilder::default()
                        .query(Query::new_nearest(vec![1.0f32, 0.0, 0.0, 0.0]))
                        .using("dense")
                        .limit(10u64),
                )
                .query(Query::new_nearest(VectorInput::new_multi(vec![
                    vec![1.0f32, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                ])))
                .using("late")
                .limit(3u64),
        )
        .await
        .expect("multivector rerank over gRPC");

    let scores: Vec<f32> = response.result.iter().map(|p| p.score).collect();
    assert!(!scores.is_empty(), "max_sim rerank returned nothing");
    for score in &scores {
        assert!(
            score.is_finite() && *score >= 0.0,
            "max_sim score {score} is not a finite non-negative similarity (all: {scores:?})"
        );
    }

    drop_collection(&client, &name).await;
}

/// §3.4 — Qdrant applies the BM25 IDF modifier at query time, in-process, with
/// no inference service, and scores exactly `Σ_i q_i · d_i · IDF(i)`.
///
/// That identity is what buys us BM25 for free: no `tantivy`, no client-side
/// lexical encoder, no separate index. MemPro's ablation makes BM25 the
/// highest-leverage channel (LoCoMo 84.93 → 72.25 without it), so this is the
/// single most load-bearing capability in the store.
#[tokio::test]
async fn bm25_idf_scoring_identity() {
    let client = myelin_core::store::qdrant::client(&qdrant_config()).expect("build qdrant client");
    let name = scratch_name("bm25_idf");

    // Sparse-only collection; the IDF modifier is the point of the test.
    let mut sparse = SparseVectorsConfigBuilder::default();
    sparse.add_named_vector_params(
        "lex",
        SparseVectorParamsBuilder::default().modifier(Modifier::Idf),
    );
    client
        .create_collection(CreateCollectionBuilder::new(&name).sparse_vectors_config(sparse))
        .await
        .expect("create sparse-only scratch collection with modifier = idf");

    // Term 11 appears in two of three documents, term 22 in one.
    let points = vec![
        PointStruct::new(
            1u64,
            NamedVectors::default()
                .add_vector("lex", Vector::new_sparse(vec![11u32, 22], vec![1.2, 0.8])),
            Payload::new(),
        ),
        PointStruct::new(
            2u64,
            NamedVectors::default().add_vector("lex", Vector::new_sparse(vec![11u32], vec![1.5])),
            Payload::new(),
        ),
        PointStruct::new(
            3u64,
            NamedVectors::default().add_vector("lex", Vector::new_sparse(vec![33u32], vec![2.0])),
            Payload::new(),
        ),
    ];
    upsert(&client, &name, points).await;

    let response = client
        .query(
            QueryPointsBuilder::new(&name)
                .query(Query::new_nearest(VectorInput::new_sparse(
                    vec![11u32, 22],
                    vec![1.0, 1.0],
                )))
                .using("lex")
                .limit(10u64),
        )
        .await
        .expect("sparse idf query");

    // IDF(t) = ln((N − n(t) + 0.5) / (n(t) + 0.5) + 1), statistics per shard.
    let idf = |n_docs: f32, n_containing: f32| -> f32 {
        ((n_docs - n_containing + 0.5) / (n_containing + 0.5) + 1.0).ln()
    };
    let n = 3.0;
    // score = Σ_i q_i · d_i · IDF(i), with q_11 = q_22 = 1.0.
    let expected_1 = 1.2 * idf(n, 2.0) + 0.8 * idf(n, 1.0);
    let expected_2 = 1.5 * idf(n, 2.0);

    let scored: Vec<(u64, f32)> = response
        .result
        .iter()
        .filter_map(|p| match p.id.as_ref()?.point_id_options.as_ref()? {
            qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => Some((*n, p.score)),
            qdrant_client::qdrant::point_id::PointIdOptions::Uuid(_) => None,
        })
        .collect();

    // Point 3 shares no term with the query, so it must not match at all.
    assert_eq!(
        scored.len(),
        2,
        "expected exactly the two term-sharing points, got {scored:?}"
    );

    let score_of = |id: u64| -> f32 {
        scored
            .iter()
            .find(|(p, _)| *p == id)
            .unwrap_or_else(|| panic!("point {id} missing from {scored:?}"))
            .1
    };

    for (id, expected) in [(1u64, expected_1), (2u64, expected_2)] {
        let got = score_of(id);
        assert!(
            (got - expected).abs() < EPS,
            "point {id}: Qdrant scored {got}, but Σ q·d·IDF = {expected}; \
             BM25 IDF scoring is no longer `sum_i q_i * d_i * IDF(i)`"
        );
    }

    // The probe's observed values, restated so a drift in *our* arithmetic is
    // also caught: 1.3486677 and 0.7050055.
    assert!((expected_1 - 1.348_667_7).abs() < EPS, "IDF formula drifted: {expected_1}");
    assert!((expected_2 - 0.705_005_5).abs() < EPS, "IDF formula drifted: {expected_2}");

    drop_collection(&client, &name).await;
}
