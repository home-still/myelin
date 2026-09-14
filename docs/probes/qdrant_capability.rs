// Reproducible capability probe for Qdrant 1.19.1 — the source of the measurements in
// docs/research/00-verified-environment.md §3.1-3.3. Ran green against the live instance at
// 192.168.1.110:6334 on 2026-09-14 with qdrant-client 1.19.0:
//
//   create: ok
//   upsert(named dense+multivector+sparse): ok
//   rrf: [(Num(1), 0.8333334), (Num(2), 0.8333334)]        // = 1/2 + 1/3  => RRF k = 1
//   maxsim-rerank: [1.0, 1.0]                               // server-side late interaction works
//   rrf-single-list scores (rank1..n): [0.5, 0.33333334, 0.25, 0.2, 0.16666667, 0.14285715]
//   cleanup: ok
//
// Not a workspace member; kept as reproducible standalone evidence for
// docs/research/00-verified-environment.md §3. The four findings above are asserted, and will fail
// loudly on regression, in crates/myelin-core/tests/qdrant_capability.rs (milestone M0) — run it with
// `cargo test -p myelin-core --features integration`.
// Cargo.toml used here:  qdrant-client = { version = "1", features = ["serde"] }, tokio (full), anyhow

use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeleteCollectionBuilder, Distance, MultiVectorComparator,
    MultiVectorConfigBuilder, PointStruct, QueryPointsBuilder, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, UpsertPointsBuilder, VectorParamsBuilder, VectorsConfigBuilder,
    Fusion, PrefetchQueryBuilder, Query,
};
use qdrant_client::{Payload, Qdrant};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let c = Qdrant::from_url("http://192.168.1.110:6334").build()?;
    let name = "mp_grpc_probe";
    let _ = c.delete_collection(DeleteCollectionBuilder::new(name)).await;

    let mut vc = VectorsConfigBuilder::default();
    vc.add_named_vector_params("dense", VectorParamsBuilder::new(4, Distance::Cosine));
    vc.add_named_vector_params(
        "late",
        VectorParamsBuilder::new(4, Distance::Cosine)
            .multivector_config(MultiVectorConfigBuilder::new(MultiVectorComparator::MaxSim)),
    );
    let mut sv = SparseVectorsConfigBuilder::default();
    sv.add_named_vector_params("lex", SparseVectorParamsBuilder::default());

    c.create_collection(
        CreateCollectionBuilder::new(name)
            .vectors_config(vc)
            .sparse_vectors_config(sv),
    )
    .await?;
    println!("create: ok");

    use qdrant_client::qdrant::{NamedVectors, Vector};
    let p1 = PointStruct::new(
        1u64,
        NamedVectors::default()
            .add_vector("dense", Vector::new_dense(vec![1.0, 0.0, 0.0, 0.0]))
            .add_vector("late", Vector::new_multi(vec![vec![1.0,0.0,0.0,0.0], vec![0.0,1.0,0.0,0.0]]))
            .add_vector("lex", Vector::new_sparse(vec![7u32, 42], vec![0.9, 0.4])),
        Payload::new(),
    );
    let p2 = PointStruct::new(
        2u64,
        NamedVectors::default()
            .add_vector("dense", Vector::new_dense(vec![0.0, 1.0, 0.0, 0.0]))
            .add_vector("late", Vector::new_multi(vec![vec![0.0,0.0,1.0,0.0]]))
            .add_vector("lex", Vector::new_sparse(vec![42u32, 99], vec![0.7, 0.5])),
        Payload::new(),
    );
    c.upsert_points(UpsertPointsBuilder::new(name, vec![p1, p2]).wait(true)).await?;
    println!("upsert(named dense+multivector+sparse): ok");

    // hybrid RRF fusion, server-side
    let r = c.query(
        QueryPointsBuilder::new(name)
            .add_prefetch(PrefetchQueryBuilder::default().query(Query::new_nearest(vec![1.0f32,0.0,0.0,0.0])).using("dense").limit(10u64))
            .add_prefetch(PrefetchQueryBuilder::default().query(Query::new_nearest(qdrant_client::qdrant::VectorInput::new_sparse(vec![42u32], vec![1.0]))).using("lex").limit(10u64))
            .query(Query::new_fusion(Fusion::Rrf))
            .limit(5u64),
    ).await?;
    println!("rrf: {:?}", r.result.iter().map(|p| (p.id.clone().map(|i| format!("{:?}", i.point_id_options)), p.score)).collect::<Vec<_>>());

    // late-interaction rerank of dense prefetch, server-side max_sim
    let r2 = c.query(
        QueryPointsBuilder::new(name)
            .add_prefetch(PrefetchQueryBuilder::default().query(Query::new_nearest(vec![1.0f32,0.0,0.0,0.0])).using("dense").limit(10u64))
            .query(Query::new_nearest(qdrant_client::qdrant::VectorInput::new_multi(vec![vec![1.0f32,0.0,0.0,0.0], vec![0.0,0.0,1.0,0.0]])))
            .using("late")
            .limit(3u64),
    ).await?;
    println!("maxsim-rerank: {:?}", r2.result.iter().map(|p| p.score).collect::<Vec<_>>());


    // pin the RRF constant: doc at rank 3 in exactly one list
    let mut pts = Vec::new();
    for i in 0..4u64 {
        let mut d = vec![0.0f32; 4];
        d[(i % 4) as usize] = 1.0;
        pts.push(PointStruct::new(
            100 + i,
            NamedVectors::default()
                .add_vector("dense", Vector::new_dense(vec![1.0 - 0.1 * (i as f32), 0.1 * (i as f32), 0.0, 0.0]))
                .add_vector("late", Vector::new_multi(vec![d.clone()]))
                .add_vector("lex", Vector::new_sparse(vec![1000u32 + i as u32], vec![1.0])),
            Payload::new(),
        ));
    }
    c.upsert_points(UpsertPointsBuilder::new(name, pts).wait(true)).await?;
    let r3 = c.query(
        QueryPointsBuilder::new(name)
            .add_prefetch(PrefetchQueryBuilder::default().query(Query::new_nearest(vec![1.0f32,0.0,0.0,0.0])).using("dense").limit(10u64))
            .query(Query::new_fusion(Fusion::Rrf))
            .limit(10u64),
    ).await?;
    println!("rrf-single-list scores (rank1..n): {:?}", r3.result.iter().map(|p| p.score).collect::<Vec<_>>());

    c.delete_collection(DeleteCollectionBuilder::new(name)).await?;
    println!("cleanup: ok");
    Ok(())
}
