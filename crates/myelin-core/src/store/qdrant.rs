//! Qdrant access. **gRPC only.**
//!
//! Do not add a REST path. On Qdrant 1.19.1 a multivector *query* over the REST
//! API fails with `422 Validation error in JSON body: [internal.query.indices:
//! must be unique]` — the untagged `VectorInput` enum mis-parses a nested float
//! matrix as a sparse vector. Multivector *upsert* over REST is fine; only the
//! query path is broken, and the query path is the one late-interaction rerank
//! needs. Measured in `docs/research/00-verified-environment.md` §3.3 and pinned
//! by `tests/qdrant_capability.rs::server_side_maxsim_rerank_works`.
//!
//! One collection carries all three retrieval channels (`PLAN.md` §5.1):
//! `dense` (bge-m3, Cosine), optional `late` (multivector max_sim), and `lex`
//! (sparse, `modifier: idf`). `lex` is computed **inside Qdrant** from the
//! record text via `qdrant/bm25` — verified over gRPC by
//! `tests/qdrant_capability.rs::bm25_document_inference_over_grpc`, so there is
//! no client-side BM25 encoder and no second index.

use std::collections::HashMap;

use chrono::{DateTime, SecondsFormat, Utc};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, DeleteCollectionBuilder,
    DeletePointsBuilder, Distance, Document, FieldType, Filter, GetPointsBuilder, Modifier,
    MultiVectorComparator, MultiVectorConfigBuilder, NamedVectors, PointId, PointStruct,
    PointsIdsList, Query, QueryPointsBuilder, ScrollPointsBuilder, SetPayloadPointsBuilder,
    SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, UpdateStatus, UpsertPointsBuilder, Value, Vector, VectorInput,
    VectorParamsBuilder, VectorsConfigBuilder,
};
use qdrant_client::{Payload, Qdrant};
use uuid::Uuid;

use crate::config::QdrantConfig;
use crate::error::{MyelinError, Result};
use crate::model::record::MemoryRecord;

/// Named dense channel.
pub const DENSE: &str = "dense";
/// Named multivector (late-interaction) channel. Optional; see §5.1.
pub const LATE: &str = "late";
/// Named sparse channel, BM25 with the IDF modifier.
pub const LEX: &str = "lex";
/// The model name Qdrant computes in-process. Arithmetic, not a neural model,
/// which is why it works without an inference service.
pub const BM25_MODEL: &str = "qdrant/bm25";

/// Build a gRPC client for the configured endpoint.
///
/// The explicit timeout is not decoration. `qdrant-client` defaults to 5 s,
/// and [`QdrantStore::ensure_collection`] issues ten control-plane calls in a
/// row — one `create_collection` plus nine `create_field_index`. Against a
/// Qdrant that is concurrently indexing another collection this exceeds 5 s
/// and fails with `Cancelled: Timeout expired`, which is how two
/// simultaneous integration tests found it.
///
/// This is a *failure* bound, not a latency target: the read path measures
/// itself (`RecallTrace`), so a generous ceiling here cannot hide a slow
/// query — it only stops a loaded server from looking like a broken one.
pub fn client(cfg: &QdrantConfig) -> Result<Qdrant> {
    Ok(Qdrant::from_url(&cfg.url)
        .timeout(std::time::Duration::from_secs(60))
        .build()?)
}

fn epoch(t: DateTime<Utc>) -> i64 {
    t.timestamp()
}

fn rfc3339(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Nanos, true)
}

/// A record plus the vectors we produced for it. `dense` is ours to compute
/// (Qdrant has no local inference service for neural models); `lex` is derived
/// server-side from the text.
pub struct IndexItem<'a> {
    pub record: &'a MemoryRecord,
    pub dense: Vec<f32>,
    /// Optional ColBERT-style multivector. Left empty until M3 settles whether
    /// `fastembed` exposes bge-m3's ColBERT head at all (§5.1 risk i).
    pub late: Option<Vec<Vec<f32>>>,
}

/// One hit from a single retrieval channel, with the channel's own score.
/// `text` rides along from the payload so the fast path never needs a SQLite
/// round trip (§7.1 targets p95 < 100 ms).
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredHit {
    pub id: Uuid,
    pub score: f32,
    pub text: String,
    /// The stored dense vector, requested only when the caller asked for it
    /// ([`SearchScope`] is about *which* points; this is about how much of
    /// each point comes back). `compose`'s cosine near-duplicate suppression
    /// is inert without it, and returning ~1024 floats per hit on every
    /// recall is not free, so it is opt-in.
    pub vector: Option<Vec<f32>>,
}

/// The scope predicates a search applies *before* ranking.
///
/// A struct rather than four positional `&str`/`Option<&str>` arguments
/// because transposing two of them at a call site is silent — they all have
/// the same type — and a transposed scope predicate is precisely the leak
/// this type exists to prevent (C12).
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchScope<'a> {
    /// Mandatory. There is no read path that spans tenants.
    pub tenant: &'a str,
    pub namespace: Option<&'a str>,
    pub agent: Option<&'a str>,
    pub session: Option<&'a str>,
}

/// The two channels, each ranked by its own scorer. Deliberately NOT fused
/// here: fusion is ours (§5.2).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HybridLists {
    pub dense: Vec<ScoredHit>,
    pub lex: Vec<ScoredHit>,
}

/// The payload fields the scope filter and the reconciler both read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadSnapshot {
    pub tenant: String,
    pub agent: String,
    pub session: Option<String>,
    pub namespace: String,
    pub kind: String,
    pub trust_tier: String,
    /// Seconds since the epoch. `None` for points written before the field
    /// entered the payload (pre-migration): the reconciler then flags the point
    /// as `missing_t_valid` — a migration gap, *not* `payload_drift` — and
    /// restores it via `sync_payload` on `--repair`. Carried because a
    /// `t_valid` that drifts from the ledger's is invisible everywhere else:
    /// M19 shipped 162,181 records stamped with the build date and nothing
    /// noticed for six milestones.
    pub t_valid: Option<i64>,
    pub t_invalid: Option<i64>,
}

pub struct QdrantStore {
    client: Qdrant,
    collection: String,
}

impl QdrantStore {
    pub fn new(cfg: &QdrantConfig) -> Result<Self> {
        Ok(Self {
            client: client(cfg)?,
            collection: cfg.collection.clone(),
        })
    }

    pub fn with_collection(cfg: &QdrantConfig, collection: impl Into<String>) -> Result<Self> {
        Ok(Self {
            client: client(cfg)?,
            collection: collection.into(),
        })
    }

    pub fn collection(&self) -> &str {
        &self.collection
    }

    pub fn client(&self) -> &Qdrant {
        &self.client
    }

    pub async fn exists(&self) -> Result<bool> {
        Ok(self.client.collection_exists(&self.collection).await?)
    }

    /// Create the collection and its payload indexes if absent. Idempotent.
    pub async fn ensure_collection(&self, dense_dim: u64, late: bool) -> Result<()> {
        if self.exists().await? {
            return Ok(());
        }

        let mut vectors = VectorsConfigBuilder::default();
        vectors.add_named_vector_params(DENSE, VectorParamsBuilder::new(dense_dim, Distance::Cosine));
        if late {
            vectors.add_named_vector_params(
                LATE,
                VectorParamsBuilder::new(dense_dim, Distance::Cosine).multivector_config(
                    MultiVectorConfigBuilder::new(MultiVectorComparator::MaxSim),
                ),
            );
        }

        let mut sparse = SparseVectorsConfigBuilder::default();
        sparse.add_named_vector_params(
            LEX,
            SparseVectorParamsBuilder::default().modifier(Modifier::Idf),
        );

        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection)
                    .vectors_config(vectors)
                    .sparse_vectors_config(sparse),
            )
            .await?;

        // Scope-before-routing (§2 finding 3): these predicates mask
        // inadmissible shards before ranking, so they must be indexed.
        for (field, ty) in [
            ("tenant", FieldType::Keyword),
            ("agent", FieldType::Keyword),
            ("session", FieldType::Keyword),
            ("namespace", FieldType::Keyword),
            ("kind", FieldType::Keyword),
            ("trust_tier", FieldType::Keyword),
            ("entity_ids", FieldType::Keyword),
            ("t_valid", FieldType::Integer),
            ("t_invalid", FieldType::Integer),
        ] {
            self.client
                .create_field_index(CreateFieldIndexCollectionBuilder::new(
                    &self.collection,
                    field,
                    ty,
                ))
                .await?;
        }
        Ok(())
    }

    /// Payload mirrors the ledger's filterable columns. `text` rides along so
    /// the fast path (§7.1, p95 < 100 ms) never needs a SQLite round trip.
    pub fn payload_of(record: &MemoryRecord) -> Payload {
        let mut map: HashMap<String, Value> = HashMap::new();
        map.insert("tenant".into(), record.scope.tenant.clone().into());
        map.insert("agent".into(), record.scope.agent.clone().into());
        map.insert(
            "session".into(),
            match &record.scope.session {
                Some(s) => s.clone().into(),
                None => Value::from(""),
            },
        );
        map.insert("namespace".into(), record.scope.namespace.clone().into());
        map.insert("kind".into(), record.kind.as_str().into());
        map.insert("trust_tier".into(), record.trust.tier.as_str().into());
        map.insert("text".into(), record.text.clone().into());
        map.insert("t_valid".into(), epoch(record.validity.t_valid).into());
        map.insert(
            "t_invalid".into(),
            // 0 means "still believed": Qdrant range filters cannot express
            // "field absent OR > now" in one predicate, and a sentinel keeps
            // the scope filter to a single range clause.
            record.validity.t_invalid.map(epoch).unwrap_or(0).into(),
        );
        map.insert(
            "t_ingested_rfc3339".into(),
            rfc3339(record.validity.t_ingested).into(),
        );
        map.insert(
            "entity_ids".into(),
            record
                .entities
                .iter()
                .map(|e| Value::from(e.phrase.clone()))
                .collect::<Vec<_>>()
                .into(),
        );
        Payload::from(map)
    }

    pub async fn upsert(&self, items: &[IndexItem<'_>]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let points: Vec<PointStruct> = items
            .iter()
            .map(|item| {
                let mut vectors = NamedVectors::default()
                    .add_vector(DENSE, Vector::new_dense(item.dense.clone()))
                    // Server-side BM25: Qdrant tokenizes, stems, drops
                    // stopwords and stores TF weights; IDF is applied at query
                    // time. No client-side lexical encoder (§5.1).
                    .add_vector(
                        LEX,
                        Vector::from(Document::new(item.record.text.clone(), BM25_MODEL)),
                    );
                if let Some(late) = &item.late {
                    vectors = vectors.add_vector(LATE, Vector::new_multi(late.clone()));
                }
                PointStruct::new(
                    item.record.id.to_string(),
                    vectors,
                    Self::payload_of(item.record),
                )
            })
            .collect();

        let response = self
            .client
            .upsert_points(UpsertPointsBuilder::new(&self.collection, points).wait(true))
            .await?;
        let status = response
            .result
            .ok_or_else(|| MyelinError::Store("upsert returned no result".into()))?
            .status;
        if status != UpdateStatus::Completed as i32 {
            return Err(MyelinError::Store(format!(
                "upsert did not complete: status {status}"
            )));
        }
        Ok(())
    }

    pub async fn delete_points(&self, ids: &[Uuid]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let list = PointsIdsList {
            ids: ids.iter().map(|i| PointId::from(i.to_string())).collect(),
        };
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection)
                    .points(list)
                    .wait(true),
            )
            .await?;
        Ok(())
    }

    /// Exact point count for the collection.
    ///
    /// `exact(true)`: the approximate count is a cardinality estimate, and
    /// the only caller compares it against a SQL `COUNT(*)` to decide
    /// whether a ledger and a collection are the same store.
    pub async fn count(&self) -> Result<u64> {
        use qdrant_client::qdrant::CountPointsBuilder;

        let response = self
            .client
            .count(CountPointsBuilder::new(&self.collection).exact(true))
            .await?;
        Ok(response.result.map(|r| r.count).unwrap_or_default())
    }

    /// Overwrite the mutable payload fields of an existing point. Used by
    /// [`crate::store::reconcile`] to repair flag drift without re-embedding.
    pub async fn sync_payload(&self, record: &MemoryRecord) -> Result<()> {
        let list = PointsIdsList {
            ids: vec![PointId::from(record.id.to_string())],
        };
        self.client
            .set_payload(
                SetPayloadPointsBuilder::new(&self.collection, Self::payload_of(record))
                    .points_selector(list)
                    .wait(true),
            )
            .await?;
        Ok(())
    }

    pub async fn get_payload(&self, id: Uuid) -> Result<Option<PayloadSnapshot>> {
        let response = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.collection, vec![PointId::from(id.to_string())])
                    .with_payload(true),
            )
            .await?;
        Ok(response
            .result
            .first()
            .map(|p| snapshot_from(&p.payload)))
    }

    /// Nearest live records in a scope, by dense similarity.
    ///
    /// This exists because the obvious client-side alternative — pull the
    /// candidate pool from SQLite and embed all of it — is quadratic in
    /// corpus size. Measured: consolidating one LoCoMo conversation that way
    /// managed 43 episodes in 900 s, because by episode 43 every candidate
    /// was re-embedding a ~200-record pool. Qdrant already holds these
    /// vectors; asking it is one round trip and no embedding at all.
    ///
    /// Filters mirror the ledger's admissibility rules (C7, I3): same tenant
    /// and namespace, not retracted, not quarantined. `t_invalid` uses the 0
    /// sentinel written by [`QdrantStore::payload_of`].
    pub async fn search_dense(
        &self,
        vector: Vec<f32>,
        tenant: &str,
        namespace: &str,
        limit: u64,
    ) -> Result<Vec<(Uuid, f32)>> {
        use qdrant_client::qdrant::Condition;

        let filter = Filter {
            must: vec![
                Condition::matches("tenant", tenant.to_string()),
                Condition::matches("namespace", namespace.to_string()),
                Condition::matches("t_invalid", 0i64),
            ],
            must_not: vec![Condition::matches(
                "trust_tier",
                "quarantined".to_string(),
            )],
            ..Default::default()
        };

        let response = self
            .client
            .query(
                QueryPointsBuilder::new(&self.collection)
                    .query(Query::new_nearest(vector))
                    .using(DENSE)
                    .filter(filter)
                    .limit(limit),
            )
            .await?;

        Ok(response
            .result
            .iter()
            .filter_map(|p| {
                let id = p.id.as_ref()?.point_id_options.as_ref()?;
                match id {
                    qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u) => {
                        Uuid::parse_str(u.as_str()).ok().map(|id| (id, p.score))
                    }
                    qdrant_client::qdrant::point_id::PointIdOptions::Num(_) => None,
                }
            })
            .collect())
    }

    /// Both retrieval channels, ranked separately, in **one** round trip.
    ///
    /// This is the shape `PLAN.md` §5.2 requires: we need the dense list and
    /// the lexical list as *lists*, not as a server-fused result, because
    /// Qdrant's own RRF uses `k = 1` and we fuse at `k = 60`
    /// (`pipeline::fuse`). A `prefetch` would only hand back the fused top-N,
    /// so the two queries go out as a `query_batch` — one request, one
    /// round trip, two independent rankings.
    ///
    /// The lexical side sends **raw text**, not a client-side sparse vector:
    /// Qdrant tokenizes, stems and IDF-weights it in-process
    /// (`bm25_document_inference_over_grpc`), which is why there is no
    /// `tantivy` and no second index.
    ///
    /// Every member of `scope` becomes a pre-ranking `must` condition. An
    /// `agent` or `session` predicate that the caller supplied and this
    /// filter dropped would be a cross-agent read that no later stage can
    /// undo, because fusion and rerank only ever narrow by score.
    pub async fn hybrid_search(
        &self,
        dense: Vec<f32>,
        text: &str,
        scope: SearchScope<'_>,
        kinds: &[&str],
        limit: u64,
        with_vectors: bool,
    ) -> Result<HybridLists> {
        use qdrant_client::qdrant::{Condition, QueryBatchPointsBuilder};

        let mut must = vec![
            Condition::matches("tenant", scope.tenant.to_string()),
            Condition::matches("t_invalid", 0i64),
        ];
        if let Some(ns) = scope.namespace {
            must.push(Condition::matches("namespace", ns.to_string()));
        }
        if let Some(agent) = scope.agent {
            must.push(Condition::matches("agent", agent.to_string()));
        }
        if let Some(session) = scope.session {
            must.push(Condition::matches("session", session.to_string()));
        }
        if !kinds.is_empty() {
            must.push(Condition::matches(
                "kind",
                kinds.iter().map(|k| k.to_string()).collect::<Vec<_>>(),
            ));
        }
        let filter = Filter {
            must,
            // I3: quarantined material is invisible to every read path.
            must_not: vec![Condition::matches("trust_tier", "quarantined".to_string())],
            ..Default::default()
        };

        // The lex-only ablation arm passes no dense vector. An empty vector
        // is not a degenerate case Qdrant tolerates: it rejects the entire
        // batch with `Vector dimension error: expected dim: 1024, got 0`. So
        // the sub-query is omitted rather than emptied, which also makes that
        // arm's measured latency honest.
        let want_dense = !dense.is_empty();
        let mut queries: Vec<qdrant_client::qdrant::QueryPoints> = Vec::with_capacity(2);
        if want_dense {
            let mut builder = QueryPointsBuilder::new(&self.collection)
                .query(Query::new_nearest(dense))
                .using(DENSE)
                .filter(filter.clone())
                .limit(limit)
                .with_payload(true);
            if with_vectors {
                // Name the dense vector explicitly: `with_vectors(true)`
                // would also ship the server-side BM25 sparse vector, which
                // nothing downstream reads.
                builder = builder.with_vectors(
                    qdrant_client::qdrant::with_vectors_selector::SelectorOptions::Include(
                        qdrant_client::qdrant::VectorsSelector {
                            names: vec![DENSE.to_string()],
                        },
                    ),
                );
            }
            queries.push(builder.into());
        }
        queries.push(
            QueryPointsBuilder::new(&self.collection)
                .query(Query::new_nearest(VectorInput::from(Document::new(
                    text.to_string(),
                    BM25_MODEL,
                ))))
                .using(LEX)
                .filter(filter)
                .limit(limit)
                .with_payload(true)
                .into(),
        );

        let response = self
            .client
            .query_batch(QueryBatchPointsBuilder::new(&self.collection, queries))
            .await?;

        let mut lists = response
            .result
            .iter()
            .map(|batch| {
                batch
                    .result
                    .iter()
                    .filter_map(|p| {
                        let id = p.id.as_ref()?.point_id_options.as_ref()?;
                        let text = p
                            .payload
                            .get("text")
                            .and_then(|v| v.as_str().map(|s| s.to_string()))
                            .unwrap_or_default();
                        let vector = dense_vector_of(p);
                        match id {
                            qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u) => {
                                Uuid::parse_str(u.as_str()).ok().map(|id| ScoredHit {
                                    id,
                                    score: p.score,
                                    text,
                                    vector,
                                })
                            }
                            qdrant_client::qdrant::point_id::PointIdOptions::Num(_) => None,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Order is the order the queries were submitted in.
        let lex = lists.pop().unwrap_or_default();
        let dense = if want_dense {
            lists.pop().unwrap_or_default()
        } else {
            Vec::new()
        };
        Ok(HybridLists { dense, lex })
    }

    /// Every point id in a namespace, with the payload fields reconcile checks.
    pub async fn scroll_namespace(&self, namespace: &str) -> Result<Vec<(Uuid, PayloadSnapshot)>> {
        let mut out = Vec::new();
        let mut offset: Option<PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(&self.collection)
                .filter(Filter::must([qdrant_client::qdrant::Condition::matches(
                    "namespace",
                    namespace.to_string(),
                )]))
                .with_payload(true)
                .limit(256);
            if let Some(o) = offset.clone() {
                builder = builder.offset(o);
            }
            let response = self.client.scroll(builder).await?;
            for point in &response.result {
                let Some(id) = point.id.as_ref() else { continue };
                let Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) =
                    id.point_id_options.as_ref()
                else {
                    continue;
                };
                out.push((
                    Uuid::parse_str(u)
                        .map_err(|e| MyelinError::Store(format!("bad point uuid {u}: {e}")))?,
                    snapshot_from(&point.payload),
                ));
            }
            offset = response.next_page_offset;
            if offset.is_none() {
                break;
            }
        }
        out.sort_by_key(|(id, _)| *id);
        Ok(out)
    }

    pub async fn drop_collection(&self) -> Result<()> {
        self.client
            .delete_collection(DeleteCollectionBuilder::new(&self.collection))
            .await?;
        Ok(())
    }
}

fn snapshot_from(payload: &HashMap<String, Value>) -> PayloadSnapshot {
    let s = |k: &str| -> String {
        payload
            .get(k)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default()
    };
    let session = s("session");
    PayloadSnapshot {
        tenant: s("tenant"),
        agent: s("agent"),
        session: if session.is_empty() {
            None
        } else {
            Some(session)
        },
        namespace: s("namespace"),
        kind: s("kind"),
        trust_tier: s("trust_tier"),
        t_valid: payload.get("t_valid").and_then(|v| v.as_integer()),
        t_invalid: payload
            .get("t_invalid")
            .and_then(|v| v.as_integer())
            .filter(|i| *i != 0),
    }
}

/// The `DENSE` vector of a scored point, when the query asked for vectors.
///
/// Returns `None` rather than an empty vector for an absent or non-dense
/// payload: `compose` treats `None` as "fall back to exact-text dedup", and a
/// zero-length vector would make every cosine `NaN` instead.
fn dense_vector_of(point: &qdrant_client::qdrant::ScoredPoint) -> Option<Vec<f32>> {
    use qdrant_client::qdrant::{vector_output, vectors_output::VectorsOptions};

    let out = match point.vectors.as_ref()?.vectors_options.as_ref()? {
        VectorsOptions::Vector(v) => v,
        VectorsOptions::Vectors(named) => named.vectors.get(DENSE)?,
    };
    match out.vector.as_ref() {
        Some(vector_output::Vector::Dense(d)) => Some(d.data.clone()),
        // Pre-1.16 servers send the flattened `data` field instead of the
        // typed oneof; both wire shapes are live in the fleet.
        _ => {
            #[allow(deprecated)]
            let legacy = &out.data;
            (!legacy.is_empty()).then(|| legacy.clone())
        }
    }
}
