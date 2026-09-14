//! Qdrant access. **gRPC only.**
//!
//! Do not add a REST path. On Qdrant 1.19.1 a multivector *query* over the REST
//! API fails with `422 Validation error in JSON body: [internal.query.indices:
//! must be unique]` — the untagged `VectorInput` enum mis-parses a nested float
//! matrix as a sparse vector. Multivector *upsert* over REST is fine; only the
//! query path is broken, and the query path is the one late-interaction rerank
//! needs. Measured in `docs/research/00-verified-environment.md` §3.3 and pinned
//! by `tests/qdrant_capability.rs::server_side_maxsim_rerank_works`.

use qdrant_client::Qdrant;

use crate::config::QdrantConfig;
use crate::error::Result;

/// Build a gRPC client for the configured endpoint.
pub fn client(cfg: &QdrantConfig) -> Result<Qdrant> {
    Ok(Qdrant::from_url(&cfg.url).build()?)
}
