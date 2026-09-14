//! Reranking (`PLAN.md` §3.1, §2 finding 2).
//!
//! **This is the highest-leverage stage in retrieval, by a wide margin.**
//! MS MARCO dev MRR@10: BM25-anserini **18.7** → BERT-large cross-encoder over
//! BM25 top-1000 **36.5** → RocketQAv2-ERNIE **40.1**
//! (`10.18653/v1_2021.emnlp-main.224` Table 3). Fusion contributes ~1–2 points
//! by comparison. Run both; rerank is the lever.
//!
//! Two implementations are planned and only one is built. `cross.rs` is a
//! bge-reranker cross-encoder and is the default. Late-interaction max_sim in
//! Qdrant stays unbuilt until M4 answers the two open questions in §5.1:
//! whether our `fastembed` build exposes bge-m3's ColBERT head at all, and
//! whether late interaction beats a cross-encoder at *equal latency* on our
//! data. Until both are answered the `late` vector channel stays unpopulated,
//! because a ColBERT vector is 1024-d **per token** — two orders of magnitude
//! more storage than the dense vector for the same record.

use async_trait::async_trait;

use crate::error::Result;

pub mod cross;

#[async_trait]
pub trait Reranker: Send + Sync {
    /// Stable identifier, recorded in the run manifest.
    fn id(&self) -> &str;

    /// Score each document against the query. Higher is more relevant.
    ///
    /// Returns one score per input document, in input order. Implementations
    /// must not reorder: the caller owns ordering, because it also owns the
    /// tie-break rule that keeps runs reproducible.
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>>;
}
