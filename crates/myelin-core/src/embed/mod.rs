//! Dense embedding. Qdrant has no local inference service for neural models
//! (`500 InferenceService URL not configured`, measured in
//! `docs/research/00-verified-environment.md` §3.5), so dense vectors are ours
//! to produce.
//!
//! **House rule: no silent CPU fallback.** `hs-distill/src/embed/mod.rs`
//! establishes it and the reason is latency honesty — a run that quietly
//! dropped to CPU produces latency numbers that are not the numbers we claim.
//! A backend asked for CUDA that cannot have CUDA must error, not degrade.
//!
//! `local.rs` (fastembed + ort) and `remote.rs` (hs-distill HTTP / ollama
//! bge-m3) land in M3; this is the seam they plug into.

use async_trait::async_trait;

use crate::error::Result;

pub mod remote;

#[async_trait]
pub trait Embedder: Send + Sync {
    /// Dimensionality of the dense vectors this embedder produces.
    fn dim(&self) -> u64;

    /// Stable identifier recorded in the run manifest and in
    /// [`crate::store::export::MemoryConfigJson`]. Changing the embedder must
    /// change this string, because it invalidates a built memory (R3).
    fn id(&self) -> &str;

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
