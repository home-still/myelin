//! HTTP dense embedder against an OpenAI-compatible `/v1/embeddings`
//! endpoint — llama.cpp's server on `big` (Qwen3-Embedding-8B Q8_0 at
//! `192.168.1.110:5811`) and ollama.
//!
//! # Empty-body guard (R7)
//!
//! llama-swap returns `200` with a **zero-byte body** when the upstream model
//! fails to load (`docs/research/00-verified-environment.md` §7.2). The body is
//! read as text and checked for emptiness *before* JSON parsing, mirroring
//! [`crate::llm::openai`]. Deserializing first would surface a serde error —
//! the wrong diagnosis, and exactly what cost ~21 h of silent stall the first
//! time.
//!
//! # Matryoshka truncation
//!
//! Qwen3-Embedding is trained with Matryoshka Representation Learning (Kusupati
//! et al., 2022 — "Matryoshka Representation Learning", NeurIPS 2022): a prefix
//! of the full vector is itself a valid embedding at a lower dimensionality.
//! This is what reconciles the 4096-d model with the 1024-d dense channel
//! `PLAN.md` §5.1 specifies — we request the full vector and truncate to the
//! prefix locally. Truncation breaks the L2 norm, and cosine distance (the
//! Qdrant `Cosine` comparator) requires unit vectors, so the prefix is
//! re-normalized after truncation. A vector shorter than the target is an
//! error, never padded: padding would invent zero-magnitude dimensions that
//! distort the cosine geometry.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::error::{MyelinError, Result};

use super::Embedder;

/// OpenAI-compatible `/v1/embeddings` client.
pub struct RemoteEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    target_dim: u64,
    /// Stable id written into `MemoryConfigJson`. Must change when
    /// `target_dim` changes — a different dimension is a different memory
    /// (R3, `PLAN.md` §4.1).
    id: String,
}

impl RemoteEmbedder {
    /// `base_url` is the `/v1` root, e.g. `http://192.168.1.110:5811/v1`.
    /// `target_dim` is the desired output dimensionality; the server's
    /// full-dimension vector is truncated (Matryoshka) and re-normalized to
    /// this width.
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        target_dim: u64,
    ) -> Result<Self> {
        let model: String = model.into();
        let base_url = base_url.into().trim_end_matches('/').to_string();
        // A cold GGUF spawn off local NVMe takes ~15 s and a 27B off NFS took
        // ~4.5 min; the default 30 s would time out a cold start and look like
        // a model failure — same rationale as `llm::openai::OpenAiLlm::new`.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .map_err(|e| MyelinError::Store(format!("http client: {e}")))?;

        let id = format!("{model}@{target_dim}");

        Ok(Self {
            client,
            base_url,
            model,
            target_dim,
            id,
        })
    }
}

#[async_trait]
impl Embedder for RemoteEmbedder {
    fn dim(&self) -> u64 {
        self.target_dim
    }

    fn id(&self) -> &str {
        &self.id
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = format!("{}/embeddings", self.base_url);
        let body = json!({
            "model": self.model,
            "input": texts,
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| MyelinError::Store(format!("{}: request to {url} failed: {e}", self.model)))?;

        let status = response.status();
        let body_text = response
            .text()
            .await
            .map_err(|e| MyelinError::Store(format!("{}: reading body: {e}", self.model)))?;

        // R7, transport half: an empty 200 is a model-load failure, not
        // malformed JSON.
        if body_text.trim().is_empty() {
            return Err(MyelinError::EmptyCompletion {
                model: self.model.clone(),
            });
        }
        if !status.is_success() {
            return Err(MyelinError::Store(format!(
                "{}: HTTP {status}: {}",
                self.model,
                body_text.chars().take(400).collect::<String>()
            )));
        }

        let parsed: EmbeddingsResponse = serde_json::from_str(&body_text).map_err(|e| {
            MyelinError::Store(format!(
                "{}: unparseable embeddings response: {e}; body was {:?}",
                self.model,
                body_text.chars().take(400).collect::<String>()
            ))
        })?;

        if parsed.data.len() != texts.len() {
            return Err(MyelinError::Store(format!(
                "{}: expected {} embeddings, got {}",
                self.model,
                texts.len(),
                parsed.data.len()
            )));
        }

        // Re-index by the server's `index` field so output order matches input
        // order. The OpenAI spec does not guarantee ordering, and batching
        // backends (llama.cpp, ollama) may return out of order.
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);

        let target = self.target_dim as usize;
        data.into_iter()
            .map(|d| fit_dim(d.embedding, target))
            .collect()
    }
}

/// Truncate a full-dimension embedding to `target` dimensions (Matryoshka
/// prefix) and L2-re-normalize. The prefix is a valid embedding per MRL
/// (Kusupati et al., 2022), but truncation breaks the unit norm that cosine
/// distance requires, so re-normalization is mandatory. A vector shorter than
/// the target is an error — padding would inject zero-magnitude dimensions
/// that distort cosine geometry. A zero vector (degenerate) returns zeros
/// rather than NaN — the division is guarded.
fn fit_dim(v: Vec<f32>, target: usize) -> Result<Vec<f32>> {
    if v.len() < target {
        return Err(MyelinError::Store(format!(
            "embedder returned {} dimensions, need {target}",
            v.len()
        )));
    }

    let mut out = v.into_iter().take(target).collect::<Vec<_>>();

    // L2 normalize. Guard against a zero vector producing NaN.
    let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut out {
            *x /= norm;
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    /// Position in the original `input` array. Used to restore input order.
    #[serde(default)]
    index: usize,
}

// ---------------------------------------------------------------------------
// Tests — no network; exercise the pure `fit_dim` helper directly.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::fit_dim;

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn fit_dim_truncates_4096_to_1024_and_renormalizes() {
        // Build a 4096-d vector: first 1024 components carry signal, the rest
        // are noise. After truncation the norm must be ~1.0 and length 1024.
        let mut v = vec![0.0f32; 4096];
        for (i, slot) in v.iter_mut().enumerate().take(1024) {
            *slot = (i as f32 + 1.0) * 0.001;
        }
        for slot in v.iter_mut().skip(1024) {
            *slot = 0.5;
        }

        let out = fit_dim(v, 1024).expect("truncation should succeed");
        assert_eq!(out.len(), 1024);
        let norm = l2_norm(&out);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "truncated vector must be unit norm, got {norm}"
        );
    }

    #[test]
    fn fit_dim_returns_unit_norm_when_already_at_target() {
        // A 4-d vector already at target length — truncation is a no-op, but
        // renormalization still applies.
        let v = vec![3.0, 4.0, 0.0, 0.0]; // norm = 5.0
        let out = fit_dim(v, 4).expect("no-op truncation should succeed");
        assert_eq!(out.len(), 4);
        let norm = l2_norm(&out);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "already-target vector must be unit norm, got {norm}"
        );
        // Spot-check the direction is preserved: [0.6, 0.8, 0, 0].
        assert!((out[0] - 0.6).abs() < 1e-5);
        assert!((out[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn fit_dim_errors_when_target_exceeds_input() {
        let v = vec![1.0, 2.0, 3.0];
        let err = fit_dim(v, 10).expect_err("target > input must error");
        let msg = err.to_string();
        assert!(
            msg.contains("need 10"),
            "error should mention the needed dimension, got: {msg}"
        );
    }

    #[test]
    fn fit_dim_zero_vector_does_not_produce_nan() {
        // A degenerate all-zero vector: the norm guard must prevent division by
        // zero, yielding zeros (not NaN) rather than panicking.
        let v = vec![0.0; 8];
        let out = fit_dim(v, 4).expect("zero vector should not error");
        assert_eq!(out.len(), 4);
        for x in &out {
            assert!(
                !x.is_nan(),
                "zero vector must not produce NaN after normalization"
            );
            assert!(x.is_finite(), "output must be finite");
        }
    }
}