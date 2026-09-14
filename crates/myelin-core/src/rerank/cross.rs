//! Cross-encoder reranking via llama.cpp's `/v1/rerank` endpoint.
//!
//! `bge-reranker-v2-m3` served by `llama-server --reranking`. A cross-encoder
//! sees the query and the document *together*, which is exactly why it beats
//! bi-encoder similarity: it can model term interaction that a dot product
//! between two independently-computed vectors cannot.
//!
//! It is also why it is expensive — one forward pass per (query, document)
//! pair — and therefore why it runs over the fused head rather than the whole
//! candidate pool.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::error::{MyelinError, Result};

use super::Reranker;

pub struct CrossEncoder {
    client: reqwest::Client,
    url: String,
    model: String,
}

impl CrossEncoder {
    /// `base_url` is the server root, e.g. `http://127.0.0.1:5812`.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(600))
                .build()
                .map_err(|e| MyelinError::Store(format!("http client: {e}")))?,
            url: format!("{}/v1/rerank", base_url.into().trim_end_matches('/')),
            model: model.into(),
        })
    }
}

#[derive(Deserialize)]
struct RerankResponse {
    #[serde(default)]
    results: Vec<RerankResult>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl Reranker for CrossEncoder {
    fn id(&self) -> &str {
        &self.model
    }

    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let response = self
            .client
            .post(&self.url)
            .json(&json!({
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": documents.len(),
            }))
            .send()
            .await
            .map_err(|e| MyelinError::Store(format!("{}: rerank request: {e}", self.model)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| MyelinError::Store(format!("{}: reading body: {e}", self.model)))?;

        // Same ordering as the LLM client and for the same reason: an empty
        // 200 is a model-load failure, and parsing first misdiagnoses it as
        // malformed JSON (R7).
        if body.trim().is_empty() {
            return Err(MyelinError::EmptyCompletion {
                model: self.model.clone(),
            });
        }
        if !status.is_success() {
            return Err(MyelinError::Store(format!(
                "{}: HTTP {status}: {}",
                self.model,
                body.chars().take(400).collect::<String>()
            )));
        }

        let parsed: RerankResponse = serde_json::from_str(&body).map_err(|e| {
            MyelinError::Store(format!(
                "{}: unparseable rerank response: {e}; body was {:?}",
                self.model,
                body.chars().take(400).collect::<String>()
            ))
        })?;

        if parsed.results.len() != documents.len() {
            return Err(MyelinError::Store(format!(
                "{}: reranked {} of {} documents",
                self.model,
                parsed.results.len(),
                documents.len()
            )));
        }

        // The endpoint returns results sorted by score, not by input order.
        // Restoring input order is the trait's contract: the caller owns
        // ordering and its deterministic tie-break.
        let mut scores = vec![f32::NEG_INFINITY; documents.len()];
        for r in parsed.results {
            let slot = scores.get_mut(r.index).ok_or_else(|| {
                MyelinError::Store(format!(
                    "{}: rerank index {} out of range for {} documents",
                    self.model,
                    r.index,
                    documents.len()
                ))
            })?;
            *slot = r.relevance_score;
        }
        Ok(scores)
    }
}
