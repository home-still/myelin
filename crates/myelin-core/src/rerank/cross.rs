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

/// Longest document sent to the cross-encoder, in characters.
///
/// The server reports its limit in tokens (8192 physical batch) and the query
/// plus special tokens share that budget. At the worst density measured on
/// this corpus — about 1.5 characters per token on line-broken text, see
/// `MAX_EMBED_CHARS` — 8,000 characters is roughly 5,300 tokens, leaving the
/// query ample room.
const MAX_RERANK_CHARS: usize = 8000;

/// Cap each document at [`MAX_RERANK_CHARS`], leaving shorter ones untouched.
fn truncate_for_rerank(documents: &[String]) -> Vec<String> {
    documents
        .iter()
        .map(|d| {
            if d.chars().count() <= MAX_RERANK_CHARS {
                d.clone()
            } else {
                d.chars().take(MAX_RERANK_CHARS).collect()
            }
        })
        .collect()
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
        // Truncate for scoring only. The cross-encoder has a fixed physical
        // batch and a memory system has no maximum record length, so a long
        // record must cost relevance precision rather than kill the query --
        // measured: a LongMemEval_S record produced "input (9771 tokens) is
        // too large to process (current batch size: 8192)" and failed the
        // whole run. Unlike the embedder, which mean-pools chunks because a
        // stored vector must represent the whole record, this genuinely can
        // discard the tail: the score decides ordering, and the reader still
        // receives the record's full text either way.
        let truncated = truncate_for_rerank(documents);
        let documents = truncated.as_slice();

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

#[cfg(test)]
mod tests {
    use super::{truncate_for_rerank, MAX_RERANK_CHARS};

    /// The cap exists to stop HTTP 500s, so nothing may exceed it — and
    /// counting must be in characters, not bytes, or one multi-byte document
    /// slips through and fails the whole query.
    #[test]
    fn long_documents_are_capped_and_short_ones_are_untouched() {
        let short = "already fine".to_string();
        let long = "x".repeat(MAX_RERANK_CHARS * 2);
        let multibyte = "é".repeat(MAX_RERANK_CHARS * 2);
        let out = truncate_for_rerank(&[short.clone(), long, multibyte]);
        assert_eq!(out[0], short, "short documents must not be rewritten");
        assert_eq!(out[1].chars().count(), MAX_RERANK_CHARS);
        assert_eq!(out[2].chars().count(), MAX_RERANK_CHARS);
    }

    /// One score per input document is the contract the caller relies on to
    /// zip scores back onto candidates; truncation must not drop or add rows.
    #[test]
    fn truncation_preserves_document_count_and_order() {
        let docs: Vec<String> = (0..5)
            .map(|i| format!("{i}").repeat(MAX_RERANK_CHARS * 2))
            .collect();
        let out = truncate_for_rerank(&docs);
        assert_eq!(out.len(), docs.len());
        for (i, d) in out.iter().enumerate() {
            assert!(d.starts_with(&i.to_string()), "order changed at {i}");
        }
    }
}
