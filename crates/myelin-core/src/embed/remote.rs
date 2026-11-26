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
    /// Largest input sent to the embedder in one piece, in characters. See
    /// [`DEFAULT_MAX_EMBED_CHARS`].
    max_chars: usize,
    /// Stable id written into `MemoryConfigJson`. Must change when
    /// `target_dim` changes — a different dimension is a different memory
    /// (R3, `PLAN.md` §4.1). It also names a `max_chars` other than the
    /// default, because that changes the vector of every longer record.
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

        let id = embedder_id(&model, target_dim, DEFAULT_MAX_EMBED_CHARS);

        Ok(Self {
            client,
            base_url,
            model,
            target_dim,
            max_chars: DEFAULT_MAX_EMBED_CHARS,
            id,
        })
    }

    /// Send inputs of up to `max_chars` characters to the embedder whole.
    ///
    /// Raise it above [`DEFAULT_MAX_EMBED_CHARS`] only for an embedder that
    /// takes that many characters in one input: a record under the cap is
    /// embedded as itself, a record over it as the weighted mean of its
    /// pieces, which is an approximation. Zero is refused: no text fits it.
    pub fn with_max_chars(mut self, max_chars: usize) -> Result<Self> {
        if max_chars == 0 {
            return Err(MyelinError::Config(
                "embed.max_chars is 0: no input could be sent to the embedder".into(),
            ));
        }
        self.max_chars = max_chars;
        self.id = embedder_id(&self.model, self.target_dim, max_chars);
        Ok(self)
    }
}

/// `model@dim`, with `+c<max_chars>` when the cap is not the default.
fn embedder_id(model: &str, target_dim: u64, max_chars: usize) -> String {
    if max_chars == DEFAULT_MAX_EMBED_CHARS {
        format!("{model}@{target_dim}")
    } else {
        format!("{model}@{target_dim}+c{max_chars}")
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

        // Long inputs are split, embedded separately, and mean-pooled rather
        // than sent whole — see `DEFAULT_MAX_EMBED_CHARS`.
        if texts.iter().any(|t| t.chars().count() > self.max_chars) {
            return self.embed_chunked(texts).await;
        }
        self.embed_batch(texts).await
    }
}

impl RemoteEmbedder {
    /// Embed inputs of any length by splitting the over-long ones.
    ///
    /// One vector out per text in, so callers never learn that a split
    /// happened: a record is still one record.
    async fn embed_chunked(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Flatten every text into its chunks, embed all chunks in one
        // request, then pool back. A per-text request would turn one
        // oversized turn into N round trips.
        let mut flat: Vec<String> = Vec::with_capacity(texts.len());
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(texts.len());
        for text in texts {
            let start = flat.len();
            for chunk in split_for_embedding(text, self.max_chars) {
                flat.push(chunk);
            }
            spans.push((start, flat.len()));
        }
        let weights: Vec<usize> = flat.iter().map(|piece| piece.chars().count()).collect();

        let vectors = self.embed_batch(&flat).await?;

        let mut out = Vec::with_capacity(texts.len());
        for (start, end) in spans {
            out.push(mean_pool(
                &vectors[start..end],
                &weights[start..end],
                self.target_dim as usize,
            ));
        }
        Ok(out)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let body = json!({
            "model": self.model,
            "input": texts,
        });

        let (status, body_text) = crate::net::send_retrying(
            || self.client.post(&url).json(&body),
            &self.model,
            &format!("request to {url}"),
        )
        .await?;

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

/// Largest text sent to the embedder in one piece, in characters, unless
/// [`RemoteEmbedder::with_max_chars`] says otherwise.
///
/// bge-m3's context is 8192 tokens and ollama returns HTTP 400 — not a
/// truncation — once an input exceeds it. Our `approx_tokens` estimate of
/// chars/4 is badly wrong on the text that actually triggers this: a
/// LongMemEval_S turn of 12,240 characters, newline-separated short phrases,
/// tokenizes to more than 8192, i.e. under 1.5 chars per token rather than 4.
/// Prose is nearer 4; lists, code and markup are nearer 1.5.
///
/// 4,000 characters is therefore ~2,700 tokens even at that worst observed
/// density, leaving room for a tokenizer denser still. Chosen for the failure
/// mode rather than the average, because the average never fails.
///
/// Measured later (ollama 0.40.3, 2026-11-26): the limit a stock ollama
/// `bge-m3` enforces is its 2,048-token batch, not the 8,192 context. A
/// piece over 2,048 tokens is either embedded from its first 2,048 tokens
/// with HTTP 200 (pieces of 2,364, 4,502 and 8,002 tokens were) or refused
/// with HTTP 413 (a piece of 2,078 tokens was); length did not say which.
/// An ollama model created with `num_batch 8192` embedded every one of
/// them whole, and that is the embedder a larger cap is for.
pub const DEFAULT_MAX_EMBED_CHARS: usize = 4000;

/// Split on line boundaries where possible, hard-cutting only when a single
/// line is itself over the cap.
///
/// Cutting mid-sentence produces a chunk that embeds to neither neighbour's
/// meaning — the same reasoning as `lmev2::chunk_tree` splitting on element
/// boundaries rather than every N bytes.
fn split_for_embedding(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in text.split_inclusive('\n') {
        if !cur.is_empty() && cur.chars().count() + line.chars().count() > max_chars {
            chunks.push(std::mem::take(&mut cur));
        }
        if line.chars().count() > max_chars {
            let mut buf = String::new();
            for c in line.chars() {
                buf.push(c);
                if buf.chars().count() >= max_chars {
                    chunks.push(std::mem::take(&mut buf));
                }
            }
            cur.push_str(&buf);
        } else {
            cur.push_str(line);
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks
}

/// Mean of unit vectors, each weighted by the length of the piece it
/// embeds, re-normalized.
///
/// The weights are what make the result stand for the text. The split is
/// greedy, so its last piece is whatever was left over, and an equal mean
/// let that remainder count as much as everything before it: a 4,045-character
/// record became half a 91-character tail. In a store of 476,000 such records
/// 19 % were split, the smaller piece had a median of 260 characters, and
/// those vectors scored 0.05-0.17 higher against unrelated queries than the
/// same text embedded whole (two unit vectors that are not parallel sum to
/// less than 2, so normalizing the sum raises every cosine). They filled 17
/// and 23 of the top 25 dense hits of two queries. Weighting by length is
/// the usual form of this pooling: OpenAI Cookbook, "Embedding texts that are
/// longer than the model's maximum context length"
/// (`np.average(chunk_embeddings, axis=0, weights=chunk_lens)`, then
/// normalize).
///
/// Re-normalization is not optional: the Qdrant `Cosine` comparator assumes
/// unit vectors and the mean of several unit vectors is not one. Same reason
/// `fit_dim` re-normalizes after Matryoshka truncation.
fn mean_pool(vectors: &[Vec<f32>], weights: &[usize], dim: usize) -> Vec<f32> {
    if vectors.len() == 1 {
        return vectors[0].clone();
    }
    let mut acc = vec![0.0f32; dim];
    for (v, weight) in vectors.iter().zip(weights) {
        let weight = *weight as f32;
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += *x * weight;
        }
    }
    let norm = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for a in acc.iter_mut() {
            *a /= norm;
        }
    }
    acc
}


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
    use super::{
        fit_dim, mean_pool, split_for_embedding, RemoteEmbedder, DEFAULT_MAX_EMBED_CHARS,
    };

    const MAX_EMBED_CHARS: usize = DEFAULT_MAX_EMBED_CHARS;

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

    /// The cap exists to stop HTTP 400s, so no chunk may exceed it — not even
    /// when one line is longer than the cap by itself.
    #[test]
    fn no_chunk_exceeds_the_cap() {
        let unbroken = "x".repeat(MAX_EMBED_CHARS * 3 + 7);
        let lines = "short line\n".repeat(2000);
        let mixed = format!("{lines}{unbroken}\n{lines}");
        for text in [&unbroken, &lines, &mixed] {
            for chunk in split_for_embedding(text, MAX_EMBED_CHARS) {
                assert!(
                    chunk.chars().count() <= MAX_EMBED_CHARS,
                    "chunk of {} chars exceeds cap {MAX_EMBED_CHARS}",
                    chunk.chars().count()
                );
            }
        }
    }

    /// Splitting must not lose or duplicate a single character. A silent drop
    /// here is unrecoverable: the record keeps its full text, and only the
    /// vector would be wrong, so nothing downstream could ever notice.
    #[test]
    fn splitting_preserves_every_character() {
        let unbroken = "y".repeat(MAX_EMBED_CHARS * 2 + 13);
        let text = format!("alpha\nbeta\n{unbroken}\ngamma\n{}", "z".repeat(9000));
        assert_eq!(split_for_embedding(&text, MAX_EMBED_CHARS).concat(), text);
    }

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(
            split_for_embedding("hello\nworld", MAX_EMBED_CHARS),
            vec!["hello\nworld"]
        );
    }

    /// Cosine in Qdrant assumes unit vectors; the mean of unit vectors is not
    /// one, so pooling has to renormalize or every score is quietly wrong.
    #[test]
    fn mean_pool_returns_a_unit_vector() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let pooled = mean_pool(&[a, b], &[10, 10], 4);
        let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm was {norm}");
        // Two pieces of one length: equidistant between the two inputs.
        assert!((pooled[0] - pooled[1]).abs() < 1e-6);
    }

    /// A single chunk must come back byte-identical, not round-tripped
    /// through the pooling arithmetic, or every ordinary record's vector
    /// would drift.
    #[test]
    fn mean_pool_of_one_is_the_identity() {
        let v = vec![0.6f32, 0.8, 0.0, 0.0];
        assert_eq!(mean_pool(std::slice::from_ref(&v), &[7], 4), v);
    }

    /// The split is greedy, so its last piece is a remainder of any size.
    /// The piece sizes here are a real record's: 3,954 characters and a
    /// 91-character tail. The pooled vector has to stay the long piece's,
    /// not land halfway to the tail's.
    #[test]
    fn a_short_remainder_does_not_count_as_much_as_the_text_before_it() {
        const LONG_PIECE_CHARS: usize = 3954;
        const TAIL_CHARS: usize = 91;
        let long_piece = vec![1.0f32, 0.0, 0.0, 0.0];
        let tail = vec![0.0f32, 1.0, 0.0, 0.0];
        let pooled = mean_pool(&[long_piece, tail], &[LONG_PIECE_CHARS, TAIL_CHARS], 4);
        let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm was {norm}");
        // Cosine with the long piece. An equal mean gives 0.707.
        assert!(pooled[0] > 0.999, "cosine with the long piece was {}", pooled[0]);
        let expected_ratio = TAIL_CHARS as f32 / LONG_PIECE_CHARS as f32;
        assert!(
            (pooled[1] / pooled[0] - expected_ratio).abs() < 1e-5,
            "tail to long-piece ratio was {}",
            pooled[1] / pooled[0]
        );
    }

    /// A record a little over the default cap is two pieces at the default
    /// and itself under a cap that holds it, so an embedder that can take it
    /// whole is given it whole.
    #[test]
    fn a_cap_that_holds_the_text_leaves_it_in_one_piece() {
        const RAISED_CAP_CHARS: usize = 8000;
        let line = "a line of a record, forty-one chars long\n";
        let text = line.repeat(MAX_EMBED_CHARS / line.len() + 2);
        assert!(text.chars().count() > MAX_EMBED_CHARS);
        assert!(text.chars().count() < RAISED_CAP_CHARS);

        assert_eq!(split_for_embedding(&text, MAX_EMBED_CHARS).len(), 2);
        assert_eq!(split_for_embedding(&text, RAISED_CAP_CHARS), vec![text]);
    }

    /// The cap is part of what a vector is, so a non-default one shows in
    /// the embedder id, and a cap nothing fits under is refused.
    #[test]
    fn a_raised_cap_is_named_in_the_id_and_zero_is_refused() {
        const RAISED_CAP_CHARS: usize = 8000;
        let new = || RemoteEmbedder::new("http://127.0.0.1:1/v1", "bge-m3", 1024);
        let stock = new().expect("client");
        assert_eq!(super::Embedder::id(&stock), "bge-m3@1024");

        let same = new().and_then(|e| e.with_max_chars(MAX_EMBED_CHARS)).expect("default cap");
        assert_eq!(super::Embedder::id(&same), "bge-m3@1024");

        let raised = new().and_then(|e| e.with_max_chars(RAISED_CAP_CHARS)).expect("raised cap");
        assert_eq!(super::Embedder::id(&raised), "bge-m3@1024+c8000");

        let Err(refused) = new().and_then(|e| e.with_max_chars(0)) else {
            panic!("a cap of zero was accepted");
        };
        assert!(refused.to_string().contains("max_chars"), "{refused}");
    }
}
