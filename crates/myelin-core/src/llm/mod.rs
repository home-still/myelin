//! LLM access (`PLAN.md` §3.1, §6.2).
//!
//! # R7 — an empty completion is a failure, never a null answer
//!
//! llama-swap answers `HTTP 200` with a **zero-byte body** when the upstream
//! child fails to spawn — typically a `cudaMalloc` OOM because another tenant
//! holds the card. That masking hid a ~21 h stall on `big`
//! (`docs/research/00-verified-environment.md` §7.1–7.2), and it was observed
//! live again during this build: `qwen3.8-27b` sat in `state: "starting"` with
//! 8 GiB free against a ~22.7 GiB requirement.
//!
//! So the check is structural, not advisory. Implementors write
//! [`Llm::raw_complete`]; [`Llm::complete`] is the provided method that
//! validates, and every caller goes through it. There is no path that
//! propagates an empty completion as a valid answer.
//!
//! Note the one legitimate empty case: a tool-call response carries
//! `"content": ""` and populated `tool_calls`. That is a real answer and is
//! allowed through.

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::{MyelinError, Result};

pub mod openai;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// A tool the model may call. Shape matches the OpenAI function-calling wire
/// format, which llama.cpp's server implements under `--jinja`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpec>,
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// JSON Schema to constrain output, for `extract` (§6.2) and
    /// `consolidate` (§6.3), where a free-form answer is unusable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
    /// Let the model emit a reasoning trace.
    ///
    /// **Default off, and that is load-bearing.** Qwen3.5-9B through
    /// llama.cpp spends its entire completion budget on `reasoning_content`
    /// for structured tasks: measured, a two-line episode burned all 512
    /// tokens and returned `content: ""`, versus **18 tokens** with thinking
    /// off. At one call per episode over a corpus that is the difference
    /// between a feasible ingest and an infeasible one. `investigate` (§7.2)
    /// turns it back on, where deliberation is the point.
    ///
    /// Note `reasoning_effort` does **not** work on this model — only
    /// `enable_thinking` does. Verified against the live server.
    #[serde(default)]
    pub thinking: bool,
    /// Nucleus and top-k sampling, and a seed, for the one arm that samples
    /// (M44 R2). `None` leaves the server's default in place, so every
    /// temperature-0 call is byte-identical to before these fields existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl CompletionRequest {
    /// Temperature 0 by default: every number this system reports has to be
    /// reproducible, and the run manifest records the temperature it used.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            temperature: 0.0,
            max_tokens: None,
            json_schema: None,
            thinking: false,
            top_p: None,
            top_k: None,
            seed: None,
        }
    }

    pub fn with_thinking(mut self, thinking: bool) -> Self {
        self.thinking = thinking;
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_schema(mut self, schema: serde_json::Value) -> Self {
        self.json_schema = Some(schema);
        self
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Sample instead of decoding greedily. The one caller is M44 R2's
    /// thinking reader, at the Qwen3 Technical Report's thinking-mode
    /// setting (`10.48550/arxiv.2505.09388`: temperature 0.6, top-p 0.95,
    /// top-k 20). A sampled run must also carry a [`Self::with_seed`] or its
    /// number cannot be reproduced.
    pub fn with_sampling(mut self, temperature: f32, top_p: f32, top_k: u32) -> Self {
        self.temperature = temperature;
        self.top_p = Some(top_p);
        self.top_k = Some(top_k);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Completion {
    pub text: String,
    /// The model's thinking trace, when the server returned one
    /// (`reasoning_content` under llama.cpp's `--reasoning-format`). Never
    /// part of the answer: callers that grade or parse `text` must not see
    /// it, and callers that diagnose a wrong answer must (M44 R2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub usage: Usage,
}

impl Completion {
    /// A completion is empty when it carries neither text nor a tool call.
    /// A tool call with empty content is a real answer.
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.tool_calls.is_empty()
    }
}

#[async_trait]
pub trait Llm: Send + Sync {
    /// Stable model identifier, recorded in the run manifest.
    fn id(&self) -> &str;

    /// Implementors provide the transport. Do not call this directly.
    async fn raw_complete(&self, req: &CompletionRequest) -> Result<Completion>;

    /// **R7 enforcement point.** Every caller uses this.
    async fn complete(&self, req: &CompletionRequest) -> Result<Completion> {
        let completion = self.raw_complete(req).await?;
        if completion.is_empty() {
            // Two different failures that look identical from here. Telling
            // them apart is the difference between debugging VRAM and
            // debugging a token budget.
            if completion.finish_reason.as_deref() == Some("length") {
                return Err(MyelinError::BudgetExhausted {
                    model: self.id().to_string(),
                    max_tokens: req.max_tokens,
                });
            }
            return Err(MyelinError::EmptyCompletion {
                model: self.id().to_string(),
            });
        }
        Ok(completion)
    }
}

/// Structured output. Kept a free function rather than a trait method so
/// [`Llm`] stays object-safe.
///
/// Models fence JSON in markdown often enough that stripping it is not
/// leniency, it is the common case; anything else is a parse error the caller
/// must see.
pub async fn complete_json<T: DeserializeOwned>(
    llm: &dyn Llm,
    req: &CompletionRequest,
) -> Result<T> {
    let completion = llm.complete(req).await?;
    let raw = strip_json_fence(&completion.text);
    serde_json::from_str(raw).map_err(|e| {
        MyelinError::Store(format!(
            "{}: structured output did not parse as the requested schema: {e}; body was {:?}",
            llm.id(),
            truncate(raw, 300)
        ))
    })
}

fn strip_json_fence(s: &str) -> &str {
    let t = s.trim();
    let Some(rest) = t.strip_prefix("```") else {
        return t;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    rest.trim_start_matches('\n')
        .strip_suffix("```")
        .unwrap_or(rest)
        .trim()
}

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Canned(Completion);

    #[async_trait]
    impl Llm for Canned {
        fn id(&self) -> &str {
            "canned"
        }
        async fn raw_complete(&self, _req: &CompletionRequest) -> Result<Completion> {
            Ok(self.0.clone())
        }
    }

    fn completion(text: &str, tool_calls: Vec<ToolCall>) -> Completion {
        Completion {
            reasoning: None,
            text: text.into(),
            tool_calls,
            finish_reason: None,
            usage: Usage::default(),
        }
    }

    fn req() -> CompletionRequest {
        CompletionRequest::new(vec![Message::user("hi")])
    }

    /// R7: the llama-swap failure mode — a 200 that carries nothing — must
    /// surface as an error at the trait boundary.
    #[tokio::test]
    async fn empty_completion_is_an_error() {
        for body in ["", "   ", "\n\t "] {
            let llm = Canned(completion(body, vec![]));
            let err = llm
                .complete(&req())
                .await
                .expect_err("R7: empty completion must not be propagated");
            assert!(
                matches!(err, MyelinError::EmptyCompletion { .. }),
                "wrong error for {body:?}: {err}"
            );
        }
    }

    /// Budget exhaustion is a different diagnosis from a dead model, and the
    /// error has to say which: one sends you to `nvidia-smi`, the other to
    /// the token budget.
    #[tokio::test]
    async fn a_truncated_reasoning_trace_is_not_reported_as_a_dead_model() {
        let llm = Canned(Completion {
            reasoning: None,
            text: String::new(),
            tool_calls: vec![],
            finish_reason: Some("length".into()),
            usage: Usage::default(),
        });
        let err = llm
            .complete(&req().with_max_tokens(512))
            .await
            .expect_err("empty content is still an error");
        match err {
            MyelinError::BudgetExhausted { max_tokens, .. } => {
                assert_eq!(max_tokens, Some(512))
            }
            other => panic!("expected BudgetExhausted, got {other}"),
        }
    }

    /// Thinking is off by default: on for structured extraction it consumes
    /// the entire budget (measured 512 tokens vs 18 on the live model).
    #[test]
    fn thinking_is_off_by_default() {
        assert!(!CompletionRequest::new(vec![]).thinking);
        assert!(CompletionRequest::new(vec![]).with_thinking(true).thinking);
    }

    /// Sampling is opt-in and explicit: a default request is greedy with no
    /// sampling keys at all, so the wire body of every temperature-0 call is
    /// unchanged by the fields M44 R2 added.
    #[test]
    fn sampling_is_unset_by_default_and_set_together_by_the_builder() {
        let plain = CompletionRequest::new(vec![]);
        assert_eq!((plain.temperature, plain.top_p, plain.top_k, plain.seed), (0.0, None, None, None));
        let sampled = CompletionRequest::new(vec![]).with_sampling(0.6, 0.95, 20).with_seed(7);
        assert_eq!(
            (sampled.temperature, sampled.top_p, sampled.top_k, sampled.seed),
            (0.6, Some(0.95), Some(20), Some(7))
        );
    }

    /// A tool call legitimately carries empty content; rejecting it would
    /// break every agentic path.
    #[tokio::test]
    async fn tool_call_with_empty_content_is_a_valid_answer() {
        let llm = Canned(completion(
            "",
            vec![ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: "{}".into(),
            }],
        ));
        let got = llm.complete(&req()).await.expect("tool call is an answer");
        assert_eq!(got.tool_calls.len(), 1);
    }

    #[tokio::test]
    async fn structured_output_tolerates_a_markdown_fence() {
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Fact {
            subject: String,
            value: i32,
        }
        for body in [
            r#"{"subject":"a","value":1}"#,
            "```json\n{\"subject\":\"a\",\"value\":1}\n```",
            "```\n{\"subject\":\"a\",\"value\":1}\n```",
        ] {
            let llm = Canned(completion(body, vec![]));
            let got: Fact = complete_json(&llm, &req()).await.expect(body);
            assert_eq!(
                got,
                Fact {
                    subject: "a".into(),
                    value: 1
                }
            );
        }
    }

    #[tokio::test]
    async fn structured_output_reports_the_body_it_could_not_parse() {
        let llm = Canned(completion("not json at all", vec![]));
        let err = complete_json::<serde_json::Value>(&llm, &req())
            .await
            .expect_err("must not silently succeed");
        assert!(err.to_string().contains("not json at all"), "got: {err}");
    }
}
