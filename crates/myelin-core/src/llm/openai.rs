//! OpenAI-compatible client, for llama-swap (`:8081/v1`) and ollama
//! (`:11434/v1`) on `big`.
//!
//! The transport-level half of R7 lives here: llama-swap's failure mode is a
//! `200` whose **body is zero bytes**, so the body is read as text and checked
//! for emptiness *before* JSON parsing. Deserializing first would surface
//! "EOF while parsing a value" — a parse bug, which is the wrong diagnosis and
//! exactly what cost ~21 h of silent stall the first time.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::error::{MyelinError, Result};

use super::{Completion, CompletionRequest, Llm, Role, ToolCall, Usage};

pub struct OpenAiLlm {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OpenAiLlm {
    /// `base_url` is the `/v1` root, e.g. `http://192.168.1.110:8081/v1`.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Ok(Self {
            client: reqwest::Client::builder()
                // A cold GGUF spawn off local NVMe takes ~15 s and a 27B off
                // NFS took ~4.5 min; the default 30 s would time out a cold
                // start and look like a model failure.
                .timeout(std::time::Duration::from_secs(600))
                .build()
                .map_err(|e| MyelinError::Store(format!("http client: {e}")))?,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
        })
    }

    fn body(&self, req: &CompletionRequest) -> serde_json::Value {
        let messages: Vec<_> = req
            .messages
            .iter()
            .map(|m| {
                json!({
                    "role": match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                    },
                    "content": m.content,
                })
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": req.temperature,
            "stream": false,
        });

        if let Some(n) = req.max_tokens {
            body["max_tokens"] = json!(n);
        }
        if !req.tools.is_empty() {
            body["tools"] = json!(req
                .tools
                .iter()
                .map(|t| json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                }))
                .collect::<Vec<_>>());
        }
        if let Some(schema) = &req.json_schema {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": { "name": "myelin", "strict": true, "schema": schema },
            });
        }
        body
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    message: ChoiceMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct ChoiceMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ApiToolCall>,
}

#[derive(Deserialize)]
struct ApiToolCall {
    #[serde(default)]
    id: String,
    function: ApiFunction,
}

#[derive(Deserialize)]
struct ApiFunction {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize, Default)]
struct ApiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

#[async_trait]
impl Llm for OpenAiLlm {
    fn id(&self) -> &str {
        &self.model
    }

    async fn raw_complete(&self, req: &CompletionRequest) -> Result<Completion> {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&self.body(req))
            .send()
            .await
            .map_err(|e| MyelinError::Store(format!("{}: request to {url} failed: {e}", self.model)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| MyelinError::Store(format!("{}: reading body: {e}", self.model)))?;

        // R7, transport half. This ordering is deliberate: an empty 200 is a
        // model-load failure, not malformed JSON.
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

        let parsed: ChatResponse = serde_json::from_str(&body).map_err(|e| {
            MyelinError::Store(format!(
                "{}: unparseable chat response: {e}; body was {:?}",
                self.model,
                body.chars().take(400).collect::<String>()
            ))
        })?;

        let choice = parsed.choices.into_iter().next().ok_or_else(|| {
            // Also R7 territory: a well-formed envelope with no choices is
            // still no answer.
            MyelinError::EmptyCompletion {
                model: self.model.clone(),
            }
        })?;

        Ok(Completion {
            text: choice.message.content.unwrap_or_default(),
            tool_calls: choice
                .message
                .tool_calls
                .into_iter()
                .map(|t| ToolCall {
                    id: t.id,
                    name: t.function.name,
                    arguments: t.function.arguments,
                })
                .collect(),
            finish_reason: choice.finish_reason,
            usage: parsed
                .usage
                .map(|u| Usage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                })
                .unwrap_or_default(),
        })
    }
}
