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

        // llama.cpp routes this into the Jinja chat template. `enable_thinking`
        // is the key Qwen3.5 honours; `reasoning_effort` is silently ignored
        // by it, which is why the knob is spelled this way.
        body["chat_template_kwargs"] = json!({ "enable_thinking": req.thinking });

        if let Some(n) = req.max_tokens {
            body["max_tokens"] = json!(n);
        }
        // Sampling keys ride only when set (M44 R2), so a greedy request's
        // body carries none of them.
        if let Some(p) = req.top_p {
            body["top_p"] = json!(p);
        }
        if let Some(k) = req.top_k {
            body["top_k"] = json!(k);
        }
        if let Some(s) = req.seed {
            body["seed"] = json!(s);
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
        let (status, body) = crate::net::send_retrying(
            || self.client.post(&url).json(&self.body(req)),
            &self.model,
            &format!("request to {url}"),
        )
        .await?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Message;

    fn client() -> OpenAiLlm {
        OpenAiLlm::new("http://127.0.0.1:1", "m").expect("client")
    }

    /// The wire body of a default request carries no sampling key and pins
    /// thinking off; a sampled, seeded, thinking request carries all of
    /// them. Every myelin number before M44 R2 was produced by the first
    /// shape, and the second must not leak into it.
    #[test]
    fn sampling_and_thinking_reach_the_wire_only_when_asked_for() {
        let plain = client().body(&CompletionRequest::new(vec![Message::user("q")]));
        assert_eq!(plain["temperature"], 0.0);
        assert_eq!(plain["chat_template_kwargs"]["enable_thinking"], false);
        for key in ["top_p", "top_k", "seed", "max_tokens"] {
            assert!(plain.get(key).is_none(), "{key} must be absent by default");
        }

        let sampled = client().body(
            &CompletionRequest::new(vec![Message::user("q")])
                .with_thinking(true)
                .with_sampling(0.6, 0.95, 20)
                .with_seed(7)
                .with_max_tokens(1184),
        );
        assert_eq!(sampled["chat_template_kwargs"]["enable_thinking"], true);
        // f32 on the request, f64 in the JSON tree: compare with tolerance.
        let close = |v: &serde_json::Value, want: f64| (v.as_f64().unwrap() - want).abs() < 1e-6;
        assert!(close(&sampled["temperature"], 0.6), "{}", sampled["temperature"]);
        assert!(close(&sampled["top_p"], 0.95), "{}", sampled["top_p"]);
        assert_eq!(sampled["top_k"], 20);
        assert_eq!(sampled["seed"], 7);
        assert_eq!(sampled["max_tokens"], 1184);
    }
}
