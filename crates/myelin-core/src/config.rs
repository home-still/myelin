//! Configuration, layered exactly as the house pattern does
//! (`home-still/crates/hs-distill/src/config.rs`): serialized defaults, then
//! `~/.myelin/config.yml`, then `MYELIN_`-prefixed environment variables.
//!
//! `split("__")` makes nesting addressable from the environment, so the Qdrant
//! endpoint is `MYELIN_QDRANT__URL`.

use figment::{
    providers::{Env, Format, Serialized, Yaml},
    Figment,
};
use serde::{Deserialize, Serialize};

use crate::error::{MyelinError, Result};

/// Path of the config file, relative to the user's home directory.
pub const CONFIG_REL_PATH: &str = ".myelin/config.yml";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MyelinConfig {
    pub qdrant: QdrantConfig,
    pub llm: LlmConfig,
    pub embed: EmbedConfig,
    pub rerank: RerankConfig,
}

/// The reader. Served directly by `llama-server`, not through llama-swap:
/// llama-swap runs strict swap (one model resident) and the write path needs
/// the reader and the embedder at the same time.
///
/// The default is localhost because `big` firewalls these ports; the
/// workstation reaches them through an SSH tunnel. See `ops/big/README.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// OpenAI-compatible `/v1` root.
    pub url: String,
    pub model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:5810/v1".into(),
            model: "qwen3.5-9b".into(),
        }
    }
}

/// The cross-encoder. Separate `llama-server` for the same reason the reader
/// is: llama-swap is strict-swap, and the read path needs the embedder and
/// the reranker live at once.
///
/// `bge-reranker-v2-m3` Q8_0 is 606 MiB — small enough to co-reside with the
/// 6 GiB reader on a card that also hosts a live voice assistant, which is
/// the whole reason a 568M-parameter reranker was chosen over a 7B one. It
/// buys the largest single gain in the pipeline: MS MARCO MRR@10 goes
/// 18.7 → 36.5 with a cross-encoder over BM25 (`PLAN.md` §2 finding 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankConfig {
    /// Server root, *not* the `/v1` path: llama.cpp exposes `/v1/rerank`.
    pub url: String,
    pub model: String,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:5813".into(),
            model: "bge-reranker-v2-m3".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedConfig {
    pub url: String,
    pub model: String,
    /// Output width after Matryoshka truncation. The served model emits 4096;
    /// 1024 is the `dense` channel `PLAN.md` §5.1 specifies, and it is 4x less
    /// storage per record. Changing this changes the embedder id and so
    /// invalidates a built memory (R3), which is the point.
    pub dim: u64,
}

impl Default for EmbedConfig {
    /// bge-m3 through ollama, not the Qwen3-Embedding-8B that M2 stood up.
    ///
    /// Three measured reasons, in order of weight:
    ///
    /// 1. `PLAN.md` §5.1 specifies the `dense` channel as **bge-m3, 1024-d**.
    ///    bge-m3 is 1024-d natively, so the Matryoshka truncation in
    ///    [`crate::embed::remote`] is not needed at all here.
    /// 2. VRAM. The 8B embedder costs 8,858 MiB. `big` is shared with a live
    ///    voice assistant (~2.6 GB) and with ollama, which its pipeline wakes
    ///    on demand (~5.6 GB); holding the 8B alongside both left 890 MiB
    ///    free, against an operator request to leave ~5 GB. bge-m3 is 1.2 GB
    ///    and is already resident in the process that keeps waking anyway.
    /// 3. ollama's port is reachable from the workstation; the direct
    ///    `llama-server` ports are firewalled and need an SSH tunnel.
    ///
    /// The dense channel is the *second* lever, not the first: MemPro's
    /// ablation costs 12.68 points for removing BM25 and 2.36 for removing
    /// the embedder. Whether Qwen3-Embedding-8B earns its 7.7 GB is an M4
    /// ablation question, and `ops/big/serve-models.sh MYELIN_EMBEDDER=qwen`
    /// stands it back up to answer it.
    fn default() -> Self {
        Self {
            url: "http://192.168.1.110:11434/v1".into(),
            model: "bge-m3".into(),
            dim: 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QdrantConfig {
    /// gRPC endpoint. Port 6334, never 6333 — see [`crate::store::qdrant`].
    pub url: String,
    pub collection: String,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://192.168.1.110:6334".into(),
            collection: "myelin_memory".into(),
        }
    }
}

impl MyelinConfig {
    pub fn load() -> Result<Self> {
        let config_path = dirs::home_dir()
            .unwrap_or_default()
            .join(CONFIG_REL_PATH);

        Figment::from(Serialized::defaults(Self::default()))
            .merge(Yaml::file(&config_path))
            .merge(Env::prefixed("MYELIN_").split("__"))
            .extract()
            .map_err(|e| MyelinError::Config(e.to_string()))
    }
}
