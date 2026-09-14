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
