//! A named store: one tenant, addressed by a stable name, with the rules of
//! its use.
//!
//! The tenant stays the isolation boundary (C12). A store adds what a tenant
//! string cannot say: whether an agent reads it without asking, whether an
//! agent may write to it, and whether its records carry real event dates.

use serde::{Deserialize, Serialize};

use crate::error::{MyelinError, Result};

/// Longest store name, in bytes. With [`is_store_name`]: `^[a-z][a-z0-9_]{0,31}$`.
pub const MAX_STORE_NAME_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct StoreDef {
    /// `^[a-z][a-z0-9_]{0,31}$`.
    pub name: String,
    /// The tenant every record of this store is written under. One store per
    /// tenant and one tenant per store.
    pub tenant: String,
    /// What the store holds, for the agent choosing where to look.
    pub description: String,
    /// Read when a caller names neither a tenant nor a store.
    pub default_read: bool,
    /// Whether the agent surface may `remember` into it and soft-`forget`
    /// from it.
    pub agent_writes: bool,
    /// Whether its records carry real event timestamps. `false` suppresses
    /// the date mechanisms of a read that does not say otherwise.
    pub dated: bool,
}

/// `^[a-z][a-z0-9_]{0,31}$`
pub fn is_store_name(name: &str) -> bool {
    let mut chars = name.chars();
    name.len() <= MAX_STORE_NAME_LEN
        && matches!(chars.next(), Some(c) if c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

impl StoreDef {
    /// Refuse a definition the registry must never hold.
    pub fn validate(&self) -> Result<()> {
        if !is_store_name(&self.name) {
            return Err(MyelinError::StoreDefinition(format!(
                "store name {:?} must match ^[a-z][a-z0-9_]{{0,31}}$",
                self.name
            )));
        }
        if self.tenant.trim().is_empty() {
            return Err(MyelinError::StoreDefinition(format!(
                "store {:?} needs a tenant",
                self.name
            )));
        }
        if self.description.trim().is_empty() {
            return Err(MyelinError::StoreDefinition(format!(
                "store {:?} needs a description: it is what an agent chooses a store by",
                self.name
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::is_store_name;

    #[test]
    fn a_store_name_is_a_lowercase_identifier_of_at_most_32_bytes() {
        for ok in ["a", "academic", "user_memory", "w3b", &"a".repeat(32)] {
            assert!(is_store_name(ok), "{ok:?}");
        }
        for bad in [
            "",
            "Academic",
            "1st",
            "_x",
            "a-b",
            "a b",
            "a.b",
            "é",
            &"a".repeat(33),
        ] {
            assert!(!is_store_name(bad), "{bad:?}");
        }
    }
}
