//! Benchmark dataset loaders and checksum-pinned fetch (`PLAN.md` §3.3).
//!
//! Datasets are pinned by SHA-256 because a silently-corrupted corpus is the
//! worst failure mode: every downstream number becomes meaningless and the
//! discrepancy is invisible until someone re-derives the digest by hand.

pub mod locomo;
pub mod lmev2;
pub mod longmemeval;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use sha2::{Digest, Sha256};

/// A dataset file pinned to an exact digest and byte count.
///
/// The `sha256` and `bytes` fields are compile-time constants so that a
/// checksum drift is a source-level change, not a silent runtime regression.
pub struct PinnedFile {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
}

/// The 10-conversation LoCoMo release.  Measured: 2 805 274 bytes, digest
/// `79fa87e9…68698ff4`.
pub const LOCOMO: PinnedFile = PinnedFile {
    name: "locomo10.json",
    url: "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    sha256: "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    bytes: 2_805_274,
};

/// Compute the SHA-256 of a file, returned as a lowercase hex string.
pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    Ok(hex::encode(&digest))
}

/// Download `file` into `dest_dir` unless a copy with a matching digest is
/// already present.
///
/// Returns the path to the verified file.  On digest mismatch the error names
/// both the expected and the actual digest so the failure is self-diagnosing
/// (`PLAN.md` §1.1 — reproducibility requires a pinned corpus).
pub async fn fetch_pinned(file: &PinnedFile, dest_dir: &Path) -> anyhow::Result<PathBuf> {
    let dest = dest_dir.join(file.name);

    // Fast path: the file already exists and its digest matches.
    if dest.exists() {
        let actual = sha256_file(&dest)?;
        if actual == file.sha256 {
            return Ok(dest);
        }
        // Digest mismatch — fall through to re-download.
    }

    std::fs::create_dir_all(dest_dir)
        .with_context(|| format!("creating {}", dest_dir.display()))?;

    let resp = reqwest::get(file.url)
        .await
        .context("requesting dataset")?
        .error_for_status()
        .context("dataset HTTP status")?;

    let bytes = resp.bytes().await.context("reading dataset body")?;
    std::fs::write(&dest, &bytes).with_context(|| format!("writing {}", dest.display()))?;

    let actual = {
        let digest = Sha256::digest(&bytes);
        hex::encode(&digest)
    };
    if actual != file.sha256 {
        bail!(
            "checksum mismatch for {}: expected {}, got {}",
            file.name,
            file.sha256,
            actual
        );
    }
    Ok(dest)
}

// ---------------------------------------------------------------------------
// hex encoding — kept local to avoid pulling a `hex` crate dependency for one
// function.
// ---------------------------------------------------------------------------

mod hex {
    fn encode_byte(b: u8) -> [u8; 2] {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        [DIGITS[(b >> 4) as usize], DIGITS[(b & 0xf) as usize]]
    }

    pub fn encode(data: &[u8]) -> String {
        let mut out = String::with_capacity(data.len() * 2);
        for &b in data {
            let [hi, lo] = encode_byte(b);
            out.push(hi as char);
            out.push(lo as char);
        }
        out
    }
}