//! The run manifest (`EVALUATION.md` §9).
//!
//! **This is the reproducibility contract, and the report renderer refuses
//! rows without one.** A number with no manifest cannot be compared to
//! anything, including a later run of the same code: it does not say which
//! commit produced it, which corpus it read, which models answered, or what
//! the operating point was.
//!
//! Two properties are load-bearing and both are enforced here rather than by
//! convention:
//!
//! 1. **`commit` is the working-tree state, not just `HEAD`.** A run from a
//!    dirty tree records `<sha>-dirty`, because `HEAD` alone would claim
//!    reproducibility the artifact cannot deliver.
//! 2. **Serialization is deterministic.** Field order is the struct's, maps
//!    are `BTreeMap`, so two manifests for identical runs are byte-identical
//!    and `diff` is a usable tool for "what changed between these runs".

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// `<sha>` or `<sha>-dirty`.
    pub commit: String,
    pub started_at: DateTime<Utc>,
    pub duration_s: f64,
    pub benchmark: Benchmark,
    pub backend: Backend,
    pub models: Models,
    pub seeds: Vec<u64>,
    pub repeats: usize,
    pub hardware: Hardware,
    pub results: Results,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agentic: Option<Agentic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Benchmark {
    pub name: String,
    pub tier: String,
    pub domain: String,
    /// SHA-256 of the corpus file actually read. Pinning the *name* of a
    /// dataset is not pinning the dataset.
    pub data_sha256: String,
    pub n_questions: usize,
    pub subset: String,
    /// LME-V2 scores 295 of 451 questions deterministically and routes 156
    /// to a judge. Reporting the mix is what lets a reader discount the
    /// judged half if they distrust it.
    pub scorer_mix: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Backend {
    pub name: String,
    pub version: String,
    pub operating_point: String,
    pub memory_config: serde_json::Value,
    pub runtime_overrides: BTreeMap<String, serde_json::Value>,
    pub store: Store,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Store {
    pub qdrant_version: String,
    pub collection: String,
    pub points: usize,
    pub graph_edges: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Models {
    pub reader: String,
    pub controller: String,
    pub temperature: f32,
    pub embedder: String,
    pub judge: String,
    pub judge_mode: String,
    /// Hash of the judge prompt, not the prompt. A judge whose prompt
    /// changed is a different judge, and this is how a reader notices.
    pub judge_prompt_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hardware {
    pub host: String,
    pub gpu: String,
    pub gpu_tenant: String,
    pub vram_peak_mib: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Results {
    pub overall_full_set: f64,
    pub static_accuracy: f64,
    pub dynamic_accuracy: f64,
    pub procedure_accuracy: f64,
    pub gotchas_accuracy: f64,
    pub abstention_precision: f64,
    pub abstention_recall: f64,
    /// The 295 deterministically-scored questions, reported separately
    /// because G1 requires a column that no judge touched.
    pub judge_free_subset_accuracy: f64,
    pub memory_query_avg_seconds: f64,
    pub latency: Latency,
    pub tokens: Tokens,
    pub ci95: BTreeMap<String, [f64; 2]>,
    pub lafs: Lafs,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Latency {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub split: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Tokens {
    pub ingest: u64,
    pub per_query_prompt: u64,
    pub per_query_completion: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Lafs {
    pub tier: String,
    pub gain: f64,
    pub absolute: f64,
    pub operating_points: Vec<OperatingPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatingPoint {
    pub name: String,
    pub acc: f64,
    pub latency: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Agentic {
    pub steps_to_answer: BTreeMap<String, f64>,
    pub tool_selection_error_rate: f64,
    pub wasted_retrieval_fraction: f64,
    pub conflict_gate_fire_rate: f64,
    pub trace_path: String,
}

/// `<sha>` from `HEAD`, suffixed `-dirty` when the tree has uncommitted
/// changes.
///
/// The suffix is the point. A leaderboard submission built from a dirty tree
/// is not reproducible from its commit, and recording the bare sha would
/// claim otherwise.
pub fn commit_id() -> Result<String> {
    let sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .context("git rev-parse")?;
    anyhow::ensure!(sha.status.success(), "git rev-parse HEAD failed");
    let sha = String::from_utf8(sha.stdout)?.trim().to_string();

    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .context("git status")?;
    let dirty = !String::from_utf8_lossy(&status.stdout).trim().is_empty();
    Ok(if dirty { format!("{sha}-dirty") } else { sha })
}

impl Manifest {
    /// Write to `dir/manifest.json`, pretty-printed and newline-terminated.
    ///
    /// Pretty-printed because these are read by humans in review and diffed
    /// between runs; a single-line JSON blob makes both useless.
    pub fn write(&self, dir: &Path) -> Result<std::path::PathBuf> {
        std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
        let path = dir.join("manifest.json");
        let mut json = serde_json::to_string_pretty(self)?;
        json.push('\n');
        std::fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
        Ok(path)
    }

    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
    }

    /// Refuse a manifest that cannot support the claim it is attached to.
    ///
    /// Called by the report renderer before a row is emitted. Every check
    /// here corresponds to a way a number has actually been made
    /// incomparable in published work: an unpinned corpus, a dirty tree, a
    /// missing model name, an operating point with no latency.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.commit.is_empty(), "manifest has no commit");
        anyhow::ensure!(
            !self.commit.ends_with("-dirty"),
            "manifest {} was produced from a dirty working tree; the run is not \
             reproducible from its commit",
            self.commit
        );
        anyhow::ensure!(
            self.benchmark.data_sha256.len() == 64,
            "benchmark.data_sha256 must be a full SHA-256; pinning a dataset's name is \
             not pinning the dataset"
        );
        anyhow::ensure!(self.benchmark.n_questions > 0, "benchmark.n_questions is zero");
        anyhow::ensure!(!self.models.reader.is_empty(), "models.reader is empty");
        anyhow::ensure!(!self.models.embedder.is_empty(), "models.embedder is empty");
        anyhow::ensure!(
            !self.backend.operating_point.is_empty(),
            "backend.operating_point is empty; LAFS is defined over named points"
        );
        anyhow::ensure!(
            self.duration_s > 0.0,
            "duration_s is zero; a run that took no time did not happen"
        );
        for op in &self.results.lafs.operating_points {
            anyhow::ensure!(
                op.latency > 0.0,
                "operating point {:?} has zero latency; LAFS is an accuracy-LATENCY \
                 frontier and a point with no cost would dominate it for free",
                op.name
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> Manifest {
        Manifest {
            commit: "a".repeat(40),
            started_at: Utc::now(),
            duration_s: 12.5,
            benchmark: Benchmark {
                name: "lme_v2".into(),
                tier: "small".into(),
                domain: "web".into(),
                data_sha256: "b".repeat(64),
                n_questions: 240,
                subset: "full".into(),
                scorer_mix: BTreeMap::from([
                    ("deterministic".to_string(), 295),
                    ("llm".to_string(), 156),
                ]),
            },
            backend: Backend {
                name: "myelin".into(),
                version: "0.1.0".into(),
                operating_point: "fast".into(),
                memory_config: serde_json::json!({"memory_type": "myelin"}),
                runtime_overrides: BTreeMap::from([("k".to_string(), serde_json::json!(6))]),
                store: Store {
                    qdrant_version: "1.19.1".into(),
                    collection: "myelin_lme_v2_small".into(),
                    points: 1,
                    graph_edges: 0,
                },
            },
            models: Models {
                reader: "qwen3.5-9b".into(),
                controller: "qwen3.5-9b".into(),
                temperature: 0.6,
                embedder: "bge-m3".into(),
                judge: "gpt-5.2".into(),
                judge_mode: "leaderboard".into(),
                judge_prompt_sha256: "c".repeat(64),
            },
            seeds: vec![1],
            repeats: 1,
            hardware: Hardware {
                host: "big".into(),
                gpu: "RTX 3090 24576MiB".into(),
                gpu_tenant: "coding".into(),
                vram_peak_mib: 7042,
            },
            results: Results {
                lafs: Lafs {
                    tier: "small".into(),
                    operating_points: vec![OperatingPoint {
                        name: "fast".into(),
                        acc: 55.0,
                        latency: 0.9,
                    }],
                    ..Default::default()
                },
                ..Default::default()
            },
            agentic: None,
        }
    }

    #[test]
    fn a_dirty_tree_is_refused() {
        let mut m = valid();
        m.commit = format!("{}-dirty", "a".repeat(40));
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("dirty"), "got: {err}");
    }

    #[test]
    fn an_unpinned_corpus_is_refused() {
        let mut m = valid();
        m.benchmark.data_sha256 = "locomo10.json".into();
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("SHA-256"), "got: {err}");
    }

    #[test]
    fn a_free_operating_point_is_refused() {
        // LAFS is an accuracy-latency frontier. A point reported at zero
        // latency dominates every real point for nothing, which is how a
        // frontier metric gets gamed by accident.
        let mut m = valid();
        m.results.lafs.operating_points[0].latency = 0.0;
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("latency"), "got: {err}");
    }

    #[test]
    fn round_trips_byte_identically() {
        let m = valid();
        let dir = std::env::temp_dir().join(format!("myelin-manifest-{}", uuid::Uuid::new_v4()));
        let path = m.write(&dir).unwrap();
        let first = std::fs::read_to_string(&path).unwrap();
        let reloaded = Manifest::read(&path).unwrap();
        reloaded.write(&dir).unwrap();
        let second = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            first, second,
            "manifests must be byte-stable so `diff` answers 'what changed between \
             these two runs'"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
