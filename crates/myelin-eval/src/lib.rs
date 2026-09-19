//! `myelin-eval` — the evaluation harness (`PLAN.md` §3.3).
//!
//! "A binary plus a library of scorers and runners." The library half exists
//! so the same code path measures `myelin` and every baseline
//! (`FullContext`, `DenseOnly`, `Bm25Only`, `HybridNoRerank`) — without that,
//! every number the project reports is self-graded.
//!
//! It is also why the loaders live here rather than in the binary: a dataset
//! reader that only the CLI can reach cannot be unit-tested against a fixture
//! or reused by a runner.

pub mod ablate;
pub mod adjudicate_probe;
pub mod attack;
pub mod attack_live;
pub mod bench;
pub mod build;
pub mod datasets;
pub mod evidence_audit;
pub mod judge;
pub mod manifest;
pub mod phrases;
pub mod standing;
pub mod temporal;
