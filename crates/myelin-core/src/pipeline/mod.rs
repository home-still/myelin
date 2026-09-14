//! Write and read pipelines (`PLAN.md` §6, §7).
//!
//! `ingest → extract → consolidate → index` on the write side;
//! `scope-filter → retrieve → fuse → rerank → (graph-expand) → compose` on the
//! read side. Fusion lands first because it is pure arithmetic with a measured
//! reference value, so it can be settled before any model exists.

pub mod compose;
pub mod consolidate;
pub mod extract;
pub mod fuse;
pub mod index;
pub mod ingest;
pub mod investigate;
pub mod retrieve;
pub mod write;
