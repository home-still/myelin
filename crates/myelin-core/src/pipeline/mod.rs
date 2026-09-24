//! Write and read pipelines (`PLAN.md` §6, §7).
//!
//! `ingest → extract → consolidate → index` on the write side;
//! `scope-filter → retrieve → graph-expand → fuse → rerank → compose` on the
//! read side. Fusion lands first because it is pure arithmetic with a measured
//! reference value, so it can be settled before any model exists.
//!
//! `graph-expand` is a third ranked list, not a stage after fusion: PPR over
//! the phrase↔record incidence graph joins `dense` and `lex` in one `rrf`
//! call ([`retrieve::RetrieveConfig::graph`]).

pub mod adjudicate;
pub mod compose;
pub mod consolidate;
pub mod decompose;
pub mod events;
pub mod events_block;
pub mod extract;
pub mod fuse;
pub mod index;
pub mod ingest;
pub mod investigate;
pub mod phrases;
pub mod retrieve;
pub mod select;
pub mod trajectory_agent;
pub mod trajectory_tools;
pub mod write;
