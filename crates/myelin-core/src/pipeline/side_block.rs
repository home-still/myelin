//! A side block: records of one kind, ranked against the question and
//! appended after the evidence, never competing with it.
//!
//! Two record kinds are served this way, each behind its own question gate:
//!
//! - **Events** (M73b, `docs/measurements/m73b-dated-events.md`). When the
//!   question names a past day ("what did I buy **10 days ago**?",
//!   [`crate::time::question_window`]), the events dated inside that window.
//!   M73 measured it without a reader: in 7 of 10 targeted temporal losses
//!   the top-ranked in-window event states the answer, and the top-ranked
//!   turn does in 0 of 8. The event calendar is Chronos's (Sen et al.,
//!   `10.48550/arXiv.2603.16862` §3.1), which keeps events in their own index
//!   "enabling independent retrieval over each representation".
//! - **Profile records** (M20b, `docs/measurements/m20b-ranked-profile.md`).
//!   When the question asks for advice
//!   ([`crate::pipeline::query_shape::is_advice_request`]), the user's stated
//!   dispositions. M20 chose them by recency and reached the gold
//!   preference's source turn in 1 of 30 questions; ranked by the question,
//!   the top 8 reach it in 18 of 30 (M20b stage 0). Profile-guided ranking
//!   is PPRO's load-bearing component (Jiang et al. 2026, arXiv 2607.00017),
//!   and PrefEval finds that retrieving the stated preference is the best
//!   remedy for preference-unaware answers (Zhao et al. 2025, arXiv
//!   2502.09597).
//!
//! Selection reads every record of the kind in the question's scope from the
//! ledger and ranks them all with the cross-encoder. A tenant holds tens to a
//! few hundred such records, so no vector prefilter is needed, and the
//! ranking is the one stage 0 measured. The base evidence is never
//! reordered, rewritten or dropped: a run with the block differs from one
//! without only by what is appended.

use chrono::Utc;

use crate::error::{MyelinError, Result};
use crate::model::evidence::{EvidenceItem, EvidenceKind, EvidenceSet};
use crate::model::query::Recall;
use crate::model::record::{MemoryRecord, RecordKind, SourceRef};
use crate::pipeline::compose::{compose, weakest_trust, ComposeConfig, Ranked};
use crate::pipeline::ingest::approx_tokens;
use crate::pipeline::query_shape::is_advice_request;
use crate::rerank::Reranker;
use crate::store::ledger::Ledger;

/// Events appended per question (M50c's design).
pub const EVENTS_M: usize = 3;
/// Profile records appended per question: M20's block size, so M20b changes
/// only how the eight are chosen.
pub const PROFILE_M: usize = 8;
/// Each block's own token budget, separate from the evidence's.
pub const SIDE_BUDGET_TOKENS: usize = 512;
/// The most records of one kind a scope may hold. Above it the block
/// refuses rather than rank a truncated set.
pub const SIDE_MAX_RECORDS: usize = 2000;

/// Which records a side block serves, and behind which question gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideKind {
    /// Dated events (stored as `semantic` records by `events-build`), gated
    /// on the question naming a past day.
    Events,
    /// Dispositions (`profile` records), gated on an advice request.
    Profile,
}

impl SideKind {
    fn record_kind(self) -> RecordKind {
        match self {
            SideKind::Events => RecordKind::Semantic,
            SideKind::Profile => RecordKind::Profile,
        }
    }

    /// Opens the block, so the reader sees it as a separate list.
    pub fn header(self) -> &'static str {
        match self {
            SideKind::Events => {
                "[events] Dated events from the event calendar, on the day the question names:"
            }
            SideKind::Profile => {
                "[profile] What the user has said about their own preferences and situation:"
            }
        }
    }

    /// The source a header item carries.
    pub fn mechanism(self) -> &'static str {
        match self {
            SideKind::Events => "m73b:events",
            SideKind::Profile => "m20b:profile",
        }
    }

    fn m(self) -> usize {
        match self {
            SideKind::Events => EVENTS_M,
            SideKind::Profile => PROFILE_M,
        }
    }
}

/// A side ledger, its reranker, and one kind of record.
pub struct SideBlock<'a> {
    pub kind: SideKind,
    pub ledger: &'a Ledger,
    pub reranker: &'a dyn Reranker,
}

/// How a side block composes: stamped with each record's date, but without a
/// `[timeline]` (the evidence carries one) and without re-resolving relative
/// phrases. An event's date is resolved once, when it is written; resolving
/// its quoted phrase again appended a contradicting date to every M50 event.
pub fn side_compose_config(m: usize) -> ComposeConfig {
    ComposeConfig {
        k: m,
        max_tokens: SIDE_BUDGET_TOKENS,
        timeline: false,
        resolve_relative: false,
        ..ComposeConfig::default()
    }
}

impl<'a> SideBlock<'a> {
    pub fn new(kind: SideKind, ledger: &'a Ledger, reranker: &'a dyn Reranker) -> Self {
        Self {
            kind,
            ledger,
            reranker,
        }
    }

    /// The records this question may see, or `None` when its gate is shut.
    async fn candidates(&self, query: &Recall) -> Result<Option<Vec<MemoryRecord>>> {
        let window = match self.kind {
            SideKind::Events => {
                let Some(asked_on) = query.as_of else {
                    return Ok(None);
                };
                let Some(window) = crate::time::question_window(&query.text, asked_on) else {
                    return Ok(None);
                };
                Some(window)
            }
            SideKind::Profile => {
                if !is_advice_request(&query.text) {
                    return Ok(None);
                }
                None
            }
        };
        let all = self
            .ledger
            .visible_of_kind(
                &query.scope,
                self.kind.record_kind(),
                Utc::now(),
                SIDE_MAX_RECORDS as i64,
            )
            .await?;
        if all.len() >= SIDE_MAX_RECORDS {
            return Err(MyelinError::Store(format!(
                "{} holds {SIDE_MAX_RECORDS} or more {:?} records in scope {:?}; the side block ranks all of them and refuses a truncated set",
                self.kind.mechanism(),
                self.kind.record_kind(),
                query.scope.tenant
            )));
        }
        Ok(Some(match window {
            None => all,
            Some(w) => all
                .into_iter()
                .filter(|r| {
                    let d = r.validity.t_valid.date_naive();
                    w.lo <= d && d <= w.hi
                })
                .collect(),
        }))
    }

    /// Append the question's top records of this kind after everything in
    /// `set`, headed by [`SideKind::header`]. Returns how many were appended.
    /// A shut gate, or no record passing it, appends nothing, not even the
    /// header, so the row is byte-identical to a run without the block.
    pub async fn append(&self, query: &Recall, set: &mut EvidenceSet) -> Result<usize> {
        let Some(records) = self.candidates(query).await? else {
            return Ok(0);
        };
        if records.is_empty() {
            return Ok(0);
        }
        let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
        let scores = self.reranker.rerank(&query.text, &texts).await?;
        if scores.len() != records.len() {
            return Err(MyelinError::Store(format!(
                "reranker returned {} scores for {} records",
                scores.len(),
                records.len()
            )));
        }
        let mut ranked: Vec<Ranked> = records
            .into_iter()
            .zip(scores)
            .map(|(record, score)| Ranked {
                record,
                score,
                vector: None,
                window: None,
            })
            .collect();
        // Best first; ties by id so a rerun composes the same block.
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.record.id.cmp(&b.record.id))
        });
        let m = self.kind.m();
        ranked.truncate(m);
        let block = compose(ranked, &side_compose_config(m));
        if block.items.is_empty() {
            return Ok(0);
        }
        // A view over the records it frames: nil record, the mechanism as
        // its source, and the weakest trust among them.
        let header = EvidenceItem {
            kind: EvidenceKind::Text,
            value: self.kind.header().to_string(),
            record_id: uuid::Uuid::nil(),
            source: SourceRef::doc(self.kind.mechanism()),
            score: 0.0,
            trust: weakest_trust(block.items.iter().map(|i| i.trust)),
        };
        let n = block.items.len();
        set.tokens += approx_tokens(&header.value) + block.tokens;
        set.items.push(header);
        set.items.extend(block.items);
        Ok(n)
    }
}
