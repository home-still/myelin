//! M50c: events beside turns, not instead
//! (`docs/measurements/m50c-events-beside-turns.md`).
//!
//! M50 and M50b put dated events into the same store as the conversation
//! turns, so at k = 6 the two competed for the same slots. On LoCoMo the
//! events displaced a gold turn on 74 rows and restored one on 17, and the
//! arm netted −0.13 even though events rescued +11.6 on the reader's declines.
//!
//! Chronos (Sen et al., `10.48550/arXiv.2603.16862` §3.1) never lets them
//! compete. Events live in their own index and turns in another, "enabling
//! independent retrieval over each representation". This is that
//! arrangement. The base retrieval runs exactly as before, and afterwards the
//! top `m` events for the same question, drawn from a separate events-only
//! index and reranked like the turns, are appended under their own token
//! budget. The base items are never reordered, rewritten or dropped. A run
//! with the block and one without differ only by the appended block, so the
//! arm's delta is the events.

use crate::error::Result;
use crate::model::evidence::{EvidenceItem, EvidenceKind, EvidenceSet};
use crate::model::query::{Budget, Mode, Recall};
use crate::model::record::SourceRef;
use crate::pipeline::compose::{weakest_trust, ComposeConfig};
use crate::pipeline::ingest::approx_tokens;
use crate::pipeline::retrieve::{RetrieveConfig, Retriever};

/// Events appended per question (the M50c design's recommendation).
pub const DEFAULT_EVENTS_M: usize = 3;
/// The block's own token budget, separate from the turns' budget.
pub const DEFAULT_EVENTS_BUDGET_TOKENS: usize = 512;
/// The source a header item carries, so the reader of an evidence set can
/// tell the block's framing from retrieved text.
pub const EVENTS_MECHANISM: &str = "m50c:events";
/// Opens the block, so the reader sees events as a separate, dated list.
pub const EVENTS_HEADER: &str = "[events] Dated events from the event calendar:";

/// How the events index composes: as the turns do, but without a second
/// `[timeline]` view and without re-resolving relative dates.
///
/// - The base evidence already carries a `[timeline]`.
/// - Every event is written with its date already resolved, both in its text
///   (`[2023-05-20 — "yesterday", said 2023-05-21]`) and as its valid time.
///   Resolving the quoted phrase a second time anchors it on the *resolved*
///   date and appends a contradicting one: `(yesterday = 2023-05-19)` beside
///   an event that happened on 2023-05-20. That defect reached the reader in
///   every M50, M50b and M50c run (found 2026-09-26 in
///   `runs/m50_pilot_s1`), so the events index composes with
///   [`ComposeConfig::resolve_relative`] off.
pub fn events_retrieve_config() -> RetrieveConfig {
    RetrieveConfig {
        compose: ComposeConfig {
            timeline: false,
            resolve_relative: false,
            ..ComposeConfig::default()
        },
        ..RetrieveConfig::default()
    }
}

/// The events index and how much of it one question may use.
pub struct EventsBlock<'a> {
    pub retriever: &'a Retriever<'a>,
    pub m: usize,
    pub budget_tokens: usize,
}

impl<'a> EventsBlock<'a> {
    pub fn new(retriever: &'a Retriever<'a>) -> Self {
        Self {
            retriever,
            m: DEFAULT_EVENTS_M,
            budget_tokens: DEFAULT_EVENTS_BUDGET_TOKENS,
        }
    }

    /// Retrieve the top events for `query`'s question in its scope and time,
    /// and append them after everything already in `set`, headed by
    /// [`EVENTS_HEADER`]. Returns how many events were appended. None found
    /// means nothing is appended, not even the header.
    pub async fn append(&self, query: &Recall, set: &mut EvidenceSet) -> Result<usize> {
        let events_query = Recall {
            budget: Budget {
                k: self.m,
                tokens: self.budget_tokens,
                max_steps: query.budget.max_steps,
            },
            mode: Mode::Recall,
            kinds: None,
            ..query.clone()
        };
        let (events, _) = self.retriever.recall(&events_query).await?;
        if events.items.is_empty() {
            return Ok(0);
        }
        // A view over the events it frames: nil record, the mechanism as its
        // source, and the weakest trust among them (`compose::weakest_trust`).
        let header = EvidenceItem {
            kind: EvidenceKind::Text,
            value: EVENTS_HEADER.to_string(),
            record_id: uuid::Uuid::nil(),
            source: SourceRef::doc(EVENTS_MECHANISM),
            score: 0.0,
            trust: weakest_trust(events.items.iter().map(|i| i.trust)),
        };
        let n = events.items.len();
        set.tokens += approx_tokens(&header.value) + events.tokens;
        set.items.push(header);
        set.items.extend(events.items);
        Ok(n)
    }
}
