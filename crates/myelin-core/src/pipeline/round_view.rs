//! M86: the round view (`docs/measurements/m86-round-view.md`).
//!
//! Ingest cuts a session into episodes of at most 512 approximate tokens
//! ([`crate::pipeline::ingest::SegmentConfig`]). A LongMemEval_S assistant
//! reply is long (median ~599 tokens), so the cap usually falls between a
//! short user turn and the reply that answers it. The user's words end one
//! episode, and the reply opens the next record. Measured on the shipped
//! round-5 replicates (2026-09-30):
//! - 487 of 500 rows' evidence holds at least one record that opens with an
//!   assistant turn;
//! - 20 of 43 preference losses have the reply without the user turn it
//!   answers;
//! - preference scores 44% in that state against 72% with the user turn
//!   held, and temporal scores 68% against 93%.
//!
//! So when an episode chosen for the reader opens with an assistant turn,
//! this view shows the user turn it answers, taken from the preceding
//! episode of the same session, as its own item directly before it. The
//! evidence the read path chose is never reordered, rewritten or dropped.
//! The added turn is beside the budget, not inside it, so nothing is
//! displaced (M50b's lesson: events that took turns' slots cost more than
//! they gave).
//!
//! The grounds:
//! - LongMemEval (Wu et al. 2024, arXiv 2410.10813): the user-assistant
//!   round is the best value granularity for memory retrieval.
//! - JustMem (Chen et al. 2026, arXiv 2609.19877): replaying the source
//!   session for fidelity-sensitive evidence took single-session-assistant
//!   from 32.14 to 94.64.
//! - MemLoc (arXiv 2609.07093): keep the full memory context and add cues.
//!   Showing only an extract of the evidence cost 4.5 to 9.0 points.
//!
//! Session ids are read only to find the predecessor. They never reach the
//! reader: the added item renders the turn's text under its date, as every
//! other item does. That matters on LongMemEval_S, where evidence sessions
//! are named `answer_*`.

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{MyelinError, Result};
use crate::model::evidence::{EvidenceItem, EvidenceSet};
use crate::model::query::Recall;
use crate::model::record::{MemoryRecord, RecordKind};
use crate::pipeline::compose::{compose, ComposeConfig, Ranked};
use crate::pipeline::ingest::approx_tokens;
use crate::pipeline::turn_windows::turn_spans;
use crate::pipeline::user_words::{SEGMENT_SEPARATOR, SPEAKERS};
use crate::store::ledger::Ledger;

/// The longest prompting turn shown, in approximate tokens. A longer one is
/// left out and counted, never cut. Measured on LongMemEval_S's haystacks:
/// the user turn before an assistant reply is p50 46, p95 114 and p99 318
/// tokens; 256 admits 98.7% of them and every gold user turn outside
/// single-session-assistant (the longest is 246).
pub const ROUND_PROMPT_MAX_TOKENS: usize = 256;
/// The most episodes a scope may hold. Above it the view refuses rather than
/// look for predecessors in a truncated set. A LongMemEval_S tenant holds
/// about 330.
pub const ROUND_VIEW_MAX_EPISODES: usize = 2000;
/// Names the mechanism in errors and in the run's stats.
pub const ROUND_VIEW_MECHANISM: &str = "m86:round-view";
/// An episode that opens with this turn answers a turn it does not hold.
const ASSISTANT_TURN_PREFIX: &str = "assistant: ";
/// The only turn the view shows.
const USER_TURN_PREFIX: &str = "user: ";

/// What the view did on one row, or summed over a run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundViewStats {
    /// Evidence episodes that open with an assistant turn.
    pub candidates: usize,
    /// Candidates shown with the user turn they answer.
    pub fired: usize,
    /// The candidate opens its session.
    pub no_predecessor: usize,
    /// The preceding episode does not end on the turn just before the
    /// candidate's first one (an empty turn between them, or a turn whose
    /// text holds a line that reads as a new turn).
    pub not_adjacent: usize,
    /// The preceding episode ends on an assistant turn.
    pub predecessor_not_user: usize,
    /// The preceding episode is already in the evidence.
    pub predecessor_in_evidence: usize,
    /// The prompting turn is over [`ROUND_PROMPT_MAX_TOKENS`].
    pub prompt_over_cap: usize,
    /// An evidence episode whose source doc is `<session>#<index>` in shape
    /// but not in content. Refused and counted.
    pub malformed_doc: usize,
    /// Tokens the added items cost, as compose counts them.
    pub added_tokens: usize,
}

impl RoundViewStats {
    pub fn add(&mut self, other: &RoundViewStats) {
        self.candidates += other.candidates;
        self.fired += other.fired;
        self.no_predecessor += other.no_predecessor;
        self.not_adjacent += other.not_adjacent;
        self.predecessor_not_user += other.predecessor_not_user;
        self.predecessor_in_evidence += other.predecessor_in_evidence;
        self.prompt_over_cap += other.prompt_over_cap;
        self.malformed_doc += other.malformed_doc;
        self.added_tokens += other.added_tokens;
    }
}

/// Where a source doc sits in its session.
#[derive(Debug, PartialEq, Eq)]
enum TurnDoc<'a> {
    /// No `#`: not a LongMemEval_S turn (LoCoMo's `D1:3`, LME-V2's
    /// `<trajectory>:events`, a mechanism's name).
    NotATurn,
    Turn { session: &'a str, index: usize },
    /// A `#` with no session before it or no turn index after it.
    Malformed,
}

fn turn_doc(doc: &str) -> TurnDoc<'_> {
    match doc.rsplit_once(SEGMENT_SEPARATOR) {
        None => TurnDoc::NotATurn,
        Some((session, index)) if !session.is_empty() => match index.parse::<usize>() {
            Ok(index) => TurnDoc::Turn { session, index },
            Err(_) => TurnDoc::Malformed,
        },
        Some(_) => TurnDoc::Malformed,
    }
}

/// The view over one scope's episodes.
pub struct RoundView<'a> {
    ledger: &'a Ledger,
    compose: ComposeConfig,
}

impl<'a> RoundView<'a> {
    /// `compose` is the run's own configuration, so the added item is dated
    /// and annotated exactly as its neighbours are.
    pub fn new(ledger: &'a Ledger, compose: ComposeConfig) -> Self {
        Self { ledger, compose }
    }

    /// Show, before each evidence episode that opens with an assistant turn,
    /// the user turn it answers. Evidence with no LongMemEval_S turn in it
    /// is returned untouched without a ledger read.
    pub async fn attach(&self, query: &Recall, set: &mut EvidenceSet) -> Result<RoundViewStats> {
        let any_turn = set
            .items
            .iter()
            .any(|it| !it.record_id.is_nil() && turn_doc(&it.source.doc) != TurnDoc::NotATurn);
        if !any_turn {
            return Ok(RoundViewStats::default());
        }
        let episodes = self
            .ledger
            .visible_of_kind(
                &query.scope,
                RecordKind::Episodic,
                Utc::now(),
                ROUND_VIEW_MAX_EPISODES as i64,
            )
            .await?;
        if episodes.len() >= ROUND_VIEW_MAX_EPISODES {
            return Err(MyelinError::Store(format!(
                "{ROUND_VIEW_MECHANISM}: scope {:?} holds {ROUND_VIEW_MAX_EPISODES} or more episodes; the view refuses to look for predecessors in a truncated set",
                query.scope.tenant
            )));
        }
        attach_rounds(set, &episodes, &self.compose)
    }
}

/// The view over `episodes`, the scope's visible episodes.
pub(crate) fn attach_rounds(
    set: &mut EvidenceSet,
    episodes: &[MemoryRecord],
    cfg: &ComposeConfig,
) -> Result<RoundViewStats> {
    let mut stats = RoundViewStats::default();
    let by_id: HashMap<Uuid, &MemoryRecord> = episodes.iter().map(|r| (r.id, r)).collect();
    // Each session's episodes by the index of their first turn.
    let mut sessions: HashMap<&str, Vec<(usize, &MemoryRecord)>> = HashMap::new();
    for record in episodes {
        match turn_doc(&record.provenance.source.doc) {
            TurnDoc::Turn { session, index } => sessions.entry(session).or_default().push((index, record)),
            TurnDoc::Malformed => tracing::warn!(
                record = %record.id,
                doc = %record.provenance.source.doc,
                "{ROUND_VIEW_MECHANISM}: episode source doc is not <session>#<index>; it is never taken as a predecessor"
            ),
            TurnDoc::NotATurn => {}
        }
    }
    for list in sessions.values_mut() {
        list.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.id.cmp(&b.1.id)));
    }
    let in_evidence: HashSet<Uuid> = set.items.iter().map(|it| it.record_id).collect();

    let mut inserts: Vec<(usize, EvidenceItem, usize)> = Vec::new();
    for (position, item) in set.items.iter().enumerate() {
        if item.record_id.is_nil() {
            continue;
        }
        // Only episodes: a fact or a profile record quoting a turn has no
        // round of its own.
        let Some(record) = by_id.get(&item.record_id) else {
            continue;
        };
        let (session, index) = match turn_doc(&item.source.doc) {
            TurnDoc::NotATurn => continue,
            TurnDoc::Malformed => {
                tracing::warn!(
                    record = %item.record_id,
                    doc = %item.source.doc,
                    "{ROUND_VIEW_MECHANISM}: evidence source doc is not <session>#<index>; refused"
                );
                stats.malformed_doc += 1;
                continue;
            }
            TurnDoc::Turn { session, index } => (session, index),
        };
        if !record.text.starts_with(ASSISTANT_TURN_PREFIX) {
            continue;
        }
        stats.candidates += 1;
        let Some(&(before, predecessor)) = sessions
            .get(session)
            .and_then(|list| list.iter().rev().find(|(i, _)| *i < index))
        else {
            stats.no_predecessor += 1;
            continue;
        };
        let spans = turn_spans(&predecessor.text, &SPEAKERS);
        let Some(last) = spans.last().cloned() else {
            stats.not_adjacent += 1;
            continue;
        };
        if before + spans.len() != index {
            stats.not_adjacent += 1;
            continue;
        }
        let prompt = &predecessor.text[last.clone()];
        if !prompt.starts_with(USER_TURN_PREFIX) {
            stats.predecessor_not_user += 1;
            continue;
        }
        if in_evidence.contains(&predecessor.id) {
            stats.predecessor_in_evidence += 1;
            continue;
        }
        if approx_tokens(prompt) > ROUND_PROMPT_MAX_TOKENS {
            stats.prompt_over_cap += 1;
            continue;
        }
        // The predecessor's last turn as a window, so it renders as every
        // windowed item does: dated, with an elision where turns were left
        // out, quoting the record it came from.
        let block = compose(
            vec![Ranked {
                record: (*predecessor).clone(),
                score: item.score,
                vector: None,
                window: Some(vec![last]),
            }],
            &ComposeConfig {
                k: 1,
                max_tokens: ROUND_PROMPT_MAX_TOKENS,
                timeline: false,
                ..cfg.clone()
            },
        );
        let tokens = block.tokens;
        let Some(added) = block.items.into_iter().next() else {
            return Err(MyelinError::Store(format!(
                "{ROUND_VIEW_MECHANISM}: compose admitted nothing for the turn before record {}",
                item.record_id
            )));
        };
        inserts.push((position, added, tokens));
    }
    // Last position first, so earlier positions stay valid.
    for (position, added, tokens) in inserts.into_iter().rev() {
        set.items.insert(position, added);
        set.tokens += tokens;
        stats.fired += 1;
        stats.added_tokens += tokens;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::evidence::EvidenceKind;
    use crate::model::record::*;

    fn episode(doc: &str, text: &str) -> MemoryRecord {
        MemoryRecord {
            id: Uuid::new_v4(),
            kind: RecordKind::Episodic,
            scope: Scope::new("t", "a", "ns"),
            text: text.into(),
            entities: vec![],
            validity: Validity {
                t_valid: Utc::now(),
                t_invalid: None,
                t_ingested: Utc::now(),
                t_expired: None,
            },
            provenance: Provenance {
                source: SourceRef::doc(doc),
                contributed_by: ActorId::new("u"),
                written_by: ActorId::new("w"),
                derived_from: vec![],
            },
            trust: Trust::asserted(),
            salience: Salience::default(),
            links: vec![],
        }
    }

    fn evidence_of(records: &[&MemoryRecord]) -> EvidenceSet {
        EvidenceSet {
            items: records
                .iter()
                .map(|r| EvidenceItem {
                    kind: EvidenceKind::Text,
                    value: r.text.clone(),
                    record_id: r.id,
                    source: r.provenance.source.clone(),
                    score: 1.0,
                    trust: r.trust.tier,
                })
                .collect(),
            tokens: 0,
            trace: vec![],
            dropped_for_tokens: 0,
            k_bound: false,
        }
    }

    fn cfg() -> ComposeConfig {
        ComposeConfig {
            stamp_valid_time: false,
            resolve_relative: false,
            ..ComposeConfig::default()
        }
    }

    /// A reply-only episode gets the user turn it answers, as its own item
    /// directly before it; the rest of the evidence is untouched.
    #[test]
    fn a_reply_is_shown_with_the_turn_it_answers() {
        let before = episode("s1#0", "user: hi\nassistant: hello\nuser: I just made a beef stew, any slow cooker ideas?");
        let reply = episode("s1#3", "assistant: Slow cooker recipes are the best! Try a chili.");
        let other = episode("s2#0", "user: unrelated\nassistant: ok");
        let mut set = evidence_of(&[&other, &reply]);
        let stats = attach_rounds(&mut set, &[before.clone(), reply.clone(), other.clone()], &cfg()).unwrap();
        assert_eq!(stats.candidates, 1);
        assert_eq!(stats.fired, 1);
        assert_eq!(set.items.len(), 3);
        assert_eq!(set.items[0].record_id, other.id);
        assert_eq!(set.items[1].record_id, before.id);
        assert!(set.items[1].value.contains("user: I just made a beef stew"));
        assert!(!set.items[1].value.contains("assistant: hello"));
        assert_eq!(set.items[2].record_id, reply.id);
        assert_eq!(set.tokens, stats.added_tokens);
        assert!(stats.added_tokens > 0);
    }

    #[test]
    fn a_predecessor_already_in_evidence_is_not_repeated() {
        let before = episode("s1#0", "user: I love hiking");
        let reply = episode("s1#1", "assistant: Hiking is great.");
        let mut set = evidence_of(&[&before, &reply]);
        let stats = attach_rounds(&mut set, &[before.clone(), reply.clone()], &cfg()).unwrap();
        assert_eq!((stats.candidates, stats.fired, stats.predecessor_in_evidence), (1, 0, 1));
        assert_eq!(set.items.len(), 2);
    }

    #[test]
    fn a_predecessor_ending_on_the_assistant_is_skipped() {
        let before = episode("s1#0", "user: hi\nassistant: hello");
        let reply = episode("s1#2", "assistant: and another thing");
        let mut set = evidence_of(&[&reply]);
        let stats = attach_rounds(&mut set, &[before, reply.clone()], &cfg()).unwrap();
        assert_eq!((stats.fired, stats.predecessor_not_user), (0, 1));
        assert_eq!(set.items.len(), 1);
    }

    #[test]
    fn a_long_prompting_turn_is_left_out_whole_not_cut() {
        let long = format!("user: {}", "word ".repeat(ROUND_PROMPT_MAX_TOKENS + 1));
        let before = episode("s1#0", &long);
        let reply = episode("s1#1", "assistant: sure");
        let mut set = evidence_of(&[&reply]);
        let stats = attach_rounds(&mut set, &[before, reply], &cfg()).unwrap();
        assert_eq!((stats.fired, stats.prompt_over_cap), (0, 1));
        assert_eq!(set.items.len(), 1);
    }

    /// An empty turn skipped at ingest leaves a gap in the turn indices; the
    /// view refuses rather than show a turn that may not be the prompt.
    #[test]
    fn a_gap_between_episodes_is_not_bridged() {
        let before = episode("s1#0", "user: hi");
        let reply = episode("s1#2", "assistant: hello");
        let mut set = evidence_of(&[&reply]);
        let stats = attach_rounds(&mut set, &[before, reply], &cfg()).unwrap();
        assert_eq!((stats.fired, stats.not_adjacent), (0, 1));
    }

    #[test]
    fn a_session_opening_reply_has_no_predecessor() {
        let reply = episode("s1#0", "assistant: welcome");
        let mut set = evidence_of(&[&reply]);
        let stats = attach_rounds(&mut set, &[reply], &cfg()).unwrap();
        assert_eq!((stats.candidates, stats.no_predecessor, stats.fired), (1, 1, 0));
    }

    /// LoCoMo's turns are dialogue ids and its speakers are people: nothing
    /// is a candidate and the evidence comes back byte-identical.
    #[test]
    fn locomo_shaped_evidence_is_untouched() {
        let a = episode("D1:3", "Caroline: I went to a support group\nMelanie: nice");
        let b = episode("D1:5", "Melanie: assistant: not a role here");
        let mut set = evidence_of(&[&a, &b]);
        let before = serde_json::to_string(&set.items).unwrap();
        let stats = attach_rounds(&mut set, &[a, b], &cfg()).unwrap();
        assert_eq!(stats, RoundViewStats::default());
        assert_eq!(serde_json::to_string(&set.items).unwrap(), before);
    }

    #[test]
    fn a_malformed_turn_doc_is_refused_and_counted() {
        let bad = episode("s1#x", "assistant: hello");
        let also_bad = episode("#3", "assistant: hello");
        let mut set = evidence_of(&[&bad, &also_bad]);
        let stats = attach_rounds(&mut set, &[bad, also_bad], &cfg()).unwrap();
        assert_eq!((stats.malformed_doc, stats.fired), (2, 0));
        assert_eq!(set.items.len(), 2);
    }

    #[test]
    fn turn_docs_parse_by_their_last_separator() {
        assert_eq!(turn_doc("answer_4be1b6b4_2#7"), TurnDoc::Turn { session: "answer_4be1b6b4_2", index: 7 });
        assert_eq!(turn_doc("a#b#12"), TurnDoc::Turn { session: "a#b", index: 12 });
        assert_eq!(turn_doc("D1:3"), TurnDoc::NotATurn);
        assert_eq!(turn_doc("s1#"), TurnDoc::Malformed);
        assert_eq!(turn_doc("#4"), TurnDoc::Malformed);
    }

    /// The added item never carries the session id: only the turn's text,
    /// under its date, reaches the reader.
    #[test]
    fn the_added_item_does_not_show_the_session_id() {
        let before = episode("answer_abc#0", "user: my favourite is jasmine tea");
        let reply = episode("answer_abc#1", "assistant: Jasmine is lovely.");
        let mut set = evidence_of(&[&reply]);
        let dated = ComposeConfig {
            resolve_relative: false,
            ..ComposeConfig::default()
        };
        attach_rounds(&mut set, &[before, reply], &dated).unwrap();
        assert!(set.items.iter().all(|it| !it.value.contains("answer_")));
        assert!(set.items[0].value.contains("user: my favourite is jasmine tea"));
    }
}
