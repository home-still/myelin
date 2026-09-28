//! M76: the user's own words, from the sessions the evidence already reached
//! (`docs/measurements/m76-user-words.md`).
//!
//! When a question asks for advice
//! ([`crate::pipeline::query_shape::is_advice_request`]), the answer is judged
//! on whether it builds on what the user said about themselves. On
//! LongMemEval_S the evidence usually reaches the right conversation but not
//! the user's sentence. Measured reader-free on M57's shipped run
//! (2026-09-27):
//! - the session holding the preference is in evidence for **29 of 30**
//!   preference questions;
//! - the gold turn itself is there for only **13**;
//! - every gold turn is a *user* turn. It ranks below the assistant's long
//!   replies, which fill the 512-token episodes the evidence is made of.
//!
//! So this block reads the other segments of those same sessions from the
//! ledger, keeps only the user's turns, ranks them against the question with
//! the cross-encoder, and appends the best of them within its own budget.
//! Taking every user turn of those sessions would hold every gold turn in 29
//! of 30, for a median of about 700 tokens.
//!
//! The grounds:
//! - PrefEval (Zhao et al. 2025, `10.48550/arxiv.2502.09597`): retrieving the
//!   user's stated preference is the best remedy for preference-unaware
//!   answers on every model it tests, and lost-in-the-middle applies to
//!   preferences too.
//! - CueMem (Wang et al. 2026, arXiv 2609.12354): context is rebuilt from the
//!   *source turns*, and it is best on LongMemEval's single-session-preference
//!   and single-session-user strata.
//!
//! This is not M20b. M20b appended *extracted* profile records, and the reader
//! anchored on one of them and narrowed its advice to that one fact (M20b
//! result, 2026-09-27). This block appends the user's turns verbatim: a
//! windowed view of the stored episode, each line quoting the record it came
//! from. The base evidence is never reordered, rewritten or dropped, and a
//! shut gate appends nothing at all.

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;

use chrono::Utc;
use uuid::Uuid;

use crate::error::{MyelinError, Result};
use crate::model::evidence::{EvidenceItem, EvidenceKind, EvidenceSet};
use crate::model::query::Recall;
use crate::model::record::{MemoryRecord, RecordKind, SourceRef};
use crate::pipeline::compose::{compose, weakest_trust, ComposeConfig, Ranked};
use crate::pipeline::ingest::approx_tokens;
use crate::pipeline::query_shape::is_advice_request;
use crate::pipeline::turn_windows::{render, turn_spans};
use crate::rerank::Reranker;
use crate::store::ledger::Ledger;

/// The block's own token budget, outside the evidence's. Every user turn of
/// the evidence's sessions costs a median of about 700 tokens on the 30
/// preference questions, so this holds all of them for most questions and
/// the best-ranked ones for the rest.
pub const USER_WORDS_BUDGET_TOKENS: usize = 1024;
/// The most episodes a scope may hold. Above it the block refuses rather than
/// rank a truncated set. A LongMemEval_S tenant holds about 330.
pub const USER_WORDS_MAX_EPISODES: usize = 2000;
/// Opens the block, so the reader sees it as a separate list.
pub const USER_WORDS_HEADER: &str =
    "[your words] What the user said about themselves, in the conversations above:";
/// The source the header item carries.
pub const USER_WORDS_MECHANISM: &str = "m76:user-words";
/// The speakers of a LongMemEval_S episode (`ingest` writes each turn as
/// `<role>: <text>`).
const SPEAKERS: [&str; 2] = ["user", "assistant"];
/// The turn prefix kept.
const USER_TURN_PREFIX: &str = "user: ";
/// Separates a session from its segment in an episode's source doc
/// (`<session>#<segment>`).
const SEGMENT_SEPARATOR: char = '#';

/// The user's turns, ranked, from the ledger's episodes of the evidence's
/// sessions.
pub struct UserWords<'a> {
    pub ledger: &'a Ledger,
    pub reranker: &'a dyn Reranker,
}

impl<'a> UserWords<'a> {
    pub fn new(ledger: &'a Ledger, reranker: &'a dyn Reranker) -> Self {
        Self { ledger, reranker }
    }

    /// Append the question's best user turns after everything in `set`,
    /// headed by [`USER_WORDS_HEADER`]. Returns how many turns were appended.
    /// A shut gate, evidence with no session, or no user turn outside the
    /// evidence appends nothing, not even the header, so the row is
    /// byte-identical to a run without the block.
    pub async fn append(&self, query: &Recall, set: &mut EvidenceSet) -> Result<usize> {
        if !is_advice_request(&query.text) {
            return Ok(0);
        }
        if evidence_sessions(set).is_empty() {
            return Ok(0);
        }
        let episodes = self
            .ledger
            .visible_of_kind(
                &query.scope,
                RecordKind::Episodic,
                Utc::now(),
                USER_WORDS_MAX_EPISODES as i64,
            )
            .await?;
        if episodes.len() >= USER_WORDS_MAX_EPISODES {
            return Err(MyelinError::Store(format!(
                "{USER_WORDS_MECHANISM}: scope {:?} holds {USER_WORDS_MAX_EPISODES} or more episodes; the block ranks all of its sessions' turns and refuses a truncated set",
                query.scope.tenant
            )));
        }
        append_user_turns(self.reranker, &query.text, episodes, set).await
    }
}

/// The sessions the evidence's episodes came from, by their source doc
/// (`<session>#<segment>`). Items without a record (headers, the timeline) or
/// without a segment marker are not episodes and name no session.
pub fn evidence_sessions(set: &EvidenceSet) -> HashSet<String> {
    set.items
        .iter()
        .filter(|it| !it.record_id.is_nil())
        .filter_map(|it| session_of(&it.source.doc))
        .collect()
}

fn session_of(doc: &str) -> Option<String> {
    doc.rsplit_once(SEGMENT_SEPARATOR).map(|(session, _)| session.to_string())
}

/// One user turn: its episode, its byte range, its score.
struct Turn {
    episode: usize,
    range: Range<usize>,
    score: f32,
}

/// The block over `episodes`: the user turns of the evidence's sessions that
/// are not already in the evidence, ranked by `reranker` and taken best-first
/// while the rendered block fits [`USER_WORDS_BUDGET_TOKENS`]. The first turn
/// is always taken, as `compose` always admits its top item.
pub(crate) async fn append_user_turns(
    reranker: &dyn Reranker,
    question: &str,
    episodes: Vec<MemoryRecord>,
    set: &mut EvidenceSet,
) -> Result<usize> {
    let sessions = evidence_sessions(set);
    let in_evidence: HashSet<Uuid> = set.items.iter().map(|it| it.record_id).collect();
    let episodes: Vec<MemoryRecord> = episodes
        .into_iter()
        .filter(|r| !in_evidence.contains(&r.id))
        .filter(|r| session_of(&r.provenance.source.doc).is_some_and(|s| sessions.contains(&s)))
        .collect();
    let mut turns: Vec<Turn> = Vec::new();
    for (e, record) in episodes.iter().enumerate() {
        for range in turn_spans(&record.text, &SPEAKERS) {
            if record.text[range.clone()].starts_with(USER_TURN_PREFIX) {
                turns.push(Turn {
                    episode: e,
                    range,
                    score: 0.0,
                });
            }
        }
    }
    if turns.is_empty() {
        return Ok(0);
    }
    let texts: Vec<String> = turns
        .iter()
        .map(|t| episodes[t.episode].text[t.range.clone()].to_string())
        .collect();
    let scores = reranker.rerank(question, &texts).await?;
    if scores.len() != turns.len() {
        return Err(MyelinError::Store(format!(
            "reranker returned {} scores for {} user turns",
            scores.len(),
            turns.len()
        )));
    }
    for (t, s) in turns.iter_mut().zip(scores) {
        t.score = s;
    }
    // Best first; ties by record id, then position, so a rerun composes the
    // same block.
    turns.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| episodes[a.episode].id.cmp(&episodes[b.episode].id))
            .then_with(|| a.range.start.cmp(&b.range.start))
    });

    // Greedy under the budget, charged as `compose` charges: the rendered
    // window of each episode, elisions included.
    let mut chosen: BTreeMap<usize, (Vec<Range<usize>>, f32)> = BTreeMap::new();
    let cost = |chosen: &BTreeMap<usize, (Vec<Range<usize>>, f32)>| -> usize {
        chosen
            .iter()
            .map(|(e, (w, _))| approx_tokens(&render(&episodes[*e].text, w)))
            .sum()
    };
    for turn in &turns {
        let entry = chosen.entry(turn.episode).or_insert_with(|| (Vec::new(), turn.score));
        entry.0.push(turn.range.clone());
        entry.0.sort_by_key(|r| r.start);
        let first = chosen.values().map(|(w, _)| w.len()).sum::<usize>() == 1;
        if !first && cost(&chosen) > USER_WORDS_BUDGET_TOKENS {
            let entry = chosen.get_mut(&turn.episode).ok_or_else(|| {
                MyelinError::Store(format!("{USER_WORDS_MECHANISM}: lost episode {}", turn.episode))
            })?;
            entry.0.retain(|r| *r != turn.range);
            if entry.0.is_empty() {
                chosen.remove(&turn.episode);
            }
        }
    }
    let n_turns: usize = chosen.values().map(|(w, _)| w.len()).sum();
    let mut ranked: Vec<Ranked> = chosen
        .into_iter()
        .map(|(e, (window, score))| Ranked {
            record: episodes[e].clone(),
            score,
            vector: None,
            window: Some(window),
        })
        .collect();
    // An episode ranks by its best turn, which is the score it was entered at.
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.record.id.cmp(&b.record.id))
    });
    let block = compose(
        ranked,
        &ComposeConfig {
            k: USER_WORDS_MAX_EPISODES,
            max_tokens: USER_WORDS_BUDGET_TOKENS,
            timeline: false,
            resolve_relative: false,
            ..ComposeConfig::default()
        },
    );
    if block.items.is_empty() {
        return Ok(0);
    }
    let header = EvidenceItem {
        kind: EvidenceKind::Text,
        value: USER_WORDS_HEADER.to_string(),
        record_id: Uuid::nil(),
        source: SourceRef::doc(USER_WORDS_MECHANISM),
        score: 0.0,
        trust: weakest_trust(block.items.iter().map(|i| i.trust)),
    };
    set.tokens += approx_tokens(&header.value) + block.tokens;
    set.items.push(header);
    set.items.extend(block.items);
    Ok(n_turns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::record::*;

    /// Scores a turn by the keywords it holds.
    struct Keywords;

    #[async_trait::async_trait]
    impl Reranker for Keywords {
        fn id(&self) -> &str {
            "keywords"
        }
        async fn rerank(&self, _query: &str, documents: &[String]) -> Result<Vec<f32>> {
            Ok(documents
                .iter()
                .map(|d| if d.contains("GOLD") { 10.0 } else { 1.0 })
                .collect())
        }
    }

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

    const ADVICE: &str = "Any tips on what to bake for my colleagues?";

    /// The user's turn from another segment of a session the evidence holds
    /// is appended, with its assistant turns left out; a session the evidence
    /// never reached contributes nothing.
    #[tokio::test]
    async fn the_block_holds_the_user_turns_of_the_evidence_sessions_only() {
        let seen = episode("s1#0", "user: can you share a cake recipe?\nassistant: a very long reply");
        let sibling = episode(
            "s1#1",
            "assistant: more cake talk\nuser: GOLD my lemon poppyseed cake was a hit\nassistant: lovely",
        );
        let elsewhere = episode("s2#0", "user: GOLD I hate baking\nassistant: ok");
        let mut set = evidence_of(&[&seen]);
        let before = set.items.len();
        let n = append_user_turns(&Keywords, ADVICE, vec![seen.clone(), sibling.clone(), elsewhere], &mut set)
            .await
            .expect("append");
        assert_eq!(n, 1, "one user turn, from the sibling segment");
        assert_eq!(set.items[before].value, USER_WORDS_HEADER);
        let block: Vec<&str> = set.items[before + 1..].iter().map(|i| i.value.as_str()).collect();
        assert_eq!(block.len(), 1);
        assert!(block[0].contains("user: GOLD my lemon poppyseed cake was a hit"), "{}", block[0]);
        assert!(!block[0].contains("assistant"), "{}", block[0]);
        assert!(!block[0].contains("I hate baking"), "another session: {}", block[0]);
        assert_eq!(set.items[before + 1].record_id, sibling.id, "the line quotes its record");
    }

    /// A record already in the evidence is not repeated, and evidence with no
    /// sibling segment appends nothing, not even the header.
    #[tokio::test]
    async fn nothing_outside_the_evidence_appends_nothing() {
        let seen = episode("s1#0", "user: GOLD cake\nassistant: reply");
        let mut set = evidence_of(&[&seen]);
        let n = append_user_turns(&Keywords, ADVICE, vec![seen.clone()], &mut set)
            .await
            .expect("append");
        assert_eq!(n, 0);
        assert_eq!(set.items.len(), 1, "byte-identical: no header");
    }

    /// Items with no record, or a doc with no segment marker, name no session.
    #[test]
    fn only_episodes_name_sessions() {
        let ep = episode("s1#3", "user: hi");
        let fact = episode("a-fact", "the user likes tea");
        let mut set = evidence_of(&[&ep, &fact]);
        set.items.push(EvidenceItem {
            kind: EvidenceKind::Text,
            value: "[timeline]".into(),
            record_id: Uuid::nil(),
            source: SourceRef::doc("x#1"),
            score: 0.0,
            trust: TrustTier::Asserted,
        });
        let sessions = evidence_sessions(&set);
        assert_eq!(sessions, HashSet::from(["s1".to_string()]));
    }

    /// Past the budget, the best-ranked turns stay and the rest are left out.
    #[tokio::test]
    async fn the_budget_keeps_the_best_turns() {
        let seen = episode("s1#0", "user: hello\nassistant: hi");
        let filler = "word ".repeat(700);
        let long = episode(
            "s1#1",
            &format!("user: {filler}\nassistant: ok\nuser: GOLD I love lemon\nassistant: nice\nuser: {filler}"),
        );
        let mut set = evidence_of(&[&seen]);
        let before = set.items.len();
        let n = append_user_turns(&Keywords, ADVICE, vec![seen.clone(), long], &mut set)
            .await
            .expect("append");
        let block = &set.items[before + 1].value;
        assert!(block.contains("GOLD I love lemon"), "{block}");
        assert_eq!(n, 2, "the gold turn and one 700-word turn fit; the second does not");
        assert_eq!(block.matches("user: word").count(), 1, "{block}");
    }
}
