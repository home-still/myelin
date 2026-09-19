//! Segmentation: a stream of turns becomes `Episodic` records (`PLAN.md` §6.1).
//!
//! **Boundary signals, not fixed windows.** LLMs segment text into events about
//! as humans do (`10.48550/arxiv.2502.06975` RQ2), and the signals that matter
//! are topic shift, task boundary and elapsed time. Two of those three are
//! deterministic and free — a session break *is* a task boundary, and a long
//! gap *is* a temporal one — so they are the primary splitters here. A token
//! cap is only a safety valve for a session that never pauses.
//!
//! **Turns are never split.** `03-retrieval.md` §7 row 0 is explicit that
//! multi-turn units stay intact for episodic material: half a turn is not an
//! event, and its `dia_id` provenance stops resolving. A single turn larger
//! than the cap therefore becomes its own oversized episode rather than being
//! cut or dropped.
//!
//! **Raw episodes are stored losslessly** and derived records point back at
//! them (Zep's bidirectional episode↔semantic index,
//! `06-consolidation-forgetting.md` §1). Nothing in this module summarises.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::model::record::{
    ActorId, EntityRef, MemoryRecord, Provenance, RecordKind, Salience, Scope, SourceRef, Trust,
    Validity,
};
use crate::store::ids::record_id;

/// One unit of input: a conversational turn, a tool result, or a trajectory
/// state. `source` is what makes the resulting record admissible (I2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    pub speaker: String,
    pub text: String,
    /// When the turn happened in the world, if the corpus says. Drives the
    /// elapsed-time boundary and becomes the episode's `t_valid`.
    pub at: Option<DateTime<Utc>>,
    pub source: SourceRef,
    /// Corpus-level grouping (a LoCoMo session, an LME-V2 trajectory). A
    /// change here is always a boundary.
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentConfig {
    /// Safety valve only. 512 is the `arXiv:2407.01219` §3.2.1 Table 3
    /// starting point — best on faithfulness (97.59), while **256 wins on
    /// relevancy** (97.78 vs 97.41) — on a single-document eval with a
    /// different generator. It is a parameter to tune on our own eval, not a
    /// settled result.
    pub max_tokens: usize,
    /// A silence this long ends the episode.
    pub max_gap_minutes: i64,
    /// Never merge across corpus units.
    pub respect_units: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            max_gap_minutes: 60,
            respect_units: true,
        }
    }
}

/// Why an episode ended. Recorded because segmentation quality is an input to
/// extraction quality, and `PLAN.md` §13 names extraction as the dominant
/// error source — when recall is bad we need to know whether the boundary or
/// the model was at fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryReason {
    UnitChange,
    TimeGap,
    TokenCap,
    EndOfStream,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeDraft {
    pub unit: String,
    pub turns: Vec<Turn>,
    pub ended_by: BoundaryReason,
    pub approx_tokens: usize,
}

impl EpisodeDraft {
    /// The lossless rendering that gets stored and embedded. Speaker labels
    /// are kept: "who said it" is load-bearing for LoCoMo's speaker-attributed
    /// questions.
    pub fn render(&self) -> String {
        self.turns
            .iter()
            .map(|t| format!("{}: {}", t.speaker, t.text))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The episode as only one speaker said it. A disposition is a property of
    /// the person who stated it, and on LongMemEval_S the user is 12.6% of the
    /// corpus — so a profile pass that reads whole episodes costs 8x what it
    /// needs to and reads 53.5M tokens of assistant prose that states nobody's
    /// preferences.
    pub fn render_speaker(&self, speaker: &str) -> String {
        self.turns
            .iter()
            .filter(|t| t.speaker == speaker)
            .map(|t| format!("{}: {}", t.speaker, t.text))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn t_valid(&self) -> DateTime<Utc> {
        self.turns
            .iter()
            .filter_map(|t| t.at)
            .min()
            .unwrap_or_else(Utc::now)
    }

    /// Natural key for the v5 id, so replaying a corpus is idempotent
    /// (`PLAN.md` §4: same logical episode → same point, update not duplicate).
    pub fn natural_key(&self) -> String {
        let first = self
            .turns
            .first()
            .map(|t| t.source.doc.as_str())
            .unwrap_or("");
        let last = self
            .turns
            .last()
            .map(|t| t.source.doc.as_str())
            .unwrap_or("");
        format!("episode\u{1f}{}\u{1f}{first}\u{1f}{last}", self.unit)
    }

    /// Build the `Episodic` record. Provenance spans the turn range, which is
    /// what makes "show me the turns behind this" answerable (I2, C6b).
    pub fn to_record(
        &self,
        scope: &Scope,
        contributed_by: &ActorId,
        written_by: &ActorId,
    ) -> MemoryRecord {
        let now = Utc::now();
        let t_valid = self.t_valid();
        let first = self
            .turns
            .first()
            .map(|t| t.source.clone())
            .unwrap_or_else(|| SourceRef::doc(&self.unit));
        MemoryRecord {
            id: record_id(scope, &self.natural_key()),
            kind: RecordKind::Episodic,
            scope: scope.clone(),
            text: self.render(),
            entities: self
                .turns
                .iter()
                .map(|t| EntityRef::new(t.speaker.clone()))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            validity: Validity {
                t_valid,
                t_invalid: None,
                t_ingested: now,
                t_expired: None,
            },
            provenance: Provenance {
                source: first,
                contributed_by: contributed_by.clone(),
                written_by: written_by.clone(),
                derived_from: Vec::new(),
            },
            trust: Trust::asserted(),
            salience: Salience::default(),
            links: Vec::new(),
        }
    }
}

/// Approximate token count.
///
/// Deliberately a heuristic and deliberately not a tokenizer dependency: the
/// serving model is Qwen, so a `cl100k` count would be precise about the wrong
/// vocabulary. Segmentation only needs this to decide *whether* a run-on
/// session has gone on too long, and the real budget accounting in `compose`
/// (§7.3) uses the serving model's own `/tokenize` endpoint where exactness
/// actually matters.
///
/// ~4 chars per token is the standard English approximation; whitespace-run
/// counting keeps it stable on the punctuation-heavy accessibility trees in
/// LME-V2 trajectories.
pub fn approx_tokens(s: &str) -> usize {
    let chars = s.chars().count();
    let words = s.split_whitespace().count();
    // Take the larger of the two estimates: code and markup under-count by
    // words, prose under-counts by chars/4.
    (chars / 4).max(words)
}

/// Split a turn stream into episodes.
///
/// Turns are assumed to be in corpus order. The function is total: every input
/// turn appears in exactly one output episode, which the tests assert, because
/// a segmenter that silently drops material would show up only as an
/// unexplained recall ceiling much later.
pub fn segment(turns: &[Turn], cfg: &SegmentConfig) -> Vec<EpisodeDraft> {
    let mut episodes = Vec::new();
    let mut current: Vec<Turn> = Vec::new();
    let mut tokens = 0usize;

    let flush =
        |current: &mut Vec<Turn>, tokens: &mut usize, reason, out: &mut Vec<EpisodeDraft>| {
            if current.is_empty() {
                return;
            }
            out.push(EpisodeDraft {
                unit: current[0].unit.clone(),
                turns: std::mem::take(current),
                ended_by: reason,
                approx_tokens: *tokens,
            });
            *tokens = 0;
        };

    for turn in turns {
        let turn_tokens = approx_tokens(&turn.text);

        if let Some(previous) = current.last() {
            let unit_changed = cfg.respect_units && previous.unit != turn.unit;
            let gapped = match (previous.at, turn.at) {
                (Some(a), Some(b)) => b - a > Duration::minutes(cfg.max_gap_minutes),
                _ => false,
            };
            // Cap check happens only when something is already buffered, so a
            // single oversized turn survives as its own episode instead of
            // being cut in half.
            let over_cap = tokens + turn_tokens > cfg.max_tokens;

            let reason = if unit_changed {
                Some(BoundaryReason::UnitChange)
            } else if gapped {
                Some(BoundaryReason::TimeGap)
            } else if over_cap {
                Some(BoundaryReason::TokenCap)
            } else {
                None
            };
            if let Some(reason) = reason {
                flush(&mut current, &mut tokens, reason, &mut episodes);
            }
        }

        tokens += turn_tokens;
        current.push(turn.clone());
    }
    flush(
        &mut current,
        &mut tokens,
        BoundaryReason::EndOfStream,
        &mut episodes,
    );
    episodes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(unit: &str, speaker: &str, text: &str, minute: i64) -> Turn {
        Turn {
            speaker: speaker.into(),
            text: text.into(),
            at: Some(DateTime::from_timestamp(minute * 60, 0).unwrap()),
            source: SourceRef::doc(format!("{unit}:{minute}")),
            unit: unit.into(),
        }
    }

    fn cfg() -> SegmentConfig {
        SegmentConfig {
            max_tokens: 20,
            max_gap_minutes: 60,
            respect_units: true,
        }
    }

    /// Totality. A dropped turn is invisible until it shows up as a recall
    /// ceiling nobody can explain.
    #[test]
    fn every_turn_lands_in_exactly_one_episode() {
        let turns: Vec<Turn> = (0..25)
            .map(|i| turn(if i < 12 { "s1" } else { "s2" }, "A", "short text here", i))
            .collect();
        let episodes = segment(&turns, &cfg());

        let emitted: Vec<&Turn> = episodes.iter().flat_map(|e| e.turns.iter()).collect();
        assert_eq!(emitted.len(), turns.len());
        for (got, want) in emitted.iter().zip(&turns) {
            assert_eq!(*got, want, "order or content changed");
        }
    }

    /// A session break is a task boundary and always splits, even mid-thought
    /// and even when the token budget is nowhere near full.
    #[test]
    fn unit_change_always_splits() {
        let turns = vec![
            turn("s1", "A", "hi", 0),
            turn("s1", "B", "hello", 1),
            turn("s2", "A", "new session", 2),
        ];
        let episodes = segment(&turns, &cfg());
        assert_eq!(episodes.len(), 2, "{episodes:#?}");
        assert_eq!(episodes[0].ended_by, BoundaryReason::UnitChange);
        assert_eq!(episodes[0].turns.len(), 2);
        assert_eq!(episodes[1].unit, "s2");
    }

    #[test]
    fn a_long_silence_splits() {
        let turns = vec![
            turn("s1", "A", "hi", 0),
            turn("s1", "B", "hello", 5),
            turn("s1", "A", "much later", 5000),
        ];
        let episodes = segment(&turns, &cfg());
        assert_eq!(episodes.len(), 2);
        assert_eq!(episodes[0].ended_by, BoundaryReason::TimeGap);
        assert_eq!(episodes[1].turns.len(), 1);
    }

    /// The cap splits between turns, never inside one.
    #[test]
    fn token_cap_splits_at_a_turn_boundary() {
        let turns: Vec<Turn> = (0..6)
            .map(|i| turn("s1", "A", "one two three four five six seven", i))
            .collect();
        let episodes = segment(&turns, &cfg());
        assert!(episodes.len() > 1, "cap never fired: {episodes:#?}");
        assert!(episodes
            .iter()
            .take(episodes.len() - 1)
            .all(|e| e.ended_by == BoundaryReason::TokenCap));
        // No turn was cut: each episode's text is a concatenation of whole turns.
        for e in &episodes {
            for t in &e.turns {
                assert!(e.render().contains(&t.text));
            }
        }
    }

    /// A turn bigger than the whole budget must survive intact rather than be
    /// truncated or dropped.
    #[test]
    fn an_oversized_single_turn_becomes_its_own_episode() {
        let huge = "word ".repeat(500);
        let turns = vec![
            turn("s1", "A", "small", 0),
            turn("s1", "B", &huge, 1),
            turn("s1", "A", "small again", 2),
        ];
        let episodes = segment(&turns, &cfg());
        let carrying: Vec<_> = episodes
            .iter()
            .filter(|e| e.turns.iter().any(|t| t.text.len() == huge.len()))
            .collect();
        assert_eq!(carrying.len(), 1, "oversized turn was split or lost");
        assert_eq!(carrying[0].turns.len(), 1);
        assert!(carrying[0].approx_tokens > cfg().max_tokens);
    }

    /// Replaying the same corpus must hit the same ids — otherwise every
    /// re-ingest duplicates the store instead of updating it.
    #[test]
    fn episode_ids_are_stable_across_runs() {
        let turns = vec![turn("s1", "A", "hi", 0), turn("s1", "B", "hello", 1)];
        let scope = Scope::new("t", "a", "ns");
        let (u, w) = (ActorId::new("u"), ActorId::new("w"));

        let first = segment(&turns, &cfg())[0].to_record(&scope, &u, &w);
        let second = segment(&turns, &cfg())[0].to_record(&scope, &u, &w);
        assert_eq!(first.id, second.id);
        assert_eq!(first.text, second.text);

        // A different tenant must not collide onto the same point (C12).
        let other = Scope::new("other-tenant", "a", "ns");
        assert_ne!(
            first.id,
            segment(&turns, &cfg())[0].to_record(&other, &u, &w).id
        );
    }

    /// The episode is the raw material, not a summary of it.
    #[test]
    fn rendering_is_lossless_and_speaker_attributed() {
        let turns = vec![
            turn("s1", "Caroline", "Hey Mel! Good to see you!", 0),
            turn("s1", "Melanie", "I have been well.", 1),
        ];
        let e = &segment(&turns, &cfg())[0];
        let rendered = e.render();
        for t in &turns {
            assert!(rendered.contains(&t.text), "lost turn text: {rendered}");
            assert!(rendered.contains(&t.speaker), "lost speaker: {rendered}");
        }
    }

    /// `t_valid` is when the episode happened in the world, not when we read
    /// it — the distinction bi-temporal validity exists for.
    #[test]
    fn t_valid_is_the_earliest_turn_time() {
        let turns = vec![turn("s1", "A", "first", 10), turn("s1", "B", "second", 11)];
        let e = &segment(&turns, &cfg())[0];
        assert_eq!(e.t_valid(), DateTime::from_timestamp(600, 0).unwrap());

        let scope = Scope::new("t", "a", "ns");
        let record = e.to_record(&scope, &ActorId::new("u"), &ActorId::new("w"));
        assert_eq!(record.validity.t_valid, e.t_valid());
        assert!(record.validity.t_ingested > record.validity.t_valid);
    }

    #[test]
    fn empty_input_yields_no_episodes() {
        assert!(segment(&[], &cfg()).is_empty());
    }

    /// Turns with no timestamp must not fabricate a gap boundary.
    #[test]
    fn missing_timestamps_do_not_split() {
        let mut turns = vec![turn("s1", "A", "a", 0), turn("s1", "B", "b", 1)];
        turns[0].at = None;
        turns[1].at = None;
        let episodes = segment(&turns, &cfg());
        assert_eq!(episodes.len(), 1);
    }
}
