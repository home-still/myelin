//! M66: turn windows. Evidence at the granularity of turns, not segments
//! (`docs/measurements/m66-turn-windows.md`).
//!
//! An episode is a segment of up to 512 tokens, about 13 conversational turns
//! ([`crate::pipeline::ingest::SegmentConfig`]). On LoCoMo a question gets
//! `k = 6` items, so it reaches only about 2.3 episodes, and most of each one
//! is text around the turn that matters. The 2026-09-25 loss anatomy put 114
//! multi-hop losses there: 63 rows held only part of their gold turns, 43
//! held none, and the reranked pool held them far more often (0.959) than the
//! emitted set did (0.863, M65).
//!
//! This module ranks **turns**, not episodes:
//! - Every turn of every episode in the reranked pool is scored against the
//!   question by the same cross-encoder, which also sidesteps its 512-token
//!   input limit that truncates a whole episode.
//! - Units are then taken best-first across the whole pool.
//! - An episode is kept as its best turns plus `radius` neighbours on each
//!   side, the rest elided.
//! - A fact or other non-episodic record competes as itself, scored as
//!   before.
//!
//! The grounds:
//! - QueryLink (Hu et al. 2026, `10.18653/v1/2026.findings-acl.765`) expands
//!   each hit by ±c neighbouring turns. With c = 0 its average drops 11.36.
//! - JustMem (Chen et al. 2026, arXiv 2609.19877) found fine units ranked
//!   globally beat whole-session packs by 10.72 points.
//! - MemPro's "focused evidence snippets" iteration (Liu et al. 2026, arXiv
//!   2606.00619, App. A.1) gained +1.47.
//! - RECOMP (Xu et al. 2023, arXiv 2310.04408) is extractive compression.
//! - "Context Length Alone Hurts" (Du et al. 2025,
//!   `10.18653/v1/2025.findings-emnlp.1264`) is why the unused turns go.
//!
//! The record is never rewritten. A window is a set of byte ranges into
//! `record.text`, rendered at compose time with `…` where turns were left
//! out, so every emitted line still quotes the memory it came from.

use std::collections::HashMap;
use std::ops::Range;

use uuid::Uuid;

use crate::error::{MyelinError, Result};
use crate::model::record::{MemoryRecord, RecordKind};
use crate::rerank::Reranker;

/// Neighbours kept on each side of a selected turn: QueryLink's c = 2.
pub const DEFAULT_TURN_WINDOW_RADIUS: usize = 2;
/// Marks turns left out of a windowed episode.
pub const ELISION: &str = "…";

/// Byte ranges of the turns in an episode's text.
///
/// An episode renders its turns as `Speaker: text` joined by newlines
/// ([`crate::pipeline::ingest::EpisodeDraft::render`]), and its `entities`
/// are exactly those speakers. A line that opens with one of them starts a
/// turn. Any other line continues the turn before it, because a turn's own
/// text may contain newlines. The first line always opens a turn. No range
/// includes the newline that separates it from the next.
pub fn turn_spans(text: &str, speakers: &[&str]) -> Vec<Range<usize>> {
    let mut spans: Vec<Range<usize>> = Vec::new();
    let mut offset = 0usize;
    for line in text.split('\n') {
        let start = offset;
        let end = start + line.len();
        offset = end + 1;
        let opens = spans.is_empty()
            || speakers.iter().any(|s| {
                line.strip_prefix(s)
                    .is_some_and(|rest| rest.starts_with(": "))
            });
        match spans.last_mut() {
            Some(last) if !opens => last.end = end,
            _ => spans.push(start..end),
        }
    }
    spans
}

/// The byte ranges kept for `centers` (turn indices), each widened by
/// `radius` turns on both sides, merged where they touch or overlap, and
/// in text order.
pub fn window_ranges(spans: &[Range<usize>], centers: &[usize], radius: usize) -> Vec<Range<usize>> {
    let last = match spans.len().checked_sub(1) {
        Some(last) => last,
        None => return Vec::new(),
    };
    let mut turns: Vec<(usize, usize)> = centers
        .iter()
        .filter(|&&c| c <= last)
        .map(|&c| (c.saturating_sub(radius), (c + radius).min(last)))
        .collect();
    turns.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (lo, hi) in turns {
        match merged.last_mut() {
            // Adjacent turn runs join: nothing is elided between them.
            Some(prev) if lo <= prev.1 + 1 => prev.1 = prev.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    merged
        .into_iter()
        .map(|(lo, hi)| spans[lo].start..spans[hi].end)
        .collect()
}

/// `text` with only `window`'s ranges, joined by [`ELISION`] lines, with an
/// elision at either end where turns were cut there. An empty window is the
/// whole text.
pub fn render(text: &str, window: &[Range<usize>]) -> String {
    let (Some(first), Some(last)) = (window.first(), window.last()) else {
        return text.to_string();
    };
    let mut out = String::new();
    if first.start > 0 {
        out.push_str(ELISION);
        out.push('\n');
    }
    for (i, r) in window.iter().enumerate() {
        if i > 0 {
            out.push('\n');
            out.push_str(ELISION);
            out.push('\n');
        }
        out.push_str(&text[r.clone()]);
    }
    if last.end < text.len() {
        out.push('\n');
        out.push_str(ELISION);
    }
    out
}

/// One reranked candidate in the pool, with its record in hand.
pub struct PoolItem {
    pub id: Uuid,
    pub score: f32,
    pub record: MemoryRecord,
}

/// A selected candidate: its record-level score for non-episodic records,
/// its best turn's score for an episode, and the episode's window.
pub struct Windowed {
    pub id: Uuid,
    pub score: f32,
    pub record: MemoryRecord,
    /// `None` for a non-episodic record, which is emitted whole.
    pub window: Option<Vec<Range<usize>>>,
}

/// What one selection cost, for the trace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WindowStats {
    pub turns_scored: usize,
    pub episodes_windowed: usize,
}

/// Rank the pool's turns and records together and take `slots` of them.
///
/// Units are every turn of every episode (scored now, on its own text) and
/// every other record (its pool score, also a cross-encoder score against
/// the same question). Walking them best-first:
/// - a unit whose record is not yet taken takes a slot, and an episode's
///   window starts at that turn;
/// - a later turn of an episode already taken opens a second window there,
///   **only while fewer than `k` units are taken**, counting records and
///   extra windows alike. So `k` bounds the windows as well as the records:
///   a turn that merely sits in an episode already taken does not ride in
///   free (JustMem ranks and counts its units the same way).
///
/// Past `k` the walk only fills the slack that `compose` needs for dedup and
/// the token budget. The caller passes `slots = 3k`, as `recall` always has,
/// and each slack record keeps just its best turn's window.
///
/// Ties break on pool order, then turn order, so a run is reproducible.
pub async fn select(
    reranker: &dyn Reranker,
    question: &str,
    pool: Vec<PoolItem>,
    k: usize,
    slots: usize,
    radius: usize,
) -> Result<(Vec<Windowed>, WindowStats)> {
    let spans: Vec<Option<Vec<Range<usize>>>> = pool
        .iter()
        .map(|p| {
            (p.record.kind == RecordKind::Episodic).then(|| {
                let speakers: Vec<&str> =
                    p.record.entities.iter().map(|e| e.phrase.as_str()).collect();
                turn_spans(&p.record.text, &speakers)
            })
        })
        .collect();

    let mut docs: Vec<String> = Vec::new();
    let mut owners: Vec<(usize, usize)> = Vec::new();
    for (i, s) in spans.iter().enumerate() {
        if let Some(s) = s {
            for (t, r) in s.iter().enumerate() {
                docs.push(pool[i].record.text[r.clone()].to_string());
                owners.push((i, t));
            }
        }
    }
    let turn_scores = if docs.is_empty() {
        Vec::new()
    } else {
        reranker.rerank(question, &docs).await?
    };
    if turn_scores.len() != docs.len() {
        return Err(MyelinError::Store(format!(
            "reranker returned {} scores for {} turns",
            turn_scores.len(),
            docs.len()
        )));
    }

    // (score, pool index, turn index or None for a whole record).
    let mut units: Vec<(f32, usize, Option<usize>)> = owners
        .iter()
        .zip(&turn_scores)
        .map(|(&(i, t), &s)| (s, i, Some(t)))
        .collect();
    units.extend(
        pool.iter()
            .enumerate()
            .filter(|(i, _)| spans[*i].is_none())
            .map(|(i, p)| (p.score, i, None)),
    );
    units.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut taken: Vec<(usize, f32)> = Vec::new();
    let mut centers: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut units_taken = 0usize;
    for (score, i, turn) in units {
        let already = centers.contains_key(&i) || taken.iter().any(|(j, _)| *j == i);
        if already {
            if units_taken < k {
                if let (Some(t), Some(c)) = (turn, centers.get_mut(&i)) {
                    c.push(t);
                    units_taken += 1;
                }
            }
            continue;
        }
        if taken.len() >= slots {
            break;
        }
        taken.push((i, score));
        units_taken += 1;
        if let Some(t) = turn {
            centers.insert(i, vec![t]);
        }
    }

    let stats = WindowStats {
        turns_scored: docs.len(),
        episodes_windowed: centers.len(),
    };
    let mut pool: Vec<Option<PoolItem>> = pool.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(taken.len());
    for (i, score) in taken {
        let Some(item) = pool[i].take() else { continue };
        let window = match (&spans[i], centers.get(&i)) {
            (Some(s), Some(c)) => Some(window_ranges(s, c, radius)),
            _ => None,
        };
        out.push(Windowed {
            id: item.id,
            score,
            record: item.record,
            window,
        });
    }
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPISODE: &str = "Caroline: hi\nMelanie: hello there\nCaroline: I went to a support group\nit was great\nMelanie: nice\nCaroline: bye";

    #[test]
    fn turns_split_on_the_episodes_own_speakers_and_keep_continuation_lines() {
        let spans = turn_spans(EPISODE, &["Caroline", "Melanie"]);
        let turns: Vec<&str> = spans.iter().map(|r| &EPISODE[r.clone()]).collect();
        assert_eq!(
            turns,
            vec![
                "Caroline: hi",
                "Melanie: hello there",
                "Caroline: I went to a support group\nit was great",
                "Melanie: nice",
                "Caroline: bye",
            ]
        );
    }

    #[test]
    fn a_colon_line_from_someone_not_in_the_episode_is_a_continuation() {
        let text = "user: my list\nNote: buy milk\nassistant: ok";
        let spans = turn_spans(text, &["user", "assistant"]);
        assert_eq!(spans.len(), 2);
        assert_eq!(&text[spans[0].clone()], "user: my list\nNote: buy milk");
    }

    #[test]
    fn a_window_keeps_the_turn_and_its_neighbours_and_marks_the_cut() {
        let spans = turn_spans(EPISODE, &["Caroline", "Melanie"]);
        let window = window_ranges(&spans, &[3], 1);
        assert_eq!(
            render(EPISODE, &window),
            "…\nCaroline: I went to a support group\nit was great\nMelanie: nice\nCaroline: bye"
        );
        let window = window_ranges(&spans, &[0], 1);
        assert_eq!(render(EPISODE, &window), "Caroline: hi\nMelanie: hello there\n…");
    }

    #[test]
    fn two_windows_merge_when_they_touch_and_elide_between_when_they_do_not() {
        let spans = turn_spans(EPISODE, &["Caroline", "Melanie"]);
        // Radius 0: turns 0 and 1 touch and join, so nothing is elided.
        assert_eq!(window_ranges(&spans, &[1, 0], 0), vec![spans[0].start..spans[1].end]);
        let window = window_ranges(&spans, &[0, 4], 0);
        assert_eq!(render(EPISODE, &window), "Caroline: hi\n…\nCaroline: bye");
    }

    /// Scores a turn by the keywords it holds, so the walk is checkable.
    struct Keywords;

    #[async_trait::async_trait]
    impl Reranker for Keywords {
        fn id(&self) -> &str {
            "keywords"
        }
        async fn rerank(&self, _query: &str, documents: &[String]) -> Result<Vec<f32>> {
            Ok(documents
                .iter()
                .map(|d| {
                    if d.contains("GOLD") {
                        10.0
                    } else if d.contains("near") {
                        5.0
                    } else {
                        0.0
                    }
                })
                .collect())
        }
    }

    fn item(kind: RecordKind, text: &str, score: f32) -> PoolItem {
        use crate::model::record::*;
        let id = Uuid::new_v4();
        PoolItem {
            id,
            score,
            record: MemoryRecord {
                id,
                kind,
                scope: Scope::new("t", "a", "ns"),
                text: text.into(),
                entities: vec![EntityRef::new("A"), EntityRef::new("B")],
                validity: Validity {
                    t_valid: chrono::Utc::now(),
                    t_invalid: None,
                    t_ingested: chrono::Utc::now(),
                    t_expired: None,
                },
                provenance: Provenance {
                    source: SourceRef::doc("d"),
                    contributed_by: ActorId::new("u"),
                    written_by: ActorId::new("w"),
                    derived_from: vec![],
                },
                trust: Trust::asserted(),
                salience: Salience::default(),
                links: vec![],
            },
        }
    }

    /// Turns outrank whole episodes: an episode ranked last in the pool
    /// whose one turn holds the answer is taken first, cut to that turn and
    /// its neighbours, and a fact competes on its own pool score.
    #[tokio::test]
    async fn the_walk_ranks_turns_across_the_pool_and_cuts_each_episode_to_its_window() {
        let long = "A: a\nB: b\nA: c\nB: d\nA: e";
        let pool = vec![
            item(RecordKind::Episodic, long, 9.0),
            item(RecordKind::Semantic, "a fact near it", 7.0),
            item(RecordKind::Episodic, "A: x\nB: y\nA: z GOLD\nB: w\nA: v", 1.0),
        ];
        let ids: Vec<Uuid> = pool.iter().map(|p| p.id).collect();
        let (out, stats) = select(&Keywords, "q", pool, 2, 6, 1).await.expect("select");
        assert_eq!(stats.turns_scored, 10);
        let order: Vec<Uuid> = out.iter().map(|w| w.id).collect();
        assert_eq!(order, vec![ids[2], ids[1], ids[0]], "turn 10.0, fact 7.0, turn 0.0");
        let gold = &out[0];
        assert_eq!(
            render(&gold.record.text, gold.window.as_deref().unwrap_or_default()),
            "…\nB: y\nA: z GOLD\nB: w\n…"
        );
        assert!(out[1].window.is_none(), "a fact is emitted whole");
    }

    /// A second strong turn opens a second window only while fewer than `k`
    /// units (records plus extra windows) are taken; weak turns never ride
    /// in free, even when the pool is smaller than `k`.
    #[tokio::test]
    async fn windows_grow_only_while_fewer_than_k_units_are_taken() {
        let text = "A: GOLD one\nB: b\nA: c\nB: d\nA: near two";
        let shown = |w: &Windowed| render(&w.record.text, w.window.as_deref().unwrap_or_default());
        let episode_of = |out: &[Windowed]| -> String {
            out.iter().find(|w| w.window.is_some()).map(shown).unwrap_or_default()
        };

        let (out, _) = select(&Keywords, "q", vec![item(RecordKind::Episodic, text, 1.0)], 2, 6, 0)
            .await
            .expect("select");
        assert_eq!(episode_of(&out), "A: GOLD one\n…\nA: near two", "two units, k = 2");

        let pool = || {
            vec![
                item(RecordKind::Episodic, text, 1.0),
                item(RecordKind::Semantic, "near fact", 6.0),
            ]
        };
        let (out, _) = select(&Keywords, "q", pool(), 3, 6, 0).await.expect("select");
        assert_eq!(
            episode_of(&out),
            "A: GOLD one\n…\nA: near two",
            "gold turn, fact, then the 5.0 turn as the third unit"
        );
        let (out, _) = select(&Keywords, "q", pool(), 2, 6, 0).await.expect("select");
        assert_eq!(
            episode_of(&out),
            "A: GOLD one\n…",
            "k = 2 is filled by the gold turn and the fact; the 5.0 turn is slack"
        );
    }

    #[test]
    fn a_window_covering_every_turn_is_the_record_verbatim() {
        let spans = turn_spans(EPISODE, &["Caroline", "Melanie"]);
        let window = window_ranges(&spans, &[2], 10);
        assert_eq!(render(EPISODE, &window), EPISODE);
    }
}
