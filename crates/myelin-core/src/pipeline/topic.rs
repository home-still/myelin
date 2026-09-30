//! M89a: topic boundaries inside a session (`docs/measurements/m89-typed-memory.md`).
//!
//! Ingest ends an episode at a session change, a long silence or a 512-token
//! cap ([`crate::pipeline::ingest`]). The third boundary signal named there,
//! a topic shift, was never computed. On LoCoMo that leaves a ~13-turn
//! episode holding two or three topics. When the reader held every gold turn
//! and still answered wrong (119 of 329 losses under the bar's judge,
//! 2026-09-30), it usually took a nearby detail from another topic in the
//! same chunk.
//!
//! This module finds where a session changes topic, deterministically:
//! 1. **Key utterances.** A lexical rule drops acknowledgements, greetings
//!    and restatements from the *similarity sequence* only. Every turn stays
//!    in its episode, because a segmenter that drops material shows up only
//!    as an unexplained recall ceiling.
//! 2. **Depth scores** over the cosine similarity of adjacent key
//!    utterances, as TextTiling scores gaps between blocks (Hearst 1997,
//!    "TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages",
//!    Computational Linguistics 23(1), ACL J97-1003).
//! 3. **Boundaries** at local maxima of depth at or above the session's own
//!    `mean − σ/2`. That is LeanMem's adaptive valley rule (arXiv
//!    2608.03463, §3.1) and TextTiling's liberal cutoff.
//!
//! Grounds for topic-sized memory units: SeCom (Pan et al. 2025, arXiv
//! 2502.05589) finds segment-level memory beats both turn- and session-level
//! on LOCOMO, by up to 11.98 GPT4Score. LeanMem's topic segmentation is
//! worth +2.08 in its own ablation, and LightMem segments by topic too
//! (arXiv 2510.18866).

use std::collections::BTreeSet;

use crate::embed::Embedder;
use crate::error::{MyelinError, Result};
use crate::pipeline::consolidate::cosine;
use crate::pipeline::ingest::{segment_at, EpisodeDraft, SegmentConfig, Turn};

/// Utterances on each side of a gap whose similarity is averaged (TextTiling's
/// block size, in utterances).
pub const TOPIC_WINDOW: usize = 2;
/// The boundary threshold is `mean − TOPIC_SIGMA_FRACTION · σ` of the
/// session's depth scores (TextTiling's liberal cutoff; LeanMem §3.1).
pub const TOPIC_SIGMA_FRACTION: f32 = 0.5;
/// The fewest key utterances a topic segment may hold. A shorter one is a
/// digression, not a topic.
pub const MIN_KEY_PER_SEGMENT: usize = 2;
/// An utterance with fewer words than this states nothing on its own.
pub const KEY_MIN_WORDS: usize = 4;
/// An utterance this short that opens with an acknowledgement is one.
pub const ACK_MAX_WORDS: usize = 8;
/// Share of an utterance's content words already in the one before it, at or
/// above which it restates rather than adds.
pub const RESTATE_OVERLAP: f32 = 0.8;
/// A content word is at least this long.
const CONTENT_MIN_CHARS: usize = 3;
/// How acknowledgements and greetings open.
const ACK_CUES: [&str; 22] = [
    "hi", "hey", "hello", "bye", "goodbye", "thanks", "thank you", "ok", "okay", "sure", "yes",
    "yeah", "yep", "wow", "cool", "nice", "great", "awesome", "sounds good", "you too",
    "take care", "see you",
];

/// How topics are found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopicConfig {
    pub window: usize,
    pub sigma_fraction: f32,
    pub min_key_per_segment: usize,
}

impl Default for TopicConfig {
    fn default() -> Self {
        Self {
            window: TOPIC_WINDOW,
            sigma_fraction: TOPIC_SIGMA_FRACTION,
            min_key_per_segment: MIN_KEY_PER_SEGMENT,
        }
    }
}

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn content_words(text: &str) -> BTreeSet<String> {
    words(text)
        .into_iter()
        .filter(|w| w.chars().count() >= CONTENT_MIN_CHARS)
        .collect()
}

/// Which utterances carry content: the rest are left out of the similarity
/// sequence (never out of the episode). A restatement is judged against the
/// last key utterance, not the turn just before, which is often an
/// acknowledgement between the two.
pub fn key_utterances(texts: &[&str]) -> Vec<bool> {
    let mut out = Vec::with_capacity(texts.len());
    let mut last_key: Option<BTreeSet<String>> = None;
    for text in texts {
        let w = words(text);
        let lower = w.join(" ");
        let ack = w.len() <= ACK_MAX_WORDS
            && ACK_CUES
                .iter()
                .any(|c| lower == *c || lower.starts_with(&format!("{c} ")));
        let mine = content_words(text);
        let restates = last_key.as_ref().is_some_and(|before| {
            !mine.is_empty()
                && mine.iter().filter(|w| before.contains(*w)).count() as f32 / mine.len() as f32
                    >= RESTATE_OVERLAP
        });
        let key = w.len() >= KEY_MIN_WORDS && !ack && !restates;
        if key {
            last_key = Some(mine);
        }
        out.push(key);
    }
    out
}

/// Depth at each gap `i` (between key utterances `i` and `i + 1`) whose
/// windows are complete on both sides: the mean similarity of the `w` gaps
/// before it and the `w` after it, less twice its own (LeanMem §3.1).
pub fn depth_scores(sims: &[f32], w: usize) -> Vec<(usize, f32)> {
    if w == 0 || sims.len() < 2 * w + 1 {
        return Vec::new();
    }
    (w..sims.len() - w)
        .map(|i| {
            let left = sims[i - w..i].iter().sum::<f32>() / w as f32;
            let right = sims[i + 1..=i + w].iter().sum::<f32>() / w as f32;
            (i, left + right - 2.0 * sims[i])
        })
        .collect()
}

/// The gaps that are topic boundaries: strict local maxima of depth (deeper
/// than each neighbouring gap that has a depth, so a flat run is never a
/// valley) at or above `mean − sigma_fraction · σ`, deepest first, each at
/// least
/// `min_key` key utterances from every boundary already taken and from
/// either end of the sequence (`n_key` utterances long).
pub fn topic_boundaries(depths: &[(usize, f32)], sigma_fraction: f32, min_key: usize, n_key: usize) -> Vec<usize> {
    if depths.is_empty() {
        return Vec::new();
    }
    let n = depths.len() as f32;
    let mean = depths.iter().map(|(_, d)| d).sum::<f32>() / n;
    let sd = (depths.iter().map(|(_, d)| (d - mean).powi(2)).sum::<f32>() / n).sqrt();
    // A session whose depths do not vary has no valley to find.
    if sd == 0.0 {
        return Vec::new();
    }
    let tau = mean - sigma_fraction * sd;
    let mut candidates: Vec<(usize, f32)> = depths
        .iter()
        .enumerate()
        .filter(|(j, (_, d))| {
            let left = j.checked_sub(1).map(|k| depths[k].1);
            let right = depths.get(j + 1).map(|p| p.1);
            *d >= tau && left.is_none_or(|l| *d > l) && right.is_none_or(|r| *d > r)
        })
        .map(|(_, p)| *p)
        .collect();
    // Deepest first; ties by position, so a rerun cuts the same places.
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut taken: Vec<usize> = Vec::new();
    for (gap, _) in candidates {
        // The segment after gap `g` starts at key utterance `g + 1`.
        let start = gap + 1;
        let fits_ends = start >= min_key && n_key - start >= min_key;
        let fits_others = taken.iter().all(|t| (t + 1).abs_diff(start) >= min_key);
        if fits_ends && fits_others {
            taken.push(gap);
        }
    }
    taken.sort_unstable();
    taken
}

/// Episodes for `turns`: [`segment_at`] with an episode also starting at every
/// topic boundary found inside a unit. Similarity is over the turns' own
/// text, without speaker names, which would make every turn resemble its
/// speaker's others.
pub async fn segment_topics(
    turns: &[Turn],
    embedder: &dyn Embedder,
    topics: &TopicConfig,
    cfg: &SegmentConfig,
) -> Result<Vec<EpisodeDraft>> {
    let mut starts: BTreeSet<usize> = BTreeSet::new();
    let mut lo = 0usize;
    while lo < turns.len() {
        let hi = (lo..turns.len())
            .find(|&j| turns[j].unit != turns[lo].unit)
            .unwrap_or(turns.len());
        let texts: Vec<&str> = turns[lo..hi].iter().map(|t| t.text.as_str()).collect();
        let key: Vec<usize> = key_utterances(&texts)
            .into_iter()
            .enumerate()
            .filter_map(|(j, k)| k.then_some(lo + j))
            .collect();
        if key.len() >= 2 * topics.window + 2 {
            let batch: Vec<String> = key.iter().map(|&j| turns[j].text.clone()).collect();
            let vectors = embedder.embed(&batch).await?;
            if vectors.len() != batch.len() {
                return Err(MyelinError::Store(format!(
                    "m89a topics: the embedder returned {} vectors for {} utterances of unit {:?}",
                    vectors.len(),
                    batch.len(),
                    turns[lo].unit
                )));
            }
            let sims: Vec<f32> = vectors.windows(2).map(|p| cosine(&p[0], &p[1])).collect();
            let depths = depth_scores(&sims, topics.window);
            for gap in topic_boundaries(&depths, topics.sigma_fraction, topics.min_key_per_segment, key.len()) {
                starts.insert(key[gap + 1]);
            }
        }
        lo = hi;
    }
    Ok(segment_at(turns, cfg, &starts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::record::SourceRef;
    use crate::pipeline::ingest::BoundaryReason;

    #[test]
    fn acknowledgements_short_turns_and_restatements_are_not_key() {
        let texts = [
            "Hey Mel! Good to see you!",
            "I went to a LGBTQ support group yesterday and it was powerful.",
            "Wow, that's awesome!",
            "The LGBTQ support group yesterday was powerful.",
            "Thanks!",
            "My kids and I went camping at the lake last weekend and loved it.",
        ];
        assert_eq!(key_utterances(&texts), vec![false, true, false, false, false, true]);
    }

    #[test]
    fn depth_needs_both_windows_and_peaks_at_the_valley() {
        assert!(depth_scores(&[0.9, 0.9, 0.9, 0.9], 2).is_empty());
        let d = depth_scores(&[0.9, 0.9, 0.1, 0.9, 0.9], 2);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].0, 2);
        assert!((d[0].1 - 1.6).abs() < 1e-6);
    }

    #[test]
    fn boundaries_are_deep_local_maxima_spaced_from_each_other_and_the_ends() {
        // One clear valley at gap 4 among ten key utterances.
        let sims = [0.8, 0.8, 0.8, 0.8, 0.1, 0.8, 0.8, 0.8, 0.8];
        let depths = depth_scores(&sims, 2);
        assert_eq!(topic_boundaries(&depths, 0.5, 2, 10), vec![4]);
        // Flat similarity has no valley.
        let flat = depth_scores(&[0.5; 9], 2);
        assert!(topic_boundaries(&flat, 0.5, 2, 10).is_empty());
        // A valley one utterance from the end is too close to it.
        let near_end = [(5usize, 0.2f32), (6, 0.1), (7, 2.0)];
        assert!(topic_boundaries(&near_end, 0.5, 2, 9).is_empty());
    }

    /// An embedder that places each text on the axis of the topic word it
    /// holds.
    struct TopicAxes;
    #[async_trait::async_trait]
    impl Embedder for TopicAxes {
        fn dim(&self) -> u64 {
            2
        }
        fn id(&self) -> &str {
            "topic-axes"
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| if t.to_lowercase().contains("camping") { vec![0.0, 1.0] } else { vec![1.0, 0.0] })
                .collect())
        }
    }

    fn turn(unit: &str, text: &str, i: usize) -> Turn {
        Turn {
            speaker: if i.is_multiple_of(2) { "Caroline".into() } else { "Melanie".into() },
            text: text.into(),
            at: None,
            source: SourceRef::doc(format!("D1:{i}")),
            unit: unit.into(),
        }
    }

    #[tokio::test]
    async fn a_session_is_cut_where_its_topic_changes_and_no_turn_is_lost() {
        let texts = [
            "I went to the support group meeting and it was powerful for me.",
            "The support group sounds like a really good place for you to be.",
            "Yes the support group people were kind and open with their stories.",
            "Wow!",
            "I am glad the support group helped you feel accepted and heard.",
            "My family went camping at the lake last weekend with the kids.",
            "Camping at the lake sounds fun, did the kids enjoy the campfire?",
            "The kids loved camping and roasting marshmallows by the lake.",
            "We should plan camping again next month when the weather is warm.",
        ];
        let turns: Vec<Turn> = texts.iter().enumerate().map(|(i, t)| turn("s1", t, i)).collect();
        let episodes = segment_topics(&turns, &TopicAxes, &TopicConfig::default(), &SegmentConfig::default())
            .await
            .unwrap();
        assert_eq!(episodes.len(), 2);
        assert_eq!(episodes[0].ended_by, BoundaryReason::TopicShift);
        assert!(episodes[1].turns[0].text.starts_with("My family went camping"));
        let total: usize = episodes.iter().map(|e| e.turns.len()).sum();
        assert_eq!(total, turns.len());
    }

    struct Short;
    #[async_trait::async_trait]
    impl Embedder for Short {
        fn dim(&self) -> u64 {
            2
        }
        fn id(&self) -> &str {
            "short"
        }
        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![1.0, 0.0]])
        }
    }

    #[tokio::test]
    async fn an_embedder_that_returns_the_wrong_count_is_refused() {
        let texts = [
            "My sister started a pottery class downtown this spring.",
            "The garden tomatoes finally ripened after all that rain.",
            "Our basketball team won the regional tournament on Sunday.",
            "I adopted a rescue puppy named Biscuit from the shelter.",
            "The new library branch opens near the train station soon.",
            "Grandma shared her secret lasagna recipe at dinner.",
            "We repainted the kitchen cabinets a pale shade of green.",
            "Traffic on the bridge made everyone late for the concert.",
        ];
        let turns: Vec<Turn> = texts.iter().enumerate().map(|(i, t)| turn("s1", t, i)).collect();
        assert!(segment_topics(&turns, &Short, &TopicConfig::default(), &SegmentConfig::default())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn a_short_session_gets_no_topic_boundary_and_needs_no_embedding() {
        let turns: Vec<Turn> = (0..4)
            .map(|i| turn("s1", "a substantive sentence about the weekend plans", i))
            .collect();
        let episodes = segment_topics(&turns, &Short, &TopicConfig::default(), &SegmentConfig::default())
            .await
            .unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].ended_by, BoundaryReason::EndOfStream);
    }
}
