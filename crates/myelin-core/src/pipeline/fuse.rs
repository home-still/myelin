//! Rank fusion (`PLAN.md` §5.2).
//!
//! Fusion is **ours, not Qdrant's**. Qdrant's server-side RRF uses `k = 1`,
//! measured: single-list ranks 1..6 score `0.5, 0.33333334, 0.25, 0.2,
//! 0.16666667, 0.14285715` = `1/(1 + rank)`, pinned by
//! `tests/qdrant_capability.rs::server_side_rrf_uses_k_equals_one`.
//!
//! `k = 1` is aggressively top-rank biased. That is the wrong default when one
//! channel is systematically better than the other — and per §2 finding 1 it
//! is: MemPro's ablation drops LoCoMo 84.93 → 72.25 without BM25 but only to
//! 82.57 without the dense embedder. With `k = 1` a single list's rank-1 hit
//! (0.5) outweighs a document ranked 2nd in *both* lists (0.333 + 0.333), so
//! agreement between channels is worth less than a single confident channel.
//! Cormack's `k = 60` flattens the head enough that cross-channel agreement
//! wins, which is the behaviour hybrid retrieval is for.
//!
//! Both are reachable: `k` is a parameter so the harness can A/B `k ∈ {1, 60}`
//! as an ablation (M4), and the chosen value is pinned into
//! [`crate::store::export::MemoryConfigJson`] so a built memory records which
//! one produced it.

use std::collections::HashMap;

use uuid::Uuid;

/// Cormack/Clarke/Buettcher's canonical constant.
pub const DEFAULT_RRF_K: f32 = 60.0;
/// What Qdrant does server-side. Kept named so the ablation reads clearly.
pub const QDRANT_RRF_K: f32 = 1.0;

/// One channel's ranking, best first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedList {
    pub channel: &'static str,
    pub ids: Vec<Uuid>,
}

impl RankedList {
    pub fn new(channel: &'static str, ids: Vec<Uuid>) -> Self {
        Self { channel, ids }
    }
}

/// Reciprocal rank fusion: `score(d) = Σ_l 1 / (k + rank_l(d))`, rank 1-based.
///
/// Ties break on id so the output is deterministic — a benchmark that reorders
/// equal-scored documents between runs produces unreproducible numbers.
pub fn rrf(lists: &[RankedList], k: f32) -> Vec<(Uuid, f32)> {
    let mut scores: HashMap<Uuid, f32> = HashMap::new();
    for list in lists {
        for (i, id) in list.ids.iter().enumerate() {
            let rank = (i + 1) as f32;
            *scores.entry(*id).or_insert(0.0) += 1.0 / (k + rank);
        }
    }
    let mut out: Vec<(Uuid, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    /// Reproduces the measured Qdrant series exactly. If this drifts, our
    /// model of what the server does is wrong.
    #[test]
    fn k_equals_one_reproduces_the_measured_qdrant_series() {
        let ids: Vec<Uuid> = (1..=6).map(id).collect();
        let fused = rrf(&[RankedList::new("dense", ids)], QDRANT_RRF_K);
        let got: Vec<f32> = fused.iter().map(|(_, s)| *s).collect();
        let want = [0.5, 0.333_333_34, 0.25, 0.2, 0.166_666_67, 0.142_857_15];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "got {got:?}, want {want:?}");
        }
    }

    /// The reason we fuse client-side, stated as a test.
    ///
    /// `A` is rank 1 in one channel and absent from the other. `B` is deep in
    /// both (rank 5 and rank 6). The two constants **order these oppositely**:
    ///
    /// ```text
    /// k = 1   A = 1/2   = 0.5000    B = 1/6  + 1/7  = 0.3095   -> A first
    /// k = 60  A = 1/61  = 0.0164    B = 1/65 + 1/66 = 0.0305   -> B first
    /// ```
    ///
    /// That inversion is the whole reason fusion is ours. `k = 1` says "one
    /// channel was very confident"; `k = 60` says "both channels agree". Per
    /// §2 finding 1 the channels are not interchangeable, so which of those we
    /// mean has to be a decision we own, not a server default.
    #[test]
    fn k_inverts_the_ordering_of_confidence_versus_agreement() {
        let (a, b) = (id(0xA), id(0xB));
        let filler: Vec<Uuid> = (0x10..0x20).map(id).collect();
        let lists = [
            RankedList::new("dense", vec![a, filler[0], filler[1], filler[2], b]),
            RankedList::new(
                "lex",
                vec![filler[3], filler[4], filler[5], filler[6], filler[7], b],
            ),
        ];

        let at_1 = rrf(&lists, QDRANT_RRF_K);
        assert_eq!(
            at_1[0].0, a,
            "k=1 must favour the single confident channel: {at_1:?}"
        );

        let at_60 = rrf(&lists, DEFAULT_RRF_K);
        assert_eq!(
            at_60[0].0, b,
            "k=60 must favour cross-channel agreement: {at_60:?}"
        );
    }

    /// The mechanism behind that inversion: `k` controls how fast score decays
    /// with rank. At k = 1 rank 1 is worth 1.5x rank 2; at k = 60 it is worth
    /// 1.02x, so rank barely matters and membership dominates.
    #[test]
    fn k_controls_rank_decay() {
        let ids: Vec<Uuid> = (1..=2).map(id).collect();
        let one = rrf(&[RankedList::new("dense", ids.clone())], QDRANT_RRF_K);
        let sixty = rrf(&[RankedList::new("dense", ids)], DEFAULT_RRF_K);

        let decay_1 = one[0].1 / one[1].1;
        let decay_60 = sixty[0].1 / sixty[1].1;
        assert!(
            (decay_1 - 1.5).abs() < 1e-5,
            "k=1 rank decay should be 1.5, got {decay_1}"
        );
        assert!(
            (decay_60 - 62.0 / 61.0).abs() < 1e-5,
            "k=60 rank decay should be 62/61, got {decay_60}"
        );
        assert!(decay_60 < decay_1);
    }

    /// A document in neither list scores nothing; a document in every list
    /// beats one in a single list at equal rank.
    #[test]
    fn agreement_beats_a_lone_channel_at_equal_rank() {
        let (a, b) = (id(1), id(2));
        let fused = rrf(
            &[
                RankedList::new("dense", vec![a, b]),
                RankedList::new("lex", vec![b]),
            ],
            DEFAULT_RRF_K,
        );
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, b, "two-channel hit must lead: {fused:?}");
    }

    #[test]
    fn empty_input_fuses_to_nothing() {
        assert!(rrf(&[], DEFAULT_RRF_K).is_empty());
        assert!(rrf(&[RankedList::new("dense", vec![])], DEFAULT_RRF_K).is_empty());
    }

    /// Determinism: equal scores must not reorder between runs, or repeated
    /// benchmark runs disagree for no reason.
    #[test]
    fn ties_break_deterministically() {
        let ids: Vec<Uuid> = (1..=5).map(id).collect();
        let lists = [RankedList::new("dense", ids.clone())];
        let first = rrf(&lists, DEFAULT_RRF_K);
        for _ in 0..8 {
            assert_eq!(rrf(&lists, DEFAULT_RRF_K), first);
        }
    }
}
