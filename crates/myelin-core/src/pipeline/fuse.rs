//! Rank fusion (`PLAN.md` §5.2).
//!
//! Fusion is **ours, not Qdrant's**. Qdrant's server-side RRF uses `k = 1`,
//! measured: single-list ranks 1..6 score `0.5, 0.33333334, 0.25, 0.2,
//! 0.16666667, 0.14285715` = `1/(1 + rank)`, pinned by
//! `tests/qdrant_capability.rs::server_side_rrf_uses_k_equals_one`.
//!
//! **The default is `k = 1`, and that is a measurement overruling a
//! prediction.**
//!
//! The prediction, written here first, was that `k = 60` should win. `k = 1`
//! is aggressively top-rank biased: a single list's rank-1 hit (0.5)
//! outweighs a document ranked 2nd in *both* lists (0.333 + 0.333), so
//! agreement between channels is worth less than one confident channel.
//! Cormack's 60 flattens the head until cross-channel agreement wins, which
//! sounded like the behaviour hybrid retrieval is for.
//!
//! M4 measured the opposite, on 997 dev questions and confirmed on 985
//! held-out ones (`docs/measurements/m4-ablation.md`):
//!
//! | arm | dev recall@6 | holdout |
//! |---|---|---|
//! | `hybrid`, k=60 | 0.8252 | 0.7830 |
//! | `hybrid`, k=1 | **0.8815** | **0.8579** |
//!
//! The mechanism is the same asymmetry the prediction invoked, pointing the
//! other way. Cormack tuned 60 for fusing TREC runs of *comparable* quality,
//! where flattening lets agreement arbitrate. Our channels are not
//! comparable — BM25 alone scores 0.8666 and dense alone 0.7314 — so
//! flattening the head averages a strong ranking with a weak one and drags
//! the good hits down. Sharpening keeps BM25's confident head and lets dense
//! contribute only where it is also confident.
//!
//! Two consequences worth stating plainly. Qdrant's server-side RRF already
//! uses `k = 1`, so the argument that we *must* fuse client-side to escape
//! it is gone; what remains is R4 — `rrf_k` has to be a query parameter, and
//! the ablation needs both lists unfused. And `k` stays a parameter with
//! both constants named, because a corpus whose channels are evenly matched
//! would flip this back.
//!
//! The chosen value is pinned into
//! [`crate::store::export::MemoryConfigJson`] so a built memory records
//! which one produced it.

use std::collections::HashMap;

use uuid::Uuid;

/// Measured best on LoCoMo (M4), and what Qdrant does server-side.
pub const DEFAULT_RRF_K: f32 = 1.0;
/// Cormack/Clarke/Buettcher's canonical constant. Kept named because it is
/// the right default for evenly-matched channels, and because the ablation
/// reads clearly with both spelled out.
pub const CORMACK_RRF_K: f32 = 60.0;

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
        let fused = rrf(&[RankedList::new("dense", ids)], DEFAULT_RRF_K);
        let got: Vec<f32> = fused.iter().map(|(_, s)| *s).collect();
        let want = [0.5, 0.333_333_34, 0.25, 0.2, 0.166_666_67, 0.142_857_15];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "got {got:?}, want {want:?}");
        }
    }

    /// What `k` actually controls, stated as a test.
    ///
    /// `A` is rank 1 in one channel and absent from the other. `B` is deep in
    /// both (rank 5 and rank 6). The two constants **order these oppositely**:
    ///
    /// ```text
    /// k = 1   A = 1/2   = 0.5000    B = 1/6  + 1/7  = 0.3095   -> A first
    /// k = 60  A = 1/61  = 0.0164    B = 1/65 + 1/66 = 0.0305   -> B first
    /// ```
    ///
    /// `k = 1` says "one channel was very confident"; `k = 60` says "both
    /// channels agree". M4 measured which of those is right *for this
    /// corpus* and the answer was confidence, by +5.6 points on dev and
    /// +7.5 on holdout — because BM25 and dense are far apart here (0.8666
    /// vs 0.7314), so averaging them drags the strong ranking down. The
    /// inversion this test pins is the mechanism behind that result, and it
    /// is why `k` stays a parameter: evenly-matched channels would flip it.
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

        let at_1 = rrf(&lists, DEFAULT_RRF_K);
        assert_eq!(
            at_1[0].0, a,
            "k=1 must favour the single confident channel: {at_1:?}"
        );

        let at_60 = rrf(&lists, CORMACK_RRF_K);
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
        let one = rrf(&[RankedList::new("dense", ids.clone())], DEFAULT_RRF_K);
        let sixty = rrf(&[RankedList::new("dense", ids)], CORMACK_RRF_K);

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
