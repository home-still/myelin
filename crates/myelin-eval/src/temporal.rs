//! Date-aware deterministic scoring for temporal answers.
//!
//! # Why this exists
//!
//! `docs/measurements/m13-temporal-axis.md` closed with a directive: *"Any
//! future temporal work should fix the scorer before it fixes the reader, or
//! it will keep measuring string overlap against `The sunday before 25 May
//! 2023`"*. Token F1 ([`crate::bench::token_f1`]) compares *words*. On the
//! LoCoMo category-2 stratum — the temporal one — that means an answer of
//! `2023-06-27` against a gold of `The week before 27 June 2023` earns 0.50
//! for naming the anchor it was asked to offset *from*, and an answer of
//! `2023-05-03` against `first week of May 2023` earns 0.25 for being right.
//!
//! So the direction of the fix is the opposite of the intuition: token F1 was
//! not under-crediting correct dates, it was handing out **partial credit for
//! date-shaped near-misses**.
//!
//! # The model: closed intervals of whole days
//!
//! The grammar itself lives in [`myelin_core::time`] — M19 moved it there so
//! the read path could resolve relative references against a record's own date
//! ([`myelin_core::time::resolve_relative`]) through the same parser that
//! grades the answers. This module is the scoring half: every temporal
//! expression resolves to either a [`DayRange`] — a closed interval of whole
//! days — or a [`Temporal::Duration`], a span with no anchor canonicalised to
//! days, and scoring is then arithmetic on days rather than overlap on tokens.
//!
//! ```text
//! score = |response ∩ gold| / |response|
//! ```
//!
//! is **precision against the gold interval, not F1**. That asymmetry is
//! load-bearing and must not be swapped for a symmetric measure:
//!
//! - gold `June 2023`, response `2023-06-15` → `1/1` = **1.0**. Gold's
//!   granularity is the limit of what the corpus knows; a more precise answer
//!   consistent with it is correct. Symmetric F1 over days would score this
//!   0.065 and reintroduce exactly the failure M13 found.
//! - gold `7 May 2023`, response `May 2023` → `1/31` = **0.032**. A vaguer
//!   answer than the question admits is penalised — which is why precision
//!   and not recall.
//! - vagueness cannot game it upward: `2023` against gold `June 2023` scores
//!   `30/365`.
//!
//! # The grammar fails closed, and so does this
//!
//! An expression [`myelin_core::time`]'s grammar does not cover parses as
//! `None`, and [`temporal_score`] then returns `None` so its caller keeps
//! token F1 — the safe direction. The count of `None` golds *within* the
//! temporal stratum is reported in `docs/measurements/m14-temporal-scorer.md`
//! as the grammar's own coverage limit, rather than hidden.

use chrono::Datelike;
use myelin_core::time::{parse_gold, parse_response};

pub use myelin_core::time::{DayRange, Temporal, TemporalKind};

/// Score a response against a gold answer on the day line.
///
/// `None` when the gold answer is not temporal — the caller then keeps token
/// F1. See the module docs for why the interval score is precision and not F1.
pub fn temporal_score(response: &str, gold: &str) -> Option<(f64, TemporalKind)> {
    match parse_gold(gold)? {
        Temporal::Range(g) => {
            let Some(Temporal::Range(r)) = parse_response(response, Some(g.lo.year())) else {
                // Gold names a time and the answer names none: no credit, and
                // no fallback to string overlap, which is the thing being
                // fixed.
                return Some((0.0, TemporalKind::Interval));
            };
            Some((
                g.overlap(r) as f64 / r.days() as f64,
                TemporalKind::Interval,
            ))
        }
        Temporal::Duration(g) => {
            let d = match parse_response(response, None) {
                Some(Temporal::Duration(d)) => d,
                _ => return Some((0.0, TemporalKind::Duration)),
            };
            // Exact on canonical days: `three months` == `3 months` ==
            // `nearly three months` == 90, and `19 days` != `11 days`.
            Some((f64::from(u8::from(d == g)), TemporalKind::Duration))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_score_is_precision_against_gold() {
        // A more precise answer consistent with a coarser gold is correct.
        let (s, k) = temporal_score("2023-06-15", "June 2023").unwrap();
        assert_eq!(k, TemporalKind::Interval);
        assert!((s - 1.0).abs() < 1e-12, "got {s}");
        // A vaguer answer than the question admits is penalised.
        let (s, _) = temporal_score("May 2023", "7 May 2023").unwrap();
        assert!((s - 1.0 / 31.0).abs() < 1e-12, "got {s}");
        // Vagueness cannot game it upward.
        let (s, _) = temporal_score("2023", "June 2023").unwrap();
        assert!((s - 30.0 / 365.0).abs() < 1e-12, "got {s}");
        // The headline: the anchor is not the offset.
        let (s, _) = temporal_score("2023-06-27", "The week before 27 June 2023").unwrap();
        assert_eq!(s, 0.0);
        // A gold that names a time and an answer that names none: no credit,
        // and no fallback to string overlap.
        let (s, _) = temporal_score("sometime in the spring", "7 May 2023").unwrap();
        assert_eq!(s, 0.0);
    }

    #[test]
    fn duration_score_is_exact_on_canonical_days() {
        assert_eq!(
            temporal_score("3 months", "nearly three months"),
            Some((1.0, TemporalKind::Duration))
        );
        assert_eq!(
            temporal_score("11 days", "19 days"),
            Some((0.0, TemporalKind::Duration))
        );
        // A duration gold against a date answer earns nothing, not a fallback.
        assert_eq!(
            temporal_score("May 2023", "19 days"),
            Some((0.0, TemporalKind::Duration))
        );
    }

    #[test]
    fn non_temporal_gold_yields_none_so_the_caller_keeps_token_f1() {
        assert_eq!(temporal_score("Boston", "Boston"), None);
        assert_eq!(temporal_score("2023-05-01", "Tokyo"), None);
    }
}
