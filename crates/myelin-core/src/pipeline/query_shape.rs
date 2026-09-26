//! M72: what shape a question has, decided without a model.
//!
//! A counting or summing question ("how many doctors did I visit?", "how
//! much did I spend on workshops in total?") needs **every** mention across
//! sessions, not the best one. On the shipped LongMemEval_S run, 26 of the 31
//! multi-session questions LongMemEval's own grader marked wrong were of this
//! shape. Most held only part of their gold turns and undercounted: 1 of 5
//! mentions retrieved gave "2" against a gold of 4 (loss anatomy, 2026-09-25,
//! `docs/measurements/m72-aggregation-depth.md`).
//!
//! The mechanism this feeds is question-shape-aware retrieval depth:
//! - MemPro's "adaptive retrieval depth" iteration (Liu et al. 2026, arXiv
//!   2606.00619, App. A.1) gained +1.34;
//! - JustMem's planner types an aggregate operation and composes for it
//!   (Chen et al. 2026, arXiv 2609.19877). Its aggregation questions went
//!   65.96 → 78.72 on LongMemEval_S.
//!
//! Here the type is a deterministic cue list, not a planner call. `recall`
//! allows no model in its loop (`PLAN.md` §7.1), and a cue list is auditable.

use crate::time::is_interval_question;

/// Cues that a question counts or sums across occurrences.
const AGGREGATION_CUES: [&str; 6] = [
    "how many ",
    "how much ",
    " in total",
    "total ",
    "altogether",
    "combined",
];

/// Does the question count or sum across occurrences?
///
/// An elapsed-time question ("how many days since…") is excluded: it has one
/// answer between two events, and [`is_interval_question`] already routes it
/// to the dated timeline. A question that sums durations across occurrences
/// still counts when its unit is not one of those cues ("how many hours of
/// jogging and yoga…").
pub fn is_aggregation_question(text: &str) -> bool {
    let lower = text.to_lowercase();
    AGGREGATION_CUES.iter().any(|c| lower.contains(c)) && !is_interval_question(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_and_summing_questions_are_aggregations() {
        for q in [
            "How many different doctors did I visit?",
            "How many musical instruments do I currently own?",
            "How much total money did I spend on attending workshops in the last four months?",
            "How many hours of jogging and yoga did I do last week?",
            "What is the combined cost of the two tickets?",
        ] {
            assert!(is_aggregation_question(q), "{q}");
        }
    }

    #[test]
    fn elapsed_time_and_single_fact_questions_are_not() {
        for q in [
            "How many days ago did I watch the Super Bowl?",
            "How long have I been working at my current job?",
            "What brand of shampoo do I currently use?",
            "When did Caroline go to the LGBTQ support group?",
        ] {
            assert!(!is_aggregation_question(q), "{q}");
        }
    }
}
