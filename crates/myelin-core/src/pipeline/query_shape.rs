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

/// Cues that a question asks for advice, a recommendation or an opinion.
const ADVICE_CUES: [&str; 12] = [
    "recommend",
    "suggest",
    "tips",
    "ideas",
    "advice",
    "what should i",
    "can you help",
    "could you help",
    "help me",
    "i'm looking for",
    "i am looking for",
    "do you think",
];

/// Cues that the question asks to recall advice already given, not for new
/// advice ("the hostel you recommended", "remind me of…").
const RECALL_CUES: [&str; 11] = [
    "you recommended",
    "you suggested",
    "you mentioned",
    "you told me",
    "you gave me",
    "you provided",
    "remind me",
    "our previous",
    "our last",
    "we discussed",
    "we talked",
];

/// Does the question ask for new advice, a recommendation or an opinion?
///
/// M20b's gate for the ranked `[profile]` block. A request for advice is
/// where the user's own stated preferences decide the answer: PrefEval
/// (Zhao et al. 2025, arXiv 2502.09597) finds preference following under
/// 10% zero-shot and best when the stated preference is retrieved into
/// context. Memora (`10.18653/v1/2026.findings-acl.1337`) and AlpsBench
/// (`10.1145/3805712.3808634`) find that injected profiles bias answers
/// elsewhere, so the block is gated here and not composed for every
/// question.
///
/// Measured on LongMemEval_S before any row (2026-09-26): it fires on 29 of
/// the 30 `single-session-preference` questions and on none of the other
/// 470. The 14 `single-session-assistant` questions that ask "what was the
/// hostel you recommended" are excluded by [`RECALL_CUES`].
pub fn is_advice_request(text: &str) -> bool {
    let lower = text.to_lowercase();
    ADVICE_CUES.iter().any(|c| lower.contains(c)) && !RECALL_CUES.iter().any(|c| lower.contains(c))
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

    #[test]
    fn requests_for_advice_are_advice_requests() {
        for q in [
            "Can you recommend some resources where I can learn more about video editing?",
            "I'm planning my meal prep next week, any suggestions for new recipes?",
            "I'm a bit anxious about getting around Tokyo. Do you have any helpful tips?",
            "I've been feeling nostalgic lately. Do you think it would be a good idea to attend my high school reunion?",
        ] {
            assert!(is_advice_request(q), "{q}");
        }
    }

    #[test]
    fn recalling_old_advice_and_plain_facts_are_not() {
        for q in [
            "Can you remind me of the name of the romantic Italian restaurant in Rome you recommended for dinner?",
            "I'm looking back at our previous conversation about building a cocktail bar. You recommended five bottles.",
            "How many different doctors did I visit?",
            "What kitchen appliance did I buy 10 days ago?",
        ] {
            assert!(!is_advice_request(q), "{q}");
        }
    }
}
