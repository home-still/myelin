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

/// Modal words that ask what is likely rather than what was recorded.
const INFERENCE_MODALS: [&str; 5] = ["likely", "might", "would", "could", "probably"];

/// Words by which the user addresses the assistant or speaks of themself.
const USER_PERSON_WORDS: [&str; 8] = ["i", "i'm", "i've", "i'd", "me", "my", "you", "your"];

/// Lowercased words of `text`, apostrophes kept inside a word ("i'm").
fn person_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .replace('\u{2019}', "'")
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// Does the question ask what is *likely*: a modal (likely, might, would,
/// could, probably) that is not addressed to the assistant ("could you",
/// "you would")?
///
/// LoCoMo's open-domain category asks for an inference from what the
/// conversation shows ("Would Caroline still want to pursue counseling…?",
/// gold "Likely no"; Maharana et al. 2024, `10.18653/v1/2024.acl-long.747`).
/// Measured on every question before any row (2026-09-29): it fires on 42 of
/// LoCoMo's 96 open-domain questions, 1 of 841 single-hop, and none of the 446
/// adversarial, multi-hop or temporal. On LongMemEval_S it fires on 5
/// preference questions, 1 multi-session, and none of the 30 abstention
/// traps.
pub fn is_inference_question(text: &str) -> bool {
    let words = person_words(text);
    words.iter().enumerate().any(|(i, w)| {
        INFERENCE_MODALS.contains(&w.as_str())
            && !(i > 0 && words[i - 1] == "you")
            && words.get(i + 1).is_none_or(|next| next != "you")
    })
}

/// M84: does the question ask for advice or an inference rather than for a
/// recorded fact (`docs/measurements/m84-non-recall.md`)?
///
/// Two shapes:
/// - an advice request ([`is_advice_request`]) that the user makes of the
///   assistant, in the first or second person. LoCoMo asks recall questions
///   *about* advice in the third person ("What advice did Calvin receive…?"),
///   and 12 of its adversarial questions carry an advice cue; none is in the
///   first or second person;
/// - an inference question ([`is_inference_question`]).
///
/// A decline is the wrong response to either: abstention is for questions
/// the memories cannot answer, and refusing a request that calls for a
/// response is over-abstention (Wen et al. 2024, "Know Your Limits", TACL,
/// `10.1162/tacl_a_00754`; Brahman et al. 2024, "The Art of Saying No",
/// `10.52202/079017-1573`). Measured before any row: 30 of LongMemEval_S's 30
/// preference questions and 1 multi-session question; 42 LoCoMo open-domain
/// and 1 single-hop. It matches no LongMemEval_S abstention trap and no
/// LoCoMo adversarial question.
pub fn is_non_recall_request(text: &str) -> bool {
    let addressed = person_words(text).iter().any(|w| USER_PERSON_WORDS.contains(&w.as_str()));
    (is_advice_request(text) && addressed) || is_inference_question(text)
}

/// M87: cues that a question asks for a list outright.
const ENUMERATION_CUES: [&str; 4] = ["what are some", "what kinds of", "what types of", "what sorts of"];
/// How far after "what"/"which" the head noun may sit before the verb.
const ENUMERATION_SPAN_MAX_WORDS: usize = 6;
/// The verb that ends a wh-phrase ("What countries **has** …").
const AUXILIARIES: [&str; 10] = ["has", "have", "had", "did", "does", "do", "are", "were", "is", "was"];
/// Plural agreement after an empty wh-phrase ("What **are** Nate's hobbies?").
const PLURAL_AUXILIARIES: [&str; 2] = ["are", "were"];
/// Subjects that make "what are" a question about a person, not a list.
const PERSONAL_SUBJECTS: [&str; 4] = ["i", "you", "we", "they"];
/// "kind of", "type of": the noun after `of` is the head, not these.
const KIND_WORDS: [&str; 6] = ["kind", "kinds", "type", "types", "sort", "sorts"];
/// Plurals that do not end in `s`.
const IRREGULAR_PLURALS: [&str; 4] = ["children", "people", "men", "women"];
/// Words ending in `s` that are not plurals.
const NON_PLURAL_S: [&str; 14] = [
    "news", "series", "species", "lens", "always", "perhaps", "sometimes", "towards", "afterwards",
    "besides", "whereas", "across", "plus", "yes",
];
/// The shortest regular plural the test accepts ("pets", "toys").
const PLURAL_MIN_CHARS: usize = 4;

fn is_plural_noun(word: &str) -> bool {
    if !word.chars().all(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    IRREGULAR_PLURALS.contains(&word)
        || (word.len() >= PLURAL_MIN_CHARS
            && word.ends_with('s')
            && !word.ends_with("ss")
            && !word.ends_with("us")
            && !word.ends_with("is")
            && !NON_PLURAL_S.contains(&word))
}

/// M87: does the question ask for a list of things
/// (`docs/measurements/m87-enumeration-depth.md`)?
///
/// "What European countries has Maria been to?" and "What activities has
/// Melanie done with her family?" need every mention across sessions, as a
/// count does ([`is_aggregation_question`]). On the shipped LoCoMo run, 80
/// of 83 multi-hop losses missed at least one gold turn, and 58 of the 83
/// asked for a list or a count.
///
/// Deterministic and auditable, like the aggregation cues. The question is
/// a list when:
/// - it carries a list cue ("what are some", "what kinds of", …), or
/// - the words between its first "what"/"which" and the first auxiliary
///   (within [`ENUMERATION_SPAN_MAX_WORDS`]) end on a plural head noun. The
///   head is the word before `of`, unless that is a kind word ("kind of
///   place"), in which case it is the span's last word. Or
/// - the auxiliary follows the wh-word directly, is plural ("are", "were"),
///   and is not followed by a personal subject ("What are Nate's hobbies?").
///
/// Grounds: JustMem types list operations and composes for them (Chen et al.
/// 2026, arXiv 2609.19877); MemPro's adaptive retrieval depth (arXiv
/// 2606.00619, +1.34). Depth stays question-gated because length alone costs
/// accuracy even with perfect retrieval (Du et al. 2025,
/// `10.18653/v1/2025.findings-emnlp.1264`).
pub fn is_enumeration_question(text: &str) -> bool {
    let lower = text.to_lowercase();
    if ENUMERATION_CUES.iter().any(|c| lower.contains(c)) {
        return true;
    }
    let words = person_words(text);
    let Some(wh) = words.iter().position(|w| w == "what" || w == "which") else {
        return false;
    };
    let after = &words[wh + 1..];
    let Some(aux) = after
        .iter()
        .take(ENUMERATION_SPAN_MAX_WORDS + 1)
        .position(|w| AUXILIARIES.contains(&w.as_str()))
    else {
        return false;
    };
    let span = &after[..aux];
    if span.is_empty() {
        return PLURAL_AUXILIARIES.contains(&after[aux].as_str())
            && after
                .get(aux + 1)
                .is_some_and(|next| !PERSONAL_SUBJECTS.contains(&next.as_str()));
    }
    let head = match span.iter().position(|w| w == "of") {
        Some(of) if of > 0 && !KIND_WORDS.contains(&span[of - 1].as_str()) => &span[of - 1],
        _ => &span[span.len() - 1],
    };
    is_plural_noun(head)
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
    fn inference_and_advice_requests_are_not_recall() {
        for q in [
            "Would Caroline still want to pursue counseling as a career if she hadn't received support growing up?",
            "What would Caroline's political leaning likely be?",
            "What might John's degree be in?",
            "I'm planning my meal prep next week, any suggestions for new recipes?",
            "I've been sneezing quite a bit lately. Do you think it might be my living room?",
        ] {
            assert!(is_non_recall_request(q), "{q}");
        }
    }

    #[test]
    fn recall_questions_and_advice_about_others_are_recall() {
        for q in [
            "What advice did Calvin receive from the chef at the music festival?",
            "I was wondering if you could remind me of the name of that restaurant you recommended.",
            "Could you tell me what I bought 10 days ago?",
            "What did Caroline realize after her charity race?",
            "How many plants did I initially plant for tomatoes and chili peppers?",
        ] {
            assert!(!is_non_recall_request(q), "{q}");
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

    #[test]
    fn list_questions_are_enumerations() {
        for q in [
            "What European countries has Maria been to?",
            "What activities has Melanie done with her family?",
            "Which of James's family members has he visited?",
            "What are Nate's favorite desserts?",
            "What are some changes Caroline has faced?",
            "What kinds of books does Joanna read?",
            "What instruments does Melanie play?",
            "Which pets do Andrew and Audrey have?",
            "What sports did the children try?",
        ] {
            assert!(is_enumeration_question(q), "{q}");
        }
    }

    #[test]
    fn single_things_and_personal_questions_are_not() {
        for q in [
            "What book did Melanie read from Caroline's suggestion?",
            "What kind of place does Caroline want to create for people?",
            "What percentage of packed shoes did I wear?",
            "Which pair of shoes did I buy?",
            "What is Caroline's identity?",
            "What did Caroline research?",
            "What are you planning for the weekend?",
            "What lens does John use?",
            "Where did Jon go?",
            "What does Caroline's necklace symbolize?",
        ] {
            assert!(!is_enumeration_question(q), "{q}");
        }
    }
}
