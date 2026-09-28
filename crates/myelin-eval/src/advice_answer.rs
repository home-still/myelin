//! M78: an advice request answered as structure, grounded in the user's own
//! words (`docs/measurements/m78-advice-answer-structure.md`).
//!
//! # Why structure
//!
//! On LongMemEval_S's 30 preference questions the reader holds the user's own
//! sentence on 28 of them (M76) and still declines about seven times a seed.
//! It cites `READER_SYSTEM`'s "reply exactly: I don't know" (M76b). M77's
//! clause changed *what* its advice says but hardly *whether* it answers:
//! declines went from 8.3 to 6.3 a seed. This project's one law is that the
//! reader obeys structure where it ignores instructions (M19, M40, M42, M43).
//!
//! So the final answer to an advice request is a schema with no place to
//! decline:
//! - the user's stated preferences, each quoted from a memory it cites;
//! - then a recommendation.
//!
//! The thinking trace stays free. Only the content after it is constrained,
//! because format constraints on the reasoning itself cost accuracy (Tam et
//! al. 2024, *Let Me Speak Freely?*, arXiv 2408.02442).
//!
//! The order follows *Attribute First, then Generate* (Slobodkin et al. 2024,
//! ACL, arXiv 2403.17104): pick the source spans, then write from them.
//! PrefEval (Zhao et al. 2025, `10.48550/arxiv.2502.09597`) finds that a
//! retrieved preference plus a reminder is what makes an answer follow the
//! preference. Its "unhelpful" failure, refusing for a perceived lack of
//! context, is the decline this removes.
//!
//! # The check is in code
//!
//! A quoted preference is kept only if the quote occurs, case- and
//! whitespace-insensitively, inside a *user* turn of the memory it cites. That
//! is an independent check, not the model's word, in the spirit of ALCE's
//! citation check (Gao et al. 2023, `10.18653/v1/2023.emnlp-main.398`) and M71b's
//! `accept_grounded`. The rendered answer opens "You told me", so a quote from
//! the assistant's turn would be a false statement, and it is dropped.
//!
//! # Field order is the wire order
//!
//! llama.cpp generates fields in the order the schema's properties arrive, and
//! serde_json here sends them alphabetically
//! (`docs/measurements/defect-2026-09-28-schema-field-order.md`). The names
//! are chosen so the alphabetical order is the intended one:
//! `preferences` < `recommendation`, and `memory` < `quote`. A test pins the
//! order on the serialized schema.

use serde::Deserialize;

/// The most preferences an answer may quote.
pub const ADVICE_PREFERENCES_MAX: usize = 4;
/// A quote's ceiling. A stated preference is one sentence. This stays well
/// under the lengths llama.cpp's grammar compiler refuses (`build.rs`).
pub const ADVICE_QUOTE_MAX_CHARS: usize = 160;
/// A quote's floor, so that "I" or "the" cannot pass as a stated preference.
pub const ADVICE_QUOTE_MIN_CHARS: usize = 12;
/// The recommendation's ceiling.
pub const ADVICE_RECOMMENDATION_MAX_CHARS: usize = 600;
/// Completion room after the trace: four quotes and a recommendation at about
/// 3.5 characters a token, with room for the JSON around them.
pub const ADVICE_ANSWER_TOKENS: u32 = 512;
/// The turn prefix of the user's own words, as `ingest` writes it.
const USER_TURN_PREFIX: &str = "user: ";
/// The speakers of a LongMemEval_S episode.
const SPEAKERS: [&str; 2] = ["user", "assistant"];
/// Opens the rendered answer.
const YOU_TOLD_ME: &str = "You told me";

/// The constrained answer, as the model writes it.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct AdviceAnswer {
    pub preferences: Vec<StatedPreference>,
    pub recommendation: String,
}

/// One stated preference: the memory it is quoted from (the `[n]` the reader
/// was shown) and the quote.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct StatedPreference {
    pub memory: usize,
    pub quote: String,
}

/// The schema for an advice answer over `memories` shown memories.
///
/// There is no decline field. At least one preference is required, and a
/// memory index must be one the reader was shown.
pub fn advice_answer_schema(memories: usize) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "preferences": {
                "type": "array",
                "minItems": 1,
                "maxItems": ADVICE_PREFERENCES_MAX,
                "items": {
                    "type": "object",
                    "properties": {
                        "memory": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": memories.saturating_sub(1)
                        },
                        "quote": {
                            "type": "string",
                            "minLength": ADVICE_QUOTE_MIN_CHARS,
                            "maxLength": ADVICE_QUOTE_MAX_CHARS
                        }
                    },
                    "required": ["memory", "quote"],
                    "additionalProperties": false
                }
            },
            "recommendation": {
                "type": "string",
                "minLength": 1,
                "maxLength": ADVICE_RECOMMENDATION_MAX_CHARS
            }
        },
        "required": ["preferences", "recommendation"],
        "additionalProperties": false
    })
}

/// Lowercase, with every run of whitespace collapsed to one space.
fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// A turn's text after the labels `compose` puts in front of an item
/// (`[2023-05-24] `, `[untrusted source] `, and so on).
fn strip_labels(turn: &str) -> &str {
    let mut rest = turn.trim_start();
    while let Some(after) = rest.strip_prefix('[') {
        match after.split_once("] ") {
            Some((_, tail)) => rest = tail.trim_start(),
            None => break,
        }
    }
    rest
}

/// Whether `quote` occurs inside one of the user's own turns in `memory`.
pub fn quoted_from_user(memory: &str, quote: &str) -> bool {
    let quote = normalize(quote.trim_matches(|c: char| c == '"' || c.is_whitespace()));
    if quote.chars().count() < ADVICE_QUOTE_MIN_CHARS {
        return false;
    }
    myelin_core::pipeline::turn_windows::turn_spans(memory, &SPEAKERS)
        .into_iter()
        .filter_map(|span| strip_labels(&memory[span]).strip_prefix(USER_TURN_PREFIX))
        .any(|words| normalize(words).contains(&quote))
}

/// The preferences whose quote is verified in the memory it cites.
pub fn verified(answer: &AdviceAnswer, memories: &[String]) -> Vec<StatedPreference> {
    answer
        .preferences
        .iter()
        .filter(|p| memories.get(p.memory).is_some_and(|m| quoted_from_user(m, &p.quote)))
        .cloned()
        .collect()
}

/// The answer the user reads: the verified quotes, then the recommendation.
/// `None` when nothing verifies; the caller then declines.
pub fn render(verified: &[StatedPreference], recommendation: &str) -> Option<String> {
    if verified.is_empty() {
        return None;
    }
    let quotes = verified
        .iter()
        .map(|p| format!("\"{}\"", p.quote.trim().trim_matches('"')))
        .collect::<Vec<_>>()
        .join("; ");
    Some(format!("{YOU_TOLD_ME}: {quotes}. {}", recommendation.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPISODE: &str = "[2023-05-24] user: I've been getting into turbinado sugar, it adds a richer flavor.\nassistant: Turbinado is great in coffee and on muffins.\nuser: I usually bake on Sundays.";

    /// llama.cpp writes fields in the order they arrive, and they arrive
    /// alphabetically. The intended order is recall first, then the
    /// recommendation, and within a preference the citation first.
    #[test]
    fn the_wire_order_is_preferences_then_recommendation_and_memory_then_quote() {
        let wire = serde_json::to_string(&advice_answer_schema(6)).expect("serialise");
        let at = |key: &str| wire.find(&format!("\"{key}\":")).expect(key);
        assert!(at("preferences") < at("recommendation"), "{wire}");
        assert!(at("memory") < at("quote"), "{wire}");
    }

    #[test]
    fn the_schema_has_no_decline_and_bounds_every_field() {
        let schema = advice_answer_schema(6);
        let props = schema["properties"].as_object().expect("properties");
        assert_eq!(props.len(), 2, "no field to decline in");
        let pref = &schema["properties"]["preferences"];
        assert_eq!(pref["minItems"], 1);
        assert_eq!(pref["maxItems"], ADVICE_PREFERENCES_MAX);
        assert_eq!(pref["items"]["properties"]["memory"]["maximum"], 5);
        assert_eq!(pref["items"]["properties"]["quote"]["maxLength"], ADVICE_QUOTE_MAX_CHARS);
        assert_eq!(
            schema["properties"]["recommendation"]["maxLength"],
            ADVICE_RECOMMENDATION_MAX_CHARS
        );
    }

    #[test]
    fn a_quote_counts_only_inside_a_user_turn() {
        assert!(quoted_from_user(EPISODE, "turbinado sugar, it adds a richer   FLAVOR"));
        assert!(quoted_from_user(EPISODE, "\"I usually bake on Sundays.\""));
        assert!(
            !quoted_from_user(EPISODE, "Turbinado is great in coffee"),
            "the assistant's words are not the user's"
        );
        assert!(!quoted_from_user(EPISODE, "I prefer demerara sugar"), "not in the memory");
        assert!(!quoted_from_user(EPISODE, "sugar"), "too short to be a preference");
    }

    #[test]
    fn a_windowed_view_is_read_turn_by_turn() {
        let windowed = "[2023-05-21] …\nuser: I'm into indie rock, especially Arctic Monkeys.";
        assert!(quoted_from_user(windowed, "especially Arctic Monkeys"));
    }

    #[test]
    fn only_quotes_verified_in_the_memory_they_cite_are_kept() {
        let memories = vec!["[2023-05-01] user: I work from home.".to_string(), EPISODE.to_string()];
        let answer = AdviceAnswer {
            preferences: vec![
                StatedPreference { memory: 1, quote: "it adds a richer flavor".into() },
                StatedPreference { memory: 0, quote: "it adds a richer flavor".into() },
                StatedPreference { memory: 7, quote: "I usually bake on Sundays".into() },
            ],
            recommendation: "Try a turbinado crumb cake.".into(),
        };
        let kept = verified(&answer, &memories);
        assert_eq!(kept, vec![answer.preferences[0].clone()], "wrong memory and out-of-range dropped");
        assert_eq!(
            render(&kept, &answer.recommendation).as_deref(),
            Some("You told me: \"it adds a richer flavor\". Try a turbinado crumb cake.")
        );
    }

    #[test]
    fn nothing_verified_renders_nothing() {
        assert_eq!(render(&[], "Try a crumb cake."), None);
    }

    #[test]
    fn the_answer_parses_from_the_constrained_json() {
        let text = r#"{"preferences":[{"memory":1,"quote":"it adds a richer flavor"}],"recommendation":"Bake."}"#;
        let parsed: AdviceAnswer = serde_json::from_str(text).expect("parse");
        assert_eq!(parsed.preferences[0].memory, 1);
        assert_eq!(parsed.recommendation, "Bake.");
    }
}
