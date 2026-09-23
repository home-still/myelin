//! M50 — Chronos-style event tuples, extracted per conversation session.
//!
//! Chronos (Sen et al., `10.48550/arXiv.2603.16862` §3.1) decomposes raw
//! dialogue into *(subject, verb, object)* events with resolved datetime
//! ranges and 2–4 lexical aliases, indexed beside the raw turns. Its
//! ablation (Table 3) attributes the largest single loss to removing that
//! events index: **93.1 → 58.6** with GPT-4o, but only **94.8 → 92.2** with
//! Opus — the weaker the reader, the more the events carry. Ours is a 9B.
//!
//! Two deliberate departures from the paper, both forced by this corpus and
//! this reader:
//!
//! * **The model never does date arithmetic.** Chronos has the extractor
//!   emit ISO ranges itself. Here the model copies the time phrase verbatim
//!   (`when`) and [`resolve_when`] resolves it against the session's date
//!   with M19's closed grammar ([`crate::time::resolve_relative`] for
//!   relative forms, [`crate::time::parse_response`] for absolute ones),
//!   which fails closed instead of guessing. A 9B subtracting weeks is the
//!   error M46 measured the reader making at answer time.
//! * **Extraction is date-free, so it is cacheable by content.**
//!   LongMemEval_S reuses sessions across haystacks (25,112 slots, 18,821
//!   distinct contents) but re-dates 5,283 of those slots. A date-free
//!   extraction is the same for every copy; resolution is per copy, free.
//!
//! Windowing follows Chronos exactly: at most 25 turns per call with a
//! 5-turn overlap (§3.1; "most sessions contain fewer than 25 turns").

use chrono::{Datelike, NaiveDate};
use serde::{Deserialize, Serialize};

use crate::llm::{complete_json, CompletionRequest, Llm, Message};
use crate::time::{parse_response, resolve_relative, DayRange, Temporal};
use crate::Result;

/// Turns per extraction call (Chronos §3.1).
pub const WINDOW_TURNS: usize = 25;
/// Turns shared by consecutive windows (Chronos §3.1).
pub const WINDOW_OVERLAP_TURNS: usize = 5;
/// Events one call may return. LoCoMo sessions average 21.6 turns of
/// personal news; 16 leaves room without inviting padding.
pub const MAX_EVENTS_PER_CALL: u64 = 16;
/// Chronos generates "2-4 lexical aliases for each event" (§3.1).
pub const MIN_ALIASES: u64 = 2;
pub const MAX_ALIASES: u64 = 4;
/// Completion ceiling for one call: 16 events at ~120 tokens each, plus the
/// JSON frame. A truncated body fails to parse and is refused, not guessed.
pub const EXTRACT_MAX_TOKENS: u32 = 2400;

// Field ceilings, all far under llama.cpp's 2,000-character grammar limit
// (`myelin-eval::build::MAX_SCHEMA_MAX_LENGTH`).
const SUBJECT_MAX_CHARS: u64 = 80;
const VERB_MAX_CHARS: u64 = 60;
const OBJECT_MAX_CHARS: u64 = 240;
const WHEN_MAX_CHARS: u64 = 60;
const ALIAS_MAX_CHARS: u64 = 120;

/// The extraction instruction. Chronos publishes the method, not the prompt,
/// so this states §3.1's rules in our own words: SVO with no pronouns, the
/// time phrase copied rather than computed (our departure), aliases in
/// different vocabulary (their "bought Fitbit" example, verbatim).
///
/// It must not show the empty object literally. The first draft ended on
/// `Return {"events": []} when the session states no event` and, measured
/// on LoCoMo conv-26 through the grammar-constrained reader, returned an
/// empty list for **16 of 19** sessions — including "I went to a LGBTQ
/// support group yesterday". A constrained 9B copies the literal it was
/// shown. The instruction now describes when the list may be empty instead:
/// the same conversation then gave **70 events over 19 sessions, one empty**,
/// every `when` copied verbatim ("yesterday", "last year", "last Saturday").
pub const EVENTS_SYSTEM: &str = r#"You extract events from one conversation session for a long-term memory.

An event is something that happened, is happening, or is planned: an action, a purchase, a visit, a start or a stop, a move, a decision, a change of job, home, habit, health or relationship. Every event has a subject, a verb and an object.

For each event write:
- subject: who did it, by the speaker's name as the transcript labels it (a transcript labelled "user:" and "assistant:" gives "the user" and "the assistant"). Never "I", "you", "he", "she" or "they".
- verb: the action, past tense for what was done ("bought", "started"), "plans to <verb>" for what is planned.
- object: what the action was done to or with, specific enough to stand alone: names, titles, places, brands, numbers and amounts exactly as said.
- when: the time expression the conversation attaches to this event, copied word for word ("last Tuesday", "two weeks ago", "in 2019", "on May 7", "next month"). Leave it empty when the conversation gives no time for the event. Never work out or convert a date yourself.
- aliases: 2 to 4 short paraphrases of the event in completely different words (synonyms, categories, related terms) so a search phrased differently still finds it. Example: "bought a Fitbit" -> "picked up a fitness tracker", "got a step counter", "purchased a wearable".

Rules:
- Only events this session states. Do not infer events nobody said.
- Skip greetings, opinions about nothing that happened, and general advice.
- One event per occurrence.
- Go through the session turn by turn: most turns that share personal news state at least one event.
- The list is empty only for a session in which nobody reports anything that happened or is planned."#;

/// JSON schema for one extraction call. Field order is the generation
/// order: the event is fixed before its time phrase is copied, and the
/// aliases come last, paraphrasing what is already written (the M42/M43
/// ordering lesson: a later field conditions on the earlier ones).
pub fn events_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["events"],
        "properties": {
            "events": {
                "type": "array",
                "maxItems": MAX_EVENTS_PER_CALL,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["subject", "verb", "object", "when", "aliases"],
                    "properties": {
                        "subject": {"type": "string", "maxLength": SUBJECT_MAX_CHARS},
                        "verb": {"type": "string", "maxLength": VERB_MAX_CHARS},
                        "object": {"type": "string", "maxLength": OBJECT_MAX_CHARS},
                        "when": {"type": "string", "maxLength": WHEN_MAX_CHARS},
                        "aliases": {
                            "type": "array",
                            "minItems": MIN_ALIASES,
                            "maxItems": MAX_ALIASES,
                            "items": {"type": "string", "maxLength": ALIAS_MAX_CHARS}
                        }
                    }
                }
            }
        }
    })
}

/// One event as the model states it: date-free, so it holds for every copy
/// of the session whatever that copy's date.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedEvent {
    pub subject: String,
    pub verb: String,
    pub object: String,
    pub when: String,
    pub aliases: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EventList {
    events: Vec<ExtractedEvent>,
}

/// One turn of a session, as the extractor reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionTurn {
    pub speaker: String,
    pub text: String,
}

/// Turn index ranges for the extraction calls over a session of `n` turns:
/// windows of [`WINDOW_TURNS`] advancing by `WINDOW_TURNS -
/// WINDOW_OVERLAP_TURNS`, the last one ending at `n`. Empty for `n == 0`.
pub fn windows(n: usize) -> Vec<std::ops::Range<usize>> {
    let stride = WINDOW_TURNS - WINDOW_OVERLAP_TURNS;
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < n {
        let end = (start + WINDOW_TURNS).min(n);
        out.push(start..end);
        if end == n {
            break;
        }
        start += stride;
    }
    out
}

fn render(turns: &[SessionTurn]) -> String {
    turns
        .iter()
        .map(|t| format!("{}: {}", t.speaker, t.text))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract a session's events: one schema-constrained call per window.
///
/// Events repeated across an overlap are dropped by exact
/// `(subject, verb, object, when)` match; paraphrased repeats survive and
/// are counted by the caller as ordinary records. A call whose body does not
/// parse fails the whole session — the caller retries it on the next run
/// rather than storing a partial one.
pub async fn extract_session(llm: &dyn Llm, turns: &[SessionTurn]) -> Result<Vec<ExtractedEvent>> {
    let mut out: Vec<ExtractedEvent> = Vec::new();
    for w in windows(turns.len()) {
        let request = CompletionRequest::new(vec![
            Message::system(EVENTS_SYSTEM),
            Message::user(render(&turns[w])),
        ])
        .with_schema(events_schema())
        .with_max_tokens(EXTRACT_MAX_TOKENS);
        let list: EventList = complete_json(llm, &request).await?;
        for e in list.events {
            let dup = out.iter().any(|o| {
                o.subject == e.subject && o.verb == e.verb && o.object == e.object && o.when == e.when
            });
            if !dup {
                out.push(e);
            }
        }
    }
    Ok(out)
}

/// When an event happened, as far as the session lets us say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventTime {
    /// The session attached no time: the event is dated by when it was said.
    Said,
    /// A time phrase the closed grammar resolved against the session date.
    Stated { phrase: String, range: DayRange },
    /// A time phrase outside the grammar ("a few weeks ago", "recently").
    /// Kept verbatim in the text; the record is dated by when it was said.
    Unresolved { phrase: String },
}

impl EventTime {
    /// The record's `t_valid`: the first day the event could have happened
    /// when the phrase resolved, the session's day otherwise.
    pub fn t_valid(&self, said: NaiveDate) -> NaiveDate {
        match self {
            EventTime::Stated { range, .. } => range.lo,
            EventTime::Said | EventTime::Unresolved { .. } => said,
        }
    }
}

/// Resolve an event's `when` against the day the session happened.
///
/// Relative forms first (`last Tuesday`, `two weeks ago`, `next month`),
/// then absolute ones (`in 2019`, `May 7`), borrowing the session's year
/// only for a phrase that names a month — `parse_response`'s own rule. A
/// duration ("for two weeks") has no position in time and is unresolved.
pub fn resolve_when(when: &str, said: NaiveDate) -> EventTime {
    let phrase = when.trim();
    if phrase.is_empty() {
        return EventTime::Said;
    }
    if let Some(r) = resolve_relative(phrase, said).into_iter().next() {
        return EventTime::Stated { phrase: phrase.to_string(), range: r.range };
    }
    if let Some(Temporal::Range(range)) = parse_response(phrase, Some(said.year())) {
        return EventTime::Stated { phrase: phrase.to_string(), range };
    }
    EventTime::Unresolved { phrase: phrase.to_string() }
}

/// The record text: the tuple, when it happened and when it was said, then
/// the aliases — which ride in the text so the BM25 channel can match them,
/// the use Chronos puts them to (§3.1, "robust keyword matching").
pub fn event_text(e: &ExtractedEvent, time: &EventTime, said: NaiveDate) -> String {
    let tuple = format!("{} {} {}", e.subject.trim(), e.verb.trim(), e.object.trim());
    let when = match time {
        EventTime::Said => format!("[said {said}]"),
        EventTime::Stated { phrase, range } if range.lo == range.hi => {
            format!("[{} — \"{phrase}\", said {said}]", range.lo)
        }
        EventTime::Stated { phrase, range } => {
            format!("[{} to {} — \"{phrase}\", said {said}]", range.lo, range.hi)
        }
        EventTime::Unresolved { phrase } => format!("[\"{phrase}\", said {said}]"),
    };
    let aliases: Vec<&str> = e
        .aliases
        .iter()
        .map(|a| a.trim())
        .filter(|a| !a.is_empty())
        .collect();
    if aliases.is_empty() {
        format!("{tuple} {when}")
    } else {
        format!("{tuple} {when}\nalso: {}", aliases.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).expect("valid test date")
    }

    fn ev(when: &str) -> ExtractedEvent {
        ExtractedEvent {
            subject: "Caroline".into(),
            verb: "attended".into(),
            object: "an LGBTQ support group".into(),
            when: when.into(),
            aliases: vec!["went to a peer meeting".into(), "joined a pride circle".into()],
        }
    }

    #[test]
    fn windows_follow_chronos_25_with_5_overlap() {
        assert!(windows(0).is_empty());
        assert_eq!(windows(12), vec![0..12]);
        assert_eq!(windows(25), vec![0..25]);
        assert_eq!(windows(26), vec![0..25, 20..26]);
        assert_eq!(windows(47), vec![0..25, 20..45, 40..47]);
        // Every turn is covered and consecutive windows share exactly 5.
        for n in 1..200 {
            let w = windows(n);
            assert_eq!(w[0].start, 0);
            assert_eq!(w.last().map(|r| r.end), Some(n));
            for pair in w.windows(2) {
                assert_eq!(pair[0].end - pair[1].start, WINDOW_OVERLAP_TURNS);
            }
        }
    }

    #[test]
    fn empty_when_is_dated_by_the_session() {
        let said = d(2023, 5, 8);
        assert_eq!(resolve_when("  ", said), EventTime::Said);
        assert_eq!(EventTime::Said.t_valid(said), said);
    }

    #[test]
    fn relative_when_resolves_against_the_session_day() {
        let said = d(2023, 5, 8); // a Monday
        match resolve_when("yesterday", said) {
            EventTime::Stated { range, .. } => assert_eq!(range, DayRange::day(d(2023, 5, 7))),
            other => panic!("expected Stated, got {other:?}"),
        }
        let t = resolve_when("two weeks ago", said);
        assert!(matches!(t, EventTime::Stated { .. }), "{t:?}");
    }

    #[test]
    fn absolute_when_borrows_the_session_year_only_for_a_month() {
        let said = d(2023, 5, 8);
        match resolve_when("in 2019", said) {
            EventTime::Stated { range, .. } => {
                assert_eq!(range.lo, d(2019, 1, 1));
                assert_eq!(range.hi, d(2019, 12, 31));
            }
            other => panic!("expected Stated, got {other:?}"),
        }
        match resolve_when("on May 7", said) {
            EventTime::Stated { range, .. } => assert_eq!(range.lo, d(2023, 5, 7)),
            other => panic!("expected Stated, got {other:?}"),
        }
    }

    #[test]
    fn vague_or_duration_when_is_unresolved_not_guessed() {
        let said = d(2023, 5, 8);
        for phrase in ["a few weeks ago", "recently", "for two weeks"] {
            let t = resolve_when(phrase, said);
            assert_eq!(t, EventTime::Unresolved { phrase: phrase.into() }, "{phrase}");
            assert_eq!(t.t_valid(said), said);
        }
    }

    #[test]
    fn text_carries_tuple_dates_and_aliases() {
        let said = d(2023, 5, 8);
        let t = resolve_when("yesterday", said);
        assert_eq!(
            event_text(&ev("yesterday"), &t, said),
            "Caroline attended an LGBTQ support group [2023-05-07 — \"yesterday\", said 2023-05-08]\n\
             also: went to a peer meeting; joined a pride circle"
        );
        assert_eq!(
            event_text(&ev(""), &EventTime::Said, said),
            "Caroline attended an LGBTQ support group [said 2023-05-08]\n\
             also: went to a peer meeting; joined a pride circle"
        );
    }

    #[test]
    fn schema_orders_fields_and_bounds_every_string() {
        let s = events_schema();
        let item = &s["properties"]["events"]["items"];
        let req: Vec<&str> = item["required"]
            .as_array()
            .expect("required is an array")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert_eq!(req, ["subject", "verb", "object", "when", "aliases"]);
        assert_eq!(item["properties"]["aliases"]["minItems"], MIN_ALIASES);
        assert_eq!(item["properties"]["aliases"]["maxItems"], MAX_ALIASES);
    }
}
