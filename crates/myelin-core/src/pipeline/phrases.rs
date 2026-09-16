//! Deterministic phrase extraction for the graph route (`PLAN.md` §5.4).
//!
//! **No model in the loop, and the same function on both sides.** HippoRAG
//! seeds PPR from LLM NER over every passage and every query. That is the
//! cost this repo has already refused twice: `build.rs` disables extraction
//! for LME-V2 and LongMemEval_S because a fact-extraction pass over those
//! corpora extrapolates to hundreds of GPU-hours. So the phrase nodes are
//! derived from the text itself, cheaply and reproducibly.
//!
//! The other reason this is one function rather than two: index-side and
//! query-side extraction must agree exactly. Asymmetric extraction produces
//! query seeds that can never match a stored phrase, and the graph channel
//! then contributes nothing but background reset mass — a uniform ranking,
//! which is pure noise in RRF.
//!
//! Why not [`crate::pipeline::ingest::Episode::to_record`]'s `entities`: it
//! sets an episodic record's entities to **the turn speakers only**, so the
//! incidence rows it produces are one hub node per conversation and carry no
//! bridging signal at all.
//!
//! Capitalization is the whole heuristic. It is a weak proxy for "named
//! entity" and it is stated as such: the measurement in
//! `docs/measurements/m12-graph-route.md` bounds what it buys.

use crate::model::record::MemoryRecord;
use crate::store::ledger::IncidenceRow;

/// Per-record edge ceiling, so one long record cannot dominate the graph and
/// so the namespace's edge count stays linear in records.
pub const MAX_PHRASES_PER_TEXT: usize = 32;

/// Words that are capitalized only because they start a sentence.
///
/// Without this list every sentence donates `the`, `i` and `when` as phrase
/// nodes, and a node incident to every record is a hub that connects
/// everything to everything. Sorted, and searched with `binary_search`;
/// `stopwords_are_sorted` is the test that keeps that true.
pub const STOPWORDS: &[&str] = &[
    "a", "about", "after", "again", "also", "an", "and", "are", "as", "at", "be", "because",
    "been", "before", "but", "by", "can", "could", "did", "do", "does", "during", "for", "from",
    "had", "has", "have", "he", "her", "here", "hers", "his", "how", "i", "if", "in", "is", "it",
    "its", "just", "me", "my", "no", "not", "of", "oh", "ok", "okay", "on", "or", "our", "over",
    "she", "should", "so", "some", "still", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "under", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "why", "will", "with", "would", "yeah", "yes", "you",
    "your",
];

fn is_stopword(lower: &str) -> bool {
    STOPWORDS.binary_search(&lower).is_ok()
}

/// One token plus whether the separator immediately before it held anything
/// other than spaces or tabs.
struct Tok<'a> {
    text: &'a str,
    /// A run of capitalized tokens is only a multi-word name if nothing
    /// punctuates it. Episodes render as `"Speaker: text\nSpeaker: text"`
    /// ([`crate::pipeline::ingest::EpisodeDraft::render`]), so without this
    /// every turn boundary would donate a `speaker firstword` phrase —
    /// degree-1 junk that no query-side seed reproduces, and that would
    /// consume half of [`MAX_PHRASES_PER_TEXT`] on real conversational text.
    hard_gap: bool,
}

/// A token is a maximal run of alphanumerics, `-` and `'`; everything else
/// separates. Hyphens and apostrophes stay inside a token so `check-in` and
/// `Caroline's` survive as one surface, and so an ISO date is one token.
fn tokens(text: &str) -> Vec<Tok<'_>> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    let mut hard = true;
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() || c == '-' || c == '\'' {
            if start.is_none() {
                start = Some(i);
            }
            continue;
        }
        if let Some(s) = start.take() {
            out.push(Tok {
                text: &text[s..i],
                hard_gap: std::mem::replace(&mut hard, false),
            });
        }
        if c != ' ' && c != '\t' {
            hard = true;
        }
    }
    if let Some(s) = start {
        out.push(Tok {
            text: &text[s..],
            hard_gap: hard,
        });
    }
    out
}

/// `1900..=2100`, four digits. Years bridge knowledge-update chains and are
/// never capitalized, so they are collected explicitly.
fn is_year(token: &str) -> bool {
    token.len() == 4
        && token.bytes().all(|b| b.is_ascii_digit())
        && matches!(token.parse::<u16>(), Ok(y) if (1900..=2100).contains(&y))
}

/// `YYYY-MM-DD`. Checked by shape rather than by regex: no new dependency,
/// and the calendar is not validated because an implausible date is still a
/// perfectly good bridge between two records that both name it.
fn is_iso_date(token: &str) -> bool {
    let b = token.as_bytes();
    b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && b.iter()
            .enumerate()
            .all(|(i, c)| matches!(i, 4 | 7) || c.is_ascii_digit())
}

/// Phrases extracted from text with no model in the loop, used identically at
/// index time and at query time.
///
/// Sorted and deduplicated, so the incidence write order does not depend on
/// where a phrase happened to appear in the text.
pub fn phrases(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut run: Vec<String> = Vec::new();

    // A capitalized run is flushed as one phrase, plus its members when there
    // is more than one — so `"New York City"` in a query still seeds the
    // `new york` some other record stored.
    let flush = |run: &mut Vec<String>, out: &mut Vec<String>| {
        match run.len() {
            0 => {}
            1 => out.push(run.remove(0)),
            _ => {
                out.push(run.join(" "));
                out.append(run);
            }
        }
        run.clear();
    };

    for tok in tokens(text) {
        let lower = tok.text.to_lowercase();
        // A capitalized stopword breaks the run rather than joining it.
        // Otherwise a sentence-initial article fuses onto the first real
        // entity and `"The Berlin office"` donates `the berlin` — a node
        // that bridges nothing and that no query-side seed would ever
        // reproduce unless the query opened the same way.
        let capitalized =
            tok.text.chars().next().is_some_and(char::is_uppercase) && !is_stopword(&lower);
        if capitalized {
            if tok.hard_gap {
                flush(&mut run, &mut out);
            }
            run.push(lower);
            continue;
        }
        flush(&mut run, &mut out);
        if is_year(tok.text) || is_iso_date(tok.text) {
            out.push(lower);
        }
    }
    flush(&mut run, &mut out);

    out.retain(|p| p.chars().count() >= 2 && !is_stopword(p));
    out.sort_unstable();
    out.dedup();
    out.truncate(MAX_PHRASES_PER_TEXT);
    out
}

/// The union of a record's extracted entities and `phrases(&record.text)`, as
/// `incidence` rows at weight 1.0.
///
/// One function so a fresh ingest ([`crate::pipeline::index::Indexer::index`])
/// and a backfill (`myelin-eval phrases`) produce byte-identical rows. Weight
/// does not distinguish the two sources: an extracted-entity-vs-derived-phrase
/// weight ratio would be an unmeasured knob.
pub fn incidence_rows(record: &MemoryRecord) -> Vec<IncidenceRow> {
    let mut all: Vec<String> = record
        .entities
        .iter()
        .map(|e| e.phrase.trim().to_lowercase())
        .filter(|p| !p.is_empty())
        .collect();
    all.extend(phrases(&record.text));
    all.sort_unstable();
    all.dedup();
    all.truncate(MAX_PHRASES_PER_TEXT);
    all.into_iter()
        .map(|phrase| IncidenceRow {
            phrase,
            record_id: record.id,
            weight: 1.0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::record::{ActorId, EntityRef, Scope, SourceRef};
    use crate::pipeline::ingest::{BoundaryReason, EpisodeDraft, Turn};

    fn p(text: &str) -> Vec<String> {
        phrases(text)
    }

    #[test]
    fn sentence_initial_stopword_is_not_a_phrase() {
        // "The" is capitalized and must neither stand alone nor fuse onto
        // "Berlin": a `the berlin` node bridges nothing.
        assert_eq!(
            p("The Berlin office is run by Acme Robotics."),
            vec!["acme", "acme robotics", "berlin", "robotics"]
        );
    }

    #[test]
    fn capitalized_run_emits_the_run_and_its_members() {
        assert_eq!(
            p("New York City hosts it."),
            vec!["city", "new", "new york city", "york"]
        );
    }

    #[test]
    fn years_and_iso_dates_are_phrases_but_other_numbers_are_not() {
        assert_eq!(
            p("The move was in 1998 and again on 2026-05-21."),
            vec!["1998", "2026-05-21"]
        );
        // Out of range, wrong width, and a plain quantity: none of them bridge.
        assert_eq!(p("It cost 3 of them in 1776 and 12345."), Vec::<String>::new());
    }

    #[test]
    fn single_character_tokens_are_dropped() {
        // "X" is capitalized but one char, so it cannot be a phrase; the
        // two-token run it belongs to still is.
        assert_eq!(p("X marks Vault Door."), vec!["door", "vault", "vault door"]);
        assert_eq!(p("A Y."), Vec::<String>::new());
    }

    #[test]
    fn punctuation_breaks_a_capitalized_run() {
        // The rendered form of an episode is `"Speaker: text"`, so without a
        // break at the colon every turn donates a `speaker firstword` phrase.
        assert_eq!(
            p("Caroline: Berlin was cold. Dana agreed."),
            vec!["berlin", "caroline", "dana"]
        );
        // A newline is a turn boundary in the same rendering.
        assert_eq!(p("Acme\nRobotics"), vec!["acme", "robotics"]);
    }

    #[test]
    fn output_is_sorted_deduplicated_and_capped() {
        let text = (0..50)
            .map(|i| format!("Name{i:02}"))
            .collect::<Vec<_>>()
            .join(". ");
        let got = p(&text);
        assert_eq!(got.len(), MAX_PHRASES_PER_TEXT);
        assert!(got.windows(2).all(|w| w[0] < w[1]), "{got:?}");

        // Repetition donates one node, not three.
        assert_eq!(p("Berlin. Berlin! Berlin?"), vec!["berlin"]);
    }

    #[test]
    fn stopwords_are_sorted() {
        // `is_stopword` is a binary search; an unsorted list would silently
        // stop matching some of these words.
        assert!(
            STOPWORDS.windows(2).all(|w| w[0] < w[1]),
            "STOPWORDS must be sorted and unique"
        );
    }

    #[test]
    fn incidence_rows_union_entities_with_derived_phrases() {
        let draft = EpisodeDraft {
            unit: "u".into(),
            turns: vec![Turn {
                speaker: "Caroline".into(),
                text: "Caroline moved to Berlin.".into(),
                at: None,
                source: SourceRef::doc("d"),
                unit: "u".into(),
            }],
            ended_by: BoundaryReason::EndOfStream,
            approx_tokens: 8,
        };
        let mut record = draft.to_record(
            &Scope::new("t", "a", "ns"),
            &ActorId::new("u1"),
            &ActorId::new("a1"),
        );
        record.entities.push(EntityRef::new("  ACME Robotics "));
        record.entities.push(EntityRef::new("   "));

        let rows = incidence_rows(&record);
        let got: Vec<&str> = rows.iter().map(|r| r.phrase.as_str()).collect();
        // `to_record` contributes the speaker as an entity and the speaker
        // prefix is also in the rendered text, so `caroline` arrives from
        // both sources and is one row, not two. The blank entity is dropped
        // rather than becoming an empty-string node.
        assert_eq!(got, vec!["acme robotics", "berlin", "caroline"]);
        assert!(rows
            .iter()
            .all(|r| r.record_id == record.id && r.weight == 1.0));
    }
}
