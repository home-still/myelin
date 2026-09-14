//! Typed loader for the LoCoMo-10 benchmark (`PLAN.md` §1.1).
//!
//! LoCoMo is a long-conversation memory benchmark: 10 conversations, each with
//! multiple sessions of dialogue turns and a QA set spanning five categories.
//! Category 5 is the adversarial/unanswerable set — its 446 items must be
//! excluded to arrive at the 1 540 "comparable" total that papers quote, but
//! **both** totals are reportable.  Reporting a LoCoMo accuracy without saying
//! which denominator was used makes the number incomparable (`PLAN.md` §1.1).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

/// A single dialogue turn inside a session.
///
/// `img_url` and `blip_caption` are present only on turns that carry an image
/// attachment; `dia_id` follows the `D{session}:{turn}` convention used as
/// evidence citation throughout the QA set.
#[derive(Debug, Clone, Deserialize)]
pub struct Turn {
    pub speaker: String,
    pub dia_id: String,
    pub text: String,
    #[serde(default)]
    pub img_url: Option<Vec<String>>,
    #[serde(default)]
    pub blip_caption: Option<String>,
}

/// One session within a conversation, ordered by its numeric index.
#[derive(Debug, Clone)]
pub struct Session {
    pub index: u32,
    pub date_time: Option<String>,
    pub turns: Vec<Turn>,
}

/// A QA item.  Category 5 is adversarial; `answer` may be absent on some of
/// those items and is not always a plain string, so it is kept as a
/// `serde_json::Value`.
#[derive(Debug, Clone, Deserialize)]
pub struct QaItem {
    pub question: String,
    #[serde(default)]
    pub answer: Option<Value>,
    pub evidence: Vec<String>,
    pub category: u8,
}

/// One LoCoMo conversation: sessions of turns plus the QA set and the three
/// pre-computed summary blocks (kept as opaque `Value` since no current code
/// needs their structure).
#[derive(Debug, Clone)]
pub struct LocomoConversation {
    pub sample_id: String,
    pub sessions: Vec<Session>,
    pub qa: Vec<QaItem>,
    pub event_summary: Value,
    pub observation: Value,
    pub session_summary: Value,
}

/// Raw serde intermediate mirroring the on-disk JSON shape.
#[derive(Deserialize)]
struct RawConversation {
    sample_id: String,
    conversation: Value,
    qa: Vec<QaItem>,
    event_summary: Value,
    observation: Value,
    session_summary: Value,
}

/// Aggregate QA counts, broken down by category.
///
/// `comparable_1540` is `total - adversarial` (i.e. category 5 dropped).  Both
/// the full 1 986 and the 1 540 are standard reporting denominators; code that
/// prints one without naming it is a bug.
#[derive(Debug, Clone)]
pub struct QaCounts {
    pub by_category: BTreeMap<u8, usize>,
    pub total: usize,
    pub comparable_1540: usize,
    pub adversarial: usize,
}

/// Parse `data/locomo10.json` into typed conversations.
///
/// The `conversation` object interleaves `session_N` arrays with
/// `session_N_date_time` strings (plus `speaker_a`/`speaker_b`).  Sessions are
/// paired up and sorted by **numeric** N, not lexicographically — `session_10`
/// must follow `session_9`, not `session_1`.
pub fn load(path: &Path) -> Result<Vec<LocomoConversation>> {
    let raw_text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let raw: Vec<RawConversation> = serde_json::from_str(&raw_text)
        .context("parsing LoCoMo JSON")?;

    let mut out = Vec::with_capacity(raw.len());
    for rc in raw {
        let sessions = parse_sessions(&rc.conversation)?;
        out.push(LocomoConversation {
            sample_id: rc.sample_id,
            sessions,
            qa: rc.qa,
            event_summary: rc.event_summary,
            observation: rc.observation,
            session_summary: rc.session_summary,
        });
    }
    Ok(out)
}

fn parse_sessions(conv: &Value) -> Result<Vec<Session>> {
    let map = conv
        .as_object()
        .context("conversation is not an object")?;

    // Collect (index, turns, date_time) triples keyed by numeric session N.
    let mut entries: Vec<(u32, Vec<Turn>, Option<String>)> = Vec::new();

    for (key, val) in map {
        if let Some(rest) = key.strip_prefix("session_") {
            if let Some(num_str) = rest.strip_suffix("_date_time") {
                // session_N_date_time -> String
                if let Ok(num) = num_str.parse::<u32>() {
                    let dt = val.as_str().map(|s| s.to_string());
                    // Find or create the entry for this N.
                    if let Some(slot) = entries.iter_mut().find(|e| e.0 == num) {
                        slot.2 = dt;
                    } else {
                        entries.push((num, Vec::new(), dt));
                    }
                }
            } else if let Ok(num) = rest.parse::<u32>() {
                // session_N -> array of turns
                let turns: Vec<Turn> = serde_json::from_value(val.clone())
                    .with_context(|| format!("parsing turns for session_{}", num))?;
                if let Some(slot) = entries.iter_mut().find(|e| e.0 == num) {
                    slot.1 = turns;
                } else {
                    entries.push((num, turns, None));
                }
            }
        }
        // Ignore speaker_a / speaker_b and anything else.
    }

    // Sort by numeric index — NOT lexicographic key order.
    entries.sort_by_key(|e| e.0);

    let sessions = entries
        .into_iter()
        .map(|(index, turns, date_time)| Session {
            index,
            date_time,
            turns,
        })
        .collect();
    Ok(sessions)
}

/// Compute QA category counts across a set of conversations.
pub fn qa_counts(conversations: &[LocomoConversation]) -> QaCounts {
    let mut by_category: BTreeMap<u8, usize> = BTreeMap::new();
    let mut total = 0;
    for conv in conversations {
        for qa in &conv.qa {
            *by_category.entry(qa.category).or_insert(0) += 1;
            total += 1;
        }
    }
    let adversarial = *by_category.get(&5).unwrap_or(&0);
    QaCounts {
        by_category,
        total,
        comparable_1540: total - adversarial,
        adversarial,
    }
}

// ---------------------------------------------------------------------------
// Unit tests — no network, small inline fixtures.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `session_10` must sort after `session_9`, not after `session_1`.
    /// This is the whole reason we parse the key as a number rather than
    /// relying on serde's map iteration or lexicographic sort.
    #[test]
    fn session_ordering_is_numeric_not_lexicographic() {
        let conv_json = serde_json::json!({
            "speaker_a": "A",
            "speaker_b": "B",
            "session_1": [
                {"speaker": "A", "dia_id": "D1:1", "text": "first"}
            ],
            "session_2": [
                {"speaker": "B", "dia_id": "D2:1", "text": "second"}
            ],
            "session_1_date_time": "2023-01-01",
            "session_2_date_time": "2023-01-02",
            "session_10": [
                {"speaker": "A", "dia_id": "D10:1", "text": "tenth"}
            ],
            "session_10_date_time": "2023-01-10",
            "session_9": [
                {"speaker": "B", "dia_id": "D9:1", "text": "ninth"}
            ],
            "session_9_date_time": "2023-01-09",
        });

        let sessions = parse_sessions(&conv_json).unwrap();
        let indices: Vec<u32> = sessions.iter().map(|s| s.index).collect();
        assert_eq!(indices, vec![1, 2, 9, 10]);
        assert_eq!(sessions[2].turns[0].text, "ninth");
        assert_eq!(sessions[3].turns[0].text, "tenth");
        assert_eq!(sessions[0].date_time.as_deref(), Some("2023-01-01"));
    }

    /// `QaCounts` arithmetic on a small fixture: total, adversarial, and the
    /// comparable_1540 subtraction.
    #[test]
    fn qa_counts_arithmetic() {
        // Inline fixture: 2 + 3 + 1 category-5 items across two conversations.
        let convs = vec![
            LocomoConversation {
                sample_id: "test-1".into(),
                sessions: vec![],
                qa: vec![
                    QaItem {
                        question: "q1".into(),
                        answer: Some(Value::String("a1".into())),
                        evidence: vec!["D1:1".into()],
                        category: 1,
                    },
                    QaItem {
                        question: "q2".into(),
                        answer: Some(Value::String("a2".into())),
                        evidence: vec!["D1:2".into()],
                        category: 5,
                    },
                    QaItem {
                        question: "q3".into(),
                        answer: None,
                        evidence: vec!["D2:1".into()],
                        category: 5,
                    },
                ],
                event_summary: Value::Null,
                observation: Value::Null,
                session_summary: Value::Null,
            },
            LocomoConversation {
                sample_id: "test-2".into(),
                sessions: vec![],
                qa: vec![
                    QaItem {
                        question: "q4".into(),
                        answer: Some(Value::String("a4".into())),
                        evidence: vec!["D1:1".into()],
                        category: 2,
                    },
                    QaItem {
                        question: "q5".into(),
                        answer: Some(Value::String("a5".into())),
                        evidence: vec!["D2:1".into()],
                        category: 5,
                    },
                ],
                event_summary: Value::Null,
                observation: Value::Null,
                session_summary: Value::Null,
            },
        ];

        let counts = qa_counts(&convs);
        assert_eq!(counts.total, 5);
        assert_eq!(counts.adversarial, 3);
        assert_eq!(counts.comparable_1540, 2);
        assert_eq!(counts.by_category.get(&1), Some(&1));
        assert_eq!(counts.by_category.get(&2), Some(&1));
        assert_eq!(counts.by_category.get(&5), Some(&3));
        assert_eq!(counts.by_category.len(), 3);
    }
}