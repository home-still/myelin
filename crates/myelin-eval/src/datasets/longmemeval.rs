//! Typed loader for LongMemEval JSON (`PLAN.md` §1.1).
//!
//! LongMemEval items carry a `question_type`, a `question`, an `answer`, a
//! `question_date`, and a `haystack_sessions` list — each session is a list of
//! `{role, content}` turns, some of which carry a `has_answer` flag indicating
//! whether that turn contains the evidence needed to answer the question.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

/// A single turn inside a LongMemEval haystack session.
#[derive(Debug, Clone, Deserialize)]
pub struct HaystackTurn {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub has_answer: Option<bool>,
}

/// One session in the haystack — an ordered list of turns.
pub type HaystackSession = Vec<HaystackTurn>;

/// One LongMemEval evaluation item.
#[derive(Debug, Clone, Deserialize)]
pub struct LongMemEvalItem {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    /// 32 of the 500 answers are bare integers rather than strings, so this
    /// cannot be `String`. Use [`LongMemEvalItem::answer_text`]; `Value`'s own
    /// `to_string` would wrap strings in quotes and poison token overlap.
    pub answer: Value,
    pub question_date: String,
    #[serde(default)]
    pub haystack_dates: Option<Vec<String>>,
    #[serde(default)]
    pub haystack_session_ids: Option<Vec<String>>,
    pub haystack_sessions: Vec<HaystackSession>,
}

impl LongMemEvalItem {
    /// The gold answer as plain text, whether the corpus stored it as a
    /// string or a number.
    pub fn answer_text(&self) -> String {
        match &self.answer {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }

    /// LongMemEval marks unanswerable questions with an `_abs` suffix on the
    /// id; there is no separate field. 30 of the 500 are abstention items.
    pub fn is_abstention(&self) -> bool {
        self.question_id.ends_with("_abs")
    }
}

/// Parse a LongMemEval JSON file (a top-level array of items).
pub fn load(path: &Path) -> Result<Vec<LongMemEvalItem>> {
    let raw_text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let items: Vec<LongMemEvalItem> = serde_json::from_str(&raw_text)
        .context("parsing LongMemEval JSON")?;
    Ok(items)
}