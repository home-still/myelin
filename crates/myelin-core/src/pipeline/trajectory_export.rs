//! M69: myelin serves the files the winning LME-V2 controller reads
//! (`docs/measurements/m69-trajectory-export.md`).
//!
//! AgentRunbook-C, the LongMemEval-V2 authors' own memory (Wu et al. 2026,
//! `10.48550/arXiv.2605.12493` §4.2), keeps every trajectory as
//! `trajectories/<id>/trajectory.json` and lets a coding agent read the
//! files with a shell. With our local controller it scored 82.98 on the
//! 47-question pilot (M54), where myelin's own bounded tools reached 36.17
//! (M62b). The paired trace analysis (2026-09-25) found the reader side
//! identical: the whole gap is which states the controller names, and the
//! file view is what names them well. Cao et al. 2026 (arXiv 2603.20432)
//! report the same shape: text in files, worked with shell and code, beats
//! bespoke retrieval tools.
//!
//! So this module renders myelin's structured copy of each trajectory (M62's
//! `trajectory` and `trajectory_state` tables) as that file, **byte for
//! byte**. The harness writes it with Python's
//! `json.dumps(simplified, indent=2, ensure_ascii=True) + "\n"`
//! (`vendor/longmemeval-v2/memory_modules/trajectory_store.py`, `save_json`),
//! and [`trajectory_json`] reproduces exactly that serialisation of exactly
//! that record:
//! - `actions`: the states' non-empty actions, in order;
//! - `start_url`: as stored (the harness takes the first state's URL, and
//!   so did the build that stored it);
//! - `screenshot`: `screenshots/<index:04>.png` (`screenshot_name_for_state`
//!   with the harness's default suffix; every LME-V2 screenshot is a PNG).
//!
//! Measured against the harness's own files for all 200 LME-V2-Small
//! trajectories: 200 byte-identical (`myelin-eval trajectories-export
//! --check`). Screenshots are not part of it: myelin stores text, and the
//! M54 controller is text-only by pre-registration.

use std::fmt::Write as _;
use std::path::Path;

use crate::error::{MyelinError, Result};
use crate::model::trajectory::AgentTrajectory;
use crate::store::ledger::Ledger;

/// The file each trajectory is served as, under `<out>/<id>/`.
pub const TRAJECTORY_FILE: &str = "trajectory.json";
/// `screenshot_name_for_state`'s suffix when the source file has none of its
/// own, which on LME-V2 is every screenshot.
pub const SCREENSHOT_SUFFIX: &str = ".png";
/// Python's `json.dumps(indent=2)`.
const INDENT: &str = "  ";

/// One JSON value of the simplified trajectory, in the only shapes it has.
enum Json<'a> {
    Str(&'a str),
    OptStr(Option<&'a str>),
    Int(u32),
    Strs(Vec<&'a str>),
    Object(Vec<(&'static str, Json<'a>)>),
    Objects(Vec<Json<'a>>),
}

/// `s` as Python's `json.dumps(..., ensure_ascii=True)` writes a string:
/// quotes, backslash and the five short control escapes; every other
/// character outside printable ASCII (0x20..=0x7E) as lowercase `\uXXXX`,
/// with a surrogate pair above the Basic Multilingual Plane.
fn push_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    // Infallible: writing to a String.
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

fn push_value(out: &mut String, v: &Json<'_>, depth: usize) {
    let pad = |out: &mut String, d: usize| {
        for _ in 0..d {
            out.push_str(INDENT);
        }
    };
    match v {
        Json::Str(s) => push_str(out, s),
        Json::OptStr(None) => out.push_str("null"),
        Json::OptStr(Some(s)) => push_str(out, s),
        Json::Int(n) => {
            let _ = write!(out, "{n}");
        }
        Json::Strs(items) if items.is_empty() => out.push_str("[]"),
        Json::Strs(items) => {
            out.push_str("[\n");
            for (i, s) in items.iter().enumerate() {
                pad(out, depth + 1);
                push_str(out, s);
                out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
            }
            pad(out, depth);
            out.push(']');
        }
        Json::Objects(items) if items.is_empty() => out.push_str("[]"),
        Json::Objects(items) => {
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                pad(out, depth + 1);
                push_value(out, item, depth + 1);
                out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
            }
            pad(out, depth);
            out.push(']');
        }
        Json::Object(fields) if fields.is_empty() => out.push_str("{}"),
        Json::Object(fields) => {
            out.push_str("{\n");
            for (i, (key, value)) in fields.iter().enumerate() {
                pad(out, depth + 1);
                push_str(out, key);
                out.push_str(": ");
                push_value(out, value, depth + 1);
                out.push_str(if i + 1 < fields.len() { ",\n" } else { "\n" });
            }
            pad(out, depth);
            out.push('}');
        }
    }
}

/// The trajectory's `trajectory.json`, byte for byte as the harness writes it.
pub fn trajectory_json(t: &AgentTrajectory) -> String {
    let actions: Vec<&str> = t
        .states
        .iter()
        .filter_map(|s| s.action.as_deref())
        .filter(|a| !a.trim().is_empty())
        .collect();
    let screenshots: Vec<String> = t
        .states
        .iter()
        .map(|s| format!("screenshots/{:04}{SCREENSHOT_SUFFIX}", s.state_index))
        .collect();
    let states = t
        .states
        .iter()
        .zip(&screenshots)
        .map(|(s, shot)| {
            Json::Object(vec![
                ("state_index", Json::Int(s.state_index)),
                ("step", Json::Int(s.step)),
                ("url", Json::Str(&s.url)),
                ("action", Json::OptStr(s.action.as_deref())),
                ("thoughts", Json::OptStr(s.thought.as_deref())),
                ("text", Json::Str(&s.accessibility_tree)),
                ("screenshot", Json::Str(shot)),
            ])
        })
        .collect();
    let doc = Json::Object(vec![
        ("id", Json::Str(&t.header.id)),
        ("goal", Json::Str(&t.header.goal)),
        ("outcome", Json::Str(&t.header.outcome)),
        ("start_url", Json::Str(&t.header.start_url)),
        ("actions", Json::Strs(actions)),
        ("states", Json::Objects(states)),
    ]);
    let mut out = String::new();
    push_value(&mut out, &doc, 0);
    out.push('\n');
    out
}

/// Write every trajectory of `tenant` in `namespace` as
/// `<out>/<id>/trajectory.json`. Returns how many were written. Refuses an
/// empty result: a tenant with no trajectories is a wrong argument, not an
/// export of nothing.
pub async fn export_tenant(
    ledger: &Ledger,
    namespace: &str,
    tenant: &str,
    out: &Path,
) -> Result<usize> {
    let all = ledger.trajectories_in_namespace(namespace).await?;
    let mine: Vec<&AgentTrajectory> = all
        .iter()
        .filter(|t| t.header.scope.tenant == tenant)
        .collect();
    if mine.is_empty() {
        return Err(MyelinError::Store(format!(
            "no trajectories for tenant {tenant:?} in namespace {namespace:?}"
        )));
    }
    for t in &mine {
        let dir = out.join(&t.header.id);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join(TRAJECTORY_FILE), trajectory_json(t))?;
    }
    Ok(mine.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::record::Scope;
    use crate::model::trajectory::{TrajectoryHeader, TrajectoryState};

    /// The expected bytes are Python's own `json.dumps(simplified, indent=2,
    /// ensure_ascii=True) + "\n"` for the same record, generated with
    /// CPython 3.9: accents, an astral emoji as a surrogate pair, quotes,
    /// newline, tab, a C0 control, DEL, null fields and an empty string.
    #[test]
    fn the_file_is_byte_for_byte_what_pythons_json_dumps_writes() {
        let t = AgentTrajectory {
            header: TrajectoryHeader {
                id: "t1".into(),
                scope: Scope::new("t", "a", "ns"),
                record_id: uuid::Uuid::nil(),
                goal: "Find caf\u{e9} \u{1F600} \"x\"".into(),
                environment: "web".into(),
                start_url: "http://a/b".into(),
                outcome: "success".into(),
            },
            states: vec![
                TrajectoryState {
                    state_index: 0,
                    step: 0,
                    url: "http://a/b".into(),
                    action: None,
                    thought: None,
                    accessibility_tree: "line1\nline2\ttab\u{1}\u{7f}".into(),
                },
                TrajectoryState {
                    state_index: 1,
                    step: 1,
                    url: "http://a/c".into(),
                    action: Some("click(12)".into()),
                    thought: Some("ok".into()),
                    accessibility_tree: String::new(),
                },
            ],
        };
        let expected = "{\n  \"id\": \"t1\",\n  \"goal\": \"Find caf\\u00e9 \\ud83d\\ude00 \\\"x\\\"\",\n  \"outcome\": \"success\",\n  \"start_url\": \"http://a/b\",\n  \"actions\": [\n    \"click(12)\"\n  ],\n  \"states\": [\n    {\n      \"state_index\": 0,\n      \"step\": 0,\n      \"url\": \"http://a/b\",\n      \"action\": null,\n      \"thoughts\": null,\n      \"text\": \"line1\\nline2\\ttab\\u0001\\u007f\",\n      \"screenshot\": \"screenshots/0000.png\"\n    },\n    {\n      \"state_index\": 1,\n      \"step\": 1,\n      \"url\": \"http://a/c\",\n      \"action\": \"click(12)\",\n      \"thoughts\": \"ok\",\n      \"text\": \"\",\n      \"screenshot\": \"screenshots/0001.png\"\n    }\n  ]\n}\n";
        assert_eq!(trajectory_json(&t), expected);
    }

    #[test]
    fn an_empty_action_list_is_written_as_python_writes_it() {
        let mut out = String::new();
        push_value(&mut out, &Json::Object(vec![("actions", Json::Strs(vec![]))]), 0);
        assert_eq!(out, "{\n  \"actions\": []\n}");
    }
}
