//! M62: what a trajectory controller may call, and the evidence it hands the
//! reader (`docs/measurements/m62-native-trajectory-tools.md`).
//!
//! Every tool reads the ledger's structured trajectories
//! ([`Ledger::trajectories`] and friends), never raw files, under one
//! [`ScopeFilter`]. Every tool's output is **bounded**: an LME-V2 state's page
//! averages ~34 KB (~8k tokens), so an unbounded "show me state 7" would fill
//! the controller's window in a few calls. The controller searches and reads
//! windows; the reader, as in AgentRunbook-C, receives whole spans.
//!
//! The evidence rendering is ported from the LME-V2 authors' AgentRunbook-C
//! (`vendor/longmemeval-v2/memory_modules/codex.py`: `format_actions`,
//! `format_span_header`, `format_state_text`,
//! `_build_memory_context_from_output`; `10.48550/arXiv.2605.12493` §4.2), so
//! the reader sees the evidence shape M54 measured at 82.98 on its pilot.

use std::collections::HashMap;

use uuid::Uuid;

use crate::error::{MyelinError, Result};
use crate::model::evidence::{EvidenceItem, EvidenceKind, EvidenceSet};
use crate::model::query::ScopeFilter;
use crate::model::record::{SourceRef, TrustTier};
use crate::model::trajectory::{TrajectoryHeader, TrajectoryState, TrajectoryStep};
use crate::pipeline::compose::weakest_trust;
use crate::pipeline::ingest::approx_tokens;
use crate::store::ledger::Ledger;

/// At most this many states across all spans of one answer
/// (`codex.py` `MAX_TOTAL_SPAN_STATES`).
pub const MAX_TOTAL_SPAN_STATES: u32 = 20;
/// Lines one `grep` returns.
pub const GREP_MAX_LINES: usize = 40;
/// States one `grep` scans for lines, so a common word cannot pull the whole
/// store through the controller.
pub const GREP_MAX_STATES: u32 = 400;
/// Characters kept of one matching line.
pub const GREP_LINE_MAX_CHARS: usize = 240;
/// Page lines one `read` returns.
pub const READ_WINDOW_LINES: usize = 150;

/// The mechanism name a controller-written evidence item carries as its
/// source, so a reader of the set can tell it from retrieved state text.
pub const NOTES_MECHANISM: &str = "m62:trajectory-notes";

/// One span of one trajectory, inclusive at both ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanRequest {
    pub trajectory: String,
    pub first: u32,
    pub last: u32,
}

pub struct TrajectoryTools<'a> {
    ledger: &'a Ledger,
    filter: ScopeFilter,
}

impl<'a> TrajectoryTools<'a> {
    pub fn new(ledger: &'a Ledger, filter: ScopeFilter) -> Self {
        Self { ledger, filter }
    }

    async fn header(&self, id: &str) -> Result<TrajectoryHeader> {
        self.ledger
            .trajectories(&self.filter)
            .await?
            .into_iter()
            .find(|h| h.id == id)
            .ok_or_else(|| MyelinError::Store(format!("no trajectory {id:?} in this memory")))
    }

    /// Every trajectory in scope, one line each: id, start URL, outcome,
    /// goal. Sorted by start URL so similar surfaces sit together, as the
    /// authors' `TRAJECTORY_SUMMARY_FULL.md` is.
    pub async fn list(&self) -> Result<String> {
        let mut headers = self.ledger.trajectories(&self.filter).await?;
        headers.sort_by(|a, b| a.start_url.cmp(&b.start_url).then(a.id.cmp(&b.id)));
        let mut out = format!("{} trajectories (id | start URL | outcome | goal)\n", headers.len());
        for h in &headers {
            out.push_str(&format!(
                "- {} | {} | {} | {}\n",
                h.id,
                h.start_url,
                h.outcome,
                one_line(&h.goal)
            ));
        }
        Ok(out)
    }

    /// One trajectory's goal, start URL, outcome and numbered actions, each
    /// with the state it led to.
    pub async fn summary(&self, id: &str) -> Result<String> {
        let h = self.header(id).await?;
        let steps = self.ledger.trajectory_steps(&self.filter, id).await?;
        let last = steps.last().map_or(0, |s| s.state_index);
        let mut out = format!(
            "Trajectory {id} (states 0-{last})\nStart URL: {}\nGoal: {}\nOutcome: {}\nActions:\n",
            h.start_url, h.goal, h.outcome
        );
        let mut n = 0;
        for s in steps.iter().filter(|s| has_action(s)) {
            n += 1;
            out.push_str(&format!(
                "{n}. {} -> state {} ({})\n",
                s.action.as_deref().unwrap_or_default(),
                s.state_index,
                s.url
            ));
        }
        if n == 0 {
            out.push_str("1. <no actions recorded>\n");
        }
        Ok(out)
    }

    /// Lines containing `needle` (case-insensitive), in the form
    /// `<trajectory> state <i>: <line>`, across every trajectory or within
    /// one. Bounded by [`GREP_MAX_LINES`] and [`GREP_MAX_STATES`], and says
    /// so when it was cut.
    pub async fn grep(&self, needle: &str, trajectory: Option<&str>) -> Result<String> {
        let states = self
            .ledger
            .trajectory_states_containing(&self.filter, needle, trajectory, GREP_MAX_STATES)
            .await?;
        let lower = needle.to_lowercase();
        let mut lines = Vec::new();
        let mut truncated = states.len() as u32 == GREP_MAX_STATES;
        'states: for (traj, st) in &states {
            for line in state_lines(st) {
                if line.to_lowercase().contains(&lower) {
                    if lines.len() == GREP_MAX_LINES {
                        truncated = true;
                        break 'states;
                    }
                    lines.push(format!("{traj} state {}: {}", st.state_index, clip(line, GREP_LINE_MAX_CHARS)));
                }
            }
        }
        if lines.is_empty() {
            return Ok(format!("no state contains {needle:?}\n"));
        }
        let mut out = lines.join("\n");
        out.push('\n');
        if truncated {
            out.push_str("(more matches not shown; narrow the search or name a trajectory)\n");
        }
        Ok(out)
    }

    /// A window of one state: its URL and action, then page lines
    /// `from_line..from_line + READ_WINDOW_LINES`, numbered.
    pub async fn read(&self, id: &str, state: u32, from_line: usize) -> Result<String> {
        let found = self
            .ledger
            .trajectory_states(&self.filter, id, state, state)
            .await?;
        let Some(st) = found.first() else {
            self.header(id).await?;
            return Err(MyelinError::Store(format!("trajectory {id} has no state {state}")));
        };
        let page: Vec<&str> = st.accessibility_tree.lines().collect();
        if from_line >= page.len().max(1) {
            return Err(MyelinError::Store(format!(
                "state {state} of {id} has {} page lines; line {from_line} is past the end",
                page.len()
            )));
        }
        let to = (from_line + READ_WINDOW_LINES).min(page.len());
        let mut out = format!(
            "Trajectory {id} state {state} (step {})\nURL: {}\nAction: {}\nPage lines {from_line}-{} of {}:\n",
            st.step,
            st.url,
            st.action.as_deref().unwrap_or("<none>"),
            to.saturating_sub(1),
            page.len()
        );
        for (i, line) in page[from_line..to].iter().enumerate() {
            out.push_str(&format!("{:>5} {line}\n", from_line + i));
        }
        Ok(out)
    }

    /// The reader's evidence: the controller's notes, the list of spans, and
    /// each span with its header and every state, as AgentRunbook-C's
    /// `_build_memory_context_from_output` lays them out.
    ///
    /// Refuses more than [`MAX_TOTAL_SPAN_STATES`] states in all, a reversed
    /// span, an unknown trajectory and a state the trajectory does not have.
    /// The authors drop such a span and carry on; here the controller is told
    /// and may correct it, so no answer is built on a span that was silently
    /// discarded.
    pub async fn evidence(&self, notes: &str, spans: &[SpanRequest]) -> Result<EvidenceSet> {
        let mut total = 0u32;
        for sp in spans {
            if sp.last < sp.first {
                return Err(MyelinError::Store(format!(
                    "span {} states {}-{} is reversed",
                    sp.trajectory, sp.first, sp.last
                )));
            }
            total += sp.last - sp.first + 1;
        }
        if total > MAX_TOTAL_SPAN_STATES {
            return Err(MyelinError::Store(format!(
                "{total} states requested; at most {MAX_TOTAL_SPAN_STATES} in all"
            )));
        }

        let mut headers: HashMap<String, TrajectoryHeader> = HashMap::new();
        let mut trust: HashMap<Uuid, TrustTier> = HashMap::new();
        let mut rendered: Vec<(SpanRequest, TrajectoryHeader, Vec<TrajectoryStep>, Vec<TrajectoryState>)> =
            Vec::new();
        for sp in spans {
            let h = match headers.get(&sp.trajectory) {
                Some(h) => h.clone(),
                None => {
                    let h = self.header(&sp.trajectory).await?;
                    headers.insert(sp.trajectory.clone(), h.clone());
                    h
                }
            };
            if let std::collections::hash_map::Entry::Vacant(slot) = trust.entry(h.record_id) {
                let anchor = self.ledger.get(h.record_id).await?.ok_or_else(|| {
                    MyelinError::Store(format!("anchor record {} of {} is gone", h.record_id, h.id))
                })?;
                slot.insert(anchor.trust.tier);
            }
            let states = self
                .ledger
                .trajectory_states(&self.filter, &sp.trajectory, sp.first, sp.last)
                .await?;
            let want = (sp.last - sp.first + 1) as usize;
            if states.len() != want {
                return Err(MyelinError::Store(format!(
                    "span {} states {}-{}: the trajectory holds {} of those {want} states",
                    sp.trajectory,
                    sp.first,
                    sp.last,
                    states.len()
                )));
            }
            let steps = self.ledger.trajectory_steps(&self.filter, &sp.trajectory).await?;
            rendered.push((sp.clone(), h, steps, states));
        }

        let mut items: Vec<EvidenceItem> = Vec::new();
        let mut span_items: Vec<EvidenceItem> = Vec::new();
        for (idx, (sp, h, steps, states)) in rendered.iter().enumerate() {
            let tier = trust[&h.record_id];
            let item = |value: String, first: u32, last: u32| EvidenceItem {
                kind: EvidenceKind::Text,
                value,
                record_id: h.record_id,
                source: SourceRef::span(h.id.clone(), first, last),
                score: 0.0,
                trust: tier,
            };
            span_items.push(item(format_span_header(idx + 1, h, steps, sp.first, sp.last), sp.first, sp.last));
            for st in states {
                span_items.push(item(format_state_text(st), st.state_index, st.state_index));
            }
        }
        if !notes.trim().is_empty() {
            items.push(EvidenceItem {
                kind: EvidenceKind::Text,
                value: format!("{}\n", notes.trim()),
                record_id: Uuid::nil(),
                source: SourceRef::doc(NOTES_MECHANISM),
                score: 0.0,
                trust: weakest_trust(span_items.iter().map(|i| i.trust)),
            });
        }
        if !rendered.is_empty() {
            let mut list = String::from("## Trajectory State Spans");
            for (sp, ..) in &rendered {
                list.push_str(&format!("\n- {}: states {}-{}", sp.trajectory, sp.first, sp.last));
            }
            let view = |value: String| EvidenceItem {
                kind: EvidenceKind::Text,
                value,
                record_id: Uuid::nil(),
                source: SourceRef::doc(NOTES_MECHANISM),
                score: 0.0,
                trust: weakest_trust(span_items.iter().map(|i| i.trust)),
            };
            items.push(view(list + "\n"));
            items.push(view("## Linked Evidence\n".into()));
        }
        items.extend(span_items);
        let tokens = items.iter().map(|i| approx_tokens(&i.value)).sum();
        Ok(EvidenceSet {
            items,
            tokens,
            ..EvidenceSet::default()
        })
    }
}

fn has_action(s: &TrajectoryStep) -> bool {
    s.action.as_deref().is_some_and(|a| !a.trim().is_empty())
}

fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clip(s: &str, max_chars: usize) -> String {
    let t = s.trim();
    match t.char_indices().nth(max_chars) {
        Some((cut, _)) => format!("{}…", &t[..cut]),
        None => t.to_string(),
    }
}

/// The searchable lines of a state: its URL, its action, then its page.
fn state_lines(st: &TrajectoryState) -> impl Iterator<Item = &str> {
    std::iter::once(st.url.as_str())
        .chain(st.action.as_deref())
        .chain(st.accessibility_tree.lines())
}

/// `codex.py` `format_actions`: every non-empty action, numbered.
fn format_actions(steps: &[TrajectoryStep]) -> String {
    let actions: Vec<&str> = steps
        .iter()
        .filter(|s| has_action(s))
        .filter_map(|s| s.action.as_deref())
        .collect();
    if actions.is_empty() {
        return "1. <no actions recorded>".into();
    }
    actions
        .iter()
        .enumerate()
        .map(|(i, a)| format!("{}. {a}", i + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

/// `codex.py` `format_span_header`.
fn format_span_header(
    span_index: usize,
    h: &TrajectoryHeader,
    steps: &[TrajectoryStep],
    first: u32,
    last: u32,
) -> String {
    format!(
        "### Trajectory span {span_index}: {} states {first}-{last}\n\nGoal\n- {}\n\nStart URL\n- {}\n\nActions\n{}\n\nLinked state evidence\n",
        h.id,
        h.goal,
        h.start_url,
        format_actions(steps)
    )
}

/// `codex.py` `format_state_text` in the `axtree` evidence mode M54 fixed.
fn format_state_text(st: &TrajectoryState) -> String {
    let action = st
        .action
        .as_deref()
        .filter(|a| !a.trim().is_empty())
        .unwrap_or("<none>");
    format!(
        "State {} (step {})\n- URL: {}\n- Action: {action}\n- AXTree:\n{}\n",
        st.state_index, st.step, st.url, st.accessibility_tree
    )
}
