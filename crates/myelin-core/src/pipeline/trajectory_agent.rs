//! M62: myelin's own trajectory controller
//! (`docs/measurements/m62-native-trajectory-tools.md`).
//!
//! A model reads the stored agent trajectories through the bounded tools in
//! [`super::trajectory_tools`] and ends by naming spans, which become the
//! reader's evidence in AgentRunbook-C's layout. The rules it follows are the
//! LME-V2 authors' own controller instructions
//! (`vendor/longmemeval-v2/memory_modules/assets/agentrunbook_c/INSTRUCTION.md`,
//! `10.48550/arXiv.2605.12493` §4.2), rewritten for these tools in place of
//! a shell and files.
//!
//! **One schema-constrained action per step** (user decision, 2026-09-24).
//! The server forces every reply into [`action_schema`], so an action always
//! parses (grammar-constrained decoding: Geng et al., EMNLP 2023,
//! `10.18653/v1/2023.emnlp-main.674`), and the model states a thought before
//! each action (ReAct: Yao et al., ICLR 2023, `10.48550/arXiv.2210.03629`).
//! This is the path `investigate`'s `reflect()` already runs on every query.
//!
//! What the model gets wrong is fed back, not papered over. A tool the model
//! misuses (unknown trajectory, a state it lacks, a span over budget) answers
//! with the error as its observation and the model may correct itself. When
//! the step or context budget is spent the model must answer. An answer
//! still refused after [`FORCED_ANSWER_ATTEMPTS`] is an error for the caller,
//! never an empty or invented evidence set.

use serde::Deserialize;

use crate::error::{MyelinError, Result};
use crate::llm::{complete_json, CompletionRequest, Llm, Message};
use crate::model::evidence::{EvidenceSet, TraceStep};
use crate::pipeline::ingest::approx_tokens;
use crate::pipeline::trajectory_tools::{SpanRequest, TrajectoryTools, MAX_TOTAL_SPAN_STATES};

/// Tool calls before the answer is forced.
pub const DEFAULT_MAX_STEPS: usize = 16;
/// Completion budget of one action. An action is a short JSON object; the
/// answer carries the notes, so this is sized for the answer.
pub const DEFAULT_MAX_TOKENS_PER_STEP: u32 = 2048;
/// Transcript size at which exploration stops and the answer is forced, so
/// the controller never runs into its server's context window mid-search.
pub const DEFAULT_CONTEXT_BUDGET_TOKENS: usize = 48_000;
/// Answers tried once the budget is spent: the first, and one correction.
pub const FORCED_ANSWER_ATTEMPTS: usize = 2;
/// Upper bound on the `thought` field. llama.cpp's grammar compiler rejects
/// `maxLength` of 2000 and above (`myelin-eval` `build::MAX_SCHEMA_MAX_LENGTH`).
const THOUGHT_MAX_CHARS: u64 = 1500;
/// Sent when the budget is spent. M62's pilot: 36 of 47 answers were forced
/// and 19 named no span, several after the controller had found the right
/// state and not finished reading it ("the exploration budget was exhausted
/// before I could read the accessibility tree of state 2").
pub const FORCED_ANSWER_MESSAGE: &str = "The exploration budget is spent. Answer now. Name the spans \
of the states you identified, whether or not you finished reading them: the reader sees every named \
state in full.";
/// Separates alternatives in a `grep` pattern, as the authors'
/// `inspect_trajectory.py --match "Delete Review|Previous"` does.
const GREP_ALTERNATIVES: char = '|';

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrajectoryAgentConfig {
    pub max_steps: usize,
    pub max_tokens_per_step: u32,
    pub context_budget_tokens: usize,
    /// Let the controller think before each action. Off, like `reflect()`:
    /// the `thought` field is its reasoning channel.
    pub thinking: bool,
}

impl Default for TrajectoryAgentConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            max_tokens_per_step: DEFAULT_MAX_TOKENS_PER_STEP,
            context_budget_tokens: DEFAULT_CONTEXT_BUDGET_TOKENS,
            thinking: false,
        }
    }
}

/// The authors' rules (INSTRUCTION.md), for these tools.
const RULES: &str = "\
You are a fast memory retrieval module. Recorded agent trajectories from a customized web \
environment are stored; each is a goal, a start URL, an outcome, and ordered states (a page's \
accessibility tree, the URL, and the action taken next). Collect the evidence a downstream reader \
needs to answer the question, and nothing more. Be quick and do not over-explore.

Each reply is one JSON action: state your thought, then pick one tool.
- summary {trajectory}: that trajectory's goal, outcome and numbered actions, with the state each led to.
- grep {pattern, trajectory?}: lines containing the text, case-insensitive; separate alternatives \
with |. Name a trajectory to search only there.
- read {trajectory, state, from_line?}: a window of one state's page.
- answer {memory_markdown, spans}: finish.

Workflow.
1. Triage the question before opening anything. For a direct lookup, find one exact state showing the \
requested field, value, button or page, and prefer a single clean span. For a comparison, find the \
supporting state from one trajectory per side. For a procedure, stay within one workflow family unless \
the question asks for a pattern shared across workflows.
2. Shortlist a few likely trajectories from the list you are given (goal, start URL, outcome). Prefer \
the exact same product, page or workflow family over merely related ones. Verify with summary, grep \
within a shortlisted trajectory, and read.
3. If the evidence contradicts the question, its premise may be wrong (a nonexistent feature, step or \
procedure): say so plainly so the reader can abstain, and still include the contradicting evidence.
4. If the exact evidence is missing, incomplete or contradictory, do not extrapolate from numeric \
progressions, nearby rows, similar buttons or similar workflows. Preserve the uncertainty for the reader.

Answer.
- memory_markdown has two sections only. \"## Support Analysis\": where the supporting evidence is, \
pointing to the spans, or that the premise is wrong and where. \"## Relevant Procedure and Hint Notes\": \
relevant procedure and observations.
- spans: zero-based inclusive state indices, most important first. Usually no more than 3 states per \
span; at most 20 states across all spans. One span proves one point; avoid redundant trajectories.
- Reject nearby-but-not-exact matches: never substitute a similar field, row, tab, header, button or state.
- If you find no useful evidence, answer with a minimal memory_markdown and no spans.
- Naming a span is enough. The reader receives every state you name in full, so you do not need to \
read a whole page before naming it: read only to choose between candidate states. A state you \
identified but did not finish reading still belongs in your spans.";

/// The tools a controller step may name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Summary,
    Grep,
    Read,
    Answer,
}

impl ToolKind {
    pub fn name(self) -> &'static str {
        match self {
            ToolKind::Summary => "summary",
            ToolKind::Grep => "grep",
            ToolKind::Read => "read",
            ToolKind::Answer => "answer",
        }
    }

    /// This tool's action: its own required fields and nothing else.
    ///
    /// M62 and M62b used one flat object with every field optional and extra
    /// keys allowed. M62b's web half then issued 268 `read` actions, and 267
    /// came back "read needs a trajectory and a state": the grammar let the
    /// model omit `state` (or name it something else), and it did so again
    /// and again. One strict object per tool makes a `read` without a state
    /// unwritable (grammar-constrained decoding, Geng et al. 2023,
    /// `10.18653/v1/2023.emnlp-main.674`).
    fn schema(self) -> serde_json::Value {
        let thought = serde_json::json!({"type": "string", "maxLength": THOUGHT_MAX_CHARS});
        let (props, required): (serde_json::Value, Vec<&str>) = match self {
            ToolKind::Summary => (
                serde_json::json!({"trajectory": {"type": "string"}}),
                vec!["trajectory"],
            ),
            ToolKind::Grep => (
                serde_json::json!({"pattern": {"type": "string"}, "trajectory": {"type": "string"}}),
                vec!["pattern"],
            ),
            ToolKind::Read => (
                serde_json::json!({
                    "trajectory": {"type": "string"},
                    "state": {"type": "integer", "minimum": 0},
                    "from_line": {"type": "integer", "minimum": 0}
                }),
                vec!["trajectory", "state"],
            ),
            ToolKind::Answer => (
                serde_json::json!({
                    "memory_markdown": {"type": "string"},
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "trajectory": {"type": "string"},
                                "first": {"type": "integer", "minimum": 0},
                                "last": {"type": "integer", "minimum": 0}
                            },
                            "required": ["trajectory", "first", "last"],
                            "additionalProperties": false
                        }
                    }
                }),
                vec!["memory_markdown", "spans"],
            ),
        };
        let mut properties = serde_json::Map::new();
        properties.insert("thought".into(), thought);
        properties.insert("tool".into(), serde_json::json!({"const": self.name()}));
        if let serde_json::Value::Object(extra) = props {
            properties.extend(extra);
        }
        let mut req = vec!["thought", "tool"];
        req.extend(required);
        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": req,
            "additionalProperties": false
        })
    }
}

/// The action form every reply is constrained to: exactly one of `tools`'
/// strict objects. All of them while exploring, only `answer` once the
/// budget is spent.
pub fn action_schema(tools: &[ToolKind]) -> serde_json::Value {
    match tools {
        [only] => only.schema(),
        _ => serde_json::json!({"anyOf": tools.iter().map(|t| t.schema()).collect::<Vec<_>>()}),
    }
}

const EXPLORE_TOOLS: [ToolKind; 4] = [ToolKind::Summary, ToolKind::Grep, ToolKind::Read, ToolKind::Answer];
const ANSWER_ONLY: [ToolKind; 1] = [ToolKind::Answer];

#[derive(Debug, Clone, Deserialize)]
struct Action {
    thought: String,
    tool: String,
    trajectory: Option<String>,
    state: Option<u32>,
    from_line: Option<u32>,
    pattern: Option<String>,
    memory_markdown: Option<String>,
    spans: Option<Vec<SpanJson>>,
}

#[derive(Debug, Clone, Deserialize)]
struct SpanJson {
    trajectory: String,
    first: u32,
    last: u32,
}

/// What one action produced: an observation for the model, or the answer.
enum Outcome {
    Observation { text: String, hits: usize },
    Answered(EvidenceSet),
}

/// What one controller run produced.
#[derive(Debug, Clone)]
pub struct AgentRun {
    /// The reader's evidence, with the trace of every action.
    pub evidence: EvidenceSet,
    /// Whether the answer came from a budget-forced step.
    pub forced: bool,
    /// Observations that reported a misused tool or a refused answer.
    pub tool_errors: usize,
}

pub struct TrajectoryAgent<'a> {
    llm: &'a dyn Llm,
    tools: TrajectoryTools<'a>,
    config: TrajectoryAgentConfig,
}

impl<'a> TrajectoryAgent<'a> {
    pub fn new(llm: &'a dyn Llm, tools: TrajectoryTools<'a>, config: TrajectoryAgentConfig) -> Self {
        Self { llm, tools, config }
    }

    /// Explore, answer, and return the reader's evidence with the trace of
    /// every action, whether the answer was forced, and how many tool errors
    /// the model was shown.
    pub async fn run(&self, question: &str) -> Result<AgentRun> {
        let listing = self.tools.list().await?;
        let mut messages = vec![
            Message::system(RULES),
            Message::user(format!(
                "<question>\n{question}\n</question>\n<trajectories>\n{listing}</trajectories>"
            )),
        ];
        let mut trace: Vec<TraceStep> = Vec::new();
        let mut tool_errors = 0usize;

        for step in 0..self.config.max_steps {
            if transcript_tokens(&messages) > self.config.context_budget_tokens {
                break;
            }
            let (action, raw) = self.next_action(&messages, &EXPLORE_TOOLS).await?;
            match self.act(&action).await? {
                Outcome::Answered(mut set) => {
                    trace.push(trace_step(step, &action, 0));
                    set.trace = trace;
                    return Ok(AgentRun { evidence: set, forced: false, tool_errors });
                }
                Outcome::Observation { text, hits } => {
                    tool_errors += usize::from(is_error(&text));
                    trace.push(trace_step(step, &action, hits));
                    messages.push(Message::assistant(raw));
                    // BATS (Liu et al. 2025, `10.48550/arXiv.2511.17006`):
                    // agents lack budget awareness, so each observation says
                    // how many steps remain before the answer is forced.
                    let left = self.config.max_steps - step - 1;
                    messages.push(Message::user(format!("{text}\n(steps left before you must answer: {left})")));
                }
            }
        }

        messages.push(Message::user(
            FORCED_ANSWER_MESSAGE,
        ));
        let mut last_refusal = String::new();
        for attempt in 0..FORCED_ANSWER_ATTEMPTS {
            let step = self.config.max_steps + attempt;
            let (action, raw) = self.next_action(&messages, &ANSWER_ONLY).await?;
            match self.act(&action).await? {
                Outcome::Answered(mut set) => {
                    trace.push(trace_step(step, &action, 0));
                    set.trace = trace;
                    return Ok(AgentRun { evidence: set, forced: true, tool_errors });
                }
                Outcome::Observation { text, .. } => {
                    tool_errors += usize::from(is_error(&text));
                    trace.push(trace_step(step, &action, 0));
                    last_refusal = text.clone();
                    messages.push(Message::assistant(raw));
                    messages.push(Message::user(text));
                }
            }
        }
        Err(MyelinError::Store(format!(
            "trajectory controller: no acceptable answer after {FORCED_ANSWER_ATTEMPTS} forced attempts; last: {last_refusal}"
        )))
    }

    async fn next_action(&self, messages: &[Message], tools: &[ToolKind]) -> Result<(Action, String)> {
        let request = CompletionRequest::new(messages.to_vec())
            .with_schema(action_schema(tools))
            .with_max_tokens(self.config.max_tokens_per_step)
            .with_thinking(self.config.thinking);
        let action: Action = complete_json(self.llm, &request).await?;
        if !tools.iter().any(|t| t.name() == action.tool) {
            return Err(MyelinError::Store(format!(
                "trajectory controller chose {:?}, outside the allowed {:?}: the server did not apply the schema",
                action.tool,
                tools.iter().map(|t| t.name()).collect::<Vec<_>>()
            )));
        }
        // The transcript keeps the action as the model's own turn, re-serialised
        // so every assistant message is one clean JSON object.
        let raw = serde_json::to_string(&serde_json::json!({
            "thought": action.thought,
            "tool": action.tool,
            "trajectory": action.trajectory,
            "state": action.state,
            "from_line": action.from_line,
            "pattern": action.pattern,
            "memory_markdown": action.memory_markdown,
            "spans": action.spans.as_ref().map(|v| v.iter().map(|s| serde_json::json!({
                "trajectory": s.trajectory, "first": s.first, "last": s.last
            })).collect::<Vec<_>>()),
        }))?;
        Ok((action, raw))
    }

    /// Run one action. A misuse the model can correct comes back as an
    /// observation; only a store failure is an error.
    async fn act(&self, a: &Action) -> Result<Outcome> {
        let observe = |text: String, hits: usize| Ok(Outcome::Observation { text, hits });
        match a.tool.as_str() {
            "summary" => match &a.trajectory {
                Some(t) => self.tool_result(self.tools.summary(t).await),
                None => observe("error: summary needs a trajectory".into(), 0),
            },
            "grep" => {
                let Some(pattern) = a.pattern.as_deref() else {
                    return observe("error: grep needs a pattern".into(), 0);
                };
                let mut out = String::new();
                let mut hits = 0;
                for alt in pattern.split(GREP_ALTERNATIVES).map(str::trim).filter(|p| !p.is_empty()) {
                    match self.tools.grep(alt, a.trajectory.as_deref()).await {
                        Ok(text) => {
                            hits += text.lines().filter(|l| l.contains(" state ")).count();
                            out.push_str(&text);
                        }
                        Err(e) => out.push_str(&format!("error: {e}\n")),
                    }
                }
                if out.is_empty() {
                    out = "error: grep needs a non-empty pattern".into();
                }
                observe(out, hits)
            }
            "read" => match (&a.trajectory, a.state) {
                (Some(t), Some(s)) => {
                    let from = a.from_line.unwrap_or(0) as usize;
                    self.tool_result(self.tools.read(t, s, from).await)
                }
                _ => observe("error: read needs a trajectory and a state".into(), 0),
            },
            "answer" => {
                let spans: Vec<SpanRequest> = a
                    .spans
                    .iter()
                    .flatten()
                    .map(|s| SpanRequest {
                        trajectory: s.trajectory.clone(),
                        first: s.first,
                        last: s.last,
                    })
                    .collect();
                let notes = a.memory_markdown.as_deref().unwrap_or("");
                match self.tools.evidence(notes, &spans).await {
                    Ok(set) => Ok(Outcome::Answered(set)),
                    Err(MyelinError::Store(why)) => observe(
                        format!(
                            "error: the answer was refused: {why}. Correct the spans (at most {MAX_TOTAL_SPAN_STATES} states in all) and answer again."
                        ),
                        0,
                    ),
                    Err(e) => Err(e),
                }
            }
            other => Err(MyelinError::Store(format!(
                "trajectory controller chose unknown tool {other:?}: the server did not apply the schema"
            ))),
        }
    }

    /// A tool's misuse is the model's to correct; anything else propagates.
    fn tool_result(&self, r: Result<String>) -> Result<Outcome> {
        match r {
            Ok(text) => Ok(Outcome::Observation { text, hits: 1 }),
            Err(MyelinError::Store(why)) => Ok(Outcome::Observation {
                text: format!("error: {why}"),
                hits: 0,
            }),
            Err(e) => Err(e),
        }
    }
}

/// An observation that reports a misuse rather than tool output. `grep`
/// joins several alternatives, so an error can start any of its lines.
fn is_error(observation: &str) -> bool {
    observation.lines().any(|l| l.starts_with("error:"))
}

fn transcript_tokens(messages: &[Message]) -> usize {
    messages.iter().map(|m| approx_tokens(&m.content)).sum()
}

fn trace_step(step: usize, a: &Action, hits: usize) -> TraceStep {
    let query = match a.tool.as_str() {
        "grep" => format!(
            "{}{}",
            a.pattern.as_deref().unwrap_or(""),
            a.trajectory.as_deref().map(|t| format!(" in {t}")).unwrap_or_default()
        ),
        "read" => format!(
            "{} state {} from {}",
            a.trajectory.as_deref().unwrap_or("?"),
            a.state.map_or("?".into(), |s| s.to_string()),
            a.from_line.unwrap_or(0)
        ),
        "answer" => a
            .spans
            .iter()
            .flatten()
            .map(|s| format!("{}:{}-{}", s.trajectory, s.first, s.last))
            .collect::<Vec<_>>()
            .join(", "),
        _ => a.trajectory.clone().unwrap_or_default(),
    };
    TraceStep {
        step,
        action: a.tool.clone(),
        query,
        hits,
    }
}
