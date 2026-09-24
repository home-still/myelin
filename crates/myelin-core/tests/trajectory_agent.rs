//! M62: the trajectory controller's loop, against a scripted model.
//! Hermetic: SQLite only; the "model" replays fixed JSON actions and records
//! every request it was sent.

mod trajectory_fixture;

use std::sync::Mutex;

use async_trait::async_trait;
use myelin_core::error::{MyelinError, Result};
use myelin_core::llm::{Completion, CompletionRequest, Llm, Role, Usage};
use myelin_core::pipeline::trajectory_agent::{
    TrajectoryAgent, TrajectoryAgentConfig, FORCED_ANSWER_ATTEMPTS,
};
use myelin_core::pipeline::trajectory_tools::TrajectoryTools;
use trajectory_fixture::{filter, stored};

struct Scripted {
    replies: Mutex<Vec<String>>,
    seen: Mutex<Vec<CompletionRequest>>,
}

impl Scripted {
    fn new(replies: &[serde_json::Value]) -> Self {
        let mut r: Vec<String> = replies.iter().map(|v| v.to_string()).collect();
        r.reverse();
        Self {
            replies: Mutex::new(r),
            seen: Mutex::new(Vec::new()),
        }
    }
    fn requests(&self) -> Vec<CompletionRequest> {
        self.seen.lock().expect("lock").clone()
    }
}

#[async_trait]
impl Llm for Scripted {
    fn id(&self) -> &str {
        "scripted"
    }
    async fn raw_complete(&self, req: &CompletionRequest) -> Result<Completion> {
        self.seen.lock().expect("lock").push(req.clone());
        let text = self
            .replies
            .lock()
            .expect("lock")
            .pop()
            .ok_or_else(|| MyelinError::Store("script exhausted".into()))?;
        Ok(Completion {
            text,
            reasoning: None,
            tool_calls: Vec::new(),
            finish_reason: Some("stop".into()),
            usage: Usage::default(),
        })
    }
}

fn tools_of(req: &CompletionRequest) -> Vec<String> {
    req.json_schema.as_ref().expect("every step is schema-constrained")["properties"]["tool"]["enum"]
        .as_array()
        .expect("enum")
        .iter()
        .map(|v| v.as_str().expect("str").to_string())
        .collect()
}

#[tokio::test]
async fn a_search_read_answer_run_yields_the_spans_and_its_trace() {
    let (ledger, t1, _) = stored().await;
    let llm = Scripted::new(&[
        serde_json::json!({"thought": "the bio edit is t1", "tool": "grep", "pattern": "Biography|robot", "trajectory": "t1"}),
        serde_json::json!({"thought": "check the page", "tool": "read", "trajectory": "t1", "state": 2}),
        serde_json::json!({"thought": "found it", "tool": "answer",
            "memory_markdown": "## Support Analysis\nt1 state 2 shows the bio.",
            "spans": [{"trajectory": "t1", "first": 2, "last": 2}]}),
    ]);
    let agent = TrajectoryAgent::new(&llm, TrajectoryTools::new(&ledger, filter()), TrajectoryAgentConfig::default());
    let set = agent.run("What bio did I set?").await.unwrap();

    assert_eq!(set.items[0].value, "## Support Analysis\nt1 state 2 shows the bio.\n");
    let state = set.items.iter().find(|i| i.value.starts_with("State 2 (step 2)")).expect("state 2");
    assert_eq!(state.record_id, t1.header.record_id);
    let actions: Vec<(&str, &str)> = set.trace.iter().map(|t| (t.action.as_str(), t.query.as_str())).collect();
    assert_eq!(
        actions,
        vec![
            ("grep", "Biography|robot in t1"),
            ("read", "t1 state 2 from 0"),
            ("answer", "t1:2-2")
        ]
    );
    assert!(set.trace[0].hits >= 2, "both alternatives matched: {:?}", set.trace[0]);

    let requests = llm.requests();
    assert_eq!(requests.len(), 3);
    let first = &requests[0];
    assert_eq!(first.messages[0].role, Role::System);
    assert!(first.messages[0].content.contains("Reject nearby-but-not-exact matches"));
    assert!(first.messages[1].content.contains("<question>\nWhat bio did I set?\n</question>"));
    assert!(first.messages[1].content.contains("- t1 | http://localhost:9080/ | success |"), "the list comes up front");
    assert_eq!(tools_of(first), vec!["summary", "grep", "read", "answer"]);
    // The model's action and the tool's observation alternate after that.
    let third = &requests[2];
    assert_eq!(third.messages.len(), 6);
    assert_eq!(third.messages[2].role, Role::Assistant);
    assert!(third.messages[3].content.contains("t1 state 2: [116] textbox 'Biography'"));
    assert!(third.messages[5].content.starts_with("Trajectory t1 state 2 (step 2)"));
}

#[tokio::test]
async fn a_misused_tool_or_refused_answer_is_fed_back_for_correction() {
    let (ledger, ..) = stored().await;
    let llm = Scripted::new(&[
        serde_json::json!({"thought": "read it", "tool": "read", "trajectory": "t1"}),
        serde_json::json!({"thought": "wrong id", "tool": "summary", "trajectory": "t9"}),
        serde_json::json!({"thought": "too far", "tool": "answer", "memory_markdown": "x",
            "spans": [{"trajectory": "t1", "first": 2, "last": 9}]}),
        serde_json::json!({"thought": "fixed", "tool": "answer", "memory_markdown": "x",
            "spans": [{"trajectory": "t1", "first": 2, "last": 3}]}),
    ]);
    let agent = TrajectoryAgent::new(&llm, TrajectoryTools::new(&ledger, filter()), TrajectoryAgentConfig::default());
    let set = agent.run("q").await.unwrap();
    assert!(set.items.iter().any(|i| i.value.starts_with("State 3 (step 3)")));

    let last = llm.requests().pop().expect("request").messages;
    let observations: Vec<&str> = last
        .iter()
        .filter(|m| m.role == Role::User)
        .skip(1)
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(observations[0], "error: read needs a trajectory and a state");
    assert!(observations[1].starts_with("error: no trajectory \"t9\""), "{:?}", observations[1]);
    assert!(observations[2].contains("holds 2 of those 8 states"), "{:?}", observations[2]);
}

#[tokio::test]
async fn a_spent_budget_forces_an_answer_only_step() {
    let (ledger, ..) = stored().await;
    let llm = Scripted::new(&[
        serde_json::json!({"thought": "look", "tool": "grep", "pattern": "Save"}),
        serde_json::json!({"thought": "look more", "tool": "grep", "pattern": "Forums"}),
        serde_json::json!({"thought": "must answer", "tool": "answer", "memory_markdown": "## Support Analysis\nNothing exact.", "spans": []}),
    ]);
    let config = TrajectoryAgentConfig {
        max_steps: 2,
        ..TrajectoryAgentConfig::default()
    };
    let agent = TrajectoryAgent::new(&llm, TrajectoryTools::new(&ledger, filter()), config);
    let set = agent.run("q").await.unwrap();
    assert_eq!(set.items.len(), 1, "notes alone: no spans named");

    let requests = llm.requests();
    assert_eq!(tools_of(&requests[1]), vec!["summary", "grep", "read", "answer"]);
    assert_eq!(tools_of(&requests[2]), vec!["answer"], "the forced step allows only answer");
    let forced = &requests[2].messages;
    assert!(forced.last().expect("msg").content.starts_with("The exploration budget is spent."));
}

#[tokio::test]
async fn an_answer_still_refused_after_the_forced_attempts_is_an_error() {
    let (ledger, ..) = stored().await;
    let bad = serde_json::json!({"thought": "x", "tool": "answer", "memory_markdown": "x",
        "spans": [{"trajectory": "nope", "first": 0, "last": 0}]});
    let llm = Scripted::new(&vec![bad; FORCED_ANSWER_ATTEMPTS]);
    let config = TrajectoryAgentConfig {
        max_steps: 0,
        ..TrajectoryAgentConfig::default()
    };
    let agent = TrajectoryAgent::new(&llm, TrajectoryTools::new(&ledger, filter()), config);
    let err = agent.run("q").await.expect_err("no acceptable answer");
    assert!(err.to_string().contains("no acceptable answer"), "{err}");
    assert_eq!(llm.requests().len(), FORCED_ANSWER_ATTEMPTS);
}

#[tokio::test]
async fn a_reply_outside_the_schema_is_an_error_not_a_guess() {
    let (ledger, ..) = stored().await;
    // A server that ignored the schema: the tool is not one of the allowed.
    let llm = Scripted::new(&[serde_json::json!({"thought": "x", "tool": "shell"})]);
    let agent = TrajectoryAgent::new(&llm, TrajectoryTools::new(&ledger, filter()), TrajectoryAgentConfig::default());
    let err = agent.run("q").await.expect_err("schema not applied");
    assert!(err.to_string().contains("did not apply the schema"), "{err}");
}
