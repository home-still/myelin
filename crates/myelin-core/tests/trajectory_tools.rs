//! M62: the trajectory controller's tools and the evidence they hand the
//! reader (`myelin_core::pipeline::trajectory_tools`). Hermetic: SQLite only.

mod trajectory_fixture;

use myelin_core::model::evidence::EvidenceKind;
use myelin_core::model::record::{ActorId, SourceRef, TrustTier};
use myelin_core::pipeline::trajectory_tools::{
    SpanRequest, TrajectoryTools, GREP_MAX_LINES, MAX_TOTAL_SPAN_STATES, NOTES_MECHANISM,
    READ_WINDOW_LINES,
};
use trajectory_fixture::{filter, goal_record, scope, state, stored, t1};

fn span(t: &str, first: u32, last: u32) -> SpanRequest {
    SpanRequest {
        trajectory: t.into(),
        first,
        last,
    }
}

#[tokio::test]
async fn list_names_every_trajectory_sorted_by_start_url() {
    let (ledger, ..) = stored().await;
    let tools = TrajectoryTools::new(&ledger, filter());
    let out = tools.list().await.unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "2 trajectories (id | start URL | outcome | goal)");
    // :7770 sorts before :9080.
    assert!(lines[1].starts_with("- t2 | http://localhost:7770/ | failure |"), "{out}");
    assert!(lines[2].starts_with("- t1 | http://localhost:9080/ | success |"), "{out}");
}

#[tokio::test]
async fn summary_numbers_the_actions_and_names_their_states() {
    let (ledger, ..) = stored().await;
    let tools = TrajectoryTools::new(&ledger, filter());
    let out = tools.summary("t1").await.unwrap();
    assert!(out.starts_with("Trajectory t1 (states 0-3)\n"), "{out}");
    assert!(out.contains("Outcome: success"), "{out}");
    assert!(out.contains("1. click('68') -> state 1"), "{out}");
    assert!(out.contains("3. click('250') -> state 3"), "{out}");
    assert!(!out.contains("4."), "the initial state has no action: {out}");
    assert!(tools.summary("nope").await.is_err());
}

#[tokio::test]
async fn grep_finds_lines_case_insensitively_across_or_within_trajectories() {
    let (ledger, ..) = stored().await;
    let tools = TrajectoryTools::new(&ledger, filter());

    let everywhere = tools.grep("BIOGRAPHY", None).await.unwrap();
    assert!(everywhere.contains("t1 state 2 line 1: [116] textbox 'Biography'"), "{everywhere}");
    assert!(everywhere.contains("t2 state 1 action: fill('5', 'biography')"), "{everywhere}");

    let within = tools.grep("biography", Some("t2")).await.unwrap();
    assert!(!within.contains("t1 state"), "{within}");

    let none = tools.grep("zebra", None).await.unwrap();
    assert_eq!(none, "no state contains \"zebra\"\n");
    assert!(tools.grep("   ", None).await.is_err(), "an empty search matches everything");
}

#[tokio::test]
async fn grep_is_bounded_and_says_when_it_was_cut() {
    let (ledger, ..) = stored().await;
    let actor = ActorId::new("builder");
    let goal = goal_record(&scope(), "long");
    let mut long = t1(goal.id);
    long.header.id = "long".into();
    let page: String = (0..GREP_MAX_LINES + 10)
        .map(|i| format!("[{i}] link 'Save draft {i}'\n"))
        .collect();
    long.states = vec![state(0, None, &page)];
    ledger
        .apply(
            &myelin_core::model::delta::Delta::Add { record: Box::new(goal) },
            &actor,
        )
        .await
        .unwrap();
    ledger.put_trajectory(&long, &actor).await.unwrap();

    let tools = TrajectoryTools::new(&ledger, filter());
    let out = tools.grep("save draft", Some("long")).await.unwrap();
    let hits = out.lines().filter(|l| l.starts_with("long state 0 ")).count();
    assert_eq!(hits, GREP_MAX_LINES);
    assert!(out.ends_with("(more matches not shown; narrow the search or name a trajectory)\n"));
}

#[tokio::test]
async fn read_returns_a_numbered_window_and_refuses_past_the_end() {
    let (ledger, ..) = stored().await;
    let tools = TrajectoryTools::new(&ledger, filter());
    let out = tools.read("t1", 2, 0).await.unwrap();
    assert!(out.starts_with("Trajectory t1 state 2 (step 2)\nURL: http://localhost:9080/page2\n"), "{out}");
    assert!(out.contains("Action: fill('116', 'I am a robot')"), "{out}");
    assert!(out.contains("Thought: I will fill('116', 'I am a robot')"), "the recorded thought is shown: {out}");
    assert!(out.contains("Page lines 0-1 of 2:"), "{out}");
    assert!(out.contains("    1 [116] textbox 'Biography'"), "{out}");
    const { assert!(READ_WINDOW_LINES > 2) };

    assert!(tools.read("t1", 2, 2).await.is_err(), "line 2 of a 2-line page");
    assert!(tools.read("t1", 9, 0).await.is_err(), "no state 9");
    assert!(tools.read("nope", 0, 0).await.is_err());
}

#[tokio::test]
async fn evidence_is_laid_out_as_agentrunbook_c_and_keeps_provenance() {
    let (ledger, t1, t2) = stored().await;
    let tools = TrajectoryTools::new(&ledger, filter());
    let set = tools
        .evidence("The bio was set in t1 state 2.", &[span("t1", 1, 2), span("t2", 2, 2)])
        .await
        .unwrap();

    let values: Vec<&str> = set.items.iter().map(|i| i.value.as_str()).collect();
    // notes, span list, "Linked Evidence", then per span: header + states.
    assert_eq!(values.len(), 3 + (1 + 2) + (1 + 1));
    assert_eq!(values[0], "The bio was set in t1 state 2.\n");
    assert_eq!(values[1], "## Trajectory State Spans\n- t1: states 1-2\n- t2: states 2-2\n");
    assert_eq!(values[2], "## Linked Evidence\n");
    assert!(values[3].starts_with("### Trajectory span 1: t1 states 1-2\n\nGoal\n- change the bio of t1\n"));
    assert!(values[3].contains("Actions\n1. click('68')\n2. fill('116', 'I am a robot')\n3. click('250')\n"));
    assert_eq!(
        values[5],
        "State 2 (step 2)\n- URL: http://localhost:9080/page2\n- Action: fill('116', 'I am a robot')\n- AXTree:\n[1] RootWebArea 'Edit biography'\n[116] textbox 'Biography' value='I am a robot'\n"
    );
    assert!(values[6].starts_with("### Trajectory span 2: t2 states 2-2\n"));

    // Retrieved text cites its trajectory's anchor and state range; the
    // controller's own words are a view with no record behind them.
    let state2 = &set.items[5];
    assert_eq!(state2.record_id, t1.header.record_id);
    assert_eq!(state2.source, SourceRef::span("t1", 2, 2));
    assert_eq!(set.items[7].record_id, t2.header.record_id);
    for view in &set.items[..3] {
        assert!(view.record_id.is_nil());
        assert_eq!(view.source, SourceRef::doc(NOTES_MECHANISM));
        assert_eq!(view.trust, TrustTier::Asserted, "weakest of the spans it frames");
    }
    assert!(set.items.iter().all(|i| i.kind == EvidenceKind::Text));
    assert!(set.tokens > 0);
}

#[tokio::test]
async fn evidence_refuses_spans_it_cannot_honour() {
    let (ledger, ..) = stored().await;
    let tools = TrajectoryTools::new(&ledger, filter());
    let too_many: Vec<SpanRequest> = (0..=MAX_TOTAL_SPAN_STATES)
        .map(|_| span("t1", 0, 0))
        .collect();
    let refused = tools.evidence("", &too_many).await;
    assert!(refused.as_ref().is_err_and(|e| e.to_string().contains("at most 20")), "{refused:?}");
    assert!(tools.evidence("", &[span("t1", 2, 1)]).await.is_err(), "reversed");
    let past = tools.evidence("", &[span("t1", 2, 9)]).await;
    assert!(
        past.as_ref().is_err_and(|e| e.to_string().contains("holds 2 of those 8")),
        "{past:?}"
    );
    assert!(tools.evidence("", &[span("nope", 0, 0)]).await.is_err());

    // Nothing found is an answer too: the notes alone, least trusted.
    let empty = tools.evidence("No trajectory mentions it.", &[]).await.unwrap();
    assert_eq!(empty.items.len(), 1);
    assert_eq!(empty.items[0].trust, TrustTier::Untrusted);
}
