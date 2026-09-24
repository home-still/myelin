//! M62: agent trajectories stored state by state, beside the episodic
//! records (`docs/measurements/m62-native-trajectory-tools.md`).
//!
//! What the trajectory tools rely on: a trajectory reads back exactly and in
//! state order; it is visible exactly when its anchor record is; it is never
//! replaced or edited; forgetting the anchor forgets it; and an export carries
//! it. Hermetic: SQLite only.

use chrono::{Duration, Utc};
use myelin_core::model::delta::Delta;
use myelin_core::model::query::ScopeFilter;
use myelin_core::model::record::{
    ActorId, MemoryRecord, Provenance, RecordKind, Salience, Scope, SourceRef, Trust, Validity,
};
use myelin_core::model::trajectory::{AgentTrajectory, TrajectoryHeader, TrajectoryState};
use myelin_core::store::export::{export_namespace, import_namespace, MemoryConfigJson};
use myelin_core::store::ids::record_id;
use myelin_core::store::ledger::Ledger;

const TENANT: &str = "small/web";
const NS: &str = "myelin";

fn scope() -> Scope {
    Scope::new(TENANT, "myelin", NS)
}

fn goal_record(sc: &Scope, traj: &str) -> MemoryRecord {
    let now = Utc::now();
    MemoryRecord {
        id: record_id(sc, &format!("{traj}#goal")),
        kind: RecordKind::Episodic,
        scope: sc.clone(),
        text: format!("goal: Goal: change the bio of {traj}"),
        entities: Vec::new(),
        validity: Validity {
            t_valid: now - Duration::hours(1),
            t_invalid: None,
            t_ingested: now - Duration::hours(1),
            t_expired: None,
        },
        provenance: Provenance {
            source: SourceRef::doc(format!("{traj}:goal")),
            contributed_by: ActorId::new("builder"),
            written_by: ActorId::new("builder"),
            derived_from: Vec::new(),
        },
        trust: Trust::asserted(),
        salience: Salience::default(),
        links: Vec::new(),
    }
}

fn state(i: u32, action: Option<&str>) -> TrajectoryState {
    TrajectoryState {
        state_index: i,
        step: i,
        url: format!("http://localhost:9080/page{i}"),
        action: action.map(str::to_string),
        thought: action.map(|a| format!("I will {a}")),
        accessibility_tree: format!("[1] RootWebArea 'Page {i}'\n[2] button 'Save'"),
    }
}

fn trajectory(sc: &Scope, id: &str, anchor: uuid::Uuid) -> AgentTrajectory {
    AgentTrajectory {
        header: TrajectoryHeader {
            id: id.to_string(),
            scope: sc.clone(),
            record_id: anchor,
            goal: format!("change the bio of {id}"),
            environment: "reddit".into(),
            start_url: "http://localhost:9080/".into(),
            outcome: "success".into(),
        },
        states: vec![
            state(0, None),
            state(1, Some("click('68')")),
            state(2, Some("fill('116', 'I am a robot')")),
            state(3, Some("click('250')")),
        ],
    }
}

/// A ledger holding trajectory `t1` in [`scope`], anchored on its goal record.
async fn stored() -> (Ledger, AgentTrajectory) {
    let ledger = Ledger::open_memory().await.unwrap();
    let actor = ActorId::new("builder");
    let goal = goal_record(&scope(), "t1");
    let traj = trajectory(&scope(), "t1", goal.id);
    ledger
        .apply(&Delta::Add { record: Box::new(goal) }, &actor)
        .await
        .unwrap();
    ledger.put_trajectory(&traj, &actor).await.unwrap();
    (ledger, traj)
}

fn filter() -> ScopeFilter {
    ScopeFilter::tenant(TENANT).with_namespace(NS)
}

#[tokio::test]
async fn a_trajectory_reads_back_exactly_and_in_state_order() {
    let (ledger, traj) = stored().await;

    let headers = ledger.trajectories(&filter()).await.unwrap();
    assert_eq!(headers, vec![traj.header.clone()]);

    let steps = ledger.trajectory_steps(&filter(), "t1").await.unwrap();
    let indices: Vec<u32> = steps.iter().map(|s| s.state_index).collect();
    assert_eq!(indices, vec![0, 1, 2, 3]);
    assert_eq!(steps[0].action, None, "the initial state has no action");
    assert_eq!(steps[2].action.as_deref(), Some("fill('116', 'I am a robot')"));

    let span = ledger.trajectory_states(&filter(), "t1", 1, 2).await.unwrap();
    assert_eq!(span, traj.states[1..=2].to_vec());
    let past_the_end = ledger.trajectory_states(&filter(), "t1", 3, 99).await.unwrap();
    assert_eq!(past_the_end, traj.states[3..].to_vec());
}

#[tokio::test]
async fn a_trajectory_is_visible_exactly_when_its_anchor_is() {
    let (ledger, traj) = stored().await;

    // Another tenant, another namespace, another agent: nothing.
    let other_tenant = ScopeFilter::tenant("small/enterprise");
    assert!(ledger.trajectories(&other_tenant).await.unwrap().is_empty());
    assert!(ledger.trajectory_steps(&other_tenant, "t1").await.unwrap().is_empty());
    assert!(ledger
        .trajectory_states(&other_tenant, "t1", 0, 3)
        .await
        .unwrap()
        .is_empty());
    let other_ns = ScopeFilter::tenant(TENANT).with_namespace("events");
    assert!(ledger.trajectories(&other_ns).await.unwrap().is_empty());
    let other_agent = filter().with_agent("someone-else");
    assert!(ledger.trajectories(&other_agent).await.unwrap().is_empty());

    // Retracting the anchor hides the trajectory from every read.
    ledger
        .apply(
            &Delta::Delete {
                target: traj.header.record_id,
                reason: "retracted".into(),
            },
            &ActorId::new("user"),
        )
        .await
        .unwrap();
    assert!(ledger.trajectories(&filter()).await.unwrap().is_empty());
    assert!(ledger.trajectory_steps(&filter(), "t1").await.unwrap().is_empty());
    assert!(ledger
        .trajectory_states(&filter(), "t1", 0, 3)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn bad_writes_are_refused() {
    let (ledger, traj) = stored().await;
    let actor = ActorId::new("builder");

    let again = ledger.put_trajectory(&traj, &actor).await;
    assert!(
        again.as_ref().is_err_and(|e| e.to_string().contains("already stored")),
        "{again:?}"
    );

    let orphan = trajectory(&scope(), "t2", record_id(&scope(), "nowhere"));
    let refused = ledger.put_trajectory(&orphan, &actor).await;
    assert!(
        refused.as_ref().is_err_and(|e| e.to_string().contains("not in the ledger")),
        "{refused:?}"
    );

    // An anchor in another tenant cannot vouch for this one.
    let elsewhere = Scope::new("small/enterprise", "myelin", NS);
    let foreign_goal = goal_record(&elsewhere, "t3");
    let foreign_id = foreign_goal.id;
    ledger
        .apply(&Delta::Add { record: Box::new(foreign_goal) }, &actor)
        .await
        .unwrap();
    let cross = ledger
        .put_trajectory(&trajectory(&scope(), "t3", foreign_id), &actor)
        .await;
    assert!(
        cross.as_ref().is_err_and(|e| e.to_string().contains("is in small/enterprise")),
        "{cross:?}"
    );

    let mut shuffled = trajectory(&scope(), "t4", traj.header.record_id);
    shuffled.states.swap(1, 2);
    let unordered = ledger.put_trajectory(&shuffled, &actor).await;
    assert!(
        unordered.as_ref().is_err_and(|e| e.to_string().contains("strictly increase")),
        "{unordered:?}"
    );

    let mut empty = trajectory(&scope(), "t5", traj.header.record_id);
    empty.states.clear();
    assert!(ledger.put_trajectory(&empty, &actor).await.is_err());

    let reversed = ledger.trajectory_states(&filter(), "t1", 3, 1).await;
    assert!(
        reversed.as_ref().is_err_and(|e| e.to_string().contains("reversed")),
        "{reversed:?}"
    );

    // None of the refused writes left anything behind.
    assert_eq!(ledger.trajectories(&filter()).await.unwrap().len(), 1);
}

#[tokio::test]
async fn a_stored_trajectory_is_never_edited() {
    let (ledger, _) = stored().await;
    let edit = sqlx::query("UPDATE trajectory SET goal = 'something else' WHERE id = 't1'")
        .execute(ledger.pool())
        .await;
    assert!(
        edit.as_ref().is_err_and(|e| e.to_string().contains("never mutated")),
        "{edit:?}"
    );
    let edit_state =
        sqlx::query("UPDATE trajectory_state SET action = 'x' WHERE trajectory = 't1'")
            .execute(ledger.pool())
            .await;
    assert!(edit_state.is_err(), "{edit_state:?}");
}

#[tokio::test]
async fn forgetting_the_anchor_forgets_the_trajectory() {
    let (ledger, traj) = stored().await;
    ledger
        .hard_delete(traj.header.record_id, &ActorId::new("user"), "forget me")
        .await
        .unwrap();
    assert!(ledger.trajectories_in_namespace(NS).await.unwrap().is_empty());
    let (states,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM trajectory_state")
        .fetch_one(ledger.pool())
        .await
        .unwrap();
    assert_eq!(states, 0, "the states cascade with their trajectory");
}

#[tokio::test]
async fn an_export_carries_trajectories_and_reimports_them() {
    let (ledger, traj) = stored().await;
    let config = MemoryConfigJson::new("bge-m3", 1024);
    let bundle = export_namespace(&ledger, NS, &config).await.unwrap();
    assert_eq!(bundle.trajectories, vec![traj.clone()]);

    let fresh = Ledger::open_memory().await.unwrap();
    import_namespace(&fresh, &bundle, &config).await.unwrap();
    assert_eq!(fresh.trajectories_in_namespace(NS).await.unwrap(), vec![traj]);
    let again = export_namespace(&fresh, NS, &config).await.unwrap();
    assert_eq!(again, bundle, "export -> import -> export is byte-faithful");
}
