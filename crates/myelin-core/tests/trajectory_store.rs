//! M62: agent trajectories stored state by state, beside the episodic
//! records (`docs/measurements/m62-native-trajectory-tools.md`).
//!
//! What the trajectory tools rely on: a trajectory reads back exactly and in
//! state order; it is visible exactly when its anchor record is; it is never
//! replaced or edited; forgetting the anchor forgets it; and an export carries
//! it. Hermetic: SQLite only.

mod trajectory_fixture;

use myelin_core::model::delta::Delta;
use myelin_core::model::query::ScopeFilter;
use myelin_core::model::record::{ActorId, Scope};
use myelin_core::store::export::{export_namespace, import_namespace, MemoryConfigJson};
use myelin_core::store::ids::record_id;
use myelin_core::store::ledger::Ledger;
use trajectory_fixture::{filter, goal_record, scope, t1, NS, TENANT};

/// The fixture ledger, keeping only `t1` for the tests that follow one
/// trajectory.
async fn stored() -> (Ledger, myelin_core::model::trajectory::AgentTrajectory) {
    let (ledger, t1, _t2) = trajectory_fixture::stored().await;
    (ledger, t1)
}

#[tokio::test]
async fn a_trajectory_reads_back_exactly_and_in_state_order() {
    let (ledger, traj) = stored().await;

    let headers = ledger.trajectories(&filter()).await.unwrap();
    let ids: Vec<&str> = headers.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, vec!["t1", "t2"], "ordered by id");
    assert_eq!(headers[0], traj.header);

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
    let left: Vec<String> = ledger
        .trajectories(&filter())
        .await
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(left, vec!["t2"], "only the retracted anchor's trajectory disappears");
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

    let mut orphan = t1(record_id(&scope(), "nowhere"));
    orphan.header.id = "orphan".into();
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
    let mut cross_traj = t1(foreign_id);
    cross_traj.header.id = "t3".into();
    let cross = ledger.put_trajectory(&cross_traj, &actor).await;
    assert!(
        cross.as_ref().is_err_and(|e| e.to_string().contains("is in small/enterprise")),
        "{cross:?}"
    );

    let mut shuffled = t1(traj.header.record_id);
    shuffled.header.id = "t4".into();
    shuffled.states.swap(1, 2);
    let unordered = ledger.put_trajectory(&shuffled, &actor).await;
    assert!(
        unordered.as_ref().is_err_and(|e| e.to_string().contains("strictly increase")),
        "{unordered:?}"
    );

    let mut empty = t1(traj.header.record_id);
    empty.header.id = "t5".into();
    empty.states.clear();
    assert!(ledger.put_trajectory(&empty, &actor).await.is_err());

    let reversed = ledger.trajectory_states(&filter(), "t1", 3, 1).await;
    assert!(
        reversed.as_ref().is_err_and(|e| e.to_string().contains("reversed")),
        "{reversed:?}"
    );

    // None of the refused writes left anything behind.
    assert_eq!(ledger.trajectories(&filter()).await.unwrap().len(), 2);
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
    let left: Vec<String> = ledger
        .trajectories_in_namespace(NS)
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.header.id)
        .collect();
    assert_eq!(left, vec!["t2"], "only the forgotten anchor's trajectory goes");
    let (states,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM trajectory_state WHERE trajectory = 't1'")
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
    assert_eq!(bundle.trajectories.len(), 2);
    assert_eq!(bundle.trajectories[0], traj);

    let fresh = Ledger::open_memory().await.unwrap();
    import_namespace(&fresh, &bundle, &config).await.unwrap();
    assert_eq!(fresh.trajectories_in_namespace(NS).await.unwrap(), bundle.trajectories);
    let again = export_namespace(&fresh, NS, &config).await.unwrap();
    assert_eq!(again, bundle, "export -> import -> export is byte-faithful");
}
