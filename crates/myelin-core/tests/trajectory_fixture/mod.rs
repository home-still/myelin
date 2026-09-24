//! A ledger holding small agent trajectories (M62), shared by the store and
//! tool tests. Hermetic: SQLite only.

#![allow(dead_code)]

use chrono::{Duration, Utc};
use myelin_core::model::delta::Delta;
use myelin_core::model::query::ScopeFilter;
use myelin_core::model::record::{
    ActorId, MemoryRecord, Provenance, RecordKind, Salience, Scope, SourceRef, Trust, Validity,
};
use myelin_core::model::trajectory::{AgentTrajectory, TrajectoryHeader, TrajectoryState};
use myelin_core::store::ids::record_id;
use myelin_core::store::ledger::Ledger;

pub const TENANT: &str = "small/web";
pub const NS: &str = "myelin";

pub fn scope() -> Scope {
    Scope::new(TENANT, "myelin", NS)
}

pub fn filter() -> ScopeFilter {
    ScopeFilter::tenant(TENANT).with_namespace(NS)
}

pub fn goal_record(sc: &Scope, traj: &str) -> MemoryRecord {
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

pub fn state(i: u32, action: Option<&str>, page: &str) -> TrajectoryState {
    TrajectoryState {
        state_index: i,
        step: i,
        url: format!("http://localhost:9080/page{i}"),
        action: action.map(str::to_string),
        thought: action.map(|a| format!("I will {a}")),
        accessibility_tree: page.to_string(),
    }
}

/// `t1`: a four-state bio edit that starts at the forum's root.
pub fn t1(anchor: uuid::Uuid) -> AgentTrajectory {
    AgentTrajectory {
        header: TrajectoryHeader {
            id: "t1".into(),
            scope: scope(),
            record_id: anchor,
            goal: "change the bio of t1".into(),
            environment: "reddit".into(),
            start_url: "http://localhost:9080/".into(),
            outcome: "success".into(),
        },
        states: vec![
            state(0, None, "[1] RootWebArea 'Postmill'\n[2] link 'Forums'"),
            state(1, Some("click('68')"), "[1] RootWebArea 'Settings'\n[89] link 'User settings'"),
            state(
                2,
                Some("fill('116', 'I am a robot')"),
                "[1] RootWebArea 'Edit biography'\n[116] textbox 'Biography' value='I am a robot'",
            ),
            state(3, Some("click('250')"), "[1] RootWebArea 'Profile'\n[250] button 'Save'"),
        ],
    }
}

/// `t2`: a three-state search that starts on a shop page.
pub fn t2(anchor: uuid::Uuid) -> AgentTrajectory {
    AgentTrajectory {
        header: TrajectoryHeader {
            id: "t2".into(),
            scope: scope(),
            record_id: anchor,
            goal: "find the cheapest biography book".into(),
            environment: "shopping".into(),
            start_url: "http://localhost:7770/".into(),
            outcome: "failure".into(),
        },
        states: vec![
            state(0, None, "[1] RootWebArea 'One Stop Market'\n[5] searchbox 'Search'"),
            state(1, Some("fill('5', 'biography')"), "[1] RootWebArea 'Search results'\n[40] link 'A Biography $12'"),
            state(2, Some("click('40')"), "[1] RootWebArea 'A Biography'\n[77] StaticText 'Price: $12.00'"),
        ],
    }
}

/// A ledger holding `t1` and `t2` in [`scope`], each anchored on its goal record.
pub async fn stored() -> (Ledger, AgentTrajectory, AgentTrajectory) {
    let ledger = Ledger::open_memory().await.expect("ledger");
    let actor = ActorId::new("builder");
    let mut out = Vec::new();
    for (id, make) in [("t1", t1 as fn(uuid::Uuid) -> AgentTrajectory), ("t2", t2)] {
        let goal = goal_record(&scope(), id);
        let traj = make(goal.id);
        ledger
            .apply(&Delta::Add { record: Box::new(goal) }, &actor)
            .await
            .expect("goal record");
        ledger.put_trajectory(&traj, &actor).await.expect("trajectory");
        out.push(traj);
    }
    let t2 = out.pop().expect("t2");
    let t1 = out.pop().expect("t1");
    (ledger, t1, t2)
}
