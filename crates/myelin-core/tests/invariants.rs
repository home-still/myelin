//! Property tests for the five storage invariants of `PLAN.md` §4 — the M1
//! exit criterion.
//!
//! Hermetic: SQLite only, no network. Each property drives the real
//! [`Ledger`] through the real write path and asserts the invariant over
//! generated inputs, so a regression in enforcement fails here rather than in
//! a benchmark three milestones later.

use chrono::{Duration, Utc};
use myelin_core::model::delta::Delta;
use myelin_core::model::query::ScopeFilter;
use myelin_core::model::record::{
    ActorId, EntityRef, MemoryRecord, Provenance, RecordKind, Salience, Scope, SourceRef, Trust,
    TrustTier, Validity,
};
use myelin_core::store::ids::record_id;
use myelin_core::store::ledger::Ledger;
use proptest::prelude::*;
use tokio::runtime::Runtime;
use uuid::Uuid;

// ── Fixtures ────────────────────────────────────────────────────

fn scope(tenant: &str) -> Scope {
    Scope::new(tenant, "agent-1", "ns-test")
}

fn record(key: &str, kind: RecordKind, sc: &Scope, text: &str, derived: Vec<Uuid>) -> MemoryRecord {
    let now = Utc::now();
    MemoryRecord {
        id: record_id(sc, key),
        kind,
        scope: sc.clone(),
        text: text.to_string(),
        entities: vec![EntityRef::new("phrase-a")],
        validity: Validity {
            t_valid: now - Duration::hours(1),
            t_invalid: None,
            t_ingested: now - Duration::hours(1),
            t_expired: None,
        },
        provenance: Provenance {
            source: SourceRef::span("doc-1", 0, 10),
            contributed_by: ActorId::new("user-1"),
            written_by: ActorId::new("agent-1"),
            derived_from: derived,
        },
        trust: Trust::asserted(),
        salience: Salience::default(),
        links: Vec::new(),
    }
}

fn actor() -> ActorId {
    ActorId::new("test")
}

fn rt() -> Runtime {
    Runtime::new().expect("tokio runtime")
}

async fn ledger_with_episodes(n: usize) -> (Ledger, Scope, Vec<Uuid>) {
    let ledger = Ledger::open_memory().await.expect("open ledger");
    let sc = scope("tenant-a");
    let mut ids = Vec::new();
    for i in 0..n {
        let r = record(
            &format!("ep-{i}"),
            RecordKind::Episodic,
            &sc,
            &format!("episode text {i}"),
            Vec::new(),
        );
        ids.push(r.id);
        ledger
            .apply(&Delta::Add { record: Box::new(r) }, &actor())
            .await
            .expect("add episode");
    }
    (ledger, sc, ids)
}

// ── I1 — no in-place mutation ───────────────────────────────────

/// UPDATE writes a *new* record and retracts the predecessor. The predecessor's
/// text survives byte-identical, which is what makes "what did we believe on
/// Tuesday" answerable.
#[test]
fn i1_update_preserves_predecessor_verbatim() {
    rt().block_on(async {
        let (ledger, sc, ids) = ledger_with_episodes(1).await;
        let original = ledger.get(ids[0]).await.unwrap().unwrap();

        let replacement = record(
            "ep-0-v2",
            RecordKind::Episodic,
            &sc,
            "corrected episode text",
            Vec::new(),
        );
        let replacement_id = replacement.id;
        ledger
            .apply(
                &Delta::Update {
                    target: ids[0],
                    replacement: Box::new(replacement),
                    reason: "correction".into(),
                },
                &actor(),
            )
            .await
            .unwrap();

        let after = ledger.get(ids[0]).await.unwrap().unwrap();
        assert_eq!(after.text, original.text, "I1: predecessor text mutated");
        assert_eq!(after.scope, original.scope, "I1: predecessor scope mutated");
        assert_eq!(
            after.provenance, original.provenance,
            "I1: predecessor provenance mutated"
        );
        assert!(
            after.validity.t_invalid.is_some(),
            "I1: UPDATE must retract the predecessor"
        );
        assert!(
            ledger.get(replacement_id).await.unwrap().is_some(),
            "I1: UPDATE must write a new record"
        );
    });
}

/// The invariant is enforced by the database, so a writer that bypasses
/// `apply` entirely still cannot mutate content.
#[test]
fn i1_direct_content_mutation_is_rejected_by_the_database() {
    rt().block_on(async {
        let (ledger, _sc, ids) = ledger_with_episodes(1).await;

        let err = sqlx::query("UPDATE record SET text = 'tampered' WHERE id = ?")
            .bind(ids[0].to_string())
            .execute(ledger.pool())
            .await
            .expect_err("I1: raw content UPDATE must be rejected");
        assert!(
            err.to_string().contains("I1"),
            "expected the I1 trigger to fire, got: {err}"
        );

        let still = ledger.get(ids[0]).await.unwrap().unwrap();
        assert_eq!(still.text, "episode text 0");
    });
}

/// `t_invalid` is write-once: a retraction is a historical fact, not a flag.
#[test]
fn i1_t_invalid_is_write_once() {
    rt().block_on(async {
        let (ledger, _sc, ids) = ledger_with_episodes(1).await;
        ledger
            .apply(
                &Delta::Delete {
                    target: ids[0],
                    reason: "superseded".into(),
                },
                &actor(),
            )
            .await
            .unwrap();
        let first = ledger.get(ids[0]).await.unwrap().unwrap().validity.t_invalid;
        assert!(first.is_some());

        let err = sqlx::query("UPDATE record SET t_invalid = '2000-01-01T00:00:00.000000000Z' WHERE id = ?")
            .bind(ids[0].to_string())
            .execute(ledger.pool())
            .await
            .expect_err("I1: rewriting t_invalid must be rejected");
        assert!(err.to_string().contains("write-once"), "got: {err}");

        let after = ledger.get(ids[0]).await.unwrap().unwrap().validity.t_invalid;
        assert_eq!(first, after);
    });
}

/// C9: the event log rejects UPDATE and DELETE outright.
#[test]
fn c9_event_log_is_append_only() {
    rt().block_on(async {
        let (ledger, _sc, _ids) = ledger_with_episodes(1).await;

        let upd = sqlx::query("UPDATE event SET reason = 'rewritten'")
            .execute(ledger.pool())
            .await
            .expect_err("C9: event UPDATE must be rejected");
        assert!(upd.to_string().contains("append-only"), "got: {upd}");

        let del = sqlx::query("DELETE FROM event")
            .execute(ledger.pool())
            .await
            .expect_err("C9: event DELETE must be rejected");
        assert!(del.to_string().contains("append-only"), "got: {del}");
    });
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Over any interleaving of deltas, no record's content, scope or
    /// provenance ever differs from what was first written.
    #[test]
    fn i1_content_is_stable_under_arbitrary_delta_sequences(
        ops in prop::collection::vec(0u8..3, 1..12),
    ) {
        rt().block_on(async move {
            let (ledger, sc, ids) = ledger_with_episodes(3).await;
            let mut original = std::collections::HashMap::new();
            for id in &ids {
                let r = ledger.get(*id).await.unwrap().unwrap();
                original.insert(*id, (r.text.clone(), r.scope.clone(), r.provenance.clone()));
            }

            for (n, op) in ops.iter().enumerate() {
                let target = ids[n % ids.len()];
                let delta = match op {
                    0 => {
                        let r = record(
                            &format!("gen-{n}"),
                            RecordKind::Episodic,
                            &sc,
                            &format!("generated {n}"),
                            Vec::new(),
                        );
                        Delta::Add { record: Box::new(r) }
                    }
                    1 => Delta::Update {
                        target,
                        replacement: Box::new(record(
                            &format!("gen-upd-{n}"),
                            RecordKind::Episodic,
                            &sc,
                            &format!("replacement {n}"),
                            Vec::new(),
                        )),
                        reason: "prop".into(),
                    },
                    _ => Delta::Delete { target, reason: "prop".into() },
                };
                // Re-adding an existing id or updating a retracted record is a
                // legitimate rejection; only silent corruption matters here.
                let _ = ledger.apply(&delta, &actor()).await;
            }

            for (id, (text, scope_, prov)) in original {
                let now = ledger.get(id).await.unwrap().unwrap();
                prop_assert_eq!(now.text, text, "I1: text changed for {}", id);
                prop_assert_eq!(now.scope, scope_, "I1: scope changed for {}", id);
                prop_assert_eq!(now.provenance, prov, "I1: provenance changed for {}", id);
            }
            Ok(())
        })?;
    }
}

// ── I2 — no provenance, no admissibility ────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// However many provenance-less rows a foreign writer inserts, `visible()`
    /// returns exactly the legitimate complement.
    ///
    /// Note the entry point: I1's trigger makes it *impossible* to null out an
    /// existing record's provenance, so the only way this state arises is
    /// insertion — a corrupt import, a second writer, or manual surgery. The
    /// test drives that path rather than a mutation that cannot happen.
    #[test]
    fn i2_records_without_provenance_are_never_visible(
        n_good in 1usize..5,
        n_orphan in 0usize..4,
    ) {
        rt().block_on(async move {
            let (ledger, sc, good) = ledger_with_episodes(n_good).await;

            let mut orphans = Vec::new();
            for i in 0..n_orphan {
                let r = record(
                    &format!("orphan-{i}"),
                    RecordKind::Episodic,
                    &sc,
                    &format!("unsourced {i}"),
                    Vec::new(),
                );
                sqlx::query(
                    "INSERT INTO record (
                        id, kind, tenant, agent, session, namespace, text, entities,
                        t_valid, t_invalid, t_ingested, t_expired,
                        trust_tier, trust_score, trust_checks,
                        prov_source, prov_contributed_by, prov_written_by, prov_derived_from,
                        salience
                     ) VALUES (?,?,?,?,NULL,?,?,'[]',?,NULL,?,NULL,'asserted',0.5,'[]',
                               NULL,'u','a','[]','{}')",
                )
                .bind(r.id.to_string())
                .bind(r.kind.as_str())
                .bind(&sc.tenant)
                .bind(&sc.agent)
                .bind(&sc.namespace)
                .bind(&r.text)
                .bind(r.validity.t_valid.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true))
                .bind(r.validity.t_ingested.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true))
                .execute(ledger.pool())
                .await
                .expect("insert unsourced row");
                orphans.push(r.id);
            }

            let visible = ledger
                .visible(&ScopeFilter::tenant(&sc.tenant), Utc::now(), 100)
                .await
                .unwrap();
            let seen: std::collections::HashSet<_> = visible.iter().map(|r| r.id).collect();

            for id in &orphans {
                prop_assert!(!seen.contains(id), "I2: {} surfaced without provenance", id);
            }
            prop_assert_eq!(seen.len(), n_good);
            for id in &good {
                prop_assert!(seen.contains(id));
            }

            let reported = ledger.records_missing_provenance().await.unwrap();
            let mut expected = orphans.clone();
            expected.sort_unstable();
            prop_assert_eq!(reported, expected);
            Ok(())
        })?;
    }
}

// ── I3 — quarantine is invisible ────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// No quarantined record reaches a read path, for any assignment of tiers.
    #[test]
    fn i3_quarantined_records_are_invisible(
        quarantine in prop::collection::vec(any::<bool>(), 6),
    ) {
        rt().block_on(async move {
            let (ledger, sc, ids) = ledger_with_episodes(6).await;

            let mut hidden = Vec::new();
            for (id, hide) in ids.iter().zip(&quarantine) {
                if *hide {
                    ledger.quarantine_existing(*id, "prop test").await.unwrap();
                    hidden.push(*id);
                }
            }

            let visible = ledger
                .visible(&ScopeFilter::tenant(&sc.tenant), Utc::now(), 100)
                .await
                .unwrap();
            for r in &visible {
                prop_assert_ne!(r.trust.tier, TrustTier::Quarantined);
                prop_assert!(!hidden.contains(&r.id), "I3: {} leaked into a read path", r.id);
            }
            prop_assert_eq!(visible.len(), 6 - hidden.len());
            Ok(())
        })?;
    }
}

/// Staged writes are reachable only through the explicit review accessor.
#[test]
fn i3_staged_writes_only_surface_through_review() {
    rt().block_on(async {
        let ledger = Ledger::open_memory().await.unwrap();
        let sc = scope("tenant-a");
        let staged = record("staged", RecordKind::Semantic, &sc, "suspicious", Vec::new());

        ledger.quarantine(&staged, "poison pattern").await.unwrap();

        let visible = ledger
            .visible(&ScopeFilter::tenant(&sc.tenant), Utc::now(), 100)
            .await
            .unwrap();
        assert!(visible.is_empty(), "I3: staged write reached a read path");

        let review = ledger.review_quarantine(10).await.unwrap();
        assert_eq!(review.len(), 1);
        assert_eq!(review[0].record.text, "suspicious");
        assert_eq!(review[0].reason, "poison pattern");
    });
}

// ── I4 — semantic records must be grounded ──────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// A `Semantic` record is accepted exactly when every ancestor it names
    /// exists; empty or dangling lineage is always rejected.
    #[test]
    fn i4_semantic_lineage_must_resolve(
        n_real in 0usize..3,
        n_fake in 0usize..3,
    ) {
        rt().block_on(async move {
            let (ledger, sc, ids) = ledger_with_episodes(3).await;

            let mut derived: Vec<Uuid> = ids.iter().take(n_real).copied().collect();
            for i in 0..n_fake {
                derived.push(Uuid::from_u128(0xdead_0000_0000_0000_0000_0000_0000_0000 + i as u128));
            }

            let sem = record("sem-1", RecordKind::Semantic, &sc, "abstracted fact", derived);
            let result = ledger.apply(&Delta::Add { record: Box::new(sem) }, &actor()).await;

            let should_pass = n_real > 0 && n_fake == 0;
            prop_assert_eq!(
                result.is_ok(),
                should_pass,
                "I4: n_real={} n_fake={} gave {:?}",
                n_real,
                n_fake,
                result.err().map(|e| e.to_string())
            );
            Ok(())
        })?;
    }
}

/// Non-semantic kinds are exempt: an episode is a primary observation and has
/// nothing to be derived from.
#[test]
fn i4_only_semantic_records_require_lineage() {
    rt().block_on(async {
        let ledger = Ledger::open_memory().await.unwrap();
        let sc = scope("tenant-a");
        for kind in [
            RecordKind::Episodic,
            RecordKind::Procedural,
            RecordKind::Working,
        ] {
            let r = record(
                &format!("k-{}", kind.as_str()),
                kind,
                &sc,
                "primary observation",
                Vec::new(),
            );
            ledger
                .apply(&Delta::Add { record: Box::new(r) }, &actor())
                .await
                .unwrap_or_else(|e| panic!("{kind:?} must not require lineage: {e}"));
        }
    });
}

// ── I5 — deletion cascades to descendants ───────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Hard-deleting a source removes exactly its transitive closure and
    /// nothing else. Survivors must be untouched: an over-eager cascade is as
    /// much a bug as a leak.
    #[test]
    fn i5_hard_delete_removes_exactly_the_descendant_closure(
        depth in 1usize..4,
        breadth in 1usize..3,
    ) {
        rt().block_on(async move {
            let (ledger, sc, roots) = ledger_with_episodes(2).await;
            let (root, bystander) = (roots[0], roots[1]);

            // Chain of semantic records derived from `root`.
            let mut frontier = vec![root];
            let mut doomed = vec![root];
            for d in 0..depth {
                let mut next = Vec::new();
                for b in 0..breadth {
                    let r = record(
                        &format!("sem-{d}-{b}"),
                        RecordKind::Semantic,
                        &sc,
                        &format!("derived {d}.{b}"),
                        frontier.clone(),
                    );
                    let id = r.id;
                    ledger.apply(&Delta::Add { record: Box::new(r) }, &actor()).await.unwrap();
                    next.push(id);
                    doomed.push(id);
                }
                frontier = next;
            }

            // One semantic record grounded in the bystander must survive.
            let survivor = record(
                "sem-survivor",
                RecordKind::Semantic,
                &sc,
                "unrelated",
                vec![bystander],
            );
            let survivor_id = survivor.id;
            ledger.apply(&Delta::Add { record: Box::new(survivor) }, &actor()).await.unwrap();

            let deleted = ledger.hard_delete(root, &actor(), "unlearn").await.unwrap();

            doomed.sort_unstable();
            doomed.dedup();
            prop_assert_eq!(deleted.clone(), doomed.clone(), "I5: wrong closure deleted");

            for id in &doomed {
                prop_assert!(ledger.get(*id).await.unwrap().is_none(), "I5: {} survived", id);
            }
            prop_assert!(ledger.get(bystander).await.unwrap().is_some(), "I5: bystander deleted");
            prop_assert!(ledger.get(survivor_id).await.unwrap().is_some(), "I5: survivor deleted");

            // The deletion itself is on the record (C11).
            let events = ledger.events(Some(root)).await.unwrap();
            prop_assert!(
                events.iter().any(|e| e.kind == "hard_delete"),
                "C11: hard delete must be audited"
            );
            Ok(())
        })?;
    }
}

// ── C2 / C12 — isolation and access control ─────────────────────

/// A read path is scoped to one tenant. This is the structural defense that
/// removes MINJA's shared-bank premise (§2 finding 9, C12).
#[test]
fn c12_reads_never_cross_tenants() {
    rt().block_on(async {
        let ledger = Ledger::open_memory().await.unwrap();
        for tenant in ["tenant-a", "tenant-b"] {
            let sc = scope(tenant);
            let r = record(
                &format!("rec-{tenant}"),
                RecordKind::Episodic,
                &sc,
                &format!("secret of {tenant}"),
                Vec::new(),
            );
            ledger
                .apply(&Delta::Add { record: Box::new(r) }, &actor())
                .await
                .unwrap();
        }

        for tenant in ["tenant-a", "tenant-b"] {
            let seen = ledger
                .visible(&ScopeFilter::tenant(tenant), Utc::now(), 100)
                .await
                .unwrap();
            assert_eq!(seen.len(), 1, "{tenant} saw {} records", seen.len());
            assert_eq!(seen[0].scope.tenant, tenant);
        }
    });
}

/// C2: `ℳ(u,a,t)` needs both edges, and revocation is edge removal.
#[test]
fn c2_bipartite_acl_requires_both_edges() {
    rt().block_on(async {
        let ledger = Ledger::open_memory().await.unwrap();
        assert!(!ledger.may_access("u1", "a1", "ns").await.unwrap());

        ledger.grant_user_agent("u1", "a1").await.unwrap();
        assert!(
            !ledger.may_access("u1", "a1", "ns").await.unwrap(),
            "C2: user→agent alone must not grant access"
        );

        ledger.grant_agent_namespace("a1", "ns").await.unwrap();
        assert!(ledger.may_access("u1", "a1", "ns").await.unwrap());

        ledger.revoke_agent_namespace("a1", "ns").await.unwrap();
        assert!(
            !ledger.may_access("u1", "a1", "ns").await.unwrap(),
            "C2: revocation is edge removal"
        );
    });
}

/// `explain` (I4 / C10): the lineage tree resolves to the primary sources.
#[test]
fn lineage_tree_resolves_to_primary_sources() {
    rt().block_on(async {
        let (ledger, sc, ids) = ledger_with_episodes(2).await;
        let sem = record(
            "sem-root",
            RecordKind::Semantic,
            &sc,
            "abstracted",
            ids.clone(),
        );
        let sem_id = sem.id;
        ledger
            .apply(&Delta::Add { record: Box::new(sem) }, &actor())
            .await
            .unwrap();

        let tree = ledger.lineage(sem_id).await.unwrap().expect("lineage");
        assert_eq!(tree.id, sem_id);
        assert_eq!(tree.ancestors.len(), 2);
        let mut ancestor_ids: Vec<_> = tree.ancestors.iter().map(|a| a.id).collect();
        ancestor_ids.sort_unstable();
        let mut expected = ids.clone();
        expected.sort_unstable();
        assert_eq!(ancestor_ids, expected);
        assert!(tree.ancestors.iter().all(|a| a.ancestors.is_empty()));
    });
}

/// Ids are content-addressed within a (tenant, namespace), so replaying the
/// same trajectory updates rather than duplicates — and two tenants asserting
/// the same fact never collide onto one point.
#[test]
fn record_ids_are_deterministic_and_tenant_scoped() {
    let a = scope("tenant-a");
    let b = scope("tenant-b");
    assert_eq!(record_id(&a, "fact"), record_id(&a, "fact"));
    assert_ne!(record_id(&a, "fact"), record_id(&b, "fact"));
    assert_ne!(record_id(&a, "fact"), record_id(&a, "other"));
}

// ── §5.4 — 1-hop personalized PageRank ──────────────────────────

/// PPR over the bipartite phrase↔record incidence must rank records reachable
/// from the query's phrases above records that share nothing with it.
///
/// This is the property the graph route exists for: 1-hop + PPR beats naive
/// neighbour expansion R@5 **72.5 vs 59.2** (`07-graph-memory.md` §5). A graph
/// that cannot separate connected from disconnected records buys nothing over
/// dense retrieval and should be deleted rather than tuned.
#[cfg(feature = "graph")]
#[test]
fn ppr_ranks_seed_connected_records_above_disconnected_ones() {
    use myelin_core::store::graph::{IncidenceGraph, DEFAULT_DAMPING, DEFAULT_ITERATIONS};

    rt().block_on(async {
        let (ledger, _sc, ids) = ledger_with_episodes(4).await;
        // ids[0], ids[1] share "berlin"; ids[2] shares nothing with the query;
        // ids[3] is connected through a high-degree, low-specificity phrase.
        ledger.set_incidence("berlin", ids[0], 1.0).await.unwrap();
        ledger.set_incidence("berlin", ids[1], 1.0).await.unwrap();
        ledger.set_incidence("sourdough", ids[2], 1.0).await.unwrap();
        for id in &ids {
            ledger.set_incidence("the", *id, 1.0).await.unwrap();
        }

        let graph = IncidenceGraph::load(&ledger, None).await.unwrap();
        assert_eq!(graph.record_count(), 4);
        assert_eq!(graph.phrase_count(), 3);
        assert_eq!(graph.edge_count(), 7);

        let ranked = graph.personalized_pagerank(
            &["berlin".to_string()],
            DEFAULT_DAMPING,
            DEFAULT_ITERATIONS,
        );
        let score = |id| {
            ranked
                .iter()
                .find(|(r, _)| *r == id)
                .map(|(_, s)| *s)
                .expect("record ranked")
        };

        assert!(
            score(ids[0]) > score(ids[2]),
            "PPR did not separate a seed-connected record from a disconnected one: {ranked:?}"
        );
        assert!(
            score(ids[1]) > score(ids[2]),
            "PPR did not separate a seed-connected record from a disconnected one: {ranked:?}"
        );
        assert_eq!(
            ranked[0].0.min(ranked[1].0),
            ids[0].min(ids[1]),
            "the two berlin records must take the top two slots: {ranked:?}"
        );

        // A query whose phrases are absent still ranks, via the background
        // reset mass, rather than returning nothing.
        let unseeded = graph.personalized_pagerank(
            &["nonexistent".to_string()],
            DEFAULT_DAMPING,
            DEFAULT_ITERATIONS,
        );
        assert_eq!(unseeded.len(), 4);
    });
}
