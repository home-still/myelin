//! SQLite ledger: truth, time, permission and lineage (`PLAN.md` §5.3).
//!
//! Qdrant answers "which records are similar". This answers "which records may
//! be seen and are still believed". Neither is authoritative alone, which is
//! why [`crate::store::reconcile`] is a first-class operation.
//!
//! # Invariants
//!
//! The four storage invariants of `PLAN.md` §4 are enforced *by the database*,
//! not by caller discipline, because caller discipline is what drift is made
//! of:
//!
//! - **I1** — `record` content, scope and provenance are frozen by a `BEFORE
//!   UPDATE` trigger; `t_invalid` is write-once by a second trigger; the
//!   `event` log rejects `UPDATE` and `DELETE` outright (C9).
//! - **I2** — [`Ledger::visible`] requires `prov_source IS NOT NULL`. The Rust
//!   type makes provenance mandatory, so a null can only arrive by import,
//!   manual edit or corruption — exactly the cases this catches.
//! - **I3** — [`Ledger::visible`] excludes `trust_tier = 'quarantined'`.
//!   [`Ledger::review_quarantine`] is the only accessor that returns them.
//! - **I4** — [`Ledger::apply`] rejects a `Semantic` record whose
//!   `derived_from` is empty or names a missing ancestor.
//! - **I5** — [`Ledger::hard_delete`] cascades to every transitive descendant.

use std::str::FromStr;

use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions, SqliteRow};
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::error::{MyelinError, Result};
use crate::model::delta::{AppliedDelta, Delta};
use crate::model::query::ScopeFilter;
use crate::model::record::{
    ActorId, EntityRef, Link, LinkKind, MemoryRecord, Provenance, RecordKind, Salience, Scope,
    SourceRef, Trust, TrustTier, Validity,
};

const SCHEMA: &str = include_str!("schema.sql");

fn fmt_time(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Nanos, true)
}

fn parse_time(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|t| t.with_timezone(&Utc))
        .map_err(|e| MyelinError::Store(format!("bad timestamp {s:?}: {e}")))
}

fn parse_uuid(s: &str) -> Result<Uuid> {
    Uuid::parse_str(s).map_err(|e| MyelinError::Store(format!("bad uuid {s:?}: {e}")))
}

fn sql(e: sqlx::Error) -> MyelinError {
    MyelinError::Store(e.to_string())
}

/// One row of the append-only decision log (C10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub seq: i64,
    pub at: DateTime<Utc>,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_id: Option<Uuid>,
    pub actor: ActorId,
    pub reason: String,
    pub detail: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkRow {
    pub src: Uuid,
    pub dst: Uuid,
    pub relation: LinkKind,
    pub at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IncidenceRow {
    pub phrase: String,
    pub record_id: Uuid,
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuarantineRow {
    pub id: Uuid,
    pub record: MemoryRecord,
    pub reason: String,
    pub at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AclEdge {
    pub from: String,
    pub to: String,
    pub granted_at: DateTime<Utc>,
}

/// A lineage node, for `explain` (I4 / C10 made usable).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineageNode {
    pub id: Uuid,
    pub kind: RecordKind,
    pub text: String,
    pub ancestors: Vec<LineageNode>,
}

pub struct Ledger {
    pool: SqlitePool,
}

impl Ledger {
    /// Open (creating if absent) a file-backed ledger and apply the schema.
    pub async fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(path.as_ref())
            .create_if_missing(true)
            .foreign_keys(true);
        Self::from_options(opts, 5).await
    }

    /// In-process ledger. Capped at one connection because each SQLite
    /// `:memory:` connection is a *separate* database.
    pub async fn open_memory() -> Result<Self> {
        let opts = SqliteConnectOptions::from_str("sqlite::memory:")
            .map_err(sql)?
            .foreign_keys(true);
        Self::from_options(opts, 1).await
    }

    async fn from_options(opts: SqliteConnectOptions, max_conn: u32) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(max_conn)
            .connect_with(opts)
            .await
            .map_err(sql)?;
        sqlx::raw_sql(SCHEMA).execute(&pool).await.map_err(sql)?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    // ── Write path ──────────────────────────────────────────────

    /// The only way the store mutates (`PLAN.md` §4).
    pub async fn apply(&self, delta: &Delta, actor: &ActorId) -> Result<AppliedDelta> {
        let now = Utc::now();
        let mut tx = self.pool.begin().await.map_err(sql)?;

        let applied = match delta {
            Delta::Add { record } => {
                Self::check_lineage(&mut tx, record).await?;
                Self::insert_record(&mut tx, record).await?;
                Self::insert_lineage_links(&mut tx, record, now).await?;
                AppliedDelta {
                    op: "add".into(),
                    subject: Some(record.id),
                    written: vec![record.id],
                    invalidated: Vec::new(),
                }
            }
            Delta::Update {
                target,
                replacement,
                ..
            } => {
                if !Self::exists(&mut tx, *target).await? {
                    return Err(MyelinError::Store(format!(
                        "UPDATE target {target} does not exist"
                    )));
                }
                Self::check_lineage(&mut tx, replacement).await?;
                Self::insert_record(&mut tx, replacement).await?;
                Self::insert_lineage_links(&mut tx, replacement, now).await?;
                // The predecessor stops being believed exactly where its
                // successor starts, not when the write happened: retracting
                // at `now` leaves a window in which both rows are live, and
                // a bi-temporal store whose belief intervals overlap cannot
                // answer "what did we believe at t". `consolidate::supersede`
                // documents the same shared boundary.
                Self::set_t_invalid(&mut tx, *target, replacement.validity.t_valid).await?;
                Self::insert_link(&mut tx, replacement.id, *target, LinkKind::Supersedes, now)
                    .await?;
                AppliedDelta {
                    op: "update".into(),
                    subject: Some(*target),
                    written: vec![replacement.id],
                    invalidated: vec![*target],
                }
            }
            Delta::Delete { target, .. } => {
                if !Self::exists(&mut tx, *target).await? {
                    return Err(MyelinError::Store(format!(
                        "DELETE target {target} does not exist"
                    )));
                }
                Self::set_t_invalid(&mut tx, *target, now).await?;
                AppliedDelta {
                    op: "delete".into(),
                    subject: Some(*target),
                    written: Vec::new(),
                    invalidated: vec![*target],
                }
            }
            Delta::Noop { target, .. } => AppliedDelta {
                op: "noop".into(),
                subject: *target,
                written: Vec::new(),
                invalidated: Vec::new(),
            },
        };

        let detail = serde_json::to_string(delta)?;
        sqlx::query(
            "INSERT INTO event (at, kind, record_id, actor, reason, detail) VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(fmt_time(now))
        .bind(delta.op())
        .bind(delta.subject().map(|u| u.to_string()))
        .bind(actor.as_str())
        .bind(delta.reason())
        .bind(detail)
        .execute(&mut *tx)
        .await
        .map_err(sql)?;

        tx.commit().await.map_err(sql)?;
        Ok(applied)
    }

    /// I4: a `Semantic` record must name where it was abstracted from, and
    /// every ancestor must exist.
    async fn check_lineage(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        record: &MemoryRecord,
    ) -> Result<()> {
        if !record.requires_lineage() {
            return Ok(());
        }
        if record.provenance.derived_from.is_empty() {
            return Err(MyelinError::Store(format!(
                "I4: semantic record {} has empty derived_from",
                record.id
            )));
        }
        for ancestor in &record.provenance.derived_from {
            if !Self::exists(tx, *ancestor).await? {
                return Err(MyelinError::Store(format!(
                    "I4: semantic record {} derives from missing ancestor {ancestor}",
                    record.id
                )));
            }
        }
        Ok(())
    }

    async fn exists(tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>, id: Uuid) -> Result<bool> {
        let row = sqlx::query("SELECT 1 FROM record WHERE id = ?")
            .bind(id.to_string())
            .fetch_optional(&mut **tx)
            .await
            .map_err(sql)?;
        Ok(row.is_some())
    }

    async fn insert_record(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        r: &MemoryRecord,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO record (
                id, kind, tenant, agent, session, namespace, text, entities,
                t_valid, t_invalid, t_ingested, t_expired,
                trust_tier, trust_score, trust_checks,
                prov_source, prov_contributed_by, prov_written_by, prov_derived_from,
                salience
             ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        )
        .bind(r.id.to_string())
        .bind(r.kind.as_str())
        .bind(&r.scope.tenant)
        .bind(&r.scope.agent)
        .bind(r.scope.session.as_deref())
        .bind(&r.scope.namespace)
        .bind(&r.text)
        .bind(serde_json::to_string(&r.entities)?)
        .bind(fmt_time(r.validity.t_valid))
        .bind(r.validity.t_invalid.map(fmt_time))
        .bind(fmt_time(r.validity.t_ingested))
        .bind(r.validity.t_expired.map(fmt_time))
        .bind(r.trust.tier.as_str())
        .bind(r.trust.score)
        .bind(serde_json::to_string(&r.trust.checks)?)
        .bind(serde_json::to_string(&r.provenance.source)?)
        .bind(r.provenance.contributed_by.as_str())
        .bind(r.provenance.written_by.as_str())
        .bind(serde_json::to_string(&r.provenance.derived_from)?)
        .bind(serde_json::to_string(&r.salience)?)
        .execute(&mut **tx)
        .await
        .map_err(sql)?;
        Ok(())
    }

    async fn insert_lineage_links(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        r: &MemoryRecord,
        now: DateTime<Utc>,
    ) -> Result<()> {
        for ancestor in &r.provenance.derived_from {
            Self::insert_link(tx, r.id, *ancestor, LinkKind::DerivedFrom, now).await?;
        }
        for link in &r.links {
            Self::insert_link(tx, r.id, link.target, link.relation, now).await?;
        }
        Ok(())
    }

    async fn insert_link(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        src: Uuid,
        dst: Uuid,
        relation: LinkKind,
        at: DateTime<Utc>,
    ) -> Result<()> {
        sqlx::query("INSERT OR IGNORE INTO link (src, dst, relation, at) VALUES (?, ?, ?, ?)")
            .bind(src.to_string())
            .bind(dst.to_string())
            .bind(relation.as_str())
            .bind(fmt_time(at))
            .execute(&mut **tx)
            .await
            .map_err(sql)?;
        Ok(())
    }

    async fn set_t_invalid(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        id: Uuid,
        at: DateTime<Utc>,
    ) -> Result<()> {
        // Write-once: the trigger aborts a second, differing write. Skipping
        // already-invalidated rows keeps DELETE-after-DELETE idempotent rather
        // than fatal.
        sqlx::query("UPDATE record SET t_invalid = ? WHERE id = ? AND t_invalid IS NULL")
            .bind(fmt_time(at))
            .bind(id.to_string())
            .execute(&mut **tx)
            .await
            .map_err(sql)?;
        Ok(())
    }

    // ── Read path ───────────────────────────────────────────────

    /// **The only scope-filtered read accessor.** I2 and I3 are enforced in
    /// [`Ledger::visible_inner`], so any new read path that goes through this
    /// function or its kind-filtered sibling inherits them.
    pub async fn visible(
        &self,
        filter: &ScopeFilter,
        now: DateTime<Utc>,
        limit: i64,
    ) -> Result<Vec<MemoryRecord>> {
        self.visible_inner(filter, None, now, limit).await
    }

    /// The same read, narrowed to one kind.
    ///
    /// Exists for `RecordKind::Profile`: a disposition is fetched by scope
    /// rather than by relevance, so it cannot come off the vector path. The
    /// ordering is `visible`'s — for profiles the newest-ingested is the
    /// current disposition, and a superseded one is already excluded by the
    /// `t_invalid` clause the shared body carries.
    pub async fn visible_of_kind(
        &self,
        filter: &ScopeFilter,
        kind: RecordKind,
        now: DateTime<Utc>,
        limit: i64,
    ) -> Result<Vec<MemoryRecord>> {
        self.visible_inner(filter, Some(kind), now, limit).await
    }

    async fn visible_inner(
        &self,
        filter: &ScopeFilter,
        kind: Option<RecordKind>,
        now: DateTime<Utc>,
        limit: i64,
    ) -> Result<Vec<MemoryRecord>> {
        let now_s = fmt_time(now);
        let kind_s = kind.map(|k| k.as_str());
        let rows = sqlx::query(
            "SELECT * FROM record
             WHERE tenant = ?
               AND (? IS NULL OR namespace = ?)
               AND (? IS NULL OR agent = ?)
               AND (? IS NULL OR session = ?)
               AND (? IS NULL OR kind = ?)
               AND prov_source IS NOT NULL
               AND trust_tier <> 'quarantined'
               AND t_valid <= ?
               AND (t_invalid IS NULL OR t_invalid > ?)
               AND (t_expired IS NULL OR t_expired > ?)
             ORDER BY t_ingested DESC, id
             LIMIT ?",
        )
        .bind(&filter.tenant)
        .bind(filter.namespace.as_deref())
        .bind(filter.namespace.as_deref())
        .bind(filter.agent.as_deref())
        .bind(filter.agent.as_deref())
        .bind(filter.session.as_deref())
        .bind(filter.session.as_deref())
        .bind(kind_s)
        .bind(kind_s)
        .bind(&now_s)
        .bind(&now_s)
        .bind(&now_s)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(sql)?;

        rows.iter().map(row_to_record).collect()
    }

    /// Unfiltered fetch by id — for admin paths and `explain`, never for
    /// composing evidence.
    pub async fn get(&self, id: Uuid) -> Result<Option<MemoryRecord>> {
        let row = sqlx::query("SELECT * FROM record WHERE id = ?")
            .bind(id.to_string())
            .fetch_optional(&self.pool)
            .await
            .map_err(sql)?;
        row.as_ref().map(row_to_record).transpose()
    }

    pub async fn count(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) AS n FROM record")
            .fetch_one(&self.pool)
            .await
            .map_err(sql)?;
        Ok(row.get::<i64, _>("n"))
    }

    /// Records admissible at `now` — the same predicate [`Ledger::visible`]
    /// applies, minus the scope.
    ///
    /// This is the count the vector index is supposed to mirror.
    /// [`Ledger::count`] includes every superseded row the store has ever
    /// held, so comparing *that* against a point count reports drift on any
    /// store that has ever taken an update.
    pub async fn count_live(&self, now: DateTime<Utc>) -> Result<i64> {
        let now_s = fmt_time(now);
        let row = sqlx::query(
            "SELECT COUNT(*) AS n FROM record
             WHERE prov_source IS NOT NULL
               AND trust_tier <> 'quarantined'
               AND t_valid <= ?
               AND (t_invalid IS NULL OR t_invalid > ?)
               AND (t_expired IS NULL OR t_expired > ?)",
        )
        .bind(&now_s)
        .bind(&now_s)
        .bind(&now_s)
        .fetch_one(&self.pool)
        .await
        .map_err(sql)?;
        Ok(row.get::<i64, _>("n"))
    }

    /// Record that an existing fact was encountered again.
    ///
    /// `salience` is the one JSON column the schema declares mutable
    /// (I1 freezes content, scope and provenance, not this). Only
    /// `access_count` and `last_access` move: `strength` is the decay
    /// model's output, and writing it here would be inventing a policy the
    /// model has not been given.
    pub async fn touch_salience(&self, id: Uuid) -> Result<()> {
        sqlx::query(
            "UPDATE record
             SET salience = json_set(
                     salience,
                     '$.access_count',
                     COALESCE(json_extract(salience, '$.access_count'), 0) + 1,
                     '$.last_access', ?)
             WHERE id = ?",
        )
        .bind(fmt_time(Utc::now()))
        .bind(id.to_string())
        .execute(&self.pool)
        .await
        .map_err(sql)?;
        Ok(())
    }

    /// Mark one ingest unit as fully written, and ask whether it already is.
    ///
    /// A row count cannot answer this. The first full LoCoMo run died
    /// mid-conversation on conv-44 with 79 records and all 62 episodes
    /// already committed, so both "has any records" and "has all its
    /// episodes" would have declared it done and skipped the consolidation
    /// that never ran. Completion is a fact about the *process*, so it is
    /// recorded as one — an audit event (C9), which is where "what happened
    /// to this unit" already lives.
    pub async fn mark_unit_complete(
        &self,
        tenant: &str,
        actor: &ActorId,
        detail: serde_json::Value,
    ) -> Result<()> {
        self.log("unit_complete", None, actor, tenant, detail).await
    }

    pub async fn unit_is_complete(&self, tenant: &str) -> Result<bool> {
        let row = sqlx::query(
            "SELECT COUNT(*) AS n FROM event WHERE kind = 'unit_complete' AND reason = ?",
        )
        .bind(tenant)
        .fetch_one(&self.pool)
        .await
        .map_err(sql)?;
        Ok(row.get::<i64, _>("n") > 0)
    }

    /// Ids whose `prov_source` is NULL and which have not yet been dealt with.
    /// These are invisible to every read path (I2); this is how
    /// [`crate::store::reconcile`] finds them.
    ///
    /// Already-quarantined rows are excluded: quarantine *is* the repair for a
    /// record whose source cannot be reconstructed, so continuing to report
    /// them would make `reconcile` permanently dirty.
    pub async fn records_missing_provenance(&self) -> Result<Vec<Uuid>> {
        let rows = sqlx::query(
            "SELECT id FROM record
             WHERE prov_source IS NULL AND trust_tier <> 'quarantined'
             ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(sql)?;
        rows.iter()
            .map(|r| parse_uuid(r.get::<String, _>("id").as_str()))
            .collect()
    }

    /// Live record ids whose provenance names a source document under
    /// `doc_prefix`, within one scope.
    ///
    /// Exists for I4. A record that abstracts over other records must name
    /// them ([`crate::model::record::MemoryRecord::requires_lineage`]), and
    /// a caller minting one — the LME-V2 events pool is the only one today
    /// — knows its source documents but not their ids. `visible_of_kind`
    /// cannot serve this: it is scoped, not source-filtered, and a single
    /// LME-V2 tenant holds ~38k records, so resolving 200 trajectories
    /// through it would be 200 full-tenant scans.
    ///
    /// `SourceRef::doc` serialises as `{"doc":"<id>"}`, so a trajectory's
    /// states are the documents under `<traj_id>:` — the prefix this takes.
    /// `_` and `%` in the prefix are escaped, because a trajectory id is
    /// caller data and `LIKE` would otherwise treat it as a pattern.
    pub async fn ids_from_source_docs(
        &self,
        filter: &ScopeFilter,
        doc_prefix: &str,
    ) -> Result<Vec<Uuid>> {
        let escaped = doc_prefix
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!(r#"{{"doc":"{escaped}%"#);
        let rows = sqlx::query(
            r#"SELECT id FROM record
               WHERE tenant = ?
                 AND (? IS NULL OR namespace = ?)
                 AND t_invalid IS NULL
                 AND prov_source LIKE ? ESCAPE '\'
               ORDER BY id"#,
        )
        .bind(&filter.tenant)
        .bind(filter.namespace.as_deref())
        .bind(filter.namespace.as_deref())
        .bind(&pattern)
        .fetch_all(&self.pool)
        .await
        .map_err(sql)?;
        rows.iter()
            .map(|r| parse_uuid(r.get::<String, _>("id").as_str()))
            .collect()
    }

    // ── Quarantine (C4 / I3) ────────────────────────────────────

    pub async fn quarantine(&self, record: &MemoryRecord, reason: &str) -> Result<()> {
        let now = Utc::now();
        sqlx::query(
            "INSERT OR REPLACE INTO quarantine (id, record, reason, at) VALUES (?, ?, ?, ?)",
        )
        .bind(record.id.to_string())
        .bind(serde_json::to_string(record)?)
        .bind(reason)
        .bind(fmt_time(now))
        .execute(&self.pool)
        .await
        .map_err(sql)?;
        self.log(
            "quarantine",
            Some(record.id),
            &ActorId::new("system"),
            reason,
            serde_json::json!({}),
        )
        .await
    }

    /// Mark a record already in the projection as quarantined, making it
    /// invisible to every read path (I3).
    pub async fn quarantine_existing(&self, id: Uuid, reason: &str) -> Result<()> {
        sqlx::query("UPDATE record SET trust_tier = 'quarantined' WHERE id = ?")
            .bind(id.to_string())
            .execute(&self.pool)
            .await
            .map_err(sql)?;
        self.log(
            "quarantine",
            Some(id),
            &ActorId::new("system"),
            reason,
            serde_json::json!({}),
        )
        .await
    }

    /// The only accessor that returns quarantined material (I3).
    pub async fn review_quarantine(&self, limit: i64) -> Result<Vec<QuarantineRow>> {
        let rows = sqlx::query("SELECT * FROM quarantine ORDER BY at, id LIMIT ?")
            .bind(limit)
            .fetch_all(&self.pool)
            .await
            .map_err(sql)?;
        rows.iter()
            .map(|r| {
                Ok(QuarantineRow {
                    id: parse_uuid(r.get::<String, _>("id").as_str())?,
                    record: serde_json::from_str(r.get::<String, _>("record").as_str())?,
                    reason: r.get("reason"),
                    at: parse_time(r.get::<String, _>("at").as_str())?,
                })
            })
            .collect()
    }

    // ── Lineage and cascade ─────────────────────────────────────

    /// I4 made usable: the ancestor tree behind a record.
    pub async fn lineage(&self, id: Uuid) -> Result<Option<LineageNode>> {
        let Some(record) = self.get(id).await? else {
            return Ok(None);
        };
        let mut ancestors = Vec::new();
        for a in &record.provenance.derived_from {
            if let Some(node) = Box::pin(self.lineage(*a)).await? {
                ancestors.push(node);
            }
        }
        Ok(Some(LineageNode {
            id: record.id,
            kind: record.kind,
            text: record.text,
            ancestors,
        }))
    }

    /// Every record transitively derived from `id`.
    pub async fn descendants(&self, id: Uuid) -> Result<Vec<Uuid>> {
        let rows = sqlx::query(
            "WITH RECURSIVE d(id) AS (
                 SELECT src FROM link WHERE dst = ? AND relation = 'derived_from'
                 UNION
                 SELECT l.src FROM link l JOIN d ON l.dst = d.id AND l.relation = 'derived_from'
             )
             SELECT id FROM d ORDER BY id",
        )
        .bind(id.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(sql)?;
        rows.iter()
            .map(|r| parse_uuid(r.get::<String, _>("id").as_str()))
            .collect()
    }

    /// C11 unlearn. **I5:** deleting a source deletes every descendant.
    ///
    /// The plan's alternative — re-deriving descendants from the surviving
    /// sources — needs the extraction model and lands with the write path in
    /// M3. Deletion is the conservative branch: it cannot leave a descendant
    /// grounded in material we were asked to erase.
    pub async fn hard_delete(&self, id: Uuid, actor: &ActorId, reason: &str) -> Result<Vec<Uuid>> {
        let mut doomed = self.descendants(id).await?;
        doomed.push(id);
        doomed.sort_unstable();
        doomed.dedup();

        let now = Utc::now();
        let mut tx = self.pool.begin().await.map_err(sql)?;
        for victim in &doomed {
            let v = victim.to_string();
            sqlx::query("DELETE FROM incidence WHERE record_id = ?")
                .bind(&v)
                .execute(&mut *tx)
                .await
                .map_err(sql)?;
            sqlx::query("DELETE FROM link WHERE src = ? OR dst = ?")
                .bind(&v)
                .bind(&v)
                .execute(&mut *tx)
                .await
                .map_err(sql)?;
            sqlx::query("DELETE FROM record WHERE id = ?")
                .bind(&v)
                .execute(&mut *tx)
                .await
                .map_err(sql)?;
            sqlx::query(
                "INSERT INTO event (at, kind, record_id, actor, reason, detail) VALUES (?,?,?,?,?,?)",
            )
            .bind(fmt_time(now))
            .bind("hard_delete")
            .bind(&v)
            .bind(actor.as_str())
            .bind(reason)
            .bind(serde_json::json!({ "root": id.to_string() }).to_string())
            .execute(&mut *tx)
            .await
            .map_err(sql)?;
        }
        tx.commit().await.map_err(sql)?;
        Ok(doomed)
    }

    // ── Links, incidence, ACL ───────────────────────────────────

    pub async fn links(&self, namespace: Option<&str>) -> Result<Vec<LinkRow>> {
        let rows = match namespace {
            Some(ns) => {
                sqlx::query(
                    "SELECT l.* FROM link l JOIN record r ON r.id = l.src
                 WHERE r.namespace = ? ORDER BY l.src, l.dst, l.relation",
                )
                .bind(ns)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query("SELECT * FROM link ORDER BY src, dst, relation")
                    .fetch_all(&self.pool)
                    .await
            }
        }
        .map_err(sql)?;
        rows.iter()
            .map(|r| {
                Ok(LinkRow {
                    src: parse_uuid(r.get::<String, _>("src").as_str())?,
                    dst: parse_uuid(r.get::<String, _>("dst").as_str())?,
                    relation: LinkKind::parse(r.get::<String, _>("relation").as_str())?,
                    at: parse_time(r.get::<String, _>("at").as_str())?,
                })
            })
            .collect()
    }

    /// Every edge touching `id`, in either direction.
    ///
    /// [`Ledger::links`] cannot answer this: with a namespace it joins only
    /// on the *source* record, so an inbound edge from another namespace is
    /// invisible, and without one it returns the whole `link` table for the
    /// caller to filter in Rust. A link is directional metadata about a
    /// pair, not a possession of the source's namespace.
    pub async fn links_incident(&self, id: Uuid) -> Result<Vec<LinkRow>> {
        let v = id.to_string();
        let rows =
            sqlx::query("SELECT * FROM link WHERE src = ? OR dst = ? ORDER BY src, dst, relation")
                .bind(&v)
                .bind(&v)
                .fetch_all(&self.pool)
                .await
                .map_err(sql)?;
        rows.iter()
            .map(|r| {
                Ok(LinkRow {
                    src: parse_uuid(r.get::<String, _>("src").as_str())?,
                    dst: parse_uuid(r.get::<String, _>("dst").as_str())?,
                    relation: LinkKind::parse(r.get::<String, _>("relation").as_str())?,
                    at: parse_time(r.get::<String, _>("at").as_str())?,
                })
            })
            .collect()
    }

    /// Links whose endpoints no longer exist — pure drift.
    pub async fn dangling_links(&self) -> Result<Vec<LinkRow>> {
        let rows = sqlx::query(
            "SELECT l.* FROM link l
             WHERE NOT EXISTS (SELECT 1 FROM record WHERE id = l.src)
                OR NOT EXISTS (SELECT 1 FROM record WHERE id = l.dst)
             ORDER BY l.src, l.dst, l.relation",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(sql)?;
        rows.iter()
            .map(|r| {
                Ok(LinkRow {
                    src: parse_uuid(r.get::<String, _>("src").as_str())?,
                    dst: parse_uuid(r.get::<String, _>("dst").as_str())?,
                    relation: LinkKind::parse(r.get::<String, _>("relation").as_str())?,
                    at: parse_time(r.get::<String, _>("at").as_str())?,
                })
            })
            .collect()
    }

    pub async fn delete_link(&self, src: Uuid, dst: Uuid, relation: LinkKind) -> Result<()> {
        sqlx::query("DELETE FROM link WHERE src = ? AND dst = ? AND relation = ?")
            .bind(src.to_string())
            .bind(dst.to_string())
            .bind(relation.as_str())
            .execute(&self.pool)
            .await
            .map_err(sql)?;
        Ok(())
    }

    pub async fn set_incidence(&self, phrase: &str, record_id: Uuid, weight: f32) -> Result<()> {
        sqlx::query(
            "INSERT INTO incidence (phrase, record_id, weight) VALUES (?,?,?)
             ON CONFLICT(phrase, record_id) DO UPDATE SET weight = excluded.weight",
        )
        .bind(phrase)
        .bind(record_id.to_string())
        .bind(weight)
        .execute(&self.pool)
        .await
        .map_err(sql)?;
        Ok(())
    }

    pub async fn incidence(&self, namespace: Option<&str>) -> Result<Vec<IncidenceRow>> {
        let rows = match namespace {
            Some(ns) => {
                sqlx::query(
                    "SELECT i.* FROM incidence i JOIN record r ON r.id = i.record_id
                 WHERE r.namespace = ? ORDER BY i.phrase, i.record_id",
                )
                .bind(ns)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query("SELECT * FROM incidence ORDER BY phrase, record_id")
                    .fetch_all(&self.pool)
                    .await
            }
        }
        .map_err(sql)?;
        rows.iter()
            .map(|r| {
                Ok(IncidenceRow {
                    phrase: r.get("phrase"),
                    record_id: parse_uuid(r.get::<String, _>("record_id").as_str())?,
                    weight: r.get("weight"),
                })
            })
            .collect()
    }

    /// Incidence scoped to one tenant — the read path's entry point.
    ///
    /// [`Ledger::incidence`] is namespace-wide and is the export/analysis
    /// path. A query needs the tenant's subgraph: `bench` scopes every query
    /// to `tenant` and leaves `namespace` at the corpus slug, which is
    /// 162,254 records for LongMemEval_S. C12 forbids the cross-tenant read
    /// the namespace-wide load would perform.
    ///
    /// Served by the existing `record_scope (tenant, namespace, agent)` and
    /// `incidence_record (record_id)` indexes.
    pub async fn incidence_for_tenant(
        &self,
        tenant: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<IncidenceRow>> {
        let rows = sqlx::query(
            "SELECT i.* FROM incidence i JOIN record r ON r.id = i.record_id
             WHERE r.tenant = ?
               AND (? IS NULL OR r.namespace = ?)
             ORDER BY i.phrase, i.record_id",
        )
        .bind(tenant)
        .bind(namespace)
        .bind(namespace)
        .fetch_all(&self.pool)
        .await
        .map_err(sql)?;
        rows.iter()
            .map(|r| {
                Ok(IncidenceRow {
                    phrase: r.get("phrase"),
                    record_id: parse_uuid(r.get::<String, _>("record_id").as_str())?,
                    weight: r.get("weight"),
                })
            })
            .collect()
    }

    /// Write many incidence rows in one transaction, replacing the existing
    /// rows of every record the batch mentions.
    ///
    /// Authoritative-per-record is what both callers need. Re-indexing a
    /// record must replace its edges rather than accumulate them, and a
    /// pre-cutover row that [`crate::pipeline::phrases::incidence_rows`] no
    /// longer emits — the 32-phrase cap can drop one — would otherwise
    /// survive as a stale edge that no reconcile direction looks for.
    /// [`Ledger::set_incidence`] stays for single-row upserts.
    pub async fn replace_incidence_batch(&self, rows: &[IncidenceRow]) -> Result<usize> {
        if rows.is_empty() {
            return Ok(0);
        }
        // Collected first so the delete order is deterministic.
        let touched: std::collections::BTreeSet<Uuid> = rows.iter().map(|r| r.record_id).collect();

        let mut tx = self.pool.begin().await.map_err(sql)?;
        for id in &touched {
            sqlx::query("DELETE FROM incidence WHERE record_id = ?")
                .bind(id.to_string())
                .execute(&mut *tx)
                .await
                .map_err(sql)?;
        }
        for row in rows {
            sqlx::query(
                "INSERT INTO incidence (phrase, record_id, weight) VALUES (?,?,?)
                 ON CONFLICT(phrase, record_id) DO UPDATE SET weight = excluded.weight",
            )
            .bind(&row.phrase)
            .bind(row.record_id.to_string())
            .bind(row.weight)
            .execute(&mut *tx)
            .await
            .map_err(sql)?;
        }
        tx.commit().await.map_err(sql)?;
        Ok(rows.len())
    }

    pub async fn orphan_incidence(&self) -> Result<Vec<IncidenceRow>> {
        let rows = sqlx::query(
            "SELECT i.* FROM incidence i
             WHERE NOT EXISTS (SELECT 1 FROM record WHERE id = i.record_id)
             ORDER BY i.phrase, i.record_id",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(sql)?;
        rows.iter()
            .map(|r| {
                Ok(IncidenceRow {
                    phrase: r.get("phrase"),
                    record_id: parse_uuid(r.get::<String, _>("record_id").as_str())?,
                    weight: r.get("weight"),
                })
            })
            .collect()
    }

    pub async fn delete_incidence(&self, phrase: &str, record_id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM incidence WHERE phrase = ? AND record_id = ?")
            .bind(phrase)
            .bind(record_id.to_string())
            .execute(&self.pool)
            .await
            .map_err(sql)?;
        Ok(())
    }

    /// C2: bipartite ACL `G_UA(t)`. Revocation is edge removal.
    pub async fn grant_user_agent(&self, user: &str, agent: &str) -> Result<()> {
        sqlx::query("INSERT OR REPLACE INTO acl_ua (user_id, agent_id, granted_at) VALUES (?,?,?)")
            .bind(user)
            .bind(agent)
            .bind(fmt_time(Utc::now()))
            .execute(&self.pool)
            .await
            .map_err(sql)?;
        Ok(())
    }

    /// C2: bipartite ACL `G_AR(t)`.
    pub async fn grant_agent_namespace(&self, agent: &str, namespace: &str) -> Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO acl_ar (agent_id, namespace, granted_at) VALUES (?,?,?)",
        )
        .bind(agent)
        .bind(namespace)
        .bind(fmt_time(Utc::now()))
        .execute(&self.pool)
        .await
        .map_err(sql)?;
        Ok(())
    }

    pub async fn revoke_user_agent(&self, user: &str, agent: &str) -> Result<()> {
        sqlx::query("DELETE FROM acl_ua WHERE user_id = ? AND agent_id = ?")
            .bind(user)
            .bind(agent)
            .execute(&self.pool)
            .await
            .map_err(sql)?;
        Ok(())
    }

    pub async fn revoke_agent_namespace(&self, agent: &str, namespace: &str) -> Result<()> {
        sqlx::query("DELETE FROM acl_ar WHERE agent_id = ? AND namespace = ?")
            .bind(agent)
            .bind(namespace)
            .execute(&self.pool)
            .await
            .map_err(sql)?;
        Ok(())
    }

    /// `ℳ(u,a,t)`: the user reaches the agent and the agent reaches the
    /// namespace. Both edges must be present.
    pub async fn may_access(&self, user: &str, agent: &str, namespace: &str) -> Result<bool> {
        let row = sqlx::query(
            "SELECT 1 FROM acl_ua ua JOIN acl_ar ar ON ar.agent_id = ua.agent_id
             WHERE ua.user_id = ? AND ua.agent_id = ? AND ar.namespace = ?",
        )
        .bind(user)
        .bind(agent)
        .bind(namespace)
        .fetch_optional(&self.pool)
        .await
        .map_err(sql)?;
        Ok(row.is_some())
    }

    pub async fn acl_edges(&self) -> Result<(Vec<AclEdge>, Vec<AclEdge>)> {
        let ua = sqlx::query("SELECT * FROM acl_ua ORDER BY user_id, agent_id")
            .fetch_all(&self.pool)
            .await
            .map_err(sql)?;
        let ar = sqlx::query("SELECT * FROM acl_ar ORDER BY agent_id, namespace")
            .fetch_all(&self.pool)
            .await
            .map_err(sql)?;
        let ua = ua
            .iter()
            .map(|r| {
                Ok(AclEdge {
                    from: r.get("user_id"),
                    to: r.get("agent_id"),
                    granted_at: parse_time(r.get::<String, _>("granted_at").as_str())?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let ar = ar
            .iter()
            .map(|r| {
                Ok(AclEdge {
                    from: r.get("agent_id"),
                    to: r.get("namespace"),
                    granted_at: parse_time(r.get::<String, _>("granted_at").as_str())?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((ua, ar))
    }

    // ── Audit log (C10) ─────────────────────────────────────────

    pub async fn log(
        &self,
        kind: &str,
        record_id: Option<Uuid>,
        actor: &ActorId,
        reason: &str,
        detail: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO event (at, kind, record_id, actor, reason, detail) VALUES (?,?,?,?,?,?)",
        )
        .bind(fmt_time(Utc::now()))
        .bind(kind)
        .bind(record_id.map(|u| u.to_string()))
        .bind(actor.as_str())
        .bind(reason)
        .bind(detail.to_string())
        .execute(&self.pool)
        .await
        .map_err(sql)?;
        Ok(())
    }

    pub async fn events(&self, record_id: Option<Uuid>) -> Result<Vec<Event>> {
        let rows = match record_id {
            Some(id) => {
                sqlx::query("SELECT * FROM event WHERE record_id = ? ORDER BY seq")
                    .bind(id.to_string())
                    .fetch_all(&self.pool)
                    .await
            }
            None => {
                sqlx::query("SELECT * FROM event ORDER BY seq")
                    .fetch_all(&self.pool)
                    .await
            }
        }
        .map_err(sql)?;
        rows.iter()
            .map(|r| {
                Ok(Event {
                    seq: r.get("seq"),
                    at: parse_time(r.get::<String, _>("at").as_str())?,
                    kind: r.get("kind"),
                    record_id: r
                        .get::<Option<String>, _>("record_id")
                        .map(|s| parse_uuid(&s))
                        .transpose()?,
                    actor: ActorId(r.get("actor")),
                    reason: r.get("reason"),
                    detail: serde_json::from_str(r.get::<String, _>("detail").as_str())
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .collect()
    }

    /// Records in a namespace, ordered deterministically — the export spine
    /// and the reconciler's left-hand side.
    ///
    /// Rows with a NULL `prov_source` are skipped: they cannot be materialized
    /// as a [`MemoryRecord`] at all, and [`Ledger::records_missing_provenance`]
    /// reports them separately. Without this, a single corrupt row would make
    /// both `export` and `reconcile` fail on the exact namespace they exist to
    /// repair.
    pub async fn records_in_namespace(&self, namespace: &str) -> Result<Vec<MemoryRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM record
             WHERE namespace = ? AND prov_source IS NOT NULL
             ORDER BY id",
        )
        .bind(namespace)
        .fetch_all(&self.pool)
        .await
        .map_err(sql)?;
        rows.iter().map(row_to_record).collect()
    }

    /// Keyset-paginated record scan, so a backfill never materialises a whole
    /// namespace — [`Ledger::records_in_namespace`] loads all 162,254
    /// LongMemEval_S rows at once and stays the reconcile path.
    ///
    /// `id` is `TEXT PRIMARY KEY`, so `id > ?` is a total lexicographic order
    /// and paging can neither skip nor repeat a row. `prov_source IS NOT
    /// NULL` keeps I2: a record that is inadmissible must not gain graph
    /// edges.
    pub async fn records_after(
        &self,
        namespace: &str,
        after_id: Option<Uuid>,
        limit: i64,
    ) -> Result<Vec<MemoryRecord>> {
        let after = after_id.map(|u| u.to_string());
        let rows = sqlx::query(
            "SELECT * FROM record
             WHERE namespace = ?
               AND prov_source IS NOT NULL
               AND (? IS NULL OR id > ?)
             ORDER BY id
             LIMIT ?",
        )
        .bind(namespace)
        .bind(after.as_deref())
        .bind(after.as_deref())
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(sql)?;
        rows.iter().map(row_to_record).collect()
    }

    /// Restore a bundle verbatim (R3).
    ///
    /// Rows are inserted directly, not replayed through [`Ledger::apply`],
    /// because a faithful re-import must reproduce the *stored* state —
    /// already-invalidated records, quarantined material, original event
    /// sequence numbers — rather than mint fresh timestamps. Event `seq` is
    /// carried across explicitly so an export of the imported ledger is
    /// byte-identical to the original.
    pub async fn import_bundle(&self, bundle: &crate::store::export::ExportBundle) -> Result<()> {
        let mut tx = self.pool.begin().await.map_err(sql)?;

        for record in &bundle.records {
            Self::insert_record(&mut tx, record).await?;
        }
        for link in &bundle.links {
            sqlx::query("INSERT OR IGNORE INTO link (src, dst, relation, at) VALUES (?,?,?,?)")
                .bind(link.src.to_string())
                .bind(link.dst.to_string())
                .bind(link.relation.as_str())
                .bind(fmt_time(link.at))
                .execute(&mut *tx)
                .await
                .map_err(sql)?;
        }
        for row in &bundle.incidence {
            sqlx::query(
                "INSERT OR REPLACE INTO incidence (phrase, record_id, weight) VALUES (?,?,?)",
            )
            .bind(&row.phrase)
            .bind(row.record_id.to_string())
            .bind(row.weight)
            .execute(&mut *tx)
            .await
            .map_err(sql)?;
        }
        for row in &bundle.quarantine {
            sqlx::query(
                "INSERT OR REPLACE INTO quarantine (id, record, reason, at) VALUES (?,?,?,?)",
            )
            .bind(row.id.to_string())
            .bind(serde_json::to_string(&row.record)?)
            .bind(&row.reason)
            .bind(fmt_time(row.at))
            .execute(&mut *tx)
            .await
            .map_err(sql)?;
        }
        for edge in &bundle.acl_ua {
            sqlx::query(
                "INSERT OR REPLACE INTO acl_ua (user_id, agent_id, granted_at) VALUES (?,?,?)",
            )
            .bind(&edge.from)
            .bind(&edge.to)
            .bind(fmt_time(edge.granted_at))
            .execute(&mut *tx)
            .await
            .map_err(sql)?;
        }
        for edge in &bundle.acl_ar {
            sqlx::query(
                "INSERT OR REPLACE INTO acl_ar (agent_id, namespace, granted_at) VALUES (?,?,?)",
            )
            .bind(&edge.from)
            .bind(&edge.to)
            .bind(fmt_time(edge.granted_at))
            .execute(&mut *tx)
            .await
            .map_err(sql)?;
        }
        for event in &bundle.events {
            sqlx::query(
                "INSERT INTO event (seq, at, kind, record_id, actor, reason, detail) VALUES (?,?,?,?,?,?,?)",
            )
            .bind(event.seq)
            .bind(fmt_time(event.at))
            .bind(&event.kind)
            .bind(event.record_id.map(|u| u.to_string()))
            .bind(event.actor.as_str())
            .bind(&event.reason)
            .bind(event.detail.to_string())
            .execute(&mut *tx)
            .await
            .map_err(sql)?;
        }

        tx.commit().await.map_err(sql)?;
        Ok(())
    }
}

fn row_to_record(row: &SqliteRow) -> Result<MemoryRecord> {
    let source: Option<String> = row.get("prov_source");
    let source = source.ok_or_else(|| {
        MyelinError::Store(format!(
            "I2: record {} has no provenance source",
            row.get::<String, _>("id")
        ))
    })?;
    let derived_from: Vec<Uuid> =
        serde_json::from_str(row.get::<String, _>("prov_derived_from").as_str())?;

    Ok(MemoryRecord {
        id: parse_uuid(row.get::<String, _>("id").as_str())?,
        kind: RecordKind::parse(row.get::<String, _>("kind").as_str())?,
        scope: Scope {
            tenant: row.get("tenant"),
            agent: row.get("agent"),
            session: row.get("session"),
            namespace: row.get("namespace"),
        },
        text: row.get("text"),
        entities: serde_json::from_str::<Vec<EntityRef>>(
            row.get::<String, _>("entities").as_str(),
        )?,
        validity: Validity {
            t_valid: parse_time(row.get::<String, _>("t_valid").as_str())?,
            t_invalid: row
                .get::<Option<String>, _>("t_invalid")
                .map(|s| parse_time(&s))
                .transpose()?,
            t_ingested: parse_time(row.get::<String, _>("t_ingested").as_str())?,
            t_expired: row
                .get::<Option<String>, _>("t_expired")
                .map(|s| parse_time(&s))
                .transpose()?,
        },
        provenance: Provenance {
            source: serde_json::from_str::<SourceRef>(&source)?,
            contributed_by: ActorId(row.get("prov_contributed_by")),
            written_by: ActorId(row.get("prov_written_by")),
            derived_from,
        },
        trust: Trust {
            tier: TrustTier::parse(row.get::<String, _>("trust_tier").as_str())?,
            score: row.get("trust_score"),
            checks: serde_json::from_str(row.get::<String, _>("trust_checks").as_str())?,
        },
        salience: serde_json::from_str::<Salience>(row.get::<String, _>("salience").as_str())?,
        links: Vec::<Link>::new(),
    })
}
