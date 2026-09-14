-- myelin ledger schema (PLAN.md §5.3).
--
-- The triggers below are the enforcement mechanism for invariants I1 and C9.
-- They live in the database rather than in Rust because the whole point of a
-- dual store is that the projection can be written by more than one process,
-- and an invariant only enforced on one write path is not an invariant.

CREATE TABLE IF NOT EXISTS event (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    kind      TEXT NOT NULL,
    record_id TEXT,
    actor     TEXT NOT NULL,
    reason    TEXT NOT NULL,
    detail    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS event_record ON event (record_id);
CREATE INDEX IF NOT EXISTS event_kind ON event (kind);

-- C9: append-only. No UPDATE, no DELETE, ever.
CREATE TRIGGER IF NOT EXISTS event_no_update BEFORE UPDATE ON event
BEGIN
    SELECT RAISE(ABORT, 'C9: event log is append-only');
END;

CREATE TRIGGER IF NOT EXISTS event_no_delete BEFORE DELETE ON event
BEGIN
    SELECT RAISE(ABORT, 'C9: event log is append-only');
END;

CREATE TABLE IF NOT EXISTS record (
    id                  TEXT PRIMARY KEY,
    kind                TEXT NOT NULL,
    tenant              TEXT NOT NULL,
    agent               TEXT NOT NULL,
    session             TEXT,
    namespace           TEXT NOT NULL,
    text                TEXT NOT NULL,
    entities            TEXT NOT NULL,
    t_valid             TEXT NOT NULL,
    t_invalid           TEXT,
    t_ingested          TEXT NOT NULL,
    t_expired           TEXT,
    trust_tier          TEXT NOT NULL,
    trust_score         REAL NOT NULL,
    trust_checks        TEXT NOT NULL,
    -- Nullable only so that I2 has something to catch. The Rust type makes
    -- provenance mandatory, so a NULL here means import, manual edit or
    -- corruption — and such a row is invisible to every read path.
    prov_source         TEXT,
    prov_contributed_by TEXT NOT NULL,
    prov_written_by     TEXT NOT NULL,
    prov_derived_from   TEXT NOT NULL,
    salience            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS record_scope ON record (tenant, namespace, agent);
CREATE INDEX IF NOT EXISTS record_live ON record (tenant, namespace, t_invalid, t_expired);

-- I1: a record is never mutated in place. Content, scope, provenance and the
-- two creation timestamps are frozen at insert. Only the retraction columns
-- (t_invalid, t_expired) and mutable state (trust, salience) may change.
CREATE TRIGGER IF NOT EXISTS record_immutable BEFORE UPDATE ON record
FOR EACH ROW WHEN
       OLD.text                IS NOT NEW.text
    OR OLD.kind                IS NOT NEW.kind
    OR OLD.tenant              IS NOT NEW.tenant
    OR OLD.agent               IS NOT NEW.agent
    OR OLD.session             IS NOT NEW.session
    OR OLD.namespace           IS NOT NEW.namespace
    OR OLD.t_valid             IS NOT NEW.t_valid
    OR OLD.t_ingested          IS NOT NEW.t_ingested
    OR OLD.prov_source         IS NOT NEW.prov_source
    OR OLD.prov_contributed_by IS NOT NEW.prov_contributed_by
    OR OLD.prov_written_by     IS NOT NEW.prov_written_by
    OR OLD.prov_derived_from   IS NOT NEW.prov_derived_from
BEGIN
    SELECT RAISE(ABORT, 'I1: record content, scope and provenance are immutable');
END;

-- I1: t_invalid records when a fact stopped being true. It is set once by
-- UPDATE or DELETE and is never overwritten or cleared.
CREATE TRIGGER IF NOT EXISTS record_t_invalid_write_once BEFORE UPDATE ON record
FOR EACH ROW WHEN OLD.t_invalid IS NOT NULL AND NEW.t_invalid IS NOT OLD.t_invalid
BEGIN
    SELECT RAISE(ABORT, 'I1: t_invalid is write-once');
END;

CREATE TABLE IF NOT EXISTS link (
    src      TEXT NOT NULL,
    dst      TEXT NOT NULL,
    relation TEXT NOT NULL,
    at       TEXT NOT NULL,
    PRIMARY KEY (src, dst, relation)
);
CREATE INDEX IF NOT EXISTS link_dst ON link (dst, relation);

-- The bipartite phrase <-> record structure PPR runs over (PLAN.md §5.4).
CREATE TABLE IF NOT EXISTS incidence (
    phrase    TEXT NOT NULL,
    record_id TEXT NOT NULL,
    weight    REAL NOT NULL,
    PRIMARY KEY (phrase, record_id)
);
CREATE INDEX IF NOT EXISTS incidence_record ON incidence (record_id);

-- C2: bipartite ACL. Revocation is edge removal; views are projected, never
-- copied.
CREATE TABLE IF NOT EXISTS acl_ua (
    user_id    TEXT NOT NULL,
    agent_id   TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    PRIMARY KEY (user_id, agent_id)
);

CREATE TABLE IF NOT EXISTS acl_ar (
    agent_id   TEXT NOT NULL,
    namespace  TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, namespace)
);

-- C4: staged writes awaiting promotion.
CREATE TABLE IF NOT EXISTS quarantine (
    id     TEXT PRIMARY KEY,
    record TEXT NOT NULL,
    reason TEXT NOT NULL,
    at     TEXT NOT NULL
);
