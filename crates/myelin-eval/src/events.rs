//! M50 — the events calendar, as two passes over a conversational corpus.
//!
//! **Extract** (`events-extract`) reads every distinct session once and asks
//! the reader for Chronos's event tuples (`10.48550/arXiv.2603.16862` §3.1,
//! [`myelin_core::pipeline::events`]). It touches no store: the result is a
//! JSONL cache keyed by the SHA-256 of the session as the extractor read it,
//! so the pass is resumable, shardable across hosts (`--shard i/n`), and
//! reusable across stores. It needs the reader and nothing else, which is
//! what lets it run on a second host while the first one's reader is held
//! for a measurement that must run alone.
//!
//! **Build** (`events-build`) resolves each event's `when` against the date
//! of *that copy* of the session — LongMemEval_S re-dates 5,283 of its
//! 25,112 session slots — and writes one `Semantic` record per event,
//! derived from that session's episodic records (I4). It needs the
//! embedder and the store and never calls the reader.
//!
//! One path each: a session whose extraction fails is left out of the cache
//! and the pass exits non-zero; the build refuses to start while any
//! session it needs is missing from the cache. Nothing is written partial.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write as _};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use futures_util::StreamExt;
use myelin_core::config::MyelinConfig;
use myelin_core::embed::remote::RemoteEmbedder;
use myelin_core::llm::openai::OpenAiLlm;
use myelin_core::model::query::ScopeFilter;
use myelin_core::model::record::{ActorId, RecordKind, Scope, SourceRef};
use myelin_core::pipeline::events::{
    event_text, extract_session, resolve_when, EventTime, ExtractedEvent, SessionTurn,
};
use myelin_core::pipeline::ingest::Turn;
use myelin_core::pipeline::write::{WritePath, WriteStats};
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::QdrantStore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::build::{parse_session_time, turns_for};
use crate::datasets::{locomo, longmemeval};

/// Progress line cadence, in sessions, for each pass.
/// What an event's source document carries after its session key, so a side
/// ledger can be checked to hold only events among its `semantic` records.
pub const EVENT_DOC_MARKER: &str = "@event";
const EXTRACT_PROGRESS_EVERY: usize = 25;
const BUILD_PROGRESS_EVERY: usize = 100;

/// One copy of one session inside one memory: what the extractor reads,
/// and everything the build needs to date and attach its events.
#[derive(Debug, Clone)]
pub struct SessionSlot {
    pub scope: Scope,
    /// Stable within the tenant: LoCoMo's `D<n>`, LongMemEval's session id.
    pub session_key: String,
    /// Source-document prefix of this session's episodic records, for I4.
    pub lineage_prefix: String,
    /// When this copy of the session happened.
    pub said: DateTime<Utc>,
    pub turns: Vec<SessionTurn>,
    /// SHA-256 of the session as rendered for the extractor. The cache key.
    pub content_sha: String,
}

fn sha_of(turns: &[SessionTurn]) -> String {
    let mut h = Sha256::new();
    for t in turns {
        h.update(t.speaker.as_bytes());
        h.update([0x1f]);
        h.update(t.text.as_bytes());
        h.update([0x1e]);
    }
    format!("{:x}", h.finalize())
}

/// LoCoMo: one tenant per conversation, one slot per session, turns exactly
/// as `build` segmented them (photo captions folded in, via `turns_for`).
pub fn locomo_slots(path: &Path, limit: Option<usize>) -> Result<Vec<SessionSlot>> {
    let convs = locomo::load(path)?;
    let mut out = Vec::new();
    for conv in convs.iter().take(limit.unwrap_or(usize::MAX)) {
        let scope = Scope::new(format!("locomo/{}", conv.sample_id), "myelin", "locomo");
        let all = turns_for(conv);
        for session in &conv.sessions {
            let unit = format!("{}#session{}", conv.sample_id, session.index);
            let said = session
                .date_time
                .as_deref()
                .and_then(parse_session_time)
                .with_context(|| {
                    format!("{} session {} has no parseable date; an event cannot be dated", conv.sample_id, session.index)
                })?;
            let turns: Vec<SessionTurn> = all
                .iter()
                .filter(|t| t.unit == unit)
                .map(|t| SessionTurn { speaker: t.speaker.clone(), text: t.text.clone() })
                .collect();
            if turns.is_empty() {
                continue;
            }
            let session_key = format!("D{}", session.index);
            out.push(SessionSlot {
                scope: scope.clone(),
                lineage_prefix: format!("{session_key}:"),
                session_key,
                said,
                content_sha: sha_of(&turns),
                turns,
            });
        }
    }
    Ok(out)
}

/// LongMemEval_S: one tenant per question, one slot per haystack session,
/// dated by that haystack's own `haystack_dates` entry. `questions`, when
/// non-empty, restricts to those tenants (M50's pilot population) and must
/// name only ids the corpus has.
pub fn longmemeval_slots(path: &Path, limit: Option<usize>, questions: &[String]) -> Result<Vec<SessionSlot>> {
    let mut items = longmemeval::load(path)?;
    if !questions.is_empty() {
        let known: HashSet<&str> = items.iter().map(|it| it.question_id.as_str()).collect();
        let unknown: Vec<&String> = questions.iter().filter(|q| !known.contains(q.as_str())).collect();
        anyhow::ensure!(unknown.is_empty(), "--questions names ids not in {}: {:?}", path.display(), unknown);
        items.retain(|it| questions.iter().any(|q| q == &it.question_id));
    }
    let mut out = Vec::new();
    for item in items.iter().take(limit.unwrap_or(usize::MAX)) {
        let scope = Scope::new(format!("lme_s/{}", item.question_id), "myelin", "longmemeval_s");
        let ids = item
            .haystack_session_ids
            .as_ref()
            .with_context(|| format!("{} has no haystack_session_ids", item.question_id))?;
        let dates = item
            .haystack_dates
            .as_ref()
            .with_context(|| format!("{} has no haystack_dates", item.question_id))?;
        anyhow::ensure!(
            ids.len() == item.haystack_sessions.len() && dates.len() == ids.len(),
            "{}: {} sessions, {} ids, {} dates",
            item.question_id,
            item.haystack_sessions.len(),
            ids.len(),
            dates.len()
        );
        for ((sid, date), session) in ids.iter().zip(dates).zip(&item.haystack_sessions) {
            let said = parse_session_time(date)
                .with_context(|| format!("{} session {sid}: unparseable date {date:?}", item.question_id))?;
            // Empty turns are skipped exactly as `build_longmemeval_s` skips them.
            let turns: Vec<SessionTurn> = session
                .iter()
                .filter(|t| !t.content.trim().is_empty())
                .map(|t| SessionTurn { speaker: t.role.clone(), text: t.content.clone() })
                .collect();
            if turns.is_empty() {
                continue;
            }
            out.push(SessionSlot {
                scope: scope.clone(),
                session_key: sid.clone(),
                lineage_prefix: format!("{sid}#"),
                said,
                content_sha: sha_of(&turns),
                turns,
            });
        }
    }
    Ok(out)
}

/// One line of the extraction cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLine {
    pub sha256: String,
    /// The reader that extracted it, from the run's config.
    pub model: String,
    pub events: Vec<ExtractedEvent>,
}

/// Read every cache file into one map. A key that appears twice with
/// different events is refused: two shards disagreeing is a fact to look
/// at, not a tie to break.
pub fn load_cache(paths: &[String]) -> Result<HashMap<String, Vec<ExtractedEvent>>> {
    let mut map: HashMap<String, Vec<ExtractedEvent>> = HashMap::new();
    for p in paths {
        let file = std::fs::File::open(p).with_context(|| format!("open {p}"))?;
        for (n, line) in std::io::BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let c: CacheLine =
                serde_json::from_str(&line).with_context(|| format!("{p}:{}: not a cache line", n + 1))?;
            match map.get(&c.sha256) {
                Some(prev) if prev != &c.events => {
                    anyhow::bail!("{p}:{}: session {} extracted twice with different events", n + 1, c.sha256)
                }
                _ => {
                    map.insert(c.sha256, c.events);
                }
            }
        }
    }
    Ok(map)
}

/// `i/n`, both integers, `i < n`.
pub fn parse_shard(s: &str) -> Result<(usize, usize)> {
    let (i, n) = s
        .split_once('/')
        .with_context(|| format!("--shard {s:?}: expected i/n"))?;
    let (i, n): (usize, usize) = (i.trim().parse()?, n.trim().parse()?);
    anyhow::ensure!(n > 0 && i < n, "--shard {s:?}: need 0 <= i < n");
    Ok((i, n))
}

#[derive(Debug, Default)]
pub struct ExtractReport {
    pub distinct: usize,
    pub in_shard: usize,
    pub cached: usize,
    pub extracted: usize,
    pub failed: usize,
    pub events: usize,
    pub wall_secs: f64,
}

/// Extract every distinct session in `slots` (this shard's share of them)
/// that `out` does not already hold, `concurrency` calls in flight, one
/// line appended and flushed per session as it finishes.
pub async fn extract(
    slots: &[SessionSlot],
    out: &Path,
    shard: (usize, usize),
    concurrency: usize,
) -> Result<ExtractReport> {
    let cfg = MyelinConfig::load().context("load myelin config")?;
    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("reader client")?;

    // Distinct contents in a fixed order, so `--shard` partitions the same
    // way on every host.
    let mut by_sha: HashMap<&str, &SessionSlot> = HashMap::new();
    for s in slots {
        by_sha.entry(s.content_sha.as_str()).or_insert(s);
    }
    let mut keys: Vec<&str> = by_sha.keys().copied().collect();
    keys.sort_unstable();
    let mut report = ExtractReport { distinct: keys.len(), ..Default::default() };
    let mine: Vec<&str> = keys
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % shard.1 == shard.0)
        .map(|(_, k)| k)
        .collect();
    report.in_shard = mine.len();

    let done: HashSet<String> = if out.exists() {
        load_cache(&[out.display().to_string()])?.into_keys().collect()
    } else {
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        HashSet::new()
    };
    let todo: Vec<&SessionSlot> = mine
        .iter()
        .filter(|k| !done.contains(**k))
        .map(|k| by_sha[k])
        .collect();
    report.cached = report.in_shard - todo.len();
    eprintln!(
        "  events-extract: {} distinct sessions, {} in shard {}/{}, {} cached, {} to extract",
        report.distinct, report.in_shard, shard.0, shard.1, report.cached, todo.len()
    );

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .with_context(|| format!("open {}", out.display()))?;
    let started = Instant::now();
    let total = todo.len();
    let llm_ref = &llm;
    let mut stream = futures_util::stream::iter(todo)
        .map(|slot| async move { (slot, extract_session(llm_ref, &slot.turns).await) })
        .buffer_unordered(concurrency.max(1));
    let mut n = 0usize;
    while let Some((slot, result)) = stream.next().await {
        n += 1;
        match result {
            Ok(events) => {
                report.extracted += 1;
                report.events += events.len();
                let line = CacheLine {
                    sha256: slot.content_sha.clone(),
                    model: cfg.llm.model.clone(),
                    events,
                };
                writeln!(file, "{}", serde_json::to_string(&line)?)?;
                file.flush()?;
            }
            Err(e) => {
                report.failed += 1;
                eprintln!(
                    "  {} {}: extraction failed, left out of the cache ({e})",
                    slot.scope.tenant, slot.session_key
                );
            }
        }
        if n.is_multiple_of(EXTRACT_PROGRESS_EVERY) || n == total {
            let secs = started.elapsed().as_secs_f64();
            eprintln!(
                "  [{n:>6}/{total}] events={} failed={} {:.1}s/session",
                report.events,
                report.failed,
                secs / n as f64
            );
        }
    }
    report.wall_secs = started.elapsed().as_secs_f64();
    Ok(report)
}

#[derive(Debug, Default)]
pub struct BuildEventsReport {
    pub slots: usize,
    pub resumed: usize,
    pub events: usize,
    pub stated: usize,
    pub unresolved: usize,
    pub said: usize,
    pub total: WriteStats,
    pub wall_secs: f64,
}

fn day_start(d: NaiveDate) -> DateTime<Utc> {
    Utc.from_utc_datetime(&d.and_time(chrono::NaiveTime::MIN))
}

/// Write the cached events of every slot into `collection`/`ledger`.
pub async fn build(
    slots: &[SessionSlot],
    cache: &HashMap<String, Vec<ExtractedEvent>>,
    collection: &str,
    ledger_path: &Path,
    concurrency: usize,
) -> Result<BuildEventsReport> {
    let missing = slots.iter().filter(|s| !cache.contains_key(&s.content_sha)).count();
    anyhow::ensure!(
        missing == 0,
        "{missing} of {} session slots have no cached extraction; run events-extract until it \
         exits cleanly, then build. Nothing was written.",
        slots.len()
    );

    let cfg = MyelinConfig::load().context("load myelin config")?;
    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("reader client")?;
    let embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)
        .context("embedder client")?;
    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    store.ensure_collection(cfg.embed.dim, false).await.context("ensure collection")?;
    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;

    let started = Instant::now();
    let mut report = BuildEventsReport { slots: slots.len(), ..Default::default() };
    for (i, slot) in slots.iter().enumerate() {
        let unit_key = format!("{}#events/{}", slot.scope.tenant, slot.session_key);
        if ledger.unit_is_complete(&unit_key).await? {
            report.resumed += 1;
            continue;
        }
        let events = &cache[&slot.content_sha];
        let said_day = slot.said.date_naive();
        let mut turns = Vec::with_capacity(events.len());
        for (k, e) in events.iter().enumerate() {
            let time = resolve_when(&e.when, said_day);
            match &time {
                EventTime::Stated { .. } => report.stated += 1,
                EventTime::Unresolved { .. } => report.unresolved += 1,
                EventTime::Said => report.said += 1,
            }
            let at = match &time {
                EventTime::Stated { .. } => day_start(time.t_valid(said_day)),
                EventTime::Said | EventTime::Unresolved { .. } => slot.said,
            };
            // `@event` rather than the session's own `#`/`:` separator, so
            // an event's source never falls under its session's lineage
            // prefix and a retried session cannot name its own events as
            // ancestors.
            let doc = format!("{}{EVENT_DOC_MARKER}{k}", slot.session_key);
            turns.push(Turn {
                speaker: "event".into(),
                text: event_text(e, &time, said_day),
                at: Some(at),
                source: SourceRef::doc(doc.clone()),
                // One unit per event: the segmenter never merges across
                // units, so each event is its own record.
                unit: doc,
            });
        }
        let mut written = 0usize;
        if !turns.is_empty() {
            let filter = ScopeFilter::tenant(&slot.scope.tenant).with_namespace(&slot.scope.namespace);
            let ancestors = ledger.ids_from_source_docs(&filter, &slot.lineage_prefix).await?;
            anyhow::ensure!(
                !ancestors.is_empty(),
                "{} {}: no episodic records under source prefix {:?} in {}. The events pass \
                 abstracts over the episodic store and cannot run before it (I4).",
                slot.scope.tenant,
                slot.session_key,
                slot.lineage_prefix,
                ledger_path.display()
            );
            let mut write = WritePath::new(&llm, &embedder, &store, &ledger);
            write.record_kind = RecordKind::Semantic;
            write.extract_facts = false;
            write.derived_from = ancestors;
            write.concurrency = concurrency.max(1);
            let stats = write
                .insert(&slot.scope, &turns)
                .await
                .with_context(|| format!("write events for {} {}", slot.scope.tenant, slot.session_key))?;
            written = turns.len();
            report.total.merge(&stats);
        }
        report.events += written;
        ledger
            .mark_unit_complete(
                &unit_key,
                &ActorId::new("myelin-eval"),
                serde_json::json!({"session": slot.session_key, "events": written, "sha256": slot.content_sha}),
            )
            .await?;
        if (i + 1).is_multiple_of(BUILD_PROGRESS_EVERY) || i + 1 == slots.len() {
            eprintln!(
                "  [{:>6}/{}] events={} stated={} unresolved={} said={} {:.1}s",
                i + 1,
                slots.len(),
                report.events,
                report.stated,
                report.unresolved,
                report.said,
                started.elapsed().as_secs_f64()
            );
        }
    }
    report.wall_secs = started.elapsed().as_secs_f64();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_parses_and_refuses_out_of_range() {
        assert_eq!(parse_shard("0/1").ok(), Some((0, 1)));
        assert_eq!(parse_shard("1/2").ok(), Some((1, 2)));
        assert!(parse_shard("2/2").is_err());
        assert!(parse_shard("0/0").is_err());
        assert!(parse_shard("x").is_err());
    }

    #[test]
    fn sha_depends_on_speaker_and_text_boundaries() {
        let a = [SessionTurn { speaker: "ab".into(), text: "c".into() }];
        let b = [SessionTurn { speaker: "a".into(), text: "bc".into() }];
        assert_ne!(sha_of(&a), sha_of(&b));
        assert_eq!(sha_of(&a), sha_of(&a.clone()));
    }

    #[test]
    fn cache_refuses_two_shards_that_disagree() {
        let dir = std::env::temp_dir().join(format!("myelin-events-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let ev = |o: &str| ExtractedEvent {
            subject: "the user".into(),
            verb: "bought".into(),
            object: o.into(),
            when: String::new(),
            aliases: vec!["x".into(), "y".into()],
        };
        let line = |o: &str| {
            serde_json::to_string(&CacheLine { sha256: "k".into(), model: "m".into(), events: vec![ev(o)] })
                .expect("serialise")
        };
        let (p1, p2, p3) = (dir.join("a.jsonl"), dir.join("b.jsonl"), dir.join("c.jsonl"));
        std::fs::write(&p1, line("a Fitbit") + "\n").expect("write");
        std::fs::write(&p2, line("a Fitbit") + "\n").expect("write");
        std::fs::write(&p3, line("a Garmin") + "\n").expect("write");
        let s = |p: &Path| p.display().to_string();
        assert_eq!(load_cache(&[s(&p1), s(&p2)]).map(|m| m.len()).ok(), Some(1));
        assert!(load_cache(&[s(&p1), s(&p3)]).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn locomo_slots_cover_every_session_with_its_own_date() {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/locomo10.json"));
        if !path.exists() {
            eprintln!("skipping: {} not fetched", path.display());
            return;
        }
        let slots = locomo_slots(path, None).expect("locomo slots");
        assert_eq!(slots.len(), 272, "LoCoMo has 272 sessions");
        assert!(slots.iter().all(|s| s.lineage_prefix.starts_with('D') && s.lineage_prefix.ends_with(':')));
        let turns: usize = slots.iter().map(|s| s.turns.len()).sum();
        assert_eq!(turns, 5_882, "every LoCoMo turn is read exactly once");
    }
}
