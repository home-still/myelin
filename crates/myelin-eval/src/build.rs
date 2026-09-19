//! `myelin-eval build` — drive a corpus through the write path (`PLAN.md` M3).
//!
//! The milestone asks for three numbers per corpus: **records/unit, tokens,
//! wall time**. They are reported per conversation and in total, because an
//! average over ten conversations hides the one that produced nothing.
//!
//! Ingest granularity is one whole unit (R2): a LoCoMo conversation goes in as
//! a conversation, and segmentation into episodes is ours to do.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use myelin_core::config::MyelinConfig;
use myelin_core::embed::remote::RemoteEmbedder;
use myelin_core::llm::openai::OpenAiLlm;
use myelin_core::model::record::{ActorId, Scope, SourceRef};
use myelin_core::pipeline::ingest::Turn;
use myelin_core::pipeline::write::{WritePath, WriteStats};
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::QdrantStore;
use myelin_core::store::reconcile::reconcile;

use crate::datasets::lmev2;
use crate::datasets::longmemeval;
use crate::datasets::locomo::{self, LocomoConversation};

/// Parse a corpus session timestamp.
///
/// Two formats, because the two corpora write time two ways and both feed
/// the same `t_valid`:
///
/// - LoCoMo: `1:56 pm on 8 May, 2023`
/// - LongMemEval_S: `2023/05/20 (Sat) 02:21`
///
/// The LongMemEval form was **not handled until M19**, and the cost was not
/// a missing field: `WritePath` falls back to the ingest time, so all 162,181
/// records of `longmemeval_s` carried the *build date* as their `t_valid`,
/// every one of the 500 memories showed the reader `[2026-09-15]` under
/// `ComposeConfig::stamp_valid_time`, and the 133 temporal-reasoning
/// questions were being asked of a corpus with no time in it. Found by
/// M19's `--timeline` arm, whose dated index came out as six entries at
/// `+0d`. See `docs/measurements/m19-temporal-resolution.md`.
///
/// A turn with no parseable time simply has none — the segmenter treats a
/// missing timestamp as "no gap evidence" rather than inventing one.
///
/// `pub(crate)` because `bench` needs the same cleanup to derive a
/// conversation's reference date; a second copy of the `" on "`/comma
/// handling would drift.
pub(crate) fn parse_session_time(s: &str) -> Option<DateTime<Utc>> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    // LongMemEval_S first: its `(Sat)` weekday is redundant with the date
    // and `%a` will only match it in the right position, so a false positive
    // is not possible.
    for fmt in ["%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, fmt) {
            return Some(Utc.from_utc_datetime(&naive));
        }
    }
    let cleaned = trimmed.replace(" on ", " ").replace(',', "");
    // Real LoCoMo always carries a time ("1:56 pm on 8 May, 2023"); the
    // date-only branch is a fallback for corpora that do not.
    for fmt in ["%l:%M %P %d %B %Y", "%I:%M %p %d %B %Y"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(&cleaned, fmt) {
            return Some(Utc.from_utc_datetime(&naive));
        }
    }
    if let Ok(date) = NaiveDate::parse_from_str(&cleaned, "%d %B %Y") {
        return date.and_hms_opt(0, 0, 0).map(|n| Utc.from_utc_datetime(&n));
    }
    None
}

/// LoCoMo turns in wire order, with photo captions folded into the text.
///
/// Public because the ablation harness re-runs the exact same segmentation to
/// reconstruct which turns each episode covers; a second, drifting copy of
/// this function would silently mis-score every retrieval.
pub fn turns_for(conv: &LocomoConversation) -> Vec<Turn> {
    let mut turns = Vec::new();
    for session in &conv.sessions {
        let at = session.date_time.as_deref().and_then(parse_session_time);
        for t in &session.turns {
            let mut text = t.text.clone();
            // A shared photo is part of what was said. Dropping the caption
            // loses evidence that LoCoMo questions are scored against.
            if let Some(caption) = &t.blip_caption {
                if !caption.trim().is_empty() {
                    text.push_str(&format!(" [shared an image: {caption}]"));
                }
            }
            turns.push(Turn {
                speaker: t.speaker.clone(),
                text,
                at,
                // `dia_id` is LoCoMo's own evidence pointer, so it IS the
                // provenance the harness will check answers against.
                source: SourceRef::doc(t.dia_id.clone()),
                unit: format!("{}#session{}", conv.sample_id, session.index),
            });
        }
    }
    turns
}


/// Check the ledger against the vector store, and optionally repair.
///
/// Run at the end of every build because a partial write does not announce
/// itself. The first full LoCoMo run left two records in the ledger with no
/// vector — invisible to every read path while still being exported and
/// counted — and it was only found by diffing the two stores by hand
/// afterwards. A build that ends silently on a drifted memory is a build
/// whose numbers are measured on a corpus nobody has checked.
async fn check_drift(
    ledger: &Ledger,
    store: &QdrantStore,
    embedder: &RemoteEmbedder,
    namespace: &str,
    repair: bool,
) -> Result<()> {
    let report = reconcile(ledger, store, namespace, Some(embedder), repair)
        .await
        .context("reconcile after build")?;
    if report.total() == 0 {
        // No content or scope drift. `total()` deliberately excludes
        // `missing_t_valid` (a migration gap, not a content invariant), so a
        // pre-migration corpus reconciles clean on every invariant and only
        // needs the one-time field migration — the build must not fail on it.
        if report.missing_t_valid > 0 {
            if report.repaired {
                eprintln!(
                    "  reconcile: brought t_valid to {} point(s) ({namespace}) — \
                     pre-migration",
                    report.missing_t_valid
                );
            } else {
                eprintln!(
                    "  reconcile: {} point(s) predate the t_valid field ({namespace}); \
                     re-run with --repair to migrate them",
                    report.missing_t_valid
                );
            }
        } else {
            eprintln!("  reconcile: clean ({namespace})");
        }
        return Ok(());
    }
    eprintln!(
        "  reconcile: {} drift item(s) in {namespace} — missing_vectors={} qdrant_orphans={} \
         payload_drift={} stale_points={} dangling_links={} orphan_incidence={} \
         missing_provenance={} missing_t_valid={} (repaired={})",
        report.total(),
        report.missing_vectors.len(),
        report.qdrant_orphans.len(),
        report.payload_drift.len(),
        report.stale_points.len(),
        report.dangling_links.len(),
        report.orphan_incidence.len(),
        report.missing_provenance.len(),
        report.missing_t_valid,
        report.repaired,
    );
    anyhow::ensure!(
        repair,
        "memory is drifted; re-run with --repair (nothing downstream should be \
         measured on an unchecked corpus)"
    );
    Ok(())
}

pub struct BuildReport {
    pub per_unit: Vec<(String, WriteStats)>,
    pub total: WriteStats,
    pub wall_secs: f64,
    /// Sessions the build walked, and how many of them carried no date this
    /// build could parse.
    ///
    /// An unparseable session date is not an error anywhere in the write
    /// path: `EpisodeDraft::t_valid` falls back to `Utc::now()`, so the
    /// record silently gets the *build date* as its valid time. That is
    /// exactly the M19 incident — 162,181 LongMemEval records stamped
    /// 2026-09-15 because `parse_session_time` only understood LoCoMo's
    /// format — and it survived six milestones because nothing counted it.
    pub sessions_total: usize,
    pub sessions_without_date: usize,
}

/// Ingest LoCoMo conversations into a namespace, one tenant per conversation.
///
/// Per-conversation tenants are not cosmetic: LoCoMo answers must come from
/// the conversation that asked, and a shared tenant would let retrieval leak
/// across conversations and silently inflate every score. It also exercises
/// the C12 isolation path on real data.
pub async fn build_locomo(
    path: &Path,
    collection: &str,
    ledger_path: &Path,
    limit: Option<usize>,
    repair: bool,
) -> Result<BuildReport> {
    let cfg = MyelinConfig::load().context("load myelin config")?;

    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("reader client")?;
    let embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)
        .context("embedder client")?;

    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    store
        .ensure_collection(cfg.embed.dim, false)
        .await
        .context("ensure collection")?;

    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;

    let conversations = locomo::load(path)?;
    let conversations: Vec<_> = match limit {
        Some(n) => conversations.into_iter().take(n).collect(),
        None => conversations,
    };

    let started = Instant::now();
    let mut report = BuildReport {
        per_unit: Vec::new(),
        total: WriteStats::default(),
        wall_secs: 0.0,
        sessions_total: 0,
        sessions_without_date: 0,
    };

    // Resume: skip a conversation the ledger records as fully ingested.
    //
    // Not a convenience. A LoCoMo conversation costs ~13 minutes of GPU on a
    // shared card, and the first full run died at conv-44 after six had
    // committed. Re-running those six to reach the seventh spends 80 minutes
    // of someone else's GPU on records the ledger already holds, and the v5
    // ids mean the work is discarded as duplicates anyway.
    //
    // The predicate is the `unit_complete` audit event, NOT a row count:
    // conv-44 died holding 79 records and all 62 of its episodes, so any
    // count-based test would have skipped the consolidation that never ran.
    let mut resumed = 0usize;

    for conv in &conversations {
        let scope = Scope::new(format!("locomo/{}", conv.sample_id), "myelin", "locomo");
        if ledger.unit_is_complete(&scope.tenant).await? {
            resumed += 1;
            eprintln!("  {:<8} already ingested, skipping", conv.sample_id);
            continue;
        }
        // Same tally as the LongMemEval path: a session whose date does not
        // parse produces episodes stamped with the build date.
        report.sessions_total += conv.sessions.len();
        report.sessions_without_date += conv
            .sessions
            .iter()
            .filter(|s| s.date_time.as_deref().and_then(parse_session_time).is_none())
            .count();
        let turns = turns_for(conv);

        let mut write = WritePath::new(&llm, &embedder, &store, &ledger);
        write.progress = true;
        let stats = write
            .insert(&scope, &turns)
            .await
            .with_context(|| format!("ingest {}", conv.sample_id))?;

        eprintln!(
            "  {:<8} turns={:<5} episodes={:<4} adj={:<3} candidates={:<5} add={:<5} upd={:<4} dup={:<5} quar={:<4} rej={:<4} {:.1}s",
            conv.sample_id,
            stats.turns,
            stats.episodes,
            stats.adjudicated_out,
            stats.candidates,
            stats.added,
            stats.updated,
            stats.duplicates,
            stats.quarantined,
            stats.rejected,
            stats.wall_ms as f64 / 1000.0,
        );

        ledger
            .mark_unit_complete(
                &scope.tenant,
                &ActorId::new("myelin-eval"),
                serde_json::json!({
                    "turns": stats.turns,
                    "episodes": stats.episodes,
                    "added": stats.added,
                    "wall_ms": stats.wall_ms,
                }),
            )
            .await
            .context("mark unit complete")?;

        report.total.merge(&stats);
        report.per_unit.push((conv.sample_id.clone(), stats));
    }

    report.wall_secs = started.elapsed().as_secs_f64();
    if resumed > 0 {
        eprintln!("  resumed: {resumed} conversation(s) already in the ledger");
    }
    check_drift(&ledger, &store, &embedder, "locomo", repair).await?;
    Ok(report)
}

/// Ingest an LME-V2 tier into one tenant **per domain** (`PLAN.md` M3).
///
/// Two tenants, not 451: see [`crate::datasets::lmev2`] for the measurement
/// behind that, and for why the accessibility trees are chunked rather than
/// dropped or deduplicated.
///
/// Extraction is off. That is `WritePath::extract_facts`'s documented case:
/// a fact-extraction pass over this corpus extrapolates to ~250 GPU-hours,
/// and LME-V2's own paper reports a controller over raw trajectories beating
/// their extracted RAG memory by +16.3/+13.1.
#[allow(clippy::too_many_arguments)]
pub async fn build_lmev2(
    trajectories: &Path,
    haystack_path: &Path,
    questions_path: &Path,
    tier: &str,
    collection: &str,
    ledger_path: &Path,
    limit: Option<usize>,
    repair: bool,
) -> Result<BuildReport> {
    let cfg = MyelinConfig::load().context("load myelin config")?;

    let questions = lmev2::load_questions(questions_path)?;
    let haystack = lmev2::load_haystack(haystack_path)?;
    let by_domain = lmev2::haystacks_by_domain(&haystack, &questions)?;

    // `limit` truncates *per domain* so a smoke run still exercises both
    // tenants; truncating the global set would silently test only one.
    let mut domain_of: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut wanted: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (domain, ids) in &by_domain {
        for id in ids.iter().take(limit.unwrap_or(usize::MAX)) {
            domain_of.insert(id.clone(), domain.clone());
            wanted.insert(id.clone());
        }
        eprintln!(
            "  {tier}/{domain}: {} trajectories ({} in haystack)",
            limit.map_or(ids.len(), |n| n.min(ids.len())),
            ids.len()
        );
    }

    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("reader client")?;
    let embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)
        .context("embedder client")?;

    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    store
        .ensure_collection(cfg.embed.dim, false)
        .await
        .context("ensure collection")?;

    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;

    let started = Instant::now();
    let mut report = BuildReport {
        per_unit: Vec::new(),
        total: WriteStats::default(),
        wall_secs: 0.0,
        sessions_total: 0,
        sessions_without_date: 0,
    };

    // Stream the 1.2 GB file on a blocking thread and hand trajectories to
    // the async write path over a depth-2 channel.
    //
    // Buffering all 200 first would hold ~172 MB (measured mean 862 KB per
    // trajectory) for no benefit. Depth 2 bounds that to ~2 MB and still
    // overlaps the file read with GPU work, which is the only overlap
    // available here: the reader is idle because extraction is off.
    let want_count = wanted.len();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<lmev2::Trajectory>(2);
    let traj_path = trajectories.to_path_buf();
    let reader = tokio::task::spawn_blocking(move || {
        lmev2::for_each_trajectory(&traj_path, &wanted, |t| {
            tx.blocking_send(t)
                .map_err(|_| anyhow::anyhow!("ingest stopped before the file was consumed"))
        })
    });

    while let Some(traj) = rx.recv().await {
        let domain = &domain_of[&traj.id];
        let scope = Scope::new(format!("{tier}/{domain}"), "myelin", tier);
        let turns = lmev2::turns_for(&traj);

        let mut write = WritePath::new(&llm, &embedder, &store, &ledger);
        write.extract_facts = false;
        let stats = write
            .insert(&scope, &turns)
            .await
            .with_context(|| format!("ingest {}", traj.id))?;

        eprintln!(
            "  {:<10} {:<10} turns={:<5} episodes={:<5} add={:<5} dup={:<5} {:.1}s",
            traj.id,
            domain,
            stats.turns,
            stats.episodes,
            stats.added,
            stats.duplicates,
            stats.wall_ms as f64 / 1000.0,
        );

        report.total.merge(&stats);
        report.per_unit.push((traj.id.clone(), stats));
    }

    // A trajectory named by the haystack but absent from the file would
    // silently shrink the memory, so the count is checked rather than logged.
    let found = reader.await.context("trajectory reader task")??;
    if found != want_count || report.per_unit.len() != want_count {
        anyhow::bail!(
            "haystack names {want_count} trajectories; read {found}, ingested {}",
            report.per_unit.len()
        );
    }

    report.wall_secs = started.elapsed().as_secs_f64();
    check_drift(&ledger, &store, &embedder, tier, repair).await?;
    Ok(report)
}

/// Ingest LongMemEval_S: 500 questions, each with its own haystack.
///
/// Unlike LME-V2-Small — where 200 trajectories collapse to two byte-identical
/// haystacks and therefore two memories — every LongMemEval_S question carries
/// an independent conversation history. So this writes **500 separate
/// memories**, one tenant per `question_id`, and a question may only ever be
/// answered from its own. Sharing one tenant would leak 499 other haystacks
/// into every recall and turn the benchmark into a different, much easier one.
///
/// Extraction is off, for the same reason as `build_lmev2`: 61.2M tokens of
/// haystack through a fact-extraction pass is hundreds of GPU-hours, and the
/// episodic path is what the throughput measurement in
/// `docs/measurements/m3-write-path.md` is calibrated on.
#[allow(clippy::too_many_arguments)]
pub async fn build_longmemeval_s(
    dataset: &Path,
    collection: &str,
    ledger_path: &Path,
    limit: Option<usize>,
    repair: bool,
) -> Result<BuildReport> {
    let cfg = MyelinConfig::load().context("load myelin config")?;
    let mut items = longmemeval::load(dataset).context("load longmemeval_s")?;
    if let Some(n) = limit {
        items.truncate(n);
    }
    eprintln!("  longmemeval_s: {} questions, one memory each", items.len());

    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("reader client")?;
    let embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)
        .context("embedder client")?;

    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    store
        .ensure_collection(cfg.embed.dim, false)
        .await
        .context("ensure collection")?;
    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;

    let started = Instant::now();
    let mut report = BuildReport {
        per_unit: Vec::new(),
        total: WriteStats::default(),
        wall_secs: 0.0,
        sessions_total: 0,
        sessions_without_date: 0,
    };
    // Resume, exactly as `build_locomo` does. Without this a crash at unit
    // 341 of 500 costs a full re-walk: the write path is *safe* to repeat
    // (episode ids are content-derived, so an existing record is a no-op)
    // but not *cheap* — it re-embeds every episode of every completed
    // tenant. Measured in M19: 29 s per already-ingested unit, which is 4
    // hours over the corpus. Completion is a fact about the process, so it
    // is recorded as an audit event rather than inferred from a row count.
    let mut resumed = 0usize;

    for (qi, item) in items.iter().enumerate() {
        let scope = Scope::new(
            format!("lme_s/{}", item.question_id),
            "myelin",
            "longmemeval_s",
        );
        if ledger.unit_is_complete(&scope.tenant).await? {
            resumed += 1;
            continue;
        }

        let mut turns = Vec::new();
        for (si, session) in item.haystack_sessions.iter().enumerate() {
            // `haystack_dates` is parallel to `haystack_sessions`. It is the
            // only absolute time in the corpus, and temporal-reasoning is 133
            // of the 500 questions, so losing it costs a whole question type.
            let at = item
                .haystack_dates
                .as_ref()
                .and_then(|d| d.get(si))
                .and_then(|s| parse_session_time(s));
            // Counted, not shrugged at: `None` here means the episode will be
            // stamped with the build date instead (M19).
            report.sessions_total += 1;
            if at.is_none() {
                report.sessions_without_date += 1;
            }
            let sid = item
                .haystack_session_ids
                .as_ref()
                .and_then(|ids| ids.get(si).cloned())
                .unwrap_or_else(|| format!("session{si}"));
            for (ti, turn) in session.iter().enumerate() {
                if turn.content.trim().is_empty() {
                    continue;
                }
                turns.push(Turn {
                    speaker: turn.role.clone(),
                    text: turn.content.clone(),
                    at,
                    source: SourceRef::doc(format!("{sid}#{ti}")),
                    unit: sid.clone(),
                });
            }
        }

        let mut write = WritePath::new(&llm, &embedder, &store, &ledger);
        write.extract_facts = false;
        let stats = write
            .insert(&scope, &turns)
            .await
            .with_context(|| format!("ingest {}", item.question_id))?;

        if qi % 25 == 0 || qi + 1 == items.len() {
            eprintln!(
                "  [{:>3}/{}] {:<24} turns={:<5} episodes={:<5} add={:<5} dup={:<5} {:.1}s",
                qi + 1,
                items.len(),
                item.question_id,
                stats.turns,
                stats.episodes,
                stats.added,
                stats.duplicates,
                stats.wall_ms as f64 / 1000.0,
            );
        }

        ledger
            .mark_unit_complete(
                &scope.tenant,
                &ActorId::new("myelin-eval"),
                serde_json::json!({
                    "turns": stats.turns,
                    "episodes": stats.episodes,
                    "wall_ms": stats.wall_ms,
                }),
            )
            .await?;
        report.total.merge(&stats);
        report.per_unit.push((item.question_id.clone(), stats));
    }
    if resumed > 0 {
        eprintln!("  resumed: {resumed} of {} units already ingested", items.len());
    }

    report.wall_secs = started.elapsed().as_secs_f64();
    check_drift(&ledger, &store, &embedder, "longmemeval_s", repair).await?;
    Ok(report)
}

impl BuildReport {
    pub fn print(&self, units: usize) {
        let t = &self.total;
        println!("\n=== write path, {units} units ===");
        println!("turns              {}", t.turns);
        println!("episodes stored    {}", t.episodes);
        // A gate that removes episodes must be visible in the number of
        // episodes it removed, or a shrinking corpus looks like a
        // segmentation change (M15).
        println!("  adjudicated out  {}", t.adjudicated_out);
        // The four-op counters describe SEMANTIC deltas only. On an
        // episodic-only corpus they are all zero while `episodes stored` is
        // large, and reading `added 0` as "nothing was written" is the
        // obvious misreading, so the block is labelled.
        println!("semantic candidates {}", t.candidates);
        println!("  add             {}", t.added);
        println!("  update          {}", t.updated);
        println!("  delete          {}", t.deleted);
        println!("  noop            {}", t.noop);
        println!("  duplicate       {}", t.duplicates);
        println!("  quarantined     {}", t.quarantined);
        println!("  rejected        {}", t.rejected);
        println!("approx tokens in   {}", t.approx_tokens);
        // Printed unconditionally, including the zero: "0 of 1,500" is the
        // evidence that the dates landed. A silent counter proves nothing.
        println!(
            "sessions w/o date  {} of {}",
            self.sessions_without_date, self.sessions_total
        );
        println!("records/unit       {:.2}", t.records_per_unit(units));
        println!("wall               {:.1}s", self.wall_secs);
        println!(
            "  adjudicate       {:.1}s
  extract          {:.1}s
  consolidate      {:.1}s
  index            {:.1}s",
            t.adjudicate_ms as f64 / 1000.0,
            t.extract_ms as f64 / 1000.0,
            t.consolidate_ms as f64 / 1000.0,
            t.index_ms as f64 / 1000.0,
        );
        if self.wall_secs > 0.0 {
            println!(
                "throughput         {:.1} episodes/s, {:.0} input tokens/s",
                t.episodes as f64 / self.wall_secs,
                t.approx_tokens as f64 / self.wall_secs,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LoCoMo's own date format must actually parse, or every episode loses
    /// its `t_valid` and the temporal questions become unanswerable.
    #[test]
    fn locomo_session_timestamps_parse() {
        let parsed = parse_session_time("1:56 pm on 8 May, 2023").expect("should parse");
        assert_eq!(parsed.format("%Y-%m-%d %H:%M").to_string(), "2023-05-08 13:56");

        let midnight = parse_session_time("7 May, 2023").expect("date-only should parse");
        assert_eq!(midnight.format("%Y-%m-%d").to_string(), "2023-05-07");
    }

    /// LongMemEval_S's format, which went unparsed from M6 to M19 and cost
    /// the whole corpus its `t_valid`: `WritePath` falls back to the ingest
    /// time, so every memory showed the reader the build date instead of the
    /// conversation date.
    #[test]
    fn longmemeval_session_timestamps_parse() {
        let parsed = parse_session_time("2023/05/20 (Sat) 02:21").expect("should parse");
        assert_eq!(parsed.format("%Y-%m-%d %H:%M").to_string(), "2023-05-20 02:21");
        // The weekday is redundant with the date and is allowed to be absent.
        let no_day = parse_session_time("2023/05/30 23:40").expect("should parse");
        assert_eq!(no_day.format("%Y-%m-%d %H:%M").to_string(), "2023-05-30 23:40");
    }

    /// An unparseable timestamp must yield `None`, not a fabricated date: a
    /// wrong `t_valid` is worse than a missing one because it silently
    /// reorders the bi-temporal history.
    #[test]
    fn an_unparseable_timestamp_is_none() {
        assert!(parse_session_time("").is_none());
        assert!(parse_session_time("sometime last spring").is_none());
    }
}
