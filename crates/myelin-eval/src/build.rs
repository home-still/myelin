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

use crate::datasets::lmev2;
use crate::datasets::locomo::{self, LocomoConversation};

/// LoCoMo timestamps look like `1:56 pm on 8 May, 2023`. A turn with no
/// parseable time simply has none — the segmenter treats a missing timestamp
/// as "no gap evidence" rather than inventing one.
fn parse_locomo_time(s: &str) -> Option<DateTime<Utc>> {
    let cleaned = s.trim().replace(" on ", " ").replace(',', "");
    if cleaned.is_empty() {
        return None;
    }
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
        let at = session.date_time.as_deref().and_then(parse_locomo_time);
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

pub struct BuildReport {
    pub per_unit: Vec<(String, WriteStats)>,
    pub total: WriteStats,
    pub wall_secs: f64,
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
        let turns = turns_for(conv);

        let mut write = WritePath::new(&llm, &embedder, &store, &ledger);
        write.progress = true;
        let stats = write
            .insert(&scope, &turns)
            .await
            .with_context(|| format!("ingest {}", conv.sample_id))?;

        eprintln!(
            "  {:<8} turns={:<5} episodes={:<4} candidates={:<5} add={:<5} upd={:<4} dup={:<5} quar={:<4} rej={:<4} {:.1}s",
            conv.sample_id,
            stats.turns,
            stats.episodes,
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
pub async fn build_lmev2(
    trajectories: &Path,
    haystack_path: &Path,
    questions_path: &Path,
    tier: &str,
    collection: &str,
    ledger_path: &Path,
    limit: Option<usize>,
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
    Ok(report)
}

impl BuildReport {
    pub fn print(&self, units: usize) {
        let t = &self.total;
        println!("\n=== write path, {units} units ===");
        println!("turns              {}", t.turns);
        println!("episodes stored    {}", t.episodes);
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
        println!("records/unit       {:.2}", t.records_per_unit(units));
        println!("wall               {:.1}s", self.wall_secs);
        println!(
            "  extract          {:.1}s
  consolidate      {:.1}s
  index            {:.1}s",
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
        let parsed = parse_locomo_time("1:56 pm on 8 May, 2023").expect("should parse");
        assert_eq!(parsed.format("%Y-%m-%d %H:%M").to_string(), "2023-05-08 13:56");

        let midnight = parse_locomo_time("7 May, 2023").expect("date-only should parse");
        assert_eq!(midnight.format("%Y-%m-%d").to_string(), "2023-05-07");
    }

    /// An unparseable timestamp must yield `None`, not a fabricated date: a
    /// wrong `t_valid` is worse than a missing one because it silently
    /// reorders the bi-temporal history.
    #[test]
    fn an_unparseable_timestamp_is_none() {
        assert!(parse_locomo_time("").is_none());
        assert!(parse_locomo_time("sometime last spring").is_none());
    }
}
