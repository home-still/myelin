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
use myelin_core::model::record::{Scope, SourceRef};
use myelin_core::pipeline::ingest::Turn;
use myelin_core::pipeline::write::{WritePath, WriteStats};
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::QdrantStore;

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

    for conv in &conversations {
        let scope = Scope::new(format!("locomo/{}", conv.sample_id), "myelin", "locomo");
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

        report.total.merge(&stats);
        report.per_unit.push((conv.sample_id.clone(), stats));
    }

    report.wall_secs = started.elapsed().as_secs_f64();
    Ok(report)
}

impl BuildReport {
    pub fn print(&self, units: usize) {
        let t = &self.total;
        println!("\n=== write path, {units} units ===");
        println!("turns              {}", t.turns);
        println!("episodes           {}", t.episodes);
        println!("candidates         {}", t.candidates);
        println!("added              {}", t.added);
        println!("updated            {}", t.updated);
        println!("deleted            {}", t.deleted);
        println!("noop               {}", t.noop);
        println!("duplicates         {}", t.duplicates);
        println!("quarantined        {}", t.quarantined);
        println!("rejected           {}", t.rejected);
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
