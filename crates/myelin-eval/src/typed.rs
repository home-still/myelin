//! M89b — typed memory for LoCoMo, as two passes over a built store
//! (`docs/measurements/m89-typed-memory.md`).
//!
//! **Extract** (`typed-extract`) reads every dataset episode of a LoCoMo store
//! and asks the reader for its typed units ([`myelin_core::pipeline::typed`]).
//! It touches no store. The result is a JSONL cache keyed by the SHA-256 of
//! the prompt version, the episode's speakers and its text, so the pass is
//! resumable, shardable (`--shard i/n`) and reusable across stores built from
//! the same episodes. This is the events pass's pattern (M50, `events.rs`).
//!
//! **Build** (`typed-build`) turns each episode's cached extraction into
//! Profile, Event and Gist records and writes them beside the episodes. It
//! refuses to start while any episode is uncached, so nothing is written
//! partial. It also reports the stage's reader-free gate:
//! - `reach_all`: the share of answerable questions whose every gold turn
//!   lies in some stored unit's span;
//! - the ignore rate;
//! - gist grounding;
//! - how often an event's `when` was copied verbatim.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write as _};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::Utc;
use futures_util::StreamExt;
use myelin_core::config::MyelinConfig;
use myelin_core::embed::remote::RemoteEmbedder;
use myelin_core::llm::openai::OpenAiLlm;
use myelin_core::model::delta::Delta;
use myelin_core::model::record::{ActorId, MemoryRecord, RecordKind};
use myelin_core::pipeline::index::Indexer;
use myelin_core::pipeline::typed::{episode_turns, extract as extract_episode, route, typed_records, Extraction, Route, TYPED_PROMPT_VERSION};
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::QdrantStore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ablate::{conversation_lines, episode_turn_ids};
use crate::datasets::locomo;
use crate::events::parse_shard;

/// Progress is printed every this many episodes.
const PROGRESS_EVERY: usize = 50;
/// LoCoMo's adversarial category, which has no gold turns to reach.
const ADVERSARIAL_CATEGORY: u8 = 5;
/// A word the gist audit checks: a name (capitalised) or a number.
const GROUNDING_MIN_CHARS: usize = 2;

/// One line of the extraction cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLine {
    pub sha256: String,
    pub version: String,
    pub model: String,
    pub extraction: Extraction,
}

/// The cache key of an episode: the prompt version, its speakers and its
/// text. The same episode in another store has the same key.
pub fn episode_sha(episode: &MemoryRecord) -> String {
    let mut h = Sha256::new();
    h.update(TYPED_PROMPT_VERSION.as_bytes());
    for e in &episode.entities {
        h.update(b"\x1f");
        h.update(e.phrase.as_bytes());
    }
    h.update(b"\x1e");
    h.update(episode.text.as_bytes());
    format!("{:x}", h.finalize())
}

/// Every cache file, as one map. A line from another prompt version is
/// refused; so is a key extracted twice with different results.
pub fn load_cache(paths: &[String]) -> Result<HashMap<String, Extraction>> {
    let mut map: HashMap<String, Extraction> = HashMap::new();
    for p in paths {
        let file = std::fs::File::open(p).with_context(|| format!("open {p}"))?;
        for (n, line) in std::io::BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let c: CacheLine =
                serde_json::from_str(&line).with_context(|| format!("{p}:{}: not a cache line", n + 1))?;
            anyhow::ensure!(
                c.version == TYPED_PROMPT_VERSION,
                "{p}:{}: extracted under prompt version {:?}, not {TYPED_PROMPT_VERSION:?}",
                n + 1,
                c.version
            );
            match map.get(&c.sha256) {
                Some(prev) if prev != &c.extraction => {
                    anyhow::bail!("{p}:{}: episode {} extracted twice with different units", n + 1, c.sha256)
                }
                _ => {
                    map.insert(c.sha256, c.extraction);
                }
            }
        }
    }
    Ok(map)
}

/// The store's dataset episodes: episodic records of LoCoMo tenants whose
/// first turn is a dataset turn. A record written through the API is left
/// out, as `ablate`'s coverage leaves it out.
async fn dataset_episodes(ledger: &Ledger, lines: &HashMap<String, Vec<(String, String)>>) -> Result<Vec<MemoryRecord>> {
    let mut out = Vec::new();
    for record in ledger
        .records_in_namespace("locomo")
        .await
        .context("load episodes")?
        .into_iter()
        .filter(|r| r.kind == RecordKind::Episodic)
    {
        let Some(turns) = lines.get(&record.scope.tenant) else {
            continue;
        };
        if episode_turn_ids(turns, &record)?.is_some() {
            out.push(record);
        }
    }
    out.sort_by_key(|r| r.id);
    Ok(out)
}

#[derive(Debug, Default)]
pub struct ExtractReport {
    pub episodes: usize,
    pub in_shard: usize,
    pub cached: usize,
    pub extracted: usize,
    pub failed: usize,
    pub wall_secs: f64,
}

/// Extract every dataset episode of `ledger_path` (this shard's share) that
/// `out` does not already hold, `concurrency` calls in flight, one line
/// appended and flushed per episode.
pub async fn extract(dataset: &Path, ledger_path: &Path, out: &Path, shard: &str, concurrency: usize) -> Result<ExtractReport> {
    let shard = parse_shard(shard)?;
    let cfg = MyelinConfig::load().context("load myelin config")?;
    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("reader client")?;
    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;
    let lines = conversation_lines(&locomo::load(dataset)?);
    let episodes = dataset_episodes(&ledger, &lines).await?;
    let mut report = ExtractReport { episodes: episodes.len(), ..Default::default() };
    let mine: Vec<&MemoryRecord> = episodes
        .iter()
        .enumerate()
        .filter(|(i, _)| i % shard.1 == shard.0)
        .map(|(_, e)| e)
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
    let todo: Vec<&MemoryRecord> = mine.into_iter().filter(|e| !done.contains(&episode_sha(e))).collect();
    report.cached = report.in_shard - todo.len();
    eprintln!(
        "  typed-extract: {} episodes, {} in shard {}/{}, {} cached, {} to extract",
        report.episodes, report.in_shard, shard.0, shard.1, report.cached, todo.len()
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
        .map(|episode| async move { (episode, extract_episode(llm_ref, episode).await) })
        .buffer_unordered(concurrency.max(1));
    let mut n = 0usize;
    while let Some((episode, result)) = stream.next().await {
        n += 1;
        match result {
            Ok(extraction) => {
                report.extracted += 1;
                let line = CacheLine {
                    sha256: episode_sha(episode),
                    version: TYPED_PROMPT_VERSION.to_string(),
                    model: cfg.llm.model.clone(),
                    extraction,
                };
                writeln!(file, "{}", serde_json::to_string(&line)?)?;
                file.flush()?;
            }
            Err(e) => {
                report.failed += 1;
                eprintln!("  {} {}: extraction failed, left out of the cache ({e})", episode.scope.tenant, episode.id);
            }
        }
        if n.is_multiple_of(PROGRESS_EVERY) || n == total {
            eprintln!(
                "  [{n:>5}/{total}] failed={} {:.1}s/episode",
                report.failed,
                started.elapsed().as_secs_f64() / n as f64
            );
        }
    }
    report.wall_secs = started.elapsed().as_secs_f64();
    Ok(report)
}

/// The stage's reader-free gate, from the episodes and their extractions.
#[derive(Debug, Default, Serialize)]
pub struct ReachReport {
    pub episodes: usize,
    pub units: usize,
    pub ignored: usize,
    pub profile_units: usize,
    pub event_units: usize,
    pub record_units: usize,
    /// Answerable questions (categories 1-4) with gold turns.
    pub questions: usize,
    /// ... whose every gold turn lies in some stored (non-ignored) unit's span.
    pub reach_all: f64,
    /// Share of a gist's names and numbers found in its cited turns.
    pub gist_grounding: f64,
    /// Share of non-empty event `when`s found verbatim in the cited turns.
    pub when_verbatim: f64,
    /// Share of all dataset turns inside some stored unit's span.
    pub turns_stored: f64,
}

/// Names (capitalised words) and numbers of a text, the words a gist must not
/// invent.
fn checkable_words(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| w.chars().count() >= GROUNDING_MIN_CHARS)
        .filter(|w| w.chars().next().is_some_and(|c| c.is_uppercase() || c.is_ascii_digit()))
        .map(str::to_lowercase)
        .collect()
}

/// `reach_all` and the audits, computed from the dataset, the store's
/// episodes and the cache alone. No model and no GPU.
pub fn reach(
    conversations: &[locomo::LocomoConversation],
    episodes: &[MemoryRecord],
    cache: &HashMap<String, Extraction>,
) -> Result<ReachReport> {
    let lines = conversation_lines(conversations);
    let mut report = ReachReport { episodes: episodes.len(), ..Default::default() };
    let mut stored: HashSet<(String, String)> = HashSet::new();
    let (mut gist_words, mut gist_found, mut whens, mut whens_verbatim) = (0usize, 0usize, 0usize, 0usize);
    for episode in episodes {
        let Some(turns) = lines.get(&episode.scope.tenant) else {
            continue;
        };
        let Some(ids) = episode_turn_ids(turns, episode)? else {
            continue;
        };
        let extraction = cache
            .get(&episode_sha(episode))
            .with_context(|| format!("episode {} has no cached extraction", episode.id))?;
        let texts = episode_turns(episode);
        let (mut next_event, mut next_gist) = (0usize, 0usize);
        for unit in &extraction.units {
            report.units += 1;
            let r = route(unit);
            let cited: String = unit
                .cites
                .iter()
                .filter_map(|c| texts.get(*c).map(|t| t.1.to_lowercase()))
                .collect::<Vec<_>>()
                .join("\n");
            match r {
                Route::Ignore => {
                    report.ignored += 1;
                    continue;
                }
                Route::Profile => report.profile_units += 1,
                Route::Event => {
                    report.event_units += 1;
                    if let Some(e) = extraction.events.get(next_event) {
                        let when = e.when.trim().to_lowercase();
                        if !when.is_empty() {
                            whens += 1;
                            whens_verbatim += usize::from(cited.contains(&when));
                        }
                    }
                    next_event += 1;
                }
                Route::Record => {
                    report.record_units += 1;
                    if let Some(g) = extraction.gists.get(next_gist) {
                        for w in checkable_words(g) {
                            gist_words += 1;
                            gist_found += usize::from(cited.contains(&w));
                        }
                    }
                    next_gist += 1;
                }
            }
            let (Some(&lo), Some(&hi)) = (unit.cites.iter().min(), unit.cites.iter().max()) else {
                continue;
            };
            for id in ids.iter().take(hi + 1).skip(lo) {
                stored.insert((episode.scope.tenant.clone(), id.clone()));
            }
        }
    }
    let all_turns: usize = lines.values().map(Vec::len).sum();
    report.turns_stored = stored.len() as f64 / all_turns.max(1) as f64;
    let (mut reached, mut questions) = (0usize, 0usize);
    for conv in conversations {
        let tenant = format!("locomo/{}", conv.sample_id);
        for qa in conv.qa.iter().filter(|q| q.category != ADVERSARIAL_CATEGORY) {
            let gold: Vec<&String> = qa.evidence.iter().collect();
            if gold.is_empty() {
                continue;
            }
            questions += 1;
            // As `ablate` takes them: a malformed id is unreachable in both.
            reached += usize::from(gold.iter().all(|g| stored.contains(&(tenant.clone(), (*g).clone()))));
        }
    }
    report.questions = questions;
    report.reach_all = reached as f64 / questions.max(1) as f64;
    report.gist_grounding = gist_found as f64 / gist_words.max(1) as f64;
    report.when_verbatim = whens_verbatim as f64 / whens.max(1) as f64;
    Ok(report)
}

#[derive(Debug, Default)]
pub struct BuildReport {
    pub reach: ReachReport,
    pub written: HashMap<&'static str, usize>,
    pub existing: usize,
    pub wall_secs: f64,
}

/// Write every dataset episode's typed records into `collection`/`ledger`,
/// beside the episodes. Refuses while any episode is uncached.
pub async fn build(dataset: &Path, ledger_path: &Path, collection: &str, cache: &HashMap<String, Extraction>) -> Result<BuildReport> {
    let conversations = locomo::load(dataset)?;
    let lines = conversation_lines(&conversations);
    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;
    let episodes = dataset_episodes(&ledger, &lines).await?;
    anyhow::ensure!(!episodes.is_empty(), "{} holds no LoCoMo episodes to type", ledger_path.display());
    let missing = episodes.iter().filter(|e| !cache.contains_key(&episode_sha(e))).count();
    anyhow::ensure!(
        missing == 0,
        "{missing} of {} episodes have no cached extraction; run typed-extract until it exits \
         cleanly, then build. Nothing was written.",
        episodes.len()
    );
    let reach = reach(&conversations, &episodes, cache)?;

    let cfg = MyelinConfig::load().context("load myelin config")?;
    let embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim).context("embedder client")?;
    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    let indexer = Indexer::new(&embedder, &store, &ledger);
    let actor = ActorId::new("myelin-eval");
    let started = Instant::now();
    let mut report = BuildReport { reach, ..Default::default() };
    for (i, episode) in episodes.iter().enumerate() {
        let records = typed_records(episode, &cache[&episode_sha(episode)], Utc::now())
            .with_context(|| format!("typed records for episode {}", episode.id))?;
        let mut fresh = Vec::new();
        for record in records {
            if ledger.get(record.id).await?.is_some() {
                report.existing += 1;
                continue;
            }
            ledger
                .apply(&Delta::Add { record: Box::new(record.clone()) }, &actor)
                .await
                .with_context(|| format!("add typed record for episode {}", episode.id))?;
            *report.written.entry(record.kind.as_str()).or_default() += 1;
            fresh.push(record);
        }
        if !fresh.is_empty() {
            indexer.index(&fresh).await.with_context(|| format!("index typed records of {}", episode.id))?;
        }
        if (i + 1).is_multiple_of(PROGRESS_EVERY) || i + 1 == episodes.len() {
            eprintln!("  [{:>5}/{}] written={:?} existing={}", i + 1, episodes.len(), report.written, report.existing);
        }
    }
    report.wall_secs = started.elapsed().as_secs_f64();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_numbers_are_what_a_gist_must_not_invent() {
        assert_eq!(
            checkable_words("Caroline's counsellor recommended 3 books: Becoming Nicole."),
            vec!["caroline's", "becoming", "nicole"]
        );
    }
}
