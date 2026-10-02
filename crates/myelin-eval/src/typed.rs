//! M89b — typed memory for LoCoMo, as two passes over a built store
//! (`docs/measurements/m89-typed-memory.md`).
//!
//! **Extract** (`typed-extract`) reads every dataset episode of a LoCoMo store
//! and asks the reader for its typed units ([`myelin_core::pipeline::typed`]).
//! It touches no store. The result is a JSONL cache keyed by the SHA-256 of
//! the prompt version, the episode's speakers and its text, so the pass is
//! resumable, shardable (`--shard i/n`) and reusable across stores built from
//! the same episodes. One cache holds one reader's extractions. This is the
//! events pass's pattern (M50, `events.rs`).
//!
//! **Build** (`typed-build`) turns each episode's cached extraction into
//! Profile, Event and Gist records and writes them beside the episodes. It
//! refuses before writing anything while any episode is uncached, while any
//! cached extraction is one [`typed_records`] refuses, while the store holds
//! typed records of another extraction, or while the collection is missing.
//! It indexes every typed record, so a run stopped between the ledger and the
//! index is healed by the next one, and it ends on a reconcile of the ledger
//! against the collection. It also reports the stage's reader-free gate:
//! - `reach_all`: the share of answerable questions whose every gold turn
//!   lies in some written record's span;
//! - the ignore rate;
//! - gist grounding;
//! - how often an event's `when` was copied verbatim.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Read as _, Seek as _, SeekFrom, Write as _};
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
use myelin_core::store::reconcile::reconcile;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::ablate::{conversation_lines, episode_turn_ids};
use crate::datasets::locomo;
use crate::events::parse_shard;

/// Progress is printed every this many episodes.
const PROGRESS_EVERY: usize = 50;
/// The namespace a LoCoMo store files its records under.
const NAMESPACE: &str = "locomo";
/// LoCoMo's adversarial category: its questions have no answer in the
/// conversation, so `reach_all` counts the answerable categories 1-4.
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

/// Every cache file as one map, with the reader that extracted it.
#[derive(Debug, Default)]
pub struct TypedCache {
    /// The reader every line was extracted by; `None` only for an empty cache.
    pub model: Option<String>,
    pub extractions: HashMap<String, Extraction>,
}

/// Every cache file, as one map. A line from another prompt version is
/// refused; so is a line from another reader than the lines before it (the
/// key does not name the reader, so a mixed cache would build one store out
/// of two readers' units), and a key extracted twice with different results.
pub fn load_cache(paths: &[String]) -> Result<TypedCache> {
    let mut cache = TypedCache::default();
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
            match &cache.model {
                Some(model) => anyhow::ensure!(
                    *model == c.model,
                    "{p}:{}: extracted by {:?}, but the lines before it by {model:?}; one cache \
                     holds one reader's extractions",
                    n + 1,
                    c.model
                ),
                None => cache.model = Some(c.model.clone()),
            }
            match cache.extractions.get(&c.sha256) {
                Some(prev) if prev != &c.extraction => {
                    anyhow::bail!("{p}:{}: episode {} extracted twice with different units", n + 1, c.sha256)
                }
                _ => {
                    cache.extractions.insert(c.sha256, c.extraction);
                }
            }
        }
    }
    Ok(cache)
}

/// Refuse to append to a cache whose last line is unterminated. A run killed
/// mid-line leaves one; the next line appended would be glued onto it and
/// make the whole cache unloadable.
fn ensure_terminated(path: &Path) -> Result<()> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    if file.metadata()?.len() == 0 {
        return Ok(());
    }
    file.seek(SeekFrom::End(-1))?;
    let mut last = [0u8; 1];
    file.read_exact(&mut last)?;
    anyhow::ensure!(
        last[0] == b'\n',
        "{}: the last line is unterminated (a run stopped mid-write); remove it, then resume",
        path.display()
    );
    Ok(())
}

/// One cache line, newline included, in a single `write_all`. `writeln!` on a
/// `File` writes the text and the newline separately, and a kill between the
/// two leaves a line the next append is glued onto.
fn append_line(file: &mut std::fs::File, line: &CacheLine) -> Result<()> {
    let mut text = serde_json::to_string(line)?;
    text.push('\n');
    file.write_all(text.as_bytes())?;
    Ok(())
}

/// The store's dataset episodes among `records`: live episodic records of
/// LoCoMo tenants whose first turn is a dataset turn. A record written
/// through the API is left out, as `ablate`'s coverage leaves it out. So is
/// a forgotten (invalidated) episode: its typed records would be written
/// live and restate what was forgotten.
fn dataset_episodes(records: &[MemoryRecord], lines: &HashMap<String, Vec<(String, String)>>) -> Result<Vec<MemoryRecord>> {
    let now = Utc::now();
    let mut out = Vec::new();
    for record in records.iter().filter(|r| r.kind == RecordKind::Episodic && r.is_admissible_at(now)) {
        let Some(turns) = lines.get(&record.scope.tenant) else {
            continue;
        };
        if episode_turn_ids(turns, record)?.is_some() {
            out.push(record.clone());
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
/// appended per episode.
pub async fn extract(dataset: &Path, ledger_path: &Path, out: &Path, shard: &str, concurrency: usize) -> Result<ExtractReport> {
    let shard = parse_shard(shard)?;
    // `Ledger::open` creates a missing file: a misspelled path would type
    // nothing and exit as if it were done.
    anyhow::ensure!(
        ledger_path.exists(),
        "missing {}; typed-extract reads the episodes of a built store",
        ledger_path.display()
    );
    let cfg = MyelinConfig::load().context("load myelin config")?;
    let llm = OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model).context("reader client")?;
    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;
    let lines = conversation_lines(&locomo::load(dataset)?);
    let records = ledger.records_in_namespace(NAMESPACE).await.context("load episodes")?;
    let episodes = dataset_episodes(&records, &lines)?;
    anyhow::ensure!(!episodes.is_empty(), "{} holds no LoCoMo episodes to type", ledger_path.display());
    let mut report = ExtractReport { episodes: episodes.len(), ..Default::default() };
    let mine: Vec<&MemoryRecord> = episodes
        .iter()
        .enumerate()
        .filter(|(i, _)| i % shard.1 == shard.0)
        .map(|(_, e)| e)
        .collect();
    report.in_shard = mine.len();
    let done: HashSet<String> = if out.exists() {
        ensure_terminated(out)?;
        let cache = load_cache(&[out.display().to_string()])?;
        if let Some(model) = &cache.model {
            anyhow::ensure!(
                *model == cfg.llm.model,
                "{} holds extractions by {model:?}, not the configured reader {:?}; extract into a \
                 new --out, or point the config back at {model:?}",
                out.display(),
                cfg.llm.model
            );
        }
        cache.extractions.into_keys().collect()
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
                append_line(&mut file, &line)?;
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

/// The stage's reader-free gate, from the episodes, their extractions and
/// the records those extractions write.
#[derive(Debug, Default, Serialize)]
pub struct ReachReport {
    /// The reader the extractions came from.
    pub model: Option<String>,
    pub episodes: usize,
    pub units: usize,
    pub ignored: usize,
    pub profile_units: usize,
    pub event_units: usize,
    pub record_units: usize,
    /// Answerable questions (categories 1-4) with gold turns.
    pub questions: usize,
    /// ... whose every gold turn lies in some written record's span.
    pub reach_all: f64,
    /// Share of a gist's names and numbers found in its cited turns.
    pub gist_grounding: f64,
    /// Share of non-empty event `when`s found verbatim in the cited turns.
    pub when_verbatim: f64,
    /// Share of all dataset turns inside some written record's span.
    pub turns_stored: f64,
}

/// A text's words for the grounding audit, whole: split on anything but
/// letters, digits and apostrophes, with surrounding apostrophes and a
/// possessive `'s` removed. "Caroline's" is the word "Caroline", and "she"
/// is never found inside "shelter".
fn grounding_words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .map(|w| {
            let w = w.trim_matches('\'');
            w.strip_suffix("'s").unwrap_or(w)
        })
        .filter(|w| !w.is_empty())
}

/// Names (capitalised words) and numbers of a text, the words a gist must not
/// invent.
fn checkable_words(text: &str) -> Vec<String> {
    grounding_words(text)
        .filter(|w| w.chars().count() >= GROUNDING_MIN_CHARS)
        .filter(|w| w.chars().next().is_some_and(|c| c.is_uppercase() || c.is_ascii_digit()))
        .map(str::to_lowercase)
        .collect()
}

/// How many of `gist`'s checkable words are words of the cited turns, of how
/// many it has.
fn gist_grounding(gist: &str, cited: &[&str]) -> (usize, usize) {
    let said: HashSet<String> = cited.iter().flat_map(|l| grounding_words(l)).map(str::to_lowercase).collect();
    let words = checkable_words(gist);
    (words.iter().filter(|w| said.contains(*w)).count(), words.len())
}

/// One dataset episode, its cached extraction, and the typed records that
/// extraction writes.
pub struct Typed<'a> {
    pub episode: &'a MemoryRecord,
    pub extraction: &'a Extraction,
    pub records: Vec<MemoryRecord>,
}

/// `reach_all` and the audits, from the dataset and the typed episodes alone.
/// No model and no GPU. A turn counts as stored only inside the span of a
/// record that is written, the span read as `ablate`'s coverage reads it.
pub fn reach(conversations: &[locomo::LocomoConversation], typed: &[Typed<'_>]) -> Result<ReachReport> {
    let lines = conversation_lines(conversations);
    let mut report = ReachReport { episodes: typed.len(), ..Default::default() };
    let mut stored: HashSet<(String, String)> = HashSet::new();
    let (mut gist_words, mut gist_found, mut whens, mut whens_verbatim) = (0usize, 0usize, 0usize, 0usize);
    for t in typed {
        let tenant = &t.episode.scope.tenant;
        let turns = lines
            .get(tenant)
            .with_context(|| format!("episode {} is in no conversation of the dataset", t.episode.id))?;
        let ids = episode_turn_ids(turns, t.episode)?
            .with_context(|| format!("episode {} does not start at a dataset turn", t.episode.id))?;
        let texts = episode_turns(t.episode);
        let (mut next_event, mut next_gist) = (0usize, 0usize);
        for unit in &t.extraction.units {
            report.units += 1;
            let cited: Vec<&str> = unit
                .cites
                .iter()
                .map(|c| {
                    texts.get(*c).map(|l| l.1.as_str()).with_context(|| {
                        format!("episode {} has no turn {c} for a unit to cite", t.episode.id)
                    })
                })
                .collect::<Result<_>>()?;
            match route(unit) {
                Route::Ignore => report.ignored += 1,
                Route::Profile => report.profile_units += 1,
                Route::Event => {
                    report.event_units += 1;
                    let e = t
                        .extraction
                        .events
                        .get(next_event)
                        .with_context(|| format!("episode {} is missing event {next_event}", t.episode.id))?;
                    next_event += 1;
                    let when = e.when.trim().to_lowercase();
                    if !when.is_empty() {
                        whens += 1;
                        whens_verbatim += usize::from(cited.join("\n").to_lowercase().contains(&when));
                    }
                }
                Route::Record => {
                    report.record_units += 1;
                    let g = t
                        .extraction
                        .gists
                        .get(next_gist)
                        .with_context(|| format!("episode {} is missing gist {next_gist}", t.episode.id))?;
                    next_gist += 1;
                    let (found, words) = gist_grounding(g, &cited);
                    gist_found += found;
                    gist_words += words;
                }
            }
        }
        for record in &t.records {
            let span = record
                .provenance
                .source
                .span
                .with_context(|| format!("typed record {} carries no span", record.id))?;
            let (lo, hi) = (span.start as usize, span.end as usize);
            anyhow::ensure!(
                lo <= hi && hi < ids.len(),
                "record {} cites turns {lo}..={hi} of an episode with {} turns",
                record.id,
                ids.len()
            );
            for id in &ids[lo..=hi] {
                stored.insert((tenant.clone(), id.clone()));
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

/// A typed record this pass writes: a Profile, Event or Gist, live, with a
/// span of one dataset episode's turns. A profile written any other way (the
/// profile pass, `remember`) carries no span.
fn is_typed_record(r: &MemoryRecord, episodes: &HashSet<Uuid>, now: chrono::DateTime<Utc>) -> bool {
    matches!(r.kind, RecordKind::Profile | RecordKind::Event | RecordKind::Gist)
        && r.provenance.source.span.is_some()
        && matches!(r.provenance.derived_from.as_slice(), [a] if episodes.contains(a))
        && r.is_admissible_at(now)
}

#[derive(Debug, Default)]
pub struct BuildReport {
    pub reach: ReachReport,
    pub written: HashMap<&'static str, usize>,
    pub existing: usize,
    pub wall_secs: f64,
}

/// Write every dataset episode's typed records into `collection`/`ledger`,
/// beside the episodes. Everything that can refuse does so before the first
/// write.
pub async fn build(dataset: &Path, ledger_path: &Path, collection: &str, cache: &TypedCache) -> Result<BuildReport> {
    // `Ledger::open` creates a missing file, and this pass writes beside a
    // built store, never a new one.
    anyhow::ensure!(
        ledger_path.exists(),
        "missing {}; typed-build writes beside a copy of a built store",
        ledger_path.display()
    );
    let conversations = locomo::load(dataset)?;
    let lines = conversation_lines(&conversations);
    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;
    let records = ledger.records_in_namespace(NAMESPACE).await.context("load records")?;
    let episodes = dataset_episodes(&records, &lines)?;
    anyhow::ensure!(!episodes.is_empty(), "{} holds no LoCoMo episodes to type", ledger_path.display());
    let missing = episodes.iter().filter(|e| !cache.extractions.contains_key(&episode_sha(e))).count();
    anyhow::ensure!(
        missing == 0,
        "{missing} of {} episodes have no cached extraction; run typed-extract until it exits \
         cleanly, then build. Nothing was written.",
        episodes.len()
    );
    // Every episode's records before anything is written, so an extraction
    // `typed_records` refuses stops the build here rather than after part
    // of the store is written.
    let now = Utc::now();
    let typed: Vec<Typed<'_>> = episodes
        .iter()
        .map(|episode| {
            let extraction = cache
                .extractions
                .get(&episode_sha(episode))
                .with_context(|| format!("episode {} has no cached extraction", episode.id))?;
            let records = typed_records(episode, extraction, now)
                .with_context(|| format!("typed records for episode {}", episode.id))?;
            Ok(Typed { episode, extraction, records })
        })
        .collect::<Result<_>>()?;
    // One generation of typed records per store. Record ids hash the
    // generated text, so another extraction (prompt version, reader, rerun)
    // writes new ids beside the old records, and both would be retrieved.
    let planned: HashSet<Uuid> = typed.iter().flat_map(|t| t.records.iter().map(|r| r.id)).collect();
    let episode_ids: HashSet<Uuid> = episodes.iter().map(|e| e.id).collect();
    let foreign = records
        .iter()
        .filter(|r| is_typed_record(r, &episode_ids, now) && !planned.contains(&r.id))
        .count();
    anyhow::ensure!(
        foreign == 0,
        "{} already holds {foreign} typed record(s) this cache does not write (another prompt \
         version, reader or extraction); type a fresh copy of the store. Nothing was written.",
        ledger_path.display()
    );
    let mut reach = reach(&conversations, &typed)?;
    reach.model = cache.model.clone();

    let cfg = MyelinConfig::load().context("load myelin config")?;
    let embedder = RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim).context("embedder client")?;
    let mut qdrant_cfg = cfg.qdrant.clone();
    qdrant_cfg.collection = collection.to_string();
    let store = QdrantStore::new(&qdrant_cfg).context("qdrant store")?;
    // Never `ensure_collection`: a misspelled name would be created empty,
    // and the typed records would be indexed beside no episodes.
    anyhow::ensure!(
        store.exists().await?,
        "collection {collection:?} does not exist; typed-build writes beside a copy of a built \
         store (ops/big/qdrant-copy.sh). Nothing was written."
    );
    let indexer = Indexer::new(&embedder, &store, &ledger);
    let actor = ActorId::new("myelin-eval");
    let started = Instant::now();
    let mut report = BuildReport { reach, ..Default::default() };
    let n_episodes = typed.len();
    let mut all: Vec<MemoryRecord> = Vec::new();
    for (i, t) in typed.into_iter().enumerate() {
        for record in t.records {
            if ledger.get(record.id).await?.is_some() {
                report.existing += 1;
            } else {
                ledger
                    .apply(&Delta::Add { record: Box::new(record.clone()) }, &actor)
                    .await
                    .with_context(|| format!("add typed record for episode {}", t.episode.id))?;
                *report.written.entry(record.kind.as_str()).or_default() += 1;
            }
            all.push(record);
        }
        if (i + 1).is_multiple_of(PROGRESS_EVERY) || i + 1 == n_episodes {
            eprintln!("  [{:>5}/{n_episodes}] written={:?} existing={}", i + 1, report.written, report.existing);
        }
    }
    // Every typed record, the ones an earlier run applied included: an upsert
    // is idempotent, and a run that stopped between the ledger and the index
    // is healed here instead of being skipped as already written.
    indexer.index(&all).await.context("index typed records")?;
    // Nothing is measured on a store whose ledger and collection disagree
    // (`build`'s `check_drift`).
    let drift = reconcile(&ledger, &store, NAMESPACE, None, false)
        .await
        .context("reconcile after typed-build")?;
    anyhow::ensure!(
        drift.is_clean(),
        "{} and {collection:?} disagree after typed-build: missing_vectors={} qdrant_orphans={} \
         payload_drift={} stale_points={} dangling_links={} orphan_incidence={} \
         missing_provenance={}; is the collection a copy of the same store?",
        ledger_path.display(),
        drift.missing_vectors.len(),
        drift.qdrant_orphans.len(),
        drift.payload_drift.len(),
        drift.stale_points.len(),
        drift.dangling_links.len(),
        drift.orphan_incidence.len(),
        drift.missing_provenance.len()
    );
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
            vec!["caroline", "becoming", "nicole"]
        );
    }

    /// Whole words: a faithful possessive is grounded by the speaker's own
    /// label, and an invented name or number is not found inside a longer
    /// word.
    #[test]
    fn gist_grounding_matches_whole_words() {
        let cited = ["Caroline: My counsellor recommended three books: Becoming Nicole, Gender Outlaw and Redefining Realness."];
        let faithful = "Caroline's counsellor recommended three books: Becoming Nicole, Gender Outlaw and Redefining Realness.";
        let (found, words) = gist_grounding(faithful, &cited);
        assert_eq!(found, words);
        let invented = gist_grounding("Jon sold 20 cars to Al.", &["Jonathan: I finally sold my car in 2023."]);
        assert_eq!(invented, (0, 3));
    }

    #[test]
    fn a_cache_holds_one_readers_extractions() {
        let dir = std::env::temp_dir().join(format!("myelin-typed-cache-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let line = |sha: &str, model: &str| CacheLine {
            sha256: sha.into(),
            version: TYPED_PROMPT_VERSION.into(),
            model: model.into(),
            extraction: Extraction { units: vec![], events: vec![], gists: vec![] },
        };
        let path = dir.join("c.jsonl");
        let mut file = std::fs::File::create(&path).unwrap();
        append_line(&mut file, &line("a", "qwen3.5-9b")).unwrap();
        append_line(&mut file, &line("b", "qwen3.5-9b")).unwrap();
        let one = load_cache(&[path.display().to_string()]).unwrap();
        assert_eq!(one.model.as_deref(), Some("qwen3.5-9b"));
        assert_eq!(one.extractions.len(), 2);
        ensure_terminated(&path).unwrap();
        append_line(&mut file, &line("c", "bonsai-2-27b")).unwrap();
        assert!(load_cache(&[path.display().to_string()]).is_err());
        // A torn last line is refused before anything is appended to it.
        let torn = dir.join("torn.jsonl");
        std::fs::write(&torn, "{\"sha256\":").unwrap();
        assert!(ensure_terminated(&torn).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
