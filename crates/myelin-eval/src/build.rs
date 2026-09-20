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
use crate::datasets::locomo::{self, LocomoConversation};
use crate::datasets::longmemeval;

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
            .filter(|s| {
                s.date_time
                    .as_deref()
                    .and_then(parse_session_time)
                    .is_none()
            })
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

/// M23 D1 — mint the typed pools (events, notes) for LME-V2 into the SAME
/// store the episodic build already wrote, so one collection carries all
/// three kinds and the read path stays R4-queryable.
///
/// AgentRunbook-R's 58.6 comes from three pools with per-pool queries
/// (`10.48550/arXiv.2605.12493` §4.1); this is the write-path half. The
/// read-path half is D2's typed probes in `investigate`.
///
/// **One batched call per trajectory per pool, not AgentRunbook-R's
/// per-transition pass.** Their event pass is one call per transition —
/// 28 states per trajectory is ~43k calls on the tier-small corpus. The
/// batched form is ~1,540 calls on each pool and the plan caps it at 6
/// events per trajectory. The note template is used verbatim; the event
/// template keeps its field spec and rules verbatim and generalises only
/// the "one target transition" delivery clause into "up to 6 transitions",
/// which is the deviation the batching *is*.
///
/// Resume is per trajectory, keyed on the ledger's `unit_complete` audit
/// event with the unit string as the key — `build_locomo`'s predicate, at
/// pool granularity. A trajectory whose pools already exist is skipped, so
/// an interrupted pass costs only the trajectories it had not reached.
///
/// Every record routes through [`WritePath::insert`] with
/// [`WritePath::record_kind`] set, so dedup, embedding, indexing and the
/// ledger see ordinary records: events are `RecordKind::Semantic` under
/// unit `{id}#events`, notes are `RecordKind::Procedural` under
/// `{id}#notes`. `t_valid` stays the ingest time — the corpus has no event
/// dates (the M22 finding), and the arms that read these records run
/// `--undated`.
// Same shape and same eight arguments as `build_lmev2`, which carries the
// same allow: the two are called from one `match` arm and a struct for one
// of them would make the pair harder to read, not easier.
#[allow(clippy::too_many_arguments)]
pub async fn build_lmev2_pools(
    trajectories: &Path,
    haystack_path: &Path,
    questions_path: &Path,
    tier: &str,
    collection: &str,
    ledger_path: &Path,
    limit: Option<usize>,
    repair: bool,
) -> Result<BuildReport> {
    use myelin_core::llm::{complete_json, CompletionRequest, Message};
    use myelin_core::model::record::RecordKind;

    // The template, verbatim from the vendored AgentRunbook-R
    // (`memory_modules/support.py`, NOTE_GENERATION_SYSTEM_PROMPT,
    // prompt version `qwen_v6_retrieval_safe`).
    const NOTE_PROMPT: &str = r#"You convert one UI task trajectory into two reusable memory notes for a future agent.

Assume these notes will later be retrieved for unknown future questions.
You do not know the downstream question in advance.
Your job is to preserve the workflow and the highest-value reusable facts from the touched pages.

Write:
1. procedure_note
2. hint_note

Each note must be an object with:
- title: a short retrieval-friendly title with app/module/task context
- description: 1 short sentence describing what the note is about
- content: a bullet list string using '- ' lines

Rules:
- Use only evidence grounded in the provided goal, outcome, thoughts, annotated actions, and screenshots.
- Never write a fact unless it is directly supported by the observed run.
- Mention application / page / module names when they are visible from the actions or screenshots.
- Do not invent unseen fields, filters, modules, or outcomes.
- Prefer exact literal UI strings over paraphrases whenever a label, tab, button, menu item, module, or option is visible.
- If the run failed, procedure_note may describe the intended or attempted workflow only where the evidence supports it. Do not pretend the task succeeded.
- For failed runs, use hint_note to explain what may trip an agent up or what signal in the UI matters.
- Do not mention screenshot numbers, state numbers, or the word trajectory.
- Do not copy the internal thoughts verbatim line-by-line. Distill them into useful notes.
- procedure_note should capture the reliable core workflow only.
- hint_note should preserve only durable, high-value facts from the touched pages.
- Prefer high-signal facts that are likely to help later retrieval.
- Keep procedure_note.content to 4 to 8 bullets.
- Keep hint_note.content to 6 to 12 bullets when the evidence supports it.
- Do not output analysis, reasoning, headings, markdown fences, or any text before or after the JSON object.
- Start your answer with { and end your answer with }.

Return only valid JSON in this shape:
{"procedure_note":{"title":"...","description":"...","content":"- ...\n- ..."},"hint_note":{"title":"...","description":"...","content":"- ...\n- ..."}}"#;

    // AgentRunbook-R's event template
    // (`memory_modules/agentrunbook_r.py`, EVENT_GENERATION_SYSTEM_PROMPT),
    // with its single-transition delivery clause generalised to a capped
    // list — the only textual deviation, and the one the batching makes.
    const EVENT_PROMPT: &str = r#"You convert one UI transition from a longer task trajectory into retrieval-ready event text.

You will be given:
- the full task goal and outcome
- the full annotated action trace for the trajectory

In this dataset, actions are attached to destination states:
- transition event_0000 means state 0 -> action stored on state 1 -> state 1
- transition event_0001 means state 1 -> action stored on state 2 -> state 2

Return exactly one JSON object with this shape:
{"events":[{"overview":"...","state_transition":"..."}]}

Emit at most 6 events: pick the transitions that changed what the page could do or show — navigation into a new module, a revealed or replaced panel, a form submitted, a value or status changed, a confirmation or blocker appeared. Return an empty list if the trajectory contains no such transition.

Field requirements:
- "overview": one concise paragraph that briefly recaps the concrete task goal and places this transition in the broader workflow. Do not just say "while pursuing the goal". Mention what the agent is trying to accomplish and what stage this step represents.
- "state_transition": one concise paragraph that explicitly compares the post-state to the pre-state. Describe what happened after the action: a new page, new module, revealed panel, form fields, changed values, confirmation signal, blocker, popup, navigation, or lack of visible change.

Rules:
- Ground both fields only in the provided goal, outcome, action trace, and thoughts.
- Be retrieval-friendly for unknown future dynamic questions.
- Reuse the exact task entities and labels when they are present in the evidence. Do not rename users, products, themes, modules, or records.
- Preserve the most answer-bearing visible facts:
  - exact module, page, tab, menu, button, field, option, status, stage, entity, or label names
  - values, counts, dates, before/after states, selected options, confirmation signals, blocking signals, or newly revealed UI
  - distinctions between similar controls when the evidence supports them
- Use the annotated action text when naming what the agent clicked, typed, selected, or opened.
- In "state_transition", prioritize what changed because of the action: newly visible or replaced pages, menus, panels, dialogs, fields, values, warnings, or blockers.
- Avoid spending space on unchanged background widgets or unrelated page content unless they are needed to identify the current view.
- Do not invent unseen controls, labels, outcomes, or causal claims.
- Do not quote raw thoughts verbatim line-by-line; distill them.
- Do not output markdown fences, commentary, or extra keys.
- First character must be { and last character must be }.
"#;

    #[derive(Debug, serde::Deserialize)]
    struct Note {
        title: String,
        description: String,
        content: String,
    }
    #[derive(Debug, serde::Deserialize)]
    struct NoteSet {
        procedure_note: Note,
        hint_note: Note,
    }
    #[derive(Debug, serde::Deserialize)]
    struct Event {
        overview: String,
        state_transition: String,
    }
    #[derive(Debug, serde::Deserialize)]
    struct EventList {
        events: Vec<Event>,
    }

    fn note_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["procedure_note", "hint_note"],
            "properties": {
                "procedure_note": note(),
                "hint_note": note(),
            }
        })
    }
    fn note() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["title", "description", "content"],
            "properties": {
                "title": {"type": "string", "maxLength": 200},
                "description": {"type": "string", "maxLength": 400},
                "content": {"type": "string", "maxLength": 2000},
            }
        })
    }
    fn event_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["events"],
            "properties": {
                "events": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["overview", "state_transition"],
                        "properties": {
                            "overview": {"type": "string", "maxLength": 1000},
                            "state_transition": {"type": "string", "maxLength": 1000},
                        }
                    }
                }
            }
        })
    }

    /// The trajectory's goal/outcome/ordered steps, the evidence both
    /// templates name. Screenshots are not attached: the write path is
    /// text-only, and the tree chunks carrying the page state are already
    /// in the episodic store.
    fn trace_text(traj: &lmev2::Trajectory) -> String {
        let mut lines = vec![
            format!("Goal: {}", traj.goal),
            format!("Environment: {} ({})", traj.environment, traj.domain),
            format!("Start URL: {}", traj.start_url),
            format!("Outcome: {}", traj.outcome),
            String::from("Annotated action trace:"),
        ];
        for s in &traj.states {
            let action = s.action.as_deref().unwrap_or("(initial state)");
            let thought = s.thought.as_deref().unwrap_or("").trim();
            if !thought.is_empty() || s.action.is_some() {
                lines.push(format!("[{}] {action}", s.step));
                if !thought.is_empty() {
                    lines.push(format!("    thought: {thought}"));
                }
            }
        }
        lines.join("\n")
    }

    fn render_note(n: &Note) -> String {
        format!("{}\n{}\n{}", n.title, n.description, n.content)
    }

    let cfg = MyelinConfig::load().context("load myelin config")?;
    let questions = lmev2::load_questions(questions_path)?;
    let haystack = lmev2::load_haystack(haystack_path)?;
    let by_domain = lmev2::haystacks_by_domain(&haystack, &questions)?;

    let mut domain_of: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut wanted: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (domain, ids) in &by_domain {
        for id in ids.iter().take(limit.unwrap_or(usize::MAX)) {
            domain_of.insert(id.clone(), domain.clone());
            wanted.insert(id.clone());
        }
        eprintln!(
            "  pools {tier}/{domain}: {} trajectories",
            limit.map_or(ids.len(), |n| n.min(ids.len()))
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
    let mut trajectories_without_events = 0usize;
    let mut trajectories_without_notes = 0usize;
    let mut units_done = 0usize;

    // Same streaming reader `build_lmev2` uses: depth-2 channel, blocking
    // task, 1.2 GB file never fully resident.
    let want_count = wanted.len();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<lmev2::Trajectory>(2);
    let traj_path = trajectories.to_path_buf();
    let reader = tokio::task::spawn_blocking(move || {
        lmev2::for_each_trajectory(&traj_path, &wanted, |t| {
            tx.blocking_send(t)
                .map_err(|_| anyhow::anyhow!("pool ingest stopped before the file was consumed"))
        })
    });

    while let Some(traj) = rx.recv().await {
        let domain = &domain_of[&traj.id];
        let scope = Scope::new(format!("{tier}/{domain}"), "myelin", tier);

        // Resume: both pools complete means nothing to do. One pool missing
        // (a crash between the two inserts) redoes the whole trajectory —
        // the id scheme makes the surviving records dedup to no-ops.
        let events_done = ledger
            .unit_is_complete(&format!("{}#events", traj.id))
            .await?;
        let notes_done = ledger
            .unit_is_complete(&format!("{}#notes", traj.id))
            .await?;
        if events_done && notes_done {
            continue;
        }

        let trace = trace_text(&traj);
        let mut traj_events = 0usize;
        let mut traj_notes = 0usize;

        // ── events: RecordKind::Semantic under {id}#events ──
        if !events_done {
            let request = CompletionRequest::new(vec![
                Message::system(EVENT_PROMPT),
                Message::user(trace.clone()),
            ])
            .with_schema(event_schema())
            .with_max_tokens(2048);
            let parsed = complete_json::<EventList>(&llm, &request).await;
            match parsed {
                Ok(list) => {
                    let events: Vec<Event> = list.events.into_iter().take(6).collect();
                    let turns: Vec<Turn> = events
                        .iter()
                        .map(|e| Turn {
                            speaker: "event".into(),
                            text: format!("{}\n{}", e.overview.trim(), e.state_transition.trim()),
                            at: None,
                            source: SourceRef::doc(format!("{}:events", traj.id)),
                            unit: format!("{}#events", traj.id),
                        })
                        .collect();
                    let mut write = WritePath::new(&llm, &embedder, &store, &ledger);
                    write.record_kind = RecordKind::Semantic;
                    write.extract_facts = false;
                    let stats = write.insert(&scope, &turns).await?;
                    traj_events = turns.len();
                    report.total.merge(&stats);
                }
                Err(e) => {
                    eprintln!(
                        "  {} events: extraction failed ({e}); skipping pool",
                        traj.id
                    );
                    trajectories_without_events += 1;
                }
            }
        }

        // ── notes: RecordKind::Procedural under {id}#notes ──
        if !notes_done {
            let request = CompletionRequest::new(vec![
                Message::system(NOTE_PROMPT),
                Message::user(trace.clone()),
            ])
            .with_schema(note_schema())
            .with_max_tokens(2048);
            let parsed = complete_json::<NoteSet>(&llm, &request).await;
            match parsed {
                Ok(set) => {
                    let notes = [set.procedure_note, set.hint_note];
                    let turns: Vec<Turn> = notes
                        .iter()
                        .map(|n| Turn {
                            speaker: "note".into(),
                            text: render_note(n),
                            at: None,
                            source: SourceRef::doc(format!("{}:notes", traj.id)),
                            unit: format!("{}#notes", traj.id),
                        })
                        .collect();
                    let mut write = WritePath::new(&llm, &embedder, &store, &ledger);
                    write.record_kind = RecordKind::Procedural;
                    write.extract_facts = false;
                    let stats = write.insert(&scope, &turns).await?;
                    traj_notes = turns.len();
                    report.total.merge(&stats);
                }
                Err(e) => {
                    eprintln!(
                        "  {} notes: extraction failed ({e}); skipping pool",
                        traj.id
                    );
                    trajectories_without_notes += 1;
                }
            }
        }

        eprintln!(
            "  {:<10} {domain:<10} events={traj_events:<3} notes={traj_notes:<3} {:.1}s",
            traj.id,
            started.elapsed().as_secs_f64(),
        );

        for pool in ["#events", "#notes"] {
            if (pool == "#events" && traj_events > 0) || (pool == "#notes" && traj_notes > 0) {
                ledger
                    .mark_unit_complete(
                        &format!("{}{pool}", traj.id),
                        &ActorId::new("myelin-eval"),
                        serde_json::json!({
                            "trajectory": traj.id,
                            "pool": pool.trim_start_matches('#'),
                            "records": if pool == "#events" { traj_events } else { traj_notes },
                        }),
                    )
                    .await?;
            }
        }
        units_done += 1;
    }

    let found = reader.await.context("trajectory reader task")??;
    if found != want_count {
        anyhow::bail!("haystack names {want_count} trajectories; read {found}");
    }

    let n = units_done;
    if n > 0 {
        let event_rate = 1.0 - trajectories_without_events as f64 / n as f64;
        let note_rate = 1.0 - trajectories_without_notes as f64 / n as f64;
        eprintln!(
            "  pool extraction coverage: events {:.1}%  notes {:.1}% (below 95% means the prompt is broken)",
            event_rate * 100.0,
            note_rate * 100.0,
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
///
/// The **profile** pass is on, and it is affordable for the reason extraction
/// is not: it reads user turns only, which are 7.69M of those 61.2M tokens
/// (12.6%), and skips the model call entirely for an episode whose user said
/// nothing substantive. See [`WritePath::extract_profiles`].
///
/// `question_types` restricts the build to one stratum. It is sound rather
/// than a shortcut *because* of the per-`question_id` tenant isolation above:
/// a question may only ever be answered from its own memory, and episode ids
/// are v5 over `(namespace, natural key)`, so the 30 tenants of a stratum
/// build are byte-identical to the same 30 inside the full one.
#[allow(clippy::too_many_arguments)]
pub async fn build_longmemeval_s(
    dataset: &Path,
    collection: &str,
    ledger_path: &Path,
    limit: Option<usize>,
    question_types: Option<&[String]>,
    concurrency: usize,
    repair: bool,
) -> Result<BuildReport> {
    let cfg = MyelinConfig::load().context("load myelin config")?;
    let mut items = longmemeval::load(dataset).context("load longmemeval_s")?;
    // Stratum before `--limit`: truncating the 500 to N and *then* filtering
    // would leave a handful of rows for a 30-tenant stratum.
    if let Some(want) = question_types {
        items.retain(|it| want.iter().any(|w| w == &it.question_type));
        anyhow::ensure!(
            !items.is_empty(),
            "--question-types {want:?} matched no question in {}",
            dataset.display()
        );
    }
    if let Some(n) = limit {
        items.truncate(n);
    }
    eprintln!(
        "  longmemeval_s: {} questions, one memory each",
        items.len()
    );

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
        write.extract_profiles = true;
        // `build.rs` sets `speaker: turn.role.clone()` below, and this
        // corpus's roles are `user`/`assistant`. A corpus that uses person
        // names (LoCoMo does) must leave the pass off rather than guess.
        write.profile_speaker = Some("user".to_string());
        write.concurrency = concurrency.max(1);
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
        eprintln!(
            "  resumed: {resumed} of {} units already ingested",
            items.len()
        );
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
        // The profile pass's own yield. Zero here with a non-zero
        // `profile` wall below means the prompt found nothing, which is a
        // different failure from the pass never running (M20).
        println!("profile records    {}", t.profiles);
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
  profile          {:.1}s
  consolidate      {:.1}s
  index            {:.1}s",
            t.adjudicate_ms as f64 / 1000.0,
            t.extract_ms as f64 / 1000.0,
            t.profile_ms as f64 / 1000.0,
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
        assert_eq!(
            parsed.format("%Y-%m-%d %H:%M").to_string(),
            "2023-05-08 13:56"
        );

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
        assert_eq!(
            parsed.format("%Y-%m-%d %H:%M").to_string(),
            "2023-05-20 02:21"
        );
        // The weekday is redundant with the date and is allowed to be absent.
        let no_day = parse_session_time("2023/05/30 23:40").expect("should parse");
        assert_eq!(
            no_day.format("%Y-%m-%d %H:%M").to_string(),
            "2023-05-30 23:40"
        );
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
