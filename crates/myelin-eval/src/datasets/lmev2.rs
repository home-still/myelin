//! LongMemEval-V2 — agent trajectories (`PLAN.md` §9.1, M3/M6/M8).
//!
//! **The Small tier is two memories, not 451.** Measured on the released
//! haystack: every one of the 451 questions carries exactly 100 trajectory
//! ids, but the union is only 200 — 100 `web` (240 questions) and 100
//! `enterprise` (211 questions), and within a domain every question's
//! haystack is byte-identical. The leaderboard README says the same thing in
//! words ("the same web haystack", "the same enterprise haystack"). So the
//! build is one pass over 200 trajectories into two tenants, not 45,100
//! per-question ingests of the same material.
//!
//! **97.2% of a trajectory is accessibility tree.** Measured over the first
//! 20 trajectories: 16,760,535 of 17,248,384 characters. Projected over the
//! Small tier that is ~43M tokens with trees and ~1.2M without — a 35×
//! difference, so how the tree is handled *is* the ingest design.
//!
//! Three things follow, and each is a decision this loader makes:
//!
//! 1. **The trees cannot be dropped.** 189 of 451 questions are
//!    `static-environment(-abs)`, and their answers are read directly off a
//!    page — e.g. "excluding Edit…, what options are in the Filters
//!    dropdown" → `Incident Mobile, Incident Portal, My Open Incidents`.
//! 2. **The trees cannot be deduplicated away.** Exact-hash dedup within a
//!    trajectory removes only 19.3% (940 states → 712 distinct trees);
//!    consecutive states are byte-identical just 11.3% of the time. Pages
//!    differ by focus rings and timestamps, not by nothing.
//! 3. **The trees must be chunked, for accuracy and not merely for the
//!    embedder's 8192-token limit.** Collapsing a whole page into one 1024-d
//!    vector is the classic long-document failure: the dropdown that answers
//!    the question is one line among thousands. Chunking gives the
//!    answer-bearing region its own vector, and gives BM25 — which Qdrant
//!    computes server-side over the same text — a tight field to score.
//!
//! Measured embedder throughput is 8,279 tok/s (bge-m3, 64-text batches,
//! *while* the LoCoMo ingest was competing for the card), so ~43M tokens is
//! ~1.44 h. That is what makes an episodic ingest of the full haystack
//! affordable at all.
//!
//! No LLM runs over this corpus. `PLAN.md` §9.1 records the reason: the
//! LME-V2 authors' own controller over raw trajectories beats their extracted
//! RAG memory by +16.3/+13.1, and a fact-extraction pass over 25M+ tokens
//! would cost ~250 GPU-hours.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use myelin_core::model::record::SourceRef;
use myelin_core::pipeline::ingest::Turn;
use serde::Deserialize;

/// One recorded agent episode against a live web or enterprise application.
#[derive(Debug, Clone, Deserialize)]
pub struct Trajectory {
    pub id: String,
    pub domain: String,
    pub environment: String,
    pub goal: String,
    /// `success` or `failure`. Load-bearing: the `errors-gotchas` ability asks
    /// what went wrong, so a failed run is evidence, not noise.
    pub outcome: String,
    pub start_url: String,
    pub states: Vec<State>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct State {
    pub state_index: u32,
    pub step: u32,
    pub url: String,
    /// Playwright-style call, e.g. `fill('112', 'WAS1315884', True)`. `None`
    /// on the initial state, which has a page but no preceding action.
    #[serde(default)]
    pub action: Option<String>,
    /// `None` on the initial state and only there. Scanned across all 1,870
    /// trajectories in the release: 1,870 null `action`s and 1,868 null
    /// `thought`s, one per trajectory — the state that has a page but no
    /// preceding decision. Every other field is always present, which is why
    /// none of them is optional here.
    #[serde(default)]
    pub thought: Option<String>,
    #[serde(default)]
    pub accessibility_tree: String,
    #[serde(default)]
    pub screenshot: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Question {
    pub id: String,
    pub domain: String,
    pub environment: String,
    pub question_type: String,
    pub question: String,
    /// A `question_screenshots/*.png` path on **29** of the 451 questions;
    /// JSON `null` on the other 422. Those 29 are why the reader is served
    /// with an mmproj and why `EvidenceKind::Image` exists on the wire.
    #[serde(default)]
    pub image: Option<String>,
    pub answer: String,
    /// The deterministic scorer spec, e.g.
    /// `norm_phrase_set_match|lower=true|separators=,;`. 295 of 451 questions
    /// are judge-free because of this field, which is the column
    /// `EVALUATION.md` §9 requires to be reported separately.
    pub eval_function: String,
}

impl Question {
    /// `-abs` question types are abstention items: the correct answer is that
    /// the haystack does not contain one.
    pub fn is_abstention(&self) -> bool {
        self.question_type.ends_with("-abs")
    }
}

pub fn load_questions(path: &Path) -> Result<Vec<Question>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(i, line)| match line {
            Ok(l) if l.trim().is_empty() => None,
            Ok(l) => Some(
                serde_json::from_str::<Question>(&l)
                    .with_context(|| format!("{}:{}", path.display(), i + 1)),
            ),
            Err(e) => Some(Err(e).with_context(|| format!("{}:{}", path.display(), i + 1))),
        })
        .collect()
}

/// `question_id → trajectory ids`. Collapsed to `domain → trajectory ids` by
/// [`haystacks_by_domain`], which is the form the build actually wants.
pub fn load_haystack(path: &Path) -> Result<HashMap<String, Vec<String>>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("parse {}", path.display()))
}

/// The two real memories, with the "one haystack per domain" claim **checked**
/// rather than assumed: if a future tier ships per-question haystacks this
/// errors instead of silently building a union that no question ever had.
pub fn haystacks_by_domain(
    haystack: &HashMap<String, Vec<String>>,
    questions: &[Question],
) -> Result<HashMap<String, Vec<String>>> {
    let domain_of: HashMap<&str, &str> = questions
        .iter()
        .map(|q| (q.id.as_str(), q.domain.as_str()))
        .collect();

    let mut by_domain: HashMap<String, Vec<String>> = HashMap::new();
    let mut seen: HashMap<String, HashSet<String>> = HashMap::new();

    for (qid, trajs) in haystack {
        let domain = domain_of
            .get(qid.as_str())
            .with_context(|| format!("haystack question {qid} is not in questions.jsonl"))?;
        let set: HashSet<String> = trajs.iter().cloned().collect();
        match seen.get(*domain) {
            Some(existing) if existing != &set => anyhow::bail!(
                "domain {domain} has per-question haystacks ({} vs {} trajectories for {qid}); \
                 the one-memory-per-domain build is invalid for this tier",
                existing.len(),
                set.len()
            ),
            Some(_) => {}
            None => {
                let mut ids: Vec<String> = trajs.clone();
                ids.sort_unstable();
                by_domain.insert((*domain).to_string(), ids);
                seen.insert((*domain).to_string(), set);
            }
        }
    }
    Ok(by_domain)
}

/// Stream the wanted trajectories out of the 1.2 GB JSONL.
///
/// Streaming, not `serde_json::from_reader` over the whole file: one line is
/// up to 1.3 MB and the file is 1,195,604,539 bytes. The callback shape keeps
/// exactly one trajectory in memory at a time.
pub fn for_each_trajectory<F>(path: &Path, wanted: &HashSet<String>, mut f: F) -> Result<usize>
where
    F: FnMut(Trajectory) -> Result<()>,
{
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // 1.3 MB lines: the default 8 KiB buffer would do ~160 refills per line.
    let reader = BufReader::with_capacity(1 << 20, file);
    let mut seen = 0usize;
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("{}:{}", path.display(), i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        // Cheap pre-filter: `id` is the first field of every record, so a
        // substring test avoids parsing 1.3 MB of JSON for a trajectory we do
        // not want. Only ~200 of the file's records are ever wanted.
        if !wanted.iter().any(|id| line.contains(id.as_str())) {
            continue;
        }
        let traj: Trajectory = serde_json::from_str(&line)
            .with_context(|| format!("{}:{} parse trajectory", path.display(), i + 1))?;
        if !wanted.contains(&traj.id) {
            continue;
        }
        seen += 1;
        f(traj)?;
    }
    Ok(seen)
}

/// Target characters per accessibility-tree turn.
///
/// ~4 chars/token (the same approximation [`myelin_core::pipeline::ingest`]
/// uses), so 1,800 chars is ~450 tokens and the segmenter's 512-token cap
/// packs roughly one tree turn per episode. Splitting happens on *line*
/// boundaries because an accessibility tree is one UI element per line, and
/// cutting mid-element would produce a chunk that matches neither the element
/// name nor its role.
const TREE_CHUNK_CHARS: usize = 1_800;

fn chunk_tree(tree: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::with_capacity(TREE_CHUNK_CHARS + 256);
    for line in tree.lines() {
        if !cur.is_empty() && cur.len() + line.len() + 1 > TREE_CHUNK_CHARS {
            chunks.push(std::mem::take(&mut cur));
            cur.reserve(TREE_CHUNK_CHARS + 256);
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// One trajectory as turns, in three unit kinds.
///
/// The segmenter splits on unit change, so the unit string decides episode
/// boundaries. Three kinds, because three different questions are asked of
/// this data and mixing them costs precision:
///
/// | unit | content | serves |
/// |---|---|---|
/// | `<id>#goal` | goal, outcome, environment, start url | "which run was about X" |
/// | `<id>#steps` | every thought+action in order | `procedure`, `errors-gotchas` |
/// | `<id>#state<n>` | url + chunked accessibility tree | `static-environment`, `dynamic-environment` |
///
/// The step list is extracted *separately* rather than left inline because it
/// is 1.3% of the bytes and answers a whole ability category: buried inside a
/// tree chunk it would be outranked by the 97% that surrounds it.
pub fn turns_for(traj: &Trajectory) -> Vec<Turn> {
    let mut turns = Vec::new();

    turns.push(Turn {
        speaker: "goal".into(),
        text: format!(
            "Goal: {}\nEnvironment: {} ({})\nStart URL: {}\nOutcome: {}",
            traj.goal, traj.environment, traj.domain, traj.start_url, traj.outcome
        ),
        at: None,
        source: SourceRef::doc(format!("{}:goal", traj.id)),
        unit: format!("{}#goal", traj.id),
    });

    for state in &traj.states {
        let action = state.action.as_deref().unwrap_or("(initial state)");
        let thought = state.thought.as_deref().unwrap_or("").trim();
        if !thought.is_empty() || state.action.is_some() {
            turns.push(Turn {
                speaker: "step".into(),
                text: format!("[{}] {action}\n{thought}", state.step),
                at: None,
                source: SourceRef::doc(format!("{}:{}", traj.id, state.state_index)),
                unit: format!("{}#steps", traj.id),
            });
        }
    }

    for state in &traj.states {
        let chunks = chunk_tree(&state.accessibility_tree);
        for (i, chunk) in chunks.iter().enumerate() {
            turns.push(Turn {
                speaker: "page".into(),
                // The URL rides on every chunk: it is the only thing that
                // identifies *which page* a fragment of tree came from, and a
                // chunk from the middle of a page otherwise has no anchor a
                // question could match on.
                text: format!("URL: {}\n{chunk}", state.url),
                at: None,
                source: SourceRef::doc(format!("{}:{}:{i}", traj.id, state.state_index)),
                unit: format!("{}#state{}", traj.id, state.state_index),
            });
        }
    }

    turns
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj() -> Trajectory {
        Trajectory {
            id: "t1".into(),
            domain: "web".into(),
            environment: "webarena-reddit".into(),
            goal: "find the top post".into(),
            outcome: "success".into(),
            start_url: "http://x/".into(),
            states: vec![
                State {
                    state_index: 0,
                    step: 0,
                    url: "http://x/a".into(),
                    action: None,
                    thought: Some("look at the list".into()),
                    accessibility_tree: "RootWebArea 'a'\n\t[1] link 'first'\n".into(),
                    screenshot: None,
                },
                State {
                    state_index: 1,
                    step: 1,
                    url: "http://x/b".into(),
                    action: Some("click('1')".into()),
                    thought: Some("open it".into()),
                    accessibility_tree: "RootWebArea 'b'\n\t[2] heading 'second'\n".into(),
                    screenshot: None,
                },
            ],
        }
    }

    #[test]
    fn steps_share_one_unit_so_a_procedure_stays_contiguous() {
        let turns = turns_for(&traj());
        let steps: Vec<_> = turns.iter().filter(|t| t.speaker == "step").collect();
        assert_eq!(steps.len(), 2);
        assert!(
            steps.iter().all(|t| t.unit == "t1#steps"),
            "a procedure question needs the action sequence in one episode, \
             so every step turn must share a unit"
        );
        assert!(steps[1].text.contains("click('1')"));
    }

    #[test]
    fn each_page_state_is_its_own_unit() {
        let turns = turns_for(&traj());
        let pages: Vec<_> = turns.iter().filter(|t| t.speaker == "page").collect();
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].unit, "t1#state0");
        assert_eq!(pages[1].unit, "t1#state1");
        assert!(
            pages.iter().all(|t| t.text.starts_with("URL: http://x/")),
            "a mid-page chunk has no anchor without its URL"
        );
    }

    #[test]
    fn tree_chunks_split_on_line_boundaries() {
        let long = (0..400)
            .map(|i| format!("\t[{i}] button 'element number {i}'"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_tree(&long);
        assert!(chunks.len() > 1, "a 400-element tree must not be one chunk");
        for c in &chunks {
            assert!(
                c.lines().all(|l| l.trim().is_empty() || l.contains("button 'element number")),
                "a chunk boundary cut an element in half: {c:?}"
            );
        }
        let rejoined: String = chunks.concat();
        assert_eq!(
            rejoined.lines().count(),
            long.lines().count(),
            "chunking must be lossless"
        );
    }

    #[test]
    fn per_question_haystacks_are_rejected_not_silently_unioned() {
        let questions = vec![
            Question {
                id: "q1".into(),
                domain: "web".into(),
                environment: "e".into(),
                question_type: "static-environment".into(),
                question: "?".into(),
                image: None,
                answer: "a".into(),
                eval_function: "f".into(),
            },
            Question {
                id: "q2".into(),
                domain: "web".into(),
                environment: "e".into(),
                question_type: "procedure".into(),
                question: "?".into(),
                image: None,
                answer: "a".into(),
                eval_function: "f".into(),
            },
        ];
        let mut hay = HashMap::new();
        hay.insert("q1".to_string(), vec!["t1".to_string()]);
        hay.insert("q2".to_string(), vec!["t2".to_string()]);
        let err = haystacks_by_domain(&hay, &questions).unwrap_err();
        assert!(
            err.to_string().contains("per-question haystacks"),
            "got: {err}"
        );
    }
}
