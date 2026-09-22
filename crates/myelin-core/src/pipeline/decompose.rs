//! `decompose` — retrieve for the sub-questions, not the question (M24).
//!
//! # Why, after six nulls
//!
//! Every retrieval mechanism measured since M12 has been a way of *reordering
//! or trimming* one candidate pool drawn from **one** query: graph fusion
//! (M12, −0.7), evidence order (M13, a coin flip), width at k=25 (M19, CI
//! spans zero), MMR (M21, −5.3 and −9.8), sufficiency selection (M21/M22,
//! +3.8 on one corpus and −1.1 on the one G1 is scored on). The only
//! mechanism that moved a stratum was M19's resolved dates, which changed
//! **what the records said** rather than which records were drawn.
//!
//! That is not a coincidence. M16 and M22 both measure the system as
//! *retrieval-limited* — S = P(evidence sufficient | answer wrong) = 12.1%
//! [8.1, 17.6], so a perfect reader over today's evidence reaches 41.5 while
//! perfect retrieval reaches 76.9. Reordering a pool cannot add a record the
//! pool never contained.
//!
//! # The mechanism, from AgentRunbook-R
//!
//! AgentRunbook-R scores **58.60** on LME-V2-Small with the same Qwen3.5-9B
//! reader we serve. Its vendored implementation is in this repo
//! (`crates/myelin-eval/vendor/longmemeval-v2/memory_modules/agentrunbook_r.py`),
//! so the mechanism is read rather than inferred, and it is not the three
//! pools: **one** model call emits a structured bundle of sub-queries
//! (`QUERY_GENERATION_SYSTEM_PROMPT`, L126–162), each is retrieved
//! *separately*, and each block is reranked against the **original
//! question** (`_query_with_rerank`, L1245–1324).
//!
//! Our `investigate` is the sequential dual: one query per step, each
//! conditioned on the last step's results. M19 measured two steps at exactly
//! 0.0 and M22 measured pool selection at −1.1. Both are sequential.
//! Parallel decomposition has never been tried here.
//!
//! # Integration: N more lists into the fusion that already exists
//!
//! [`crate::pipeline::retrieve::rrf`] is already a multi-list fuser. Each
//! sub-query contributes its own dense and lexical list, so `n` sub-queries
//! add `2n` lists to the same RRF call and **every downstream stage is
//! untouched** — admissibility, rerank, selection and compose all see one
//! ordinary fused list. Two properties fall out for free:
//!
//! - A record that answers *two* sub-questions accumulates reciprocal rank
//!   from both and rises. That is precisely the multi-hop shape.
//! - The rerank already runs against `query.text`, the original question, so
//!   AgentRunbook-R's "rerank the block against the original, not the
//!   generated query" is what this codebase already did. It is also M22's
//!   independent finding: scores from different probe queries are logits on
//!   different scales and are not mutually comparable.
//!
//! **The original question's own lists are always retained.** A sub-query set
//! that misses the point can then only ever *add* candidates, so the pool
//! grows monotonically and the off-path is byte-identical to a run with zero
//! sub-queries.
//!
//! # What this must not become
//!
//! M21's verdict is that the memories which jointly answer one question
//! resemble *each other* 1.60× more than they resemble the rest of the
//! composed set (token-Jaccard 0.2235 vs 0.1399). Any de-duplication across
//! sub-query results would re-inflict exactly the damage MMR did. Union,
//! then rank. Nothing here drops a candidate.
//!
//! # It can never be a `recall` default
//!
//! `PLAN.md` §7.1 specifies `recall` as "target p95 < 100 ms, **no LLM in the
//! loop**". A model call per query violates that by construction whatever it
//! measures, so the switch ships off for `recall` regardless of outcome; its
//! home if it wins is `investigate`, which already pays for model calls.
//!
//! # It never fails the caller
//!
//! Every model-quality failure — unparseable body, empty list, prose —
//! degrades to *no sub-queries*, which is the unmodified single-query path.
//! `EmptyCompletion` is the one exception and propagates: R7 says a
//! zero-byte body is a dead model, not the model declining to decompose.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{MyelinError, Result};
use crate::llm::{complete_json, CompletionRequest, Llm, Message};

/// Hard ceiling on sub-queries, whatever the caller or the model asks for.
///
/// AgentRunbook-R caps its raw-state queries at 5 and adds one event and one
/// note query, so seven retrievals per question is the published shape. Each
/// costs an embedding and a `hybrid_search`, and the reranker then sees a
/// pool that is up to `n` times wider — the cost is real and bounded here
/// rather than at the call site.
pub const MAX_SUBQUERIES: usize = 6;

const DECOMPOSE_SYSTEM: &str = "\
You split one question into the smallest set of independent search queries \
that together retrieve everything needed to answer it.

Rules:
- Each query must target a DIFFERENT fact, entity, event or time period.
- A question that needs one fact gets ONE query. Do not invent sub-questions.
- A question that joins two or more things — a comparison, a chain, a \
before/after, a total over several occasions — gets one query per part.
- Preserve exact names, labels and dates from the question. Do not \
paraphrase an entity into a description.
- Write each query as the statement you want to find, not as a question.
- Do not answer the question.
- Do not add a query for context, background or general topic.";

/// `{"queries": ["...", "..."]}`
pub fn decomposition_schema(max: usize) -> serde_json::Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["queries"],
        "properties": {
            "queries": {
                "type": "array",
                "maxItems": max,
                "items": { "type": "string", "maxLength": 200 }
            }
        }
    })
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Decomposition {
    queries: Vec<String>,
}

pub struct Decomposer<'a> {
    llm: &'a dyn Llm,
    max_tokens: u32,
}

impl<'a> Decomposer<'a> {
    pub fn new(llm: &'a dyn Llm) -> Self {
        // Six short strings. The ceiling stops a model that starts narrating
        // from spending a reader's worth of tokens on a retrieval helper.
        Self {
            llm,
            max_tokens: 256,
        }
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    /// Sub-queries to retrieve *in addition to* `question`, at most `max`.
    ///
    /// An empty result is the correct answer for a single-fact question and
    /// is also the fallback for every model-quality failure; in both cases
    /// the caller's behaviour is the unmodified single-query path.
    ///
    /// Returned queries are trimmed, de-duplicated case-insensitively, and
    /// never equal to the question itself — the caller always retrieves the
    /// original, so repeating it would give one query two votes in the RRF
    /// fusion and quietly re-weight the channel.
    pub async fn decompose(&self, question: &str, max: usize) -> Result<Vec<String>> {
        let max = max.min(MAX_SUBQUERIES);
        if max == 0 || question.trim().is_empty() {
            return Ok(Vec::new());
        }

        let request = CompletionRequest::new(vec![
            Message::system(DECOMPOSE_SYSTEM),
            Message::user(format!(
                "<question>\n{question}\n</question>\n\
                 Return at most {max} queries. Return one query if one is enough."
            )),
        ])
        .with_schema(decomposition_schema(max))
        .with_max_tokens(self.max_tokens);

        let parsed: Decomposition = match complete_json(self.llm, &request).await {
            Ok(d) => d,
            // R7: a zero-byte body is a dead model, not a considered "this
            // question does not decompose".
            Err(e @ MyelinError::EmptyCompletion { .. }) => return Err(e),
            // Anything else is the model producing something unusable, which
            // costs the latency and changes nothing.
            Err(_) => return Ok(Vec::new()),
        };

        Ok(clean(parsed.queries, question, max))
    }
}

/// Trim, drop empties and the question itself, de-duplicate, and cap.
///
/// Free of the model and the network so the contract is testable directly:
/// this is where every guarantee the doc comment makes actually lives.
fn clean(raw: Vec<String>, question: &str, max: usize) -> Vec<String> {
    let norm = |s: &str| s.trim().to_lowercase();
    let question_norm = norm(question);
    let mut seen: Vec<String> = Vec::new();
    let mut out = Vec::with_capacity(max);
    for q in raw {
        let trimmed = q.trim();
        if trimmed.is_empty() {
            continue;
        }
        let key = norm(trimmed);
        if key == question_norm || seen.contains(&key) {
            continue;
        }
        seen.push(key);
        out.push(trimmed.to_string());
        if out.len() == max {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Completion, Usage};
    use async_trait::async_trait;

    /// One canned body, or one canned error, per construction.
    struct Canned(std::sync::Mutex<Vec<Result<Completion>>>);

    impl Canned {
        fn text(body: &str) -> Self {
            Self(std::sync::Mutex::new(vec![Ok(Completion {
                reasoning: None,
                text: body.into(),
                tool_calls: vec![],
                finish_reason: None,
                usage: Usage::default(),
            })]))
        }

        fn dead() -> Self {
            Self(std::sync::Mutex::new(vec![Err(
                MyelinError::EmptyCompletion {
                    model: "test".into(),
                },
            )]))
        }
    }

    #[async_trait]
    impl Llm for Canned {
        fn id(&self) -> &str {
            "canned"
        }
        async fn raw_complete(&self, _req: &CompletionRequest) -> Result<Completion> {
            self.0
                .lock()
                .unwrap()
                .pop()
                .unwrap_or(Err(MyelinError::Store("exhausted".into())))
        }
    }

    #[tokio::test]
    async fn a_multi_part_question_yields_one_query_per_part() {
        let llm = Canned::text(r#"{"queries": ["Alice's new job", "Bob's move to Berlin"]}"#);
        let got = Decomposer::new(&llm)
            .decompose("Did Alice change jobs before Bob moved to Berlin?", 6)
            .await
            .unwrap();
        assert_eq!(got, vec!["Alice's new job", "Bob's move to Berlin"]);
    }

    /// The off-path must be reachable from the model too: a single-fact
    /// question that decomposes into nothing leaves the caller on the
    /// unmodified single-query path.
    #[tokio::test]
    async fn an_empty_decomposition_is_not_an_error() {
        let llm = Canned::text(r#"{"queries": []}"#);
        assert!(Decomposer::new(&llm)
            .decompose("what is my dog's name", 6)
            .await
            .unwrap()
            .is_empty());
    }

    /// The caller always retrieves the original question. A sub-query equal
    /// to it would give that one query two lists in the RRF fusion and
    /// silently double its weight against the others.
    #[tokio::test]
    async fn the_question_is_never_returned_as_its_own_subquery() {
        let llm =
            Canned::text(r#"{"queries": ["  Where does Alice work?  ", "Alice's salary"]}"#);
        let got = Decomposer::new(&llm)
            .decompose("Where does Alice work?", 6)
            .await
            .unwrap();
        assert_eq!(got, vec!["Alice's salary"]);
    }

    #[tokio::test]
    async fn duplicates_and_blanks_are_dropped() {
        let llm = Canned::text(r#"{"queries": ["Alice job", "  ", "ALICE JOB", "Bob move"]}"#);
        let got = Decomposer::new(&llm).decompose("q", 6).await.unwrap();
        assert_eq!(got, vec!["Alice job", "Bob move"]);
    }

    /// The cap is the caller's, and `MAX_SUBQUERIES` is the ceiling over it:
    /// the cost of this mechanism is one embedding and one search per query,
    /// so an unbounded list is an unbounded query.
    #[tokio::test]
    async fn the_cap_is_enforced_on_the_model_not_trusted_from_it() {
        let many: Vec<String> = (0..20).map(|i| format!("q{i}")).collect();
        let body = serde_json::json!({ "queries": many }).to_string();
        assert_eq!(
            Decomposer::new(&Canned::text(&body))
                .decompose("q", 3)
                .await
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            Decomposer::new(&Canned::text(&body))
                .decompose("q", usize::MAX)
                .await
                .unwrap()
                .len(),
            MAX_SUBQUERIES
        );
        // Zero is off, and off must not even call the model — the fake
        // would still answer, so this asserts the early return by asking
        // for a cap no response could satisfy.
        assert!(Decomposer::new(&Canned::text(&body))
            .decompose("q", 0)
            .await
            .unwrap()
            .is_empty());
    }

    /// A model-quality failure costs the latency and changes nothing.
    #[tokio::test]
    async fn prose_degrades_to_the_single_query_path() {
        let llm = Canned::text("Sure! Here are some queries you might like.");
        assert!(Decomposer::new(&llm)
            .decompose("q", 6)
            .await
            .unwrap()
            .is_empty());
    }

    /// R7: a zero-byte body is a dead model and must not be read as "this
    /// question does not decompose".
    #[tokio::test]
    async fn an_empty_completion_propagates() {
        let llm = Canned::dead();
        assert!(matches!(
            Decomposer::new(&llm).decompose("q", 6).await,
            Err(MyelinError::EmptyCompletion { .. })
        ));
    }
}
