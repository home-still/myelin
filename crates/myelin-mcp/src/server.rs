//! The MCP tool surface over `myelin-core` (`PLAN.md` §3.2, §8).
//!
//! **No memory logic lives here.** Every tool is a projection of a
//! `myelin-core` call: the server owns transport, argument schemas and error
//! mapping, and nothing else. If a tool needs a decision made, the decision
//! belongs in the core where the tests are.
//!
//! `recall` is registered now because the LongMemEval-V2 adapter forwards to
//! it (`PLAN.md` §3.3) and M6's break-even run cannot happen without it. The
//! other eight tools of §8 land in M10.

use std::sync::Arc;

use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_router, ErrorData, ServerHandler};
use serde::{Deserialize, Serialize};

use myelin_core::config::MyelinConfig;
use myelin_core::embed::remote::RemoteEmbedder;
use myelin_core::llm::openai::OpenAiLlm;
use myelin_core::model::evidence::{EvidenceSet, WireItem};
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::model::delta::Delta;
use myelin_core::model::record::{ActorId, LinkKind, RecordKind, Scope, SourceRef};
use myelin_core::pipeline::ingest::Turn;
use myelin_core::pipeline::write::WritePath;
use myelin_core::pipeline::investigate::{InvestigateConfig, InvestigateTrace, Investigator};
use myelin_core::pipeline::retrieve::{RecallTrace, RetrieveConfig, Retriever};
use myelin_core::rerank::cross::CrossEncoder;
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::QdrantStore;

/// Everything a read needs, built once and shared across connections.
///
/// **R5 — concurrency-safe reads.** The benchmark harness drives queries from
/// a thread pool against one built memory, so the handler is `Clone` over an
/// `Arc` rather than per-session state. Qdrant and SQLite pools are both
/// internally shared; the ledger is read-only on this path.
pub struct Backend {
    pub store: QdrantStore,
    pub ledger: Ledger,
    pub embedder: RemoteEmbedder,
    /// The controller for `investigate`'s reflect step. Same model as the
    /// writer's judge: R4 wants one store AND one model behind both modes,
    /// so an operating point is a parameter change, not a redeployment.
    pub llm: OpenAiLlm,
    /// `None` when no reranker is configured. That is a legitimate operating
    /// point (the `hybrid_no_rerank` ablation arm), not a degraded mode.
    pub reranker: Option<CrossEncoder>,
    pub config: RetrieveConfig,
}

impl Backend {
    /// `prefetch_limit` and `rerank_depth` override the
    /// [`RetrieveConfig`] defaults; `None` keeps them. They are arguments
    /// rather than globals because a second configuration channel beside
    /// `MyelinConfig` is how an operating point silently stops being
    /// reproducible from the run manifest.
    pub async fn open(
        cfg: &MyelinConfig,
        collection: &str,
        prefetch_limit: Option<u64>,
        rerank_depth: Option<usize>,
    ) -> anyhow::Result<Self> {
        let mut qdrant_cfg = cfg.qdrant.clone();
        qdrant_cfg.collection = collection.to_string();
        let defaults = RetrieveConfig::default();
        Ok(Self {
            store: QdrantStore::new(&qdrant_cfg)?,
            ledger: Ledger::open(&cfg.ledger).await?,
            embedder: RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)?,
            llm: OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model)?,
            reranker: CrossEncoder::new(&cfg.rerank.url, &cfg.rerank.model).ok(),
            config: RetrieveConfig {
                prefetch_limit: prefetch_limit.unwrap_or(defaults.prefetch_limit),
                rerank_depth: rerank_depth.unwrap_or(defaults.rerank_depth),
                // Documented as equal to `prefetch_limit` so every fused
                // channel contributes the same depth (`retrieve.rs:125-128`).
                // `graph` itself stays off — M12 measured the PPR channel as
                // a loss.
                graph_limit: prefetch_limit.map_or(defaults.graph_limit, |p| p as usize),
                ..defaults
            },
        })
    }
}

#[derive(Clone)]
pub struct MyelinServer {
    backend: Arc<Backend>,
    tool_router: rmcp::handler::server::tool::ToolRouter<Self>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallParams {
    /// The question, verbatim. Nothing else about the caller's task is
    /// accepted — see `PLAN.md` R6: a memory backend that can see the
    /// benchmark's metadata is disqualified, so there is deliberately no
    /// field here for a question id, a category, or a gold answer.
    pub query: String,
    /// Mandatory. There is no read path that can span tenants (C12).
    pub tenant: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub session: Option<String>,
    /// Evidence-set size. Small on purpose: HiGMem retrieves 8.09 vs 99.84
    /// turns/query at P@K 0.1909 vs 0.0101, and MINJA ASR climbs 6% → 20% →
    /// 38% as k goes 3 → 5 → 10 (`PLAN.md` §2 findings 4 and 9).
    #[serde(default)]
    pub k: Option<usize>,
    #[serde(default)]
    pub budget_tokens: Option<usize>,
    /// `episodic` | `semantic` | `procedural` | `working`.
    #[serde(default)]
    pub kinds: Option<Vec<String>>,
    /// Withhold the evidence set entirely when the best cross-encoder score
    /// is below this. Lets a caller trade answered-wrong for abstained,
    /// which is the dominant term in LongMemEval-V2's score.
    #[serde(default)]
    pub tau_abstain: Option<f32>,
}

/// The `recall` response.
///
/// `items` is the R1 wire shape — `{type, value}` and nothing else — because
/// the LongMemEval-V2 `Memory.query()` contract requires exactly that list.
/// Everything the benchmark must not see, and everything we want for our own
/// manifests, is carried in the sibling fields instead of being smuggled into
/// the wire items.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct RecallResult {
    pub items: Vec<WireItem>,
    pub record_ids: Vec<String>,
    pub tokens: usize,
    pub trace: RecallTraceJson,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct RecallTraceJson {
    pub dense_hits: usize,
    pub lex_hits: usize,
    pub fused: usize,
    pub reranked: usize,
    pub admitted: usize,
    pub embed_ms: u64,
    pub search_ms: u64,
    pub rerank_ms: u64,
    pub total_ms: u64,
    pub top_score: Option<f32>,
    pub abstained: bool,
}

impl From<RecallTrace> for RecallTraceJson {
    fn from(t: RecallTrace) -> Self {
        Self {
            dense_hits: t.dense_hits,
            lex_hits: t.lex_hits,
            fused: t.fused,
            reranked: t.reranked,
            admitted: t.admitted,
            embed_ms: t.embed_ms as u64,
            search_ms: t.search_ms as u64,
            rerank_ms: t.rerank_ms as u64,
            total_ms: t.total_ms as u64,
            top_score: t.top_score,
            abstained: t.abstained,
        }
    }
}

#[tool_router]
impl MyelinServer {
    pub fn new(backend: Arc<Backend>) -> Self {
        Self {
            backend,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "recall",
        description = "Fast hybrid retrieval over one tenant's memory. BM25 + dense in one \
                       round trip, RRF-fused, cross-encoder reranked, budgeted and \
                       deduplicated. No LLM in the loop."
    )]
    async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<Json<RecallResult>, ErrorData> {
        let kinds = match &params.kinds {
            Some(names) => Some(
                names
                    .iter()
                    .map(|n| RecordKind::parse(n))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| ErrorData::invalid_params(e.to_string(), None))?,
            ),
            None => None,
        };

        let mut scope = ScopeFilter::tenant(&params.tenant);
        scope.namespace = params.namespace.clone();
        scope.agent = params.agent.clone();
        scope.session = params.session.clone();

        let defaults = Budget::default();
        let query = Recall {
            scope,
            text: params.query,
            budget: Budget {
                k: params.k.unwrap_or(defaults.k),
                tokens: params.budget_tokens.unwrap_or(defaults.tokens),
                max_steps: defaults.max_steps,
            },
            mode: Mode::Recall,
            kinds,
        };

        let mut retriever = self.retriever();
        if params.tau_abstain.is_some() {
            retriever.config.tau_abstain = params.tau_abstain;
        }
        let (evidence, trace) = retriever
            .recall(&query)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(Json(to_result(evidence, trace)))
    }

    #[tool(
        name = "investigate",
        description = "Agentic retrieval: search, reflect, search again, until the evidence is \
                       sufficient and self-consistent or the step budget runs out. Slower and \
                       more accurate than recall; returns the same evidence shape."
    )]
    async fn investigate(
        &self,
        Parameters(params): Parameters<InvestigateParams>,
    ) -> Result<Json<InvestigateResult>, ErrorData> {
        let defaults = Budget::default();
        let mut scope = ScopeFilter::tenant(&params.tenant);
        scope.namespace = params.namespace.clone();

        let query = Recall {
            scope,
            text: params.question,
            budget: Budget {
                k: params.k.unwrap_or(defaults.k),
                tokens: params.budget_tokens.unwrap_or(defaults.tokens),
                // One source of truth: the measured default lives on
                // `InvestigateConfig` with the curve that justifies it.
                max_steps: params
                    .max_steps
                    .unwrap_or(InvestigateConfig::default().max_steps),
            },
            mode: Mode::Investigate,
            kinds: None,
        };

        let retriever = self.retriever();
        let (evidence, trace) = Investigator::new(&self.backend.llm, &retriever)
            .investigate(&query)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(Json(InvestigateResult {
            items: evidence.to_wire(),
            record_ids: evidence.items.iter().map(|i| i.record_id.to_string()).collect(),
            tokens: evidence.tokens,
            queries: evidence.trace.iter().map(|t| t.query.clone()).collect(),
            trace,
        }))
    }

    #[tool(
        name = "remember",
        description = "Write one statement through the full write path: extract, consolidate against \
                       neighbours, apply a four-op delta, index. Idempotent by content."
    )]
    async fn remember(
        &self,
        Parameters(params): Parameters<RememberParams>,
    ) -> Result<Json<WriteResult>, ErrorData> {
        let scope = Scope::new(&params.tenant, params.agent.as_deref().unwrap_or("myelin"),
                               params.namespace.as_deref().unwrap_or("default"));
        let turn = Turn {
            speaker: params.speaker.unwrap_or_else(|| "user".into()),
            text: params.text,
            at: params.t_valid,
            source: SourceRef::doc(params.source.unwrap_or_else(|| "remember".into())),
            unit: params.unit.unwrap_or_else(|| "remember".into()),
        };
        self.write(&scope, &[turn]).await
    }

    #[tool(
        name = "observe",
        description = "Bulk-ingest a conversation or trajectory segment. Segmentation into episodes \
                       is the server's job, not the caller's."
    )]
    async fn observe(
        &self,
        Parameters(params): Parameters<ObserveParams>,
    ) -> Result<Json<WriteResult>, ErrorData> {
        let scope = Scope::new(&params.tenant, params.agent.as_deref().unwrap_or("myelin"),
                               params.namespace.as_deref().unwrap_or("default"));
        let unit = params.unit.unwrap_or_else(|| "observe".into());
        let turns: Vec<Turn> = params
            .turns
            .into_iter()
            .enumerate()
            .map(|(i, t)| Turn {
                speaker: t.speaker,
                text: t.text,
                at: t.at,
                source: SourceRef::doc(t.source.unwrap_or_else(|| format!("{unit}:{i}"))),
                unit: unit.clone(),
            })
            .collect();
        self.write(&scope, &turns).await
    }

    #[tool(
        name = "search",
        description = "Record stubs matching a scope filter. The primitive for caller-driven \
                       iteration: identifiers and previews, never full records."
    )]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<Json<SearchResult>, ErrorData> {
        let mut scope = ScopeFilter::tenant(&params.tenant);
        scope.namespace = params.namespace.clone();
        scope.agent = params.agent.clone();
        let limit = params.limit.unwrap_or(20).min(200) as i64;
        let records = self
            .backend
            .ledger
            .visible(&scope, chrono::Utc::now(), limit)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(Json(SearchResult {
            records: records.iter().map(RecordStub::from_record).collect(),
        }))
    }

    #[tool(
        name = "neighbors",
        description = "Records linked to this one by a typed edge. One hop; call again to walk."
    )]
    async fn neighbors(
        &self,
        Parameters(params): Parameters<NeighborsParams>,
    ) -> Result<Json<NeighborsResult>, ErrorData> {
        let id = parse_uuid(&params.record_id)?;
        let relation = match &params.relation {
            Some(r) => Some(LinkKind::parse(r).map_err(|e| ErrorData::invalid_params(e.to_string(), None))?),
            None => None,
        };
        let links = self
            .backend
            .ledger
            .links(params.namespace.as_deref())
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        let mut out = Vec::new();
        for link in links {
            // Edges are walked in both directions: `supersedes` is only
            // useful if you can ask "what replaced this?" as well as "what
            // did this replace?".
            let other = if link.src == id {
                link.dst
            } else if link.dst == id {
                link.src
            } else {
                continue;
            };
            if relation.is_some_and(|r| r != link.relation) {
                continue;
            }
            if let Some(record) = self
                .backend
                .ledger
                .get(other)
                .await
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?
            {
                out.push(NeighborStub {
                    relation: link.relation.as_str().to_string(),
                    inbound: link.dst == id,
                    record: RecordStub::from_record(&record),
                });
            }
        }
        Ok(Json(NeighborsResult { neighbors: out }))
    }

    #[tool(
        name = "forget",
        description = "Invalidate a record (soft) or erase it and everything derived from it (hard). \
                       Hard deletion requires confirm=true and cannot be undone."
    )]
    async fn forget(
        &self,
        Parameters(params): Parameters<ForgetParams>,
    ) -> Result<Json<ForgetResult>, ErrorData> {
        let id = parse_uuid(&params.record_id)?;
        let actor = ActorId::new(params.actor.as_deref().unwrap_or("mcp"));
        let reason = params.reason.unwrap_or_else(|| "forget via mcp".into());

        match params.mode.as_str() {
            "soft" => {
                let applied = self
                    .backend
                    .ledger
                    .apply(&Delta::Delete { target: id, reason: reason.clone() }, &actor)
                    .await
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                let _ = applied;
                Ok(Json(ForgetResult { mode: "soft".into(), affected: vec![id.to_string()] }))
            }
            "hard" => {
                // C11 unlearning erases descendants too, so the confirmation
                // is not ceremony: the caller usually cannot see how many
                // records are about to go.
                if !params.confirm.unwrap_or(false) {
                    return Err(ErrorData::invalid_params(
                        "hard deletion erases this record AND every record derived from it; \
                         pass confirm=true".to_string(),
                        None,
                    ));
                }
                let gone = self
                    .backend
                    .ledger
                    .hard_delete(id, &actor, &reason)
                    .await
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                self.backend
                    .store
                    .delete_points(&gone)
                    .await
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                Ok(Json(ForgetResult {
                    mode: "hard".into(),
                    affected: gone.iter().map(|u| u.to_string()).collect(),
                }))
            }
            other => Err(ErrorData::invalid_params(
                format!("mode must be \"soft\" or \"hard\", got {other:?}"),
                None,
            )),
        }
    }

    #[tool(
        name = "review_quarantine",
        description = "Staged writes awaiting a decision, with the reason each was held."
    )]
    async fn review_quarantine(
        &self,
        Parameters(params): Parameters<LimitParams>,
    ) -> Result<Json<QuarantineResult>, ErrorData> {
        let rows = self
            .backend
            .ledger
            .review_quarantine(params.limit.unwrap_or(20).min(200) as i64)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(Json(QuarantineResult {
            staged: rows
                .iter()
                .map(|r| QuarantineStub {
                    id: r.id.to_string(),
                    reason: r.reason.clone(),
                    at: r.at.to_rfc3339(),
                    record: RecordStub::from_record(&r.record),
                })
                .collect(),
        }))
    }

    #[tool(
        name = "explain",
        description = "Why this record exists: its lineage back to source episodes, and every audit \
                       event that touched it."
    )]
    async fn explain(
        &self,
        Parameters(params): Parameters<ExplainParams>,
    ) -> Result<Json<Explanation>, ErrorData> {
        let id = parse_uuid(&params.record_id)?;
        let lineage = self
            .backend
            .ledger
            .lineage(id)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?
            .ok_or_else(|| ErrorData::invalid_params(format!("no record {id}"), None))?;
        let events = self
            .backend
            .ledger
            .events(Some(id))
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(Json(Explanation {
            lineage: LineageJson::from_node(&lineage),
            events: events
                .iter()
                .map(|e| EventJson {
                    seq: e.seq,
                    at: e.at.to_rfc3339(),
                    kind: e.kind.clone(),
                    actor: e.actor.as_str().to_string(),
                    reason: e.reason.clone(),
                })
                .collect(),
        }))
    }
}


impl MyelinServer {
    /// One write path for `remember` and `observe`: the difference between
    /// them is how many turns the caller has, not what happens to them.
    async fn write(&self, scope: &Scope, turns: &[Turn]) -> Result<Json<WriteResult>, ErrorData> {
        let stats = WritePath::new(
            &self.backend.llm,
            &self.backend.embedder,
            &self.backend.store,
            &self.backend.ledger,
        )
        .insert(scope, turns)
        .await
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(Json(WriteResult {
            episodes: stats.episodes,
            candidates: stats.candidates,
            added: stats.added,
            updated: stats.updated,
            deleted: stats.deleted,
            noop: stats.noop,
            duplicates: stats.duplicates,
            quarantined: stats.quarantined,
            rejected: stats.rejected,
            adjudicated_out: stats.adjudicated_out,
            wall_ms: stats.wall_ms,
        }))
    }

    /// Both modes drive the same retriever over the same store (R4).
    fn retriever(&self) -> Retriever<'_> {
        let mut r = Retriever::new(
            &self.backend.embedder,
            &self.backend.store,
            &self.backend.ledger,
        )
        .with_config(self.backend.config.clone());
        if let Some(reranker) = &self.backend.reranker {
            r = r.with_reranker(reranker);
        }
        r
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct InvestigateParams {
    pub question: String,
    pub tenant: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub k: Option<usize>,
    #[serde(default)]
    pub budget_tokens: Option<usize>,
    /// Iteration cap. The latency knob that makes a second operating point
    /// possible from one built memory (R4).
    #[serde(default)]
    pub max_steps: Option<usize>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct InvestigateResult {
    pub items: Vec<WireItem>,
    pub record_ids: Vec<String>,
    pub tokens: usize,
    /// Every search the loop actually issued, in order — the agentic trace
    /// `EVALUATION.md` §9 requires.
    pub queries: Vec<String>,
    pub trace: InvestigateTrace,
}

fn to_result(evidence: EvidenceSet, trace: RecallTrace) -> RecallResult {
    RecallResult {
        items: evidence.to_wire(),
        record_ids: evidence
            .items
            .iter()
            .map(|i| i.record_id.to_string())
            .collect(),
        tokens: evidence.tokens,
        trace: trace.into(),
    }
}

// `router = self.tool_router` rather than the default: the default form
// calls `Self::tool_router()` and so rebuilds the whole router on every
// tool call. The field is built once in `new`.
#[rmcp::tool_handler(router = self.tool_router)]
impl ServerHandler for MyelinServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("myelin-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Agentic memory. `recall` is the fast read path; it never returns more than k \
                 items and every item is individually addressable by record id.",
            )
    }
}

// ── Tool argument and result shapes ──────────────────────────────────
//
// Every result here is a *stub*: identifiers plus enough text to decide
// whether to ask for more. `PLAN.md` §8 makes that a design rule, not a
// preference — "tool results are summaries with identifiers, never raw
// dumps (progressive disclosure)" — because a tool that returns whole
// records spends the caller's context on material it did not ask for.

fn parse_uuid(s: &str) -> Result<uuid::Uuid, ErrorData> {
    uuid::Uuid::parse_str(s)
        .map_err(|e| ErrorData::invalid_params(format!("bad record id {s:?}: {e}"), None))
}

/// How much of a record's text a stub carries.
///
/// Long enough to recognise the record, short enough that twenty of them
/// do not displace the evidence set. A caller that wants the whole text
/// asks `explain` or `recall` for it.
const PREVIEW_CHARS: usize = 180;

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct RecordStub {
    pub id: String,
    pub kind: String,
    pub preview: String,
    pub t_valid: String,
    pub trust: String,
    /// True when the record has been superseded or expired. Stubs surface
    /// this because a caller iterating with `search` is otherwise unable to
    /// tell a live fact from a historical one.
    pub invalidated: bool,
}

impl RecordStub {
    fn from_record(r: &myelin_core::model::record::MemoryRecord) -> Self {
        let mut preview: String = r.text.chars().take(PREVIEW_CHARS).collect();
        if r.text.chars().count() > PREVIEW_CHARS {
            preview.push('…');
        }
        Self {
            id: r.id.to_string(),
            kind: r.kind.as_str().to_string(),
            preview,
            t_valid: r.validity.t_valid.to_rfc3339(),
            trust: r.trust.tier.as_str().to_string(),
            invalidated: r.validity.t_invalid.is_some() || r.validity.t_expired.is_some(),
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RememberParams {
    pub text: String,
    pub tenant: String,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub t_valid: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ObserveTurn {
    pub speaker: String,
    pub text: String,
    #[serde(default)]
    pub at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ObserveParams {
    pub turns: Vec<ObserveTurn>,
    pub tenant: String,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub unit: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct WriteResult {
    pub episodes: usize,
    pub candidates: usize,
    pub added: usize,
    pub updated: usize,
    pub deleted: usize,
    pub noop: usize,
    pub duplicates: usize,
    /// Staged, not applied. A non-zero count here is the caller's cue to
    /// run `review_quarantine` (C4).
    pub quarantined: usize,
    pub rejected: usize,
    /// Episodes the injection adjudicator refused before storage (M15).
    /// Staged in quarantine like `quarantined`, but earlier: these never
    /// reached extraction at all.
    pub adjudicated_out: usize,
    pub wall_ms: u128,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    pub tenant: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct NeighborsParams {
    pub record_id: String,
    /// `supersedes` | `contradicts` | `supports` | `mentions`.
    #[serde(default)]
    pub relation: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NeighborStub {
    pub relation: String,
    /// True when the edge points *at* the queried record. Direction is the
    /// difference between "what this replaced" and "what replaced this".
    pub inbound: bool,
    pub record: RecordStub,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetParams {
    pub record_id: String,
    /// `soft` invalidates and keeps history; `hard` erases (C11 unlearning).
    pub mode: String,
    #[serde(default)]
    pub confirm: Option<bool>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ForgetResult {
    pub mode: String,
    pub affected: Vec<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LimitParams {
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct QuarantineStub {
    pub id: String,
    pub reason: String,
    pub at: String,
    pub record: RecordStub,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExplainParams {
    pub record_id: String,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LineageJson {
    pub id: String,
    pub kind: String,
    pub preview: String,
    pub ancestors: Vec<LineageJson>,
}

impl LineageJson {
    fn from_node(n: &myelin_core::store::ledger::LineageNode) -> Self {
        let mut preview: String = n.text.chars().take(PREVIEW_CHARS).collect();
        if n.text.chars().count() > PREVIEW_CHARS {
            preview.push('…');
        }
        Self {
            id: n.id.to_string(),
            kind: n.kind.as_str().to_string(),
            preview,
            ancestors: n.ancestors.iter().map(Self::from_node).collect(),
        }
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct EventJson {
    pub seq: i64,
    pub at: String,
    pub kind: String,
    pub actor: String,
    pub reason: String,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct Explanation {
    pub lineage: LineageJson,
    pub events: Vec<EventJson>,
}

// MCP requires every tool's `outputSchema.type` to be `"object"`. A tool
// returning a bare JSON array is a protocol violation that a hand-rolled
// client happily ignores and the official SDK rejects outright with
// `Input should be 'object'` — which is how these three were found. The
// wrappers are also where a cursor goes when these grow pagination.

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SearchResult {
    pub records: Vec<RecordStub>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NeighborsResult {
    pub neighbors: Vec<NeighborStub>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct QuarantineResult {
    pub staged: Vec<QuarantineStub>,
}
