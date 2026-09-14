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
use myelin_core::model::record::RecordKind;
use myelin_core::pipeline::investigate::{InvestigateTrace, Investigator};
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
    pub async fn open(cfg: &MyelinConfig, collection: &str) -> anyhow::Result<Self> {
        let mut qdrant_cfg = cfg.qdrant.clone();
        qdrant_cfg.collection = collection.to_string();
        Ok(Self {
            store: QdrantStore::new(&qdrant_cfg)?,
            ledger: Ledger::open(&cfg.ledger).await?,
            embedder: RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)?,
            llm: OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model)?,
            reranker: CrossEncoder::new(&cfg.rerank.url, &cfg.rerank.model).ok(),
            config: RetrieveConfig::default(),
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

        let retriever = self.retriever();
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
                max_steps: params.max_steps.unwrap_or(4),
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
}

impl MyelinServer {
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
