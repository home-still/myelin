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

use futures_util::future::try_join_all;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_router, ErrorData, ServerHandler};
use serde::{Deserialize, Serialize};

use myelin_core::config::MyelinConfig;
use myelin_core::embed::remote::RemoteEmbedder;
use myelin_core::llm::openai::OpenAiLlm;
use myelin_core::model::delta::Delta;
use myelin_core::model::evidence::{EvidenceSet, WireItem};
use myelin_core::model::query::{Budget, MetaCondition, Mode, Recall, ScopeFilter};
use myelin_core::model::record::{ActorId, LinkKind, RecordKind, Scope, SourceRef};
use myelin_core::model::evidence::EvidenceItem;
use myelin_core::pipeline::extract::CandidateKind;
use myelin_core::pipeline::ingest::Turn;
use myelin_core::pipeline::compose::KindQuota;
use myelin_core::pipeline::investigate::{InvestigateConfig, InvestigateTrace, Investigator};
use myelin_core::pipeline::retrieve::{RecallTrace, RetrieveConfig, Retriever};
use myelin_core::pipeline::trajectory_agent::{TrajectoryAgent, TrajectoryAgentConfig};
use myelin_core::pipeline::trajectory_tools::TrajectoryTools;
use myelin_core::pipeline::write::WritePath;
use myelin_core::rerank::cross::CrossEncoder;
use myelin_core::store::ledger::Ledger;
use myelin_core::store::qdrant::QdrantStore;
use myelin_core::embed::Embedder;
use myelin_core::llm::Llm;
use myelin_core::model::store::StoreDef;
use myelin_core::pipeline::merge::{merge_scopes, MergedEvidence};
use myelin_core::rerank::Reranker;

/// Everything a read needs, built once and shared across connections.
///
/// **R5 — concurrency-safe reads.** The benchmark harness drives queries from
/// a thread pool against one built memory, so the handler is `Clone` over an
/// `Arc` rather than per-session state. Qdrant and SQLite pools are both
/// internally shared; the ledger is read-only on this path.
pub struct Backend {
    pub store: QdrantStore,
    pub ledger: Ledger,
    /// The embedder, the model and the reranker are trait objects so a test
    /// can serve this backend with stand-ins that need no model server.
    pub embedder: Box<dyn Embedder>,
    /// The controller for `investigate`'s reflect step. Same model as the
    /// writer's judge: R4 wants one store AND one model behind both modes,
    /// so an operating point is a parameter change, not a redeployment.
    pub llm: Box<dyn Llm>,
    /// `None` when no reranker is configured. That is a legitimate operating
    /// point (the `hybrid_no_rerank` ablation arm), not a degraded mode.
    pub reranker: Option<Box<dyn Reranker>>,
    pub config: RetrieveConfig,
}

impl Backend {
    /// `prefetch_limit`, `rerank_depth` and `rerank_factor` override the
    /// [`RetrieveConfig`] defaults; `None` keeps them. They are arguments
    /// rather than globals because a second configuration channel beside
    /// `MyelinConfig` is how an operating point silently stops being
    /// reproducible from the run manifest.
    pub async fn open(
        cfg: &MyelinConfig,
        collection: &str,
        prefetch_limit: Option<u64>,
        rerank_depth: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> anyhow::Result<Self> {
        let mut qdrant_cfg = cfg.qdrant.clone();
        qdrant_cfg.collection = collection.to_string();
        let defaults = RetrieveConfig::default();
        Ok(Self {
            store: QdrantStore::new(&qdrant_cfg)?,
            ledger: Ledger::open(&cfg.ledger).await?,
            embedder: Box::new(RemoteEmbedder::new(
                &cfg.embed.url,
                &cfg.embed.model,
                cfg.embed.dim,
            )?),
            llm: Box::new(OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model)?),
            reranker: CrossEncoder::new(&cfg.rerank.url, &cfg.rerank.model)
                .ok()
                .map(|r| Box::new(r) as Box<dyn Reranker>),
            config: RetrieveConfig {
                prefetch_limit: prefetch_limit.unwrap_or(defaults.prefetch_limit),
                rerank_depth: rerank_depth.unwrap_or(defaults.rerank_depth),
                rerank_factor: rerank_factor.unwrap_or(defaults.rerank_factor),
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

/// Which tools a listener offers, and how a caller addresses a scope on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// Every tool; a scope is a tenant or a store. For operators, pipelines
    /// and benchmarks.
    Full,
    /// For agents. A scope is a store, by name: a tenant string is refused,
    /// because it would reach past the stores the agent was shown. The four
    /// tools of [`AGENT_REMOVED_TOOLS`] are not registered, `remember` writes
    /// only where a store accepts agent writes, and `forget` is soft only.
    Agent,
}

impl Surface {
    fn instructions(self) -> &'static str {
        match self {
            Self::Full => {
                "Agentic memory. `recall` is the fast read path; it never returns more than k \
                 items and every item is individually addressable by record id. A scope is a \
                 tenant or a store: `stores` lists the stores, `define_store` names one."
            }
            Self::Agent => {
                "Agentic memory, addressed by store. Call `stores` first: it lists each store, \
                 what it holds and whether you may write to it. `recall` and `investigate` read \
                 the default stores unless you name others in `stores`. Every hit carries \
                 `store`, the store it came from, and `meta`, the record's own metadata."
            }
        }
    }
}

impl std::fmt::Display for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Full => "full",
            Self::Agent => "agent",
        })
    }
}

/// Not registered on [`Surface::Agent`]: bulk ingest, bulk erasure, the
/// registry's write side and the quarantine queue are an operator's.
pub const AGENT_REMOVED_TOOLS: [&str; 4] =
    ["observe", "forget_scope", "define_store", "review_quarantine"];

/// The most stores one read may name. Every store named is one more
/// retrieval and adds its `k` items to the one cross-encoder pass.
pub const MAX_STORES_PER_READ: usize = 8;

#[derive(Clone)]
pub struct MyelinServer {
    backend: Arc<Backend>,
    surface: Surface,
    tool_router: rmcp::handler::server::tool::ToolRouter<Self>,
}

/// What one retrieval is scoped to: a tenant, and the store that named it
/// when the caller addressed a store.
struct ReadScope {
    tenant: String,
    store: Option<StoreDef>,
}

impl ReadScope {
    fn of_tenant(tenant: String) -> Self {
        Self {
            tenant,
            store: None,
        }
    }

    fn of_store(store: StoreDef) -> Self {
        Self {
            tenant: store.tenant.clone(),
            store: Some(store),
        }
    }

    fn store_name(&self) -> Option<String> {
        self.store.as_ref().map(|s| s.name.clone())
    }

    /// The caller's `dated` when it gave one, else the store's own.
    fn dated(&self, requested: Option<bool>) -> Option<bool> {
        requested.or(self.store.as_ref().map(|s| s.dated))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallParams {
    /// The question, verbatim. Nothing else about the caller's task is
    /// accepted — see `PLAN.md` R6: a memory backend that can see the
    /// benchmark's metadata is disqualified, so there is deliberately no
    /// field here for a question id, a category, or a gold answer.
    pub query: String,
    /// A tenant; the full surface only. Name a tenant or `stores`, not both.
    /// There is no read path that can span tenants (C12): a read of several
    /// stores is one retrieval per store.
    #[serde(default)]
    pub tenant: Option<String>,
    /// Stores to read, by name; the `stores` tool lists them. Empty with no
    /// `tenant` reads every default store.
    #[serde(default)]
    pub stores: Vec<String>,
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
    /// Ask the model which retrieved candidates jointly answer the question and
    /// put those first. Costs one model call per query, so it is an operating
    /// point (R4) and never a `recall` default: `PLAN.md` §7.1 pins the fast
    /// path at "no LLM in the loop".
    #[serde(default)]
    pub select: Option<bool>,
    /// Whether this memory's records carry real event timestamps.
    ///
    /// `false` suppresses all three date mechanisms — `stamp_valid_time`,
    /// `resolve_relative`, `timeline`. LME-V2 needs it: its 85,589 records all
    /// carry the ingest timestamp as `t_valid` because its trajectories are
    /// agent task logs with no dates, so the annotations those mechanisms emit
    /// resolve against a meaningless anchor.
    #[serde(default)]
    pub dated: Option<bool>,
    /// The day the question is asked, `YYYY-MM-DD` (M46). When set, every
    /// `[timeline]` entry also states its distance from this day — "28 days
    /// ago (4 weeks)" — so *how many weeks ago …?* is a lookup rather than
    /// a subtraction. No default: a backend that substituted the wall clock
    /// would emit offsets wrong by the age of the memory. A malformed date
    /// is refused, never ignored.
    #[serde(default)]
    pub as_of: Option<String>,
    /// Allocate `k`'s slots per record kind instead of handing them to one
    /// fused ranking: AgentRunbook-R's published **top-6 events, top-3
    /// notes**, raw states taking the rest
    /// ([`myelin_core::pipeline::compose::KindQuota::AGENTRUNBOOK`]).
    ///
    /// Not tunable from the wire on purpose. The allocation under test is
    /// the paper's, and a tunable one would invite fitting it to the test
    /// set. M34 measured our emitted mix at 10.6% events against their
    /// 31.6%, which is the gap this closes.
    #[serde(default)]
    pub kind_quota: Option<bool>,

    /// Split the question into at most N sub-queries and retrieve for each,
    /// fusing them into the same RRF call as the original (M24).
    ///
    /// One model call per query, so — like `select` — it is an operating
    /// point (R4) and never a `recall` default. `investigate` decomposes
    /// every probe, which is one call per step.
    #[serde(default)]
    pub decompose: Option<usize>,
    /// Conditions on the caller's record `meta`, all of which must hold. Applied
    /// before ranking, so they narrow the candidate set rather than trim the
    /// ranked list. `{"op":"eq","key":K,"value":"str"}` (also matches an array
    /// field containing the value) or `{"op":"range","key":K,"gt"?,"gte"?,"lt"?,
    /// "lte"?: i64}`. A key must match `^[a-z_][a-z0-9_]*$`; a range needs at
    /// least one bound.
    #[serde(default)]
    pub filter: Vec<MetaCondition>,
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
    /// Aligned 1:1 with `items`.
    pub hits: Vec<HitJson>,
    pub tokens: usize,
    /// One per scope read, in the order the scopes were read.
    pub traces: Vec<StoreRecallTrace>,
}

/// One scope's `recall` trace.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StoreRecallTrace {
    /// `None` on a tenant-addressed read.
    pub store: Option<String>,
    pub trace: RecallTraceJson,
}

/// One scope's `investigate` trace.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StoreInvestigateTrace {
    /// `None` on a tenant-addressed read.
    pub store: Option<String>,
    pub trace: InvestigateTrace,
}

/// What a caller needs to address one hit without a second lookup. Sits beside
/// `items`, never inside them: R1 keeps the wire item to `{type, value}`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct HitJson {
    pub record_id: String,
    pub source: String,
    pub score: f32,
    /// The record's caller metadata. `None` for a record without any and for
    /// synthetic items (`[timeline]`, `[notes]`), which have a nil record id.
    pub meta: Option<serde_json::Map<String, serde_json::Value>>,
    /// The store the hit came from. `None` on a tenant-addressed read.
    pub store: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct RecallTraceJson {
    pub dense_hits: usize,
    pub lex_hits: usize,
    pub fused: usize,
    pub reranked: usize,
    pub admitted: usize,
    /// Fused candidates handed to the reranker. `rerank_depth == reranked
    /// == admitted == k` is the signature of a second stage that can only
    /// reorder what it will emit, which is what shipped through M36.
    pub rerank_depth: usize,
    pub embed_ms: u64,
    pub search_ms: u64,
    pub rerank_ms: u64,
    pub total_ms: u64,
    pub top_score: Option<f32>,
    pub abstained: bool,
    /// What the sufficiency selector did, when it ran.
    ///
    /// On the wire because the LongMemEval-V2 harness path is the one
    /// `bench` was before M32: it drives retrieval over MCP and had no way
    /// to tell a working selector from a silent fallback to rank order. A
    /// fully degraded selecting arm emits the unselected arm's evidence
    /// set, so it reads as a clean null for a mechanism that never ran
    /// ([`myelin_core::pipeline::select::Degradation`]). `investigate`
    /// already returns its whole [`InvestigateTrace`], which carries this;
    /// `recall` reported everything about the fusion and nothing about the
    /// one stage that can silently do nothing.
    pub selected: usize,
    pub select_ms: u64,
    pub select_degraded: myelin_core::pipeline::select::Degradation,
}

impl From<RecallTrace> for RecallTraceJson {
    fn from(t: RecallTrace) -> Self {
        Self {
            dense_hits: t.dense_hits,
            lex_hits: t.lex_hits,
            fused: t.fused,
            reranked: t.reranked,
            admitted: t.admitted,
            rerank_depth: t.rerank_depth,
            embed_ms: t.embed_ms as u64,
            search_ms: t.search_ms as u64,
            rerank_ms: t.rerank_ms as u64,
            total_ms: t.total_ms as u64,
            top_score: t.top_score,
            abstained: t.abstained,
            selected: t.selected,
            select_ms: t.select_ms as u64,
            select_degraded: t.select_degraded,
        }
    }
}

#[tool_router]
impl MyelinServer {
    pub fn new(backend: Arc<Backend>, surface: Surface) -> Self {
        let mut tool_router = Self::tool_router();
        if surface == Surface::Agent {
            for name in AGENT_REMOVED_TOOLS {
                tool_router.remove_route(name);
            }
        }
        Self {
            backend,
            surface,
            tool_router,
        }
    }

    #[tool(
        name = "recall",
        description = "Fast hybrid retrieval over the stores named in `stores`, or the default \
                       stores when none is named. BM25 + dense in one round trip, RRF-fused, \
                       cross-encoder reranked, budgeted and deduplicated. No LLM in the loop. \
                       Each hit names the store it came from."
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

        let scopes = self.read_scopes(params.tenant.clone(), &params.stores).await?;
        let merge_reranker = self.merge_reranker(&scopes)?;

        let defaults = Budget::default();
        let budget = Budget {
            k: params.k.unwrap_or(defaults.k),
            tokens: params.budget_tokens.unwrap_or(defaults.tokens),
            max_steps: defaults.max_steps,
        };
        let as_of = parse_as_of(params.as_of.as_deref())?;
        let meta_filter = validated_filter(params.filter.clone())?;

        // One retrieval per scope, each over its own tenant (C12).
        let runs = try_join_all(scopes.iter().map(|scope| {
            let mut filter = ScopeFilter::tenant(&scope.tenant);
            filter.namespace = params.namespace.clone();
            filter.agent = params.agent.clone();
            filter.session = params.session.clone();
            let query = Recall {
                scope: filter,
                text: params.query.clone(),
                budget,
                mode: Mode::Recall,
                kinds: kinds.clone(),
                as_of,
                meta_filter: meta_filter.clone(),
            };
            let mut retriever = self.retriever();
            if params.tau_abstain.is_some() {
                retriever.config.tau_abstain = params.tau_abstain;
            }
            apply_operating_point(
                &mut retriever.config,
                params.select,
                scope.dated(params.dated),
                params.kind_quota,
            );
            retriever.config.decompose = params.decompose;
            async move { retriever.recall(&query).await.map_err(mcp_err) }
        }))
        .await?;

        let mut sets = Vec::with_capacity(runs.len());
        let mut traces = Vec::with_capacity(runs.len());
        for (scope, (evidence, trace)) in scopes.iter().zip(runs) {
            traces.push(StoreRecallTrace {
                store: scope.store_name(),
                trace: trace.into(),
            });
            sets.push(evidence);
        }
        let merged = merge_read(merge_reranker, &params.query, sets, &budget).await?;
        let hits = self.hits_of(&scopes, &merged).await?;
        Ok(Json(RecallResult {
            items: wire_items(&merged.items),
            record_ids: record_ids(&merged.items),
            hits,
            tokens: merged.tokens,
            traces,
        }))
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
        let scopes = self.read_scopes(params.tenant.clone(), &params.stores).await?;
        let merge_reranker = self.merge_reranker(&scopes)?;

        let defaults = Budget::default();
        let budget = Budget {
            k: params.k.unwrap_or(defaults.k),
            tokens: params.budget_tokens.unwrap_or(defaults.tokens),
            // One source of truth: the measured default lives on
            // `InvestigateConfig` with the curve that justifies it.
            max_steps: params
                .max_steps
                .unwrap_or(InvestigateConfig::default().max_steps),
        };
        let as_of = parse_as_of(params.as_of.as_deref())?;
        let meta_filter = validated_filter(params.filter.clone())?;
        // `--premise` implies the gate: the analysis rewrites the statement
        // the gate emits, so one without the other is a silently inert
        // switch — the exact class of failure M12, M14 and M20 each lost a
        // run to.
        let premise = params.premise.unwrap_or(false);
        // M48: the two digest labels are alternative arms. Refused at the
        // wire with the library's own rule, so the caller learns which
        // switch to drop rather than seeing an internal error mid-query.
        myelin_core::pipeline::investigate::digest_label(
            params
                .digest_relevance
                .unwrap_or(InvestigateConfig::default().digest_relevance),
            params
                .digest_role
                .unwrap_or(InvestigateConfig::default().digest_role),
        )
        .map_err(|e| ErrorData::invalid_params(e.to_string(), None))?;
        let cfg = InvestigateConfig {
            select_sufficient: params
                .select
                .unwrap_or(InvestigateConfig::default().select_sufficient),
            select_coverage: params
                .select_coverage
                .unwrap_or(InvestigateConfig::default().select_coverage),
            self_ask: params
                .self_ask
                .unwrap_or(InvestigateConfig::default().self_ask),
            item_digest: params
                .item_digest
                .unwrap_or(InvestigateConfig::default().item_digest),
            digest_dates: params
                .digest_dates
                .unwrap_or(InvestigateConfig::default().digest_dates),
            digest_relevance: params
                .digest_relevance
                .unwrap_or(InvestigateConfig::default().digest_relevance),
            digest_role: params
                .digest_role
                .unwrap_or(InvestigateConfig::default().digest_role),
            rerank_pool: params.pool_rerank.unwrap_or(false),
            premise_analysis: premise,
            abstain_on_insufficient: premise,
            premise_check: params
                .premise_check
                .unwrap_or(InvestigateConfig::default().premise_check),
            answerability_gate: params
                .answerability_gate
                .unwrap_or(InvestigateConfig::default().answerability_gate),
            typed_probes: params.typed_probes.unwrap_or(false),
            ..InvestigateConfig::default()
        };

        // One investigation per scope, each over its own tenant (C12).
        // `select` reaches `InvestigateConfig`, never `RetrieveConfig`: this
        // loop selects once over its accumulated pool, and a per-probe
        // selection underneath it is the arrangement M21 measured at +0.0.
        let runs = try_join_all(scopes.iter().map(|scope| {
            let mut filter = ScopeFilter::tenant(&scope.tenant);
            filter.namespace = params.namespace.clone();
            let query = Recall {
                scope: filter,
                text: params.question.clone(),
                budget,
                mode: Mode::Investigate,
                kinds: None,
                as_of,
                meta_filter: meta_filter.clone(),
            };
            let mut retriever = self.retriever();
            apply_operating_point(
                &mut retriever.config,
                None,
                scope.dated(params.dated),
                params.kind_quota,
            );
            retriever.config.decompose = params.decompose;
            async move {
                Investigator::new(self.backend.llm.as_ref(), &retriever)
                    .with_config(cfg)
                    .investigate(&query)
                    .await
                    .map_err(mcp_err)
            }
        }))
        .await?;

        let mut sets = Vec::with_capacity(runs.len());
        let mut traces = Vec::with_capacity(runs.len());
        let mut queries = Vec::new();
        for (scope, (evidence, trace)) in scopes.iter().zip(runs) {
            queries.extend(evidence.trace.iter().map(|t| t.query.clone()));
            traces.push(StoreInvestigateTrace {
                store: scope.store_name(),
                trace,
            });
            sets.push(evidence);
        }
        let merged = merge_read(merge_reranker, &params.question, sets, &budget).await?;
        let hits = self.hits_of(&scopes, &merged).await?;
        Ok(Json(InvestigateResult {
            items: wire_items(&merged.items),
            record_ids: record_ids(&merged.items),
            hits,
            tokens: merged.tokens,
            queries,
            traces,
        }))
    }

    #[tool(
        name = "trajectories",
        description = "Agent-history retrieval (M62): a controller reads the stored agent \
                       trajectories through bounded tools (summary, grep, read) and returns spans \
                       of states as AgentRunbook-C lays them out, at most 20 states. For memories \
                       built from agent trajectories; slower than investigate."
    )]
    async fn trajectories(
        &self,
        Parameters(params): Parameters<TrajectoriesParams>,
    ) -> Result<Json<TrajectoriesResult>, ErrorData> {
        let started = std::time::Instant::now();
        let target = self.one_scope(params.tenant, params.store).await?;
        let mut scope = ScopeFilter::tenant(&target.tenant);
        scope.namespace = params.namespace.clone();
        let config = TrajectoryAgentConfig {
            max_steps: params
                .max_steps
                .unwrap_or(TrajectoryAgentConfig::default().max_steps),
            ..TrajectoryAgentConfig::default()
        };
        let tools = TrajectoryTools::new(&self.backend.ledger, scope);
        let run = TrajectoryAgent::new(self.backend.llm.as_ref(), tools, config)
            .run(&params.question)
            .await
            .map_err(mcp_err)?;
        let evidence = run.evidence;
        Ok(Json(TrajectoriesResult {
            items: evidence.to_wire(),
            record_ids: evidence
                .items
                .iter()
                .map(|i| i.record_id.to_string())
                .collect(),
            tokens: evidence.tokens,
            queries: evidence
                .trace
                .iter()
                .map(|t| format!("{}: {}", t.action, t.query))
                .collect(),
            trace: TrajectoriesTrace {
                steps: evidence.trace.len(),
                stopped_because: if run.forced { "budget" } else { "answered" }.into(),
                tool_errors: run.tool_errors,
                total_ms: started.elapsed().as_millis(),
            },
        }))
    }

    #[tool(
        name = "remember",
        description = "Write one statement through the full write path: extract, consolidate against \
                       neighbours, apply a four-op delta, index. Idempotent by content. \
                       Set as_profile to assert it verbatim as a durable preference of the user \
                       instead of extracting facts from it; re-asserting a changed preference \
                       supersedes the old one."
    )]
    async fn remember(
        &self,
        Parameters(params): Parameters<RememberParams>,
    ) -> Result<Json<WriteResult>, ErrorData> {
        let target = self.one_scope(params.tenant, params.store).await?;
        self.require_writable(&target)?;
        let scope = Scope::new(
            &target.tenant,
            params.agent.as_deref().unwrap_or("myelin"),
            params.namespace.as_deref().unwrap_or("default"),
        );
        let turn = Turn {
            speaker: params.speaker.unwrap_or_else(|| "user".into()),
            text: params.text,
            at: params.t_valid,
            source: SourceRef::doc(params.source.unwrap_or_else(|| "remember".into())),
            unit: params.unit.unwrap_or_else(|| "remember".into()),
            meta: None,
        };
        if params.as_profile {
            return self.assert_profile(&scope, turn).await;
        }
        self.write(&self.write_path(), &scope, &[turn]).await
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
        let target = self.one_scope(params.tenant, params.store).await?;
        let scope = Scope::new(
            &target.tenant,
            params.agent.as_deref().unwrap_or("myelin"),
            params.namespace.as_deref().unwrap_or("default"),
        );
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
                unit: t.unit.unwrap_or_else(|| unit.clone()),
                meta: t.meta,
            })
            .collect();
        let mut wp = self.write_path();
        // `extract: false` stores each episode verbatim and stops: no model
        // call, no derived facts. The default stays today's full path.
        wp.extract_facts = params.extract.unwrap_or(true);
        self.write(&wp, &scope, &turns).await
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
        let target = self.one_scope(params.tenant, params.store).await?;
        let mut scope = ScopeFilter::tenant(&target.tenant);
        scope.namespace = params.namespace.clone();
        scope.agent = params.agent.clone();
        let limit = params.limit.unwrap_or(20).min(200) as i64;
        let records = self
            .backend
            .ledger
            .visible(&scope, chrono::Utc::now(), limit)
            .await
            .map_err(mcp_err)?;
        Ok(Json(SearchResult {
            records: records.iter().map(RecordStub::from_record).collect(),
        }))
    }

    #[tool(
        name = "profile",
        description = "What this user is known to prefer: their durable dispositions, newest first. \
                       Retrieved by scope, not by relevance — a preference is about the user, not \
                       about the question."
    )]
    async fn profile(
        &self,
        Parameters(params): Parameters<ProfileParams>,
    ) -> Result<Json<ProfileResult>, ErrorData> {
        let target = self.one_scope(params.tenant, params.store).await?;
        let mut scope = ScopeFilter::tenant(&target.tenant);
        scope.namespace = params.namespace.clone();
        scope.agent = params.agent.clone();
        let limit = params.limit.unwrap_or(20).min(200) as i64;
        let records = self
            .backend
            .ledger
            .visible_of_kind(&scope, RecordKind::Profile, chrono::Utc::now(), limit)
            .await
            .map_err(mcp_err)?;
        Ok(Json(ProfileResult {
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
            Some(r) => Some(
                LinkKind::parse(r).map_err(|e| ErrorData::invalid_params(e.to_string(), None))?,
            ),
            None => None,
        };
        // One SQL query for the edges that actually touch this record,
        // instead of `links(namespace)` + a Rust filter. `links` joined only
        // on the *source* record's namespace, so an inbound edge written
        // from another namespace never reached the walk below — and a link
        // is directional metadata about a pair, not a possession of the
        // source's namespace. Hence no namespace argument on this path.
        let links = self
            .backend
            .ledger
            .links_incident(id)
            .await
            .map_err(mcp_err)?;

        let mut out = Vec::new();
        for link in links {
            // Edges are walked in both directions: `supersedes` is only
            // useful if you can ask "what replaced this?" as well as "what
            // did this replace?".
            let other = if link.src == id { link.dst } else { link.src };
            if relation.is_some_and(|r| r != link.relation) {
                continue;
            }
            if let Some(record) = self.backend.ledger.get(other).await.map_err(mcp_err)? {
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
        if self.surface == Surface::Agent {
            self.require_agent_forgettable(id, &params.mode).await?;
        }
        let actor = ActorId::new(params.actor.as_deref().unwrap_or("mcp"));
        let reason = params.reason.unwrap_or_else(|| "forget via mcp".into());

        match params.mode.as_str() {
            "soft" => {
                let applied = self
                    .backend
                    .ledger
                    .apply(
                        &Delta::Delete {
                            target: id,
                            reason: reason.clone(),
                        },
                        &actor,
                    )
                    .await
                    .map_err(mcp_err)?;
                // Report what the ledger actually invalidated, not the id
                // the caller already knows. `Delta::Delete` invalidates one
                // record today, but the tool's contract is "what changed",
                // and echoing the input would keep saying "one" if that ever
                // stopped being true.
                Ok(Json(ForgetResult {
                    mode: "soft".into(),
                    affected: applied.invalidated.iter().map(|u| u.to_string()).collect(),
                }))
            }
            "hard" => {
                // C11 unlearning erases descendants too, so the confirmation
                // is not ceremony: the caller usually cannot see how many
                // records are about to go.
                if !params.confirm.unwrap_or(false) {
                    return Err(ErrorData::invalid_params(
                        "hard deletion erases this record AND every record derived from it; \
                         pass confirm=true"
                            .to_string(),
                        None,
                    ));
                }
                let gone = self
                    .backend
                    .ledger
                    .hard_delete(id, &actor, &reason)
                    .await
                    .map_err(mcp_err)?;
                self.backend
                    .store
                    .delete_points(&gone)
                    .await
                    .map_err(mcp_err)?;
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
        name = "namespaces",
        description = "Live record count per namespace of one tenant, ordered by namespace; one \
                       namespace when given. Soft-forgotten and quarantined records are not \
                       counted, and a namespace with none left is not listed."
    )]
    async fn namespaces(
        &self,
        Parameters(params): Parameters<NamespacesParams>,
    ) -> Result<Json<NamespacesResult>, ErrorData> {
        let target = self.one_scope(params.tenant, params.store).await?;
        let rows = self
            .backend
            .ledger
            .namespace_inventory(&target.tenant, params.namespace.as_deref(), chrono::Utc::now())
            .await
            .map_err(mcp_err)?;
        let total_records = rows.iter().map(|(_, n)| n).sum();
        Ok(Json(NamespacesResult {
            namespaces: rows
                .into_iter()
                .map(|(namespace, records)| NamespaceCount { namespace, records })
                .collect(),
            total_records,
        }))
    }

    #[tool(
        name = "forget_scope",
        description = "Hard-delete every record of one tenant + namespace, except the record ids \
                       in `keep`, with everything derived from them (C11). The namespace is \
                       required so one call cannot erase a whole tenant. Requires confirm=true \
                       and cannot be undone."
    )]
    async fn forget_scope(
        &self,
        Parameters(params): Parameters<ForgetScopeParams>,
    ) -> Result<Json<ForgetScopeResult>, ErrorData> {
        if !params.confirm {
            return Err(ErrorData::invalid_params(
                "forget_scope erases every record in the namespace except `keep`, and \
                 everything derived from them; pass confirm=true"
                    .to_string(),
                None,
            ));
        }
        let keep: std::collections::HashSet<uuid::Uuid> = params
            .keep
            .iter()
            .map(|s| parse_uuid(s))
            .collect::<Result<_, _>>()?;
        let actor = ActorId::new(params.actor.as_deref().unwrap_or("mcp"));
        let reason = params
            .reason
            .unwrap_or_else(|| "forget_scope via mcp".into());

        let target = self.one_scope(params.tenant, params.store).await?;
        let ids = myelin_core::pipeline::forget::forget_scope(
            &self.backend.ledger,
            &self.backend.store,
            &target.tenant,
            &params.namespace,
            &keep,
            &actor,
            &reason,
        )
        .await
        .map_err(mcp_err)?;
        Ok(Json(ForgetScopeResult {
            affected: ids.len(),
            ids: ids.iter().map(|u| u.to_string()).collect(),
        }))
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
            .map_err(mcp_err)?;
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
            .map_err(mcp_err)?
            .ok_or_else(|| ErrorData::invalid_params(format!("no record {id}"), None))?;
        let events = self
            .backend
            .ledger
            .events(Some(id))
            .await
            .map_err(mcp_err)?;
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

    #[tool(
        name = "stores",
        description = "The stores this memory holds: each one's name, what it holds, whether a \
                       read covers it when no store is named, whether `remember` may write to \
                       it, whether its records are dated, and how many records and namespaces \
                       it has. Call this first."
    )]
    async fn stores(&self) -> Result<Json<StoresResult>, ErrorData> {
        let defs = self.backend.ledger.stores().await.map_err(mcp_err)?;
        let now = chrono::Utc::now();
        let mut stores = Vec::with_capacity(defs.len());
        for def in defs {
            let rows = self
                .backend
                .ledger
                .namespace_inventory(&def.tenant, None, now)
                .await
                .map_err(mcp_err)?;
            stores.push(StoreInfo {
                records: rows.iter().map(|(_, n)| n).sum(),
                namespaces: rows.len(),
                name: def.name,
                description: def.description,
                default: def.default_read,
                writable: def.agent_writes,
                dated: def.dated,
            });
        }
        Ok(Json(StoresResult { stores }))
    }

    #[tool(
        name = "define_store",
        description = "Name a tenant as a store, or restate a store's description and flags. \
                       Idempotent. A name already bound to another tenant is refused, and so is \
                       a tenant already bound to another name."
    )]
    async fn define_store(
        &self,
        Parameters(def): Parameters<StoreDef>,
    ) -> Result<Json<StoreDef>, ErrorData> {
        self.backend
            .ledger
            .define_store(&def)
            .await
            .map_err(|e| match e {
                // The one core error whose text is the caller's to read: it
                // names only what the caller sent and the binding it hit.
                myelin_core::MyelinError::StoreDefinition(msg) => invalid(msg),
                other => mcp_err(other),
            })?;
        Ok(Json(def))
    }
}

impl MyelinServer {
    /// One write path for `remember` and `observe`: the difference between
    /// them is how many turns the caller has, not what happens to them.
    async fn write(
        &self,
        wp: &WritePath<'_>,
        scope: &Scope,
        turns: &[Turn],
    ) -> Result<Json<WriteResult>, ErrorData> {
        let stats = wp.insert(scope, turns).await.map_err(mcp_err)?;
        Ok(Json(WriteResult::from_stats(&stats)))
    }

    /// `remember` with `as_profile: true`. Same consolidation, same
    /// supersession, same audit event — only extraction is skipped, because
    /// the caller already wrote the disposition in its final form.
    async fn assert_profile(
        &self,
        scope: &Scope,
        turn: Turn,
    ) -> Result<Json<WriteResult>, ErrorData> {
        let stats = self
            .write_path()
            .assert_one(scope, turn, CandidateKind::Profile)
            .await
            .map_err(mcp_err)?;
        Ok(Json(WriteResult::from_stats(&stats)))
    }

    /// `hits` for a read: per-item id, source, score, the record's `meta` and
    /// the store the item came from. One ledger read per item (`k` is small);
    /// a nil id is a synthetic view with no record behind it.
    async fn hits_of(
        &self,
        scopes: &[ReadScope],
        merged: &MergedEvidence,
    ) -> Result<Vec<HitJson>, ErrorData> {
        let mut hits = Vec::with_capacity(merged.items.len());
        for (item, origin) in merged.items.iter().zip(&merged.origins) {
            let meta = if item.record_id.is_nil() {
                None
            } else {
                self.backend
                    .ledger
                    .get(item.record_id)
                    .await
                    .map_err(mcp_err)?
                    .and_then(|r| r.meta)
            };
            hits.push(HitJson {
                record_id: item.record_id.to_string(),
                source: item.source.doc.clone(),
                score: item.score,
                meta,
                store: scopes.get(*origin).and_then(ReadScope::store_name),
            });
        }
        Ok(hits)
    }

    fn write_path(&self) -> WritePath<'_> {
        WritePath::new(
            self.backend.llm.as_ref(),
            self.backend.embedder.as_ref(),
            &self.backend.store,
            &self.backend.ledger,
        )
    }

    /// Both modes drive the same retriever over the same store (R4).
    fn retriever(&self) -> Retriever<'_> {
        let mut r = Retriever::new(
            self.backend.embedder.as_ref(),
            &self.backend.store,
            &self.backend.ledger,
        )
        .with_config(self.backend.config.clone());
        if let Some(reranker) = self.backend.reranker.as_deref() {
            r = r.with_reranker(reranker);
        }
        // Unconditional, like `with_reranker`: `RetrieveConfig::select_sufficient`
        // defaults false, so a wired client is inert until an operating point asks
        // for it — and wiring it only under the switch is the exact class of failure
        // M12, M14 and M20 each lost a run to.
        r = r.with_llm(self.backend.llm.as_ref());
        r
    }

    /// The scopes a read covers: a tenant (full surface only), the stores
    /// named, or, with neither, every default store.
    async fn read_scopes(
        &self,
        tenant: Option<String>,
        stores: &[String],
    ) -> Result<Vec<ReadScope>, ErrorData> {
        if let Some(tenant) = tenant {
            self.admit_tenant()?;
            if !stores.is_empty() {
                return Err(invalid("name a tenant or stores, not both"));
            }
            return Ok(vec![ReadScope::of_tenant(tenant)]);
        }

        let mut scopes: Vec<ReadScope> = Vec::with_capacity(stores.len());
        if stores.is_empty() {
            let defined = self.backend.ledger.stores().await.map_err(mcp_err)?;
            scopes.extend(
                defined
                    .into_iter()
                    .filter(|s| s.default_read)
                    .map(ReadScope::of_store),
            );
            if scopes.is_empty() {
                return Err(invalid(match self.surface {
                    Surface::Full => "name a tenant or a store; no default store is defined",
                    Surface::Agent => "name a store; no default store is defined",
                }));
            }
        } else {
            for name in stores {
                if scopes.iter().any(|s| s.store_name().as_deref() == Some(name)) {
                    return Err(invalid(format!("store {name:?} is named twice")));
                }
                scopes.push(ReadScope::of_store(self.store_named(name).await?));
            }
        }
        if scopes.len() > MAX_STORES_PER_READ {
            return Err(invalid(format!(
                "a read covers at most {MAX_STORES_PER_READ} stores, not {}; name the ones to read",
                scopes.len()
            )));
        }
        Ok(scopes)
    }

    /// The scope of a tool that addresses exactly one: a tenant (full surface
    /// only) or a store, never both and never neither.
    async fn one_scope(
        &self,
        tenant: Option<String>,
        store: Option<String>,
    ) -> Result<ReadScope, ErrorData> {
        match (tenant, store) {
            (Some(tenant), store) => {
                self.admit_tenant()?;
                if store.is_some() {
                    return Err(invalid("name a tenant or a store, not both"));
                }
                Ok(ReadScope::of_tenant(tenant))
            }
            (None, Some(name)) => Ok(ReadScope::of_store(self.store_named(&name).await?)),
            (None, None) => Err(invalid(match self.surface {
                Surface::Full => "name a tenant or a store",
                Surface::Agent => "name a store; `stores` lists them",
            })),
        }
    }

    fn admit_tenant(&self) -> Result<(), ErrorData> {
        match self.surface {
            Surface::Full => Ok(()),
            Surface::Agent => Err(invalid(
                "this surface addresses stores by name; call stores",
            )),
        }
    }

    async fn store_named(&self, name: &str) -> Result<StoreDef, ErrorData> {
        self.backend
            .ledger
            .store(name)
            .await
            .map_err(mcp_err)?
            .ok_or_else(|| invalid(format!("no store named {name:?}; `stores` lists them")))
    }

    /// The reranker that merges a read of several scopes; `None` for a read
    /// of one, which merges nothing. Asked before any retrieval runs, so a
    /// read that cannot be merged is refused before it is paid for.
    fn merge_reranker(&self, scopes: &[ReadScope]) -> Result<Option<&dyn Reranker>, ErrorData> {
        if scopes.len() < 2 {
            return Ok(None);
        }
        match self.backend.reranker.as_deref() {
            Some(reranker) => Ok(Some(reranker)),
            None => Err(invalid("reading more than one store needs the reranker")),
        }
    }

    /// The agent surface writes only where a store accepts agent writes.
    fn require_writable(&self, target: &ReadScope) -> Result<(), ErrorData> {
        if self.surface == Surface::Full {
            return Ok(());
        }
        match &target.store {
            Some(store) if store.agent_writes => Ok(()),
            Some(store) => Err(invalid(format!(
                "store {:?} does not accept writes from this surface; `stores` marks the ones \
                 that do",
                store.name
            ))),
            None => self.admit_tenant(),
        }
    }

    /// The agent surface's `forget`: soft only, and only a record of a store
    /// that accepts agent writes.
    async fn require_agent_forgettable(&self, id: uuid::Uuid, mode: &str) -> Result<(), ErrorData> {
        if mode != "soft" {
            return Err(invalid(
                "this surface invalidates and keeps history: mode must be \"soft\"",
            ));
        }
        let record = self
            .backend
            .ledger
            .get(id)
            .await
            .map_err(mcp_err)?
            .ok_or_else(|| invalid(format!("no record {id}")))?;
        let stores = self.backend.ledger.stores().await.map_err(mcp_err)?;
        if stores
            .iter()
            .any(|s| s.agent_writes && s.tenant == record.scope.tenant)
        {
            Ok(())
        } else {
            Err(invalid(format!(
                "record {id} is not in a store that accepts writes from this surface"
            )))
        }
    }
}

/// `as_of` from the wire, or an `invalid_params` error. `None` stays `None`:
/// absence is "no anchor", not "today" (see [`Recall::as_of`]). Pinned by
/// `a_malformed_as_of_is_refused_and_an_absent_one_is_no_anchor`.
fn parse_as_of(raw: Option<&str>) -> Result<Option<chrono::NaiveDate>, ErrorData> {
    raw.map(|s| {
        chrono::NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").map_err(|e| {
            ErrorData::invalid_params(format!("as_of must be YYYY-MM-DD, got {s:?}: {e}"), None)
        })
    })
    .transpose()
}

/// The wire filter, or an `invalid_params` error naming what is wrong. Checked
/// here, at the boundary, because a key reaches Qdrant as `meta.{key}` and a
/// range with no bound would match everything while looking like a filter.
fn validated_filter(filter: Vec<MetaCondition>) -> Result<Vec<MetaCondition>, ErrorData> {
    for cond in &filter {
        let (key, bounded) = match cond {
            MetaCondition::Eq { key, .. } => (key, true),
            MetaCondition::Range {
                key,
                gt,
                gte,
                lt,
                lte,
            } => (
                key,
                gt.is_some() || gte.is_some() || lt.is_some() || lte.is_some(),
            ),
        };
        if !is_meta_key(key) {
            return Err(ErrorData::invalid_params(
                format!("filter key {key:?} must match ^[a-z_][a-z0-9_]*$"),
                None,
            ));
        }
        if !bounded {
            return Err(ErrorData::invalid_params(
                format!("range filter on {key:?} needs at least one of gt, gte, lt, lte"),
                None,
            ));
        }
    }
    Ok(filter)
}

/// `^[a-z_][a-z0-9_]*$`
pub fn is_meta_key(key: &str) -> bool {
    let mut chars = key.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase() || c == '_')
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Apply the query-time operating-point overrides to a retrieval config.
///
/// One function, called from both tools, so `recall` and `investigate` cannot
/// drift on what an operating point means. `None` is always "keep the
/// server's configured default" — the same contract `tau_abstain` has.
///
/// Takes the config rather than the `Retriever` that owns it so the decision
/// is testable without an embedder, a live Qdrant and a ledger; the same
/// reason `investigate.rs::gate_insufficient` is a free function.
fn apply_operating_point(
    cfg: &mut RetrieveConfig,
    select: Option<bool>,
    dated: Option<bool>,
    kind_quota: Option<bool>,
) {
    // Absent means "do not override", so only an explicit value moves it —
    // the same contract `dated` has, and the reason M33's adapter defect was
    // a defect: after a default flips, an omitted key stops meaning "off".
    match kind_quota {
        Some(true) => cfg.compose.kind_quota = Some(KindQuota::AGENTRUNBOOK),
        Some(false) => cfg.compose.kind_quota = None,
        None => {}
    }
    if let Some(on) = select {
        cfg.select_sufficient = on;
    }
    // Only an explicit `false` suppresses. `None` means the caller said
    // nothing, and a corpus is dated until someone says it is not.
    if dated == Some(false) {
        cfg.compose.stamp_valid_time = false;
        cfg.compose.resolve_relative = false;
        cfg.compose.timeline = false;
    }
}

/// M62's agent-history retrieval.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TrajectoriesParams {
    pub question: String,
    /// A tenant; the full surface only. Exactly one of `tenant` and `store`.
    #[serde(default)]
    pub tenant: Option<String>,
    /// A store, by name; the `stores` tool lists them.
    #[serde(default)]
    pub store: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    /// Tool calls before the answer is forced.
    #[serde(default)]
    pub max_steps: Option<usize>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TrajectoriesResult {
    pub items: Vec<WireItem>,
    pub record_ids: Vec<String>,
    pub tokens: usize,
    /// Every action the controller took, in order (`grep: Save in t1`).
    pub queries: Vec<String>,
    pub trace: TrajectoriesTrace,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TrajectoriesTrace {
    pub steps: usize,
    /// `answered`, or `budget` when the step or context budget forced it.
    pub stopped_because: String,
    /// Tool misuses and refused answers the model was shown to correct.
    pub tool_errors: usize,
    pub total_ms: u128,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct InvestigateParams {
    pub question: String,
    /// A tenant; the full surface only. Name a tenant or `stores`, not both.
    #[serde(default)]
    pub tenant: Option<String>,
    /// Stores to investigate, by name; the `stores` tool lists them. Empty
    /// with no `tenant` investigates every default store. One investigation
    /// runs per store.
    #[serde(default)]
    pub stores: Vec<String>,
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
    /// Ask the model which of the accumulated pool's records jointly answer
    /// the question and put those first, once, before `compose` truncates to
    /// `k`. One extra model call per query on a path that already pays for
    /// `max_steps` of them.
    #[serde(default)]
    pub select: Option<bool>,
    /// Whether this memory's records carry real event timestamps. `false`
    /// suppresses `stamp_valid_time`, `resolve_relative` and `timeline`; see
    /// [`RecallParams::dated`].
    #[serde(default)]
    pub dated: Option<bool>,
    /// The day the question is asked, `YYYY-MM-DD` (M46). When set, every
    /// `[timeline]` entry also states its distance from this day — "28 days
    /// ago (4 weeks)" — so *how many weeks ago …?* is a lookup rather than
    /// a subtraction. No default: a backend that substituted the wall clock
    /// would emit offsets wrong by the age of the memory. A malformed date
    /// is refused, never ignored.
    #[serde(default)]
    pub as_of: Option<String>,
    /// Allocate `k`'s slots per record kind instead of handing them to one
    /// fused ranking: AgentRunbook-R's published **top-6 events, top-3
    /// notes**, raw states taking the rest
    /// ([`myelin_core::pipeline::compose::KindQuota::AGENTRUNBOOK`]).
    ///
    /// Not tunable from the wire on purpose. The allocation under test is
    /// the paper's, and a tunable one would invite fitting it to the test
    /// set. M34 measured our emitted mix at 10.6% events against their
    /// 31.6%, which is the gap this closes.
    #[serde(default)]
    pub kind_quota: Option<bool>,

    /// Rerank the loop's accumulated pool against the original question
    /// before composing (M23 A2). Off by default; inert without a reranker
    /// wired on the server.
    #[serde(default)]
    pub pool_rerank: Option<bool>,
    /// When the loop stops unsatisfied, emit an explicit premise analysis in
    /// place of the bare insufficiency statement (M23 A3). Implies the
    /// insufficiency gate: without the gate firing there is nothing to
    /// analyse.
    #[serde(default)]
    pub premise: Option<bool>,
    /// Verify what the question assumes against the composed memories and
    /// append a `[premise]` line only when a memory contradicts it (M47).
    /// Silence appends nothing. One model call per query.
    #[serde(default)]
    pub premise_check: Option<bool>,
    /// Let the reflect gate aim each probe at a pool: `raw` | `event` |
    /// `note` (M23 D2). Meaningless until the typed pools exist in the
    /// store; off by default.
    #[serde(default)]
    pub typed_probes: Option<bool>,
    /// Judge whether the composed evidence answers the question and act on a
    /// graded verdict (M36). Replaces the loop's stop reason as the
    /// abstention trigger; `supported` leaves the evidence untouched.
    #[serde(default)]
    pub answerability_gate: Option<bool>,
    /// Ask the sufficiency selector for every needed memory instead of the
    /// fewest (M38). Inert unless `select` is on. The shipped "FEWEST"
    /// instruction costs −5.3 points of complete gold-session coverage on
    /// LongMemEval `multi-session` and −0.0 on every single-gold category.
    #[serde(default)]
    pub select_coverage: Option<bool>,
    /// Decompose the question into follow-ups, answer each from the composed
    /// evidence, and append them as one additive `[notes]` item (M39).
    /// One model call per query; investigate-only.
    #[serde(default)]
    pub self_ask: Option<bool>,
    /// State what every composed memory contributes to the question and
    /// append the contributions as one additive `[notes]` item (M40).
    /// `self_ask` with the entry count fixed by schema. investigate-only.
    #[serde(default)]
    pub item_digest: Option<bool>,
    /// Prefix each digest line with the date of the memory it came from
    /// (M41). Inert unless `item_digest` is on; costs no extra model call.
    #[serde(default)]
    pub digest_dates: Option<bool>,
    /// Drop digest lines for memories the model says do not bear on the
    /// question (M43). Inert unless `item_digest` is on; no extra model call.
    #[serde(default)]
    pub digest_relevance: Option<bool>,
    /// Type each digest entry `answers` / `context` / `irrelevant` and drop
    /// only the last (M48). An alternative to `digest_relevance`; both
    /// together is refused. Inert unless `item_digest` is on.
    #[serde(default)]
    pub digest_role: Option<bool>,
    /// Split each probe into at most N sub-queries and retrieve for each
    /// (M24). One model call per step, on top of the reflect gate's.
    #[serde(default)]
    pub decompose: Option<usize>,
    /// Conditions on record `meta`; see [`RecallParams::filter`].
    #[serde(default)]
    pub filter: Vec<MetaCondition>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct InvestigateResult {
    pub items: Vec<WireItem>,
    pub record_ids: Vec<String>,
    /// Aligned 1:1 with `items`.
    pub hits: Vec<HitJson>,
    pub tokens: usize,
    /// Every search the loop actually issued, in order — the agentic trace
    /// `EVALUATION.md` §9 requires.
    pub queries: Vec<String>,
    /// One per scope investigated, in the order the scopes were read.
    pub traces: Vec<StoreInvestigateTrace>,
}

/// The R1 wire form of a read's items: `{type, value}` and nothing else.
fn wire_items(items: &[EvidenceItem]) -> Vec<WireItem> {
    items
        .iter()
        .map(|i| WireItem {
            kind: i.kind,
            value: i.value.clone(),
        })
        .collect()
}

fn record_ids(items: &[EvidenceItem]) -> Vec<String> {
    items.iter().map(|i| i.record_id.to_string()).collect()
}

/// One evidence set for a read. One scope's evidence is returned as its
/// retrieval composed it; several are pooled and ordered by one cross-encoder
/// pass, because fused rank scores from separate retrievals do not compare
/// ([`merge_scopes`]). `reranker` is [`MyelinServer::merge_reranker`]'s
/// answer for the same scopes.
async fn merge_read(
    reranker: Option<&dyn Reranker>,
    query: &str,
    mut sets: Vec<EvidenceSet>,
    budget: &Budget,
) -> Result<MergedEvidence, ErrorData> {
    match reranker {
        Some(reranker) => merge_scopes(
            reranker,
            query,
            sets.into_iter().map(|set| set.items).collect(),
            budget.k,
            budget.tokens,
        )
        .await
        .map_err(mcp_err),
        None => {
            let only = sets.pop().unwrap_or_default();
            Ok(MergedEvidence {
                origins: vec![0; only.items.len()],
                tokens: only.tokens,
                items: only.items,
            })
        }
    }
}

fn invalid(message: impl Into<String>) -> ErrorData {
    ErrorData::invalid_params(message.into(), None)
}

// `router = self.tool_router` rather than the default: the default form
// calls `Self::tool_router()` and so rebuilds the whole router on every
// tool call. The field is built once in `new`.
#[rmcp::tool_handler(router = self.tool_router)]
impl ServerHandler for MyelinServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("myelin-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(self.surface.instructions())
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

/// Map a core error to an MCP error without telling the caller how the
/// process is wired.
///
/// `MyelinError`'s own `Display` is clean, but the `Qdrant` and `Io` variants
/// forward their inner error verbatim, and that text carries the Qdrant URL
/// or a ledger path. An MCP caller is untrusted by construction (C12), so it
/// gets the variant slug; the operator gets the whole error in the log.
///
/// The one exception is a failed write ([`myelin_core::MyelinError::is_write_failure`]):
/// its message is the failing leg's own cause (the embedder's HTTP status,
/// the index's gRPC status) and says whether anything was left behind. A
/// caller told only `write-failed` cannot tell an over-long input, which it
/// can fix, from a dead index, which it can only retry, and would resend the
/// same document unchanged.
fn mcp_err(e: myelin_core::MyelinError) -> ErrorData {
    tracing::error!(error = %e, "core error");
    if e.is_write_failure() {
        return ErrorData::internal_error(e.to_string(), None);
    }
    ErrorData::internal_error(e.kind_str().to_string(), None)
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
    /// A tenant; the full surface only. Exactly one of `tenant` and `store`.
    #[serde(default)]
    pub tenant: Option<String>,
    /// A store, by name; the `stores` tool lists them and marks the ones
    /// that accept writes.
    #[serde(default)]
    pub store: Option<String>,
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
    /// Assert `text` verbatim as a durable preference of the user, instead
    /// of extracting facts from it.
    ///
    /// Deleting a disposition is the existing `forget` tool; there is
    /// deliberately nothing new for it.
    #[serde(default)]
    pub as_profile: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ObserveTurn {
    pub speaker: String,
    pub text: String,
    #[serde(default)]
    pub at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub source: Option<String>,
    /// This turn's corpus unit. A change of unit always ends an episode, so a
    /// caller that wants one episode per turn gives each turn its own. Defaults
    /// to the call's `unit`.
    #[serde(default)]
    pub unit: Option<String>,
    /// Caller metadata (a JSON object) stored with the episode this turn lands
    /// in — the first turn's `meta` when several share an episode — returned in
    /// `recall`'s `hits` and filterable with `filter`.
    #[serde(default)]
    pub meta: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ObserveParams {
    pub turns: Vec<ObserveTurn>,
    /// Exactly one of `tenant` and `store`.
    #[serde(default)]
    pub tenant: Option<String>,
    /// A store, by name.
    #[serde(default)]
    pub store: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub unit: Option<String>,
    /// Run fact extraction over the episodes (default `true`). `false` stores
    /// each episode verbatim and writes nothing derived from it.
    #[serde(default)]
    pub extract: Option<bool>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct WriteResult {
    /// One id per episode, in draft order — new and already-present alike.
    pub record_ids: Vec<String>,
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
    /// `RecordKind::Profile` records written. The only signal that
    /// `as_profile` did anything, so it is always reported.
    pub profiles: usize,
    pub wall_ms: u128,
}

impl WriteResult {
    fn from_stats(stats: &myelin_core::pipeline::write::WriteStats) -> Self {
        Self {
            record_ids: stats.episode_ids.iter().map(|i| i.to_string()).collect(),
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
            profiles: stats.profiles,
            wall_ms: stats.wall_ms,
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// A tenant; the full surface only. Exactly one of `tenant` and `store`.
    #[serde(default)]
    pub tenant: Option<String>,
    /// A store, by name; the `stores` tool lists them.
    #[serde(default)]
    pub store: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ProfileParams {
    /// A tenant; the full surface only. Exactly one of `tenant` and `store`.
    #[serde(default)]
    pub tenant: Option<String>,
    /// A store, by name; the `stores` tool lists them.
    #[serde(default)]
    pub store: Option<String>,
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
    /// `supersedes` | `supports` | `mentions`.
    #[serde(default)]
    pub relation: Option<String>,
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
pub struct NamespacesParams {
    /// A tenant; the full surface only. Exactly one of `tenant` and `store`.
    #[serde(default)]
    pub tenant: Option<String>,
    /// A store, by name; the `stores` tool lists them.
    #[serde(default)]
    pub store: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NamespaceCount {
    pub namespace: String,
    /// Live records.
    pub records: i64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NamespacesResult {
    pub namespaces: Vec<NamespaceCount>,
    pub total_records: i64,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetScopeParams {
    /// Exactly one of `tenant` and `store`.
    #[serde(default)]
    pub tenant: Option<String>,
    /// A store, by name.
    #[serde(default)]
    pub store: Option<String>,
    /// Required: a missing namespace must not mean "the whole tenant".
    pub namespace: String,
    /// Record ids (UUIDs) to leave in place.
    #[serde(default)]
    pub keep: Vec<String>,
    /// Must be `true`; a call without it is refused.
    #[serde(default)]
    pub confirm: bool,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ForgetScopeResult {
    pub affected: usize,
    /// Every erased record id, descendants included.
    pub ids: Vec<String>,
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
pub struct ProfileResult {
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

/// One store, as the `stores` tool reports it.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StoreInfo {
    pub name: String,
    /// What the store holds.
    pub description: String,
    /// Read when a read names no store.
    pub default: bool,
    /// Whether `remember` may write to it from the agent surface.
    pub writable: bool,
    /// Whether its records carry real event dates.
    pub dated: bool,
    /// Live records.
    pub records: i64,
    /// Namespaces with at least one live record.
    pub namespaces: usize,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StoresResult {
    pub stores: Vec<StoreInfo>,
}

#[cfg(test)]
mod tests {
    /// A filter key reaches Qdrant as `meta.{key}`, and a range with no bound
    /// would match everything while looking like a filter. Both are refused at
    /// the wire, as `invalid_params`, naming the problem.
    #[test]
    fn a_malformed_filter_is_refused_at_the_wire() {
        use myelin_core::model::query::MetaCondition::{Eq, Range};
        let eq = |key: &str| Eq {
            key: key.into(),
            value: "v".into(),
        };
        let range = |key: &str, gte: Option<i64>| Range {
            key: key.into(),
            gt: None,
            gte,
            lt: None,
            lte: None,
        };

        let ok = vec![eq("category"), range("year", Some(2021)), eq("_x9")];
        assert_eq!(super::validated_filter(ok.clone()).unwrap(), ok);
        assert!(super::validated_filter(vec![]).unwrap().is_empty());

        for bad in ["", "Year", "1year", "a-b", "a.b", "a b", "yeår"] {
            let err = super::validated_filter(vec![eq(bad)]).unwrap_err();
            assert!(err.message.contains("^[a-z_][a-z0-9_]*$"), "{bad:?}: {}", err.message);
            assert!(super::validated_filter(vec![range(bad, Some(1))]).is_err(), "{bad:?}");
        }

        let err = super::validated_filter(vec![range("year", None)]).unwrap_err();
        assert!(err.message.contains("at least one"), "{}", err.message);
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    /// The wire shape the consumer sends, parsed as sent.
    #[test]
    fn the_filter_wire_shape_parses_as_documented() {
        use myelin_core::model::query::MetaCondition;
        let p: super::RecallParams = serde_json::from_value(serde_json::json!({
            "query": "q", "tenant": "t",
            "filter": [
                {"op": "eq", "key": "category", "value": "a"},
                {"op": "range", "key": "year", "gte": 2021, "lt": 2024}
            ]
        }))
        .unwrap();
        assert_eq!(
            p.filter,
            vec![
                MetaCondition::Eq { key: "category".into(), value: "a".into() },
                MetaCondition::Range {
                    key: "year".into(),
                    gt: None,
                    gte: Some(2021),
                    lt: Some(2024),
                    lte: None
                },
            ]
        );
        let none: super::RecallParams =
            serde_json::from_value(serde_json::json!({"query": "q", "tenant": "t"})).unwrap();
        assert!(none.filter.is_empty());
        // An unknown op is a parse error, not an ignored condition.
        assert!(serde_json::from_value::<super::RecallParams>(serde_json::json!({
            "query": "q", "tenant": "t", "filter": [{"op": "like", "key": "a", "value": "b"}]
        }))
        .is_err());
    }

    /// `observe` turns carry their own unit and meta; `extract` is optional.
    #[test]
    fn observe_params_accept_unit_meta_and_extract() {
        let p: super::ObserveParams = serde_json::from_value(serde_json::json!({
            "tenant": "t", "extract": false,
            "turns": [{"speaker": "document", "text": "x", "source": "s", "unit": "u1",
                       "meta": {"year": 2020}}]
        }))
        .unwrap();
        assert_eq!(p.extract, Some(false));
        assert_eq!(p.turns[0].unit.as_deref(), Some("u1"));
        assert_eq!(p.turns[0].meta.as_ref().unwrap()["year"], 2020);
        // Old callers: nothing new is required.
        let old: super::ObserveParams = serde_json::from_value(serde_json::json!({
            "tenant": "t", "turns": [{"speaker": "user", "text": "x"}]
        }))
        .unwrap();
        assert_eq!(old.extract, None);
        assert!(old.turns[0].meta.is_none());
    }

    /// Issue 216. A failed `observe` reports the failing leg's own cause to
    /// the caller: an embedder's `413` must not arrive as the bare slug
    /// `write-failed`. Every other core error still reaches a caller as its
    /// slug alone.
    #[test]
    fn a_failed_write_reaches_the_caller_with_its_cause() {
        use myelin_core::error::WriteLeg;
        use myelin_core::MyelinError;

        let embedder = MyelinError::Store("bge-m3: HTTP 413 Payload Too Large: input too long".into());

        let rolled_back = super::mcp_err(MyelinError::write_failed(WriteLeg::Embed, &embedder));
        assert!(rolled_back.message.contains("HTTP 413"), "{}", rolled_back.message);
        assert!(rolled_back.message.contains("embed"), "{}", rolled_back.message);
        assert!(rolled_back.message.contains("nothing was kept"), "{}", rolled_back.message);

        let stuck = super::mcp_err(MyelinError::WriteNotRolledBack {
            leg: WriteLeg::Index,
            cause: "qdrant: unavailable".into(),
            left_behind: "taking back 3 ledger rows failed".into(),
        });
        assert!(stuck.message.contains("unavailable"), "{}", stuck.message);
        assert!(stuck.message.contains("taking back 3 ledger rows failed"), "{}", stuck.message);

        let other = super::mcp_err(MyelinError::Store("sqlite said something about a path".into()));
        assert_eq!(other.message, "store");
    }

    /// `forget_scope` without `confirm` is a refusal, not a deserialization
    /// accident, and `namespace` has no default.
    #[test]
    fn forget_scope_params_require_a_namespace_and_default_confirm_to_false() {
        let p: super::ForgetScopeParams = serde_json::from_value(serde_json::json!({
            "tenant": "t", "namespace": "d1"
        }))
        .unwrap();
        assert!(!p.confirm);
        assert!(p.keep.is_empty());
        assert!(serde_json::from_value::<super::ForgetScopeParams>(serde_json::json!({
            "tenant": "t", "confirm": true
        }))
        .is_err());
    }

    /// M46. A date the backend cannot parse is refused with the error
    /// naming the format; it is never silently dropped, because a dropped
    /// anchor produces a timeline that reads as if the caller sent none.
    #[test]
    fn a_malformed_as_of_is_refused_and_an_absent_one_is_no_anchor() {
        assert_eq!(
            super::parse_as_of(Some("2023-04-01")).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2023, 4, 1)
        );
        assert_eq!(super::parse_as_of(None).unwrap(), None);
        let err = super::parse_as_of(Some("04/01/2023")).unwrap_err();
        assert!(err.message.contains("YYYY-MM-DD"), "{}", err.message);
    }

    use super::*;

    /// `dated: false` is the whole of the undated-corpus gate, and it has to
    /// suppress all three date mechanisms rather than the one that is easiest
    /// to find: LME-V2's 85,589 records all carry the ingest timestamp as
    /// `t_valid`, so a stamp, a resolved relative expression and a `[timeline]`
    /// block are each an annotation against a meaningless anchor.
    #[test]
    fn undated_suppresses_every_date_mechanism_and_nothing_else() {
        let mut cfg = RetrieveConfig::default();
        assert!(cfg.compose.stamp_valid_time, "precondition");
        assert!(cfg.compose.resolve_relative, "precondition");
        assert!(cfg.compose.timeline, "precondition");

        apply_operating_point(&mut cfg, None, Some(false), None);

        assert!(!cfg.compose.stamp_valid_time);
        assert!(!cfg.compose.resolve_relative);
        assert!(!cfg.compose.timeline);
        // Untouched: the gate is about dates, not about width or abstention.
        assert_eq!(cfg.compose.k, RetrieveConfig::default().compose.k);
        assert_eq!(cfg.tau_abstain, RetrieveConfig::default().tau_abstain);
        assert!(!cfg.select_sufficient);
    }

    /// `None` means "keep the server's configured default", the same contract
    /// `tau_abstain` has. A parameter that silently changed behaviour when the
    /// caller omitted it would make every prior run's manifest a lie.
    #[test]
    fn an_absent_operating_point_changes_nothing() {
        let mut cfg = RetrieveConfig::default();
        apply_operating_point(&mut cfg, None, None, None);
        assert_eq!(cfg, RetrieveConfig::default());
    }

    /// `dated: true` is not a way to turn the date mechanisms *on* over a
    /// server that has them off — it is the absence of suppression. Otherwise
    /// an operating point could resurrect a mechanism an ablation arm disabled.
    #[test]
    fn dated_true_does_not_re_enable_a_disabled_mechanism() {
        let mut cfg = RetrieveConfig::default();
        cfg.compose.timeline = false;
        apply_operating_point(&mut cfg, None, Some(true), None);
        assert!(!cfg.compose.timeline);
        assert!(cfg.compose.stamp_valid_time);
    }

    /// Both directions, because an operating point that can only be switched
    /// on cannot produce the paired base arm it has to be measured against.
    #[test]
    fn select_is_switchable_in_both_directions() {
        let mut on = RetrieveConfig::default();
        apply_operating_point(&mut on, Some(true), None, None);
        assert!(on.select_sufficient);

        let mut off = RetrieveConfig {
            select_sufficient: true,
            ..RetrieveConfig::default()
        };
        apply_operating_point(&mut off, Some(false), None, None);
        assert!(!off.select_sufficient);
    }
}
