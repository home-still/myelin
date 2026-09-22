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
use myelin_core::model::delta::Delta;
use myelin_core::model::evidence::{EvidenceSet, WireItem};
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::model::record::{ActorId, LinkKind, RecordKind, Scope, SourceRef};
use myelin_core::pipeline::extract::CandidateKind;
use myelin_core::pipeline::ingest::Turn;
use myelin_core::pipeline::compose::KindQuota;
use myelin_core::pipeline::investigate::{InvestigateConfig, InvestigateTrace, Investigator};
use myelin_core::pipeline::retrieve::{RecallTrace, RetrieveConfig, Retriever};
use myelin_core::pipeline::write::WritePath;
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
            embedder: RemoteEmbedder::new(&cfg.embed.url, &cfg.embed.model, cfg.embed.dim)?,
            llm: OpenAiLlm::new(&cfg.llm.url, &cfg.llm.model)?,
            reranker: CrossEncoder::new(&cfg.rerank.url, &cfg.rerank.model).ok(),
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
        apply_operating_point(
            &mut retriever.config,
            params.select,
            params.dated,
            params.kind_quota,
        );
        retriever.config.decompose = params.decompose;
        let (evidence, trace) = retriever.recall(&query).await.map_err(mcp_err)?;

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

        // `select` reaches `InvestigateConfig`, never `RetrieveConfig`: this
        // loop selects once over its accumulated pool, and a per-probe
        // selection underneath it is the arrangement M21 measured at +0.0.
        let mut retriever = self.retriever();
        apply_operating_point(
            &mut retriever.config,
            None,
            params.dated,
            params.kind_quota,
        );
        retriever.config.decompose = params.decompose;
        // `--premise` implies the gate: the analysis rewrites the statement
        // the gate emits, so one without the other is a silently inert
        // switch — the exact class of failure M12, M14 and M20 each lost a
        // run to.
        let premise = params.premise.unwrap_or(false);
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
            rerank_pool: params.pool_rerank.unwrap_or(false),
            premise_analysis: premise,
            abstain_on_insufficient: premise,
            answerability_gate: params
                .answerability_gate
                .unwrap_or(InvestigateConfig::default().answerability_gate),
            typed_probes: params.typed_probes.unwrap_or(false),
            ..InvestigateConfig::default()
        };
        let (evidence, trace) = Investigator::new(&self.backend.llm, &retriever)
            .with_config(cfg)
            .investigate(&query)
            .await
            .map_err(mcp_err)?;

        Ok(Json(InvestigateResult {
            items: evidence.to_wire(),
            record_ids: evidence
                .items
                .iter()
                .map(|i| i.record_id.to_string())
                .collect(),
            tokens: evidence.tokens,
            queries: evidence.trace.iter().map(|t| t.query.clone()).collect(),
            trace,
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
        let scope = Scope::new(
            &params.tenant,
            params.agent.as_deref().unwrap_or("myelin"),
            params.namespace.as_deref().unwrap_or("default"),
        );
        let turn = Turn {
            speaker: params.speaker.unwrap_or_else(|| "user".into()),
            text: params.text,
            at: params.t_valid,
            source: SourceRef::doc(params.source.unwrap_or_else(|| "remember".into())),
            unit: params.unit.unwrap_or_else(|| "remember".into()),
        };
        if params.as_profile {
            return self.assert_profile(&scope, turn).await;
        }
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
        let scope = Scope::new(
            &params.tenant,
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
        let mut scope = ScopeFilter::tenant(&params.tenant);
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
}

impl MyelinServer {
    /// One write path for `remember` and `observe`: the difference between
    /// them is how many turns the caller has, not what happens to them.
    async fn write(&self, scope: &Scope, turns: &[Turn]) -> Result<Json<WriteResult>, ErrorData> {
        let stats = self
            .write_path()
            .insert(scope, turns)
            .await
            .map_err(mcp_err)?;
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

    fn write_path(&self) -> WritePath<'_> {
        WritePath::new(
            &self.backend.llm,
            &self.backend.embedder,
            &self.backend.store,
            &self.backend.ledger,
        )
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
        // Unconditional, like `with_reranker`: `RetrieveConfig::select_sufficient`
        // defaults false, so a wired client is inert until an operating point asks
        // for it — and wiring it only under the switch is the exact class of failure
        // M12, M14 and M20 each lost a run to.
        r = r.with_llm(&self.backend.llm);
        r
    }
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
    /// Split each probe into at most N sub-queries and retrieve for each
    /// (M24). One model call per step, on top of the reflect gate's.
    #[serde(default)]
    pub decompose: Option<usize>,
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

/// Map a core error to an MCP error without telling the caller how the
/// process is wired.
///
/// `MyelinError`'s own `Display` is clean, but the `Qdrant` and `Io` variants
/// forward their inner error verbatim, and that text carries the Qdrant URL
/// or a ledger path. An MCP caller is untrusted by construction (C12), so it
/// gets the variant slug; the operator gets the whole error in the log.
fn mcp_err(e: myelin_core::MyelinError) -> ErrorData {
    tracing::error!(error = %e, "core error");
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
    /// `RecordKind::Profile` records written. The only signal that
    /// `as_profile` did anything, so it is always reported.
    pub profiles: usize,
    pub wall_ms: u128,
}

impl WriteResult {
    fn from_stats(stats: &myelin_core::pipeline::write::WriteStats) -> Self {
        Self {
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
    pub tenant: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ProfileParams {
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

#[cfg(test)]
mod tests {
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
