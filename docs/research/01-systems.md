# 01 — LLM Agent Memory System Architectures: Literature Evidence Pack

Scope: architecture extractions for every major LLM-agent memory system present in the local corpus (~9,770 papers), to ground the `myelin` design. Every system section carries a DOI; every quoted number carries a page/line pointer (chunk line range from `distill_search`) unless marked `[UNVERIFIED]`.

Shared vocabulary (per design contract): record kinds `episodic | semantic | procedural | working`; pipeline `ingest → extract → consolidate → index → retrieve → rerank → compose → forget`; primitives `dense | sparse/BM25 | hybrid+RRF | rerank | graph-expand | scope-filter`; eval axes `accuracy | token-cost | p95-latency | grounding/provenance | robustness/poisoning`.

---

## 1. MemGPT / Letta (OS-inspired virtual context management)
- **DOI**: 10.48550/arxiv.2310.08560
- **Memory record schema**: Records are message text objects in two external databases — `archival storage` (arbitrary-length text objects, "disk") and `recall storage` (message DB of full conversation history) — plus a `working context` block (fixed-size read/write unstructured text, e.g. key facts/preferences) and a FIFO queue of recent messages. Main context split into system instructions (read-only) + working context + FIFO queue (lines 58–85).
- **Write path**: Self-directed. The LLM (LLM processor) chooses via function calls (`working_context.append`, `recall_storage.search`) when to move memory; a "memory pressure" system alert at ~70% of context window warns the LLM to persist important FIFO content to working context/archival storage; at ~100% the queue manager flushes ~50% of the context window into a recursive summary (lines 83–94). Memory edits and retrieval are entirely LLM-driven, not heuristic.
- **Consolidation**: Recursive summarization of evicted messages; no merge/conflict resolution beyond replacing the head summary; evicted messages persist indefinitely in recall storage (lines 83–94).
- **Index/storage**: External context = archival storage (vector/searchable text DB) + recall storage (message log). No embedding model stated in the architecture section.
- **Retrieval**: Function-call search into recall/archival storage; multi-step retrieval by chaining function calls via `request_heartbeat=true` (lines 58–85). Pagination prevents context overflow (lines 83–94).
- **Forgetting/decay**: Eviction-based (FIFO flush) only; no time-decay formula.
- **Reported results**: OS-inspired hierarchy beats fixed-context baselines in document analysis & multi-session chat (abstract, lines 1–19); on DMR benchmark later reported at **93.4%** with gpt-4-turbo (per 10.48550/arxiv.2501.13956, Table 1, lines 132–203).
- **Limitations**: Fixed operations/structure limit adaptability (criticized by A-MEM, 10.48550/arxiv.2502.12110 lines 32–47); full-conversation context (~115k tokens) can match or exceed it.

---

## 2. Generative Agents (Park et al., 2023)
- **DOI**: 10.48550/arxiv.2304.03442 (ACM 10.1145/3586183.3606763)
- **Memory record schema**: List of memory objects, each `{natural-language description, creation timestamp, last-access timestamp}` (lines 197–209). Two record kinds: **observation** (event directly perceived) and **reflection** (higher-level thought, included in retrieval; forms trees — reflection of reflection) (lines 300–317). Everything is natural language.
- **Write path**: All experiences recorded as observations. Reflections generated **periodically when sum of importance scores of latest events exceeds threshold = 150**; reflection queries the LLM with the 100 most recent records, asks "3 most salient high-level questions", retrieves per question, then extracts insights citing evidence records (lines 300–317). Reflection ~2–3x/day.
- **Consolidation**: Reflection trees (leaves=observations, non-leaves=increasingly abstract thoughts); no dedup/conflict resolution described.
- **Index/storage**: Memory stream + embedding vectors of memory text (for relevance). Uses `gpt3.5-turbo`; no dedicated vector DB stated.
- **Retrieval**: Weighted combination of three signals, all min-max normalized to [0,1]: `score = α_recency·recency + α_importance·importance + α_relevance·relevance`, with **all α = 1**; relevance = cosine similarity between memory and query-memory embedding; importance LLM-scored at creation; top-ranked within context window to prompt (lines 283–300).
- **Forgetting/decay**: No explicit decay formula; recency via time-normalized component only.
- **Reported results**: Ablations show observation/planning/reflection each critical to interview-task believability; common errors = failed retrieval of relevant memories, fabricated embellishments, inherited formal speech (lines 58–68).
- **Limitations**: Vector-only retrieval can surface too many similar fragments; retrieval failure is a primary error source (echoed by HiGMem 10.48550/arxiv.2604.18349 lines 1–18).

---

## 3. MemoryBank / SiliconFriend (Ebbinghaus forgetting)
- **DOI**: 10.48550/arxiv.2305.10250
- **Memory record schema**: Three storage units: (1) detailed chronological multi-turn conversations (timestamps), (2) hierarchical event summaries (daily → global), (3) evolving user-personality portrait (lines 27–44).
- **Write path**: Conversations logged directly; LLM summarizes dialogs into daily-event then global-event summaries via "Summarize the events and key information in the content [dialog/events]"; portrait synthesized from interactions (lines 27–44).
- **Consolidation**: Hierarchical summarization (daily → global); no explicit dedup; portrait continuously refined.
- **Index/storage**: Vector DB via **LangChain + FAISS**; embeddings **MiniLM** (English) / **Text2vec** (Chinese) (lines 82–109).
- **Retrieval**: User's conversation acts as query for dual-tower dense retrieval; output assembled into prompt with relevant memory + global portrait + global event summary (lines 82–109).
- **Forgetting/decay — exact Ebbinghaus formula**: Retention `R = e^(−t/S)`, `R`=fraction retained, `t`=elapsed time, `e≈2.71828`, `S`=memory strength. **S initialized to 1 at first mention; on recall S += 1 and t reset to 0** (lower forget probability) (lines 60–72).
- **Reported results**: SiliconFriend tuned on **38k** psychological conversations; quantitative eval on **10 days** of conversations across **15 virtual users**, **194 probing questions** (lines 19–27, 105–132).
- **Limitations**: Author: "an exploratory and highly simplified memory updating model"; real forgetting varies by person/content (lines 60–72).

---

## 4. A-MEM (Zettelkasten agentic memory)
- **DOI**: 10.48550/arxiv.2502.12110
- **Memory record schema**: Note `m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i}` where `c_i`=content, `t_i`=timestamp, `K_i`=LLM keywords, `G_i`=LLM tags, `X_i`=LLM contextual description, `L_i`=links, and `e_i`=dense embedding of all textual components (lines 47–76).
- **Write path**: LLM-driven, no predefined workflow: `K_i, G_i, X_i ← LLM(c_i ‖ t_i ‖ P_s1)`; `e_i = f_enc[concat(c_i,K_i,G_i,X_i)]` (lines 47–76). ~1,200 tokens/memory-op (lines 319–411).
- **Consolidation/update**: (a) **Link generation**: top-k by cosine `s = e_n·e_j/(|e_n||e_j|)`, then LLM decides: `L_i ← LLM(m_n ‖ M_near ‖ P_s2)`. (b) **Memory evolution**: each neighbor updated `m_j* ← LLM(m_n ‖ M_near∖m_j ‖ m_j ‖ P_s3)`; a memory can live in multiple "boxes" (lines 74–127).
- **Index/storage**: text-encoder dense vectors; no explicit vector DB named.
- **Retrieval**: query embedding via same encoder; cosine over all notes; top-k; linked "box" memories auto-accessed (lines 74–127, 32–47).
- **Forgetting/decay**: none; structure evolves organically.
- **Reported results** (LoCoMo, GPT-4o-mini): A-MEM avg **3.45** vs LoCoMo **2.55** (+35%) and **MemGPT 1.18** (+192%); **~1,200 tokens/op vs 16,900 = 85–93% reduction**; <$0.0003/op; 5.4 s/op GPT-4o-mini, 1.1 s/op Llama3.2-1B local (lines 319–411). Ablation w/o LG&ME MultiHop F1 9.65→27.02 (Table 3).
- **Limitations**: quality bounded by underlying LLM (model-dependent descriptions/links); text-only (lines 504–529).

---

## 5. Mem0 (and Mem0^S graph)
- **DOI**: 10.48550/arxiv.2504.19413
- **Memory record schema**: natural-language facts distilled from interactions. Mem0^S: directed labeled graph `G=(V,E,L)`; nodes=entities `{type, embedding e_v, creation t_v}`; edges=triplets `(v_s, r, v_d)` (lines 27–54).
- **Write path**: incremental extraction per message pair `(m_{t-1}, m_t)` with async conversation summary `S` + recent `m=10` messages; prompt `P=(S,{m_{t-m}..m_{t-2}}, m_{t-1}, m_t)` → extraction LLM `φ(P)` returns fact set Ω (lines 27–38). Hyperparams m=10, s=10; engine GPT-4o-mini (lines 38–54).
- **Consolidation/update — LLM-selected 4-op tool call**: per fact retrieve top-`s` similar; LLM picks **ADD | UPDATE | DELETE | NOOP** (CLASSIFYOPERATION: not-similar→ADD; contradicts→DELETE; augments→UPDATE; else NOOP) (lines 27–38, 728–757).
- **Index/storage**: dense-embedding vector DB; Mem0^S uses **Neo4j** + GPT-4o-mini function calling (lines 38–54).
- **Retrieval** (Mem0^S): dual — entity-centric (locate nodes, explore in/out relationships → subgraph) and semantic-triplet (encode whole query, match triplet text, threshold+rank) (lines 38–54).
- **Forgetting/decay**: deletion only via DELETE op; no time-decay.
- **Reported results** (LOCOMO): Mem0 single-hop **F1 38.72, B1 27.13, J 67.13**; multi-hop **F1 28.64, J 51.15**; Mem0^g temporal **F1 51.55, J 58.13**; Zep edges Mem0 on open-domain **J 76.60 vs 72.93** (lines 293–306).
- **Limitations**: graph variant adds little on single/multi-hop; temporal needs explicit timestamps (OpenAI baseline <15% temporal) (lines 293–306).

---

## 6. Zep / Graphiti (temporal KG memory layer)
- **DOI**: 10.48550/arxiv.2501.13956
- **Memory record schema**: KG `G=(N,E,φ)` three subgraphs: **Episode** (raw messages, non-lossy), **Semantic Entity** (entities+facts), **Community** (clusters). Facts = predicate edges; same fact repeatable across entity pairs (hyper-edges). Every fact carries **bi-temporal** stamps `t_created`, `t_expired` (T' transactional) + `t_valid`, `t_invalid` (T validity) (lines 32–49, 64–87).
- **Write path**: ingestion per message + last `n=4` messages (2 turns); speaker auto-entity; reflexion-inspired reflection reduces hallucination; entities embedded 1024-dim cosine + full-text → LLM **entity resolution** dedup; facts embedded, **edge dedup constrained to same entity-pair**; graph writes via predefined Cypher (not LLM SQL) (lines 49–64, 64–87; appendix 382–461).
- **Consolidation/update**: **temporal extraction** (absolute+relative via `t_ref`); **edge invalidation**: new edge invalidates overlapping contradictory edges via `t_invalid = t_valid(new)`; transactionally new info wins (lines 64–87).
- **Index/storage**: **BGE-m3** (BAAI) embeddings + reranking; gpt-4o-mini-2024-07-18 graph construction (lines 132–203). Community detection = **label propagation** (not Leiden) with dynamic single-step extension (new node → plurality-neighbor community); periodic refresh; community names embedded for cosine search (lines 84–103).
- **Retrieval** = `f: S→S`, three steps: **Search φ** → **Reranker ρ** (frequency-aware, node-distance from centroid, cross-encoder LLM) → **Constructor χ** (formats facts+t_valid/t_invalid, entity name+summary, community summary) (lines 84–103).
- **Forgetting/decay**: invalidation marks edges invalid (not physical delete); no decay formula.
- **Reported results**: **DMR (gpt-4-turbo): Zep 94.8% vs MemGPT 93.4% vs recursive-summarization 35.3%**; gpt-4o-mini 98.2% (Table 1, lines 132–203). **LongMemEval_s (~115k tokens): Zep gpt-4o-mini 63.8%/3.20 s/1.6k vs Full-context 55.4%/31.3 s/115k** (+15.2% acc, ~90% latency cut, 98.6% fewer tokens); gpt-4o **71.2% vs 60.2%** (+18.5%) (Table, lines 217–335).
- **Limitations**: DMR 60-message convs fit in context; benchmark criticized (lines 203–220); entity-centric indexing loses context (HippoRAG-2 critique, 2502.14802 lines 55–70).

---

## 7. HippoRAG (hippocampal-index PPR)
- **DOI**: 10.48550/arxiv.2405.14831
- **Memory record schema**: schemaless open KG ("hippocampal index"): nodes=noun phrases `N`, edges=relations `E` (LLM OpenIE), synonymy edges `E'` when cosine of entity embeddings > threshold `τ`; passage-occurrence matrix `P` (|N|×|P|) (lines 58–67).
- **Write path**: offline: LLM 1-shot OpenIE extracts named entities, then triples (two-step); encoder `M` adds synonymy edges; builds `P` (lines 58–67).
- **Consolidation**: synonymy-edge addition only (no training); no deletion — continually addable.
- **Index/storage**: KG (LLM-built) + retrieval encoder; best backbone ColBERTv2 (lines 278–326).
- **Retrieval — exact PPR**: query → LLM extracts `C_q`; seed nodes by argmax cosine `r_i = e_k, k = argmax_j cos(M(c_i),M(e_j))`; run **Personalized PageRank (PPR)** with personalized vector `n⃗` having **equal probability on each query node, zero elsewhere**; multiply query-node probs by **node specificity `s_i = |P_i|^{-1}`** (local IDF substitute) before PPR; passage score `p⃗' = n⃗'·P`; rank (lines 58–79). PPR damping factor/α **not stated in corpus** → `[UNVERIFIED]`.
- **Forgetting/decay**: none; continual addition.
- **Reported results** (Table 4 QA): HippoRAG+ColBERTv2 vs ColBERTv2: MuSiQue EM **19.2 vs 15.5**, 2Wiki **46.6 vs 33.4**, avg **35.9 vs 30.8**; IRCoT+HippoRAG avg 38.4. Online retrieval **10–30× cheaper, 6–13× faster** than IRCoT (abstract + lines 278–326). All-recall AR@5 2Wiki **75.7 vs 25.1** (Table 6, lines 372–428).
- **Limitations**: no component fine-tuning; errors dominated by NER/OpenIE then graph search; OpenIE inconsistent on long docs; scale unvalidated (lines 482–503).

---

## 8. HippoRAG-2
- **DOI**: 10.48550/arxiv.2502.14802
- **Memory record schema**: open KG with **two node types — phrase nodes (triples) and passage nodes** integrated in one KG (fixes entity-centric context loss); LLM OpenIE triples (schemaless) (lines 55–70).
- **Write path / index**: same offline triple extraction + synonymy edges; phrase KG combined with passages (conceptual+contextual) (lines 55–70).
- **Consolidation**: synonym links; **triple filtering** (recognition-memory) — LLM filters top triples before seed selection (Figure 2, lines 55–70).
- **Retrieval**: embedding scores **both passages and triples** → seed nodes of both → triple filter → PPR; **passage-node reset probability multiplied by weight, default 0.05** (lines 495–542).
- **Reported results**: beats NV-Embed-v2 by **+5.0% (MuSiQue) / +13.9% (2Wiki) Recall@5**; query-to-triple beats NER-to-node by **+12.5% avg Recall@5** (lines 495–542). Error analysis: 26% 2-hop, 41% 3-hop, 33% 4-hop; triple-filter + graph-search are the two main error sources (lines 717–756).
- **Limitations**: recognition filter can empty candidates (18% zero-triples after); filter+search dominate errors (lines 717–756).

---

## 9. MIRIX (multi-agent compositional memory)
- **DOI**: 10.48550/arxiv.2507.07957
- **Memory record schema**: six components — **Core** (agent persona + human block; rewrite >90% capacity), **Episodic** (`event_type, summary, details, actor, timestamp`), **Semantic** (`name, summary, details, source`), **Procedural** (`entry_type, description, steps`), **Resource** (`title, summary, resource_type, content`), **Knowledge Vault** (verbatim sensitive facts) (lines 83–90).
- **Write path**: six dedicated **Memory Managers** + **Meta Memory Manager** for routing; agent generates a **topic before** answering (Active Retrieval); multiple retrieval tools (lines 16–20).
- **Consolidation**: controlled rewrite of core memory above 90% capacity (lines 83–90).
- **Index/storage**: hierarchical structured stores; hybrid on-device/cloud (Knowledge Vault local, Resource offloaded) (lines 47–67).
- **Retrieval**: Active Retrieval + Meta Memory Manager routing + multiple retrieval tools (lines 16–20).
- **Reported results** (LOCOMO LLM-judge %): MIRIX avg **85.38** vs LangMem 78.05, Zep 79.09, Mem0 62.47; **vs RAG +35% accuracy with 99.9% storage reduction; vs long-context Gemini +410% accuracy with 93.3% storage reduction** (lines 220–326).
- **Limitations**: flat systems (Letta/Mem0/ChatGPT) lack compositional structure + multimodal support (lines 16–20).

---

## 10. MemoryOS (OS-style three-tier heat)
- **DOI**: 10.48550/arxiv.2506.06326
- **Memory record schema**: **STM** dialogue pages (fixed FIFO queue), **MTM** recurring-topic segments, **LPM** user/agent preferences (User Traits 90-dim across 3 categories; User KB & Agent Traits fixed-size FIFO of 100) (lines 96–127).
- **Write path**: pages appended to STM; FIFO STM→MTM when full (lines 96–127).
- **Consolidation/update**: MTM deletion + MTM→LPM by **Heat score**: `Heat = α·N_visit + β·L_interaction + γ·R_recency`, `R_recency = exp(−Δt/μ)` with `μ = 1e+7` s. Heat ≥ `τ=5` promoted to LPM; after transition `L_interaction` reset; lowest-heat evicted over capacity (lines 96–127).
- **Index/storage**: segmented-paging per topic; no embedding model stated.
- **Retrieval**: STM=all; MTM two-stage (segment semantic relevance → pages); LPM persona; combined (lines 96–127).
- **Forgetting/decay**: heat-based eviction (formula above).
- **Reported results**: validated on long-conversation benchmarks (correctness + coherence, abstract lines 27–41).
- **Limitations**: fixed three-tier hierarchy; heat coefficients must be tuned.

---

## 11. EverMemOS (engeram-inspired lifecycle OS)
- **DOI**: 10.48550/arxiv.2601.02163
- **Memory record schema**: **MemCells** (episodic traces + atomic facts + time-bounded foresight) organized into **MemScenes** (thematic consensus + stable user profiles) (abstract lines 1–20).
- **Write path**: three phases — **Episodic Trace Formation** (dialogue → MemCells), **Semantic Consolidation** (MemCells → MemScenes), **Reconstructive Recollection** (scene-guided agentic retrieval, necessity-and-sufficiency) (lines 20–39).
- **Consolidation**: MemScene consolidation + coherent user-state (conflict detection via coherent representation) (lines 20–39).
- **Index/storage**: MemScene hierarchical; not pure vector store.
- **Retrieval**: necessity-and-sufficiency-driven agentic (limits context) (lines 20–39).
- **Reported results** (GPT-4.1-mini): **+9.2% LoCoMo, +6.7% LongMemEval** overall; SOTA on memory-augmented reasoning, strongest multi-hop & temporal (lines 1–20).
- **Limitations**: text-only; LLM ops add latency/cost (caching/batch/async mitigations); benchmarks lack ultra-long timelines (lines 491–524).

---

## 12. Nemori (adaptive distillation via prediction error)
- **DOI**: 10.48550/arxiv.2508.03341
- **Memory record schema**: episodic DB `D_e` (narrative + cue per episode) + semantic DB `D_s` (insights); management-agnostic distillation layer (Table 1, lines 17–121).
- **Write path**: partition buffer into episodes; generate `(Narrative, cue)`; embed `v=f_emb(cue‖N)`; merge-or-insert `D_e`; evoke context, synthesize anticipatory schema `P̂`, then distill insights `K←f_LLM(P_dis‖P‖P̂)` (prediction error) (Algorithm 1, lines 1071–1134).
- **Consolidation**: merge-or-insert episodes; native consolidation new/merge/conflict-resolution (lines 1071–1134).
- **Index/storage**: embeddings of `(cue‖narrative)` + insight embeddings.
- **Retrieval**: query embedding; search `D_e` (narratives + paragraphs) and `D_s`, combine into answer (Algorithm 2, lines 1071–1134).
- **Forgetting/decay**: none (agnostic).
- **Reported results** (LongMemEval_s): NEMORI avg **64.2 vs Full-Context 55.0 (gpt-4o-mini)**; **74.6 vs 65.6 (gpt-4o)** at **95–96% fewer tokens** (~3.7–4.8k vs 101k) (Table 8, lines 923–1014). Integrated into A-MEM & MemoryOS: **45–64% storage reduction** (lines 119–121).
- **Limitations**: naive management/retrieval (bottleneck for sophisticated reasoning); conceptual interfaces, case-by-case integration (lines 923–1014).

---

## 13. G-Memory (MAS hierarchical: insight/query/interaction graphs)
- **DOI**: 10.48550/arxiv.2506.07398
- **Memory record schema**: three-tier: **Insight Graph** `((κ_k, Ω_k))`, **Query Graph** `((Q_i, Ψ_i, G_inter))` status Failed/Resolved, **Interaction Graph** utterances `(A_i, m_i)` temporal edges (lines 71–104).
- **Write path**: after task completion all three levels updated agentically — distilled insights, enriched query records, trajectories + associations (lines 27–46).
- **Consolidation**: institutionalization of group knowledge via feedback (lines 71–104).
- **Index/storage**: graphs; MiniLM embeddings for query-graph cosine (lines 101–131).
- **Retrieval**: coarse query-graph cosine → **1-hop expansion** → **bi-directional traversal** upward query→insight (supporting-query intersection), downward query→interaction (LLM sparsifier extracts core subgraph) (lines 101–131).
- **Reported results**: **+20.89% embodied-action success, +10.12% knowledge-QA accuracy** without framework modification; comparable-or-lower token usage (lines 1–17).
- **Limitations**: MAS memory oversimplified in baselines; memory could amplify incorrect reasoning if LLM compromised (lines 664+).

---

## 14. GAM (hierarchical graph, encoding/consolidation decoupling)
- **DOI**: 10.48550/arxiv.2604.12285
- **Memory record schema**: `H_t = {G_topic, G_event, S_arch, E_cross}` — **Topic Associative Network** (global), **Event Progression Graph** (buffer), archived graphs + cross-links (lines 50–81).
- **Write path / consolidation**: decouples rapid perception (event buffer) from stable retention; consolidation triggered by **semantic divergence** `b_t = I(Δ(G_event, G_topic) > ε)`; merge only onto semantically-complete units (write isolation) (lines 33–55, 50–81).
- **Index/storage**: graph-based; cross-encoder base semantic score.
- **Retrieval**: graph-guided **multi-factor re-ranking**: `Score(v,q) = P_sem(v|q)·Π_k β_k^{I_k(v,q)}`, factors `β_time, β_conf, β_role` (>1); β robust over 1.0–2.0 (lines 147–168).
- **Reported results**: vs AriGraph on LoCoMo (Qwen2.5-14B): MultiHop F1 **33.32 vs 29.40**, avg F1 **40.38 vs 24.44**; GPT-4o-mini avg F1 **43.14 vs 24.20** (Tables 7, lines 1339–1451). Cost: 932 input tokens/session, 0.63 s latency/session (Table 9).
- **Limitations**: decoupling adds memory-lifecycle machinery; benchmark scope limited (LoCoMo, LongDialQA).

---

## 15. H-MEM (hierarchical memory routing)
- **DOI**: 10.18653/v1_2026.eacl-long.15 (corpus entry)
- **Memory record schema**: four layers — **Domain, Category, Memory Trace, Episode** (Figure 2, lines 24–41).
- **Write path**: LLM (DeepSeek-R1-8B) for analysis/info extraction; **memory weights** attached to top-k + user profile (lines 79–94).
- **Consolidation/update**: memory strength adjusted by **LLM-generated feedback weight** — external memory weights only, NO gradient/parameter update of base model (lines 79–94).
- **Index/storage**: **BERT encoder**; **FAISS**; top-k=10 (lines 79–94).
- **Retrieval**: encode question → cosine over memory → top-k + user profile → attach weights (confidence) → LLM (lines 79–94).
- **Reported results**: beats LoCoMo/ReadAgent/MemoryBank/MemGPT/A-MEM across five LoCoMo task types (F1 + BLEU-1) (lines 24–41).
- **Limitations**: hierarchical routing must be tuned per domain; relies on LLM-graded feedback quality.

---

## 16. HiGMem (hierarchical event-turn, LLM-guided retrieval)
- **DOI**: 10.48550/arxiv.2604.18349
- **Memory record schema**: two-level — **event layer** (semantic anchors/summaries) + **turn layer** (fine-grained evidence) (lines 18–33).
- **Write path**: LLM builds event summaries as anchors + retains turns (lines 18–33).
- **Consolidation**: hierarchical abstraction by level; no conflict resolution described.
- **Index/storage**: all-MiniLM-L6-v2 both stages; k_turn=10, k_event=10 (lines 99–163).
- **Retrieval**: LLM inspects event summaries (cheap anchors), then **predicts which turns to read** (LLM-guided pruning) instead of returning all similar fragments (lines 18–33).
- **Reported results** (LoCoMo10, GPT-4o-mini): best F1 4/5 categories; vs A-Mem: avg turns **8.09 vs 99.84** (order-of-magnitude fewer), Precision@K **0.1909 vs 0.0101**, Recall@K **0.7241 vs 0.7502** comparable; adversarial F1 **0.78 vs 0.54** (lines 99–163).
- **Limitations**: extra LLM calls; best with cheap memory-model + expensive answer-model; weaker multi-party (DialSim) (lines 254–304).

---

## 17. MOOM (competition-inhibition forgetting, role-play)
- **DOI**: 10.48550/arxiv.2509.11860
- **Memory record schema**: dual-branch — **NSB** (Narrative Summarization, hierarchical, `θ_1=6` turns, `θ_2=θ_3=5`) + **PCB** (Persona Construction, LLM persona values for fixed keys) (lines 53–87).
- **Write path**: hierarchical dialog compression (turns → level-1 units → LLM summaries) + persona snapshots with key-appropriate merge (lines 53–87).
- **Consolidation**: three merge strategies (Rule/Embedding/LLM); LLM-based merging best under capacity constraints (lines 588–613).
- **Forgetting/decay**: **competition-inhibition** theory-based forgetting constrains growth (lines 17–38); exact numeric decay expression `[UNVERIFIED]` (described qualitatively in corpus hits).
- **Reported results** (ZH-4O Qwen1.5-7B QA precision): MOOM **0.832** vs NSB-only 0.693, PCB-only 0.752; vs vanilla 0.607, InfLLM 0.570, RAPTOR 0.626, HippoRAG2 0.730 (Table 3 area, lines 306–403). Constrained 6k memory → higher QA precision than unlimited (inhibition suppresses noise) (lines 588–613).
- **Limitations**: ZH-4O Chinese-only; annotation bias; 7B fine-tuned on GPT-4 output only (lines 401–422).

---

## 18. Collaborative Memory (multi-user, dynamic ACL)
- **DOI**: 10.48550/arxiv.2505.18279
- **Memory record schema**: **memory fragments** = LLM-generated key–value pairs with **provenance** (agents, resources, creation time); split **private** (user tiers) + **shared** (cross-user) (lines 111–139).
- **Write path**: coordinator routes query; agents respond; intermediate responses written via `π^write/private` and `π^write/shared` prompt transformations (private=standalone KV; shared=universal, user details stripped) (lines 322–403, 44–72).
- **Consolidation**: no merge; fragments accumulate; **dynamic access graphs** `G_UA(t) ⊆ U×A`, `G_AR(t) ⊆ A×R` evolve (revoke/grant) (lines 44–72).
- **Index/storage**: embeddings **text-embedding-3-large**; fragments in storage table (lines 111–139).
- **Retrieval**: top-`k_user` user tier + top-`k_cross` shared tier (both default 10) satisfying provenance; cosine between subquery embedding and fragment keys (lines 111–139, 322–403).
- **Reported results** (MULTIHOP-RAG: 609 articles ~2,046 tokens, 2,556 multi-hop queries): shared memory lowers resource utilization vs single-user baselines while maintaining strict privacy compliance (lines 206–239).
- **Limitations**: policy instantiation simple-read/transformation-write; requires provenance-aware fragment storage.

---

## 19. Memory-mechanism survey (taxonomy)
- **DOI**: 10.48550/arxiv.2404.13501
- **Architecture view**: formalizes memory module; three perspectives — **sources** (inside-trial/cross-trial/external), **forms** (textual, tabular, graph, parametric), **operations** (write/read/management) (Section 5, lines ~240–643).
- **Write/read/management**: write = raw or summary (TiM relation extraction, SCM controller, MemGPT self-directed; Table 1 sources: MemoryBank, TiM, SCM, Voyager, MemoGPT, Generative Agents, etc.) (lines 368–441, 561–582). Management = reflect, merge redundant, forget early (MemoryBank, Voyager, GA reflection, GITM) (lines 561–582).
- **Evaluation**: direct (subjective/objective) vs indirect (end-to-end agent tasks) (lines 616–643).
- **Key takeaways**: memory not optional for agents (cognitive basis + evolution + apps) (lines ~240); textual form mainstream (interpretable, fast) (lines 490–513).
- **Limitations flagged**: memory overlap/temporality → need forgetting; humanoid agents need psychological fidelity/knowledge boundaries (lines 921–952).

---

## 20. Episodic-memory position paper
- **DOI**: 10.48550/arxiv.2502.06975
- **Position/architecture**: external episodic memory bridges **parametric** (weights) and **in-context** (context window): (a) **consolidation** episodes → parametric (capacity + generalization + learning before forgetting), (b) **encoding** in-context → external, (c) **retrieval** external → in-context reinstatement (Figure 1, lines 11–26).
- **Five properties**: long-term storage, explicit reasoning, single-shot learning, instance-specific content, contextualized content (lines 11–26).
- **Motivating numbers**: Linux case — 40M+ lines + decades of context (lines 1–13).
- **Design implication**: constant per-token cost + stable/improving performance; consolidation offloads to parametric memory, requiring forgetting on external store (lines 285–320).
- **Limitations/alternative views**: scenarios where episodic memory unnecessary (Section 5) (lines 11–26).

---

## 21. SECOM (Second-order Conflict-free Consolidation)
- **Status**: **NOT FOUND in local corpus.** Searched "SECOM memory consolidation second-order conflicts", "SECOM scalable consolidation memory integration", "SECom memory conflict second-order SCAM scalable", external `paper_search`. No dedicated SECOM paper indexed; only generic conflict-resolution references. → SECOM architecture **absent** from corpus; not fabricated. Design implications below drawn from present conflict mechanisms: Graphiti edge invalidation (2501.13956), Mem0 ADD/UPDATE/DELETE (2504.19413), GAM semantic-divergence consolidation (2604.12285).

---

# Cross-system synthesis

## Recurring design choices
1. **Write path is LLM-driven, not heuristic** — MemGPT, A-MEM, Mem0 (LLM picks op), Zep, G-Memory, MIRIX, EverMemOS all route memory decisions through an LLM. Survey (2404.13501) catalogues heuristic variants; NEMORI (2508.03341) argues heuristics (importance/topic/fact templates) inject bias → prefers prediction-error distillation. **Contested**: LLM-mediated writes are accurate but token/latency-expensive (2601.02163 limitation; HiGMem trade, 2604.18349).
2. **Hierarchical / multi-tier storage** — MemGPT, MemoryOS (heat), MIRIX (6 comps), G-Memory (3 graphs), GAM (buffer+topic), H-MEM (4 layers), HiGMem (event+turn).
3. **Graph memory for multi-hop** — HippoRAG(-2), Zep, Mem0^S, A-MEM, G-Memory, GAM; PPR/graph traversal is the recurring multi-hop primitive.
4. **Two-tier private/shared or episodic/semantic split** — Zep, Collaborative Memory (ACL), NEMORI, MIRIX.
5. **Consolidation by semantic-completeness trigger** — GA (importance-sum >150), GAM (divergence b_t), MemoryBank (daily→global), MOOM (turn thresholds), MemGPT (70%/100% pressure).

## Contested choices
- **Value of graph structure**: Mem0 finds graph variant hurts single/multi-hop (2504.19413: Mem0^g < Mem0 on SH/MH); Zep's open-domain edge small (J 76.60 vs 72.93). GAM vs AriGraph shows massive graph gains (43.14 vs 24.20 avg F1). → Graphs help big on temporal/integration, hurt/neutral on simple single-hop — task-dependent, not universally load-bearing.
- **Flat vs hierarchical retrieval**: A-MEM 99.84 turns, Recall@K 0.75; HiGMem 8.09 turns, Recall@K 0.72 → hierarchy gives comparable recall at ~1/12 the turns (2604.18349). But HiGMem Temporal F1 slightly below A-MEM — hierarchy can weaken chronological cues.
- **LLM-everywhere vs lightweight/heuristic**: NEMORI (distillation over designer-heuristic, −95% tokens, higher accuracy) vs MemGPT/Mem0; MOOM keeps LLM for narrative/persona but cheap rule/embedding merge first.
- **Explicit forgetting**: MemoryBank (Ebbinghaus) and MOOM (competition-inhibition) vs MemGPT/Zep (eviction/invalidation only) vs NEMORI (distillation as implicit forgetting). No consensus curve; MOOM shows constrained capacity beats unlimited QA precision via noise suppression (2509.11860 lines 588–613).

## Empirically load-bearing choices (with numbers)
1. **LLM-selected 4-op delta (Add/Update/Delete/Noop)** is the most reproduced write/consolidation primitive (Mem0; MOOM merge 2509.11860; GAM divergence 2604.12285). Converts factual conflicts into explicit DELETE (temporal consistency) — the mechanism absent systems fail.
2. **Semantic compression → token-cost dominates**: Zep −98.6% tokens at +15–18% acc and −90% latency (2501.13956); NEMORI −95–96% tokens at higher avg acc (74.6 vs 65.6 gpt-4o, 2508.03341); HiGMem 99.84→8.09 turns (2604.18349). → token-cost and p95-latency are the deciding axes; accuracy follows with the right compact evidence set.
3. **LLM-built schemaless KG + PPR is load-bearing for multi-hop**: HippoRAG 2Wiki EM 33.4→46.6, AR@5 25.1→75.7, single-step 10–30× cheaper than IRCoT (2505.14831); HippoRAG-2 dual-node + 0.05 passage reset + recognition filter (+5/+13.9 Recall@5 vs NV-Embed-v2). But extraction (NER/OpenIE/triple-filter) is the error bottleneck (26/41/33% for 2/3/4-hop) — invest in extraction correctness first.
4. **Grounding/provenance must be explicit** — Zep bi-temporal validity + episode links for citation (2501.13956); SSGM argues stale/adversarial poisoning needs ACL + decay gate `w(Δτ)=exp(−(Δτ/η)^α)` (2603.11768). Provenance is the least-popular-but-most-critical design gap.
5. **Robustness/poisoning largely unmeasured** — only Collaborative Memory (ACL graphs) and SSGM (formal, no implementation) address it; self-writing memory is a poisoning surface (G-Memory impact statement). No standard benchmark in the surveyed corpus.

### Design implications for `myelin` (Rust backend)
- Adopt **LLM-selected 4-op updates** with explicit per-fact timestamps; keep provenance (source episode/turn) for grounding + invalidation.
- Make **schemaless KG + PPR** the multi-hop path, but gate retrieval on **hybrid + scope-filter + rerank** (Zep's search→rerank→construct is the strongest single retrieval blueprint).
- Engineer for **token-cost and p95-latency**, not raw accuracy alone — every production paper wins on context compression and latency (Zep 3.2 s, NEMORI 95% fewer tokens, SwiftMem 10.8 ms search 2601.08160).
- Prefer a **lightweight encoder for memory ops + LLM only for extraction/consolidation decisions** (Mem0 gpt-4o-mini, Zep BGE-m3, H-MEM BERT/FAISS) to keep compute local.
- **Explicit forgetting with a validated decay curve** (`R=e^(−t/S)`, S+=1 on recall) unless noise suppression (MOOM) argues bounded capacity + interference-based pruning.

---

### Verification notes
- All numbers carry chunk line ranges from `distill_search` on the cited doc_id; page field where populated is given or noted.
- `[UNVERIFIED]`: HippoRAG PPR damping factor/α (not stated in corpus); MOOM competition-inhibition numeric decay formula (qualitative only in corpus hits).
- **SECOM absent**; **LightMem** referenced by name in 2601.08160/SwiftMem related-work (sensory/short/long-term + pre-compression/buffering/offline updates, lines 37–56) but its own paper is not in corpus — flagged, not fabricated.
- Papers actually read (doi list): 10.48550/arxiv.2310.08560, 2304.03442, 2305.10250, 2502.12110, 2504.19413, 2501.13956, 2405.14831, 2502.14802, 2507.07957, 2506.06326, 2601.02163, 2508.03341, 2506.07398, 2604.12285, 10.18653/v1_2026.eacl-long.15, 2604.18349, 2509.11860, 2505.18279, 2404.13501, 2502.06975, plus corroborating 2410.10813 (LongMemEval) and 2601.08160 (SwiftMem).
