# 07 — Graph Memory & Multi-Hop Reasoning Evidence Pack

**Target:** myelin (Rust agentic-memory system: backend crate + MCP server + eval harness, local 'home cloud').
**Method:** verified extractions from the local academic corpus (~9,770 papers, Qdrant). Every factual claim carries a DOI/doc_id and (where quoted) a line pointer. Numbers not verifiable in the corpus are marked `[UNVERIFIED]` with what was searched.
**Scope:** graph-structured memory + multi-hop reasoning. Systems: HippoRAG / HippoRAG-2, GraphRAG, LightRAG, Zep/Graphiti, G-Memory, GAM, QueryLink, MultiHop-RAG, entity resolution/coreference, and 'graph vs hybrid+rerank' evidence.

---

## 0. Source-citation key (doc_ids)

| Short name | Paper | doc_id | Line refs (chunk_text) |
|---|---|---|---|
| HippoRAG | Gutiérrez et al. 2024 | `10.48550_arxiv.2405.14831` | 58-67, 67-79, 278-326, 326-372, 372-428, 1368-1489 |
| HippoRAG-2 | Gutiérrez et al. 2025 | `10.48550_arxiv.2502.14802` | 55-70, 70-83, 83-99, 495-542, 717-756, 1236-1364 |
| GraphRAG | Edge et al. 2024 | `10.48550_arxiv.2404.16130` | 25-37, 44-55, 55-84, 109-132, 225-426, 768-878 |
| LightRAG | Guo et al. 2024 | `10.48550_arxiv.2410.05779` | 29-73, 73-93, 91-114, 146-363, 577-616, 633-689 |
| Zep/Graphiti | Rasmussen et al. 2025 | `10.48550_arxiv.2501.13956` | 32-64, 64-103, 132-203 |
| G-Memory | Zhang et al. 2025 | `10.48550_arxiv.2506.07398` | 27-46, 71-104, 423-499, 664-694 |
| GAM | Wu et al. 2026 | `10.48550_arxiv.2604.12285` | 122-168, 176-359 |
| QueryLink | Hu et al. 2026 | `10.18653_v1_2026.findings-acl.765` | 1-15, 164-331 |
| Best-practices RAG | Wang et al. 2024 | `10.18653_v1_2024.emnlp-main.981` | 1-35, 101-117 |
| DPR | Chen et al. 2020 | `10.18653_v1_2020.emnlp-main.550` | 436-442 |
| LongMemEval | Wu et al. 2024 | `10.48550_arxiv.2410.10813` | 1325-1434 |
| MultiHop-RAG | Tang & Yang 2023 | `10.48550_arxiv.2401.15391` | 22-28, 142-156 |
| Evolving-Memory survey | 2026 | `10.48550_arxiv.2603.11768` | 131-262 |
| Entity resolution survey | (relational) | `W1529533208` | 118-126 |

---

## 1. Graph construction procedure

### 1.1 HippoRAG (open KG via OpenIE)
- **Prompt config:** 1-shot prompting of instruction-tuned LLM (GPT-3.5; also Llama-3.1-8B/70B, REBEL). Two-step: (1) extract named entities from passage; (2) feed them back into the OpenIE prompt to extract final triples that ALSO contain concepts (noun phrases) beyond named entities. 'two-step prompt configuration leads to an appropriate balance between generality and bias towards named entities.' (`2405.14831` L58-67)
- **Node/edge types:** nodes `N` = noun-phrase/concept nodes; edges `E` = relation edges (from triples); synonymy edges `E'` added when cosine similarity between two node embeddings >= threshold `τ`; passage-aggregation matrix `P` (|N|×|P|) counting phrase-per-passage occurrences. (`2405.14831` L58-67)
- **Cost per doc (LLM calls):** 2 LLM calls per passage. Offline indexing of 10k passages: GPT-3.5-Turbo-1106 **$15, ~60 min**; Llama-3.1-8B **$0, ~120 min**; Llama-3.1-70B **$0, ~250 min** (4 H100). Baselines ColBERTv2/IRCoT: $0, 7 min. (`2405.14831` L1368-1489) → graph construction is the dominant cost (~10x time, +$15/10k-pass over IRCoT).

### 1.2 HippoRAG-2 refinements
- Same offline OpenIE plus **passage nodes**: each passage is a node, 'contains' context edge to all its phrases. Graph = phrase + passage nodes, relation + synonym + context edges. (`2502.14802` L70-83)
- **Graph stats (Llama-3.3-70B), Table 10** (`2502.14802` L1236-1364): MuSiQue 11,656 passage / 85,288 phrase / 140,830 extracted edges / **1,125,951 synonym edges** / 132,586 context edges / 1,399,367 total edges. Synomy edges dominate (~10x extracted edges) — a storage/compute red flag.

### 1.3 GraphRAG (Microsoft)
- **Prompt config:** documents → text chunks (chunk size = recall/cost knob). One multipart prompt extracts entities (name/type/description) + relationships (source/target/description) as delimited tuples; optional parallel **Claim Extraction Prompt** (dates, events, interactions). GPT-4 defaults. (`2404.16130` L55-84)
- **Cost:** chunking + 1-2 extraction calls per chunk. Graphs: **8,564 nodes/20,691 edges (Podcast)**, **15,754 nodes/19,520 edges (News)**. (`2404.16130` L225+

### 1.4 LightRAG
- Three-stage: `Recog` (LLM extracts entities+relations per chunk), `Prof` (LLM generates key→value; entities keyed by name, relations by multiple LLM keys incl. global themes), `Dedup` (LLM merges identical entities/relations). (`2410.05779` L50-73)
- **Cost:** 'LLM needs to be called total_tokens/chunk_size times' — no extra overhead; incremental = same per-new-doc cost. (`2410.05779` L91-114)

### 1.5 Zep/Graphiti
- Episodes → entity extraction (with reflection for hallucination control) → resolve against existing graph → facts with key predicate (hyper-edge for multi-entity facts) → temporal metadata. **Bi-temporal model:** `t_valid`/`t_invalid` (world) + `t'_created`/`t'_expired` (transaction). New contradictory edges invalidate prior edges. Communities via label-propagation (not Leiden) for dynamic extension. (`2501.13956` L49-103)

### 1.6 G-Memory
- Three-tier: **Insight graph** (distilled insights + supporting-query hyper-edges), **Query graph** (query + status + interaction graph), **Interaction graph** (nodes = per-agent utterances, edges temporal). (`2506.07398` L71-104)

---

## 2. Retrieval algorithm over the graph — exact parameters

### 2.1 HippoRAG Personalized PageRank (PPR) — implementable spec
- **Seed selection:** LLM 1-shot extracts query named entities `C_q`; embeddings match each to highest-cosine node: `r_i = argmax_j cos(M(c_i), M(e_j))` → seeds `R_q`. (`2405.14831` L58-67)
- **Reset vector:** personalized distribution `n` over `N`: each query node equal probability, all others 0. (`2405.14831` L67-79)
- **Node specificity:** pre-PPR multiply each query-node prob by `s_i = |P_i|^{-1}` (inverse of # passages mentioning node i) — a local IDF. (`2405.14831` L67-79)
- **PPR run:** over KG (|N| nodes, |E|+|E'| edges); **passage score = n' · P** (PPR node-prob vector × phrase×passage incidence matrix). Top passages returned. (`2405.14831` L67-79). python-igraph; damping/ε not printed → `[UNVERIFIED: exact damp α/ε; searched corpus, not found — use standard PPR α≈0.85]`.
- **Ablation (Table 5):** nodes-only 50.7/56.2; +1-hop neighbors w/o PPR 42.2/59.2; full PPR 57.1/72.5 avg R@2/R@5. **PPR beats naive neighborhood expansion.** (`2405.14831` L326-372)
- **Synonymy threshold τ:** not printed → `[UNVERIFIED τ]`.

### 2.2 HippoRAG-2 PPR (weighted reset, two node classes)
- **Seeds:** phrase nodes from top-k filtered triples (query→triple + LLM recognition-memory filter); fallback to top passages if no triples. **ALL passage nodes also seeds** ('broader activation improves multi-hop reasoning'). (`2502.14802` L83-99)
- **Reset assignment:** phrase ← ranking scores; passage ← embedding-sim × **weight 0.05** (tuned, Table 5: MuSiQue F1 best 80.5 at 0.05). (`2502.14802` L495-542)
- **Passage ranking:** top PageRank scores of passage nodes; python-igraph. Query-to-triple beats NER-to-node by **+12.5% Recall@5**. (`2502.14802` L83-99, L495-542)

### 2.3 LightRAG dual-level
- Query → LLM extracts **local keys** `k^(l)` + **global keys** `k^(g)`. Match local→entities, global→relation keys (vector DB). Gather 1-hop neighbors `N_v ∪ N_e` for higher-order relatedness. (`2410.05779` L91-114)
- **Retrieval cost:** <100 tokens + **1 API call** per query (vs GraphRAG 610k tokens for 610 level-2 communities). (`2410.05779` L633-689)

### 2.4 GraphRAG global search
- Map-reduce: shuffle summaries → chunk to token budget → parallel map (partial answer + 0-100 helpfulness; score-0 dropped) → reduce (sort desc, fill context) → final answer. (`2404.16130` L109-132)
- Token cost (Table 2): root-level C0 = **2.6%/2.3%** of source-text TS (26,657 vs 1,014,611 Podcast) — **9x-43x fewer tokens**. (`2404.16130` L768-878)

### 2.5 Zep/Graphiti
- `φ` search → `ρ` rerank → `χ` construct. Rerankers: mention-frequency, centroid node-distance, cross-encoder (best, highest cost). Retrives 20 edges + 20 entity nodes. (`2501.13956` L84-132)

### 2.6 G-Memory / GAM
- G-Memory: coarse retrieval → upward (query→insight, 1-hop, k∈{1,2}) + downward (query→interaction, LLM sparsifier). **1-hop optimal; 2-3 hop degrades** (PDDL 55.24→49.79). (`2506.07398` L423-499)
- GAM: semantic anchors + 1st-order neighbor expansion → drill-down cross-layer → multi-factor rerank `Score = P_sem(v|q)·Π β_k^{I_k}`, β over time/conf/role. (`2604.12285` L122-168)

---

## 3. Measured gain over flat vector retrieval (per dataset / metric)

### 3.1 HippoRAG QA EM/F1, top-5 (Tables 4, 6) (`2405.14831` L278-326, L372-428)

| Retriever | MuSiQue EM/F1 | 2Wiki EM/F1 | HotpotQA EM/F1 | Avg EM/F1 |
|---|---|---|---|---|
| None | 12.5/24.1 | 31.0/39.6 | 30.4/42.8 | 24.6/35.5 |
| ColBERTv2 (flat dense) | 15.5/26.4 | 33.4/43.3 | 43.4/57.7 | 30.8/42.5 |
| **HippoRAG (ColBERTv2)** | **19.2/29.8** | **46.6/59.5** | 41.8/55.0 | **35.9/48.1** |
| IRCoT (ColBERTv2) | 19.1/30.5 | 35.4/45.1 | 45.5/58.4 | 33.3/44.7 |
| IRCoT + HippoRAG | **21.9/33.3** | **47.7/62.7** | 45.7/59.2 | **38.4/51.7** |

**All-recall** AR@2/AR@5: ColBERTv2 MuSiQue (—/6.8), 2Wiki 16.1/25.1, Hotpot 37.1/33.3, avg 21.7/37.4 vs **HippoRAG MuSiQue (—/10.2 then 2Wiki 45.4/75.7), Hotpot 33.8/57.9, avg 29.8/52.0**. Gap grows 3%→6% (MuSiQue), 20%→38% (2Wiki) at top-5 → **graph gains come from retrieving the full support set.** (`2405.14831` L372-428)
- Recall: '11 and 20% for R@2 and R@5 on 2WikiMultiHopQA and around 3% on MuSiQue'; +18% R@5 (2Wiki), +4% (MuSiQue) as IRCoT retriever. (`2405.14831` L278-326)
- **Where it LOSES:** HotpotQA (41.8/55.0 < ColBERTv2 43.4/57.7) — spurious single-hop signals. (`2405.14831` L67-79)
- Cost: online/1k queries **$0.1, 3 min** vs IRCoT **$1-3, 20-40 min** (10-30x cheaper, 6-13x faster). (`2405.14831` L1368-1489)

### 3.2 HippoRAG-2 (Recall@5 vs NV-Embed-v2) (`2502.14802` L495-542)
- **+5.0% (MuSiQue), +13.9% (2Wiki)** over the strongest 7B dense baseline; HippoRAG-v1 'generally lags behind recent dense retrievers'. HippoRAG-2 F1 stable/rising as corpus grows 4x (continual learning) vs flat flat. |

### 3.3 GraphRAG vs vector RAG, global sensemaking (LLM-judge) (`2404.16130` L225-426, L768-878)
- Comprehensiveness win 72-83% (Podcast), 72-80% (News); diversity 75-82% / 62-71% — all significant. **But vector RAG wins directness (SS 75%). Gain is on whole-corpus sensemaking, NOT fact multi-hop QA** — the paper itself says ODQA benchmarks 'are oriented towards vector RAG performance.' (`2404.16130` L44-55)

### 3.4 LightRAG (win rates, Table 1) (`2410.05779` L146-363)
- vs NaiveRAG overall 67.6/61.2/84.8/60.0 (Agric/CS/Legal/Mix); diversity up to 86.4% (Legal). vs HyDE overall 75.2/58.4/73.6/57.6. **vs GraphRAG overall ~50/50 (54.4/51.6/51.6/49.6)** — a light non-community graph ≈ heavy community GraphRAG. Ablations: removing either level hurts; dropping raw text (-Origin) shows no decline, sometimes improves. (`2410.05779` L577-616)

### 3.5 MultiHop-RAG (`2401.15391` L22-28, L142-156)
- 2,556 queries: Inference 816 (31.9%), Comparison 856 (33.5%), Temporal 583 (22.8%), Null 301 (11.8%). Reranker used over embedding top-K. `[UNVERIFIED: no accuracy table extracted]`.

### 3.6 LongMemEval flat-vector reference (`2410.10813` L1325-1434)
- Dense (Contriever) ≫ BM25 (@round Recall@5 0.747 vs 0.538; NDCG 0.495 vs 0.372). **Key expansion (facts/summary/keyphrases) beats raw keys; greatest from facts. Rank-merge < key-merge.** Vector-side alternative to graph edges.

---

## 4. Storage / compute footprint & update cost

| System | Index size | Update model | Incremental cost | Note |
|---|---|---|---|---|
| HippoRAG-2 | MuSiQue 96,944 nodes, 1,399,367 edges (synonym ~1.13M) | per-passage extract | cheap per-doc; synonym scan superlinear | synonym explosion dominant |
| GraphRAG | 8.5-15.7k nodes, ~20k edges | **rebuild communities + summaries** | ~1,399x2x5,000 ≈ 14M tokens per dataset insert (`2410.05779` L667-689) | exorbitant |
| LightRAG | graph key-value | **incremental union; no rebuild** | = per-doc extract | 'eliminates need to rebuild' |
| Zep/Graphiti | tri-tier | label-prop + refresh | low per-episode; periodic community refresh | |
| G-Memory | 3 graphs | per-task update | +1.4e6 tokens for +10.32% (PDDL) vs MetaGPT-M +2.2e6 for +4.07% | token-efficient |

---

## 5. Is a graph needed at all? — balanced both-sides evidence

### 5.1 FOR a graph
1. HippoRAG: +3.7 EM avg (30.8→35.9; 42.5→48.1 F1); all-support AR@5 2Wiki 25.1→75.7; +5.0/+13.9 Recall@5 (HippoRAG-2). (`2405.14831`; `2502.14802`)
2. GraphRAG sensemaking: 72-83% comp / 62-82% div; 9-43x token cut. (`2404.16130`)
3. HippoRAG-2 continual learning: graph stable/rising under 4x growth; NV-Embed-v2 flat. (`2502.14802`)
4. Single-step multi-hop: 10-30x cheaper online than IRCoT. (`2405.14831`)
5. G-Memory +20.89% ALFWorld / +10.12% knowledge QA; GAM avg F1 30.11→36.27 vs Mem0. (`2506.07398`; `2604.12285`)
6. PPR search matters: R@5 72.5 (PPR) vs 59.2 (neighbors-only). (`2405.14831`)

### 5.2 AGAINST a separate graph
1. **Best-practices RAG (EMNLP 2024):** recommends 'Hybrid Search (BM25+dense) with HyDE as default' + monoT5/LLM-Embedder rerank — no graph needed for top TREC-DL/MS-MARCO. (`10.18653_v1_2024.emnlp-main.981` L101-117)
2. **7B dense erodes graph edge:** HippoRAG-v1 lags 7B embedders; HippoRAG-2 margin on MuSiQue only +5.0. HotpotQA: flat > graph. (`2502.14802`; `2405.14831`)
3. **Vector key-expansion ≈ graph edges:** facts/summary/keyphrase keys beat raw (LongMemEval Recall@5 0.747 vs 0.538). (`2410.10813`)
4. **LightRAG ≈ GraphRAG** on quality (~50/50) — heavy community machinery adds little for fact-level quality. (`2410.05779`)
5. **QueryLink:** flat query-memory alignment **beats graph Mem0 even in Multi-Hop** (Judge 70.21) and beats graph Zep on LongMemEval avg 69.80 vs 63.80. (`10.18653_v1_2026.findings-acl.765` L164-331) — strongest 'graphs unnecessary even for multi-hop' result.
6. **DPR-era:** BM25+DPR hybrid already beats plain DPR on NQ (57.9 vs 41.5 EM). (`10.18653_v1_2020.emnlp-main.550`)

### 5.3 Synthesis
- **Graph wins** where multi-hop *entity-bridging* is required (2Wiki +14-38 pts) and/or single-step multi-hop under latency/token budget (10-30x cheaper).
- **Graph loses/unnecessary** where facts are single-hop/noisy (HotpotQA), you can afford 7B embedders + costly rerank, queries are fact-answer (not sensemaking), or you adopt key-expansion on a good hybrid index.
- **Cheapest defensible graph = HippoRAG-style:** entity/relation extraction per chunk + PPR. Skip GraphRAG community summaries unless whole-corpus sensemaking is a product need.

---

## 6. ## Minimum viable graph layer (design recommendation)

- **Verdict:** a graph layer IS justified for multi-hop entity-bridging + single-step efficiency, but must be **lightweight (HippoRAG-style extraction + PPR), NOT GraphRAG-community-style.** GraphRAG community machinery buys mostly sensemaking diversity at huge rebuild cost / ~50-50 quality parity.
- **Adjacency:** directed labeled graph (subject, relation, object) + (node→passage) incidence for ranking. **Synonym-edge explosion (~10x extracted edges) is the primary storage risk to design around.**
- **Traversal budget:** PPR (α≈0.85 default; `[UNVERIFIED]` exact) over the KG; seeds = query entities (or query→triple, best) plus all passage nodes; passages ranked by summed PPR mass over their phrase nodes. **1-hop expansion is the safe default** (G-Memory: 2-3 hop degrades). Budget: python-igraph handles ~1-2M edges — a small local store suffices.
- **Incremental update:** new passage → 2 extraction LLM calls → merge nodes/edges → re-run PPR at query time. **No full rebuild.** Synonym edges + community labels are the only global-recompute parts; amortize periodically (Zep label-propagation template). GraphRAG full rebuild = non-goal (~14M tokens/insert).
- **Qdrant vs separate graph store:** corpus evidence says **Qdrant (vector) is necessary but NOT sufficient** for the multi-hop wins — PPR + phrase×passage incidence are graph primitives a flat vector index lacks (+5-14 Recall@5, +up to 38 pts AR@5 on 2Wiki). But the graph can be **thin**: no evidence a heavyweight graph DB is needed (HippoRAG uses python-igraph over ~1-2M edges; LightRAG gets parity via key→value graph + vector match). **Recommendation:** keep Qdrant for seed/keyword matching + hybrid RRF; add a **small local adjacency + incidence layer** (in-memory or compact embedded index) for PPR and the passage×phrase matrix. A networked graph DB (Neo4j-class) is **not supported by the evidence** at this scale.

---

## 7. Entity resolution / coreference across sessions
- **Zep/Graphiti (production reference):** embed entity name (1024-d) → cosine search + full-text on names/summaries → **LLM entity-resolution prompt** → on duplicate merge to updated name+summary. Speaker auto-extracted; n=4 messages context; edge dedup constrained to same entity-pair. (`2501.13956` L49-87)
- **HippoRAG synonym edges = implicit resolution** via cosine>τ (τ unverified). (`2405.14831` L58-67)
- **Generic:** collective resolution — matching decisions propagate through graph to match surface names to KG entities. (`W1529533208` L118-126)
- **Coreference→graph:** coreference clusters (phrases → same object) feed IE graph construction. (`W4317931697` L1112-1206)
- **Rule from evidence:** do resolution **at write time** (LLM resolve-then-merge, Zep), not retrieve time. Resolution/search is a top error source — HippoRAG-2: phrase mismatches in 26% of failing samples after filtering. (`2502.14802` L717-756)

---

## 8. Highest-leverage findings for a Rust implementation
1. **HippoRAG-style PPR over an LLM-extracted open KG is the best-justified graph primitive**: single-step multi-hop at $0.1/3min per 1k queries (10-30x cheaper than IRCoT), +5-14 Recall@5 over 7B dense, +up to 38pts all-support recall on 2Wiki. (`2405.14831`; `2502.14802`)
2. **The graph must be thin**: ~1-2M edges + node-passage incidence; python-igraph suffices. Synonym-edge explosion (~10x) is the storage risk. No heavyweight graph DB evidenced.
3. **Incremental ingestion required; community re-summarization is a non-goal.** GraphRAG rebuild ≈14M tokens/insert; HippoRAG/LightRAG/Zep ingest per-passage/episode at extraction cost. (`2410.05779`)
4. **Graph ≠ always better.** Flat hybrid+rerank, 7B embedders+key-expansion, or QueryLink alignment beat graphs on single-hop/fact and some multi-hop (LongMemEval 69.8 vs 63.8; HotpotQA flat>graph). Route by query type. (`10.18653_v1_2024.emnlp-main.981`; `10.18653_v1_2026.findings-acl.765`; `2405.14831`)
5. **1-hop expansion + PPR beats naive +1-neighbor** (R@5 72.5 vs 59.2); 2-3 hops degrade (G-Memory). Seed-node quality (query→triple + LLM filter, passage weight ≈0.05) is the load-bearing hyperparameter. (`2502.14802` L495-542)

---

## Appendix — numbers not verified (honesty log)
- Exact PPR damping `α` / iteration `ε` (`[UNVERIFIED]`; searched 'damping iteration epsilon teleport'); recommended α≈0.85 default, confirm vs paper appendix.
- Synonym-cosine threshold `τ` (`[UNVERIFIED]`).
- MultiHop-RAG numeric accuracy table (`[UNVERIFIED]`).
- **KG-RAG (Soman et al.) is NOT indexed** in the local corpus (searched 'KG-RAG Soman 2023 biomedical MedQA'); no MedQA accuracy cited here.
