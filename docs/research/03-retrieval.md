# 03 — Retrieval & Context Pruning (myelin evidence pack)

Scope: retrieval machinery that determines retrieval *quality* and *token cost*: hybrid dense+sparse, Reciprocal Rank Fusion, cross-encoder reranking, late interaction (ColBERT), query rewriting/HyDE/decomposition, iterative retrieval, chunking, embedding models (BGE-M3, E5, GTE, Qwen3-Embedding), context pruning (Provence), prompt compression (LLMLingua), and retrieval infrastructure (SwiftMem, ShardMemo, QueryLink).

Vocabulary (shared): pipeline `ingest → extract → consolidate → index → retrieve → rerank → compose → forget`; primitives `dense`, `sparse/BM25`, `hybrid+RRF`, `rerank`, `graph-expand`, `scope-filter`; eval axes `accuracy`, `token-cost`, `p95-latency`, `grounding/provenance`, `robustness/poisoning`.

**Sources verified in local corpus** (full text read/confirmed via `distill_search`/`markdown_read`):
- `10.48550_arxiv.2004.12832` — ColBERT (SIGIR 2020)
- `10.18653_v1_2022.naacl-main.272` — ColBERTv2
- `10.18653_v1_2024.emnlp-main.981` — "Searching for Best Practices in RAG" (Wang et al., EMNLP 2024) — **primary empirical anchor**
- `10.18653_v1_2021.emnlp-main.224` — RocketQAv2 (cross-encoder rerank, MRR table)
- `10.48550_arxiv.2310.05736` — LLMLingua
- `10.48550_arxiv.2501.16214` — Provence (context pruning)
- `10.48550_arxiv.2601.08160` — SwiftMem
- `10.48550_arxiv.2601.21545` — ShardMemo
- `10.18653_v1_2026.findings-acl.765` — QueryLink
- `10.18653_v1_2023.emnlp-main.322` — Query Rewriting in RAG (Rewrite-Retrieve-Read)
- `10.18653_v1_2020.emnlp-main.519` — BLINK (bi→cross-encoder two-stage; 2 ms ANN over 5.9 M candidates)
- `10.18653_v1_2024.naacl-long.463` — REPLUG
- `10.1007_978-3-031-28241-6_7` — Unified Framework for Learned Sparse Retrieval (SPLADE, hybrid motivation)

**Cited but NOT in corpus** (verified absence; flagged where relied upon):
- Cormack RRF original (SIGIR 2009) — not indexed; `k=60` constant **`[UNVERIFIED in corpus]`** (industry-standard value, see below).
- BGE-M3, E5, GTE full papers — not indexed (only citation strings found); treated as model facts from public docs, marked.
- ColBERTv2 PLAID/compression details — partially in corpus (ColBERTv2 NAACL 2022).

---

## 1. Exact algorithms / formulas

### 1.1 Reciprocal Rank Fusion (RRF)
- Cormack, Clarke & Buettcher, *"Reciprocal rank fusion outperforms condorcet and individual rank learning methods."* SIGIR 2009 (pp. 758–759). **Confirmed present as a citation** in `10.18653_v1_2024.findings-acl.372` (MEDRAG bib) — abstract-level only; full text not indexed.
- Score: `RRFscore(d) = Σ_R 1/(k + rank_R(d))`, summed over retrieval result lists R, **k a rank-smoothing constant**; **canonical k = 60** in all production RAG stacks (Qdrant, Vespa, Weaviate hybrid search). `k=60` is **`[UNVERIFIED in corpus]`** — the local corpus does not contain the RRF paper text; the 60 value is the de-facto standard from Qdrant/Vespa docs, not read here. Treat as design default, not literature-verified.
- Known properties (from ColBERTv2/practitioner consensus): score depends only on rank, not raw similarity → robust to incompatible score scales between BM25 and dense dot products (why it is favored over score interpolation for fusion).

### 1.2 Hybrid search (α-weighted score interpolation) — Verbatim formula
`10.18653_v1_2024.emnlp-main.981`, §A.3: `S_h = α·S_s + S_d`, where `S_s`, `S_d` are *normalized* sparse and dense relevance scores. **Best α = 0.3** (Table 9). Note: this is **score interpolation**, not RRF — evidence says α=0.3 beats α∈{0.1,0.5,0.7,0.9} on TREC DL19/DL20.

### 1.3 ColBERT late interaction
`10.48550_arxiv.2004.12832`: queries and documents independently BERT-encoded into per-token matrices; relevance = **MaxSim**: `Σᵢ max_j ⟨E_q[i]·E_d[j]⟩` (sum over query tokens of max cosine over document tokens). Late interaction = cheap, pruning-friendly interaction AFTER independent encoding, enabling precomputed document representations.

### 1.4 Cross-encoder reranking
`10.18653_v1_2021.emnlp-main.224` (RocketQAv2): concat `q [SEP] p`, [CLS] repr → learned linear → relevance. `10.18653_v1_2024.emnlp-main.981` §A.4: DLM rerankers fine-tuned to predict `"true"/"false"` on `(q,d)`; rank by P("true").

### 1.5 Query rewriting / decomposition / HyDE
- **Rewrite-Retrieve-Read** (`10.18653_v1_2023.emnlp-main.322`): LLM prompted to *generate* the search query (not answer); trainable small rewriter with RL reward `R_lm = EM + λ_f·F1 + λ_h·Hit`.
- **HyDE** (`10.18653_v1_2024.emnlp-main.981` §A.3): prompt LLM to write a *hypothetical answer* (pseudo-doc), embed it, retrieve by pseudo-doc similarity. **Single pseudo-doc suffices**.
- **Query decomposition**: split into sub-questions, retrieve per sub-question.

### 1.6 Provence (context pruning) — Algorithm
`10.48550_arxiv.2501.16214`: context pruning formulated as **binary sequence labeling**. A DeBERTa(-v3-large) encoder encodes (query, context) jointly; predicts a binary mask per sentence (0…all sentences flagged relevant). Trained on silver labels from Llama-3-8B-Instruct answering-with-citations on MS MARCO doc (370k queries) + Natural Questions (87k). **Key architectural move: reranking + pruning unified in ONE forward pass** — the model outputs both a relevance score (rerank) and the sentence mask (prune), so context pruning costs ~0 extra in a pipeline that already reranks.

### 1.7 LLMLingua (prompt compression)
`10.48550_arxiv.2310.05736`: coarse-to-fine. Budget controller allocates compression ratio per prompt component; iterative token-level compression using a **small LM** (Alpaca-7B or GPT-2) perplexity; instruction-tuning to align small-LM distribution to target LLM. Compression rate `τ = L̄/L`, ratio `1/τ`.

### 1.8 Infrastructure (scope-before-routing)
- **SwiftMem** `10.48550_arxiv.2601.08160`: three indexes — Temporal (sorted timeline, log-time range queries), Semantic **DAG-Tag** (routes query to bounded tag subset in `O(k(log|V| + D_max))`), Embedding (HNSW, co-consolidated by tag cluster for locality).
- **ShardMemo** `10.48550_arxiv.2601.21545`: **scope-before-routing**. Metadata predicate `ψ^τ(m,q)∈{0,1}` picks eligible shards FIRST; inadmissible shards masked to logit `-∞` (cannot consume probe budget); learned router then allocates `B_probe` shard probes **only within the admissible set**. Budget constraints: `|A(q)|≤M` (active ctx), `|P(q)|≤B_probe` (probes), `|R^B(q)|≤K` (evidence), `|U(q)|≤R` (skills).
- **QueryLink** `10.18653_v1_2026.findings-acl.765`: **Query–Memory Alignment** — project query and memory into shared space via 4 aligned granularities: `V_raw` (embedding), `V_sem/epi` (LLM-extracted intent/events), `V_kw` (entities/keywords), `V_cen = Normalize(V_raw + V_sem + V_kw)` (centroid). Plus **Coherent Memory Chunking**: chunk memories in multi-turn dialogue units `C={(u_i,r_i)...}` with dialogue-guided summarization when window W exceeded.

### 1.9 Chunking
`10.18653_v1_2024.emnlp-main.981` §3.2: sentence-level chunking recommended (token-level splits sentences; semantic-level LLM-determined breakpoints too slow). Small-to-big + sliding window help. **Chunk-size data** (Table 3, lyft_2021, text-embedding-ada-002):

| Chunk size | Avg Faithfulness | Avg Relevancy |
|---|---|---|
| 2048 | 80.37 | 91.11 |
| 1024 | 94.26 | 95.56 |
| 512 | **97.59** | 97.41 |
| 256 | 97.22 | **97.78** |
| 128 | 95.74 | 97.22 |

**512 tokens is the sweet spot** — best faithfulness, near-best relevancy, lower than 1024/2048 (which trade reliability for context).

---

## 2. Measured deltas (dataset + metric + delta over baseline)

### (a) BM25+dense+RRF vs dense alone — the core lettered question
**`10.18653_v1_2024.emnlp-main.981` is the strongest corpus anchor.** Table 7 (TREC DL19/DL20, LLM-Embedder, F1-ish metrics + latency sec):

| Method | DL20 mAP | DL20 nDCG@10 | DL19 nDCG@10 | Δ vs dense | Latency |
|---|---|---|---|---|---|
| BM25 (unsup) | 28.56 | 38.3 | 47.96 | — | 0.29 s |
| Contriever (unsup) | 23.98 | 44.54 | 42.13 | — | 0.98 s |
| **LLM-Embedder (dense)** | 45.60 | 44.66 | 68.76 | — | 0.71 s |
| + Query Rewriting | 45.16 | 67.89* | 65.62 | ↓2.1-3.1 nDCG | 2.06 s |
| + Query Decomposition | 43.30 | 66.10* | 64.95 | ↓ | 2.01 s |
| + **HyDE** | 50.94 | 75.44* | 73.94 | **+5.34 mAP / +5.2 nDCG@10 DL19** | 2.14 s |
| + **Hybrid Search** (α=0.3) | 47.72 | 72.50* | 69.80 | +2.12 mAP / +1.04 nDCG10 | 0.77 s |
| + **HyDE + Hybrid** | 53.13 | 73.34* | 72.72 | **+7.53 mAP / +3.96 nDCG10** | 2.95 s |

*(\*marked entries: the DL20 nDCG@10 column is misaligned in the OCR; DL19 nDCG@10 and DL20 mAP are the reliable columns.)*

**Reading:** (1) **Hybrid adds a solid, cheap gain over dense alone** (+2.1 mAP DL20, +1.0 nDCG DL19) at +0.06 s latency. (2) **HyDE is the single biggest additive gain** (+5.3 mAP). (3) HyDE+Hybrid stack nearly additively (+7.5 mAP) → **hybrid does NOT make HyDE redundant; they address different gaps**. (4) **Query Rewriting and Query Decomposition HURT** on this benchmark (↓nDCG) and cost 3× latency — do not enable by default.

### (b) How many chunks after reranking
`10.18653_v1_2024.emnlp-main.981` §3.6/A.4: **50 documents retrieved → rerank → top-k.** "Typically 50 documents are retrieved as input for the reranking module." The paper truncates to top-k after repacking (forward/reverse/sidewise) but **does not fix a single k**; it cites "lost-in-the-middle" ordering sensitivity. **Recommended working range: top-5…top-10** post-rerank (see §Recommended stack; no corpus paper fixes a single value, so range is `[INFERENCE]` from 50→rerank→k convention).

### (c) Reranking vs fusion — which contributes more
No corpus paper performs a clean head-to-head (fusion vs rerank on same metric). Evidence triangulation:
- **Fusion**: +2.1 mAP / +1.0 nDCG10 (above) — cheap, no model.
- **Reranking** (`10.18653_v1_2024.emnlp-main.981` §A.4, Table 10): all DLM rerankers "demonstrate a notable increase in performance across all metrics" over BM25/random; monoT5 ≈ monoBERT, **RankLLaMA best**, TILDEv2 fastest. From RocketQAv2 (`10.18653_v1_2021.emnlp-main.224` Table 3): monoBERT-style cross-encoder over BM25 top-1000 → MRR@10 36.5 (vs BM25-alone 18.7, +17.8); ColBERT (late-interaction rerank) → 34.9; RocketQAv2 reranker → 41.8.
- **Net:** reranking moves more quality than fusion (fusion ~+1-2 metric points; cross-encoder rerank is +10-17 MRR points in the retrieval-grade setting), but cost is the discriminator (cross-encoder is per-(q,d) transformer forward; fusion is O(1)). The Best-Practices paper's recommendation: **best performance+latency = Hybrid+HyDE, then rerank the (small) candidate set**. Fusion and rerank are **complementary, not redundant** — fusion widens recall cheaply, rerank sharpens precision.

### (d) When lexical beats dense
- **Learned sparse > single-vector dense out-of-domain**: `10.1007_978-3-031-28241-6_7` (MacAvaney et al. 2023): "LSR models and token-level dense models like ColBERT tend to generalize **better than single-vector dense models on BEIR**; hybrid dense+sparse 'can bring benefits for both in-domain and out-of-domain effectiveness'." So lexical/sparse wins when: (1) OOD generalization to unseen domains (BEIR zero-shot), (2) exact/rare/factual tokens (names, IDs, error codes — QueryLink `V_kw` rationale), (3) short/verbose queries, (4) when query→memory vocabulary gap is large.
- **BM25 (plain) beats Contriever** in `10.18653_v1_2024.emnlp-main.981` Table 7 on DL19 nDCG@10 (47.96 vs 42.13) — unsupervised dense is worse than classic lexical.
- QueryLink `10.18653_v1_2026.findings-acl.765`: embeddings "smooth out important details"; entity/keyword rep needed for proper nouns/technical terms → lexical layer is the precision backstop.

### (e) SwiftMem latency claim — VERIFIED with conditions
`10.48550_arxiv.2601.08160` (Huawei, 2026). Claim: **10.834 ms** (SwiftMem) vs **881.924–1231.332 ms** (baselines) on **LoCoMo with GPT-4.1-mini**, = **~81–114× lower query-time search latency** than HNSW-backed baselines (RAG-4096 1143.6 ms, LangMem 1038.8, Nemori 920.5, LightMem 881.9, EverMemOS 1231.3). Abstract also cites "10.8/12.7 ms" (LoCoMo / LongMemEval).

**Comparison conditions (must be stated):** (1) metric is **query-time ANN *search* latency only** — add-stage excluded (SwiftMem 4216 s add vs baselines up to 21448 s); (2) baselines use query-agnostic full-space HNSW; (3) the "794–1264 ms" figure in the task brief maps to baselines' 881.9–1231.3 ms range — **close but the exact 794 number is `[UNVERIFIED]`**; corpus shows 881.9–1231.3. (4) Speedup comes from **routing to a query-relevant memory subset** (DAG-Tag/temporal), NOT from a faster ANN — so it is a *scope/reachability* win (see ShardMemo). (5) Quality: SwiftMem LJ 0.7253, BLEU-1 0.4858 (best B1) but **EverMemOS wins LJ/F1** (0.8994/0.4731) — SwiftMem trades some judge score for 80-114× latency. On LoCoMo Refined SwiftMem 61.5% vs EverMemOS 58.3%.

**Verdict:** The 11.7-ms-vs-~920-ms order-of-magnitude claim is **substantiated in the corpus** (10.834 vs 881.9–1231.3 ms), but it is *search-only* and the baselines are query-agnostic. For myelin: the full query path — embedding, routing, ANN, rerank — is what the p95-latency budget must cover; SwiftMem justifies a scope-filter/graph-expand tier *before* the ANN.

### (f) ShardMemo scope-before-routing gain — VERIFIED
`10.48550_arxiv.2601.21545`. **Controlled claim (+3 F1 on LoCoMo at fixed budget): abstract states "Under matched supervision and fixed budgets, SHARD-MEMO improves over a learned router baseline by roughly +3 F1 on LoCoMo."** Confirmed in body. End-to-end vs strongest baseline: **up to +6.8 F1** on LoCoMo. Per-category (GPT-OSS-120B backbone, Table 1):

| Query type | GAM F1 | ShardMemo F1 | Δ |
|---|---|---|---|
| Single-Hop | 58.38 | 64.08 | +5.70 |
| Multi-Hop | 41.17 | 46.28 | +5.11 |
| Temporal | 59.52 | 66.34 | +6.82 |
| Open Domain | 34.10 | 40.13 | +6.03 |

- Controlled ablation (Limitations/B.4): untrained router 39.82 vs 54.21 F1 (trained) — **routing supervision matters more than the mask**; no-label cosine 47.34 recovers ~half.
- Ablation: "replacing scope-before-routing with post-filtering **lowers F1 and increases VecScan**." HotpotQA: +2.27/+1.73/+1.39 F1 at 56K/224K/448K ctx. Backbone sensitivity at Qwen3-32B: multi-hop gain shrinks +2.49 (generation-bound residual error).
- **Conclusion (f): the +3 F1 controlled claim is verified**, and the *mechanism* is that inadmissible shards never consume probe budget — a token/latency win as well as accuracy.

### Prompt compression (LLMLingua) cost numbers
`10.48550_arxiv.2310.05736`: total compute `c = (L + kL/τ + L/τ)·c_small + L/τ·c_LLM`; with small LM ≈ 1/25 LLM cost and τ=5 → `c ≈ 0.264·L·c_LLM ≈ 1/4` → **~4× compute saving at 5× compression**. Latency on GSM8K (V100-32G): end-to-end 8.6 s (uncompressed) → 4.9/2.3/1.3 s at 2×/5×/10× (1.7×/3.3×/5.7× speedup). Up to **20× compression with little performance loss** (abstract). Small-LM choice: Alpaca-7B beats GPT2-Alpaca (drops 0.99–2.06 EM points) — distribution alignment matters.

### ColBERT cost/effectiveness
`10.48550_arxiv.2004.12832`: **>170× faster and 14,000× fewer FLOPs/query than existing BERT-based models**, competitive quality. MRR@10 on MS MARCO (Table): re-rank cosine 128-dim = 34.9; end-to-end L2 = 36.0. Space: 24-dim 2-byte = 27 GiB for MS MARCO @ MRR 33.9 (−1.0); 128-dim 4-byte = 286 GiB @ 34.9. **ColBERT's end-to-end retrieval "retrieves to top-10 documents missed entirely from BM25's top-1000"** — an end-to-end-recall argument for late interaction. ColBERTv2 (`10.18653_v1_2022.naacl-main.272`) adds residual compression/centroid interaction; **PLAID-style indexing = the practical choice for a local Rust store**.

### BLINK two-stage latency
`10.18653_v1_2020.emnlp-main.519`: bi-encoder ANN over **5.9 M entities in ~2 ms**; cross-encoder rerank adds accuracy; distillation transfers cross-encoder gains back to the bi-encoder. **Direct template for a Rust backend**: HNSW bi-encoder first stage (fast, cheap) + cross-encoder rerank over top-k candidates only.

---

## 3. Computational cost (params/FLOPs/latency)

| Technique | Cost (verified) | Notes |
|---|---|---|
| BM25 | ~0.29 s/query DL19 (BestPractices) | inverted index, WAND/MaxScore pruning |
| Dense bi-encoder (LLM-Embedder) | ~0.71 s/query + ANN | ANN ~2 ms/5.9M (BLINK) |
| Hybrid (α=0.3) | ~0.77 s/query | ≈ max(dense,sparse), fusion O(1) |
| HyDE | +~1.4 s/query (2.14 vs 0.71) | LLM generation per query → **avoid in low-latency path** |
| Hybrid+HyDE | ~2.95 s/query | highest quality, 4× dense latency |
| Cross-encoder rerank (monoT5/RankLLaMA) | "notable" gain, higher latency (BestPractices); TILDEv2 10–20 ms/query | monoT5≈monoBERT; RankLLaMA best; TILDEv2 fastest (fixed collection) |
| ColBERT | >170× faster, 14,000× fewer FLOPs vs BERT rankers | 27 GiB MS MARCO @ 24-dim |
| Provence | DeBERTa-v3-large; prune fused into rerank → ~0 extra | almost "free" in rerank pipeline |
| LLMLingua | 1/4 compute at 5× compression; 1.3–4.9 s end-to-end GSM8K | small LM ~1/25 LLM cost |
| SwiftMem | 10.8 ms search (LoCoMo) vs 882–1231 ms | add-stage excluded; 81–114× |
| ShardMemo | VecScan 372–458 vectors/query; p95 measured | probe-budget controlled |

---

## 4. Failure modes

- **Hybrid score interpolation (α)**: requires score normalization; wrong α breaks it (BestPractices swept α, 0.3 optimal; 0.9 → −1.7 nDCG). RRF avoids this by using rank, but RRF k must be tuned.
- **HyDE**: hallucinated pseudo-docs can mislead retrieval (`10.18653_v1_2026.findings-acl.765` explicitly: HyDE/Query2Doc "introduce the risk of hallucinations, where inaccurate details can mislead the retrieval process"). Latency cost is large.
- **Query rewriting/decomposition**: can *hurt* — BestPractices Table 7 shows rewriting ↓nDCG on DL19 and 3× latency; hotpotQA rewriting helped only because raw multi-hop questions are bad web queries (Rewrite-Retrieve-Read) — task-dependent.
- **ColBERT**: space-hungry at full precision; 128-dim needs 286 GiB on MS MARCO (use 24-48-dim quantized PLAID). End-to-end indexing cost is one BERT pass per doc.
- **Cross-encoder rerank**: O(rankers × candidates) transformer forwards — the dominant latency term if candidate count is large; TILDEv2 needs documents pre-included in index (new/unseen docs need reindex → negates speed).
- **Chunking**: token-level splits sentences (→ recall drop); 2048-token chunks drop faithfulness to 80 (context pollution); semantic chunking too slow. **512 is the safe default**.
- **Query-agnostic full-space ANN (SwiftMem critique)**: as memory grows, full-space search is the latency bottleneck even with HNSW — ANN lowers per-candidate cost, not *which region to search*.
- **ShardMemo**: routing supervision is essential (untrained router: 39.82 vs 54.21 F1); wrong/false-excluding scope predicates cap recall; fixed shard map assumes workload stability; false inclusions waste budget.
- **LLMLingua**: small-LM/target-LLM distribution gap (Alpaca-7B vs GPT2 −1-2 EM); GPT-3.5-Turbo *could not* reconstruct compressed prompts (emergent-ability caveat); token removal can corrupt reasoning chains on reasoning tasks (Selective-Context broke 9-step chain).
- **Provence**: sentence-level only (cannot prune within-sentence); depends on silver LLM labels (~10% of cases filtered); cross-encoding makes it per-(query,context) — cost = one DeBERTa forward over the candidate set.

---

## 5. Interaction effects

- **Reranking does NOT make fusion redundant.** Fusion (hybrid) is precision-from-recall at near-zero marginal cost; rerank sharpens ordering. BestPractices pairs *both* (its default = Hybrid+HyDE, then rerank). RocketQAv2 shows rerankers built on BM25 candidate sets (not densified) still gain hugely — but ColBERT shows end-to-end retrieval recovers docs BM25's top-1000 misses → **hybrid-first recall then rerank is the robust pattern**.
- **HyDE + Hybrid stack near-additively** (+7.5 mAP ≈ +5.3 HyDE + +2.1 hybrid).
- **Provence deliberately FUSES prune into rerank**: if you already cross-encode-encode for rerank, sentence pruning is nearly free; separate prune step would double encoder cost. So **prune = free feature of the rerank stage** on the Provence design.
- **LLMLingua/Provence complement generation, not retrieval**: token pruning (LLMLingua, query-independent) and sentence pruning (Provence, query-dependent) are "orthogonal and could potentially be combined" (Provence §Related). LLMLingua also shrinks *generation* (compressed prompts produce fewer output tokens, Figure 2) → token-cost win on both sides.
- **SwiftMem/ShardMemo belong to the `scope-filter`/`graph-expand` primitive and compose with everything**: they cut the *candidate universe* before dense retrieval; rerank/fusion still apply on the (smaller) shard-local results.
- **QueryLink chunking is upstream of everything**: multi-turn-unit chunking (Coherent Memory Chunking) improves *recall* (preserves trigger–response context); a flat memory with good alignment beats complex graph topologies on LoCoMo (QueryLink) — for episodic/working memory, keep chunks coherent rather than fixed-size.

---

## 6. Six lettered questions — final answers

**(a) BM25+dense+RRF vs dense alone.** Evidence: α-interpolated hybrid (dense+sparse) over dense alone = **+2.12 mAP (DL20, 45.60→47.72)** and +1.04 nDCG@10 (DL19) at +0.06 s — `10.18653_v1_2024.emnlp-main.981` Table 7. RRF-variant (rank-based) not measured head-to-head in corpus, but the direction (hybrid > dense) is confirmed. **Answer: hybrid beats dense; budget the +sparse index.**

**(b) How many chunks after reranking.** Corpus default: **50 candidates → rerank → top-k**, k not fixed by any corpus paper; working range top-5…10 `[INFERENCE from BestPractices convention]`. Tune on eval; the "lost-in-the-middle" result (BestPractices/MEDRAG) means order matters as much as count.

**(c) Reranking or fusion contributes more.** Reranking moves far more quality (cross-encoder rerank: BM25 18.7→36.5 MRR@10, RocketQA; vs fusion ~+1-2 points) but costs per-candidate transformers. **Answer: rerank contributes more per unit quality; fusion is the cheap recall-widener — run both.**

**(d) When lexical beats dense.** Out-of-domain/BEIR generalization; exact/rare/factual tokens (names, IDs, error codes); plain-BM25 > unsupervised-dense (Contriever); sparse + hybrid robust OOD — `10.1007_978-3-031-28241-6_7`, `10.18653_v1_2024.emnlp-main.981` Table 7, QueryLink `V_kw`.

**(e) SwiftMem 11.7 ms vs 794–1264 ms.** Corpus shows **10.834 ms vs 881.9–1231.3 ms** (LoCoMo, GPT-4.1-mini, search-only, HNSW-query-agnostic baselines) = 81–114×. The 794 lower bound is `[UNVERIFIED]`; range is 881.9–1231.3. **Order-of-magnitude claim verified with the caveat it is search-latency-only.**

**(f) ShardMemo +3 F1 on LoCoMo fixed budget.** **Verified** (abstract + controlled ablations). End-to-end up to +6.8 F1; +5.11–6.82 per category over GAM; mechanism = inadmissible shards never spend probe budget.

---

## 7. Recommended retrieval stack for a local Rust implementation

Concrete stage order with parameter values and the citation justifying each. Design target: fully local, HNSW/Qdrant-class, BGE-M3 (available locally) as the single embedding/rerank backbone, p95-latency budget driven by SwiftMem's proof that scoping beats brute ANN.

```
ingest → extract → index → [scope-filter] → retrieve → rerank(+prune) → compose → forget
```

| # | Stage | Parameter / choice | Justification (citation) |
|---|---|---|---|
| 0 | Chunk | **sentence-level, 512-token chunks, 20-token overlap; small-to-big; multi-turn units for episodic/working memory** | 512 = best faithfulness 97.59 / relevancy 97.41 (`...981` Table 3); sentence-level > token/semantic (§3.2); QueryLink Coherent Memory Chunking for memory (`2026.findings-acl.765`) |
| 0 | Embed | **BGE-M3** (dense+sparse+multi-vector in one model, 1024-dim dense, bilingual — matches local availability) | BGE-M3 = unified dense/sparse/multi-vector (`2402.03216`, not in corpus → `[UNVERIFIED in corpus]`; model fact from public docs). In-corpus embedding reality: size matters (LLM-Embedder 3× smaller ≈ bge-large, `...981` §A.2) |
| 1 | Scope-filter | **ShardMemo scope-before-routing**: metadata predicates (user/session/time/namespace) → mask inadmissible shards → only then route. Budgets `B_probe≈3, K≈10` | +3 F1 controlled / +6.8 end-to-end, inadmissible shards never consume probe budget (`2601.21545`); SwiftMem shows scope cuts search 81–114× (`2601.08160`) |
| 2 | Retrieve | **Hybrid: dense (BGE-M3 dense) + sparse (BGE-M3 sparse or BM25) fused with RRF, k=60** (RRF over α-interpolation to avoid score-calibration). Retrieve top-50. | hybrid +2.1 mAP over dense (`...981` Table 7); RRF k=60 = standard but `[UNVERIFIED in corpus]`; top-50 candidate convention (`...981` §3.6) |
| 2b | Recall guard | DAG-Tag/graph-expand + temporal index for temporal/semantic-locality queries | SwiftMem DAG-Tag `O(k(log|V|+D_max))` (`2601.08160`) |
| 3 | Rerank + prune | **Cross-encoder rerank over top-50, monoT5/RankLLaMA-class (or BGE-M3-reranker). Provence-fused sentence pruning (DeBERTa) → top-5…10, order preserved** | rerank = biggest quality lever (`2021.emnlp-main.224`: 18.7→36.5 MRR); Provence: prune fused into rerank = "almost free" (`2501.16214`); keep 5-10 `[INFERENCE]` |
| 4 | (Optional, high-latency) HyDE / query rewrite | **Off by default; enable per-query or in async/consolidation path** | HyDE big gain but +1.4 s/query; rewriting can hurt (`...981` Table 7); hallucination risk (`2026.findings-acl.765`) |
| 5 | Compose | Repack forward (answer-relevant-first); cap composed tokens | lost-in-the-middle (`...981` §3.6, MEDRAG) |
| 6 | Token-cost trim | LLMLingua-style compression only for huge composed contexts (≥4× compute saved at 5× when small LM ≈ 1/25 LLM) | `2310.05736` |
| 7 | Forget | (owned by Forgetting pack) | — |

**Key takeaway for myelin:** the dominant, evidence-backed stack is *scope-before-routing → hybrid dense+sparse (RRF) → cross-encoder rerank with fused sentence pruning → top-K compose*. The two highest-leverage additions per the corpus are (1) a scope/shard layer (ShardMemo +3-6.8 F1 AND 81-114× latency via SwiftMem), and (2) HyDE only where added latency is affordable. BGE-M3 covers dense+sparse+lexical in one locally-available model, keeping the Rust backend to a single embedding service plus an inverted index.
