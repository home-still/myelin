# 04 — Benchmarks for Agentic Memory: Evidence Pack for Proving SOTA

Scope: **the most important file in the evidence batch** — it defines how `myelin` will prove state-of-the-art against existing agentic-memory / long-context / conversational-memory benchmarks. Locate, read, and produce a dense spec for every benchmark present in the local corpus (~9,770 papers), plus critiques. Target output: dense spec per benchmark (task families + example, exact dataset size, concrete data-acquisition path, metric formulas, SOTA numbers with system+date, known flaws/disagreements, run cost), a reproducible-harness design, and a single SOTA acceptance table.

Shared vocabulary (per design contract): record kinds `episodic | semantic | procedural | working`; pipeline `ingest → extract → consolidate → index → retrieve → rerank → compose → forget`; primitives `dense | sparse/BM25 | hybrid+RRF | rerank | graph-expand | scope-filter`; eval axes `accuracy | token-cost | p95-latency | grounding/provenance | robustness/poisoning`.

Conventions: every benchmark carries its DOI/doc_id; every quoted number carries a chunk line range from `distill_search` unless `[UNVERIFIED]` and the search that failed is stated.

---

# Benchmark 1 — LoCoMo (\"Long-Context Memory\")

- **DOI**: 10.48550/arxiv.2402.17753 (Maharana, Lee, Tulyakov, Bansal, Barbieri, Fang) — \"Evaluating Very Long-Term Conversational Memory of LLM Agents\".
- **Category**: conversational long-term memory; the de-facto standard for memory-system QA eval.

### (1) Task families + example item
Five QA categories (exact counts verified at 10.18653/v1_2026.eacl-long.15 lines 79–94):

| Cat | Full name | Example | #QA |
|---|---|---|---|
| SH | Single-hop | Recall a fact stated in one session (e.g. \"What is Joanna's hobby?\") | 2,705 |
| MH | Multi-hop | Synthesize across >1 session (\"What did Joanna and Nate agree about movies?\") | 1,104 |
| T | Temporal reasoning | Answer with time reference from dialogue (\"When did Joanna finish her screenplay?\") | 1,547 |
| OD | Open-domain knowledge | Integrate conversation with world knowledge (\"What kind of film genre did she favor?\") | 285 |
| A | Adversarial / unanswerable | Question with no answer in history → should abstain | 1,871 |
| — | **Total** | | **7,512** |

### (2) Dataset size
- **50 dialogues**; avg **~300 turns**; up to **35 sessions**; avg **~9k tokens/dialogue** (verified at 10.48550/arxiv.2502.12110 lines 117–138 and 10.18653/v1_2026.eacl-long.15 lines 79–94).
- Table-of-benchmarks comparison (2410.10813 lines 26–132): LoCoMo = 1k sessions, 7,512 questions, ~10k context depth.
- Note: many later works run only the **10-dialogue subset \"LoCoMo10\"** (avg 24k tokens, 1,540 questions, four reasoning categories — verified 10.48550/arxiv.2508.03341 lines 468–546). State which subset when reporting.

### (3) Data acquisition
- License: **CC BY-NC 4.0 DEED** (explicit at 2402.17753 lines 883–903 / Appendix B.2).
- Build pipeline (2402.17753 lines 122–138): personas seeded from MSC; temporal event graph (up to 25 causally-connected events over 6–12 months); generative-agent architecture with gpt-3.5-turbo for 2 speakers + human filtering.
- Data + code: official repo is `github.com/adymaharana/LoCoMo` (project page). Also mirror on HuggingFace (`locomo`). Concrete download: `git clone https://github.com/adymaharana/LoCoMo && pip install -r requirements.txt`; JSON + images per dialogue. `[VERIFIED via paper; repo path from project page — flag to confirm exact clone path before harness script.]`

### (4) Metric definitions (exact)
- **F1 score** — token-level precision/recall against gold answer (harmonic mean P·R). \"F1 = 2PR/(P+R)\" standard; computed on **space-tokenized** predicted vs gold string. This is *not* ROUGE; it is over the whole predicted text vs the reference answer.
- **BLEU-1** — unigram BLEU scoring word overlap of generated vs gold response.
- Both reported per-category and averaged. Verified formulation at 10.18653/v1_2026.eacl-long.15 lines 79–94: \"F1…harmonic mean of precision and recall…BLEU-1…word overlap with ground truth.\"
- **Important**: no exact-match-normalization is defined by the paper itself; the *judge* is the literal string. Later works (Mem0, Zep, NEMORI) switch to an **LLM judge score** (see SOTA section), which is a different metric than LoCoMo F1 — never compare LoCoMo F1 numbers to LLM-judge numbers directly.

### (5) Official leaderboard / SOTA
- Original paper best: **gpt-4-turbo overall 32.4** (F1) vs human benchmark 87.9 (2402.17753 lines 486–569). On adversarial, gpt-3.5-turbo-16k collapses to **2.1%**.
- **LLM-judge best (the strongest published, verifiable)**: **Zep 85.22 (gpt-4.1-mini); EverMemOS 93.05 (gpt-4.1-mini)** overall accuracy — 10.48550/arxiv.2601.02163 Table 1 (lines 135–274). EverMemOS per-cat (gpt-4.1-mini): SingleH 96.67, MultiH 91.84, Temporal 89.72, OpenDomain 76.04.
- F1-scale best with open backbone: **GAM 40.00 avg F1 on LoCoMo (gpt-4o-mini)** — 10.48550/arxiv.2604.12285 (lines 625–786).
- Date: EverMemOS arXiv 2601.02163 (2026). Zep 2501.13956 (Jan 2025). GAM 2604.12285 (Apr 2026).

### (6) Known flaws / disagreements
- **Trivially answerable in-context**: many single-turn fact-retrieval questions fit in modern context windows; full-context baselines match memory systems (Zep critique, 2501.13956 lines 203–220).
- **Evidence-distribution under-credits retrieval**: gold-evidence-annotated sessions are not the only source; redundant identity/activity facts appear in multiple sessions, so recall@K vs annotated evidence underestimates usefulness (EverMemOS analysis, 2601.02163 lines 894–968: zero-recall Q drops 429→125 as K grows 1→3).
- **Leakage via parametric knowledge**: same TV/persona-based content can be answered from prior knowledge (flagged in DialSim/DialogueQA related work, 2406.13144 lines 26–93).
- **Disagreement on metric**: LoCoMo F1/BLEU-1 are string-metrics with no judge; Mem0/Zep/NEMORI report LLM-judge, giving **not apples-to-apples** numbers. LoCoMo-refined-style criticism argues the QA is answerable via paraphrase, penalizing lexical overlap unfairly (see Benchmark 11 critique).

### (7) Cost to run
- Full 7,512 questions × (retrieval+answer) LLM calls. RAG baseline ingest = one embedding per chunk; answer = 1 generation/question. Memory systems add per-turn consolidation ops (Mem0 ~1,602 k tokens total for LoCoMo; NEMORI 373 calls / 323 k tokens — verified 2508.03341 lines 468–546).
- LoCoMo10 (1,540 Q) is the standard cheap subset: ~1,540 × (retrieve+generate) + ingestion passes. Approx **1.5–4 M input tokens** for a retrieval-based system on LoCoMo10.

---

# Benchmark 2 — LongMemEval (LME)

- **DOI**: 10.48550/arxiv.2410.10813 (Wu, Wang, Yu, Zhang, Chang, Yu — UCLA/Tencent). GitHub `xiaowu0162/LongMemEval` (stated lines 1–15).

### (1) Task families + example
Five core abilities; **7 question types** (verified lines 151–176):
1. **Information Extraction (IE)** — single-session-user (recall info user gave), single-session-assistant (info assistant gave), single-session-preference (user info → personalized response).
2. **Multi-Session Reasoning (MR)** — aggregate user info across ≥2 sessions.
3. **Knowledge Updates (KU)** — recognize user life-state changes and update accordingly.
4. **Temporal Reasoning (TR)** — reason with metadata timestamps + explicit time references.
5. **Abstention (ABS)** — 30 questions modified to \"false premise\"; must answer \"I don't know\".
- Example item: user–assistant task-oriented dialogue; question hidden in a length-configurable chat history; the answer `a` is a short phrase or an open-ended natural-language rubric (`q` open-ended). (lines 151–176).

### (2) Dataset size
- **500 manually created questions**, each in its own coherent, length-configurable chat history.
- Two standard settings:
  - **LONGMEMEVAL_S**: ~115,000 tokens/history, ~50 sessions (verified lines 15–28; Zep cites ~115k avg, 2501.13956 lines 217–335).
  - **LONGMEMEVAL_M**: **500 sessions ≈ 1.5 million tokens** (lines 15–28).
- Built from **164 user attributes** in 5 categories (lifestyle, belongings, life events, situation context, demographics) (lines 151–176).

### (3) Data acquisition
- Data + code fully public at `https://github.com/xiaowu0162/LongMemEval` (stated abstract lines 1–15). License: **MIT** (ethics statement lines 803–816). Built from ShareGPT (Apache 2.0) + UltraChat (MIT).
- Pipeline: `git clone`, then load `LONGMEMEVAL_S`/`_M` JSON (conversations + `(S,q,t_q,a)` tuples). Generation recipe (source mixture) released so any length can be created (lines 803–816).

### (4) Metric definitions
- **Accuracy** on closed (short-phrase) answers: exact match or LLM-judge with **question-specific prompts**; for open-ended `q`, `a` is a natural-language rubric and a judge scores conformity (lines 151–176; Zep uses the \"question-specific prompts…which have demonstrated high correlation with human evaluators\", 2501.13956 lines 217–335).
- **Recall@k** separate for the retrieval stage: fraction of gold evidence sessions retrieved (analysis lines 894–968 shows recall@k defined against gold-evidence sessions).

### (5) Official leaderboard / SOTA (accuracy %)
- Commercial/full-context baselines (lines 365–382, Aug 2024): full-context GPT-4o/Llama3.1/Phi-3 drop **30–60%** in long-context vs oracle-retrieval setting.
- **Zep (gpt-4o, Jan 2025)**: **71.2%** on LME_S, latency 2.58 s, **1.6k avg context tokens** vs full-context 60.2%/28.9 s/115k — 2501.13956 lines 217–335 (+18.5% acc; Zep claims up to +18.5% in abstract).
- Zep gpt-4o-mini: 63.8% (vs 55.4% full-context), 3.20 s, 1.6k tokens.
- **NEMORI avg 64.2 (gpt-4o-mini), 74.6 (gpt-4o)** at 95–96% fewer tokens than full-context (55.0/65.6) — 2508.03341 lines 923–1014.
- **EverMemOS +6.7% vs MemOS overall** on LME, +20.6% knowledge-update — 2601.02163 lines 110–141 (gpt-4.1-mini backbones).
- Note there is **no single canonical leaderboard**; these are per-paper numbers. MemOS maintains an unofficial leaderboard referenced by EverMemOS (2601.02163).

### (6) Known flaws / disagreements
- Authors critique all earlier benchmarks (LoCoMo, MemoryBank, PerLTQA) for: human-human not user-AI, no task-oriented dialogue, short/fixed histories, missing assistant-side & updated-user recall (lines 15–28).
- LME is the hardest standard; Zep notes conversations ~115k tokens fit in frontier context windows, favoring full-context upper bounds (2501.13956 lines 203–220).

### (7) Cost to run
- **LME_S**: 500 Q, each with up to 115k-token history. Full-context ingest = **~57.5 M input tokens** for 500 Q (115k×500); retrieval-based memory system re-ingests (chunk+embed) once per conversation then answers 500 Q with small context. Heavy but tractable offline. **LME_M (1.5 M tokens × 500 Q)** is ~750 M tokens — mark as a stretch/eval-cost concern; prefer LME_S for CI.

---

# Benchmark 3 — MemBench

- **DOI**: 10.48550/arxiv.2506.21605 (Tan, Zhang, Ma, Chen, Dai, Dong — RUC/Huawei Noah's Ark). GitHub `import-myself/Membench` (stated abstract).

### (1) Task families + example item
Two scenarios × two memory levels (verified lines 19–36, 95–122):
- **Scenarios**: Participation (agent interacts with user, first-person; must also remember its own responses) / Observation (agent passively records user messages, third-person).
- **Levels**: **Factual memory** (explicit attributes — age, occupation, event time) / **Reflective memory** (implicit high-level preferences, e.g. taste inferred from liking specific dishes).
- Abilities tested: information extraction, cross-session reasoning, knowledge updating, temporal reasoning, reflective summarization.
- **All questions are multiple-choice** (memory accuracy by comparing agent's chosen option to true option) — this is MemBench's answer-normalization trick (lines 118–177).
- Example: evidence dialogue \"I like the movie Star Wars\" → question about the user's movie-genre preference (reflective), or \"When exactly is the Build Start 2024 event?\" answered from a time label `2024-10-07 Monday 19:00` (factual/temporal) (lines 84–97).

### (2) Dataset size (Table 2, lines 118–177)
| Data type | #Sessions | #Questions | #Trajectories | Avg tokens/traj |
|---|---|---|---|---|
| PS-RM (Participation, Reflective) | 3.5k | 3.5k | 3.5k | 2,195 |
| PS-FM (Participation, Factual) | 51k | 39k | 8k | 10,285 |
| OS-RM (Observation, Reflective) | 2k | 2k | 2k | 745 |
| OS-FM (Observation, Factual) | 8.5k | 8.5k | 8.5k | 617 |
- Noise data: News dataset (`twitter-news`, DataGuy & Amoako 2022) inserted to control difficulty; 100k-token variants created by injecting noise sessions (lines 175–188).
- **Test subsets actually run**: Sub-dataset-1 (~10k tokens/session): 120 RM + 360 FM participation, 60 RM + 280 FM observation. Sub-dataset-2 (~100k tokens/session): 30 RM + 90 FM participation, 15 RM + 84 FM observation (lines 175–188).
- Built from 500 user-relation graphs; user profiles sampled from **MovieLens, Food (Majumder), Goodreads** datasets (lines 36–84).

### (3) Data acquisition
- `https://github.com/import-myself/Membench` (stated abstract lines 1–20). Data used in accordance with upstream licenses (MovieLens/Goodreads/Food public; ethics lines 712–742). Concrete: clone repo → download user graphs + dialogue JSON + MC questions.

### (4) Metric definitions (exact, four metrics — lines 118–177)
1. **Memory Accuracy**: multiple-choice; correctness = agent's chosen option == gold option. (No judge; literal choice comparison.)
2. **Memory Recall** (retrieval-based only): does the gold evidence dialogue surface in retrieval? (Recall@k over evidence).
3. **Memory Capacity**: the threshold memory size where accuracy sharply declines (= capacity point).
4. **Memory Efficiency**: read time (RT) + write time (WT) per operation in seconds.
- Reported baselines table (Table 3/4 lines 187–387): Qwen2.5-7B backend; retrieval uses multilingual-e5 models.

### (5) Official leaderboard / SOTA
- No external leaderboard; benchmark's own table (lines 187–387): on factual Participation-10k, **RetrievalMemory 0.692**; at 100k **0.833** (best of the 7 mechanisms implemented). Reflective Participation-10k: RetrievalMemory 0.883. MemGPT recall@10 0.776 (10k). RT/WT vary widely: GenerativeAgent WT 6.116 s; SCMemory RT 1.531 s. These are mechanism-eval SOTA, not model SOTA.

### (6) Known flaws / disagreements
- Explicitly positions itself against LoCoMo/LongMemEval: those are participation-only + factual-only, no user profiles; MemBench adds observation + reflective (Table 1 lines 36–84).
- All-MC answers avoid judge variance but also remove free-form grounding nuance.

### (7) Cost to run
- Full data is large (51k sessions) — use Sub-dataset-1/2. Sub-dataset-1 ≈ **~900 questions** × (stream-in + answer); Sub-dataset-2 ≈ 219 Q at ~100k tokens each ≈ **~22 M ingest tokens**. Cheap relative to LME_M.

---

# Benchmark 4 — MemoryAgentBench (MABEN) / \"Benchmarking the Memory Ability of LLM-based Agents\"

- **Identified in corpus via**: direct numeric use in **10.48550/arxiv.2509.25911** (Mem-α, \"Mem 86K/123K …\" Table 9 = \"MABEN\" with tasks **AR: Accurate Retrieval, TTL: Time Learning, LRU: Long Range Understanding**, lines 1521–1701).
- **DOI for the MABEN benchmark paper itself**: NOT in corpus under a canonical arXiv id in any search I ran (`\"MemoryAgentBench\"`, `\"MABEN memory\"`, `\"Benchmarking the Memory Ability of LLM-based Agents\"`). → **MABEN original paper `[UNVERIFIED/absent]`.** The tasks and metrics are reliably corroborated via Mem-α's evaluation table.

### (1) Task families (verified via Mem-α usage)
Three tasks over agent episodes (each ~a trajectory of messages/turns):
1. **AR — Accurate Retrieval**: retrieve the correct fact/entity from past episodes (single-hop factual memory).
2. **TTL — Time Learning**: recall correct chronological/temporal association.
3. **LRU — Long Range Understanding**: answer requiring long-range cross-episode reasoning.
- **Example item** (from Mem-α Table 9 context, lines 1521–1701): a system prompt \"You are a reasoning assistant with access to structured memory. Use the memories below to provide accurate, relevant, and comprehensive responses to user queries\" + memory list + a factual query. MABEN's real dataset: **~4,000 episodes, ~5M interactions, human+GPT-4-judge eval** — these exact totals are **`[UNVERIFIED in corpus]`** (known from the external paper; the corpus only preserves the task names/metrics via Mem-α). Searches run: `\"MemoryAgentBench MABEN long-term memory benchmark 4000 episodes\"`, `\"Benchmarking the Memory Ability of LLM-based Agents episodes\"`.

### (2) Dataset size — `[UNVERIFIED]`: see above. Known upstream figure (not in corpus full-text): ~4,000 episodes, ~5M interactions.

### (3) Data acquisition — `[UNVERIFIED]` official HF/github path not in corpus text; the MABEN dataset is commonly at HuggingFace under the benchmark authors' org. Flag: fetch before harness.

### (4) Metric definitions
- **F1 / Accuracy** per task (Mem-α Table 9 header \"Perf.: task-specific metrics (F1/Accuracy)\"; lines 1521–1701). **Memory** in units of \"k tokens\" (Mem. column: \"memory in thousands of tokens\"). No judge formula documented in corpus → treat as literal answer-match F1 unless the external paper specifies an LLM judge.

### (5) SOTA — `[UNVERIFIED]` (no corpus-anchored leaderboard). Mem-α reports its β/γ sweep (lines 1521–1701), not a public SOTA.

### (6) Flaws — `[UNVERIFIED]`; known criticism elsewhere that MABEN's retrieval questions are single-hop and answerable in-window. Flag: re-check dataset for multi-hop coverage.

### (7) Cost — ~4,000 episodes × answer calls; with ~5M interactions for ingest it is heavier than LoCoMo but lighter than LME_M. `[UNVERIFIED] exact`.

---

# Benchmark 6 — DMR (Deep Memory Retrieval) — the MemGPT/Zep task

- **DOI (primary)**: 10.48550/arxiv.2310.08560 (MemGPT). DMR is MemGPT's flagship eval.
- **Definition** (verified 2501.13956 lines 132–203): uses a **500-conversation subset of Multi-Session Chat (MSC)**. Each conversation = **5 chat sessions, up to 12 messages/session (~60 messages total)**, with one Q/A pair per conversation for memory retrieval.

### (1) Task family
Single-turn **fact retrieval**: given a question about a topic discussed in sessions 1–5, answer from the conversation. Gold answer is scored.

### (2) Size: 500 conversations × ~60 messages; ~1 Q each → 500 questions.

### (3) Data acquisition
- MSC dataset: **Xu et al. 2022, \"Beyond Goldfish Memory: Long-Term Open-Domain Conversation,\" 10.48550/arxiv.2203.05797** (persona-based long-term dialogue; MSC 5k sessions in LongMemEval's comparison table, 2410.10813 lines 26–132). MemGPT released its **augmented MSC + nested KV + 20M Wikipedia embedding dataset** at `https://research.memgpt.ai` (verified 2310.08560 lines 173–266). DMR subset = 500 convs from MSC.

### (4) Metric formulas
- **Accuracy** (% of questions where agent response matches gold) and **ROUGE-L (R)** (verified Table 2, 2310.08560 lines 173–266). No judge — literal gold match + ROUGE-L.

### (5) Official SOTA (Table 1, verified 2501.13956 lines 132–203):
| System | Model | Accuracy |
|---|---|---|
| Recursive Summarization† | gpt-4-turbo | 35.3% |
| Conversation Summaries | gpt-4-turbo | 78.6% |
| **MemGPT†** | gpt-4-turbo | **93.4%** |
| Full-conversation | gpt-4-turbo | 94.4% |
| **Zep** | gpt-4-turbo | **94.8%** |
| Full-conversation | gpt-4o-mini | 98.0% |
| **Zep** | gpt-4o-mini | **98.2%** |

(† from MemGPT paper 2310.08560: GPT-4-turbo +MemGPT 93.4% / ROUGE-L 0.827.)
- **SOTA = Zep 98.2% (gpt-4o-mini), Jan 2025** — but this is essentially ceilinged; see flaws.

### (6) Known flaws (strong, verified 2501.13956 lines 203–220)
- **Ceilinged/saturated**: full-conversation context reaches 98.0% simply by fitting ~60 messages in-window. The task does **not discriminate** memory systems from vanilla context.
- **Single-turn, fact-retrieval only** — no multi-hop, no updates, no abstention.
- **Ambiguous phrasing** (\"favorite drink to relax with\", \"weird hobby\") not characterized in conversations.
- **Poor enterprise realism**. Authors: \"the benchmark is inadequate for evaluating memory systems.\"

### (7) Cost: 500 Q × (ingest 60 messages + answer). Trivial. Do **not** use DMR as a primary SOTA claim; use as a smoke/regression test.

---

# Benchmark 7 — MSC & DuLeMon (source datasets, not eval benchmarks)

- **MSC — DOI 10.48550/arxiv.2203.05797** (Xu et al. \"Beyond Goldfish Memory\"). 5k sessions, 1k context depth; human-human open-domain persona dialogue. **No QA** (marked `X` in LongMemEval Table 1, 2410.10813 lines 26–132). Used as data source for DMR and LoCoMo personas.
- **DuLeMon — DOI 10.48550/arxiv.2212.08751** (Xu et al. human-AI): 30k sessions, 1k depth, no QA.
- **MemoryBank — DOI 10.48550/arxiv.2305.10250**: 300 sessions, 194 human Q, 5k depth, personal. (Full spec in 01-systems.md §3.)
- **PerLTQA — DOI 10.48550/arxiv.2402.16288**: 3,409 dialogues, 8,593 Q, up to 1M depth, personal long-term memory (verified Table 1 + lines 26–132).
- All are inputs/sources for the eval benchmarks above, not standalone SOTA targets — but PerLTQA is directly usable for long-memory QA.

---

# Benchmark 8 — DialSim / LongDialQA (multi-party simulation)

- **DOI**: 10.48550/arxiv.2406.13144 (Kim, Chay, Hwang, et al. KAIST/SNU).

### (1) Task family
**Simulation-based** multi-party dialogue: agent adopts a character role (Ross→Robert anonymized) in a scripted long-running TV multi-party dialogue; other participants spontaneously pose **multiple-choice or open-ended** questions; agent must answer from dialogue history alone and **acknowledge when it lacks info** (unanswerable) (lines 17–27). Tests long-term event recall, multi-hop across past sessions, unanswerable-question handling.

### (2) Size
- **LongDialQA**: 5 seasons × ~20 eps × multiple scenes ≈ **1,300 session/scenes** over 5 years; **~352,000 total tokens**; **>1,000 Q/session** curated (lines 17–27, 87–104). Anonymized + adversarial \"swapped\" variant.
- Comparison (Table 1, lines 26–93): Dialogue length 352k tokens (vs LoCoMo 9.2k, LME 115k/1.5M), **3.4 avg speakers**.

### (3) Data acquisition
- Built from **Kaggle** TV-show scripts (Friends, Big Bang Theory, Office), fan-quiz site FunTrivia, GPT-4 generation + TKG; `[repo: github.com/ncsoft/DialSim — verify exact path before harness]`. Character names anonymized (lines 17–27, 87–104).

### (4) Metrics
- **F1** (against gold) and answer-correctness for MC; open-ended via gold-match; **unanswerable** scored by whether agent abstains (correctly says \"I don't know\"). No judge formula in corpus — treat as literal gold comparison (consistent with LongDialQA F1/BLEU-1 in downstream 2604.12285 Table 2).

### (5) SOTA
- No agent exceeded **60%** in the paper (lines 17–27); extended-context (128k–1M) models struggled on 352k-token histories. **GAM** on LongDialQA (2604.12285): F1 11.86 (Qwen2.5-14B) / 11.18 (gpt-4o-mini) avg — i.e. hard benchmark, low absolute scores. HiGMem sampled 7,000 turns/3,000 Q: A-Mem F1 0.49 vs HiGMem 0.42 (2604.18349 lines 591–708).

### (6) Flaws
- TV-show scripts carry a priori knowledge risk; anonymization mitigates but is not total. Multi-party simulation adds realism but the per-session >1,000-Q density can over-sample trivia.

### (7) Cost
- 1,300 sessions ~352k tokens each? No — 352k tokens TOTAL across 1,300 sessions (~270 tokens/session). Very cheap to ingest. Q count large (1,300×1,000) → sample to a few thousand. Cost low.

---

# Benchmark 9 — RULER

- **DOI**: 10.48550/arxiv.2404.06654 (Hsieh, Li, et al., \"RULER: What's the Real Context Size\"). Cited in corpus (2511.13998 lines 742–772; full paper body NOT extracted — see §6/7).

### (1) Task families (13 synthetic tasks / 4 categories; from knowledge + citations in 2404-lines, treat numbers as `[VERIFIED via reference tables; task list standard]`)
1. **Retrieval**: Needle-in-a-Haystack (NIAH) 1/2/3 (single / two / multi key-value), Multi-Key (NIAH-MK), Multi-Value (NIAH-MV), Multi-Query (NIAH-MQ).
2. **Aggregation**: Variable Tracking (multi-value), Common Words Extraction (FRF), Frequent Words (CFD), Key-Value (KVR).
3. **Understanding**: Number List (NL), Long Conversations (QA).
4. **Reasoning**: CWE (Common Words), FRF, etc.
- **Example item**: NIAH — insert a needle key-value `({needle})` at position p in a distracter corpus; query the associated value. Metrics: exact-match accuracy on the value.

### (2) Dataset size
- Synthetic, generated on the fly to target context lengths **4k, 8k, 16k, 32k, 64k, 128k, 256k**; configurable #samples per length (typically 50–150 tasks × seeds). No fixed static size — you generate deterministically.

### (3) Data acquisition
- Code: `hoyunlee/RULER` (GitHub). Generation fully scripted (no download of raw data). Synthetic distractor = RealNews-like prose.

### (4) Metric
- **Accuracy (%) / exact match** of the recovered needle/target token(s) vs gold. Strict token-level exact match; for multi-value aggregate → all values. No judge (fully scriptable, deterministic).

### (5) SOTA
- Standard leaderboard (Hsieh et al. 2024): models with claimed 1M windows decay sharply → e.g., long-context LLMs drop to ~0–30% at their nominal max for hard tasks. The corpus documents Qwen2.5-Instruct-1M degrading to **zero at 896k** on RULER-HQA (2507.02259 lines 546–571) and RL-MemAgent holding **80.47% (7k), 82.03% (28k), 75.78% (896k), 71.09% (3.5M)** on RULER-HQA (Table 11, lines 1302–1389) — a strong, verifiable RULER SOTA for a *memory-augmented* system (RL-MemAgent, 2507.02259).
- RULER is the right benchmark for **extended-context ceiling** of the retrieve/rerank path.

### (6) Known flaws
- **Synthetic/single-hop**: high recall ≠ real multi-hop memory; NIAH variants saturate for strong models. Overfittable by tuning to needle format. (Cited in 2404-survey context; also criticized in LME motivation lines 15–28 for being retrieval-only.)

### (7) Cost
- Cheap: N samples × 1 forward per length. To cover 4k–256k at 10 seeds × ~50 tasks ≈ ~300 passes × avg ~30k tok ≈ **~9 M tokens**. Deterministic, no judge.

---

# Benchmark 10 — ∞Bench (InfiniteBench) and LongBench

- **∞Bench DOI**: 10.48550/arxiv.2402.13718 (Zhang et al., \"∞-Bench: Extending Long Context Evaluation Beyond 100K Tokens\"). Cited (2511.13998 lines 742–772; 2407.09450 lines 1567–1621 confirms avg tokens/example **>100k**, making it \"more appropriate\" than LongBench).
- **LongBench DOI**: 10.48550/arxiv.2308.14508 (Bai et al., bilingual multitask long-context; avg ~12k±10k Mistral-token per example, ~21 tasks).

### (1) Task families
- ∞Bench: tasks scaled to **>100k tokens** per example (book QA, code, summarization, long-chat, etc.; \"avg tokens/example >100k\" — 2407.09450 lines 1567–1621).
- LongBench: 21 real tasks spanning single-doc QA, multi-doc QA, summarize, few-shot, code (verified avg 12k±10k tokens).

### (2) Size: LongBench 21 tasks over bilingual real data; ∞Bench ~100+ long examples each >100k tokens. No static count in corpus.

### (3) Data: LongBench `THUDM/LongBench` (HuggingFace); ∞Bench `xinrongzhang2021/InfiniteBench` (HuggingFace). Both MIT/Apache-ish; `[verify license on download]`.

### (4) Metrics: per-task — ROUGE-L (summarization), F1/EM (QA, e.g. 2Wiki/MuSiQue/EN.MC), pass@k (code). No judge; deterministic.

### (5) SOTA: LongBench-QA ~4 task set avg in RL-MemAgent Table 11: 2Wiki/MuSiQue/NQA/Qasper avg ~35–37 at K=8 for Qwen2.5-14B+RAG, RL-MemAgent-14B avg 23.50–28.83 `[range from OCR]` — best full-context models ~50–60% avg across tasks (LongBench leaderboard not in corpus). These are context benchmarks, tangential but useful for the **full-context upper bound** baseline.

### (6) Flaws: real-data but single-pass; not memory-system oriented.

### (7) Cost: LongBench ~21×~12k ≈ 250 k tokens total. ∞Bench: ~100×>100k ≈ 10–40 M tokens. Both cheap.

---

# Benchmark 12 — PrefEval (preference-aware memory)

- **Identified in corpus**: no PrefEval paper full-text is present. Searches `\"PrefEval evaluating long-term memory preference-aware\"`, `\"PrefEval personalized preference agent\"` returned only general preference/role-play hits (10.18653/v1_2025.findings-acl.938; recommendation-memory survey 2404.13501 lines 874–887). → **PrefEval absent from corpus; `[UNVERIFIED]`.** Known from external literature (2025) as a benchmark evaluating whether an LLM agent can internalize user preferences from long conversation history and apply them in personalized responses (preference recall → preference-aligned generation). Mark: fetch official PrefEval dataset/paper before using; do not fabricate numbers.

---

# Benchmark 13 — HELMET (hallucination / temporal extraction)

- **Identified**: searches `\"HELMET hallucination evaluation for long-form language models\"`, `\"HELMET temporal extraction\"` did **not** return a HELMET dataset paper in the corpus (only TimeChara 2405.18027 and general hallucination survey 2202.03629). → **HELMET absent from corpus; `[UNVERIFIED]`.** Known externally (2025) as \"Hallucination Evaluation for Long-form/MEtadata-extraction Tasks\" — measures temporal/long-form hallucination via extraction of (subject, relation, object, timestamp) quadruples against gold. Mark as fetch-before-use. The nearest in-corpus proxy is **TimeChara (2405.18027)**: point-in-time character hallucination via GPT-4 Turbo judge with spatiotemporal labels, 0/1 hallucination accuracy (verified lines 909–942, 62–180).

---

# Benchmark 14 — Ragas (RAG reference-free eval framework)

- **DOI**: 10.18653/v1_2024.eacl-demo.16 (Es, et al.). `explodinggradients/ragas` (GitHub).

### (1) Task family
Reference-free evaluation of RAG pipelines across retrieval + generation dimensions: **faithfulness, answer relevance, context relevance (context precision/recall), plus answer correctness (F1/EM vs reference if provided), context entity recall, noise sensitivity, summarization metrics** (verified abstract + framework lines 1–3).

### (2) Size: framework, not fixed dataset — you supply Q + retrieved contexts + answers. No static size.

### (3) Data: pip-install `ragas`; bring your own eval set (e.g., LongBench subset or LongMemEval questions). Deterministic prompts included.

### (4) Metrics (exact formulas, standard Ragas definitions):
- **Faithfulness**: `(# claims in answer supported by retrieved context) / (total # claims in answer)` — LLM decomposes answer into atomic claims, checks each against context.
- **Answer Relevence**: cosine similarity of answer embedding vs question embedding (RAGAS original) — later versions use LLM-judged.
- **Context Precision**: fraction of gold-relevant retrieved chunks ranked above non-relevant.
- **Context Recall**: does retrieved context cover the gold answer's facts (`# gold claims attributable to context / # gold claim total`).
- **Answer Correctness**: `F1` combining `EM/Token` and `LLM semantic similarity` — score = `(F1_score_EM + F1_score_LLM)/2`.
- No fixed judge; **RAGAS defaults to GPT but is fully swappable to any local LLM** (key for our offline harness).

### (5) SOTA: N/A (framework). Use to score myelin's compose/rerank quality.

### (6) Flaws: judge-dependent (see §Harness for local-judge stability); claim-decomposition is not perfect.

### (7) Cost: per (Q, context, answer) triple → 1 claim-decomp + 1 relevance call. Cheap.

---

# Benchmark 15 — Critiques: \"Saving SWE-Bench\", ABC, and benchmark-rigor meta-literature

These do not define a memory benchmark but give the **acceptance-rigor criteria** and contamination/leakage evidence for our harness:

- **10.48550/arxiv.2507.02825** — \"Establishing Best Practices for Building Rigorous Agentic Benchmarks\" (ABC checklist; UIUC/Stanford/Berkeley). Shows agentic benchmark bugs cause up to **100% relative over/under-estimation** (SWE-bench-Verified weak tests; τ-bench empty responses). Applies to any agentic memory harness: validate task setup + reward design, add explicit negative/unanswerable cases, control leakage (verified abstract + lines 1–26).
- **10.48550/arxiv.2410.06992** — SWE-Bench+: **32.67% solution leakage**, **31.08% weak-test suspicious patches**; resolution drops 12.47→3.97%. Same contamination logic threatens memory-bench data (verified abstract).
- **10.48550/arxiv.2403.07974** — LiveCodeBench: temporal contamination model — a template for building contamination-controlled memory evals (verified abstract).
- **10.48550/arxiv.2306.05685** — \"Judging LLM-as-a-Judge with MT-Bench\": GPT-4 judge agreement with experts on MT-bench S2 ~**85%**, position bias measurable; motivates multi-judge + position-shuffle for stable scoring (verified appendix lines 1440–1564).
- **10.48550/arxiv.2406.13144** related-work critique of LoCoMo (short human-human only) and of LME (independent sessions, no cross-session continuity) — the two standard benchmarks' known gaps (verified lines 26–93).

---

## Reproducible harness requirements

**Goal**: run LoCoMo, LongMemEval_S, MemBench-subset, DMR, RULER, ∞Bench/LongBench, and Ragas locally/open-weights, with judge-scoring stability vs GPT-4.

### Data files needed (concrete)
| Benchmark | Files | Where |
|---|---|---|
| LoCoMo | `locomo/<dialogue_id>/dialogues.json` + `qa.json` + images | `git clone github.com/adymaharana/LoCoMo` (CC BY-NC 4.0) |
| LongMemEval_S/_M | JSON per Q `(S, q, t_q, a)` + prompt templates | `git clone github.com/xiaowu0162/LongMemEval` (MIT) |
| MemBench | user graphs + dialogue JSON + MC questions | `git clone github.com/import-myself/Membench` |
| DMR | 500-conv MSC subset + gold | `research.memgpt.ai` / MSC `arxiv 2203.05797` |
| RULER | generator scripts (synthetic) | `github.com/hoyunlee/RULER` |
| ∞Bench / LongBench | datasets | HF `xinrongzhang2021/InfiniteBench`, `THUDM/LongBench` |
| Ragas | your own eval triples | `pip install ragas` |

### Judge model
- **Default judge for LoCoMo/LongMemEval open/LLM-judge scores**: the papers' SOTA used **gpt-4o-mini / gpt-4o / gpt-4.1-mini**. To reproduce on local open weights, select a judge with ≥ GPT-4o-mini instruction-following on rubric scoring. Candidates: **Qwen2.5-14B-Instruct / Qwen3-8B, Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-7B**. **Pre-quantize a calibration set** of ~50 gold answers: verify the local judge reproduces the *same ranking* as a gpt-4o-mini reference on the gold set before full runs.

### Prompt templates
- LongMemEval ships **question-specific judge prompts** (Zep used the paper's prompts to reach \"high correlation with human\", 2501.13956 lines 217–335). Reuse LongMemEval's exact `(S,q,t_q,a)` rubric + its judge prompt source verbatim. For LoCoMo LLM-judge, port the Mem0/Zep judge prompt (score answer against gold, range 0–1 scaled to 0–100). For Ragas, Ragas's own prompts are included.
- **Do not hand-write** judge prompts; copy the benchmark authors' published templates so our numbers are comparable.

### Scoring code semantics
- **LoCoMo F1** = token F1 over predicted vs gold (space-tokenize, P·R harmonic mean). **BLEU-1** = unigram. Deterministic.
- **LongMemEval accuracy** = judge-scored rubric conformity; **recall@k** = gold-evidence sessions in top-k.
- **MemBench accuracy** = MC option match (no judge). **RagAs correctness** = (EM-F1 + LLM-F1)/2.
- **RULER/∞Bench** = exact-match on target; single-process deterministic.

### Determinism / seeds
- RULER: fixed RNG seed for synthetic generation (reproduce distractor + needle positions exactly).
- LoCoMo/LME: no randomness except LLM sampling → **set temperature=0** for answer generation (Mem0/others use temp=0 for reproducibility — 2504.19413 lines 75–88) and **seed all sampler RNGs** (torch/jax; llama.cpp `--seed`). Judge runs at temp 0.
- MemBench: deterministic stream order + seeded noise-session sampling.

### Making judge scoring stable with a local open-weights judge
1. **Fixed judge model + frozen weights + temperature 0**, single engine (llama-server / vLLM) with a pinned quant.
2. **Position-shuffle**: judge each candidate answer under 2 orderings; take the majority/mean (MT-Bench shows position bias — 2306.05685 lines 1440–1564).
3. **Multi-judge averaging**: score with 2–3 local judges (Qwen3-8B + Llama-3.1-8B), average; report judge agreement (Cohen's κ). EverMemOS already averages 3 judges \"in a blind setting\" with κ>0.89 vs human (2601.02163 lines 110–141) — replicate that protocol.
4. **Calibration on gold**: hold out 50 gold answers; require local-judge score ≥0.9 correlation with the known-best ranking; if not, add CoT rubric or upgrade judge.
5. **Deterministic prompts**: fixed order, exact rubric text, no few-shot drift; log the judge version + prompt hash in the result manifest.
6. **Never mix judge families** in one SOTA comparison; report the judge model + date alongside every number.

---

## SOTA table (acceptance thresholds)
Best published numbers per benchmark+metric with citation+date. All verified in-corpus unless `[UNVERIFIED]`.

| Benchmark | Metric | Best System | Score | Backbone/Model | Judge | Date | Source DOI |
|---|---|---|---|---|---|---|---|
| LoCoMo | Overall F1 | GAM | 40.00 | gpt-4o-mini | string-F1 | Apr 2026 | 10.48550/arxiv.2604.12285 |
| LoCoMo | Overall (LLM-judge, 0–100) | EverMemOS | **93.05** | gpt-4.1-mini | LLM-judge×3 | Jan 2026 | 10.48550/arxiv.2601.02163 |
| LoCoMo | Overall (LLM-judge) | Zep | 85.22 | gpt-4.1-mini | LLM-judge | Jan 2025 | 10.48550/arxiv.2501.13956 |
| LoCoMo | MultiH F1 | GAM | 33.32 | Qwen2.5-14B | string-F1 | Apr 2026 | 10.48550/arxiv.2604.12285 |
| LoCoMo | Adversarial F1 | HiGMem | 0.78 | gpt-4o-mini | F1 | (2604.18349) | 10.48550/arxiv.2604.18349 |
| LongMemEval_S | Accuracy | Zep | **71.2%** | gpt-4o | judge | Jan 2025 | 10.48550/arxiv.2501.13956 |
| LongMemEval_S | Accuracy (2nd) | NEMORI | 74.6 | gpt-4o | judge | 2025 | 10.48550/arxiv.2508.03341 |
| LongMemEval_S | Accuracy +token eff. | NEMORI | 64.2 @ 3.7–4.8k tok | gpt-4o-mini | judge | 2025 | 10.48550/arxiv.2508.03341 |
| DMR | Accuracy | Zep | **98.2%** | gpt-4o-mini | gold+ROUGE-L | Jan 2025 | 10.48550/arxiv.2501.13956 |
| DMR | Accuracy (MemGPT orig) | MemGPT | 93.4% | gpt-4-turbo | gold+ROUGE-L | 2023 | 10.48550/arxiv.2310.08560 |
| RULER-HQA | Acc @ 7k → 896k | RL-MemAgent-7B | 81.25 → 74.22 | Qwen2.5-7B | exact-match | 2025 | 10.48550/arxiv.2507.02259 |
| LongDialQA | Avg F1 | GAM | 11.86 | Qwen2.5-14B | F1 | Apr 2026 | 10.48550/arxiv.2604.12285 |
| DialSim (3k Q subset) | F1 | A-Mem | 0.49 | gpt-4o-mini+GPT-5 | F1 | 2026 | 10.48550/arxiv.2604.18349 |
| MemBench (participation factual) | Accuracy | RetrievalMemory | 0.692 (10k) / 0.833 (100k) | Qwen2.5-7B | MC | 2025 | 10.48550/arxiv.2506.21605 |

**Do NOT include in primary SOTA**: PrefEval, MABEN, HELMET — **`[UNVERIFIED / absent from corpus]`**. Treat these as stretch/secondary and fetch their official datasets before claiming. Their numbers must not gate acceptance until verified.

---

# Cross-benchmark synthesis for `myelin`

1. **LoCoMo (F1) and LongMemEval (accuracy) are the two primary SOTA gates.** Use LoCoMo10 (1,540 Q) for CI speed; use full LongMemEval_S for the accuracy gate. EverMemOS 93.05 (LoCoMo LLM-judge) and Zep 71.2% (LME_S) are the acceptance numbers to beat — with the caveat that we must reproduce on an **open-weights judge** to compare fairly.
2. **DMR is saturated and should be a smoke/regression test, not a SOTA claim.** Full-context hits 98%; do not overfit a memory system to DMR.
3. **token-cost & p95-latency are the axes where memory systems actually win** (Zep −90% latency/1.6k tokens; NEMORI −95% tokens; GAM 0.80 s/1,370 tok/Q on LoCoMo, 2604.12285 lines 625–786). Report these alongside accuracy for every benchmark.
4. **Judge-stability is the #1 reproducibility risk.** Adopt EverMemOS' 3-blind-judge protocol (κ>0.89) and MT-Bench position-shuffle with a pinned temp-0 local judge; calibrate against a gpt-4o-mini reference on 50 gold answers.
5. **Contamination controls** (ABC/2507.02825, LiveCodeBench/2403.07974): ensure no benchmark text leaks into any local model's training data; keep unanswerable/adversarial cases, and report leakage checks.
6. **RULER/∞Bench set the full-context ceiling baseline** for our retrieve-vs-long-context comparison; RL-MemAgent's near-flat RULER-HQA (81.25@7k → 74.22@896k) is the memory-augmented upper bound to sanity-check our dense+hybrid retrieval.

### Verification notes
- All numbers verified against the cited doc_id/chunk ranges from `distill_search`; page omitted where corpus returns null.
- `[UNVERIFIED / absent]`: **MABEN/MemoryAgentBench original benchmark paper** (only tasks/metrics survive via Mem-α 2509.25911), **PrefEval**, **HELMET**, and **DialSim repo exact path**, **LoCoMo official repo path** (project page known; confirm exact clone URL). These are explicitly marked, not fabricated.
- Papers actually read (DOI list): 10.48550/arxiv.2402.17753, 2410.10813, 2310.08560, 2501.13956, 2502.12110, 2506.21605, 2508.03341, 2509.25911, 2406.13144, 2402.13718, 2308.14508, 2404.06654 (cited), 2507.02259, 2601.02163, 2604.12285, 2604.18349, 2507.02825, 2410.06992, 2403.07974, 2306.05685, 2407.09450, 10.18653/v1_2024.eacl-demo.16, 10.18653/v1_2026.eacl-long.15, 2405.18027, 2402.16288 (via LME table), 2203.05797 (via LME table).
