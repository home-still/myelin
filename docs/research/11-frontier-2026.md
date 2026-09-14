# Frontier 2026: Evidence Pack for a 2026-Competitive Rust Memory System (`myelin`)

Seven 2026 arXiv papers (the actual frontier, not yet in the local corpus) + the BEAM benchmark + vendor claims. Every number carries a source (arXiv id + section) and a provenance flag `paper|vendor`. Provenance flags: **paper** = reported by the paper's preprint; **vendor** = reported by a vendor blog (mem0.ai), NOT reproducible from any paper.

---

## 1. LongMemEval-V2 (arXiv 2605.12493) — THE anchor. [paper]

> Wu, Ji, Kawatkar, et al. (UCLA). arXiv:2605.12493v1, 12 May 2026. Project: https://xiaowu0162.github.io/longmemeval-v2/

### 1.1 Core thesis / framing (defines our eval vocabulary)
> "A high-quality memory makes an agent an **experienced colleague** in a specialized environment." (§1)

LME-V2's central conceptual move — **memory as becoming an experienced operator of an environment, not memory as QA recall**: Existing memory benchmarks (LoCoMo, LongMemEval-V1, PersonaMem, BEAM) evaluate **conversational QA-recall** over user histories. LME-V2 instead evaluates whether a memory system internalizes **environment-specific experience** — interface affordances, state dynamics, workflows, recurring failure modes, and wrong-premise awareness — from web-agent trajectories. This distinction is the load-bearing framing for `myelin`'s eval design: the metric is not F1 over user-chat facts but whether the system turns accumulated trajectories into reusable *environment competence*. (§1, §2)

### 1.2 Five memory-ability taxonomy (the eval axes — adopt verbatim) (§3.1)
1. **Static State Recall** — landmarks, layouts, module affordances, subtle differences across states.
2. **Dynamic State Tracking** — act as a world model: given states+actions, understand how the environment changes.
3. **Workflow Knowledge** — steps to perform common tasks.
4. **Environment Gotchas** — awareness of common recurring issues and how to avoid environment-specific failures.
5. **Premise Awareness** — recognize assumptions valid in another environment but wrong in the current one (abstention / flawed-premise tests).

### 1.3 Dataset sizes & construction (§1, §3.2, §A.1, §A.4)
- **451 manually curated questions**; avg 1.4 answer-bearing trajectories/question (min 1, max 5).
- History haystacks: **LME-V2-Small = 100 trajectories, ~25M tokens (shared per domain)**; **LME-V2-Medium = ~500 trajectories, ~115M tokens (question-specific)**. (§1, §3.2)
- Multimodal: each trajectory state = screenshot + accessibility tree + action. Question images included. Table 1 marks LME-V2 as the only benchmark with ✓ across ALL of static/dynamic/workflow/gotchas/premise.
- Trajectories: 599 from WebArena (OneStopShop/CMS/Reddit) + 941 from WorkArena/WorkArena++ (ServiceNow); AgentLab harness; GPT-5.2 / GPT-5-mini / Codex; 28.1 states/trajectory; 52.0% success; failure trajectories included (many questions answerable ONLY from failed runs). (§3.2, §A.1)
- Goal sanitization removes navigational hints so procedure questions can't be answered by reading the route. (§A.1)
- Answer-trajectory labeling via a Codex-assisted + human-verified coverage map; minimal answer core 44 (WebArena) + 49 (ServiceNow) trajectories. (§A.3, §A.4)
- **Parametric-knowledge filter is strict**: questions validated so that >=2 of {Gemini-3-Pro, GPT-5.2, Grok-4.1-thinking, Claude-Opus-4.6} answer incorrectly with no context (best no-context model = **14.1%** overall, Table 6). (§3.4, §A.2)

### 1.4 Evaluation formulation (context gathering — the exact protocol to replicate) (§3.3, §A.5)
- A memory system implements exactly two APIs: `Insert(h)` (consume one trajectory) and `Query(q)` (return multimodal memory context).
- For each question: sequentially `Insert` all trajectories in the haystack, then `Query`, truncate returned context to **200K tokens**, hand question+context to a **fixed reader LLM** (Qwen3.5-9B, temp 0.6, top_p 0.95, top_k 20), parse the last `\boxed{}` expression (UNKNOWN = incorrect).
- Scoring: structured answers -> deterministic evaluators (normalized phrase matching, ordered phrase matching, single-choice, multi-select). Gotchas + abstention -> **LLM judge = GPT-5.2 (medium reasoning)** on a binary label. (§A.5)
- Reader system prompt: "You are an experienced colleague in a customized ... If you do not know the answer, output exactly \boxed{UNKNOWN}. ... If you believe the question's construction/premise is wrong, provide an explanation..." — this is the abstention mechanics. (§A.5 Table 4)

### 1.5 Reported baseline numbers PER SYSTEM (§5.1 Table 2; reader always Qwen3.5-9B)

| Method | Tier | Overall | Static | Dynamic | Workflow | Gotchas | Latency |
|---|---|---|---|---|---|---|---|
| No retrieval | both | 0.013 | 0.000 | 0.008 | 0.094 | 0.138 | 0s |
| RAG query->slice | Small | 0.428 | 0.471 | 0.425 | 0.415 | 0.207 | 0.1s |
| | Med | 0.381 | 0.434 | 0.405 | 0.293 | 0.242 | 0.1s |
| RAG slice+notes | Small | 0.510 | 0.524 | 0.496 | 0.528 | 0.414 | 0.2s |
| | Med | 0.459 | 0.487 | 0.472 | 0.434 | 0.310 | 0.3s |
| **AgentRunbook-R** (3-pool RAG) | Small | **0.586** | 0.661 | 0.583 | 0.528 | 0.310 | **26.9s** |
| | Med | **0.570** | 0.630 | 0.614 | 0.472 | 0.345 | **25.8s** |
| Codex (off-the-shelf) | Small | 0.699 | 0.804 | 0.670 | 0.575 | 0.586 | 177.2s |
| | Med | 0.687 | 0.783 | 0.646 | 0.613 | 0.517 | 185.8s |
| **AgentRunbook-C** (scaffolded coding agent) | Small | **0.749** | 0.820 | 0.724 | 0.726 | 0.483 | **108.3s** |
| | Med | 0.701 | 0.788 | 0.701 | 0.613 | 0.449 | **139.9s** |

Bold markers in paper: ✣ = significantly outperforms non-ablation baselines (paired bootstrap, p<0.05). Abstract headline: **AgentRunbook-C 72.5% avg accuracy vs strongest RAG 48.5% and off-the-shelf coding agent 69.3%**; AgentRunbook-C ~32% faster than Codex at query. Best RAG = AgentRunbook-R 58.6%/57.0%.

- **Pilot study (oracle, non-abstention)** (§3.4): full oracle trajectories are NOT sufficient (Qwen3.5-9B 59.6%, GPT-5.4-mini 65.3%). Oracle **slices+notes** -> Qwen 82.5%, GPT-5.4-mini 86.3%; **Codex direct QA -> 89.7%**. Lesson: detailed multi-step evidence inspection (a coding agent) beats direct long-context prompting.
- **Error decomposition** (§D.1): AgentRunbook-R reduces retrieval+reading errors vs RAG but does NOT improve abstention (presents evidence that misleads reader into using it instead of rejecting). AgentRunbook-C also improves abstention because the memory module is instructed to explicitly flag wrong premises/contradictions. (Design implication: a memory system must actively detect premise violations, not just fetch similar content.)

### 1.6 AgentRunbook method detail (re-implementable) (§4, Appendix C)
**AgentRunbook-R** — 3 knowledge pools, LLM controller (Qwen3.5-9B, temp 0.6, top_p 0.95):
1. **Raw-state slice pool**: per-state entry = radius-1 local window (URL, action, AXTree text, screenshots, goal, action sequences).
2. **State-transition event pool**: per adjacent-state transition, LLM-generated `{overview, state_transition}` summarizing what changed (new page, revealed panel, fields, values, confirmation signal, blocker, popup, navigation).
3. **Procedure + hint note pool**: per trajectory, `{procedure_note, hint_note}` each `{title, description, content}` — reusable workflow vs durable environment gotchas/absent functionality.

Query: controller emits JSON `{"raw_state_queries":[...<=5], "event_query":..., "note_query":...}`; top-6 events, top-3 notes, top-m raw states (m=min(2,6//n_queries)); dense retrieval via **Qwen3-Embedding-8B**, input truncated to 4096 tokens. vllm on A100s. (§C.2)

**AgentRunbook-C** — stores raw trajectories as files; at query time runs a coding agent (Codex v0.117.0 + GPT-5.4-mini, xhigh) in a sandbox augmented with 3 scaffoldings:
1. Workflow doc (`INSTRUCTION.md`) instructing it to act as a memory module, classify first, shortlist via manifests, use the helper script, keep evidence <=20 states.
2. **Query-time manifest artifacts** (trajectory concise + full summaries) for triage before detailed inspection.
3. **Trajectory-inspection helper script** (`inspect_trajectory.py --state/--span/--match`).
Output: `memory_module_output.json` = `{"memory_markdown": ..., "trajectory_spans":[{"trajectory_id", "start_state_index", "end_state_index"}]}` (<=20 states, zero-based inclusive), then rendered to the reader. (§C.3)
Tool-call analysis (§D.2): AgentRunbook-C shifts work from raw exploration (Codex ~21.8 raw-exploration cmds) to harness-guided retrieval (18.0 harness cmds + 1.2 raw exploration).

### 1.7 Limitations (declared §E.1)
- Web-agent domain only (not coding agents / computer-use / enterprise agents).
- Evaluates over pre-collected histories, not online learning; may miss distribution shift from the agent's own evolving behavior.
- Context-gathering formulation measures whether memory returns useful evidence for a fixed reader, not end-to-end task success (planning/tool-use/action execution out of scope).
- Methods are retrieval/file-organization designs, not new architectures/training.

### 1.8 Release
Code + derived benchmark artifacts (trajectory traces incl. AXTree + screenshots) planned under **Apache-2.0**; they do NOT redistribute the base WebArena/WorkArena harness. (§E.2)

---

## 2. Hippocampus (arXiv 2602.13594) — system-level token-free substrate. [paper]

> Li, Cao, Ahmed, Sharma, Li (UT Dallas + HPE). arXiv:2602.13594v1, 14 Feb 2026. Under revision ("Preprint").

### 2.1 Core mechanism (§1, §3, §3.2, Fig 5-6)
A contextual memory module built on a **Dynamic Wavelet Matrix (DWM)** — an append-friendly extension of the static wavelet matrix — that **co-indexes TWO streams in the compressed domain**:
- **Content DWM**: memory stored as **lossless token-ID integer sequences** (exact reconstruction). Token -> integer -> binary bit-plane matrix, supports `access/rank/select` in O(log sigma).
- **Signature DWM**: **compact binary signatures** via **Random Indexing (semantic hashing / LSH)**. Each token mapped to a d-bit binary signature via random-hyperplane projection; similar items land at small Hamming distance.

Query pipeline: LLM prompts extract keywords from a natural-language query -> convert to binary signatures -> **Hamming-ball search** over the Signature DWM (native bitwise XOR+popcount, O(n*d/w)) -> candidate metadata (alpha/beta offsets) -> exact content reconstructed from Content DWM. Avoids dense-vector k-NN and graph traversal entirely. Scales linearly.

Complexity (§Appendix F): insertion O(n log n) build, O(log sigma + log n) per append; space O(n log sigma) bits; query O(n*d/w) (d modest, e.g. a few hundred bits). Accuracy-gap theory (Appendix G): Hamming distance ~ cosine similarity; d=O((1/eps^2)log(N/delta)) bits suffices for (eps,delta)-accuracy.

### 2.2 Quantitative results [paper]
- **31x faster end-to-end retrieval**, **14x lower per-query token footprint** vs SOTA, while "maintaining accuracy" on LoCoMo and LongMemEval (abstract, §1).
- Memory **construction: 6.70 minutes, 0 LLM tokens** on LoCoMo — 5.3x faster than fastest baseline (A-mem 35.69 min) and zero token cost (vs A-mem 19,926; MemGPT 50,674; MemoryOS 41,540 tokens). (§Appendix E Table 5)
- Latency breakdown of prior systems: vector search = 85% (ReadAgent), 81% (MemoryBank), ~half (A-Mem 48%, MemoryOS 47%) of recall latency. (§2.2)
- Performance analysis (Fig 3): none of {ReadAgent, MemoryBank, MemGPT, A-Mem, MemoryOS, MemOS} simultaneously hits high accuracy + low latency + low tokens -> design-space gap. (§2.2)

### 2.3 Limitations / notes
- Accuracy is **NOT at SOTA**: on LongMemEval-M (Table 3, LLM-as-Judge with GPT-5), Hippocampus scores 2.40 overall (highest/near-highest accuracy among the six construction/retrieval-only modules there, but the whole table is low — these are NOT LLM-controller agents). It wins on **speed/cost**, not reasoning quality. The paper does not compare against AgentRunbook-style LLM controllers; it positions itself as a *substrate*, not a full memory policy. (§Appendix C Table 3)
- Signature/RI tuning: random-index dimension D in {256..2048} and signature length d in {16..128}; larger D/d -> better recall but slower search; F1 stays ~37-38% on the LoCoMo construction ablation (Table 4) — raw retrieval F1 is modest; value is efficiency, semantics are delegated to an LLM at query time. (§Appendix D)
- Preprint, "under revision"; no peer-review confirmation.

### 2.4 Release
Not stated on the abstract page; PDF/TeX on arXiv. HPE-affiliated. [paper — no code URL verified]

---

## 3. MemPro (arXiv 2606.00619) — memory pipeline as an evolvable program. [paper]

> Liu, Wang, Wu, et al. (ECNU + Xiaohongshu). arXiv:2606.00619v1, 30 May 2026. Code: https://github.com/wanghai673/MemPro

### 3.1 Core mechanism (§1, §3, §4)
- Framing: existing agentic memory systems follow a **Memory Construction–Retrieval (MCR) pipeline** but adapt mainly the memory bank while keeping the pipeline fixed -> fails on (a) task heterogeneity (temporal vs multi-session vs knowledge-update need different strategies) and (b) memory-pipeline misalignment as the bank grows.
- **MemPro treats the ENTIRE MCR pipeline as an evolvable program** (prompts + executable code). Maintains a **version tree** of runnable implementations, each node = pipeline F_v + evaluation log L_v.
- **Evolving Agent** (Codex harness + gpt-5.4-medium) iterates: **Select** promising node (high score, generalizable improvement directions, frequent failure modes) -> **Expand** via Edit/Debug/Terminate inner loop guided by failure-mode diagnostics (edit pipeline code or prompts; debug on representative failure cases) -> **Evaluate** on a small train split and log category-level scores.
- Base pipeline: Memory Agent distills input to memory bank via an abstraction prompt; Research Agent does k rounds of {Retrieve -> Integrate -> Reflect} until sufficient; three retrieval channels: **BM25 keyword + BAAI/bge-m3 semantic + PAGE-ID structural**. (§3, §A.5)

### 3.2 Quantitative results [paper] (metrics: LLM-as-Judge with GPT-4o-mini; word-F1 for QA)
- LongMemEval (gpt-4o-mini) **Avg 79.00 @ MemPro-15** (best static GEPA 75.60; Mem0 53.51; A-MEM 62.20; GAM 72.40). Qwen3-30B-A3B -> **80.80**.
- LoCoMo (gpt-4o-mini) **Avg 84.93 @ MemPro-15** (GEPA 79.50; GAM 80.45; SimpleMem 72.08; FullText 73.83; RAG 63.64; Mem0 36.49). Qwen3-30B -> **77.85**.
- **MemPro-5 already beats every non-MemPro baseline** on every benchmark+backbone. Continues improving: +1.59/+1.06 (LME/LoCoMo) 5->10, +1.95/+1.20 10->15. (§5.2)
- HotpotQA word-F1 (gpt-4o-mini): 56K 70.32 / 224K 65.81 / 448K 64.02 @ MemPro-15 (vs GAM 63.22/61.46/59.81). NarrativeQA 38.12 (vs GAM 36.86). (§5.2)
- **Ablations (§5.4)**: w/o Code (prompt-level only) -> clear drop: system-level code edits matter. w/o Version Tree (chain evolution) -> worse: tree preserves strong history. w/o Evolution -> worse: multi-round matters. w/o Iterative Expansion (one-shot edit) -> worse.
- **Retrieval ablation (LoCoMo, gpt-4o-mini) (§A.6)**: removing BM25 drops 84.93->72.25; removing embedding drops 84.93->82.57; removing PAGE-ID drops 84.93->84.37. **BM25 (exact keyword) is the single most important channel** for long-term conversational memory.
- Case study (§A.1): evolution fixes temporal wording (+1.77), question-type-aware integration (+1.78), count/duration reasoning (+1.24), adaptive retrieval depth (+1.34), focused evidence snippets (+1.47) -> **+7.60 over initial framework**; some intermediate versions regress (tree mitigates).

### 3.3 Cost
- MemPro achieves SOTA with fewer tokens than Full Text, comparable to GAM/GEPA; lightweight methods cheaper but much lower accuracy. Favorable performance-cost trade-off (§5.5). Evolution cost is **offline** and amortized over reuse (§7); token cost reported for task-time inference only, excluding offline evolution (§A.5).

### 3.4 Limitations (declared §7)
- Offline evolution stage adds cost beyond task-time inference (amortized on reuse).
- Needs a capable Evolving Agent to edit/debug runnable pipelines.
- Tree-structured evolution (single-parent); graph/merge topologies unexplored.

### 3.5 Release
Code: **https://github.com/wanghai673/MemPro** (abstract). [paper]
