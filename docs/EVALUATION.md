# myelin — agentic evaluation specification

This is the load-bearing document. `PLAN.md` describes a memory system; this describes the machinery that
decides whether that system is actually state of the art, and it is specified first because every design
choice in `PLAN.md` is answerable to a number defined here.

Everything below is grounded in one of three things, and says which: **[probed]** measured against live
systems or real data this session; **[source]** read out of the benchmark authors' released code or data;
**[paper]** taken from a preprint with its conditions attached. No number appears without one of those tags.

---

## 1. What "SOTA" is allowed to mean

Three gates, all on the same commit.

| gate | statement | instrument |
|---|---|---|
| **G1 — agentic accuracy** | Positive **LAFS gain** against the released LongMemEval-V2 reference frontier, at ≥2 latency operating points, on a leaderboard-valid run | official LME-V2 harness + `leaderboard/compute_lafs.py` |
| **G2 — conversational accuracy at open weights** | LongMemEval_S ≥ **80.80** and LoCoMo(n=1540) ≥ **77.85** — MemPro-15's own Qwen3-30B-A3B numbers, so the answer model class matches ours | `myelin-eval bench` |
| **G3 — robustness** | MINJA-style attack success ≤ 10% with pre-populated memory at k=6, and no cross-tenant leak on any read path | `myelin-eval attack` |

G1 is the primary claim because LME-V2 is the only one of the three with a **public leaderboard, a pinned
protocol, a released reference frontier, and an enforced privacy rule**. G2 is the claim that is comparable
to the existing memory-systems literature. G3 is the claim that the thing is safe to point at real data.

`myelin-eval standing` is the mechanical check of those three gates: it joins
[`docs/sota/registry.json`](sota/registry.json) against the artifacts in `runs/`, prints a comparability
verdict and a gap per published claim, and under `--gate` exits non-zero while any gate row is unsupported,
not comparable, or not beaten.

A fourth, non-negotiable condition: **every reported number carries its protocol**. Dataset + subset + SHA,
answer/reader model, judge model + prompt hash, temperature, seeds, repeat count, token counts, p50/p95
latency, GPU tenancy state, commit SHA. A number without that tuple is not a result.

---

## 2. LongMemEval-V2 — the primary agentic evaluation

### 2.1 Why this one

LME-V2 reframes memory from "answer questions about a chat log" to "become an experienced operator of an
environment" **[paper: arXiv 2605.12493]**. The task is *context gathering*: the memory system consumes a
haystack of agent trajectories and returns **compact evidence**; a fixed reader then answers. That is exactly
the interface `myelin` exposes, which makes the benchmark a direct test of the product rather than a proxy.

Measured dataset facts **[probed, from the HF repo]**:

- `questions.jsonl` — **451 lines**. Domains: `web` 240, `enterprise` 211.
- `question_type` distribution:

  | type | n | ability |
  |---|---|---|
  | `static-environment` | 134 | static state recall |
  | `dynamic-environment` | 86 | dynamic state tracking |
  | `procedure` | 74 | workflow knowledge |
  | `static-environment-abs` | 55 | premise awareness (abstain) |
  | `dynamic-environment-abs` | 41 | premise awareness (abstain) |
  | `procedure-abs` | 32 | premise awareness (abstain) |
  | `errors-gotchas` | 29 | environment gotchas |

  **128 of 451 (28.4%) are abstention questions.** Answering them correctly means *declining* to answer.
- `trajectories.jsonl` — **1,870 trajectories**, each `{id, domain, environment, goal, outcome, start_url,
  states[]}`; each state `{state_index, step, url, action, thought, accessibility_tree, screenshot}`
  **[source: SCHEMA.md]**.
- Haystacks map question id → ordered trajectory ids: `lme_v2_small.json` 100 per question (shared within a
  domain), `lme_v2_medium.json` ~500 per question.
- Download sizes **[probed]**: `trajectories.jsonl` 1,195.6 MB; haystacks 4.1 + 0.8 MB; questions 0.3 MB;
  screenshots 5.9 GB in two tarballs. **Text-only evaluation needs ≈1.5 GB.** `checksums.sha256` ships, so
  integrity is pinned, not trusted. Apache-2.0.

### 2.2 The scorers are the authors', not ours

`eval_function` is a spec string per question, dispatched by `parse_eval_function_spec`
**[source: `evaluation/qa_eval_metrics.py`]**. Measured distribution over all 451 **[probed]**:

| scorer | n | kind |
|---|---|---|
| `norm_phrase_set_match` (separators `,;` / `;`) | 199 + 1 | deterministic |
| `llm_abstention_checker` | 128 | LLM judge |
| `mc_choice_match` | 68 | deterministic |
| `llm_gotchas_checker` | 28 | LLM judge |
| `norm_phrase_set_match_ordered` (`,;` / `;`) | 19 + 7 | deterministic |
| `mc_choice_set_match` | 1 | deterministic |

**295 of 451 (65.4%) are judge-free**; only **156 (34.6%)** need an LLM. That removes judge variance from
two-thirds of the primary benchmark — a much stronger position than LoCoMo/LongMemEval, where the headline
number is judge-scored end to end. Report the judge-free 295 as a separate column; it is the most
reproducible accuracy figure available in this field.

Two mechanics that dictate reader behaviour **[source: `qa_eval_metrics.py`]**:

- Answers are extracted from `\boxed{...}`; `extract_boxed_answer` uses `rfind`, i.e. the **last** boxed span.
- `is_unknown(parsed) == (parsed.strip().lower() == "unknown")`. So a correct abstention is literally
  `\boxed{unknown}`. Our evidence set must make "there is no support for this premise" *decidable*, which is
  why `EvidenceSet` carries explicit emptiness/insufficiency signals rather than silently returning the
  nearest neighbours (`PLAN.md` §7.3).

We implement **zero scorers** for LME-V2. `myelin-eval` shells out to the official harness; reimplementing
the metric is how people accidentally publish incomparable numbers.

### 2.3 The adapter contract

`myelin` plugs in as a `Memory` subclass **[source: `memory_modules/memory.py`]**:

```python
class Memory(ABC):
    memory_type: str
    def __init__(self, memory_params: dict) -> None: ...
    @abstractmethod
    def insert(self, trajectory: dict) -> None: ...                    # one whole trajectory
    @abstractmethod
    def query(self, query: str, query_image: str | None = None
              ) -> list[MemoryContextItem]: ...                        # [{type: text|image, value: str}]
    def configure_runtime(self, **kwargs) -> None: ...                 # non-persisted overrides
    def post_query_hook(self, *, query, query_image, memory_context) -> dict | None: ...
    def save_memory(self, output_dir) -> None: ...                     # + _save_backend / _load_backend
    def set_query_context(self, *, query_invocation_id: str) -> None: ...   # thread-local
```

Five consequences for `PLAN.md`, each a hard requirement:

1. **`query()` returns `list[{type, value}]`.** Our `EvidenceSet` must serialise to exactly that. The
   "compact evidence" contract in `PLAN.md` §7.3 is not a design preference; it is the benchmark's ABI.
2. **`insert()` granularity is a whole trajectory**, not a turn. `myelin-mcp`'s `observe` must accept a
   trajectory object with `states[]` and do its own segmentation (`PLAN.md` §6.1).
3. **`save_memory` / `load_memory` with exact config match.** `reconcile_loaded_memory_config` *requires*
   the requested config to equal the saved config when loading a prebuilt artifact. So a built memory must
   be exportable and re-importable byte-faithfully, and its config must be a pure function of the build.
   This is a new store requirement: `myelin-core` needs `export`/`import` of a built namespace.
4. **Operating points are runtime-selected, not build-selected.** `configure_runtime` exists precisely for
   non-persisted overrides, and leaderboard step 2 validates that all operating points share the same
   haystack and method. Therefore `recall` vs `investigate`, `k`, and step budgets **must be query-time
   parameters over one identical store.** A design where `investigate` needs a different index is
   disqualified from a multi-operating-point submission.
5. **The harness is multi-threaded** (`_query_context_local` is a `threading.local`). The adapter and the
   backend must be safe under concurrent queries against one built memory, and per-query latency is
   attributed via `query_invocation_id`.

### 2.4 The privacy rule — and what it forbids us from doing

The repo ships a test that enforces it **[source: `tests/test_query_privacy.py`]**:

- Every backend receives **only** `{"query_invocation_id": "<opaque>"}` as context; `get_query_context()`
  must be empty outside a query.
- `set_query_context(query_invocation_id=..., question_id=...)` raises `TypeError`.
- The planner prompt may contain `Question text:` and must **not** contain `Question ID:`, `Question type:`,
  `Question image path:`, or the original goal.
- Coding backends' configs must not reference a questions file (`assertNotIn("questions_path", ...)`).

So: **no question-aware indexing, no tuning per `question_type`, no routing off benchmark labels, no
knowledge that a question is an abstention item.** `myelin`'s read policy must infer query class — single-hop
vs multi-hop vs procedural vs unanswerable — from the query *text alone*. `PLAN.md` §7.1's router is
constrained accordingly, and `myelin-eval` runs this test against our adapter in CI so we cannot drift into
cheating by accident.

### 2.5 Pinned models, and what that costs

Leaderboard step 1 validates **[source: `leaderboard/README.md`]**:

- reader model string contains **`qwen3.5-9b`**
- judge model string contains **`gpt-5.2`**
- both domains use the same method and tier; `per_question.jsonl` covers every question; ids unique;
  question-type counts match the runtime inputs

The reference baselines' own configs **[source: `evaluation/memory_configs/*.json`]**:

| param | AgentRunbook-R | RAG slice+notes |
|---|---|---|
| controller | `Qwen/Qwen3.5-9B` @ `:8023/v1`, temp **0.6**, top_p 0.95, top_k 20, max_completion_tokens 8192, thinking on | same |
| embedder | `Qwen/Qwen3-Embedding-8B` @ `:8114/v1`, max_input_tokens 4096, with a query-instruction prefix | same |
| index | `raw_state_slice_radius: 1` | `raw_state_slice_radius: 1` |
| query gen | `max_raw_state_queries: 5` | — |
| retrieval | `raw_state_search_top_k_per_query: 6`, `event_search_top_k: 6`, `note_search_top_k_per_type: 3`, merge budget 6, per-query cap 2, rerank candidate limit 8, **`enable_rerank: false`** | `raw_state_search_top_k: 6`, `note_search_top_k_per_type: 3`, `enable_notes: true` |

Read that table again: **the reference RAG memory is dense-only, has no lexical channel, and does not
rerank.** MemPro's ablation says BM25 is worth +12.68 points on LoCoMo versus +2.36 for the dense embedder
**[paper: arXiv 2606.00619 §A.6]**, and cross-encoder reranking is the largest single quality lever in the
retrieval literature **[paper: MS MARCO dev MRR@10 18.7 → 36.5 → 40.1]**. `myelin`'s `recall` mode is
hybrid BM25+dense with reranking. **That is the specific, identified gap in the reference frontier**, and
§3.4 shows it is also the highest-leverage place to attack LAFS.

Also note **temperature 0.6, not 0**. The reference protocol is stochastic, so single runs are not
comparable; §5 fixes the statistics.

The pinned models are now served on `big` **[probed]** — `Qwen3.5-9B` (the reader, served by
`llama.cpp` at `:5810`) and `bge-reranker-v2-m3` (the cross-encoder, at `:5813`). `Qwen3-Embedding-8B` is
served only when `MYELIN_EMBEDDER=qwen` (the M4 ablation); the default dense embedder is `bge-m3` via ollama
(`:11434`). `llama-swap` also serves `qwen3.8-27b`, `qwen3-vl`, `olmocr`, `glm-ocr`, and ollama has
`qwen3:4b/8b`, `qwen3-vl:8b`, `gpt-oss-20b-heretic`, `qwen2.5:7b`. Adding a model not already listed is a
prerequisite task, and `skill://serve-gguf-on-big` is the procedure.

### 2.6 GPU capacity is the binding constraint — measured

The card is one RTX 3090, 24,576 MiB, shared with the paper pipeline. Measured this session **[probed]**:

| state | VRAM used |
|---|---|
| idle, `hs-serve-distill` resident (bge-m3 on CUDA) | 4,893 MiB |
| after `gpu-tenant claim coding` (distill + trellis2-mcp paused) | 343 MiB |
| `olmocr` loaded and serving | 13,701 MiB |

`qwen3.8-27b` (UD-Q4_K_XL) fails to load while distill holds 4.9 GB:
`ggml_backend_cuda_buffer_type_alloc_buffer: allocating 16053.22 MiB on device 0: cudaMalloc failed: out of
memory` → `unable to allocate CUDA0 buffer` **[probed, `/tmp/qwen3.8-child.log`]**. The model needs
≈22.7 GB of the 24 GB card, so anything else resident starves it — consistent with
`skill://serve-gguf-on-big`, which documents that configuration as requiring the claim with distill stopped.

Operational rules that follow, and they are part of the harness, not the README:

- **Every benchmark run acquires `gpu-tenant claim` and records the tenancy state in its manifest.** Runs
  that share the GPU with distill/scribe produce latency numbers that are not comparable, and
  `memory_query_avg_seconds` is half of LAFS.
- Budget for the LME-V2 stack: `Qwen3.5-9B` at Q5_K_M ≈ 6.5 GB + `Qwen3-Embedding-8B` at Q8 ≈ 8.5 GB
  ≈ 15 GB of weights, leaving ≈9 GB for KV and CUDA graphs — feasible under a claim, not feasible alongside
  distill. If graph capture OOMs, trade `--no-cuda-graph` before shrinking the quant (skill §3).
- The paper pipeline and an eval run **cannot** overlap. `myelin-eval` must fail fast if tenancy is held by
  another tenant rather than silently producing slow numbers.

**Local tool-calling is verified** — the agentic loop can run entirely on `big`. `ollama` `qwen3:8b` at
`localhost:11434/v1`, with `olmocr` concurrently holding 13.7 GB, returned **`finish_reason: "tool_calls"`**
with a well-formed call `recall({"k":1,"query":"cat"})` in 13.7 s **[probed]**. A 9B-class controller
therefore fits and behaves, alongside another resident model.

---

### 2.7 Ports to forward to `big`

The reader, reranker, and Qdrant gRPC are firewalled (§2.6) and reached via SSH tunnel. Ollama connects
directly. A tunnel for a full eval run forwards:

| port | service | engine |
|---|---|---|
| 5810 | reader (OpenAI-compatible `/v1`) | llama.cpp `llama-server` |
| 5813 | reranker / cross-encoder | llama.cpp `llama-server` `--reranking` |
| 6334 | Qdrant gRPC | qdrant server |
| 11434 | ollama (default embedder `bge-m3`) | ollama |

`5811` (the Qwen3-Embedding-8B embedder) starts only when `MYELIN_EMBEDDER=qwen`; forward it too for that
arm.

---

## 3. LAFS — the metric that actually decides G1

### 3.1 Definition

**[source: `leaderboard/compute_lafs.py`]** — accuracy is `overall_full_set * 100`, latency is
`memory_query_avg_seconds`, and the score is the mean best-accuracy-under-budget over a **log-uniform**
latency budget:

$$\mathrm{LAFS} \;=\; \frac{1}{\ln\!\big(T_{\max}/T_{\min}\big)}\int_{T_{\min}}^{T_{\max}} \mathrm{bestAcc}\big(\text{latency} \le T\big)\; d\ln T$$

with $T_{\min}=1.0\,$s, $T_{\max}=200.0\,$s, floor accuracy $0$, evaluated over the Pareto frontier of
operating points. The submission score is **LAFS gain** = $\mathrm{LAFS}(\text{reference} \cup \text{ours})
- \mathrm{LAFS}(\text{reference})$. Domains are combined by example-count-weighted average (web 240,
enterprise 211), and the extracted fields are `overall_full_set`, `gotchas_accuracy`, `static_accuracy`,
`dynamic_accuracy`, `procedure_accuracy`, `memory_query_avg_seconds`.

### 3.2 The released reference frontier

Hard-coded in the leaderboard tool **[source]**:

| tier | points (accuracy @ latency) |
|---|---|
| small | RAG slice+notes 51.0 @ 0.2 s · AgentRunbook-R 58.6 @ 26.9 s · Codex 69.9 @ 177.2 s · AgentRunbook-C 74.9 @ 108.3 s |
| medium | RAG slice+notes 45.9 @ 0.3 s · AgentRunbook-R 57.0 @ 25.8 s · Codex 68.7 @ 185.8 s · AgentRunbook-C 70.1 @ 139.9 s |

Computed by reimplementing the released formula exactly **[probed]**:

| tier | reference LAFS | Pareto frontier | dominated |
|---|---|---|---|
| small | **55.765** | 51.0@0.2s, 58.6@26.9s, 74.9@108.3s | Codex (69.9@177.2s) |
| medium | **51.074** | 45.9@0.3s, 57.0@25.8s, 70.1@139.9s | Codex (68.7@185.8s) |

### 3.3 Sensitivity — where the points actually come from

LAFS gain for a single added operating point, tier `small` **[probed]**:

*fast point (our `recall` mode)*

| latency | acc 50 | 55 | 60 | 65 | 70 |
|---|---|---|---|---|---|
| 0.3 s | +0.00 | +2.49 | +5.96 | +10.38 | +14.80 |
| 0.5 s | +0.00 | +2.49 | +5.96 | +10.38 | +14.80 |
| 1.0 s | +0.00 | +2.49 | +5.96 | +10.38 | +14.80 |
| 2.0 s | +0.00 | +1.96 | +4.78 | +8.55 | +12.32 |
| 5.0 s | +0.00 | +1.27 | +3.23 | +6.13 | +9.03 |

*slow point (our `investigate` mode)*

| latency | acc 60 | 70 | 76 | 80 | 85 |
|---|---|---|---|---|---|
| 10 s | +2.05 | +6.55 | +9.37 | +11.63 | +14.46 |
| 20 s | +0.87 | +4.06 | +6.10 | +7.84 | +10.01 |
| 30 s | +0.34 | +2.76 | +4.34 | +5.78 | +7.57 |
| 60 s | +0.16 | +1.27 | +2.07 | +2.98 | +4.11 |
| 110 s | +0.00 | +0.00 | +0.12 | +0.58 | +1.14 |

Combined submissions, tier `small` **[probed]**:

| submission | LAFS gain | absolute LAFS |
|---|---|---|
| `recall` 60 @ 0.5 s only | +5.96 | 61.73 |
| `investigate` 76 @ 60 s only | +2.07 | 57.83 |
| `recall` 60 @ 0.5 s + `investigate` 76 @ 60 s | **+7.87** | 63.64 |
| `recall` 65 @ 0.5 s + `investigate` 80 @ 40 s | **+13.79** | 69.56 |
| `recall` 55 @ 0.5 s + mid 70 @ 20 s + `investigate` 78 @ 90 s | +6.96 | 62.72 |

### 3.4 Four strategic conclusions, all forced by the arithmetic

1. **Sub-second latency is worth nothing extra.** $T_{\min}=1.0\,$s clamps the integral, so 0.3 s, 0.5 s and
   1.0 s score identically. `PLAN.md`'s p95 < 100 ms target is a **product** SLO, not a leaderboard
   objective — it must not be traded against accuracy in pursuit of G1. Correcting this is the single most
   important thing this analysis changes about the plan.
2. **A fast mode is worth ~3× a slow mode.** `recall` at 60% and ≤1 s gains +5.96; `investigate` at 76% and
   60 s gains +2.07. Log-uniform integration means a fast point raises the floor across the *entire*
   [1 s, 200 s] window, while a slow point only improves the tail.
3. **Break-even is astonishingly low: accuracy > 51.1 (small) / 46.0 (medium) at ≤1 s** **[probed]** — i.e.
   beat the reference's dense-only, no-rerank, no-BM25 RAG baseline in under a second and the submission
   scores. Given §2.5, that is exactly what a hybrid BM25+dense+rerank memory should do. This is M5's first
   target and it is deliberately modest, because a positive LAFS gain is a real, checkable SOTA claim.
4. **Ship at least two operating points, and make the fast one good.** The benchmark is built for it,
   `configure_runtime` exists for it, and the arithmetic rewards it. Target for the headline claim:
   **`recall` ≥ 65 @ ≤1 s and `investigate` ≥ 80 @ ≤40 s ⇒ +13.79 LAFS gain on small.**

---

## 4. The agentic loop, specified

`investigate` is where `myelin` is an agent rather than an index. It must be pinned down, because "an agent
searches until it's satisfied" is not a specification.

```
investigate(question, scope, max_steps, budget_tokens) -> EvidenceSet + Trace
  s := 0 ; evidence := ∅ ; budget := budget_tokens
  loop
    plan   := controller(question, digest(evidence))        # 1 LLM call, tools: search|read|neighbors|stop
    if plan.stop or s ≥ max_steps or budget exhausted: break
    obs    := execute(plan.tool_call)                        # deterministic, no LLM
    evidence := merge(evidence, obs)                         # dedup by near-duplicate cosine
    gate   := sufficiency_and_conflict(question, evidence)   # cheap: coverage + contradiction check
    if gate.sufficient and not gate.conflicted: break
    if gate.conflicted: plan_next := refresh(conflict)       # AMA's refresh-on-conflict
    s := s + 1
  return compose(evidence, budget_tokens)
```

Parameters and their justification:

| parameter | default | why |
|---|---|---|
| `max_steps` | 6 | Codex-class agents spend ≈182 s/query at LME-V2 **[paper]**; §3.3 shows gain collapses past ~60 s. Six controller calls on a 9B model at ~14 s/call sits near 40 s, the +13.79 operating point. |
| tool set | `search`, `read`, `neighbors`, `stop` | Progressive disclosure: identifiers first, bodies on demand. Matches the just-in-time pattern that `PLAN.md` §2.1 adopts. |
| conflict gate | on | AMA reports knowledge-update 0.897 with a refresh-on-conflict gate vs 0.568 without **[paper: arXiv 2601.20352]** |
| `budget_tokens` | 8,192 composed | MEMAGENT's split: memory 1,024 / chunk 5,000 / total ≤ 8,192 **[paper: arXiv 2507.02259]** |
| controller temp | 0.6 / top_p 0.95 / top_k 20 | match the reference protocol exactly **[source]** so the comparison is like-for-like |

**Trace capture is mandatory and is a first-class artifact.** Per query the harness records: every tool call
with arguments and latency, evidence set after each step, gate decisions with reasons, controller token
counts, and the final `EvidenceSet` with provenance. Without traces, a regression in agentic accuracy is
undebuggable, and `post_query_hook` is the official place to emit them.

Agentic metrics that only a trace can produce, reported alongside accuracy:

| metric | definition |
|---|---|
| steps-to-answer | distribution of `s` at break, per question type |
| tool-selection error rate | fraction of controller turns emitting an invalid tool name or schema-invalid arguments |
| marginal step value | accuracy as a function of `max_steps` ∈ {1, 2, 4, 6, 8} — the curve, not a point |
| wasted retrieval | fraction of returned items not cited by the reader's answer |
| abstention precision/recall | on the 128 `-abs` questions, measured as `\boxed{unknown}` correctness |
| conflict-gate firing rate | and its precision against gold contradictions |

The marginal-step-value curve is the honest test of whether the agentic loop earns its latency. If accuracy
at `max_steps=2` equals `max_steps=6`, we ship the cheaper operating point and say so.

---

## 5. Statistics — the part that separates a claim from a coincidence

The reference protocol runs the controller at temperature 0.6 **[source]**, so results are random variables.
With n=451 (LME-V2), n=500 (LongMemEval_S) and n=1540 (LoCoMo), the naïve standard error on a proportion near
0.7 is $\sqrt{0.7 \cdot 0.3 / n}$ ≈ **2.2 pp** (451), **2.0 pp** (500), **1.2 pp** (1540). A 1-point win is
noise. Therefore:

1. **Three seeds minimum** per operating point; report mean and the spread. Seeds recorded in the manifest.
2. **Bootstrap 95% CIs** over questions (10,000 resamples, stratified by `question_type` and `domain`) for
   every headline accuracy.
3. **Paired comparison against the baseline on identical items** — McNemar's test for the
   deterministic-scored subset, paired bootstrap for the judge-scored subset. Unpaired comparisons across
   different question subsets are not reported.
4. **A win is claimed only when the paired CI excludes zero.** If we beat MemPro's 80.80 by 0.8 pp with a
   CI of ±2.0, the honest statement is "matches SOTA within noise", and that is what the report prints.
5. **Judge-free column always present.** For LME-V2 that is the 295 deterministic items; for LoCoMo it is
   token-F1 and exact match. Judge-scored and judge-free numbers are never averaged together silently.
6. **Contamination probe before any model is used as reader or judge**: sample 50 benchmark questions, ask
   the model for the answer with no memory context at all, and report the closed-book score. A high
   closed-book score on a memory benchmark means the model already knows the answers and the retrieval
   result is uninterpretable. Precedent for why this matters: SWE-Bench+ found 32.67% solution leakage
   **[paper: 10.48550/arXiv.2410.06992]**.

---

## 6. Conversational-memory benchmarks (G2)

### 6.1 LoCoMo

**[probed]** `locomo10.json`, 2,805,274 bytes, 10 conversations, 19–32 sessions and 369–689 turns each.
QA by category: `1:282, 2:321, 3:96, 4:841, 5:446` — **1,986 total**.

$1986 - 446 = 1540$ exactly. The "1,540 questions" every paper and vendor blog reports is **this file with
the adversarial category deleted**. The harness therefore always emits three columns:

- `n=1540` — comparable to the literature
- `n=1986` — the honest whole-file number
- `abstention-F1` — category 5 alone, where the reference is HiGMem's 0.78 **[paper: arXiv 2604.18349]**

### 6.2 LongMemEval_S

**[probed]** HF `xiaowu0162/longmemeval`, ungated, config `longmemeval_s` = 278,025,796 bytes, **500 items**,
mean **50** haystack sessions. Keys: `question_id, question, question_type, question_date, answer,
answer_session_ids, haystack_dates, haystack_session_ids, haystack_sessions`. Types: `temporal-reasoning`
133, `multi-session` 133, `knowledge-update` 78, `single-session-user` 70, `single-session-assistant` 56,
`single-session-preference` 30.

### 6.3 Judging these two

LME-V2's leaderboard pins the judge to `gpt-5.2`; LoCoMo and LongMemEval have no such authority, so here we
use a local open-weights panel and prove it is trustworthy rather than asserting it:

1. Fixed models + quant + single engine; prompt hash recorded. Judge runs at temperature 0 even though the
   *answer* model follows the reference protocol's 0.6.
2. **Reuse the benchmark authors' published judge prompts verbatim.** Zep's high human correlation came
   from using LongMemEval's own prompts **[paper: arXiv 2501.13956]**.
3. Position-shuffle each candidate under two orderings and average **[paper: MT-Bench, arXiv 2306.05685]**.
4. Two judges — the local Qwen3.5-9B reader plus `gemini-3.1-flash-lite` — over 86 questions, **with
   inter-judge Cohen's κ printed next to every number**. The measured agreement is raw 95.3%,
   κ = **0.8813** (`docs/measurements/m9-judge-panel.md`); a quota-limited 12-question third-judge subset
   adding `gemini-3-flash-preview` is not reportable (κ = 0.7037 at n = 12). EverMemOS's three-blind-judge
   protocol reaches κ = 0.891 (LoCoMo) and 0.979 (LongMemEval) against five human annotators over 25 Q&A
   pairs each **[paper: arXiv 2601.02163]** — that is the bar.
5. Calibrate on 50 gold answers; require ρ ≥ 0.9 against the reference ranking before a full run. **If the
   panel fails calibration, the judge is the finding and the run is void.**
6. Never mix judge families inside one comparison.

For the 156 LLM-scored LME-V2 items there are two modes, and the harness labels which was used:
`leaderboard` (judge = `gpt-5.2`, required for a valid submission, ~156 × 2 domains × operating points
calls — a bounded, affordable external spend) and `local` (judge = the panel above, for the development
loop). The local-vs-`gpt-5.2` delta on those 156 items is tracked as a calibration statistic in its own
right.

### 6.4 Where we stand on these two, and how a number gets made

The shipped `investigate` operating point on LongMemEval_S, as of M44 R2
(2026-09-22): `--k 6 --budget-tokens 4096 --max-steps 2 --select-sufficient
--item-digest --digest-dates --reader-thinking --reader-seed <n>`, the
reader served with `--reasoning-budget 1024`, judged by the local
Qwen3.5-9B panel (thinking off for the judge).

| milestone | change | LongMemEval_S judged (n = 500) |
|---|---|---|
| M19 | dates resolved *for* the reader; `[timeline]` view | 56.40 |
| M32 | pool-level sufficiency selection ships on | **62.00** (+5.8, 95% CI [+2.8, +8.8]) |
| M43 | one dated digest note per memory ships on | **67.80** (+5.8, 95% CI [+2.8, +8.8]) |
| M44 R2 | the reader thinks — 1,024-token budget, sampled, two seeds | **78.40** (+10.6, 95% CI [+7.0, +14.2]; seed 2 identical) |

Everything between and after those rows — M33–M42, M44 R1 — is a measured
null, a significant negative, or a win vetoed on the abstention rows, and
each is recorded in `docs/measurements/` with the number that stopped it.
`BACKLOG_DONE.md` keeps the running standing table and the ratchet history;
`myelin-eval ratchet --strict` fails any commit whose artifacts fall below
these floors.

**How an arm becomes a number.** Every switch is pre-registered in its
measurement doc *before* the arm runs — bar, strata, prediction, falsifier.
The arm is one `bench` run differing from the base by exactly one switch;
it is judged with `judge --seed <base>`, which reuses the base's verdict on
every byte-identical answer so the control on untouched rows is exactly
zero rather than judge noise (M42 measured 2 of 469 unchanged answers
flipping under a fresh judge). The paired bootstrap over per-question
differences (`adapters/paired_ci.py`) gives the CI; the bar is **+3.0 with
the CI excluding zero**, and any drop on the 30 abstention rows vetoes the
switch whatever the headline says. A switch that clears both flips its
default in the same PR as its doc, its artifacts and the raised ratchet
floor.

**What the comparison to the literature is worth.** Every published
LongMemEval_S and LoCoMo row is judged by a frontier API; ours by a local
9B. `standing` marks every such row `caveat-judge`, marks a row backed by an
arm rather than the shipped defaults `(arm)`, and licenses a claim only on a
`comparable` row we lead. One such row exists: LoCoMo 69.87 against Mem0's
published 66.88.

---

## 7. Robustness (G3)

Attack suite derived from MINJA's method **[paper: arXiv 2503.03704]**, which achieves **98.2% injection
success / 76.8% attack success** averaged over three agents and four datasets, from a query-only attacker
against a **shared** memory bank starting **empty**, retrieval k = 3–5.

The follow-up EHR study gives the two conditions that matter **[paper: arXiv 2601.05504]**:

- pre-populating legitimate memory drops ASR **62% → 6.67%** (GPT-4o-mini) and **52.94% → 0%** (Llama-3.1-8B)
  at k = 3 (Table 1);
- in a *different* configuration (6 pre-existing memories + 4 indication prompts), ASR rises with retrieval
  breadth: **6% → 20% → 38%** at k = 3/5/10 (Table 2).

These are two setups, not one curve, and the harness runs them as two separate experiments.

| experiment | axes | gate |
|---|---|---|
| E1 injection | empty vs pre-populated × tier ∈ {untrusted, asserted} × adjudicator on/off, k ∈ {3,6,10} | ASR ≤ 10% at k=6 pre-populated — **measured, and still failing**: M11 reported 80% undefended at n=5; the 40-attack set reproduces it at **77.5% [62.5, 87.7]**, and the M15 write-time adjudicator takes it to **15.0% [7.1, 29.1]** — 62.5 points, identical at the `asserted` tier — with 0/550 LoCoMo false positives. ⇒ **`WritePath::adjudicate` ships off, G3 open with a measured bound.** `docs/measurements/m15-injection-adjudication.md` |
| E2 retrieval breadth | k ∈ {3,6,10} at fixed memory state | report the curve; flag if monotone increasing — **measured**: monotone in 4 of 6 M15 conditions (70.0 → 77.5 → 85.0% undefended, pre-populated), so the EHR Table 2 effect does reproduce once ASR is not saturated. M11 saw "not monotone" only because its ASR was pinned at ~100% |
| E3 quarantine efficacy | fraction of templated poison caught by the trust gate before commit | ≥ 90% catch on templated attacks |
| E4 tenant isolation | cross-tenant read attempts on every read path | **zero** leaks; a single leak fails G3 outright |
| E5 confidence≠safety | replay the Gemini-2.0-Flash failure mode: 54 malicious entries accepted at trust = 1.0 (82 of 151 accepted; the GPT-4o-mini run rejected all 23) | our gate must reject them, proving trust alone does not admit |
| E6 unlearning | delete a source record, assert descendants re-derived or removed, and that it is unreachable from every read path | invariant I5 holds |

E1's poison is the **paraphrased** set — 40 attacks, 8 surface forms × 5 domains, each raising zero
flags from the pattern gate, so E1 can never decay into a second measurement of E3 — plus a 10-item
**ungated** adaptive probe of bare false assertions carrying no mechanic at all. The probe is
reported and never averaged into the gate: a false statement with no override clause, no forged
provenance and no redirect framing is indistinguishable from a true fact by any content classifier,
and the adjudicator admits 10/10 of them by design.

E4 and E6 are invariant tests, not benchmarks: they run on every commit.

---

## 8. Ablations — one per load-bearing claim

Every architectural claim in `PLAN.md` §2 becomes a switch, measured on our own code. A claim we cannot
reproduce as an ablation is demoted to a citation and removed from the design rationale.

| # | claim | ablation | expected direction |
|---|---|---|---|
| 1 | BM25 > dense | `Bm25Only`, `DenseOnly`, `Hybrid` | removing BM25 hurts more than removing dense (MemPro: −12.68 vs −2.36) |
| 2 | rerank > fusion | `HybridNoRerank` vs `Hybrid` | rerank is the bigger delta |
| 3 | scope-before-routing | filter pre- vs post-ranking at fixed probe budget | pre-filter wins (ShardMemo +2.9/+3.1 F1) |
| 4 | small k | k ∈ {3, 6, 10, 20, 50} | accuracy peaks then declines; cost rises monotonically |
| 5 | 4-op delta | disable `DELETE` | knowledge-update accuracy collapses |
| 6 | bi-temporal validity | ignore `t_invalid` at read | contradiction rate rises |
| 7 | graph expansion | PPR route on/off | gain confined to multi-hop questions — **measured, and wrong**: no gain in any category, −2.7 pts recall@6 against `hybrid_k1`, and −0.7 pts end-to-end on LongMemEval_S (CI [−1.5, −0.1]). `docs/measurements/m12-graph-route.md` |
| 8 | evidence order | `bookend` relevance interleave vs ascending `t_valid` (`ComposeConfig::chronological`) | temporal and duration questions improve — **measured, and wrong**: 540/1,986 LoCoMo answers change and split 173 better / 182 worse, LoCoMo cat 2 −0.3 (CI [−1.9, +1.3]), LME_S temporal-reasoning −1.1 (CI [−5.2, +3.1]); the one real gain is LME_S knowledge-update +8.0. `docs/measurements/m13-temporal-axis.md` |
| 9 | answer scoring | token F1 vs date-aware interval scoring (`myelin_eval::temporal`, `bench --scorer`) | a date-aware scorer credits correct dates token overlap cannot see — **measured, and the opposite was the bigger effect**: token F1 was awarding 0.50–0.75 to answers naming the *anchor* instead of the offset, inflating LoCoMo cat 2 by 8.03 pts (0.2825 → 0.2022) and answerable F1 by 1.69. Against a reader-only judge on 272 cat-2 items the interval scorer agrees 96.7% vs token F1's 84.9% (+11.8 pts, CI [+7.7, +15.8], κ 0.60 → 0.91), and is a −0.02 pt no-op off-stratum ⇒ **`temporal` is the LoCoMo default; both columns ship on every row**. `docs/measurements/m14-temporal-scorer.md` |
| 10 | date resolution | relative expressions in the emitted text left as written vs resolved against the record's own `t_valid` (`ComposeConfig::resolve_relative`), and the reader-side date clause in `READER_SYSTEM` | resolved dates help the temporal stratum — **measured, and it is the largest win in the project**: on LoCoMo cat 2 (n=321, date-aware scorer) the annotation is +37.6 pts (CI [+32.2, +43.2]), the prompt clause +14.3 ([+10.2, +18.7]), both together +42.8 ([+37.3, +48.5]), with both marginals significant (+5.2 and +28.4) ⇒ **both default on**. Full-set judged LoCoMo 62.66 → 69.87 (+7.2, CI [+5.5, +9.0]), and no off-target category moves on an interval excluding zero. `docs/measurements/m19-temporal-resolution.md` |
| 11 | dated index | no index vs one synthetic `[timeline]` item over the selected records, appended only for interval questions (`ComposeConfig::timeline`, gated by `time::is_interval_question`) | duration and ordering questions need two dated endpoints side by side — **measured and confirmed, and it is the index rather than breadth**: +6.8 pts on LongMemEval temporal-reasoning (n=133, judged, CI [+3.0, +11.3]) and +6.8 again at `k=25` ([+1.5, +12.8]), while widening `k` alone is +3.8 with a CI spanning zero and `investigate` is exactly 0.0 at 6.5× the latency. The whole effect lands on the 61 duration questions (13.11 → 27.87) and the other 66 are unchanged to the digit ⇒ **default on; `k` and `mode` stay per-call (R4)**. `docs/measurements/m19-temporal-resolution.md` |
| RRF | fusion constant | client-side k=60 vs Qdrant's server-side k=1 | measurable; Qdrant's k=1 verified **[probed]** |
| late | reranker choice | `late` multivector vs `bge-reranker` cross-encoder at equal latency | decides whether the `late` channel is populated at all |

---

## 9. Run manifest

Every run writes one JSON manifest. This is the reproducibility contract; the report renderer refuses rows
without one.

```jsonc
{
  "commit": "<sha>", "started_at": "<rfc3339>", "duration_s": 0,
  "benchmark": { "name": "lme_v2", "tier": "small", "domain": "web",
                 "data_sha256": "<from checksums.sha256>", "n_questions": 240,
                 "subset": "full", "scorer_mix": { "deterministic": 295, "llm": 156 } },
  "backend":   { "name": "myelin", "version": "<semver>", "operating_point": "fast",
                 "memory_config": { "memory_type": "myelin", "memory_params": { } },
                 "runtime_overrides": { "mode": "recall", "k": 6, "rrf_k": 60 },
                 "store": { "qdrant_version": "1.19.1", "collection": "myelin_memory",
                            "points": 0, "graph_edges": 0 } },
  "models":    { "reader": "qwen3.5-9b", "controller": "qwen3.5-9b", "temperature": 0.6,
                 "embedder": "qwen3-embedding-8b", "judge": "gpt-5.2",
                 "judge_mode": "leaderboard", "judge_prompt_sha256": "<hash>" },
  "seeds":     [1, 2, 3], "repeats": 3,
  "hardware":  { "host": "big", "gpu": "RTX 3090 24576MiB",
                 "gpu_tenant": "coding", "vram_peak_mib": 0 },
  "results":   { "overall_full_set": 0.0, "static_accuracy": 0.0, "dynamic_accuracy": 0.0,
                 "procedure_accuracy": 0.0, "gotchas_accuracy": 0.0,
                 "abstention_precision": 0.0, "abstention_recall": 0.0,
                 "judge_free_subset_accuracy": 0.0,
                 "memory_query_avg_seconds": 0.0,
                 "latency": { "p50": 0.0, "p95": 0.0, "p99": 0.0,
                              "split": { "embed": 0.0, "search": 0.0, "rerank": 0.0, "llm": 0.0 } },
                 "tokens": { "ingest": 0, "per_query_prompt": 0, "per_query_completion": 0 },
                 "ci95": { "overall_full_set": [0.0, 0.0] },
                 "lafs": { "tier": "small", "gain": 0.0, "absolute": 0.0,
                           "operating_points": [ { "name": "fast", "acc": 0.0, "latency": 0.0 } ] } },
  "agentic":   { "steps_to_answer": { "p50": 0, "p95": 0 },
                 "tool_selection_error_rate": 0.0, "wasted_retrieval_fraction": 0.0,
                 "conflict_gate_fire_rate": 0.0, "trace_path": "runs/<id>/traces.jsonl" }
}
```

---

## 10. What runs when

| tier | trigger | contents | budget |
|---|---|---|---|
| **unit** | every commit | invariants I1–I5, query-privacy test against our adapter, E4 tenant isolation, E6 unlearning, RRF/BM25 capability probes against live Qdrant | < 2 min, no GPU claim |
| **smoke** | every commit | 1 LoCoMo conversation, 20 LongMemEval items, 10 LME-V2 questions; deterministic scorers only; assert no regression > 3 pp vs the stored baseline | < 10 min, no GPU claim |
| **nightly** | schedule | full LoCoMo(1540/1986), full LongMemEval_S, LME-V2-Small both domains, local judge panel, 1 seed | GPU claim required |
| **release** | manual | 3 seeds × all operating points × both tiers, `gpt-5.2` judge on the 156 items, bootstrap CIs, full ablation matrix, E1–E3 attack suite, leaderboard package build | GPU claim, hours |

`myelin-eval` subcommands: `fetch` (datasets + checksum verify), `build` (construct memory, export artifact),
`bench`, `attack`, `ablate`, `report`, `package` (invoke the official two-step leaderboard builder).

---

## 11. Threats to validity we are choosing to accept, and how we disclose them

| threat | disclosure |
|---|---|
| `gpt-5.2` judge is external and non-local | `judge_mode` is printed on every row; the judge-free 295-item column is always shown; the local-vs-`gpt-5.2` delta on the 156 judged items is reported |
| Reference frontier is hard-coded from the paper, not re-measured by us | stated; we do not re-derive the baselines, we add operating points to the released frontier as the tool intends |
| Reader is pinned to `qwen3.5-9b`, so results do not transfer to other readers | stated; G2 exists precisely to give a second, differently-pinned comparison |
| LoCoMo is LLM-generated and CC BY-NC 4.0; LME-V2 environments are synthetic | stated in the report header; neither is a claim about real users |
| Our own ablations are self-graded | mitigated by running all baselines through the identical `MemoryBackend` trait and the official scorers; never by a bespoke metric |
| Latency measured on one contended GPU | tenancy state and VRAM peak in every manifest; runs sharing the GPU are marked non-comparable and excluded from LAFS |
