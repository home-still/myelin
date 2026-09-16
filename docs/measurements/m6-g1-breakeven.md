# M6 — G1 break-even on LongMemEval-V2 Small: **NOT MET**

`PLAN.md` M6 exit criterion: *a single `recall` operating point clears accuracy > 51.1 at ≤ 1 s.*

Measured 2026-09-14, commit `cfffebd`, against the full 85,589-episode LME-V2-Small memory, run
through the **authors' own harness** (`evaluation/harness.py`, unmodified) with our
`memory_modules` adapter.

## Result

| domain | n | overall | non-abstention | abstention | `memory_query` p50 |
|---|---|---|---|---|---|
| web | 240 | 36.7% | 42.9% | 22.2% | 1.82 s |
| enterprise | 211 | 34.6% | 39.4% | 21.4% | 2.07 s |
| **combined** | **451** | **35.7%** | — | — | — |

**Both halves of the gate fail**: 35.7% against >51.1, and 1.8–2.1 s against ≤1 s.

## Caveat that must travel with these numbers

The judge is **local Qwen3.5-9B, not the pinned `gpt-5.2`** — we have no OpenAI key. 156 of the 451
questions are LLM-judged, so those are **not leaderboard-comparable**. The 295 deterministically
scored questions are, and reporting them separately is exactly what `EVALUATION.md` §9's
`judge_free_subset_accuracy` field exists for. Any LAFS number computed from the table above would
be invalid, and none is claimed here.

## Where the accuracy goes: abstention

| category | web | enterprise |
|---|---|---|
| static | 46.7% | 39.2% |
| dynamic | 31.4% | 34.3% |
| procedure | 52.4% | 50.0% |
| gotchas | 40.0% | 28.6% |
| **static-abs** | 38.7% | 12.5% |
| **dynamic-abs** | 14.3% | 25.0% |
| **procedure-abs** | **5.0%** | 33.3% |

`procedure-abs` on `web` is the floor: 5.0% correct, **70% answered wrong**. The reader is handed
25 plausible page fragments and produces a confident procedure that does not exist.

Abstention is 30% of the question set and the arithmetic says it is the whole gap:

$$168(0.429) + 72(0.222) = 88/240 = 36.7\%,\qquad 168(0.429) + 72(0.72) = 124/240 = 51.7\%$$

Fix abstention alone, change nothing else, and `web` clears the gate.

## Four attempts, and what each measured

| # | change | result |
|---|---|---|
| 1 | k = 6, budget 2,048 | 30.8% (web) |
| 2 | k = 25, budget 10,000 | **36.7%** (web) |
| 3 | k = 60, budget 24,000 | invalid — see below |
| 4 | abstention gate on rerank score | +0.4 points, inside noise |

Attempt 3 was **not a measurement**: `rerank_depth = 25` silently capped `k`, so it re-ran k=25
with a different tie-break. Both runs returned ~9,600-token evidence sets, which is what exposed
it. Fixed in `507835c` — a config constant must not overrule a query parameter (R4).

Attempt 4 is documented in full on `RetrieveConfig::tau_abstain`. Calibrated over 120 questions,
answerable and abstention top-scores overlap almost entirely (medians **0.76** vs **0.47**), and
every threshold trades one error for the other at a loss. **A cross-encoder scores relevance, not
answer-containment** — an LME-V2 abstention question asks something plausible about an environment
the haystack really does describe, so topically relevant pages score high and the question is
still unanswerable.

Per the standing three-attempt cap, tuning stopped here.

## Two structural findings

**1. Sufficiency needs a model, and that breaks the ≤1 s point.** Relevance scoring cannot detect
"the evidence does not contain the answer". The mechanism that can is `investigate`'s reflect
gate, which costs a model call (~2 s measured). So the ≤1 s `recall` point and the abstention fix
are, on current evidence, mutually exclusive — which is consistent with `PLAN.md` targeting
`recall` ≥65 **and** `investigate` ≥80 as two separate points rather than one.

**2. The reranker does not fit in the 1 s budget on this corpus, and LoCoMo did not predict that.**
M4 measured `hybrid_rerank` at p50 **184 ms** on LoCoMo. Here the same stage is most of 1.8 s. The
difference is document length: a LoCoMo fact is ~15 tokens, an LME-V2 accessibility-tree chunk is
~450. At k=25 that is ~11,000 tokens through the cross-encoder instead of ~375. A latency budget
transferred between corpora without re-measuring is not a budget.

**Follow-up, M16 — the answerable half is retrieval's, not the reader's.**
`docs/measurements/m16-evidence-sufficiency.md` audited the evidence behind every answerable
question these two runs scored wrong: only **7.4%** [4.4, 12.0] of them had the answer in the
evidence set. A perfect reader over the evidence these runs retrieved reaches **38.8%** against the
51.0 bar; repairing retrieval reaches **67.6%** at the measured P(correct | sufficient) = 81.8%. So
finding 1 above generalises past abstention: the binding constraint on G1 is what the evidence
contains, not what the reader does with it.

## Reproduce

```bash
myelin-mcp --serve 127.0.0.1:7447 --collection myelin_lme_v2_small --ledger data/lme_v2_small.ledger
PYTHONPATH=vendor/longmemeval-v2:adapters .venv/bin/python adapters/run_myelin.py \
  --data-root /tmp/lmev2 --domain web --tier small --k 25 --budget-tokens 10000 \
  --output-dir runs/myelin_k25_web_small \
  --evaluator-base-url http://127.0.0.1:5810/v1 --evaluator-model Qwen/Qwen3.5-9B
```

Reader must be served with `--mmproj` (29 questions carry a screenshot) and a context that gives
each slot ≥ 16k tokens — `-c 16384 -np 4` yields 4,096 per slot and rejects every k=25 prompt.

---

## Follow-up: does `investigate` fix abstention?

Asked because the answer decides whether G1 is reachable at all. Same 60-question `web` subset,
same `k = 25`, same evidence budget — only the mode changes.

| mode | overall | non-abstention | abstention | `memory_query` p50 |
|---|---|---|---|---|
| `recall` | 36.7% | 46.5% | 11.8% | 1.74 s |
| `investigate`, max_steps 3 | **41.7%** | 48.8% | **23.5%** | 26.71 s |

**Directionally right, quantitatively short.** The reflect gate roughly *doubles* abstention
accuracy — which confirms the diagnosis that sufficiency, not relevance, is the missing predicate
— and it buys +5.0 points overall. Latency rises 15× to 26.7 s, comfortably inside the 40 s
`investigate` budget.

But 23.5% is not 72%. Projecting to the full `web` set:

$$168(0.488) + 72(0.235) = 82 + 17 = 99/240 = 41.2\%$$

Still short of 51.1. Abstention-by-category shows where: `procedure-abs` is **0% correct, 100%
answered wrong** even with the loop.

### The mechanism the loop already has and does not use

`Investigator` computes a model judgement of sufficiency on every step and records why it stopped
(`sufficient`, `no new evidence`, `step budget`, `no new query`). When it stops *unsatisfied* it
nevertheless returns the pool it accumulated, and the reader answers from it.

Emitting an explicit insufficiency signal instead — an empty evidence set when
`stopped_because != "sufficient"` — would apply the gate at the level where a model actually
judged sufficiency, rather than at the reranker where the signal is only relevance. That is a
materially different mechanism from the two already measured and rejected, and the data above is
what motivates it.

**Not implemented.** M6 is at its three-attempt cap, and this is a design change rather than a
parameter, so it is escalated rather than taken unilaterally.

---

## Correction: the ≤1 s latency target was my error

Exercising the vendored packager produced the authoritative frontier arithmetic and it contradicts
the target I published above. `leaderboard/compute_lafs.py` defines

$$\text{LAFS} = \frac{1}{\ln(t_{max}/t_{min})}\int_{t_{min}}^{t_{max}} \text{best\_acc\_under\_budget}(T)\ d\ln T$$

with `T_MIN = 1.0`, `T_MAX = 200.0`. `best_acc_under_budget(T)` is the best accuracy among frontier
points with latency ≤ T, so it is a **step function**. The released `small` frontier is

| point | acc | latency |
|---|---|---|
| RAG: query → slice + notes | 51.0 | 0.2 s |
| AgentRunbook-R | 58.6 | 26.9 s |
| AgentRunbook-C | 74.9 | 108.3 s |
| Codex | 69.9 | 177.2 s — *dominated, not on the frontier* |

A new point adds area only where it lifts that envelope. Solving numerically for the accuracy that
yields LAFS gain > 0:

| latency | break-even accuracy |
|---|---|
| 0.50 s | 51.00% |
| 1.00 s | 51.00% |
| 1.97 s | 51.00% |
| 26.71 s | 51.00% |
| 108.30 s | 74.90% |

**Being faster than ~26.9 s buys nothing.** RAG sits at 0.2 s, below `t_min`, so its 51.0 is the
envelope across the whole band $[1.0, 26.9)$. The `p50 ≤ 1 s` constraint I asserted was read off the
"fast operating point" framing, not computed — it does not exist. The accuracy bar of ~51 is real.

### What this invalidates and what it opens

- The `fast` operating point (35.7% @ 1.83 s) **can never score**: it is strictly under 51.0 and
  no amount of latency reduction changes that. Effort spent shaving it was wasted.
- `investigate` measures **avg 24.87 s** (p50 26.71, p95 34.03) — LAFS uses the *average*, so it
  sits 2.03 s inside the cliff at 26.9 s, on the correct side.
- That leaves roughly **23 seconds of per-query budget we are not spending**, against a gap of
  +9.8 accuracy points. Deeper reranking, more investigate steps, an explicit sufficiency signal,
  or repeated reader calls are all affordable; none of them were affordable under the imaginary
  1-second ceiling.

A point at avg 24.87 s is only 2 s from the cliff where the bar jumps 51.0 → 58.6. Any future
change that adds latency must either hold the average under ~26 s with margin, or commit to
clearing 58.6.

## M8 is blocked by M9's dependency, not by its own

`build_submission_step_1_single_operating_point.py` rejects both runs outright:

```
error: runs/myelin_k25_web_small/run_args.json evaluator_model must contain 'gpt-5.2'
```

`submission_utils.py:197` checks the **reader** against `EXPECTED_READER_MODEL_SUBSTRING =
"qwen3.5-9b"` — we pass. `submission_utils.py:202` checks the **judge** against
`EXPECTED_EVALUATOR_MODEL_SUBSTRING = "gpt-5.2"` — we fail, because scoring used the local
Qwen3.5-9B. This is the protocol's own gate, and it confirms in code what was previously only our
caveat: **these runs are not leaderboard-comparable.** One `gpt-5.2` key unblocks M8 packaging and
M9's judge panel together; they are a single external dependency, not two.
