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
