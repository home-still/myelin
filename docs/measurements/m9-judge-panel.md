# M9 — judge panel and inter-rater agreement

`PLAN.md` M9 wants a judge panel at $\kappa \ge 0.89$, because an accuracy number built on one LLM
judge inherits that model's quirks and there is no way to tell from the number alone.

Judges are the **vendored** `qa_eval_metrics.llm_abstention_checker` and `llm_gotchas_checker`,
called with only `evaluator_model` and `evaluator_base_url` varied. The prompts, parsing and retry
behaviour are the harness's own; reimplementing them would measure agreement between my prompt and
theirs. Gemini is reached through its OpenAI-compatible endpoint, so no adapter code exists.

## Only 35% of the benchmark is judged at all

| count | eval function | kind |
|---|---|---|
| 200 | `norm_phrase_set_match` | deterministic |
| 128 | `llm_abstention_checker` | **LLM** |
| 68 | `mc_choice_match` | deterministic |
| 28 | `llm_gotchas_checker` | **LLM** |
| 26 | `norm_phrase_set_match_ordered` | deterministic |
| 1 | `mc_choice_set_match` | deterministic |

156 of 451 judged, 295 deterministic. That bounds how far any judge disagreement can move a
headline — and note that *every* abstention verdict is LLM-decided, which is exactly the slice
where our numbers are weakest.

## Two judges, 86 questions (the `web` judged slice)

| judge | marked correct |
|---|---|
| Qwen3.5-9B (local, the original scorer) | 25.6% |
| `gemini-3.1-flash-lite` | 27.9% |

- raw agreement **95.3%**
- Fleiss' $\kappa$ = **0.8813**

**Gate: FAIL**, by 0.0087.

## The direction is the useful part

The concern a panel exists to test is self-grading inflation — that scoring our own runs with a
local model flatters them. It does not. The frontier judge is *more* lenient than ours: 27.9%
against 25.6%. Our numbers are, if anything, slightly harsh.

The headline impact is bounded and small:

$$(0.279 - 0.256) \times \frac{86}{240} = +0.82 \text{ points}$$

So `web` overall would move **36.7% → 37.5%** under the frontier judge, against a **14.3-point**
deficit to the 51.0 bar. **Judge choice cannot explain the G1 gap** — it is worth under one point,
and in our favour.

## Three judges, 12 questions — quota-limited, not reportable

| judge | marked correct |
|---|---|
| Qwen3.5-9B (local) | 25.0% |
| `gemini-3.1-flash-lite` | 25.0% |
| `gemini-3-flash-preview` | 25.0% |

| pair | agreement |
|---|---|
| `gemini-3-flash-preview` vs `gemini-3.1-flash-lite` | **100.0%** |
| `gemini-3-flash-preview` vs Qwen3.5-9B | 83.3% |
| `gemini-3.1-flash-lite` vs Qwen3.5-9B | 83.3% |

$\kappa = 0.7037$ at $n = 12$, where a single flipped verdict moves agreement 8.3 points. **Do not
read this as a result.** The one pattern worth noting, because it is consistent rather than
marginal: the two Gemini models agree perfectly with each other and each differs from the local
model on the same two items, which is what a family effect looks like and is the reason a panel
should not be built from one vendor.

## Why it stops here

The key is free-tier, and Gemini caps `generate_content` at **20 requests per day per model**
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). `gemini-3.1-flash-lite` sits in a higher
tier and completed all 86; `gemini-3.5-flash` and `gemini-3-flash-preview` exhausted after 0 and
12 respectively.

`adapters/judge_panel.py` therefore caches every verdict to disk keyed by `model|question_id` and
judges only what is missing, so successive days accumulate a complete panel instead of restarting.
A paid key finishes it in one pass. Without the cache each 429 would discard work already paid for.

**To finish this properly:** re-run the same command on subsequent days until the third judge
completes all 86, or supply a paid key. The command is unchanged either way:

```
python adapters/judge_panel.py runs/myelin_k25_web_small \
  --gemini-models gemini-3.1-flash-lite gemini-3-flash-preview
```
