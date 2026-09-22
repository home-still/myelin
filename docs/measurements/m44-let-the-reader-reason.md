# M44 — the reader has never been allowed to reason

## The finding

`CompletionRequest::with_thinking(true)` exists in `llm/mod.rs`, is unit
tested, and has **no production call site**. `ops/big/serve-models.sh` pins
`--chat-template-kwargs '{"enable_thinking":false}'` server-wide. Every reader
call is capped at 160 completion tokens and `READER_SYSTEM` says *"Answer in
as few words as possible … Do not explain."*

Meanwhile the vendored LongMemEval-V2 harness
(`vendor/longmemeval-v2/evaluation/harness.py:186`) sets
`reader_enable_thinking=True` **by default** with a 20,000-token completion
budget, and our own adapter (`adapters/run_myelin.py:201`) overrides it to
`False`.

Two consequences:

1. **The AgentRunbook-R row is not a same-reader comparison.** Its 58.60 came
   from a *thinking* Qwen3.5-9B with a 20,000-token budget; our 38.58 from the
   same weights with thinking off and 160 tokens. `BACKLOG_DONE.md` now labels
   it `caveat-reader-mode` until an equal-configuration number exists.
2. **Every diagnosis since M38 was taken under that configuration.** The
   2-fact collapse (M39: 79.9 → 56.7 → 40.0), the ignored instructions (M38,
   M39, M40, M43), the 63 false declines (M42) — all textbook behaviour for a
   small model denied reasoning tokens.

Tam et al., *Let Me Speak Freely?* (EMNLP Industry 2024,
`10.18653/v1/2024.emnlp-industry.91`) measured exactly this failure. Under
JSON mode, **100%** of GPT-3.5-Turbo responses placed the `answer` key before
the `reason` key, "resulting in zero-shot direct answering instead of
zero-shot chain-of-thought reasoning", and LLaMA-3-8B-Instruct lost **38.15%**
on Last Letter. Their §5.2: *"in reasoning related task, JSON-mode failed to
adhere to the order of reasoning first followed by answer causing a large drop
in final performance."* `READER_SYSTEM` is that failure mode with no reason
field at all. Their remedy is the one this project keeps re-deriving —
**structure, not instruction**: put the reasoning field first so the answer
is generated after it.

Their Table 2 carries the caveat that bounds R1: even with reasoning-first
JSON-Schema output, natural language still beat the schema on 2 of 3 reasoning
tasks for gpt-4o-mini. A reasoning *field* is a partial restoration; native
thinking (R2) is the full one.

## The mechanism

**R1 — structured reasoning, thinking off.** `BenchSwitches::reader_reasoning`
replaces the bare reader call with a strict schema
`{reasoning, answer, evidence_absent}` in that field order. `reasoning` is
bounded at 600 characters; the completion ceiling rises 160 → 480 so the
trace cannot eat the answer. Temperature stays 0, isolating *let it reason*
from *sample*. `evidence_absent: true` maps to the one decline string
`is_abstention` already recognises, so the abstention contract and M42's veto
are unchanged. An unparseable response is returned verbatim and graded as
what the model said.

This is the third application of the ordering rule M42 (`answer` before
`evidence_absent`) and M43 (`says` before `bears_on_question`) each used.

**R2 — thinking on.** `enable_thinking: true` per request, bounded thinking
budget, sampling per the Qwen3 Technical Report (`2505.09388`: temperature
0.6, top-p 0.95, top-k 20), two seeds so the CI carries sampling noise. Not in
this document's first pass; it runs after R1 lands so the two can be read
against each other.

## Pre-registration

Written before the R1 arm ran.

**Base.** The shipped operating point as of M43:
`myelin-eval bench --corpus longmemeval-s --mode investigate --k 6
--budget-tokens 4096 --max-steps 2 --select-sufficient --item-digest
--digest-dates`, judged — `runs/m43_dated_judged`, **67.80**.

**Arm.** The same command plus `--reader-reasoning` → `runs/m44_r1`, judged
with `--seed runs/m43_dated_judged` so an unchanged answer is never re-graded
(M42's method fix).

**Population.** All 500 LongMemEval_S rows.

**Primary metric.** `longmemeval_s.judge_score.n500`, paired bootstrap over
the per-question difference (`adapters/paired_ci.py`).

**Decision rule.**

- Ship `reader_reasoning` on if the judged score improves by **≥ +3.0** over
  base with a paired 95% CI excluding zero.
- **Veto:** any drop on the 30 abstention rows (base 90.00, 27/30) ships it
  off whatever the headline says.

**Predicted, specifically.** Strata are measured on the base *before* the
arm ran (`gold = k` is the number of `answer_session_ids` on the answerable
rows):

| stratum | n | base | prediction |
| --- | --- | --- | --- |
| gold = 1 | 170 | 82.35 | does not regress by more than 1.0 |
| **gold = 2** | 229 | 62.88 | **moves up** — the mechanism's target |
| gold ≥ 3 | 71 | 39.44 | moves up |
| `temporal-reasoning` | 133 | 48.12 | carries gain |
| `multi-session` | 133 | 59.40 | carries gain |
| abstention rows | 30 | 90.00 | **does not fall** |
| declines (`I don't know.`) | 76 | — | fall, without the abstention rows falling |

**Falsifier.** If R1 does not move gold = 2, and R2 does not either, the
reader is not compute-limited: M38's "reading is the gap" theory is refuted
at the cheapest possible point, and the project redirects to write-time
aggregation (M50) with far more confidence. If R1 ≈ R2 the cheap ship is the
bounded field; if R2 ≫ R1 the gain is deliberation and needs the GPU window.

**Cost.** R1: one ~60-minute run at ~7 s/row, no re-ingest. R2: 2–6
GPU-hours.

## Results

*(pending — the arm is running)*
