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

**R2 — thinking on.** `BenchSwitches::reader_thinking`: the same
`READER_SYSTEM` prompt every milestone used, `enable_thinking: true` per
request, sampled at the Qwen3 Technical Report's thinking-mode setting
(`10.48550/arxiv.2505.09388`: temperature 0.6, top-p 0.95, top-k 20) under a
recorded `--reader-seed`; two seeds so the CI carries sampling noise. The
thinking budget is **1,024 tokens**, enforced by the reader server's
`--reasoning-budget` (`MYELIN_READER_THINK_BUDGET`), which the harness cannot
read back — so `verify_thinking_budget` measures it before the first row with
a prompt that thinks far past the budget, and the run refuses to start if the
trace runs into the ceiling instead of being cut. The scorer sees `content`
only; the trace stays in `reasoning_content`. R2 runs after R1 lands so the
two can be read against each other, and it needs the reader restarted with
the budget between them.

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

## Results — R1

**Run.** `runs/m44_r1` (raw, `judge_verdicts.json` seeded from
`runs/m43_dated`: 204 judged, 244 reused) → `runs/m44_r1_judged`. 500 rows,
8.2 s/row against the base's 7.1. The control is exact: **273 byte-identical
answers, delta 0.0**.

**Headline: −0.8 (95% CI [−3.8, +2.2], p = 0.65). Null. The abstention veto
fires: 90.0 → 43.3 on the 30 rows, −46.7 [−63.3, −30.0]. `reader_reasoning`
ships off.**

| stratum | n | base | R1 | delta | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- |
| **overall** | 500 | 67.8 | 67.0 | −0.8 | [−3.8, +2.2] | 0.647 |
| answerable | 470 | 66.4 | 68.5 | +2.1 | [−0.6, +4.9] | 0.159 |
| **abstention** | 30 | 90.0 | 43.3 | **−46.7** | [−63.3, −30.0] | <0.0001 |
| gold = 1 | 170 | 82.4 | 84.1 | +1.8 | [−1.8, +5.9] | 0.451 |
| **gold = 2** | 229 | 62.9 | 65.9 | +3.1 | [−0.4, +7.0] | 0.127 |
| gold ≥ 3 | 71 | 39.4 | 39.4 | +0.0 | [−11.3, +11.3] | 1.000 |
| base declined | 76 | 46.1 | 35.5 | −10.5 | [−22.4, +1.3] | 0.100 |
| base answered | 424 | 71.7 | 72.6 | +0.9 | [−1.9, +3.8] | 0.576 |
| `multi-session` | 133 | 59.4 | 59.4 | +0.0 | [−6.8, +6.8] | 1.000 |
| `temporal-reasoning` | 133 | 48.1 | 51.1 | +3.0 | [−3.8, +9.8] | 0.442 |
| `knowledge-update` | 78 | 80.8 | 74.4 | −6.4 | [−12.8, −1.3] | 0.011 |
| `single-session-user` | 70 | 94.3 | 91.4 | −2.9 | [−8.6, +2.9] | 0.444 |
| `single-session-preference` | 30 | 40.0 | 36.7 | −3.3 | [−20.0, +13.3] | 0.852 |
| `single-session-assistant` | 56 | 98.2 | 98.2 | +0.0 | [+0.0, +0.0] | 1.000 |

### What the predictions did

1. **gold = 2 leaned and did not move**: +3.1 [−0.4, +7.0]. `multi-session`,
   the stratum M43's digest moved +11.3, moved **exactly +0.0** — 11 gained,
   11 lost. `temporal-reasoning` +3.0, not significant. The mechanism's own
   target did not respond to the mechanism.
2. **gold = 1 did not regress** (+1.8). The prediction held; it was the easy
   one.
3. **Abstention fell by half.** Predicted not to fall; fell 27 → 13 of 30.
   Seven of the fourteen answers turn absence into a quantity — `0`,
   `0 days`, `0 minutes`, `Never`, `Nothing` — and the rest borrow a
   neighbour's fact (`three months` of vintage *films* from the vintage
   *cameras* question) or invent one (`Harvard University`, `15` fish). The
   `evidence_absent` field, written *after* `reasoning` and `answer`, held
   on 13 rows and gave way on 14. It is M42's hatch with a longer run-up.
4. **`knowledge-update` −6.4 is the veto, not a `knowledge-update`
   regression**: all five rows lost there are abstention rows whose
   question happens to carry that type. On the 73 answerable
   `knowledge-update` rows the arm is unchanged.

### Declines, and what replaced them

Declines fell **76 → 35**, and every one of the 421 answerable rows the base
answered is still answered. On the **49 answerable rows the base declined**,
R1 answers 27 and is right on **11 (40.7%)** — the same conversion rate M42's
forced commit measured on its 31 committed rows (41.9%). Two mechanisms, two
prompts, one number: when this reader is made to answer instead of decline,
it is right two times in five, and on the adversarial rows it is made to
answer just as readily.

What R1 does add that M42 did not is **arithmetic**: the rows it gains are
`31` days for a base `19`, `$65` for a base `$15`, `43` years older, `3`
days in December — the reasoning field is where the subtraction happens.
That is M46's numerator, and it says the computation is worth having; M46
puts it in the evidence instead of in a field that also talks the reader out
of refusing.

### Verdict

`reader_reasoning` ships **off**. It is the third mechanism in this project
(M35, M42, M44 R1) whose gain on answerable rows is bought with the reader's
licence to refuse, and the bill is the same each time: ~40% of the
newly-answered rows are right, ~50% of the adversarial rows are lost. The
discriminator has to come from somewhere other than the model's own say-so —
that is M45's premise, and this result is its second data point.

**On the falsifier.** R1 did not move gold = 2 significantly. The
pre-registration says the theory is refuted only if **neither** arm does;
R2 — native thinking under a 1,024-token budget, sampled, two seeds — is the
remaining test, and it runs next with the trace recorded per row
(`ScoredQuestion::reader_trace`, absent on this arm, which predates it).

## Results — R2

*(pending — needs the reader restarted with `MYELIN_READER_THINK_BUDGET=1024`
after the M46 arm releases it)*
