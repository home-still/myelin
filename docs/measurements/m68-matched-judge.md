# M68 — LoCoMo graded the way the MemPro row was graded *(pre-registered 2026-09-25, before any verdict)*

## Why

The LoCoMo finish line is MemPro-15 on Qwen3-30B-A3B: **77.85**
(Liu et al. 2026, arXiv 2606.00619, Table 1). That number was graded by
**gpt-4o-mini with LightMem's LoCoMo judge prompt** (its L122 and Fig. 10).
Ours (70.52 on the fresh base, 69.87 shipped) is graded by a strict local 9B
rubric (`crates/myelin-eval/src/judge.rs`). `standing` has always marked the
comparison `caveat-judge`. It is "the one asymmetry left"
(`docs/measurements/m18-sota-standing.md`).

The two graders are not close:
- LightMem's prompt says "you should be generous with your grading — as long
  as it touches on the same topic as the gold answer, it should be counted as
  CORRECT".
- EdgeMem (Cui et al. 2026, arXiv 2609.05553) measured this family of
  lenient judges flipping **17.7%** of strict "wrong" verdicts to correct.
- A LoCoMo audit (Penfield Labs, 2026, a blog, not peer-reviewed) reports it
  accepting up to 63% of deliberately wrong answers.

So the size of our gap to 77.85 is unknown until our answers meet the same
grader. The 2026-09-25 loss anatomy found 15 of 20 near-equivalent losses
to be judge or label disagreements under the strict rubric.

## The protocol

`crates/myelin-eval/adapters/judge_lightmem.py` reproduces LightMem's grader
and nothing else:
- **The prompt** is `ACCURACY_PROMPT`, byte for byte, from
  `zjunlp/LightMem@8449d574df6bae1bdf3314a1564da65e2f37e046:experiments/locomo/llm_judge.py`
  (sha256 `62395dd3…`, recorded in every verdict file).
  - MemPro's Fig. 10 prints it without the "First, provide a short (one
    sentence) explanation" line. The code is what ran, so the code is used.
- **The call:** one user message, `temperature=0.0`,
  `response_format=json_object`, `gpt-4o-mini` pinned to OpenAI's own
  provider on OpenRouter (no fallback host).
- **Parsing:** the label goes through LightMem's `extract_json`. Correct
  means the label is exactly `CORRECT`.
- **Rows:** every category 1–4 row is judged, declines included. Category 5
  is skipped, as LightMem's `main` does.
- **Files:** the verdicts go to `<run>/judge_verdicts_lightmem.json` with
  the protocol that made them. The strict `judge_verdicts.json` is never
  touched. A cache or seed from another protocol is refused.

`standing` reads that file as a separate metric,
`locomo.judge_score_lightmem.n1540`, with judge class `frontier_api`, and
compares it with the MemPro rows under the same metric.

## What it decides and what it does not *(the user's call, 2026-09-25)*

- The **strict 9B number stays the headline and the ship gate.** Arms are
  still decided on it: +3.0 with the CI excluding zero, plus the vetoes.
- The **matched number settles only the comparison** with MemPro's rows.
  It is published next to the strict one, never instead of it.
- It uses a cloud model as a checker, which the model policy allows
  ("local AI + cloud checker"). The reader, the memory and every answer stay
  local.

## Runs

1. `runs/m63_locomo_base` (the fresh base, 70.52 strict).
2. `runs/m19_locomo_full` (shipped, 69.87 strict), seeded from 1 for
   byte-identical answers.

## Predictions (before any verdict)

- The matched score lands **74–78**. The central guess is ~75.7, from
  EdgeMem's 17.7% flip rate on our 29.48% strict-wrong. That is derived,
  across a different reader and judge.
- It moves most on **open-domain** (the most judge-sensitive category) and on
  **multi-hop**, where wrong answers are often supersets of the gold list,
  which "touches on the same topic" accepts.
- Declines stay wrong. A decline touches no topic.

**Falsifier.** If the matched score is within ±1.5 of the strict one, the
judge asymmetry is not what separates us from MemPro, and the whole gap is
the system's.

## Result *(measured 2026-09-25, 17:05–17:15; about $0.30 of judge calls in total)*

| run | strict 9B judge | **LightMem protocol (gpt-4o-mini)** | MemPro-15 (Qwen3-30B) |
|---|---|---|---|
| `runs/m63_locomo_base` (shipped settings, today's code) | 70.52 | **78.18** (1,204 / 1,540) | 77.85 |
| `runs/m19_locomo_full` (the M19 artifact) | 69.87 | **77.86** (1,199 / 1,540) | 77.85 |

**The LoCoMo gate closes on a comparable row, by a hair:**
- today's code: +0.33 over MemPro-15 on Qwen3-30B-A3B, same judge, same
  prompt, same 1,540 questions;
- the M19 artifact: +0.01.

Neither is a clear win, and the doc says so. The margin is:
- **larger than the judge's own noise.** Re-judging the base's identical
  answers flipped 6 rows (net −2, so 78.05);
- **far smaller than the reader's.** A date-rendering change alone flips 88
  rows (M64);
- **set against a different estimate.** MemPro's figure is a mean of 3 runs
  at temperature 0.7; ours is one greedy run.

**By category, matched protocol (base) vs MemPro-15 Qwen3:**

| category | n | strict | matched | MemPro | matched − MemPro |
|---|---|---|---|---|---|
| multi-hop | 282 | 59.57 | 70.57 | 75.17 | **−4.60** |
| temporal | 321 | 60.44 | 72.90 | 67.60 | +5.30 |
| open-domain | 96 | 31.25 | 36.46 | 70.83 | **−34.37** |
| single-hop | 841 | 82.52 | 87.51 | 83.47 | +4.04 |

- **The two graders disagree on 134 of 1,540 rows.** 126 rows are wrong
  under the strict rubric and right under LightMem's; 8 go the other way.
  The 126 include:
  - real equivalents (`Xeonoblade Chronicles` for `Xenoblade Chronicles`,
    ISO ranges that are the gold week);
  - generous passes (`2023-08-07` for "the week before 7 August 2023";
    partly overlapping lists).
- **No decline was passed:** 0 of 116.

**Against the predictions:**
- The range held: predicted 74–78, measured 78.18, at the top.
- Multi-hop moved most (+11.0), as predicted: supersets of the gold list
  "touch on the same topic".
- Open-domain moved least (+5.2), **against** the prediction. Its losses
  are declines and speculation misses, which no grader rescues.
- The falsifier did not fire: the matched score is 7.66 points above the
  strict one. The judge asymmetry was most of the LoCoMo gap.

**What changes:**
- `standing` publishes `locomo.judge_score_lightmem.n1540` beside the strict
  metric, reading `judge_verdicts_lightmem.json` only when it names
  LightMem's prompt sha, and refusing otherwise.
- The G2 LoCoMo gate moved to the matched MemPro row (the user's call). The
  strict comparison stays published as `caveat-judge`.
- The strict 9B judge still decides every arm.

**What it does not change:**
- The losses are real under both graders. Multi-hop trails by 4.6 and
  open-domain by 34.4 under MemPro's own judge.
- A +0.33 lead is inside the reader's variance, so LoCoMo work goes on:
  M66 turn windows (multi-hop coverage) is next.

## M68b — a second reading of MemPro's protocol *(pre-registered 2026-09-25, before any verdict)*

- **What else there is.** MemPro's paper cites LightMem and prints its
  prompt (Fig. 10). MemPro's **public repo** grades LoCoMo with a paraphrase
  of it: `wanghai673/MemPro@834b1ce:eval/locomo_test.py`
  (`JUDGE_PROMPT_TEMPLATE`, `JUDGE_SCHEMA`, `call_llm_judge`).
  - The rule is the same ("touches on the same topic → CORRECT").
  - It has no Hawaii example.
  - Its output is strict json_schema, with gpt-4o-mini at temperature 0.
  - A judge failure scores WRONG. We refuse instead, so a transport error is
    never mixed into the number.
- **Both readings are runnable, and neither can be ruled out.**
  `adapters/judge_matched.py` (renamed from `judge_lightmem.py`) carries both
  as protocols. Its `--verify-upstream` checks each prompt byte for byte
  against the pinned upstream file, and both pass.
- **The user's rule (2026-09-25):** a gate closes only if we lead under
  **every** runnable reading. `standing` publishes each reading, and the gate
  row uses the lower.
- **Prediction:** within ±1 of the LightMem reading (78.18), because the rule
  and the model are the same.
- **Falsifier.** If the MemPro reading lands below 77.85, the LoCoMo gate
  reopens. This doc then says so.
