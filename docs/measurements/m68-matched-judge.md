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
