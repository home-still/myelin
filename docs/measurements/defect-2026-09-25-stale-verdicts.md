# Defect: seeded verdicts counted on declined answers *(found and fixed 2026-09-25)*

**What was wrong.** `myelin-eval judge --seed <base>` reuses a base run's
verdict only when the answer is byte-identical (M42). The check ran only over
the rows the judge grades, and the judge never grades a decline. So when an
arm answered "I don't know." where its base had answered:
- the base's verdict stayed in the arm's `judge_verdicts.json` under the
  arm's question id;
- every scorer (`standing`, `rescore --scorer judge`, and the scratch
  `locomo_paired.py`) looked the verdict up **before** applying the decline
  rule;
- so a declined row whose base answer had been right counted as **correct**.

**How it was found.** M70 graded the shipped LongMemEval_S run with
LongMemEval's official judge and got 78.60 against our 83.40. The rows the
two graders disagreed on were decline-first answers the strict judge had
"passed". Their stored `answers` were not the answers in the run.

**The fix** (`judge.rs`, `bench.rs`, `standing.rs`):
- the judge keeps a verdict only for a row it still grades, and only for
  the answer that row holds now (`judge::reusable`);
- every scorer applies the decline rule first;
- a verdict counts only for the answer it graded (`judge::verdict_for`);
- all 16 affected runs were re-judged (no model calls: every stale row is a
  decline) and their `_judged` dirs rescored;
- tests: `a_verdict_survives_only_for_the_answer_it_graded_on_a_row_still_judged`,
  `verdict_for_refuses_a_verdict_recorded_for_another_answer`,
  `a_stale_verdict_never_scores_a_decline_or_a_changed_answer`.

## What changed *(strict 9B judge, all 500 or all 1,540 rows)*

| run | milestone | stale "correct" verdicts | was | corrected |
|---|---|---|---|---|
| `m43_dated` | M43 | 8 | 67.80 | **66.20** |
| `m44_r1` | M44 R1 | 3 | — | 66.40 |
| `m44_r2_s1` | M44 R2, the 9B base since then | 17 | 78.40 | **75.00** |
| `m44_r2_s2` | M44 R2, seed 2 | 18 | 78.40 | 74.80 |
| `m44_r2b_s1` | M44 R2b | 15 | — | 77.20 |
| `m55_bonsai_s1` | M55 | 11 | 82.20 | **80.00** |
| `m57_bonsai_premise_s1` | **M57, shipped** | 21 | **83.40** | **79.20** |
| `m63_locomo_dedupe` | M63 | 48 | 65.71 | 62.60 |
| `m63_locomo_inline` | M64 | 9 | 71.30 | 70.71 |

Also affected, by 2 to 11 rows each: M43's filtered arm, M45, M46, M48,
and the M50 and M55 pilots. Unseeded runs, including every fresh base
(`m63_locomo_base`, `m19_locomo_full`), were never affected.

**Claims, re-measured:**
- **M57 vs the 9B:** +4.20 [+1.20, +7.40] (was +5.0 [+2.2, +7.8]). It still
  clears the +3.0 bar with the CI excluding zero, so the ship decision
  stands.
- **M44 R2 vs M43:** +8.80 [+5.0, +12.8]. It still ships.
- **M55 vs the 9B:** +5.00 [+2.2, +8.0] (abstention veto unchanged).
- **M64 vs the fresh base:** +0.19 [−1.04, +1.43] (was +0.78). Still null.
- **M63 vs the fresh base:** −7.92 [−9.68, −6.17] (was −4.81). Still
  falsified, and worse.
- **LongMemEval_S against the same-size row (MemPro-15 Qwen3-30B, 80.80):
  the "past it" claim of 2026-09-24 was wrong.** Strict is 79.20 (−1.60),
  and LongMemEval's own grader gives 78.60 (M70). The gate stays open.
- **The ratchet floor for `longmemeval_s.judge_score.n500`** was lowered by
  hand from 83.40 to 79.20. It was pinned from inflated verdicts, and the
  floor must be a measured value.

**Not affected:**
- M68's LightMem and MemPro readings, and M70's official reading: those
  adapters key every verdict on (question id, answer).
- Every LME-V2 number (a different harness).
- The MINJA gate.
