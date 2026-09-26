# M72b — aggregation depth, with the pool cap fixed *(pre-registered 2026-09-26, before any row)*

## Why

M72 gave counting and summing questions k = 18 and 8,192 tokens, and
measured **+1.0 [−0.2, +2.2]** (`m72-aggregation-depth.md`). Paired under
LongMemEval's own grader, it fixed 5 multi-session counts and broke 2.

The round-4 code review found that the arm never had its depth. Every
`investigate` probe asked for `step_k` = 10 records whatever the question's
k, so the pool was 10 per probe and the evidence grew 7.9 → 13.4 items, not
to 18. PR #136 makes each probe ask for `max(step_k, k)`, and the shipped
k = 6 is unchanged.

There is a caution in the other direction:
- Karunanidhi 2026 (arXiv 2608.21230) measured LongMemEval_S multi-session
  accuracy **falling** with depth: 0.742 / 0.677 / 0.613 at k = 15 / 30 / 50.
  The reader over-counts from plausible extra context.
- Our own anatomy has 4 over-counts among the 5 count errors that held every
  gold turn.

Depth can cut both ways, and this arm measures which way it cuts here.

## The arm

- M72's exact command (`--aggregation-k 18 --aggregation-budget-tokens
  8192`) on the fixed binary, on the shipped M57 configuration.
- Only the rows `is_aggregation_question` fires on are re-run. The others
  have byte-identical prompts, and are copied from `m57`.
- Output: `runs/m72b_lme_agg`.
- Judged by the strict 9B (seeded) and the official grader.

**Stratum gate.** The switch enters the bundle only if all three hold:
1. On the multi-session rows the gate fires on, the paired difference under
   the strict judge has a 95% CI excluding zero.
2. The official difference is positive.
3. The abstention rows are unchanged.

**Reported beside it:**
- evidence items per fired row (it should now reach 18);
- the **over-count rate**: answered counts above gold, against the base;
- gold turns held (all / part / none), against M72.

**Predictions.**
- Evidence per fired row ≈ 18.
- The share holding every gold turn rises over M72's.
- Net fixes on multi-session counting: **+3 to +6**.
- Over-counts rise by 1–3. That is the price, and it is reported.

**Falsifier.** The over-count rate rises more than the under-count rate
falls, and the net is at or below zero. That is Karunanidhi's finding
reproduced, and depth ships off.
