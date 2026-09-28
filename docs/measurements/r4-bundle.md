# Round-4 bundle — the three positive mechanisms together *(pre-registered 2026-09-27, before any row)*

## Why this arm exists outside the rule

Under the user's rule, every mechanism clears its stratum gate first, and one
bundle of the passing switches must then reach +3.0. In round 4 none passed.
Three were positive under both graders, but their strata (18 to 90 rows) are
too small to separate a 2–3 row gain from zero:

| mechanism | stratum strict Δ | all-500 strict Δ | all-500 official Δ |
|---|---|---|---|
| M71b, the grounded pass covers every named thing | +6.1 [+0.0, +14.3] | +0.6 | +0.6 |
| M73b, events from the question's own date | +11.1 [−11.1, +33.3] | +0.4 | +1.0 |
| M72b, aggregation depth with the cap fixed | +4.4 [−3.3, +12.2] | +0.6 | +0.8 |

Stacked, that is about +1.6 strict (≈ 80.8) and +2.4 official (≈ 81.0),
against MemPro-15's 80.80. The user chose (2026-09-27) to measure whether the
three add up, as one 500-row arm. **This arm does not change the rule.** It
ships only if it clears the bar below *and* the user decides to ship it.

## Arm

`runs/r4_bundle_s1`: the shipped M57 configuration plus the following, over
all 500 questions on M57's serving (Bonsai 27B, 2 slots × 32K, on big itself):
- `--events-ledger data/longmemeval_s_events.ledger`, for M73b;
- `--aggregation-k 18 --aggregation-budget-tokens 8192`, for M72b.

Then `commit-arm --grounded` over its declines, for M71b
(`runs/r4_bundle_s1_grounded`). Driver: `bundle_big.sh`.

**Judged** by the strict 9B, seeded from M57, and by LongMemEval's official
grader, seeded from M57's official verdicts.

## Bar (the user's, unchanged)

- **+3.0 strict over all 500**, with the paired 95% CI excluding zero.
- Abstention not below 29/30.
- A lead under the official grader too.

## Predictions

- Additive at best: strict about +1.2 to +1.8, and official about +2.0 to
  +2.6.
- Some overlap is expected. M72b's 137 counting rows include some of M71b's
  declines, and a decline M72b already answers leaves M71b nothing to fix.
- Rows outside all three gates (about 330) should be byte-identical in
  evidence to M57's, apart from reader nondeterminism.

## Falsifier

- If the bundle lands below the best single mechanism's +0.6, the three
  interfere. Report the rows where they disagree.
