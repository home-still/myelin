# Round-4 bundle — the three positive mechanisms together *(pre-registered 2026-09-27; measured 2026-09-28: strict +1.4, vetoed; official 80.8 ties 80.80; does not ship)*

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

## Result — below the bar and vetoed; does not ship *(measured 2026-09-28)*

`runs/r4_bundle_s1` was run as 500 rows on big, 20:54 → 02:08. It was then
followed by `commit-arm --grounded` (5 of 75 declines committed), making
`runs/r4_bundle_s1_grounded`.

| reading | bundle | M57 | Δ [95% CI] | p | abstention |
|---|---|---|---|---|---|
| strict 9B | 80.6 | 79.2 | +1.4 [−0.8, +3.6] | 0.25 | **28/30** (M57 29/30) |
| official | 80.8 | 78.6 | +2.2 [+0.0, +4.6] | 0.07 | 28/30 (M57 28/30) |

- **The bar is not met.** Strict +1.4 is under +3.0, its CI crosses zero,
  and abstention fell to 28/30, so the veto fires. Under LongMemEval's own
  grader the bundle *ties* MemPro-15's 80.80 and does not lead it.
- **The predictions held.** Strict +1.2 to +1.8 and official +2.0 to +2.6
  were predicted, so the three mechanisms add. The gain concentrates in
  category 5: strict +3.8 [+0.8, +7.5], official +4.5 [+1.5, +8.3], with 5
  rows fixed and none broken.
- The falsifier (the bundle below the best single mechanism's +0.6) did not
  fire.

**The veto row is reader noise, not a mechanism.** `a96c20ee_abs` ("at which
university did I present a poster for my undergrad course research
project?") answered "Harvard University" where M57 declined. No gate fired
on it (no events block, no counting shape), its evidence is byte-identical
to M57's, and it answered before the grounded pass ran. The reader simply
answered differently.

**How much of the +1.4 is noise.** The rows no mechanism touches measure the
rerun floor directly:

| ungated rows (outside M72b's 137 and M73b's 18) | 346 |
|---|---|
| evidence byte-identical to M57 | 282 |
| of those, answers byte-identical | 215 |
| strict verdict flips on identical-evidence rows | +6 / −1 |

- On rows whose evidence did not change at all, the reader moved the score
  by +5 rows, about **+1.0 point**, from nondeterminism alone.
  - The thinking reader serves 2 slots, so batch composition differs
    between runs.
  - Another 64 ungated rows' evidence changed through the LLM selector's
    own nondeterminism.
- So the three mechanisms' attributable gain in this run is well under the
  measured +1.4. **A single full rerun cannot resolve differences of about
  one point.**
- This is why round 4's stratum arms copied M57's untouched rows. The
  bundle's full rerun reintroduced the noise those arms were designed to
  exclude.

**What this means for the gap (1.60 strict, 2.20 official).** No
combination of round 4's mechanisms closes it. The next gains must be
larger per mechanism, and they must be measured as partial reruns, or over
several seeds, so they stand above the rerun floor.
