# Round-5 bundle — seed-replicated, only the changed rows rerun *(pre-registered 2026-09-28, before any row and before M78's result)*

## Why

Round 4's bundle (`r4-bundle.md`) stacked three mechanisms that were positive
but too small for their strata:
- M71b, the grounded pass;
- M73b, dated events;
- M72b, aggregation depth.

It gave official **80.8, a tie with MemPro-15's 80.80**, and strict +1.4, and
it was vetoed by one abstention row, `a96c20ee_abs`. No mechanism touched that
row, and its evidence was byte-identical to M57's. The reader simply answered
differently. Rows no mechanism touched moved +5 net from rerun noise alone.

Since then, preference has its first working mechanism. The clause alone
(M77c) gives official +18.9 [+8.9, +30.0] on the 30 rows, and strict +11.1
[+1.1, +22.2] on the rubric-corrected judge (a post-hoc pass;
`judge-preference-rubric.md`).

## Method (the user's choice, 2026-09-28: "3 seeds, only changed rows rerun")

- **U, the rows a bundle switch can change**, is fixed by the switches' gates.
  Each gate reads the question text only, never an outcome:
  - `is_aggregation_question` (M72b): `m72_agg_ids_rust`, 137 rows;
  - `question_window` (M73b's events block): `m73b_fire`, 18 rows;
  - `is_advice_request` (M77c/M78): `m20b_ids`, 29 rows.

  **U = 183 rows.** Rows outside U hold byte-identical reader input in both
  arms by construction. A 5-row control outside U (`control_ids`, evidence
  only, bundle switches on) must reproduce M57's evidence byte for byte, or
  the run stops.
- **Seeds:** U is rerun at reader seeds 1, 2 and 3 for the bundle, and at
  seeds 2 and 3 for the base. The base's seed 1 is M57 itself, and its
  advice rows at seeds 2–3 are `m57_pref_s2`/`m57_pref_s3`. Every row outside
  U is M57's seed-1 row in every replicate of both arms. So rerun noise enters
  only where a switch acts, and the seeds average it there.
- **M71b is a post-pass.** `commit-arm --grounded` runs over each bundle
  replicate's declines. It is a greedy second call, so rows outside U get the
  same pass in every replicate.
- **Why not select rows by outcome:** rerunning "the rows the base declined"
  would regress their fresh draws toward answering and flatter the bundle.
  U is chosen on the question text for that reason.
- **Paired:** `paired_ci.py` over the three replicates a side, per-question
  means, with the bootstrap over questions. The strict reading uses the 9B
  with the preference rubric; the official reading uses LongMemEval's grader.

## Arms

- **Base:** the M57 configuration. Replicates: `m57_bonsai_premise_s1`,
  `r5_base_s2`, `r5_base_s3`.
- **Bundle:** the M57 configuration plus:
  - `--events-ledger data/longmemeval_s_events.ledger` (M73b);
  - `--aggregation-k 18 --aggregation-budget-tokens 8192` (M72b);
  - `--advice-profile-clause` (M77c). If M78 passes its gate *and* beats
    M77c's seed-averaged official preference score, as pre-registered in
    `m78-advice-answer-structure.md`, it adds `--user-words --advice-answer`.

  Then `commit-arm --grounded` (M71b). Replicates:
  `r5_bundle_s{1,2,3}_grounded`.
- Driver: `r5_big.sh "<bundle flags>"` on big.
- Cost: about 860 row-reads, roughly 9–10 h, plus about 30 min of judging.

## Bar (the user's, unchanged; read on the seed means)

- **+3.0 strict over all 500**, with the paired 95% CI excluding zero.
- Abstention, as the seed mean, not below 29/30.
- A lead under the official grader too, meaning the bundle's seed-mean
  official score is above 80.80.

## Predictions

- **Round 4's three mechanisms:** about +0.5 to +1.2 strict on the seed means.
  That is less than round 4's single-seed +1.4, since part of it was noise.
- **The clause:** about +0.6 strict and +1.1 official overall (its preference
  effect, spread over 500).
- **Together:** strict +1.0 to +2.2, and official +1.5 to +3.0. Abstention
  unchanged, since no gate fires on an abstention row and the grounded pass
  was abstention-safe in round 4's own measurement.
- **The +3.0 strict bar is expected to fail.** The run is the measurement the
  rule asks for, and its replicates are reusable: later post-pass mechanisms
  (commit-arm style) can be measured on top of them without rerunning.
- The official seed-mean is expected at about 79.6–81.1, which straddles
  80.80.

## Falsifiers

- **The control differs from M57:** rows outside U are not untouched, and the
  method does not hold. Stop.
- **Strict below +0.5:** the mechanisms do not add once noise is averaged,
  and round 4's result was mostly noise.

## Result — official seed mean 81.13, past 80.80; strict +2.5, short of the +3.0 bar *(measured 2026-09-28, 21:38)*

**The runs:**
- Bundle `r5_bundle_s{1,2,3}_grounded`: 183 rerun rows per seed, then the
  grounded pass, which committed 8, 8 and 6 of 73, 72 and 76 declines.
- Base `m57_bonsai_premise_s1`, `r5_base_s2`, `r5_base_s3`.

Readings: the strict 9B with the preference rubric, and LongMemEval's
official grader (gpt-4o-mini). `paired_ci` over three replicates a side.

| reading | bundle (seed mean) | base (seed mean) | Δ [95% CI] | p | W/L (sign p) |
|---|---|---|---|---|---|
| **strict 9B** | **80.80** (80.4, 81.2, 80.8) | 78.27 (79.0, 78.0, 77.8) | **+2.5 [+1.0, +4.2]** | 0.0010 | 26/12 (0.034) |
| **official** | **81.13** (80.2, 81.6, 81.6) | 77.87 (78.6, 78.0, 77.0) | **+3.3 [+1.7, +5.0]** | < 0.0001 | 31/11 (0.003) |
| abstention, strict | 29.33 / 30 | 29.00 / 30 | +1.1 | | 1/0 |
| abstention, official | 28.00 / 30 | 28.00 / 30 | +0.0 | | 1/1 |

**By category** (strict / official Δ):

| category | strict Δ | official Δ |
|---|---|---|
| 3, preference | +5.6 | **+16.7** (12/2) |
| 4, multi-session | **+5.0** | +3.8 |
| 5, temporal | **+2.3** | **+3.3** |
| 1, 2, 6 | within ±2.4 | within ±3.3 |

Single-session-assistant loses one row strict. Knowledge-update is flat.

**The bar:**

| criterion | result | |
|---|---|---|
| strict Δ ≥ +3.0 with CI > 0 | +2.5, CI [+1.0, +4.2] | **fails on size** (the CI clears zero) |
| abstention seed mean ≥ 29/30 and ≥ base | 29.33 ≥ 29.00 | holds |
| official seed mean > 80.80 | **81.13** | **holds** |

**As pre-registered, the bundle does not ship under the user's rule.** It is
the first measurement in this project to put LongMemEval_S past MemPro-15's
80.80 under the matched grader. The CI of +3.3 official excludes zero, it
wins 31 questions and loses 11, and abstention holds. Whether that ships is
the user's call (plan step 2.4).

**Predictions against the result:**

| prediction | result | |
|---|---|---|
| round 4's three mechanisms about +0.5 to +1.2 strict, the clause about +0.6: together +1.0 to +2.2 strict | +2.5 | above the range |
| official +1.5 to +3.0 | +3.3 | above the range |
| official seed mean 79.6–81.1 | 81.13 | at the top |
| abstention unchanged | strict +0.33, official ±0 | held |
| the +3.0 strict bar fails | +2.5 | held |

- The falsifier "strict below +0.5" did not fire.
- The control held: the 5 rows outside U reproduced M57's evidence byte for
  byte.

**Incidents, and why the numbers still stand:**
- **12:11:** home-still's shared Qdrant container restarted and killed seed
  1 at row 57. It was resumed with `bench --resume`.
- **15:00:** the driver hung in a bare `wait`, which also waits on its own
  lease-renewal loop. A continuation unit ran the remaining steps under the
  held lease.
- **16:07:** a second Qdrant restart, from home-still's rc.358 deploy,
  killed seed 3 at row 33. Recovering it also stopped base seed 2's lane.
  Both were resumed from their saved rows.
- A resumed row is read exactly as an uninterrupted one would be: same
  flags, same seed, same serving. Every rerun row exists exactly once, and
  `merge_close` asserts the 183/154 rerun rows and 500 total per run.
- Since then, reads retry through a transport failure (#168), and the
  drivers wait on named PIDs or units.

## Shipped *(user's decision, 2026-09-29)*

The user chose to ship the bundle on the official gate, even though strict
missed the +3.0 bar.
- `bench::shipped_{events_ledger, aggregation, advice_profile_clause,
  commit_grounded}` make its four mechanisms LongMemEval_S's shipped
  configuration. `standing` marks their absence as an arm, and M79's typed
  pass as one.
- **`standing` quotes seed means.** `runs/r5_bundle_seeds/replicates.json`
  names the three grounded replicates, and a replicate set outranks any
  single draw of its configuration. Otherwise `standing` would quote the
  best seed (official 81.6), which is a seed's luck.
- **Standing:**
  - `longmemeval_s.judge_score_matched.n500` is **81.13**, comparable,
    +0.33 against MemPro-15 (Qwen): **the gate closes**;
  - `judge_score.n500` is 80.80, caveat-judge.
- **The ratchet** pins the seed means. `token_f1` is lowered by hand from
  M57's single-seed 60.82 to 60.48. The base's own seed mean is 60.76, so
  the bundle costs −0.28 token F1: personalized advice answers are longer,
  and word overlap with rubric-shaped references falls while every judged
  reading rises. This is a known trade, recorded rather than hidden.
- `ratchet --strict` passes.
