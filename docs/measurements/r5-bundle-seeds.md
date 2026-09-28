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
