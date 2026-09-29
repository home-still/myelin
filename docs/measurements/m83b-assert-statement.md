# M83b — the typed statement must assert the question, not dispute it *(user decision 2026-09-29: "run M83b, then ask to ship"; pre-registered before any row)*

## Why

M83's speaker contrast took M82's LoCoMo person swaps from 34 to 4 across
three runs. It still failed one held-out run, `m51`, with 4 flips against a
limit of 3 (`m83-speaker-contrast.md`).

All four misses are statements that **dispute** the premise instead of
asserting it:
- "Nate did **not** take a picture of a sunflower…; **Joanna** did."
- "Witcher 3 inspired James to create his game, **not** a painting."
- "Sam did **not** share a photo of a kayak…; **Evan** took that photo."
- "Joanna mentioned that **Nate** has been playing… Cyberpunk 2077."

The NLI model rightly entails a correction, and the pass then committed the
answer.

## Mechanism

`commit-arm --typed-nli --speaker-contrast --assert-statement`
(`bench::{statement_disputes, negation_count}`). Before the NLI model is
asked, code keeps the decline when the statement:
1. **holds more negating words than the question** ("not", "never", "no",
   any "n't"), or
2. **names a dialogue speaker the question does not name.**

Everything else is M83 unchanged. The trace records `assert=rejected: <why>`.

Grounds: presupposition verification checks the question's own claim in
declarative form (Kim et al. 2021, arXiv 2101.00391). A statement that
negates it, or moves it to another person, is not that claim.

**Honest caveats:**
- **The rule was found on `m51`.** Post hoc there it blocks all 4 flips, and
  on M83's three runs it blocks none of the 7 right answers. `m51` is now
  reported, not gated.
- **LongMemEval_S:** the rule blocks none of M82's recorded commits (checked
  on every statement in `m82_base_s*` and `r5_bundle_s*_m82`). LongMemEval_S
  has no dialogue speakers, and no committed statement added a negation.
  So M82's LongMemEval_S result stands as M83b's, and LongMemEval_S is not
  rerun.

## Measurement (post-passes on big, the 9B reader)

| source | out | role | declines | adversarial declines |
|---|---|---|---|---|
| `m55b_locomo_bonsai` | `m83b_locomo_m55b` | **held out** (Bonsai's first pass, recall) | 706 | 414 |
| `m50b_locomo_events` | `m83b_locomo_m50b` | **held out** | 433 | 317 |
| `m63_locomo_base` | `m83b_locomo_base` | the shipped LoCoMo point, reported | 419 | 303 |
| `m51_locomo_s1` | `m83b_locomo_m51` | the run the rule was found on, reported | 572 | 370 |

Grader: the strict 9B judge, seeded from each base. The LightMem grader is
run on `m83b_locomo_base`.

## Gate

1. **Held-out flips at most 1% of each run's adversarial declines (rounded
   down):** `m83b_locomo_m55b` ≤ 4 of 414, `m83b_locomo_m50b` ≤ 3 of 317.
2. **No LoCoMo loss:** strict Δ ≥ 0 on each held-out run, over all 1,986
   rescored rows, adversarial included.

**If both hold, the ship decision goes to the user** (their instruction),
with M82 plus M83b's numbers:
- **LongMemEval_S stack:** official 81.40 and strict 81.20, against the
  shipped 81.13 and 80.80. M82's stratum CI touched 0.
- **LoCoMo:** the Δ on `m83b_locomo_base` under both graders.

## Predictions

- Held out: 0–2 flips per run, and 3–10 answerable commits per run.
- Strict Δ between 0 and +0.3 per held-out run.
- `m83b_locomo_m51`: at most 1 flip.

## Falsifiers

- **Held-out flips over the limit:** traps pass statements that assert the
  question, so the NLI model is entailing a claim the memories do not make.
  Report the statements.
- **Held-out Δ < 0:** the right commits are fewer than the flips.

## Result — passes: 0 flips on both held-out runs, and on every run *(measured 2026-09-29, 12:43–13:47, main db4f504)*

| run | role | adversarial declines | **flips** (limit) | answerable commits (strict right) | statements rejected by the rule | strict Δ, all 1,986 rows |
|---|---|---|---|---|---|---|
| `m83b_locomo_m55b` | held out | 414 | **0** (≤ 4) ✅ | 24 (9) | 9 | **+0.5 [+0.2, +0.8]** (9/0) ✅ |
| `m83b_locomo_m50b` | held out | 317 | **0** (≤ 3) ✅ | 3 (0) | 3 | +0.0 (0/0) ✅ |
| `m83b_locomo_base` | the shipped point, reported | 303 | 0 | 3 (1) | 1 | +0.1 [+0.0, +0.2] (1/0) |
| `m83b_locomo_m51` | where the rule was found, reported | 370 | 0 (M83: 4) | 8 (3) | 6 | +0.2 [+0.0, +0.4] (3/0) |

On the shipped point, LightMem's grader gives +0.1 [+0.0, +0.3] (2/0).

**Verdict:** both gate criteria hold on both held-out runs. **M82 plus M83b
now clears its LoCoMo criterion.** As pre-registered, whether it ships goes
to the user.

**The M82 line on LoCoMo, start to finish:**

| mechanism | flips, M82's design run | the worst held-out run |
|---|---|---|
| M82 (NLI alone) | 13 | 17 (`m51`) |
| M83 (+ speaker contrast) | 0 | 4 (`m51`) |
| **M83b (+ the statement must assert)** | **0** | **0** |

**Predictions:**
- held out, 0–2 flips per run: **0 and 0, held**;
- held out, 3–10 answerable commits: 24 and 3, **wrong** (the Bonsai-read
  base declines far more);
- strict Δ 0 to +0.3: +0.5 and +0.0 (the first is above the range);
- `m51` at most 1 flip: **0, held**.

**What shipping would carry, for the user's decision:**
- **LongMemEval_S stack** (M82's rows; M83b blocks none of them): official
  81.40 and strict 81.20, against the shipped 81.13 and 80.80. Abstention
  is unchanged.
  - M82's stratum CI on the 49 answerable declines, +6.1 [+0.0, +14.3],
    touched 0.
- **LoCoMo:** +0.1 on `m63_locomo_base`. LoCoMo now ships with M84's
  non-recall pass, so M83b there would need one stacked pass over
  `m84_locomo_base`.
- **A new service in the shipped path:** the NLI cross-encoder (DeBERTa-v3
  large, ~1.7 GB of VRAM) beside the reader.

## Shipped for LongMemEval_S *(user decision 2026-09-29: "ship on LongMemEval_S")*

**The ship run** (main 3f9acd7, 14:07–14:38): the full pass with the exact
shipped flags over the round-5 bundle's three replicates →
`runs/r5_nli_s{1,2,3}`, quoted as the replicate set `runs/r5_nli_seeds`.

| | seed 1 | seed 2 | seed 3 | **seed mean** | round-5 bundle |
|---|---|---|---|---|---|
| commits (answerable) | 2 | 2 | 2 | | |
| false fits | 0 | 0 | 0 | | |
| official | | | | **81.53** | 81.13 |
| strict | | | | **81.20** | 80.80 |
| abstention (official / strict, of 30) | | | | 28.00 / 29.33 | 28.00 / 29.33 |

- **Paired against the bundle:** +0.4 [+0.0, +1.0] under both graders (2/0
  per seed), and abstention +0.0.
- **It reproduces M82's recorded stack row for row.** The same two questions
  are committed on every seed.
- **Official reads 81.53 here, where M82's stack read 81.40.** The ship run's
  untouched rows keep the bundle's own official verdicts. M82's readout had
  seeded them from another run, and two untouched rows on seed 2 were graded
  differently.
- **Standing:**
  - `longmemeval_s.judge_score_matched.n500` reads **81.53**, +0.73 over
    MemPro-15 (Qwen3-30B), `comparable`, gate closed;
  - strict reads 81.20, +0.40, `caveat-judge`;
  - `ratchet --strict` holds.

**The shipped LongMemEval_S recipe, end to end:**
1. `myelin-eval bench --corpus longmemeval-s --mode investigate --k 6
   --budget-tokens 4096 --max-steps 2 --select-sufficient --item-digest
   --digest-dates --reader-thinking --reader-premise-clause --events-ledger
   data/longmemeval_s_events.ledger --aggregation-k 18
   --aggregation-budget-tokens 8192 --advice-profile-clause`. These are the
   round-5 flags (`r5-bundle-seeds.md`).
2. `myelin-eval commit-arm --run <bench> --grounded --out <grounded>` (M71b).
3. `bash ops/big/serve-nli.sh`, then `myelin-eval commit-arm --run
   <grounded> --typed-nli --speaker-contrast --assert-statement --nli-url
   http://127.0.0.1:5820 --out <shipped>`.

**On the record:**
- It shipped on the user's decision, not on M82's pre-registered rule. That
  rule's stratum criterion (+6.1 [+0.0, +14.3]) touched 0.
- The gain is two questions per seed.
