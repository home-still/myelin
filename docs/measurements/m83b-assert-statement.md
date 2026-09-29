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
