# The strict judge's preference rubric *(user decision 2026-09-28; predictions written before any re-judged verdict)*

## Why

The strict 9B judge (`src/judge.rs`) decides every arm. Until today it asked
one question of every row: does the model answer "convey the same fact as the
reference answer"?

A LongMemEval_S `single-session-preference` row has no fact. Its reference is
a rubric: "The user would prefer responses that highlight the differences
between Fender Stratocaster and Gibson Les Paul …". M77 measured what grading
such a rubric as a fact costs
(`m77-advice-profile-clause.md`, "What the gate measured"):
- On M77's 90 seed-rows the strict judge and LongMemEval's own grader
  disagreed on 21, 15 to 6.
- The 15 are answers built on the user's stated preferences, such as "Based
  on your Strat-to-Les Paul upgrade: body shape, neck, pickups …".

The user chose to give the strict judge a rubric for this row type
(2026-09-28), over letting the official grader decide the stratum or leaving
the judge as it was.

## The change

- **`Rubric::Preference`** applies to LongMemEval_S rows of type
  single-session-preference, and to nothing else. LoCoMo's own category 3
  keeps the fact rubric, because the corpus is checked by tenant first.
- **The criterion is LongMemEval's own preference template:** "The model does
  not need to reflect all the points in the rubric. The response is correct
  as long as it recalls and utilizes the user's personal information
  correctly." (Wu et al. 2024, arXiv 2410.10813, `evaluate_qa.py`,
  `get_anscheck_prompt`.) It is written in the strict grader's voice and
  reply format, and it adds one sentence the template implies: a generic
  answer that uses none of the user's information is incorrect.
- **Every other row's request is byte for byte what it was**, which a unit
  test pins.
- **Each verdict records its rubric** (`JudgeFile.rubrics`). A cached or
  seeded verdict is reused only under the rubric it was graded by, so any
  preference row judged as a fact is judged again. Files written before this
  change load as fact-graded.

## What gets re-judged

Each run is re-judged in place with the 9B on big, from its own cache, so
only the 30 preference rows are graded afresh. Then each is rescored with
`--scorer judge`. The runs:
- the base: `m57_bonsai_premise_s1`, `m57_pref_s2`, `m57_pref_s3`;
- M77: `m77_words_clause_s{1,2,3}`;
- M77c: `m77c_clause_s{1,2,3}`.

M57's re-judged strict number replaces 79.20 as the LongMemEval_S strict
headline, whichever way it moves, and is recorded with its commit.

## Predictions (before any verdict)

1. **Agreement with the official grader** on M77's 90 seed-rows rises from
   69/90 to at least 80/90.
2. **M77's strict difference on preference moves toward its official one**
   (+18.9): at least +12, with the CI excluding zero.
3. **The base's strict preference score** moves from 11.0 (seed mean) to
   within ±2 of its official 10.7.
4. **M57's strict headline moves by at most ±0.6** (±3 rows), since only 30
   rows can move.

**Re-reading M77's gate is post hoc.** Its instrument was changed after its
result was seen, and every place M77's re-read gate is quoted says so. The
change itself is not tuned to M77: the criterion is LongMemEval's, fixed
before this run.

**Falsifier.** If agreement does not rise, the rubric is not what the judges
disagreed about, and the PR that records the result also reverts the change.

## Result *(measured 2026-09-28, main `9e4ff83`)*

Nine runs were re-judged on big (unit `myelin-rubric`) from their own caches.
Only the preference rows were graded again, then each run was rescored.

**Against the predictions:**

| prediction | result | |
|---|---|---|
| 1. agreement with the official grader on M77's 90 seed-rows 69 → ≥ 80 | **74** | missed |
| 2. M77 strict on preference ≥ +12, CI excluding 0 | +10.0 [+0.0, +21.1] | missed |
| 3. base strict preference within ±2 of official 10.7 | 8.7 (10 / 8 / 8) | held |
| 4. M57's strict headline moves ≤ ±0.6 | 79.20 → **79.00** | held |

- **What moved:** strict-right/official-wrong fell from 6 to 1 (M77) and
  from about 4 to 1 (base). Official-right/strict-wrong stayed at 15. The
  rubric removed the strict judge's false credits, and it also judges some
  personalized answers more harshly than gpt-4o-mini does. For example, it
  rejects "your lemon lavender pound cake" against a rubric naming the user's
  lemon poppyseed cake. It is stricter than the official grader, which is
  what a strict judge is for.
- **Falsifier:** agreement rose (69 → 74), so it does not fire, and the change
  stays. Prediction 1's size was wrong.
- **The headline moves:** M57's strict LongMemEval_S reading is **79.00**
  (was 79.20; the preference stratum is now 10/30, was 11). The pin in
  `docs/sota/progression.json` is lowered by hand as an instrument
  correction, following the 2026-09-25 precedent. `ratchet --strict` then
  passes. The official reading (78.60) does not move.

**M77's gate, re-read on the corrected instrument (post hoc).** Seed-averaged,
on the 30 preference rows, against the three-seed base:

| arm | strict Δ [95% CI] | p | official Δ |
|---|---|---|---|
| M77 (words + clause) | +10.0 [+0.0, +21.1] | 0.051 | +18.9 [+8.9, +30.0] |
| **M77c (clause alone)** | **+11.1 [+1.1, +22.2]** | 0.020 | +18.9 [+8.9, +30.0] |

- M77 still touches zero, while M77c clears both criteria.
- Overall strict: M77c +0.7 [+0.1, +1.4]; abstention unchanged.
- **This pass is post hoc,** because the instrument changed after M77's
  result was seen. M77c therefore enters the bundle as a *candidate*. The
  bundle is the confirmatory test: seed-replicated and pre-registered.
