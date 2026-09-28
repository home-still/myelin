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
