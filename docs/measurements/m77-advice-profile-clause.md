# M77 — the preference clause for advice requests, with the user's words, over three seeds *(pre-registered 2026-09-28, before any row)*

## Why

The 30 single-session-preference questions hold LongMemEval_S's whole
official gap to MemPro-15: 13 right against its 24, a difference of 11
questions, or 2.2 points.

| what is known | measured by |
|---|---|
| The user's own sentence reaches the evidence on 28 of 30 (from 13). | M76, stage 0: +50.0 [+33.3, +66.7] |
| The reader declines anyway, and the premise clause is not why. | M76b: no declined row's reasoning mentions a premise |
| The reader cites `READER_SYSTEM`'s own line: "If the memories do not contain the answer, reply exactly: I don't know." | M76b, 10 of 12 declines |
| An unchanged reader scores 13, 10 and 9 official under seeds 1–3 (strict 11, 9, 13; declines 7, 10, 8). | `m57_pref_s2`, `m57_pref_s3` (this doc) |

`READER_PREFERENCE_CLAUSE` (M20 arm B) already answers that line: "answer
with the preferences the user themselves stated in the memories … Do not
reply I don't know when the memories state a relevant preference." M20
measured it on the 9B without the user's words: declines fell 11 → 7, and
one more question was right, with nothing for the clause to point at.

**The user decided (2026-09-28)** to test the existing clause, gated to
advice requests, with M76's words, over three seeds. No new text.

## Mechanism

`--advice-profile-clause`: when `query_shape::is_advice_request` fires, the
reader's system prompt carries `READER_PREFERENCE_CLAUSE` ahead of M57's
premise clause.
- It fires on 29 of the 30 preference questions and on none of the other
  470, so no other row can change.
- `--user-words` (M76) appends the `[your words]` block to the same 29.
- Grounds: PrefEval (Zhao et al. 2025, `10.48550/arxiv.2502.09597`) finds
  that a reminder of the stated preference, with the preference retrieved,
  turns generic answers into preference-following ones.

## Seeds, and why three

A single rerun of these 29 rows moves the official count by up to 4 and the
declines by 3 (the table above). M76b's "−16.7 [−30.0, −3.3]" against seed 1
alone becomes −8.9 [−20.0, +1.1] against the three-seed base: the base's
seed 1 was its best draw.
- Each arm runs under reader seeds 1, 2 and 3 (`--reader-seed`).
- `paired_ci.py` averages each question over its seed replicates (runs
  joined by `,`) and then bootstraps over questions. That is Miller (2024,
  arXiv 2411.00640) and Bouthillier et al. (2021, arXiv 2103.03098): average
  out within-question noise, and keep the question as the sampling unit.
- The base is `runs/m57_bonsai_premise_s1,runs/m57_pref_s2,runs/m57_pref_s3`.

## Arms (the 29 fired rows; the other 471 rows copied from M57, on M57's serving)

1. **M77, the gated arm:** `--user-words --advice-profile-clause`, giving
   `runs/m77_words_clause_s{1,2,3}`.
2. **M77c, diagnostic only:** `--advice-profile-clause` alone, giving
   `runs/m77c_clause_s{1,2,3}`. It separates the clause from the words. It
   enters no bundle unless it clears the same gate itself.

Both are judged by the strict 9B (seeded from M57) and by LongMemEval's
official grader (seeded from M57's official verdicts).

## Gate (the user's stratum rule; the statistic is seed-averaged)

On the 30 preference rows, against the three-seed base:
1. the strict paired difference has a 95% CI excluding zero, *and*
2. the official difference is positive.

Passing admits M77 to a bundle; it does not ship it. The bundle rule still
holds: one full-500 arm at +3.0 strict with a CI excluding zero, abstention
at 29 of 30 or better, and the lead under the official grader too.
Preference alone is worth at most 2.2 points, so a bundle needs round 4's
positive strata as well.

## Predictions

- Declines on the 30 fall from a seed mean of 8.3 to about 2–4 per seed.
- M77: official 10.7 → 15–19 (seed mean), strict 11.0 → 14–17.
- M77c: smaller, official about 12–14. Without the words, the reader holds
  the preference on only 13 of 30 rows.

## Falsifiers

- Declines fall but answers do not improve: the reader answers without the
  `[your words]` block. Report the rows where the block holds the gold
  sentence and the answer does not use it.
- M77c matches M77: the words are not what the clause points at.
- Declines do not fall: the clause is as inert on Bonsai as M38's rewritten
  selector clause was ("the reader ignores instructions and obeys
  structure"). The next step would then be structure, not text.

## Result — official +18.9, strict CI crosses zero; the gate fails *(measured 2026-09-28)*

The runs are `runs/m77_words_clause_s{1,2,3}` and `runs/m77c_clause_s{1,2,3}`.
They ran on big at main `33a4041` as unit `myelin-m77`: the 29 fired rows
under each seed, the other 471 copied from M57. They were judged by the strict
9B (seeded from M57) and by LongMemEval's official grader (gpt-4o-mini, on the
workstation).

**Per seed, on the 30 preference rows** (declines / official right / strict
right):

| arm | seed 1 | seed 2 | seed 3 | seed mean |
|---|---|---|---|---|
| base (M57) | 7 / 13 / 11 | 10 / 10 / 9 | 8 / 9 / 13 | 8.3 / 10.7 / 11.0 |
| **M77** (words + clause) | 8 / 17 / 13 | 6 / 16 / 13 | 5 / 16 / 14 | 6.3 / **16.3** / 13.3 |
| M77c (clause alone) | 7 / 17 / 13 | 6 / 18 / 13 | 8 / 14 / 12 | 7.0 / **16.3** / 12.7 |

**Paired and seed-averaged** (`paired_ci.py`, three replicates a side, with
the bootstrap over questions):

| arm | reading | stratum | n | arm | base | Δ [95% CI] | p |
|---|---|---|---|---|---|---|---|
| M77 | strict 9B | preference | 30 | 44.4% | 36.7% | +7.8 [−5.6, +22.2] | 0.22 |
| M77 | official | preference | 30 | 54.4% | 35.6% | **+18.9 [+8.9, +30.0]** | < 0.0001 |
| M77 | strict 9B | all | 500 | 79.7 | 79.2 | +0.5 [−0.3, +1.3] | 0.23 |
| M77 | official | all | 500 | 79.3 | 78.1 | **+1.1 [+0.5, +1.9]** | 0.0001 |
| M77c | strict 9B | preference | 30 | 42.2% | 36.7% | +5.6 [−5.6, +17.8] | 0.31 |
| M77c | official | preference | 30 | 54.4% | 35.6% | **+18.9 [+8.9, +30.0]** | < 0.0001 |

- Abstention is unchanged under both readings, since the gate fires on no
  abstention row.
- **Verdict: criterion 1 fails** (the strict CI includes zero), and
  criterion 2 holds by a wide margin. As pre-registered, M77 does not enter
  a bundle.
- Note: the three-seed base reads **78.1** official overall, below seed 1's
  78.60. Seed 1 was the base's best draw on preference, as the correction in
  `m76b-advice-without-premise.md` already found.

**Predictions against the result:**

| prediction | result | |
|---|---|---|
| declines 8.3 → 2–4 per seed | 6.3 | wrong |
| M77 official 10.7 → 15–19 | 16.3 | held |
| M77 strict 11.0 → 14–17 | 13.3 | just under |
| M77c smaller, official 12–14 | 16.3, the same as M77 | wrong |

**Falsifiers:**
- **"M77c matches M77" fires.** The clause alone gives the whole official
  gain, so the user's words add nothing measurable once the clause is on.
- **"Declines do not fall" half-fires.** Declines fell by 2 per seed, not 5.
  The clause is not inert, though. It works on the *content* of the answers,
  turning generic advice into advice built on the user's stated
  preferences, not on whether the reader answers at all. That is the first
  instruction this reader has measurably followed since M57's clause, and it
  qualifies the one law: text can move what an answer says, but not whether
  the reader declines.

## What the gate measured: the strict judge cannot grade a preference row

The two readings disagree on 21 of the 90 seed-rows of M77. On 15, official
credits the answer and strict does not; on 6 it is the other way; 34 are
right under both and 35 under neither.
- The strict judge (`judge.rs`, `JUDGE_SYSTEM`) asks one question for every
  row: does the answer "convey the same fact as the reference answer".
- A preference row has no fact. Its reference is a rubric, "The user would
  prefer responses that …".
- LongMemEval's own grader gives this type its own template: correct if the
  response "recalls and utilizes the user's personal information correctly".
- Examples the official grader credits and the strict judge rejects, all
  from seed 1:
  - Music store: "Based on your Strat-to-Les Paul upgrade: body shape, neck,
    pickups …" (rubric: Stratocaster vs Les Paul differences).
  - Meal prep: "chicken fajitas with quinoa and roasted vegetables, lentil
    bolognese over quinoa …" (rubric: quinoa and roasted vegetables).
  - Denver: "Since you love the music scene, hit up Red Rocks …" (rubric:
    their interest in live music).

This is a finding about the instrument, not a change to the verdict. The
strict judge decides every arm under the user's rule, and on this stratum it
grades a rubric as though it were a fact. Whether preference rows should be
graded by a rubric-shaped strict template is the user's call, and it is
recorded in BACKLOG_DONE.

## Re-read on the corrected strict judge *(post hoc, 2026-09-28)*

The user gave the strict judge a rubric for preference rows
(`judge-preference-rubric.md`). Re-judged, on the 30 preference rows:
- M77: strict **+10.0 [+0.0, +21.1]**, p 0.051. It still touches zero.
- **M77c (the clause alone): strict +11.1 [+1.1, +22.2]**, p 0.020, and
  official +18.9. It clears both criteria.
- **This is post hoc:** the instrument changed after the result was seen.
  M77c goes forward as a bundle candidate, and the seed-replicated bundle is
  the confirmatory test. The clause alone is also the simpler mechanism,
  since the words add nothing measurable.
