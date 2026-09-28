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
