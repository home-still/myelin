# M76b — advice requests without the premise clause *(pre-registered 2026-09-28, before any row)*

## Why

M76 put the user's own sentence into the evidence for 28 of 30 preference
questions (+50.0 [+33.3, +66.7], reader-free). Its answers did not move
(official +0.0), because the reader declined more: 11 of 30 declines, 10
with every gold turn in hand (`m76-user-words.md`).

The shipped LongMemEval_S reader carries M57's premise clause: "If the
question assumes something the memories do not state … begin your reply with
'I don't know.'" An advice request ("Can you suggest a hotel for my trip to
Miami?") assumes nothing to recall, but the reader reads it as a recall
question with no recorded answer. This is PrefEval's *unhelpful* failure,
"refusing to answer queries due to a perceived lack of context" (Zhao et al.
2025, `10.48550/arxiv.2502.09597`).

**The user decided (2026-09-28)** to gate the existing clause off for advice
requests. No clause is added; one stops applying to one question shape.

## Mechanism

`--advice-without-premise`: when `query_shape::is_advice_request` fires, the
reader's system prompt omits `READER_PREMISE_CLAUSE`. Every other question
keeps it.
- The gate is fixed since 2026-09-26: 29 of the 30 preference questions and
  none of the other 470.
- No abstention row can change, by construction.

## Arms (the 29 fired rows, other 471 copied from M57, on M57's serving)

1. **M76b, the gated arm:** `--user-words --advice-without-premise`, giving
   `runs/m76b_words_nopremise_s1`.
2. **M76c, diagnostic only:** `--advice-without-premise` alone, giving
   `runs/m76c_nopremise_s1`. It separates what dropping the clause does
   without the user's words. It cannot enter a bundle by itself unless it
   clears the same gate.

Both arms run side by side on big's two slots, as M57's two shards did. Both
are judged by the strict 9B (seeded from M57) and LongMemEval's official
grader (seeded from M57's official verdicts).

## Gate (the user's stratum rule, as for M76)

On the 30 preference rows:
1. the strict paired difference against M57 has a 95% CI excluding zero;
2. the official difference is positive.

## Predictions

- Declines on the 30 fall from 11 (M76) and 7 (M57) to about 1–3.
- M76b: official preference 13 → 17–21. M76c: smaller, about 13 → 14–16,
  since without the user's words most declines turn into generic advice.
- The strict judge moves less than the official one. Its rubric is literal,
  and generic advice fails both.

## Falsifiers

- Declines fall but answers do not improve: the reader answers generically
  and ignores the `[your words]` block. Report the rows where the block holds
  the gold sentence and the answer does not use it.
- M76c alone matches M76b: the words were never the lever; the clause was.

## Result — both arms fail; the premise-clause diagnosis was wrong *(measured 2026-09-28)*

The runs are `runs/m76b_words_nopremise_s1` and `runs/m76c_nopremise_s1`,
both at main `3be60dd`, on big. Each reruns the 29 fired rows; the other 471
rows are M57's.

| arm | reading | n | arm | M57 | Δ [95% CI] |
|---|---|---|---|---|---|
| M76b (words, no clause) | strict 9B | 30 | 33.3% | 36.7% | −3.3 [−16.7, +10.0] |
| M76b | official | 30 | 26.7% | 43.3% | **−16.7 [−30.0, −3.3]** |
| M76c (no clause) | strict 9B | 30 | 33.3% | 36.7% | −3.3 [−16.7, +10.0] |
| M76c | official | 30 | 30.0% | 43.3% | **−13.3 [−26.7, −3.3]** |

- Official: M76b fixed 0 rows and broke 5; M76c fixed 0 and broke 4.
- Abstention is unchanged (30/30 rows identical), as the construction
  guarantees.
- **Verdict:** both gate criteria fail for both arms. The switch stays off
  and enters no bundle.

**Declines rose instead of falling.** The prediction was about 1–3.

| run (the 30 preference rows) | declines | … citing a false premise | … saying the memories lack the answer | answered | answered and right (official) |
|---|---|---|---|---|---|
| M57 (shipped) | 7 | 0 | 5 | 23 | 13 |
| round-4 bundle, a pure rerun here¹ | 10 | — | — | 20 | 13 |
| M76 (words) | 11 | — | — | 19 | 13 |
| **M76b** (words, no clause) | **12** | 0 | 10 | 18 | 8 |
| **M76c** (no clause) | **13** | 0 | 11 | 17 | 9 |

¹ No round-4 mechanism fires on a preference row, so `r4_bundle_s1` is an
unchanged rerun of them: 19 of 30 have evidence identical to M57's and 12
identical answers.

- **The M76 diagnosis is falsified.** `m76-user-words.md` blamed M57's
  premise clause for the declines. None of the reader's reasoning traces on
  a declined preference row mentions an assumption or a premise, in M57 or
  in either arm. The reader cites the base prompt's own line, "If the
  memories do not contain the answer, reply exactly: I don't know." For
  example, on "What should I serve for dinner this weekend with my homegrown
  ingredients?": "there's no specific dinner recommendation in the memories
  for this weekend", so "I don't know."
- **Seven declines were a low draw.** An unchanged rerun gives 10. M76's 11
  sits inside the reader's own rerun spread; the words did not cause them.
- **Dropping the clause cost answers.** The answered rows fell from 23 to 17
  or 18, and fewer of those were right. The clause's decline-then-state
  shape seems to steer the reader toward naming what the memories do hold.
  With 30 rows and one seed this is **measured, not explained**.
- The second falsifier ("M76c matches M76b") fires in its weak form: the two
  arms are within one row of each other, and neither beats M57.

**What is left for preference.** Preference is 30 of 500 rows. MemPro-15 has
24 of them right and M57 has 13, so it holds the whole official gap
(11 rows, 2.2 points).
- The retrieval half is solved: M76 holds every gold turn on 28 of 30.
- The block that remains is `READER_SYSTEM`'s "reply exactly: I don't know",
  which the reader applies to requests that have no recorded answer.
- `READER_PREFERENCE_CLAUSE` (M20 arm B, `--profile-clause`) already counters
  that line ("Do not reply I don't know when the memories state a relevant
  preference"). It is a reader-prompt change, so under the code-first rule
  (2026-09-24) it waits for the user's decision.
- Any further preference arm needs more than one seed. An unchanged rerun
  moves declines on these 30 rows by 3.
