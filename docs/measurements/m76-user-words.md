# M76 — the user's own words, from the sessions the evidence reached *(pre-registered 2026-09-28, before any row)*

## Why

LongMemEval_S's gap to MemPro-15 (80.80) sits in the single-session strata.
Preference is the largest part: 13/30 right under LongMemEval's own grader
against MemPro's 24/30. Round 4's attempt (M20b: ranked extracted profile
records) narrowed advice to one remembered fact and failed its gate.

**Reader-free anatomy on M57's shipped run** (2026-09-27, the 30 preference
questions, `data/longmemeval_s.json` gold turns against M57's evidence):

| | questions |
|---|---|
| the session holding the preference is in evidence | **29 / 30** |
| the gold turn itself is in evidence | 13 / 30 |
| every gold turn is a *user* turn | 30 / 30 |
| all user turns of the evidence's sessions would hold every gold turn | 29 / 30 |

- The evidence reaches the right conversation. What reaches the reader,
  though, is 512-token episodes dominated by the assistant's long replies,
  and the user's short sentence ("turbinado adds a richer flavor") ranks
  below them.
- Evidence spans a median of 2 sessions (at most 4). All of their user turns
  cost a median of about 700 tokens (at most about 2,400), against about
  2,700 of evidence.
- A plain lexical ranking of those user turns already puts the gold turn
  1st in 7 of 30, in the top 3 in 19, and in the top 10 in 28.

## Mechanism

`myelin_core::pipeline::user_words` (`--user-words`). It is gated by
`query_shape::is_advice_request`: fixed on 2026-09-26, it fires on 29 of the
30 preference questions and on none of the other 470.

1. Take the sessions of the evidence's episodes (source doc
   `<session>#<segment>`).
2. Read those sessions' other segments from the run's own ledger. Records
   already in the evidence are skipped.
3. Keep only the `user:` turns, and rank them against the question with the
   cross-encoder.
4. Take them best-first while the rendered block fits 1,024 tokens
   (`USER_WORDS_BUDGET_TOKENS`). The top turn is always taken.
5. Append them after the evidence under a `[your words]` header, each as a
   windowed view of its stored episode, so every line quotes its record.

- The base evidence is never touched. A shut gate appends nothing, so 471
  rows are byte-identical by construction.
- It differs from M20b in kind: M20b gave the reader extracted dispositions;
  this gives it the user's own sentences, in context.
- Grounds:
  - PrefEval (Zhao et al. 2025, `10.48550/arxiv.2502.09597`): retrieving
    the stated preference is the best remedy for preference-unaware answers.
  - CueMem (Wang et al. 2026, arXiv 2609.12354): rebuilding context from
    source turns makes it best on single-session-preference.
  - Both were found through home-still in round 4
    (`docs/research/sota-catalog-2026-09-26.md`).

## Stage 0 — retrieval, reader-free (gate for stage 1)

- **Run:** `bench <M57 flags> --user-words --evidence-only --questions
  m20b_pref_ids` (the 30 preference questions), then `coverage`.
- **Pair:** `paired_ci.py runs/m76_words_evidence runs/m57_bonsai_premise_s1
  --gold-held --ids m20b_pref_ids`.
- **Criterion:** every-gold-turn-held on the 30 rises, with a paired 95% CI
  excluding zero.
- **Prediction:** 13 → 24–28.
- **Diagnostics:** tokens added per row, and turns appended per row.

## Stage 1 — the stratum arm (only if stage 0 passes)

- **Run:** the 29 fired rows rerun with `--user-words` on M57's serving. The
  other 471 are copied from `runs/m57_bonsai_premise_s1`.
- **Judges:** the strict 9B (seeded from M57) and LongMemEval's official
  grader (seeded from M57's official verdicts).
- **Stratum gate** (the user's rule): the switch enters a bundle only if
  1. on the 30 preference rows, the strict paired difference has a 95% CI
     excluding zero, *and*
  2. the official difference on the same rows is positive.
- Abstention cannot move (the gate fires on no abstention row), and this is
  checked.

**Predictions.**
- Stage 0 passes.
- Stage 1: preference rises by 3–6 questions under the official grader
  (13 → 16–19). Some of the 6 gold-in-evidence losses stay generic, since
  holding the sentence is not using it.

**Falsifiers.**
- Stage 0 flat: the cross-encoder ranks the preference turn below the budget
  line. Report the gold turn's rank.
- Stage 0 up and stage 1 flat or down: the reader ignores the block, or
  narrows on it as it did on M20b's. Report fixed and broken rows with the
  block's content.

## Stage 0 result — passes *(measured 2026-09-28)*

The run is `runs/m76_words_evidence`: `--user-words --evidence-only` on the
30 preference questions, on big, at main `ee44c6d`. It was paired with M57
on every-gold-turn-held (`paired_ci.py --gold-held --ids m20b_pref_ids`):

| | n | M76 | M57 | Δ [95% CI] | p |
|---|---|---|---|---|---|
| **all gold turns held** | 30 | **93.3%** (28) | 43.3% (13) | **+50.0 [+33.3, +66.7]** | < 0.0001 |

- Coverage: 28 hold all their gold turns, 1 part, 1 none. Mean gold recall
  is 0.950.
- **Diagnostics:**
  - The block was appended on 28 of 30 questions.
  - A median of 10.5 user turns per question (at most 20), about 612 tokens
    (at most about 1,076).
  - The selector declined on 10 rows, against M57's 9. It agrees with M57 on
    29 of 30. The block is appended after selection, so it cannot move the
    selector; the one difference is the LLM selector's own rerun noise
    (`r4-bundle.md`).
- The prediction (13 → 24–28) held at its top.
- The criterion holds, so stage 1 runs, as pre-registered.

## Stage 1 result — fails; the second falsifier fires *(measured 2026-09-28)*

The run is `runs/m76_words_s1`: the 29 fired rows rerun with `--user-words`
on M57's serving, with the other 471 copied from M57.

| reading | stratum | n | M76 | M57 | Δ [95% CI] |
|---|---|---|---|---|---|
| strict 9B | preference | 30 | 33.3% | 36.7% | −3.3 [−16.7, +10.0] |
| official | preference | 30 | 43.3% | 43.3% | +0.0 [−13.3, +13.3] |

- Official: 2 fixed and 2 broken. Strict: 2 fixed and 3 broken.
- Abstention is unchanged (the gate fires on no abstention row), and the
  other 471 rows are M57's.
- **Verdict:** both criteria fail. M76 does not enter a bundle and stays
  off.

**Why: the reader declines advice it now has the grounds for.**

| | declines on the 30 | … with every gold turn held | official right |
|---|---|---|---|
| M57 | 7 | — | 13 |
| M76 | **11** | **10** | 13 |

- With the user's own sentences in hand, the reader answered "I don't know."
  more often, not less. Examples:
  - "Can you suggest a hotel for my upcoming trip to Miami?"
  - "Can you recommend a show or movie for me to watch tonight?": M57 named
    a documentary; M76 declined.
  - "…recent publications or conferences?": "I don't know. The memories
    only note that recommendations were requested and made on 2023-05-22…"
- This is PrefEval's *unhelpful* failure: "refusing to answer queries due to
  a perceived lack of context".
- The cause is the reader prompt, not the memory. M57's decline-first
  premise clause tells the reader to decline when the memories do not
  support what the question assumes. An advice request assumes nothing to
  recall, yet the reader treats it as a recall question with no recorded
  answer. The more of the user's conversation it sees, the more explicitly
  it can say what the memories "only note".
- The retrieval half of M76 works: stage 0, +50.0 [+33.3, +66.7]. What
  does not work is how the reader treats the question.
