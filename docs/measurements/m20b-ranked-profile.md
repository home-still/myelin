# M20b — the user's preferences, ranked by the question *(pre-registered 2026-09-26, before any row)*

## Why

Preference questions are the whole LongMemEval_S gap.
- MemPro-15 on Qwen3-30B answers 24 of the 30 `single-session-preference`
  questions. We answer 13 under LongMemEval's own grader, and 7 under the
  strict 9B.
- That is −11 questions, or 2.2 points, the entire distance to 80.80
  (`docs/research/sota-catalog-2026-09-26.md`).

Of our 17 losses:
- 7 are declines;
- 10 are generic advice that ignores what the user said;
- in 7 of the 8 that hold no gold turn, the right session is in evidence,
  but only as the assistant's long replies. The user's one-line statement
  ("turbinado adds a richer flavor") ranks below them.

M20 built the right records and chose them by the wrong rule.
- The write pass stores the user's dispositions as `profile` records, a
  median of 124 per tenant ("The user prefers stand-up comedy on
  Netflix.").
- M20 composed 8 of them by recency, which reached a preference's source in
  1 of 30 questions: a measured null (`m20-preference-profile.md`).
- M20 §7 named the repair and deferred it to "a future milestone with a
  rule fixed in advance". This is that milestone.

Research:
- PrefEval (Zhao et al. 2025, arXiv 2502.09597) finds that zero-shot
  preference following is under 10%, and that retrieving the stated
  preference is the best remedy.
- PPRO (Jiang et al. 2026, arXiv 2607.00017) finds profile-guided ranking
  load-bearing.
- CueMem (Wang et al. 2026, arXiv 2609.12354) uses extracted records as
  cues to the turns they came from.

## Stage 0 — retrieval only *(measured 2026-09-26)*

For each of the 30 preference questions:
- rank every live `profile` record of the tenant against the question with
  the cross-encoder (bge-reranker-v2-m3 Q8_0);
- ask whether the top m include one derived from a gold turn, meaning an
  episode holding a LongMemEval `has_answer` turn.

The ground truth is the benchmark's own flags, not the ranker's.

| selection | gold-turn disposition in the block |
|---|---|
| M20: newest 8 | **1 / 30** |
| newest 3 | 0 / 30 |
| ranked, top 3 | 14 / 30 |
| **ranked, top 8** | **18 / 30** |
| ranked top 3, gold *session* | 21 / 30 |

- On the 17 questions we lose, the ranked top 8 reaches the gold turn in 8.
- In 5 of the 17, no disposition was ever derived from the gold turn. The
  write pass missed it, and no selection rule recovers those.

## The mechanism

`pipeline/side_block.rs`, `SideKind::Profile`:
1. When the question is an advice request, read every live `profile` record
   in the question's scope from a side ledger.
2. Rank them all with the cross-encoder.
3. Append the top 8 after the evidence under a
   `[profile] What the user has said about their own preferences and
   situation:` header, with their own 512-token budget.

The evidence is never reordered or dropped. It replaces M20's recency block
(`ComposeConfig::profile`, removed), so there is one profile path.

**The gate, fixed before any row.** `query_shape::is_advice_request`:
- the 12 advice cues ("recommend", "suggest", "tips", "do you think", …),
  minus the 11 cues of recalling old advice ("you recommended", "remind
  me", …);
- on LongMemEval_S it fires on **29 of the 30** preference questions and
  **0 of the other 470**;
- the miss is "Could there be a reason for this?".

## The arm

**Population.**
- The 29 rows where the gate fires are re-run.
- The other 471 prompts are byte-identical by construction: the gate is
  shut, so nothing is appended, and a test pins it. Those rows are copied
  from `runs/m57_bonsai_premise_s1`.

**Configuration.**
- M57's exact command: Bonsai PTQ1_0, thinking at 1,024 tokens, reader seed
  1, the premise clause.
- Plus `--profile-ledger data/longmemeval_s_pref.ledger`, the M20 build,
  whose profile records cover exactly these 30 tenants.
- Output: `runs/m20b_profile_s1`.

**Judging.**
- The strict 9B judge, seeded from M57's verdicts.
- LongMemEval's official grader (`judge_lme_official.py`), seeded from M57's
  official verdicts.

**Stratum gate (the user's rule, 2026-09-26).** The switch enters the
bundle only if all three hold:
1. On the 30 preference rows, the paired difference against M57 under the
   strict judge has a 95% CI excluding zero (`paired_ci.py`).
2. The official-grader difference on the same rows is positive.
3. The 30 abstention rows are unchanged.

At n = 30 the CI needs roughly +4 net questions.

**Predictions.**
- Strict: preference 7/30 → **11 to 15**.
- Official: 13/30 → **16 to 20**.
- Declines on the stratum fall from 7.
- No row outside the gate moves (by construction).

**Falsifier.** Answers that cite a disposition and still miss the rubric
(PrefEval's "inconsistency" error), or new declines. Either shows as a
stratum difference at or below zero.

**Diagnostics reported either way:**
- rows with a block, and dispositions per block;
- per lost row, whether the block held a gold-turn disposition (stage 0's
  measure on the arm's actual blocks).
