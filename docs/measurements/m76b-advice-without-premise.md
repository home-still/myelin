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
