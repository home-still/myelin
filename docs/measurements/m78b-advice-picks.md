# M78b — the advice answer selects the user's own turns *(pre-registered 2026-09-28, before any row)*

## Why

M78 made the advice answer structure, `{preferences: [{memory, quote}],
recommendation}`, with no decline field. It failed its gate
(`m78-advice-answer-structure.md`) for one measured reason:
- Asked to quote the user's stated preferences, this reader quoted the
  assistant's advice 18% of the time, and 39% of the time without the words
  block.
- The in-code check refused those quotes, and those rows declined: 8.0 a
  seed, no better than the clause alone (M77c: 7.0).

The structure worked: grammar after thinking, with no parse failures. The
free-text field did not.

## Mechanism

`--advice-picks` (M78b; `advice_answer::{user_memories, advice_picks_schema,
render_picks}`, `bench::read_advice_picks`). It is the alternative to
`--advice-answer`, and it needs `--reader-thinking` and `--user-words`.

1. **Only the user's own memories are pickable.** These are the shown
   memories in which every turn is a `user:` turn: the `[your words]` block's
   items (M76). They are detected by content, not by position.
2. **The answer selects, never quotes:**
   `{"picks": [1–3 indices], "recommendation": "…"}`. The `enum` on the
   indices admits only the user's memories. A pick cannot be the assistant's
   words, so there is nothing to verify and no field to decline in. This is
   the selector's `{keep: [int]}` shape, which this reader follows (M40), and
   *Attribute First, then Generate*'s content selection before generation
   (Slobodkin et al. 2024, ACL, arXiv 2403.17104). The field order is the
   wire order (`defect-2026-09-28-schema-field-order.md`).
3. **The answer:** `You told me: "<excerpt>"; …. <recommendation>`. Each
   excerpt is the pick's opening sentences (at least 40 characters, at most
   220, cut at a word).
4. **Gate:** an advice request whose evidence holds none of the user's words
   (1 of 29 rows in M78's runs) is read the ordinary way.
5. The thinking trace stays free, as in M78 (Tam et al. 2024, arXiv
   2408.02442).

## Arm (the 29 advice rows × reader seeds 1–3, the rest copied from M57)

`runs/m78b_picks_s{1,2,3}`: the M57 configuration plus the round-5 bundle's
switches and the picks.
- The bundle's switches: `--events-ledger … --aggregation-k 18
  --aggregation-budget-tokens 8192 --advice-profile-clause`.
- The picks: `--user-words --advice-picks`.

The advice rows are disjoint from the aggregation and past-day sets (checked),
so on these 29 rows the only switches that fire are the clause, the words and
the picks. The rows are therefore both M78b's stratum test and drop-in
replacements for the round-5 bundle's advice rows.

The driver waits for the round-5 unit to finish before it touches big's tree.
It then:
- runs the arm;
- builds `r5b_bundle_s{1,2,3}`: round 5's bundle replicates with their 29
  advice rows replaced by these;
- reruns `commit-arm --grounded` (M71b) on each;
- judges everything.

## Gate and use

- **Stratum gate** (seed-averaged, on the 30 preference rows, against the
  three-seed base): strict CI excluding zero *and* official Δ > 0.
- **Replacing M77c in the bundle:** only if the gate passes *and* M78b's
  seed-averaged official preference score beats the round-5 bundle's own on
  the same rows. That is a head-to-head against M77c under identical
  switches: `m78b_picks_s*` against `r5_bundle_s*`.
- If it replaces M77c, the bundle's bar is read on `r5b_bundle_s*_grounded`
  instead of `r5_bundle_s*_grounded`. Everything else is as
  `r5-bundle-seeds.md` pre-registered.

## Predictions

- Declines ≤ 2 per seed. Only the no-words row, and a recommendation that
  writes "I don't know", can decline.
- Official 10.7 → 18–21 (M77c: 16.3). Strict 8.7 → 13–16 (M77c: 12.0).
- No content fails to parse, and no pick is refused by `render_picks`.

## Falsifiers

- **Declines fall and official does not rise over M77c:** the forced answers
  are wrong or generic, and removing the decline adds nothing. Report the
  rows that turned from a decline into a wrong answer.
- **Picks name the user's question, not their preference.** For example,
  "Can you recommend a hotel?" picked as the stated preference: the excerpt
  then carries no preference. Report the share of picks whose excerpt is a
  request.
