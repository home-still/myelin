# M71 — the grounded second pass on LongMemEval_S *(pre-registered 2026-09-25, before any row)*

## Why

LongMemEval_S is the open gate:
- the shipped M57 run is **79.20** strict and **78.60** under LongMemEval's
  own grader (M70);
- the same-size row, MemPro-15 on Qwen3-30B, is **80.80**.

Graded officially, the run loses 107 of 500 rows (loss anatomy,
2026-09-25). **20 of those are declines with every gold turn in the
evidence**:
- multi-session 7, single-session-user 6, temporal 3, and 4 others;
- M57's reader says "I don't know." with the answer in hand.

M61 built the mechanism for that on LoCoMo: a **grounded** second pass on
the reader's own declines.
- It first lists the memories that state the answer about exactly the
  person, thing or event the question names.
- It answers only from them, or else keeps the decline.
- On Bonsai it gained **+3.38 [+2.53, +4.29]** with **0 adversarial flips**
  (`m61-grounded-refusal-override.md`).
- Grounded in Sufficient Context (Joren et al. 2024, arXiv 2411.06037):
  small models decline with sufficient context in hand.

## The arm

`commit-arm --run runs/m57_bonsai_premise_s1 --grounded --out runs/m71_lme_grounded`
replays the shipped run's own rows:
- **Answered rows (422) are copied byte for byte.**
- **The 78 declining rows are re-asked:**
  - 49 answerable and 29 abstention items;
  - same memories, `<today>` and question;
  - the base's own system prompt, rebuilt with the premise clause through
    the function the base was read with (`bench::reader_system_of_run`, new
    here; `commit-arm` used to refuse clause-bearing bases).
- **The second pass** is the same Bonsai 27B file (PTQ1_0,
  `53107f530aa52eb0`) on big, M61's grounded schema as built, without
  thinking.
- **Judging:** the strict 9B, seeded from the base (safe since the
  stale-verdict fix), and LongMemEval's own grader (M70).

It waits for big to have room, since nothing is evicted
(`scratchpad/m71_arm.sh`).

## Bar and veto

- **Ship:** +3.0 on the strict judge over all 500, with the paired 95% CI
  excluding zero.
- **Veto:** any drop on the 30 abstention rows. The base holds 29/30; M42 was
  vetoed on exactly this.
- **Reported beside it, not deciding:** the matched LongMemEval_S metric (the
  official grader). The G2 gate needs it ≥ 80.80.

## Predictions

- It fires on all 78 rows and commits ~20–30. That is M61's share and M59's
  observed 70% right.
- Answerable declines converted correctly: 10–16 of 49. **Strict +2.0 to
  +3.2**, near the bar.
- Abstention holds at 29/30. A false-premise question names something the
  memories do not state, so the grounded list stays empty. **This is the
  mechanism's claim.**
- Official: 78.60 → **80.2–81.2**, around the gate.

## Falsifiers

- Abstention drops: grounding does not guard LongMemEval's false premises
  the way it guarded LoCoMo's swapped people.
- Commits land below 50% right: the pass converts declines into wrong
  answers, not recovered ones.
