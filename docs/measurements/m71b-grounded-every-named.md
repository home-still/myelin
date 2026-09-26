# M71b — the grounded second pass, with every named thing covered *(pre-registered 2026-09-26, before any row)*

## Why

M71 re-asked M57's 78 declines under M61's grounded schema:
- **+1.20 strict [+0.2, +2.4]**, 7 right and 6 wrong among the answerable
  commits;
- vetoed by one abstention flip (`m71-lme-grounded-override.md`).

The flip was `6456829e_abs`: "How many plants did I initially plant for
tomatoes **and chili**?" The memories state tomatoes and never chili. The
pass cited the tomato memories and answered "5 tomato plants".
- LoCoMo's traps swap the person, which grounding catches (M61: 0
  adversarial flips).
- LongMemEval's traps add an unstated detail, which it did not catch.

Meanwhile the loss anatomy (2026-09-26) counts **20 declines with every
gold turn in hand**. 10 of them are declines whose own text states the
answer.

## The mechanism

One path, replacing M61's pass for every caller (`bench::commit_grounded`).
The second pass now:
1. names each person, thing or event the question names, and cites a
   memory for each, or null;
2. cites the memories that state the answer;
3. answers from them alone, and only if every named thing has a memory.

**The check is in code, not in the model's word.** `accept_grounded`
commits only when every named thing's identifying words (three letters or
more, up to one plural ending) appear in the memory cited for it. For the
trap, "chili" appears in no memory, so the decline stands whatever the
model claims. The unit test
`every_named_thing_must_be_stated_in_its_memory` replays it.

The research is unchanged from M61:
- Sufficient Context (Joren et al. 2024, arXiv 2411.06037): small models
  decline with sufficient context in hand;
- Trust-Align (Song et al. 2024, arXiv 2409.11242) on grounded refusals.

## The arm

`commit-arm --run runs/m57_bonsai_premise_s1 --grounded --out runs/m71b_lme_grounded`:
- It is M71's command, on the same base and the same Bonsai PTQ1_0 file
  with thinking off.
- The 422 answered rows are copied byte for byte. The 78 declines are
  re-asked.
- Judged by the strict 9B (seeded from the base) and LongMemEval's own
  grader.

**Stratum gate (the user's rule, 2026-09-26).** The switch enters the
bundle only if both hold:
1. On the 48 answerable declines the base holds, the paired difference
   under the strict judge has a 95% CI excluding zero.
2. **All 30 abstention rows are unchanged (29/30).**

**Predictions.**
- Commits fall from M71's 14 to **8–12**, because the text check refuses
  paraphrased citations too.
- Precision rises above M71's 54%.
- Answerable fixes: **5–7**, net of breaks.
- The trap stays declined.

**Falsifiers.**
- An abstention row flips: the check is not strict enough.
- Commits fall below 5: it is too strict to matter. In that case report
  the refused citations and do not tune the check on this population.
