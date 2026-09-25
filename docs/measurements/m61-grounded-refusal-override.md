# M61 — Bonsai's refusals, re-asked with grounding *(pre-registered 2026-09-24, before any row)*

## Why

LoCoMo is 7.98 behind its same-size row (69.87 vs 77.85). Bonsai 27B is the
better reader there *when it answers*: right on 82.3% of the rows it answers,
against the 9B's 75.8%. But it declines 292 answerable questions, and 132 of
those declines had every gold turn in the evidence (M55b). The arithmetic,
on `runs/m55b_locomo_bonsai`:
- answering those refusals at 60% right would score **78.06**, past the row;
- at 50% right, 76.17.

A prompt instruction does not move them. M59's best-guess clause converted
27 of 292, though 70% of those were right. So this arm is a mechanism.

## Prior art here, and what is different

- **M42** re-asked declining rows under `{answer, evidence_absent}` (9B,
  LongMemEval_S): +2.20, with the committed rows 41.9% right. **Vetoed:**
  two adversarial rows were talked out of refusing.
- **M45:** sample agreement does not separate answerable declines from
  traps (AUROC 0.59).
- The new element is **grounding**:
  - the second pass must first list the memories that state the answer
    **about exactly the person, thing or event the question names**, and
    it may answer only from those;
  - LoCoMo's adversarial questions are mostly a swapped person (the
    bottleneck review), which is exactly what an entity-bound citation
    refuses;
  - the field order is citation first, the reverse of M42's.
- Grounded in *Sufficient Context* (Joren et al. 2024, arXiv 2411.06037):
  small models refuse with sufficient context, and a separate sufficiency
  decision should gate the answer. The user chose the reader itself as
  that gate, so the number carries no cloud caveat.

## The arm

- `myelin-eval commit-arm --run runs/m55b_locomo_bonsai --grounded --out runs/m61_locomo_grounded`,
  with Bonsai served as M55b's reader was
  (`MYELIN_READER_MODEL=bonsai-27b`, 2 slots).
- **An exact control:** only declining rows are re-asked (answerable and
  adversarial alike, since the pass cannot know which). Every other row is
  the base's bytes. The prompt is M55b's first-pass prompt byte for byte:
  `READER_SYSTEM`, and no `<today>`.
- Judged by the 9B, seeded from M55b, so only the new answers are judged.

**Comparisons, paired over 1,986:**
1. **To ship (LoCoMo's reader becomes Bonsai plus this pass):** against the
   9B (`m19_locomo_full`, 69.87). The bar is +3.0 on judge 1–4 with the
   95% CI excluding zero. **Veto:** adversarial below 69.96.
2. **Attribution:** against M55b (Bonsai alone, 66.69).

**Predictions.**
- Of the ~292 answerable declines, 45–65% are re-answered, and those
  answers are **60–70% right**.
- Judge 1–4 lands at **74–78**.
- Adversarial falls from 92.83 but stays **≥ 85**, because grounding
  refuses speaker swaps. That is far above the 69.96 veto line.

**Falsifiers.**
- The converted answers are right less than 40% of the time, which would
  mean grounding is not selecting answerable rows.
- Adversarial falls more than 15 points, which would mean the citation
  does not bind the entity.

---

## Result *(2026-09-24 19:07)* — **safe, and too small to ship: +0.19 against the 9B, +3.38 against Bonsai, adversarial untouched**

`runs/m61_locomo_grounded`: the exact-control replay over M55b, 1,986 rows.
- 706 rows declined, and the grounded pass re-asked them; 124 committed an
  answer. The other 1,862 are byte-identical to M55b.
- Judged by the 9B, seeded: 92 fresh verdicts.

| paired over 1,986 | base | M61 | Δ | 95% CI |
|---|---|---|---|---|
| **vs the 9B** (m19), judge 1–4 | 69.87 | 70.06 | **+0.19** | [−1.43, +1.82] |
| vs Bonsai (M55b), judge 1–4 | 66.69 | 70.06 | **+3.38** | [+2.53, +4.29] |
| vs Bonsai, adversarial | 92.83 | **92.83** | 0.00 | 0 flips |
| vs Bonsai, the 292 declines | 0.00 | 17.81 | +17.81 | [+13.36, +22.26] |

Every category rises against Bonsai: multi-hop +3.55, temporal +4.67,
open-domain +5.21, single-hop +2.62.

**Against the pre-registration.**
- ✗ Bar: +0.19 against the 9B's 69.87. It does not ship.
- ✓ Veto: adversarial 92.83, with **not one adversarial row answered**.
  Grounding held perfectly.
- ✗ "45–65% of declines re-answered": it was ~124 of 706 declines
  (answerable and adversarial together), with 52 answerable rows turned
  right. Of the 292 answerable declines, 17.8% became right.
- ✗ 74–78: it is 70.06.
- Falsifiers: neither fired. Adversarial fell 0, not 15. The committed
  answers were right well above 40%.

**What it means.** An entity-bound citation is a safe gate. It is the
adversarial guard M42 lacked, and the mechanism is worth keeping. But
Bonsai cites a supporting memory for only a minority of its refusals; for
the rest it still sees none. The lever is real and small, +3.38 on Bonsai.
To reach 77.85 it has to be stacked with evidence the reader can cite.
