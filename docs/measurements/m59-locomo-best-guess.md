# M59 — LoCoMo: answer when the memories bear on the question *(pre-registered 2026-09-24)* — **result: 67.92, −1.95 vs the 9B; does not ship**

## Why

LoCoMo is the benchmark furthest from its SOTA row: the shipped 9B scores
69.87 against MemPro-15's 77.85 (Qwen3-30B). Bonsai 27B knows more, but on
LoCoMo it refuses (M55b, `runs/m55b_locomo_bonsai`):

| Bonsai, categories 1–4 (n = 1,540) | rows |
|---|---|
| right | 1,027 (82% of the 1,248 it answered) |
| wrong | 221 |
| refused | **292** (the 9B: 121) |
| refused with every gold turn in the evidence | **132** |
| refused with some gold turn in the evidence | 46 |

The evidence is the same as the 9B's: gold turns held in 1,736 rows vs
1,732. Lineage recall at k = 6 is 0.909 (M25). The losses are the decision to
answer, not retrieval.

M56b showed the lever exists. Overriding refusals that had evidence in
hand gave +3.96, even with stand-in answers right only 42% of the time.
Kadavath et al. (2022, `10.48550/arXiv.2207.05221`) find larger models "know
what they know"; here the instruction held the answer back. `READER_SYSTEM`
reads "If the memories do not contain the answer, reply exactly: I don't
know", and Bonsai takes "contain" literally on LoCoMo's inferential
questions.

## The arm

`bench --reader-best-guess` appends to `READER_SYSTEM`:

> When the memories contain information that bears on the question — even if
> the answer must be inferred, or is not stated in the words the question
> uses — give your most likely answer instead of "I don't know". Reply "I
> don't know" only when nothing in the memories bears on the question.

**Run.**
- Big serves Bonsai (`MYELIN_READER_MODEL=bonsai-27b`, 2 × 16k, no
  projector).
- The command is LoCoMo's shipped command plus the switch, in two shards,
  merged and closed with `bench --resume` while Bonsai serves:
  `bench --corpus locomo --mode recall --k 6 --max-steps 2 --reader-best-guess`
  → `runs/m59_locomo_bonsai_bestguess`.
- Judged by the 9B.

**Comparisons, paired over 1,986:**
1. **To ship:** against `runs/m19_locomo_full` (the 9B, 69.87). The bar is
   +3.0 on judge 1–4 with the 95% CI excluding zero. **Veto:** adversarial
   below 69.96.
2. **Attribution:** against `runs/m55b_locomo_bonsai` (Bonsai without the
   clause, 66.69).

**Predictions.**
- Answerable refusals fall from 292 to **≤ 150**.
- Judge 1–4 lands at **72 to 75**: +5 to +8 over Bonsai, +2 to +5 over the
  9B.
- Adversarial falls from 92.83 and stays **≥ 75**.
- The gain concentrates in open-domain and multi-hop, which have the most
  refusals per row.

**Falsifiers.**
- The newly answered rows are right less than 30% of the time: the clause
  converts refusals into wrong answers, not right ones.
- Adversarial falls below 69.96.

---

## Result *(2026-09-24 15:13)* — **does not ship: −1.95 against the 9B; the clause barely moves Bonsai's refusals**

`runs/m59_locomo_bonsai_bestguess` (1,986 rows; Bonsai PTQ1_0 recorded as the
served model). Judged by the 9B: 1,268 fresh verdicts, none from cache.
Declines are never sent to the judge.

**1. To ship — against the 9B (`m19_locomo_full`, 69.87):**

| stratum | n | 9B | M59 | Δ | 95% CI |
|---|---|---|---|---|---|
| **judge 1–4** | 1,540 | 69.87 | **67.92** | **−1.95** | **[−3.64, −0.26]** |
| multi-hop | 282 | 57.80 | 55.32 | −2.48 | [−7.09, +2.13] |
| temporal | 321 | 60.44 | 56.07 | −4.36 | [−8.41, −0.31] |
| open-domain | 96 | 29.17 | 27.08 | −2.08 | [−9.38, +5.21] |
| single-hop | 841 | 82.16 | 81.33 | −0.83 | [−2.85, +1.07] |
| adversarial | 446 | 69.96 | 93.50 | +23.54 | [+19.51, +27.80] |

It is a significant negative, so it does not ship.

**2. Attribution — against Bonsai alone (M55b, 66.69):** judge 1–4
**+1.23 [+0.32, +2.14]**, adversarial +0.67. The clause does something, but
very little.

**Predictions.**
- ✗ Answerable refusals ≤ 150: they went 292 → **272**.
- ✗ Judge 1–4 at 72–75: it is 67.92.
- ✓ Adversarial ≥ 75: it is 93.50. The clause did not tip the model into
  answering the unanswerable.
- ~ The gain in open-domain and multi-hop runs +3.1 and +2.1, both with CIs
  touching zero.

**Falsifier: did not fire.** Of Bonsai's 292 refusals, only **27** became
answers, and **19 of those 27 (70%) were right**, well above the 30% line.
7 rows went the other way, from an answer to a refusal.

**What it means.** Asking Bonsai in its prompt to give its best guess
barely changes when it answers. 265 of the 292 refusals stand, with the
evidence unchanged. When it can be moved, it is usually right, so the lever
is real but the prompt is not how to pull it. This is the day's second
prompt arm to move a model's calibration by only a few rows (M57's clause
was worth +1.2 overall on its own).

It is also the evidence behind the user's call, the same afternoon, to push
**code first**: memory-side mechanisms, not reader instructions (see
`BACKLOG.md`, "SOTA push"). M60 (the same clause plus thinking) still runs,
because it was already queued.
