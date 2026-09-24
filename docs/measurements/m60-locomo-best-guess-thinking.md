# M60 — LoCoMo: best guess, and room to think *(pre-registered 2026-09-24)*

## Why

M59 targets LoCoMo's refusals. The other large pool is **wrong answers
with the evidence in hand**: 129 of Bonsai's 221 wrong answers had every gold
turn in the evidence (`runs/m55b_locomo_bonsai`), concentrated in temporal
and multi-hop. Those are reading errors. On LongMemEval_S, thinking was the
single largest lever this project ever measured (M44 R2, +10.6, compute
limited).

M51 found thinking made the 9B cautious on LoCoMo: declines rose 121 → 197.
M59's clause is aimed at exactly that caution, so the two are measured
together. This is one new mechanism, thinking on LoCoMo for Bonsai, on top
of M59's arm.

## The arm

M59's command plus `--reader-thinking --reader-seed 1`, with the server's
1,024-token budget verified by `bench` →
`runs/m60_locomo_bonsai_bestguess_think`, judged by the 9B.

**Comparisons, paired over 1,986:**
1. **To ship:** against `runs/m19_locomo_full` (the 9B, 69.87). The bar is
   +3.0 on judge 1–4 with the CI excluding zero. **Veto:** adversarial below
   69.96.
2. **Attribution:** against M59's run. This measures thinking's effect on
   top of the clause.

**Predictions.** +2 to +4 over M59, concentrated in temporal and multi-hop.
Refusals no higher than M59's. About 3× M59's seconds per row.

**Falsifier.** Thinking brings back the caution: answerable refusals rise
by more than 30 over M59, as with M51's 9B.

---

**Deferred, 2026-09-24 ~16:40 (user).** Before any row. LME-V2 is
GPU-bound and the push is code first, so big goes to the M54 full pair
after M62b. M60 stays pre-registered as written, to run in a later
window.
