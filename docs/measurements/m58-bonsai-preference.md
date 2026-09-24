# M58 — Bonsai and the preference clause on LongMemEval_S *(pre-registered 2026-09-24)*

## Why

Bonsai misses **23 of the 30** `single-session-preference` questions
(`runs/m55_bonsai_s1_judged`). That is 4.6 points, the largest pool per
question in the benchmark, and MemPro-15 scores 80.00 on it. **12 of the 23
misses are flat "I don't know"s.**

M20 measured `READER_PREFERENCE_CLAUSE` on the old 9B reader as a null:
+3.3 [−10.0, +16.7], n = 30. The clause targets exactly this refusal:
"Do not reply I don't know when the memories state a relevant preference".
Bonsai reads instructions literally in both directions (M55, M55b), so
the null does not transfer by assumption.

## The arm

M57's command (Bonsai, the premise clause, thinking) plus the existing
`--profile-clause`, full 500 → `runs/m58_bonsai_pref_s1`, judged by the 9B.
No new code. The clause order in the prompt is fixed by `reader_system()`:
preference, then premise.

**Gate:** M20's stratum rule, pre-registered then and reused as is.
- On the preference stratum (n = 30, judged), the paired difference against
  the M57 run has a 95% CI excluding zero, **and**
- the overall score is not lower than M57's, **and**
- abstention is not lower than M57's.

**Ship comparison** (reported): against whichever LongMemEval_S point is
shipped when this lands, with the usual +3.0 bar.

**Predictions.** Preference 7/30 → **13 to 18 of 30**; about half the
refusals become rubric-satisfying suggestions. Overall +1.0 to +2.5. No
movement outside the preference stratum beyond noise, because the clause
names recommendation questions.

**Falsifier.** The clause turns abstention rows into "suggestions", which
would show as abstention falling below M57's.
