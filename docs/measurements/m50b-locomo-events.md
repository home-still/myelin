# M50b — the events calendar on LoCoMo *(measured 2026-09-23: −0.13, null)*

M50's LongMemEval_S pilot measured the events calendar at **+4.0**
[−3.0, +11.0] over 100 questions — below its +5 gate, and concentrated where
the base declined (+15.8 over 19 rows). LoCoMo's events were built the same
day (`events-build --corpus locomo`: 939 events over 272 sessions, 38% with a
resolved date; `myelin_locomo_events`, `data/locomo_events.ledger`), so this
arm costs only reading.

**Base.** `runs/m19_locomo_full` — LoCoMo's shipped point: `recall`, k = 6,
plain reader, dates resolved, timeline on. Judge (categories 1–4) **69.87**;
adversarial 69.96; 121 declines on answerable rows.

**Arm.** The same command on the events store:

```
myelin-eval bench --corpus locomo --mode recall --k 6 --max-steps 2 \
  --collection myelin_locomo_events --ledger data/locomo_events.ledger \
  --out runs/m50b_locomo_events
myelin-eval judge --run runs/m50b_locomo_events
```

Reader served as the 9B (2 × 64k, nothing else running on it). Paired by
`scratchpad/locomo_paired.py runs/m19_locomo_full runs/m50b_locomo_events`.

**Predictions.** Judge (1–4) **+2 to +4**; the gain concentrated in the 121
base declines (M50's pattern) and in multi-hop (events are cross-turn
summaries); temporal flat or up (M19 already resolves dates in turns; events
add dated tuples); adversarial **no drop — the veto**. An event is a
statement of what happened; on a question about what did not, it should not
supply an answer.

**Ships** on +3.0 with the CI excluding zero and no adversarial drop; LoCoMo's
standing row would then move. **Falsifier:** events crowding turns out of
k = 6 — rows whose base evidence held the gold `dia_id` and whose arm evidence
does not — reported before judging.

## Results — measured 2026-09-23

**Judge (categories 1–4): 69.87 → 69.74, −0.13 [−1.49, +1.23].** Null, far
below the +3.0 bar. It does not ship, and LoCoMo's standing row stays at
`runs/m19_locomo_full`.

| stratum | n | base | arm | Δ | 95% CI |
|---|---|---|---|---|---|
| judge, categories 1–4 | 1,540 | 69.87 | 69.74 | −0.13 | [−1.49, +1.23] |
| multi-hop | 282 | 57.80 | 57.45 | −0.35 | [−3.90, +2.84] |
| temporal | 321 | 60.44 | 60.44 | +0.00 | [−4.67, +4.67] |
| open-domain | 96 | 29.17 | 28.12 | −1.04 | [−5.21, +3.12] |
| single-hop | 841 | 82.16 | 82.16 | +0.00 | [−1.19, +1.19] |
| adversarial (veto) | 446 | 69.96 | 71.08 | +1.12 | [−1.35, +3.81] |
| rows the base declined | 121 | 0.00 | 11.57 | +11.57 | [+5.79, +17.36] |

Declines on answerable rows went from 121 to 116, and 1,570 of 1,986 answers
are byte-identical. The judge is deterministic on unchanged answers: 0
verdict flips over the 1,111 identical answers both runs judged. So the
paired numbers are the arm's effect, not judge noise.

**The falsifier fired, and it explains the null.** It was checked before
judging. A gold turn counts as held when its speaker and first 60 characters
appear verbatim in the evidence, which undercounts equally in both runs.

| rows (of 1,986) | count |
|---|---|
| base evidence held a gold turn | 1,732 |
| arm evidence held a gold turn | 1,687 |
| lost a gold turn the base held | 74 |
| gained one | 17 |

The pattern repeats M50's pilot. Events rescue the rows the reader would
have declined: 14 of 121 flip to correct. But at k = 6 they compete with
turns for the same slots, and the turns they displace cost as much as the
rescues gain. Temporal did not move, as on LongMemEval_S: resolved dates
were already in the turns (M19).

**What this leaves.** Events recover declines on both benchmarks (+15.8 on
LongMemEval_S's base declines, +11.6 here), so the signal is real. The
losing part is the substitution. The untested design is **events beside
turns**: k = 6 turns as today, plus the top events in their own budget.
That is a new arm (M50c), not a re-run of this one.

Artifacts: `runs/m50b_locomo_events/{aggregated_metrics.json,judge_verdicts.json}`.

