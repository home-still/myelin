# M50b — the events calendar on LoCoMo *(pre-registered 2026-09-23)*

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
