# SOTA push — implementation checklist *(opened 2026-09-24)*

The finish line is the best **same-size open-model** rows: LongMemEval_S
80.80, LoCoMo 77.85, LME-V2 74.90. Cloud models are allowed where they help,
and every number that uses one says so. Plan approved by the user
2026-09-24. Each item lands as its own PR; tick it here in that PR.

## A — LongMemEval_S (target 80.80)
- [ ] M57 result written up (`m57-decline-first.md`); strata against shipped and against M55
- [ ] If it clears: per-corpus shipped model and clause, standing / ratchet,
      README and EVALUATION rows, BACKLOG → BACKLOG_DONE. **SOTA crossing
      recorded with its commit.**
- [x] M58 pre-registered (`m58-bonsai-preference.md`)
- [ ] M58 run, judged, written up

## B — LoCoMo (target 77.85)
- [x] `READER_BEST_GUESS_CLAUSE` + `--reader-best-guess` switch, tests
- [x] M59 pre-registered (`m59-locomo-best-guess.md`)
- [ ] M59 run (full 1,986), judged, written up
- [x] M60 pre-registered (`m60-locomo-best-guess-thinking.md`)
- [ ] M60 run, judged, written up
- [ ] B3 (Jev gating of overrides) / B4 (wider k): decided from M59 and M60,
      and recorded either way

## C — LME-V2 (target 74.90)
- [x] M54 pre-registration amended (second controller, provenance)
- [x] Claim-based chunk worker; bmb Codex home and shim; bmb smoke question passes (717 s, 5 items)
- [ ] bmb worker running (started 11:32, chunk web_c08); big worker relaunched after the quick fixes
- [ ] `run_agentrunbook_c.py` records controller and reader served models
- [ ] `standing.rs` reads `memory_type` and has an `agentrunbook_c` mode,
      its pairing keys and a label; tests
- [ ] All 17 chunks done → merged prompt rows → reader pass → judged pair →
      write-up → adoption PR (if ≥ 74.90)
- [ ] M62 design doc (myelin-native route 3)

## Always
- [ ] Each arm's artifact records its served model, clause switches and
      store; standing lists arms correctly after every result
- [ ] Memory and the SOTA table updated at each crossing, with commit hashes
