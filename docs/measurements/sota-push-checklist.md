# SOTA push — implementation checklist *(opened 2026-09-24)*

The finish line is the best **same-size open-model** rows: LongMemEval_S
80.80, LoCoMo 77.85, LME-V2 74.90. Cloud models are allowed where they help,
and every number that uses one says so. Plan approved by the user
2026-09-24. Each item lands as its own PR; tick it here in that PR.

## A — LongMemEval_S (target 80.80)
- [x] M57 result written up (`m57-decline-first.md`); strata against shipped and against M55: **83.40, +5.0 [+2.2, +7.8], abstention 29/30**
- [x] It cleared: per-corpus shipped model and clause, standing / ratchet,
      README and EVALUATION rows, BACKLOG → BACKLOG_DONE. **SOTA crossing
      recorded with its commit.**
- [x] M58 pre-registered (`m58-bonsai-preference.md`)
- [ ] M58 run, judged, written up

## B — LoCoMo (target 77.85)
- [x] `READER_BEST_GUESS_CLAUSE` + `--reader-best-guess` switch, tests
- [x] M59 pre-registered (`m59-locomo-best-guess.md`)
- [x] M59 run (full 1,986), judged, written up: **67.92, −1.95 [−3.64, −0.26] vs the 9B; does not ship.** Refusals 292 → 272 only
- [x] M60 pre-registered (`m60-locomo-best-guess-thinking.md`)
- [ ] M60 run, judged, written up
- [ ] B3 (Jev gating of overrides) / B4 (wider k): decided from M59 and M60,
      and recorded either way

## C — LME-V2 (target 74.90)
- [x] M54 pre-registration amended (second controller, provenance)
- [x] Claim-based chunk worker; bmb Codex home and shim; bmb smoke question passes (717 s, 5 items)
- [x] ~~bmb worker~~ withdrawn 11:45 at the user's call (too slow, memory-heavy on a daily driver); no bmb chunk in the measurement
- [ ] big worker relaunched after the quick fixes (held by `HOLD_big`; the queue releases it), in 2-slot mode
- [x] M54 amendment 2: sib (RTX 3060) and big_mac (M1 Max) as controllers, same model file (sha256-checked), same flags
- [x] ~~big_mac~~ dropped 12:49: smoke question timed out at 1,800 s with no memory (6–7 tok/s on long contexts); no big_mac chunk in the measurement
- [x] sib smoke question passes (916 s, 9 memory items, 0 failures); sib worker running from 12:58
- [x] Unstarted chunks halved (15 → 30, same question set) so no slow host holds a long tail chunk
- [x] `run_agentrunbook_c.py` records controller and reader served models
- [x] `standing.rs` reads `memory_type` and has an `agentrunbook_c` mode,
      its pairing keys and a label; tests (`SHIPPED_LME_V2_MEMORY` stays `myelin` until adoption)
- [x] Merge and by-controller report tooling (`merge_arc_chunks.py`, `arc_by_controller.py`), tried on the pilot
- [ ] All 32 chunks done (2 + 30 halves) → merged prompt rows → reader pass → judged pair →
      write-up → adoption PR (if ≥ 74.90)
- [x] M62 design doc (`m62-native-trajectory-tools.md`)

## Always
- [ ] Each arm's artifact records its served model, clause switches and
      store; standing lists arms correctly after every result
- [ ] Memory and the SOTA table updated at each crossing, with commit hashes
