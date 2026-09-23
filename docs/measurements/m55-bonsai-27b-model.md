# M55 — a stronger local model: Ternary Bonsai 2 27B as myelin's model *(pre-registered 2026-09-23)*

## Why the model, and why now

Every measurement this week points at the 9B model as the ceiling that all
three benchmarks share:

- **LongMemEval_S was compute-limited.** Letting the reader think under a
  1,024-token budget was worth +10.6 (M44, 67.8 → 78.4), more than any
  retrieval or write-path change ever measured.
- **LME-V2 misreads the answers it is handed.** With the gold answer in the
  reader's context the 9B still answers only ~64% right, and on enterprise
  only 55% (M53's diagnosis).
- **LoCoMo turns over-cautious.** At the shipped LongMemEval_S settings the 9B
  declined 197 answerable LoCoMo questions against 121 at the plain point
  (M51).

The comparable SOTA row, MemPro-15, answers with Qwen3-30B-A3B. Ternary
Bonsai 2 27B is PrismML's ternary compression of a 27B dense Qwen model:
5.95 GB on disk, local, and already on big's 3090. Ternary weights follow
BitNet b1.58 (Ma et al., *The Era of 1-bit LLMs: All Large Language Models
are in 1.58 Bits*, arXiv:2402.17764). The project runs local models only, so
this is the strongest model in the size class that fits the card beside
big's other tenants.

**Comparability note.** A Bonsai-served result compares a 27B dense model
with MemPro-15's 30B mixture-of-experts. Total parameters are close; active
parameters per token are ~9× higher on our side. G2 ("the answer model class
matches ours") would be reported with that caveat, not silently.

## What changed first (merged before any Bonsai row exists)

1. **A bench artifact now names the model that served it.** `bench` asks the
   server (`GET /v1/models`) at the start of every run and records the model
   file as `llm_served_model` in `aggregated_metrics.json`. An unreachable
   server is an error, not a blank. `standing` treats any run served a model
   other than `Qwen3.5-9B-UD-Q4_K_XL.gguf` as an arm, so a Bonsai run can
   never be quoted as where the shipped system stands (the M22 defect).
2. **`ops/big/serve-models.sh` can serve Bonsai** with
   `MYELIN_READER_MODEL=bonsai-27b`: PrismML's llama.cpp fork (stock builds
   reject the ternary packing), the patched chat template, `-fa on`,
   `--reasoning-format deepseek`. Slots, context, q8 KV, the 1,024-token
   thinking budget and thinking-off-by-default stay the script's own, so the
   arm differs from the base only in the model.

## Pre-registration — LongMemEval_S pilot

**Population.** `m50-pilot-questions.txt`: 100 questions, all six types,
6 abstention.

**Base.** `runs/m50_pilot_base_s1` → judged `runs/m50_pilot_base_s1_judged`:
the shipped LongMemEval_S point on the 9B. **69.0** judged; abstention 4/6;
19 declines; p50 24.9 s per row.

**Arm.** The identical command with big serving Bonsai on the reader port
(2 slots × 16k, thinking budget 1,024, no projector). Bonsai is the whole
system's model: select, digest and reader. The run is split in two shards
of 50 on the two slots and merged with `bench --resume` while Bonsai is
still serving, so `llm_served_model` records Bonsai.

```
ssh big "MYELIN_READER_MODEL=bonsai-27b MYELIN_MMPROJ=0 MYELIN_READER_SLOTS=2 \
  MYELIN_READER_CTX=32768 bash -s" < ops/big/serve-models.sh
myelin-eval bench --corpus longmemeval-s --mode investigate --k 6 \
  --budget-tokens 4096 --max-steps 2 --select-sufficient --item-digest \
  --digest-dates --reader-thinking --reader-seed 1 \
  --questions docs/measurements/m50-pilot-questions.txt --out runs/m55_bonsai_pilot_s1
```

**Judge.** The same Qwen3.5-9B judge as the base, on big, re-served before
judging, seeded from the base (`judge --seed runs/m50_pilot_base_s1`), then
`rescore --scorer judge`. Only the system model differs.

**Reported.** Paired Δ with a 95% bootstrap CI; per question type;
abstention (6 rows); declines; seconds per row (a 27B should cost ~2×);
how often the thinking trace is present.

**Predictions.** **+5 to +10** overall. The gain lands in multi-session and
knowledge-update, where the 9B loses most (55.6 and 66.7 at base), and in
the 19 base declines. Abstention holds at ≥ 4/6. Seconds per row about
double.

**Gate.** **≥ +5** → a full 500-question run, then LoCoMo (does the caution
go away?) and LME-V2. **Below +5** → recorded: the model is not the lever at
this compression. **Veto:** abstention below the base's 4/6.

**Falsifier.** The ternary compression loses more than the size buys: Bonsai
at or below the base, or a thinking trace missing on most rows (the budget
not being honoured by the fork).

**Follow-up if it wins.** A reader-only swap (9B for select and digest, 27B
reading) needs a second LLM URL in `bench`. It is worth building only if
attribution matters for the next step.
