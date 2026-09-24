# M55 — a stronger local model: Ternary Bonsai 2 27B as myelin's model *(pilot measured 2026-09-23: +10.0, gate passed)*

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

## Results — LongMemEval_S pilot, measured 2026-09-23

**69.0 → 79.0, +10.0 [+3.0, +17.0], p = 0.005.** The gate was +5, so the
pilot passes. Abstention rose from 4/6 to 5/6, so the veto does not fire.

| stratum | n | 9B | Bonsai | Δ | 95% CI |
|---|---|---|---|---|---|
| overall | 100 | 69.0 | 79.0 | +10.0 | [+3.0, +17.0] |
| answerable | 94 | 69.1 | 78.7 | +9.6 | [+2.1, +17.0] |
| two gold sessions | 44 | 68.2 | 88.6 | +20.5 | [+9.1, +34.1] |
| three or more gold sessions | 17 | 52.9 | 47.1 | −5.9 | [−23.5, +11.8] |
| rows the 9B declined | 19 | 21.1 | 57.9 | +36.8 | [+15.8, +57.9] |
| knowledge-update | 15 | 66.7 | 86.7 | +20.0 | [+0.0, +40.0] |
| temporal-reasoning | 27 | 77.8 | 88.9 | +11.1 | [+0.0, +25.9] |
| multi-session | 27 | 55.6 | 63.0 | +7.4 | [−11.1, +25.9] |

Twelve rows were gained and two lost. Declines fell from 19 to 15. The
predictions held: the gain landed in knowledge-update and in the 9B's
declines, and two-session questions moved most. Questions over three or
more sessions did not move, so composition across many sessions is still the
limit.

| side measurement | 9B | Bonsai |
|---|---|---|
| seconds per row, p50 | 24.9 | 34.0 |
| thinking trace present | 100/100 | 100/100 |
| median trace length, chars | 3,247 | 526 |
| selector declined (`model_declined`) | 2 | 7 |
| answers byte-identical to the 9B's | — | 46/100 |

Bonsai thinks about a sixth as long as the 9B under the same 1,024-token
budget and still answers better. The cost is 1.4× per row, not the 2×
predicted. The artifact records `llm_served_model =
Ternary-Bonsai-2-27B-PTQ1_0.gguf`. Big's reader log confirms the judge was
the 9B: it loaded `Qwen3.5-9B-UD-Q4_K_XL.gguf` and served 47 judge calls.
The other 37 verdicts came from the base's cache for identical answers.

Artifacts: `runs/m55_bonsai_pilot_s1` and `runs/m55_bonsai_pilot_s1_judged`,
each with `aggregated_metrics.json` and `judge_verdicts.json`.

## Pre-registration — the full runs the gate calls for

Both arms run on one Bonsai server: 4 slots × 16k, thinking budget 1,024,
no projector. Both are judged by the 9B, re-served afterwards.

**M55 full: LongMemEval_S, 500 questions.**
- **Base.** `runs/m44_r2_s1_judged`: the shipped point, **78.40**.
- **Arm.** The pilot's command without `--questions` →
  `runs/m55_bonsai_s1`. It inherits the pilot's 100 rows, which came from the
  identical command. It runs the other 400 in two shards and closes the merge
  with `bench --resume` while Bonsai still serves.
- **Judge.** Seeded from `runs/m44_r2_s1`.
- **Bar.** +3.0 with the paired 95% CI excluding zero, and abstention (30
  rows) no lower than base.
- **Prediction.** +6 to +10, shrinking from the pilot's +10. At 78.40 + 6 the
  row would pass MemPro-15's 80.80.
- **If it clears.** Bonsai becomes the shipped model:
  `SHIPPED_LLM_MODEL` and the serve default move in the results PR, and G2
  carries the 27B-dense against 30B-A3B caveat above.

**M55b: LoCoMo on Bonsai, 1,986 questions.**
- **Base.** `runs/m19_locomo_full`: `recall`, k = 6, plain reader, dates
  resolved. Judge (1–4) **69.87**, adversarial 69.96, 121 declines.
- **Arm.** The identical command (`bench --corpus locomo --mode recall --k 6
  --max-steps 2`) on Bonsai → `runs/m55b_locomo_bonsai`, in two shards.
- **Bar.** +3.0 with the CI excluding zero, and adversarial not lower.
- **Prediction.** +4 to +8, with declines on answerable rows below 121 and
  the gain in multi-hop and open-domain, where the 9B is weakest (57.80,
  29.17).
- **Falsifier.** Adversarial drops, meaning a stronger model answers what it
  should refuse.

## LME-V2 plumbing (merged before any Bonsai LME-V2 row exists)

The LME-V2 protocol fixes the reader to Qwen3.5-9B for every system it
compares (`10.48550/arXiv.2605.12493`), so a Bonsai-served reader would
compare our memory against published rows read by a weaker model. The
comparable arm builds memory with Bonsai and reads with the 9B, in two
phases on one card:

1. `run_myelin.py --prompts-only` with Bonsai served. myelin's own calls
   (select, digest) go to Bonsai, and the run stops once every prompt row
   is saved.
2. `run_myelin.py --reuse-prompts-from <phase-1 dir>` with the 9B served.
   The harness reader and the judge are the 9B.

Every LME-V2 artifact now records `memory_llm_served_model` and
`reader_served_model`, each asked of its server, and a replay keeps the
model that built the memory. `standing` treats memory built by a
non-shipped model, or any reader but the protocol's 9B, as an arm. A
domain whose memory Bonsai built never pairs with one the 9B built.

