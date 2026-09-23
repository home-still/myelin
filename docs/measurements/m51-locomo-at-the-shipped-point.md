# M51 — LoCoMo at the shipped LongMemEval_S operating point *(pre-registered 2026-09-23, one bundled arm)*

## Why one arm and not three

LoCoMo's standing row is **69.87** (`runs/m19_locomo_full`, 2026-09-20): a
`recall` run, k = 6, two steps, plain reader, dates resolved and the
`[timeline]` view on. Nothing that moved LongMemEval_S since then has been
run on it:

| lever | measured on LongMemEval_S | shipped |
| --- | --- | --- |
| `select_sufficient` (M32) | +5.8 on LME-V2, the `investigate` default | on |
| dated item digest (M43) | +5.80 [+3.6, +8.0], veto clear | on |
| native thinking, 1,024 tokens (M44 R2) | +10.60, two seeds, abstention *up* | on |

Each is a switch that already cleared the bar on another corpus, so this is
the case for a bundle rather than a per-lever queue (the rule agreed with
the user on 2026-09-23: bundle switches validated elsewhere, one arm per
new mechanism). If the bundle clears, all three ship for LoCoMo and the
attribution is not needed; only if it fails is the split paid for.

The mechanism expectations carry over. LoCoMo (Maharana et al.,
`10.48550/arXiv.2402.17753`) is five categories of question over ten
long dyadic conversations; its temporal stratum is where M19's date
resolution already lives, and it is the stratum that the thinking reader
moved most on LongMemEval_S (+30 on `temporal-reasoning`, Qwen3 report
`2505.09388` §4 on thinking-mode gains on multi-step reasoning). The digest
is Chain-of-Note's reading note (`2311.09210`), one line per memory, dated
(M41/M43); on LoCoMo's multi-hop questions it should do what it did on
LongMemEval_S's two-fact rows (+18).

## Base

`runs/m19_locomo_full`, scored as `standing` scores it: categories 1–4 by
the judge (`qwen3.5-9b`), a declined answerable row as 0 (121 of 1,540 —
all of the unjudged rows are declines), category 5 by the deterministic
decline rule. `scratchpad/locomo_paired.py base base` reproduces the row
exactly.

| stratum | n | base |
| --- | --- | --- |
| judge (cat 1–4) | 1,540 | **69.87** |
| 1 multi-hop | 282 | 57.80 |
| 2 temporal | 321 | 60.44 |
| 3 open-domain | 96 | 29.17 |
| 4 single-hop | 841 | 82.16 |
| 5 adversarial (abstention accuracy) | 446 | 69.96 |
| declines in base, cat 1–4 | 121 | 0.00 |

Comparable rows: MemPro-15 on a Qwen3-30B backbone **77.85** (gap −7.98);
Mem0's paper number 66.88 (+2.99 already).

## Arm

```
myelin-eval bench --corpus locomo --mode investigate --k 6 --budget-tokens 4096 \
  --max-steps 2 --select-sufficient --item-digest --digest-dates \
  --reader-thinking --reader-seed 1 --out runs/m51_locomo_s1
myelin-eval judge --run runs/m51_locomo_s1 --seed runs/m19_locomo_full
myelin-eval rescore --run runs/m51_locomo_s1 --scorer judge --out runs/m51_locomo_s1_judged
```

Reader served as for M44 R2 (`--reasoning-budget 1024`, no budget
message, one slot, nothing co-running). Paired bootstrap per stratum above
with `scratchpad/locomo_paired.py runs/m19_locomo_full runs/m51_locomo_s1`.

**Seeds.** Seed 1 runs first (≈8 h: 1,986 rows at ~15 s). If the judge
row's CI *lower bound* clears +3.0, seed-to-seed noise (±0.5 on M44 R2 and
R2b) cannot undo it and the bundle ships on one seed. If the point estimate
clears the bar but the lower bound does not, seed 2 decides, as R2 did. If
the point estimate is under +1.0, stop: the bundle failed and the split
(thinking alone, then digest alone) is queued instead.

## Predictions, written before the run

- **Judge (cat 1–4): +8 or better.** Single-hop is 55% of the rows and
  already at 82, so the whole-row gain is smaller than LongMemEval_S's.
- **Temporal (2): +15 or better.** The largest mover, as on LongMemEval_S.
- **Multi-hop (1): +10 or better.**
- **Open-domain (3): no prediction** — 96 rows, commonsense the store
  cannot hold; reported, not gated.
- **Single-hop (4): +3 or better**, mostly from the 27 declines.
- **Adversarial (5): ≥ 69.96 — the veto.** The digest states what every
  memory contributes; on a question with no answer in the store that is an
  invitation to invent one. M43 did not trip the veto on LongMemEval_S's
  30 `_abs` rows; 446 rows will say whether that holds.
- **Declines in base (121): ≥ 40 convert.** Thinking converted LongMemEval_S
  declines without the abstention loss that M42/M45 paid.

**Falsifier.** A drop on category 5 with a gain on 1–4 means the bundle is
carrying a vetoed lever, and the split runs digest-off first (the lever
with a plausible adversarial cost), thinking still on.

**Ships:** `shipped_reader_thinking("locomo")` flips to true, the ratchet
floor and `docs/sota/progression.json` move to `runs/m51_locomo_s1`, and
the standing row changes. The `recall` path is untouched.

## Results

*(pending)*
