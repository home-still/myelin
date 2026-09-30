# Re-measured under deterministic retrieval *(2026-09-30)*

## Why

Moving myelin to its own Qdrant (PR #192) showed that retrieval depended on
Qdrant's order among tied scores. That order is internal segment order, which
a snapshot, a reindex or a copy can change.
- On LoCoMo's lexical top-50, ties changed the **order** of 85 of 100
  questions' lists and their **membership** in 20.
- On LongMemEval_S, the order of 5 of 100 and the membership of none.

Since PR #192/#193:
- a query fetches past the cutoff, re-queries the tied group when it runs to
  the fetch's end, and orders by `(score desc, id asc)`;
- a run records `tie_order: "score_then_id"`;
- `standing` marks every earlier artifact `stale-config`.

The user decided to re-measure both benchmarks on that code (2026-09-30).
Code: main `afe75b3` (#194, `--cache-ram 0`). Driver:
`myelin-det`/`myelin-det-lme` on big, `det_big.sh` and `det_lme_big.sh`.

## LoCoMo *(re-measured 2026-09-30 17:20)*

The shipped recipe, end to end:
1. `bench --corpus locomo --mode recall --k 6 --budget-tokens 4096` on the
   9B (`runs/det_locomo_base`);
2. M84's `commit-arm --non-recall` (`runs/det_locomo_m84`: 428 declines
   seen, 15 committed);
3. every judge.

**203 of 1,540 answers changed.** The other 1,337 are byte-identical to the
shipped run, so their verdicts were reused.

| reading | before (`m84_locomo_base`) | deterministic (`det_locomo_m84`) | bar | gap |
|---|---|---|---|---|
| **SimpleMem's judge (LeanMem's bar)** | 78.64 | **79.03** | LeanMem (Qwen3-8B) 84.41 | **−5.38** (gate open) |
| LightMem's judge | 78.64 | 78.38 | MemPro-15 (Qwen3-30B) 77.85 | +0.53 |
| MemPro's repo judge | 80.78 | 80.78 | | |
| matched (the lower of LightMem and MemPro) | 78.64 | **78.38** | MemPro-15 77.85 | **+0.53** (still ahead) |
| strict 9B | 70.84 | 70.91 | | |
| abstention (446 adversarial, strict) | 67.94 | **69.51** | | |

By category under SimpleMem's judge:

| category | score |
|---|---|
| multi-hop | 72.34 |
| temporal | 76.64 |
| open-domain | 44.79 |
| single-hop | 86.09 |

**Read:**
- The tie order moved LoCoMo by a quarter point either way, depending on the
  judge. It moved abstention up by 1.6.
- The standing numbers change at the second decimal and the conclusions do
  not:
  - we still lead MemPro-15 on the matched reading;
  - we still trail LeanMem by about 5.4.
- `det_locomo_m84` is the LoCoMo base every round-7 arm pairs against.

**The ratchet pins move with it (honest, recorded):** the LoCoMo LightMem
and matched readings are pinned at 78.64 and now read 78.38. The pins are
re-raised from the deterministic runs in the readout commit, not from the
arbitrary-tie runs.

## LongMemEval_S *(in progress)*

The deterministic evidence differs from the shipped replicates on:
- 95 of 500 rows at seed 1 (50 outside round 5's rerun set);
- 66 and 68 of the 183 rerun-set rows at seeds 2 and 3.

`det_lme_big.sh` re-reads those 229 rows at their seeds, splices them into
round 5's replicates, and closes each replicate. It then runs the shipped
post-passes (M71b grounded, then the NLI premise pass) and the judges.

*(result pending)*
