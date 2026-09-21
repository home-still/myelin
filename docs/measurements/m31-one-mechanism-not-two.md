# M31 — the selector and the tight budget are one mechanism, and the selector is the better one

**Measured.** M26, M27 and M29 each credited a different switch with about the same +0.06, and at
most one of them could own it. The pre-registered 2×2 settles it.

LongMemEval_S, 478 questions, shipped width (50, 25), k = 6:

| budget | select | recall@6 | pool | trunc | tok-drops | mean rec-tokens | p50 |
|---|---|---|---|---|---|---|---|
| 2048 | off | 0.8251 | 0.9665 | 0.1414 | 8.43 | 357 | 343 ms |
| **2048** | **on** | **0.8847** | 0.9665 | **0.0819** | 7.10 | 330 | 1278 ms |
| 8192 | off | 0.7643 | 0.9665 | 0.2022 | 0.00 | 498 | 337 ms |
| 8192 | on | 0.8756 | 0.9665 | 0.0909 | 0.00 | 444 | 1240 ms |

Artifact: `runs/interaction_longmemeval_s/width.json`.

---

## 1. The pre-registered rule, and the answer

From `PLAN.md` §15, fixed before the first cell ran:

> If they are the same escape mechanism, `select` at 8,192 recovers most of M29's 0.0608 while
> `select` at 2,048 adds much less than its solo +0.0585 — the interaction term is **negative and
> at least half the smaller main effect**. If the interaction is ≈ 0 they are independent levers
> and the combination is the new operating point.

| quantity | value |
|---|---|
| budget main effect (select off) | **+0.0608** |
| selector effect at 2048 | **+0.0596** |
| selector effect at 8192 | **+0.1113** |
| **interaction** | **−0.0517** |
| threshold (half the smaller main effect) | 0.0298 |

**−0.0517 exceeds 0.0298. The rule fires: they are largely the same effect.**

The cleanest way to read it is the other marginal. With the selector **off**, relaxing the budget
costs **0.0608**. With the selector **on**, relaxing the budget costs **0.0091** — the tight
budget buys almost nothing once something is deliberately choosing. The selector subsumes it.

## 2. Which one to keep

The selector, and it is not close:

- It is **stronger where the problem is worst**: +0.1113 at 8192 against +0.0596 at 2048.
- It is **general**: it recovers the loss at both budgets, leaving only 0.0091 between them.
- The tight budget is **accidental and fragile** — it works only while records happen to overflow
  a fixed ceiling, which M28 showed is a property of the corpus (LongMemEval_S's 380-token mean)
  rather than of the mechanism. On LoCoMo, where the budget barely binds, M29 measured its effect
  at −0.0030.

This retires the three-way ambiguity. M26's −0.046, M27's +0.0585 and M29's +0.0608 are **one
finding**: the cross-encoder's top-6 ordering is poor, and the amount recoverable by escaping it
is about +0.06 to +0.11 of gold-turn recall. Two of the three "mechanisms" were the same escape
by different routes.

## 3. What it still does not license

**No default flips.** This is gold-turn recall, and `PLAN.md` §7.1 forbids an LLM in the `recall`
loop whatever this measures. M21's counter-example stands and is the reason the rule was written
this way: the same selector gained 0.658 → 0.838 in recall and measured **exactly +0.0 judged**
inside `investigate`. Recall is necessary, not sufficient.

The best cell costs **+935 ms/query** over its baseline. Nothing here is free.

## 4. Reproducibility

Three cells reproduce prior runs on different binaries and a restarted stack:

| cell | this run | prior | Δ |
|---|---|---|---|
| (2048, off) | 0.8251 | 0.8251 (M26, M27, M29) | 0.0000 |
| (8192, off) | 0.7643 | 0.7643 (M29) | 0.0000 |
| (2048, on) | 0.8847 | 0.8836 (M27) | +0.0011 |

The two non-selecting cells are bit-identical across four runs. The selecting cell moves by
0.0011, which is reranker tie-break noise of the kind M13 documented — and it is the only cell
with a model call in it.

## 5. An instrument defect this exposed

`mean_emitted_rank` is **only informative when `select` is off**. `RecallTrace::pool` is captured
*after* the selector's stable partition, so on a selecting cell it measures position in the
selector's output rather than in the reranker's — and `compose` then takes the head, pinning the
value at `mean(1..k)` = 3.5 whatever the selector did. The run shows exactly that: 3.5 at both
`(8192, off)` and `(8192, on)`.

The field is now documented to say so. `mean_emitted_tokens` has no such defect and stays
readable on every cell — and it carries a small finding of its own: the selector picks **smaller**
records even with no budget pressure at all (444 vs 498 tokens at 8192), which is the
short-record preference M28 flagged, now visible without the budget as a cause.

Fixing it properly means capturing the pool before selection as well as after. That is a real
change to `RecallTrace` and it buys nothing until a run uses it, so it is named in `PLAN.md` §15
rather than half-done here.

## 6. Cost

26 minutes, 478 questions × 4 cells, one model call per query on the two selecting cells. Reader,
reranker, embedder and Qdrant all live; nothing else on the box, per §13.
