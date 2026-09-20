# M29 — the budget binds, and relaxing it makes things worse

**Measured.** M28 predicted from store-wide arithmetic that `max_tokens`, not `k`, is the binding
limit on LongMemEval_S, and pre-registered a rule: *if `tok-drops` at the shipped cell is ≥ 1.0
per query, the budget binds and compression is the mechanism to build.*

The antecedent fired — hard, at **8.44 refusals per query**. The conclusion is wrong, and the same
run says why.

| budget | recall@6 | pool | trunc | tok-drops | mean rank | mean rec-tokens | p50 |
|---|---|---|---|---|---|---|---|
| **2048** (shipped) | **0.8251** | 0.9665 | 0.1414 | **8.44** | **4.7** | **357** | 348 ms |
| 4096 | 0.7702 | 0.9665 | 0.1964 | 0.48 | 3.6 | 481 | 532 ms |
| 8192 | 0.7643 | 0.9665 | 0.2022 | 0.00 | 3.5 | 498 | 548 ms |
| 16384 | 0.7643 | 0.9665 | 0.2022 | 0.00 | 3.5 | 504 | 562 ms |

**Relaxing the budget costs −0.0608 emitted recall, monotonically.** LongMemEval_S, 478 questions,
no reader and no judge. Artifacts: `runs/budget_longmemeval_s/width.json`,
`runs/budget_locomo/width.json`, `runs/record_cost/record_cost.json`.

---

## 1. The control

LoCoMo, same grid, same code, same session:

| budget | recall@6 | tok-drops |
|---|---|---|
| 2048 | 0.9085 | 0.71 |
| 4096 | 0.9055 | 0.00 |
| 8192 | 0.9055 | 0.00 |
| 16384 | 0.9055 | 0.00 |

Where the budget barely binds, relaxing it does essentially nothing: **−0.0030**. The effect
tracks how hard the budget bound — ~12× the refusals on LongMemEval_S, ~20× the effect. That
dose–response is the control, and it is why this is a mechanism and not a coincidence.

## 2. Why a tighter budget wins

`compose`'s selection loop `continue`s rather than `break`s when a candidate does not fit. Two
consequences, and the run measured both:

- **It reaches deeper.** Mean rank of the emitted records is **4.7** at 2048 against **3.5** at
  8192. 3.5 is exactly `mean(1..6)` — at 8192 `compose` emits the literal top six by rank, every
  time. At 2048 it skips oversized candidates and keeps scanning, drawing from further down the
  reranked list.
- **It emits smaller records.** Mean emitted record is **357 tokens** at 2048 against **498** at
  8192, a 28% drop.

Both moved in the predicted direction, so both are operating; this run does not apportion between
them. What it does establish is that **escaping strict rank order is worth +0.06 gold-turn
recall** — and that the shipped 2,048 was doing it by accident.

## 3. This unifies M26, M27 and M29

Three milestones, three mechanisms, one underlying fact:

| milestone | mechanism | effect | what it does to the top-6 |
|---|---|---|---|
| M26 | widen the pool | emitted recall **falls** 0.8251 → 0.7791 | more candidates, same bad ordering |
| M27 | sufficiency selector | **+0.0585** | reorders past the top-6 |
| M29 | tight token budget | **+0.0608** | skips past the top-6 |

**The cross-encoder's top-6 ordering is the problem.** Anything that escapes it helps by about
the same amount; anything that gives it more to order hurts.

That makes a sharp, falsifiable prediction: **the selector and the tight budget are largely the
same effect and should not be additive.** Selector-on at budget 8192 should recover most of the
0.0608; selector-on at budget 2048 should add much less than its solo +0.0585. That is the next
arm and it is four cells.

## 4. What this says about compression

M28 reasoned: tokens bind ⇒ build question-aware compression, per the literature's +21.4pp at 4×
fewer tokens. **This run refutes that reasoning on this corpus.** Compression frees budget, and
freeing budget is precisely what costs 0.06 here — less skipping, shallower reach, strict rank
order restored. A compressor that did nothing but shrink records would move this corpus from the
2048 column toward the 8192 column.

The pre-registered rule was satisfied and its conclusion is still wrong. Recording that plainly
is the point of pre-registering: the antecedent (`tok-drops ≥ 1.0`) was a correct test for
*"does the budget bind"* and a bad proxy for *"is compression the fix"*, because it did not
distinguish the budget's two roles — a ceiling on tokens **and**, accidentally, a sampler that
reaches past a bad ranking.

Compression may still win, but it has to be measured against the tight budget it would be
displacing, and the M28 rule's `tok-drops strictly reduced` clause now reads as a *warning sign*
rather than a success criterion.

## 5. Two live defects this exposes

**`bench`'s default budget is worse than the library's.** `default_budget_tokens()` is 4096;
`Budget::default()` is 2048. On LongMemEval_S that is −0.055 emitted recall, applied to every
bench run since the key existed.

**Every LME-V2 G1 run operates in the worst regime.** `runs/m22_base_web` and
`runs/myelin_inv2_web_small` both carry `budget_tokens = 10000`, `k = 25` — past the point where
this grid flatlines. And the budget binds there too: costed offline from `lme_v2_small.ledger`,
85,589 live records average **463.3 tokens**, so 25 of them cost **11,582** against a 10,000
budget and only ~21.6 fit. G1 has never been measured at a budget where `k = 25` actually
delivers 25 records.

Neither is fixed here. Both are *default changes on the benchmark G1 is scored on*, and the rule
this project runs by says a default moves on a measured arm, not on an inference from another
corpus. The arm is named in `PLAN.md` §15.

## 6. What is not measured

- **Item count per query.** Mean rank and mean size separate *where* the records came from and
  *how big* they were, but not how many were emitted. A tight budget that emits four dense
  records and beats six sparse ones would be a stronger finding still; this run cannot say.
- **Answers.** This is gold-turn recall. M21's warning stands — the same selector that gained
  0.658 → 0.838 in recall measured exactly +0.0 judged inside `investigate`.
- **LME-V2 recall.** It has no per-turn gold, so only its record-cost arithmetic is available.
