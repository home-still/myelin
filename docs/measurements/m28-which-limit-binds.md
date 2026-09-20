# M28 — it was never `k`. On LongMemEval_S the token budget binds.

**Measured, entirely offline.** No GPU, no Qdrant, no network — host `big` was down for the whole
milestone. 162,181 live records read from the local SQLite ledger and costed with the production
token estimator.

| corpus | live records | mean tokens | p50 | p90 | 6 × mean | vs shipped 2,048 |
|---|---|---|---|---|---|---|
| **LongMemEval_S** | 162,181 | **380.3** | 421 | 693 | **2,282** | **tokens bind** |
| LoCoMo | 4,871 | 55.8 | 13 | 126 | 335 | `k` binds |

Artifact: `runs/record_cost/record_cost.json`. Reproduce with `sqlite3` and
`max(chars/4, words)` — which is `myelin_core::pipeline::ingest::approx_tokens` verbatim.

---

## 1. What M25, M26 and M27 actually measured

`compose` truncates on **two** limits: `k` slots *or* `max_tokens`. Every width cell those three
milestones ran used `Budget::default()` — `k = 6`, `tokens = 2048` — and **none of them recorded
which one bit**. All three attributed the loss to `k` and called it "top-k truncation".

On LongMemEval_S that attribution is wrong. Six mean-sized records cost **2,282 tokens** against a
2,048 budget, so the sixth slot is frequently unfillable. The reported `trunc` of 0.1414 is
substantially a *token-budget* loss wearing `k`'s name.

On LoCoMo, where the mean record is 56 tokens and six cost 335, `k` really does bind — which is
the cleanest available explanation for the cross-corpus asymmetry M26 found and could not account
for:

| corpus | trunc | miss | trunc ÷ miss | binding limit |
|---|---|---|---|---|
| LoCoMo | 0.0498 | 0.0410 | 1.2× | `k` |
| LongMemEval_S | 0.1414 | 0.0335 | 4.2× | **tokens** |

The two corpora were not showing "the same mechanism, different strength". They were showing
**two different mechanisms**.

## 2. The confound this puts on M27

`compose`'s selection loop `continue`s rather than `break`s when a candidate does not fit:

```rust
if !selected.is_empty() && tokens + cost > cfg.max_tokens {
    dropped_for_tokens += 1;   // M28: was a silent `continue`
    continue;
}
```

So a bound budget does not merely truncate — it **silently reshapes the emitted set toward shorter
records**, scanning past anything too big.

M27 measured the sufficiency selector at +0.0585 emitted recall. Part of that gain may be the
selector promoting *shorter* records into a budget that could not fit the longer ones — a win for
a reason that is not relevance. M27's number stands as measured; its **interpretation** is now
open, and `dropped_for_tokens` is what will close it. This is exactly the kind of thing the
project's own history says to check: a mechanism that wins for an unmeasured reason is a mechanism
that will not transfer.

## 3. What shipped

Both limits are now counted and reported, so no future milestone can attribute a loss to the
wrong one:

- `EvidenceSet::dropped_for_tokens` — candidates that had a free slot and were refused by the
  budget.
- `EvidenceSet::k_bound` — `compose` filled every slot and still had candidates.
- Both mirrored onto `RecallTrace`, aggregated onto `WidthPoint`, and printed as a `tok-drops`
  column beside `trunc` and `miss`.
- `budget_tokens` is now a **grid dimension**, and `BUDGET_GRID` sweeps the shipped width over
  2,048 → 4,096 → 8,192 → 16,384: the one axis no milestone has ever varied.

Three tests pin the attribution: a short corpus binds on `k` with **zero** token drops; a long one
binds on tokens with a non-zero count and fewer than `k` items; and the existing
"admit the single best item even if it alone blows the budget" behaviour is unchanged by the
counters.

## 4. Why this is the right next lever, per the literature

`docs/research/08-context-engineering.md` §11, from the home-still corpus:

> **2.** *"Evidence-set precision beats recall: 10 items ≈ P .19/R .72; 100 items → P .01. Cap
> k = 8–10, never grow to chase recall"* — which is M19's `k = 25` null and M26's monotonic
> decline, predicted in advance.
>
> **5.** *"Question-aware compression is cheap and lossy-tolerant: 3–5× reduction with a gain
> (LongLLMLingua **+21.4pp @4×**; 94% cost cut)"*, and it explicitly mitigates lost-in-the-middle.

The literature says do not add a seventh slot, and does say make each slot carry more. With the
binding limit now identified as **tokens** on the corpus holding 56% of the remaining headroom,
compression is aimed at the constraint rather than past it. Selective Context reports 50% token
reduction for −0.023 BERTscore; an extractive, question-aware compressor needs **no model call**,
so it is measurable on the reader-free width instrument.

## 5. What is not measured

No sweep ran. Host `big` went offline at the end of M27 — no SSH, no ping — and stayed down. The
`BUDGET_GRID` exists, is wired end to end and is unit-tested, but has never executed. The
arithmetic above is a strong prior on what it will find, not a substitute for it: 380 tokens is a
mean over the whole store, and the six records a *query* retrieves are not a uniform sample of it.

The pre-registered rule for that run, fixed here before it: **if `tok-drops` at the shipped
(50, 25, 2048) cell is ≥ 1.0 per query on LongMemEval_S, the budget is the binding limit and
compression is the mechanism to build. If it is ≈ 0, the arithmetic above is wrong about the
retrieved subset, `k` binds after all, and selection stays the lever.**
