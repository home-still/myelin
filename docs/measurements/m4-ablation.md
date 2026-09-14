# M4 — retrieval ablation on LoCoMo

`PLAN.md` M4 exit criterion and `EVALUATION.md` §8 rows 1, 2 and RRF. Measured 2026-09-14,
commit `79391e3`, against the complete LoCoMo memory (5,036 records, 4,871 live).

Metric is **recall@6 of gold evidence turns** — deterministic, no reader, no judge. A LoCoMo
question ships a list of `dia_id` turn pointers, so "did we retrieve the right thing" is decidable
by set arithmetic. Coverage is reconstructed by re-running the production segmenter, so each
episode's exact turn list is known rather than string-matched; derived records inherit coverage
through `derived_from`.

Split is by **conversation**: dev = the first 5 (997 questions), holdout = the other 5 (985).
Questions from one conversation share a memory, so a question-level split would leak.

## Dev split — 997 questions

| arm | recall@6 | Δ vs hybrid k=60 | any@6 | MRR | items | p50 | e2e p50 |
|---|---|---|---|---|---|---|---|
| `dense_only` | 0.7314 | −0.0938 | 0.7743 | 0.6333 | 6.00 | 11 ms | 51 ms |
| `bm25_only` | 0.8666 | +0.0414 | 0.9137 | 0.7805 | 5.81 | 10 ms | **10 ms** |
| `hybrid_k60` | 0.8252 | — | 0.8676 | 0.7140 | 6.00 | 11 ms | 51 ms |
| `hybrid_k1` | 0.8815 | +0.0563 | 0.9248 | 0.7399 | 6.00 | 11 ms | 51 ms |
| `hybrid_rerank` | **0.9085** | **+0.0833** | 0.9428 | **0.8144** | 5.99 | 184 ms | 224 ms |

## Holdout — 985 questions, run once

| arm | recall@6 | Δ vs hybrid k=60 |
|---|---|---|
| `dense_only` | 0.6819 | −0.1011 |
| `bm25_only` | 0.8316 | +0.0486 |
| `hybrid_k60` | 0.7830 | — |
| `hybrid_k1` | 0.8579 | +0.0749 |
| `hybrid_rerank` | **0.8923** | **+0.1093** |

**Every ordering holds.** Absolute numbers are 3–5 points lower across the board, which is what a
held-out split is for.

## Finding 1 — BM25 > dense, reproduced

`PLAN.md` §2 finding 1 predicted it from MemPro's ablation (LoCoMo 84.93 → 72.25 without BM25,
→ 82.57 without dense: −12.68 vs −2.36). Our gap, measured on our own code:

$$\text{bm25\_only} - \text{dense\_only} = 0.8666 - 0.7314 = +13.5 \text{ points (dev)},\quad +15.0 \text{ (holdout)}$$

Same direction, same order of magnitude. The claim survives as a design rationale.

It is also *stronger* than predicted here: `bm25_only` beats the configured hybrid. Per category,
dense wins only on `cat3` (open-domain, n=44 — the smallest bucket) and loses `cat5` catastrophically
(0.502 vs 0.954). LoCoMo's evidence is largely name- and noun-anchored, which is exactly where
lexical matching is strongest and where an embedding's paraphrase tolerance buys nothing.

## Finding 2 — rerank is the single biggest lever, confirmed

$$\text{hybrid\_rerank} - \text{hybrid\_k1} = +2.7,\qquad \text{vs } \texttt{hybrid\_k60} = +8.3$$

and MRR 0.7399 → 0.8144, the largest MRR move of any arm. §2 finding 2 predicted this from MS MARCO
(18.7 → 36.5 for a cross-encoder over BM25). Confirmed.

Cost: p50 11 ms → 184 ms, ~17×. At a ≤1 s operating point that is affordable; it is the obvious
first thing to drop for a faster point.

## Finding 3 — `k = 60` was wrong, and the default changed

**This refutes a decision recorded in `pipeline/fuse.rs` before any data existed.**

The prediction was that Cormack's `k = 60` should beat Qdrant's `k = 1`, because flattening the
head lets cross-channel *agreement* arbitrate rather than one channel's confidence. Measured:

| | dev | holdout |
|---|---|---|
| `hybrid`, k=60 | 0.8252 | 0.7830 |
| `hybrid`, k=1 | **0.8815** | **0.8579** |
| | **+5.6** | **+7.5** |

The mechanism is the same asymmetry the prediction invoked, pointing the other way. Cormack tuned
60 for fusing TREC runs of *comparable* quality. Our channels are 13 points apart, so flattening
averages a strong ranking with a weak one and drags the good hits down; sharpening keeps BM25's
confident head and lets dense contribute only where it is also confident.

`DEFAULT_RRF_K` is now `1.0`. `CORMACK_RRF_K` stays named and reachable, because a corpus with
evenly-matched channels would flip this back — which is why `rrf_k` is a query parameter (R4) and
not a constant.

Consequence worth stating: the argument that we *must* fuse client-side to escape Qdrant's `k = 1`
is gone — Qdrant was already doing the better thing. What still requires client-side fusion is R4
(`rrf_k` as a per-query parameter) and the ablation itself, which needs both channel lists unfused.

## Latency

Query embedding is p50 **40 ms**, which dominates every dense-using arm: `bm25_only` is 10 ms
end-to-end against 51 ms for hybrid. Retrieval itself is 10–11 ms. A latency-constrained operating
point should consider BM25-only before it considers dropping the reranker — it gives up 4.2 points
for 41 ms, where the reranker gives up 8.3 points for 173 ms.

## Reproduce

```bash
myelin-eval ablate --units 5 --k 6              # dev
myelin-eval ablate --units 5 --k 6 --holdout    # held-out confirmation
```
