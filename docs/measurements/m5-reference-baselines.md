# M5 — reference baselines

Purpose: anchor `myelin`'s numbers against the two ends of the scale. Without a floor, "36.7%" is
unreadable — it could be most of the achievable signal or almost none of it.

## `no_retrieval`, `web`, `small`

The harness's own `NoRetrievalMemory`: `insert()` is a no-op, `query()` returns nothing. The reader
sees the question and no evidence whatsoever. Run with the pinned reader (Qwen3.5-9B, thinking off)
over all 240 `web` questions.

| system | overall | non-abstention | abstention | `memory_query` avg |
|---|---|---|---|---|
| `no_retrieval` | **6.7%** | 8.3% | 2.8% | 0.00 s |
| `myelin` recall k=25 | **36.7%** | 42.9% | 22.2% | 1.83 s |
| RAG: query → slice + notes *(paper)* | 51.0% | — | — | 0.2 s |
| AgentRunbook-C *(paper)* | 74.9% | — | — | 108.3 s |

**Memory is worth +30.0 points.** That is the honest measure of what this system contributes: from
6.7% with nothing to 36.7% with our store. It is also 14.3 points short of the RAG reference, which
is the number that matters for G1.

## The reader will not abstain on its own

The load-bearing row is `no_retrieval`'s abstention column: **2.8%**. Given *literally no evidence*,
the correct answer to every abstention question is "I don't know" — it is available for free, with
no retrieval quality required. The reader gets it right 2.8% of the time. It fabricates an answer
instead, on 97.2% of questions where it has nothing whatsoever to fabricate from.

This reframes the abstention gap diagnosed in `m6-g1-breakeven.md`. Two hypotheses were live:

1. our retrieval surfaces misleading evidence, and the reader is reasonable given what it sees; or
2. the reader's disposition is to answer regardless, and no evidence set will fix that.

The 2.8% floor is decisive for (2). Evidence quality cannot explain a failure that persists when
the evidence set is *empty*. Our store already lifts abstention 2.8% → 22.2%, so retrieval is
helping — but the residual is a reader that needs to be *told* it has nothing, not shown nothing.

That is precisely the escalated design change: emit an explicit insufficiency signal when
`Investigator` stops with `stopped_because != "sufficient"`, rather than passing a weak pool and
hoping the reader draws the right conclusion. This measurement is the strongest evidence for it,
and it was obtained without touching the abstention code at all.

## Not run

`rag` requires an embedding endpoint; pointing it at our Qwen3-Embedding-8B would measure
"rag-with-our-embedder", not the published 51.0 reference, so it cannot reproduce the frontier
point and is not a substitute for it. The published value is used directly in `compute_lafs.py`'s
hard-coded frontier, which is the only place it is load-bearing.
