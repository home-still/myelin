# M24 — parallel query decomposition: retrieve for the sub-questions

**Mechanism milestone, functionally verified and not yet measured for accuracy.**
`RetrieveConfig::decompose` ships **off**. What is established here is that the mechanism does the
thing it claims — reaches records the question alone cannot — proven end to end against a live
Qdrant, not asserted from a diff.

```
cargo test -p myelin-core --features integration --test decompose_route
```

---

## 1. Why this, after six nulls

Every retrieval mechanism measured since M12 reorders or trims **one** pool drawn from **one**
query:

| mechanism | milestone | verdict |
|---|---|---|
| PPR graph channel | M12 | −0.7 on LongMemEval_S (CI [−1.5, −0.1]) |
| chronological order | M13 | coin flip: 208 better, 211 worse |
| `k = 25` width | M19 | +3.8, CI spans zero |
| MMR | M21 | **−5.3 / −9.8** — aimed at co-evidence |
| sufficiency selection | M21 / M22 | +3.8 on one corpus, −1.1 on the one G1 is scored on |
| pool-level selection | M22 | −1.1 (CI [−4.7, +2.4]) |

The one mechanism that moved a stratum was M19's resolved dates (+37.6 on LoCoMo temporal), and it
worked by changing **what the records said**, not which records were drawn.

That is not luck. M16 and M22 independently measure the system as **retrieval-limited**:
S = P(evidence sufficient | answer wrong) = 12.1% [8.1, 17.6]. A perfect reader over today's
evidence reaches **41.5** against a 51.0 break-even; perfect retrieval reaches **76.9**. Reordering
a pool cannot add a record the pool never contained.

## 2. The mechanism, read from the vendored source

AgentRunbook-R scores **58.60** on LME-V2-Small with the same Qwen3.5-9B reader we serve, against
our 36.59 — a 22.0-point gap and the only apples-to-apples comparison in the registry. Its
implementation is vendored in this repo, so the mechanism is read rather than inferred
(`crates/myelin-eval/vendor/longmemeval-v2/memory_modules/agentrunbook_r.py`):

- **One** model call emits a structured bundle —
  `{"raw_state_queries": [≤5], "event_query": str, "note_query": str}` (L126–162). The prompt
  explicitly forbids splitting one surface into attribute-level queries.
- Each sub-query is retrieved **separately**, and each block is reranked against the **original
  question**, not the generated one (L1245–1324).

Our `investigate` is the sequential dual: one query per step, each conditioned on the last step's
results. M19 measured two steps at **exactly 0.0**; M22 measured pool selection at **−1.1**. Both
sequential, both null. Parallel decomposition had never been tried here.

## 3. Integration: N more lists into the fusion that already existed

`rrf` is already a multi-list fuser, so `n` sub-queries add `2n` lists to the same call and every
stage below it — admissibility, rerank, selection, compose — sees one ordinary fused list.

Three properties fall out rather than being built:

- A record answering **two** sub-questions accumulates reciprocal rank from both and rises. That is
  the multi-hop shape.
- The rerank already runs against `query.text`, so AgentRunbook-R's "rerank against the original"
  is what this codebase already did — and it is M22's independent finding that cross-probe scores
  are logits on different scales.
- The original question's own lists are **always** retained, so a bad decomposition can only add
  candidates. With zero sub-queries the path is byte-identical to before.

Nothing is de-duplicated across sub-query results. M21 measured that co-evidence for one question
resembles *itself* 1.60× more than the rest of the composed set (token-Jaccard 0.2235 vs 0.1399), so
a redundancy filter here would be aimed precisely at the records multi-hop needs. Union, then rank.

## 4. What is verified

Four integration tests against live Qdrant, 63 records, `HashEmbedder` (lexical, so a record
sharing no tokens with the question is genuinely unreachable from it):

| test | asserts |
|---|---|
| `the_single_query_path_misses_the_second_hop` | the control: with the switch off, "Ravi relocated to Lisbon" is **not** in the evidence set |
| `a_subquery_reaches_a_record_the_question_cannot` | with it on, **both** hops are — each pulled in by a sub-query in its own vocabulary |
| `the_switch_without_an_llm_is_inert_and_says_so` | the switch alone is inert, the trace says so, and recall still works |
| `the_fused_pool_only_ever_grows` | the pool is a union and gains candidates |

The fixture carries 60 distractors deliberately: `prefetch_limit` is 50, and under that every query
retrieves the whole store, so "the pool grew" is unobservable for the arithmetic reason that it
cannot. That was a real defect in the first draft of this test, which passed for the wrong reason.

**Measured on that fixture:** the fused pool goes 61 → 62, while `admitted` is **25 → 25** because
`rerank_depth` binds. So the mechanism's effect here is not a bigger pool — it is a *different*
25 records reaching the cross-encoder. That is the honest description of what decomposition does
at this scale, and it is exactly the "changes which records are drawn" property the six prior
nulls lacked.

## 5. What is NOT measured

No accuracy number. The reader on `big` was unavailable for the whole milestone: a household voice
assistant held ollama `qwen3:8b` on a rolling 30-minute keep-alive that refreshed on use, plus a
10.5 GB llama-swap model, leaving ~4 GB against the reader's ~7 GB. Evicting a live tenant to take
a benchmark number is not a trade this project makes.

The switch therefore ships **off**, which is the standing rule for an unmeasured mechanism anyway.

## 6. The pre-registered rule, fixed before any arm runs

Decomposition ships on for `investigate` if it beats a **fresh same-code base** by **≥ +5.0 judged
points** on LoCoMo multi-hop (n=282) *or* LongMemEval multi-session (n=133), with a paired CI
excluding zero, and regresses no other stratum by more than 2.0. Anything else is a measured null
and the switch stays off.

`recall` is excluded by construction whatever the arm says: `PLAN.md` §7.1 pins that path at "no
LLM in the loop", and this costs one model call per query.

Two constraints M23 now enforces mechanically rather than by memory:

- The base must be re-measured on the same code. `decompose` is in `standing::PAIR_KEYS`, so an
  artifact that predates it is `stale-config` and unpublishable, and `ratchet --strict` fails on it.
- A decomposition run is an **arm**. `bench --decompose N` is recorded in `BenchRun`, read back by
  `rescore`, and counted by `standing`'s arm detection, so it can never publish itself as where we
  stand.

## 7. Reachable from

`bench --decompose N`, both MCP tools (`recall` and `investigate` declare the parameter), and
`adapters/run_myelin.py --decompose N`, which writes it into `memory_params` unconditionally so the
artifact records its own operating point.

On `investigate` this decomposes **every** probe — one model call per step, not per query. That is
the honest composition of the two mechanisms; special-casing the first step would make the arm
measure something the MCP path does not do.
