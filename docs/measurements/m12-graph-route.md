# M12 — the graph route: PPR as a third fused channel

`PLAN.md` §7.1 line 489 (`route? → if query classified multi-hop: PPR 1-hop expansion, merge,
re-rank`) and `docs/EVALUATION.md` §8 row 7 (*"graph expansion | PPR route on/off | gain confined to
multi-hop questions"*) are now built and measured. Row 7 was the last ablation row never run.

**Verdict: `RetrieveConfig::graph` stays `false`.** The decision rule was fixed before the runs and
both of its conditions fail. On LongMemEval_S the mechanism is a *measured regression*: −0.7 points
overall, 95% CI [−1.5, −0.1], p = 0.0255. On LoCoMo it is a precise null everywhere except
single-hop, where it wins +1.6 points — the opposite of the row-7 prediction that any gain would be
confined to multi-hop.

The mechanism, the backfill and the switch all ship. What does not ship is the default.

## What was built

| piece | where |
|---|---|
| deterministic phrase extraction, identical at index and query time | `crates/myelin-core/src/pipeline/phrases.rs` |
| tenant-scoped incidence read, batched authoritative write, keyset record scan | `Ledger::incidence_for_tenant`, `replace_incidence_batch`, `records_after` |
| per-tenant `IncidenceGraph` cache | `store::graph::GraphIndex` (`GRAPH_CACHE_TENANTS = 8`) |
| PPR as a third `RankedList` inside the one `rrf` call | `pipeline::retrieve::Retriever::recall` |
| GPU-free backfill over a built ledger | `myelin-eval phrases` |
| two one-switch ablation arms | `ablate.rs::arms` — `graph_k1`, `graph_rerank` |
| end-to-end switch, recorded in the run artifact | `myelin-eval bench --graph`, `BenchRun::graph` |
| per-category paired CIs | `adapters/paired_ci.py --by-category` |

The `graph` cargo feature is gone: it gated the module, not the mechanism, and a default read path
that can use PPR cannot have it behind a feature flag. `petgraph` is now an unconditional
dependency; `cargo build -p myelin-core --no-default-features` still builds.

`myelin-mcp` is deliberately untouched — `Backend` builds `RetrieveConfig::default()`, so the MCP
surface and the Python LME-V2 adapter behave exactly as before.

## The graph, backfilled

Pure SQLite, no GPU, no Qdrant, no re-ingest. `incidence_rows` is the single function both the
ingest path (`Indexer::index`) and the backfill call, so a fresh ingest and a backfill write
byte-identical rows.

| corpus | records | edges | distinct phrases | edges/record | wall |
|---|---|---|---|---|---|
| LoCoMo | 5,036 | 17,794 | 1,586 | 3.53 | 0.3 s |
| LongMemEval_S | 162,181 | **2,762,496** | 218,422 | 17.0 | 46 s |

Every record in both namespaces received at least one edge. Per-record edges are capped at
`MAX_PHRASES_PER_TEXT = 32` and the cap binds: LoCoMo's episodic records average 19.2 edges against
1.60 for semantic ones, because a semantic record is a one-sentence extracted fact with almost no
capitalized tokens in it. The LongMemEval_S ledger grew 964 MB → 1,390 MB.

`myelin-eval build --corpus locomo` afterwards prints `reconcile: clean (locomo)` — the backfill
writes rows only for records `records_after` returns, so `orphan_incidence` stays empty.

## §8 row 7 — the ablation (retrieval only, no reader)

LoCoMo dev split (5 conversations, 997 questions with gold evidence), `k = 6`, scored by
`qa.evidence` turn coverage. The five pre-existing arms reproduce every accuracy column of
`m4-ablation.md` to the digit — recall, `any`, `mrr` and `items` — which is the check that the
`graph: false` path is byte-identical. (Latency differs from that session, as it always will: a
shared GPU is not a stopwatch.)

| arm | recall@6 | any | mrr | items | p50 ms | p90 ms |
|---|---|---|---|---|---|---|
| dense_only | 0.7314 | 0.7743 | 0.6333 | 6.00 | 11 | 14 |
| bm25_only | 0.8666 | 0.9137 | 0.7805 | 5.81 | 10 | 13 |
| hybrid_k60 | 0.8252 | 0.8676 | 0.7140 | 6.00 | 11 | 14 |
| hybrid_k1 | **0.8815** | 0.9248 | 0.7399 | 6.00 | 11 | 14 |
| hybrid_rerank | **0.9085** | 0.9428 | 0.8144 | 5.99 | 161 | 211 |
| `graph_k1` | 0.8548 | 0.9037 | 0.7321 | 6.00 | 11 | 14 |
| `graph_rerank` | 0.8954 | 0.9348 | 0.8235 | 5.73 | 338 | 445 |

Each graph arm differs from its comparator in exactly one switch:

| pair | Δ recall@6 | Δ items | latency |
|---|---|---|---|
| `graph_k1` − `hybrid_k1` | **−0.0267** | +0.00 | unchanged (11 ms) |
| `graph_rerank` − `hybrid_rerank` | **−0.0131** | −0.26 | 161 → 338 ms (2.1×) |

Per-category recall, the block row 7 is read off:

| category | n | hybrid_k1 | graph_k1 | hybrid_rerank | graph_rerank |
|---|---|---|---|---|---|
| 1 single-hop | 142 | 0.665 | 0.612 | 0.697 | 0.663 |
| 2 temporal | 156 | 0.933 | 0.929 | 0.963 | 0.946 |
| **3 multi-hop** | 44 | 0.572 | **0.561** | 0.568 | **0.553** |
| 4 open-domain | 418 | 0.935 | 0.911 | 0.961 | 0.959 |
| 5 adversarial | 237 | 0.941 | 0.907 | 0.970 | 0.954 |

Row 7's expected direction — *gain confined to multi-hop* — is not observed. There is no gain in any
category, multi-hop included. `items` did not grow (§8 row 4 is satisfied: extra breadth is not
being bought), and the reranked arm costs 2.1× the latency for −1.3 points of recall.

## End to end, both corpora

`--mode recall --k 6`, deterministic scorer. Baselines are the M9 artifacts on disk.

| | LoCoMo base | LoCoMo `--graph` | LME_S base | LME_S `--graph` |
|---|---|---|---|---|
| token F1, answerable | 0.5307 | 0.5315 | 0.4369 | 0.4315 |
| exact match | 0.2870 | 0.2857 | 0.3489 | 0.3447 |
| abstention accuracy | 0.6996 | 0.7175 | 0.9667 | 0.9333 |
| `memory_query` p50 | 0.25 s | 0.35 s | 0.41 s | 0.45 s |
| target category mean | cat 3: 0.2007 | cat 3: **0.2161** | cat 4: 0.3420 | cat 4: **0.3351** |

Baseline reuse is sound, and it was checked rather than assumed. A 200-question LoCoMo probe at
`graph: false` reproduced the baseline's first-200 answerable F1 to the last digit
(0.4451152884361769 both). The LongMemEval_S reader had to be re-served with a larger per-slot
context (`MYELIN_READER_SLOTS=2 MYELIN_READER_CTX=65536`, 32,768/slot): a long session record
produces prompts the default 4,096 cannot hold — the observed failure was 9,716 tokens, and the
largest single record in the corpus is 76,566 chars (≈19k tokens). A 120-question probe under
the new serving config reproduced **every** baseline per-question score bit-identically, so the
change is inert for scoring.

## Paired bootstrap CIs, per category

20,000 paired resamples of the per-question difference (`adapters/paired_ci.py --by-category`).
Positive = the graph run scores higher.

### LoCoMo, n = 1,986

| stratum | n | graph | base | diff | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 1,986 | 57.3% | 56.9% | +0.5 | [−0.3, +1.2] | 0.233 |
| non-abstention | 1,540 | 53.1% | 53.1% | +0.1 | [−0.6, +0.8] | 0.832 |
| abstention | 446 | 71.7% | 70.0% | +1.8 | [−0.7, +4.3] | 0.170 |
| 1 single-hop | 282 | 44.1% | 42.5% | **+1.6** | **[+0.1, +3.1]** | 0.033 * |
| 2 temporal | 321 | 27.4% | 28.3% | −0.9 | [−2.7, +0.8] | 0.291 |
| **3 multi-hop** | 96 | 21.6% | 20.1% | +1.5 | [−2.4, +6.0] | 0.476 |
| 4 open-domain | 841 | 69.6% | 69.9% | −0.2 | [−1.0, +0.6] | 0.586 |
| 5 adversarial | 446 | 71.7% | 70.0% | +1.8 | [−0.7, +4.3] | 0.170 |

### LongMemEval_S, n = 500

| stratum | n | graph | base | diff | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 500 | 46.2% | 46.9% | **−0.7** | **[−1.5, −0.1]** | 0.026 * |
| non-abstention | 470 | 43.1% | 43.7% | −0.5 | [−1.3, +0.0] | 0.070 |
| abstention | 30 | 93.3% | 96.7% | −3.3 | [−10.0, +0.0] | 0.708 |
| 1 single-session-user | 70 | 91.1% | 91.1% | +0.0 | [+0.0, +0.0] | 1.000 |
| 2 single-session-asst | 56 | 79.5% | 79.3% | +0.2 | [+0.0, +0.6] | 0.722 |
| 3 single-session-pref | 30 | 4.8% | 4.6% | +0.2 | [−1.1, +1.3] | 0.689 |
| **4 multi-session** | 133 | 33.5% | 34.2% | −0.7 | [−1.9, +0.0] | 0.269 |
| 5 temporal | 133 | 20.0% | 22.2% | −2.1 | [−5.1, +0.1] | 0.100 |
| 6 knowledge-update | 78 | 63.9% | 63.9% | +0.0 | [+0.0, +0.0] | 1.000 |

\* = 95% CI excludes zero. Category codings differ between the corpora — LoCoMo 3 is multi-hop,
LongMemEval_S 3 is single-session-preference — so only ever compare like against like.

## The decision rule, applied

Fixed in advance: `graph` becomes `true` **iff** (a) LoCoMo cat 3 **and** LongMemEval_S cat 4 both
have paired 95% CIs excluding zero on the positive side, **and** (b) neither corpus's `overall` CI
has a lower bound below −1.0 points.

| condition | required | measured | verdict |
|---|---|---|---|
| LoCoMo cat 3 CI > 0 | lower bound > 0 | +1.5, [**−2.4**, +6.0] | fail |
| LME_S cat 4 CI > 0 | lower bound > 0 | −0.7, [**−1.9**, +0.0] | fail |
| LoCoMo overall lower bound ≥ −1.0 | ≥ −1.0 | −0.3 | pass |
| LME_S overall lower bound ≥ −1.0 | ≥ −1.0 | **−1.5** | fail |

Three of four conditions fail. **`RetrieveConfig::graph` stays `false`**, the same discipline
`tau_abstain` and `label_untrusted` already got: the switch stays, the default does not move, and
the interval is the record.

The bound this puts on the mechanism as implemented: on LoCoMo multi-hop, any true effect is inside
[−2.4, +6.0] points at n = 96 — the CI half-width is ±4.2, close to the ±3.5 `m9-paired-cis.md`
measured on the same stratum, so this is a bounded null and not an underpowered one. On
LongMemEval_S multi-session the effect is inside [−1.9, +0.0], i.e. the mechanism is at best neutral
and more likely slightly harmful there.

## Why it fails, measured rather than guessed

`RecallTrace` carries `graph_seeds`, `graph_hits` and `graph_ms` precisely so this question is
answerable. Sampled over the first 100 questions of each corpus at `graph: true`:

| corpus | seeds mean / p50 | zero-seed questions | hits mean / p50 | graph_ms mean / p50 / p90 |
|---|---|---|---|---|
| LoCoMo | 1.19 / 1 | 1 / 100 | 49.5 / 50 | 0.3 / 0 / 0 |
| LongMemEval_S | 0.42 / **0** | **79 / 100** | 10.5 / 0 | 14.7 / 0 / 67 |

And the switch's *reach* is measurable directly from the two run pairs — how many questions got a
different reader answer at all:

| corpus | answers changed | net effect of the whole run |
|---|---|---|
| LoCoMo | 272 / 1,986 (13.7%) | +0.5 pts, CI [−0.3, +1.2] |
| LongMemEval_S | **19 / 500 (3.8%)** | −0.7 pts, CI [−1.5, −0.1] |

Two distinct failures, and neither is PPR's arithmetic:

**LongMemEval_S: the channel almost never runs.** 79 of 100 sampled questions extract *zero* phrase
seeds, so the route is skipped entirely (a zero-seed PPR is uniform background mass, which is noise
in RRF, so `recall` declines to fuse it). LongMemEval questions are phrased without proper nouns —
*"What did I say about my dog's diet?"* — and the extractor only emits capitalized runs plus years
and ISO dates. That is why only 19 answers move across the whole corpus, and all 70
single-session-user questions return byte-identical answers. The questions that never fire still
pay the `graph_ms` p90 of 67 ms for nothing: every question is a fresh tenant, so the per-tenant
cache never hits and each query costs one `incidence_for_tenant` read against a 2.76M-row table.

**LoCoMo: the seed is the wrong node.** The median question yields exactly **one** seed, and that
seed is usually a speaker name. Applying the same capitalization-plus-stopword rule to all 1,986
questions: median 1 candidate per question, **85.5% name one of the conversation's two speakers,
and 60.6% contain no other candidate at all**. A speaker name is the highest-degree node in the
tenant's graph — `john` alone touches 1,134 records — so for six questions in ten the entire reset
distribution is one hub. Specificity `s_i = |P_i|^-1` correctly down-weights it, but with a single
seed there is nothing else in the distribution to weight it *against*: the walk spreads almost
uniformly over the conversation and the channel returns ~50 near-arbitrary records (hits p50 = 50,
the full `graph_limit`). RRF then fuses a near-uniform list against two informative ones, which is
what costs `hybrid_k1` 2.7 points of recall@6.

Tenant subgraphs are the right size for the mechanism to have worked — `locomo/conv-26` is 461
records / 290 phrases / 1,477 edges; `lme_s/e47becba` is 330 records / 2,542 phrases / 5,358 edges —
so this is not a sparsity problem in the index. It is a seeding problem in the query.

## What this says about the stated cause of the weak axis

`m9-locomo.md` attributes multi-hop weakness to `recall` being one round of hybrid fusion, and
HippoRAG-2 reports +13.2 EM on multi-hop from PPR over a graph. That mechanism is now implemented
faithfully — 1-hop bipartite incidence, damping 0.5, specificity weighting, all-record background
seeds — and it does not transfer, because HippoRAG's PPR is seeded by **LLM NER on every query and
every passage** and this is seeded by capitalization.

The honest reading is that the graph was never the load-bearing part: the seeding is. That is also
the part `build.rs:250`/`:376` refused on cost grounds (hundreds of GPU-hours for a fact-extraction
pass over LongMemEval_S), and the refusal still stands. A future attempt at this axis should spend
its budget on query-side entity linking — an LLM NER call per *query* is cheap, unlike per passage —
rather than on graph structure, and should be measured against these intervals.

Two other things this run establishes, independent of the verdict:

- The `graph: false` path is byte-identical to M9 across two corpora and two probes, so the switch
  can stay in the tree at zero risk to any existing number.
- `myelin-eval phrases` backfills a 162k-record ledger in 46 s with no GPU, so the graph is cheap to
  rebuild if the seeding question is ever revisited.

## Reproduce

```bash
myelin-eval phrases --corpus locomo
myelin-eval phrases --corpus longmemeval-s
MYELIN_QDRANT__URL=http://192.168.1.110:6334 myelin-eval build --corpus locomo   # reconcile: clean
myelin-eval ablate --units 5 --k 6
myelin-eval bench --corpus locomo        --mode recall --k 6 --graph
myelin-eval bench --corpus longmemeval-s --mode recall --k 6 --graph
python3 crates/myelin-eval/adapters/paired_ci.py runs/locomo_recall_graph runs/locomo_recall --by-category
python3 crates/myelin-eval/adapters/paired_ci.py runs/lme_s_recall_graph  runs/lme_s_recall  --by-category
```

Reader and reranker on `big` per `ops/big/README.md`; LongMemEval_S needs
`MYELIN_READER_SLOTS=2 MYELIN_READER_CTX=65536`.
