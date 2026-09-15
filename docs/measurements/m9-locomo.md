# M9 — LoCoMo end-to-end accuracy

All 1,986 questions across 10 conversations, `recall` at `k = 6`, deterministic scorer (no LLM
judge — see `crates/myelin-eval/src/bench.rs` for why, and for the exact normalisation).

| metric | value |
|---|---|
| token F1, answerable (cat 1–4) | **0.5307** |
| exact match, answerable | 0.2870 |
| abstention accuracy (cat 5) | **0.6996** |
| `memory_query` p50 / avg | 0.25 s / 0.28 s |

| category | n | mean | what it tests |
|---|---|---|---|
| 1 | 282 | 0.4252 | single-hop |
| 2 | 321 | 0.2825 | temporal reasoning |
| 3 | 96 | 0.2007 | multi-hop |
| 4 | 841 | 0.6985 | open-domain / commonsense |
| 5 | 446 | 0.6996 | adversarial (unanswerable) |

Multi-hop is weakest at 0.2007, which is the expected shape: our retrieval is one round of hybrid
fusion, and a question needing two joined facts gets one of them.

## The finding: abstention is 70% here and 22% on LongMemEval-V2

Same reader. Same store implementation. Same abstention concept — a question the corpus cannot
answer. LoCoMo scores **69.96%**; LongMemEval-V2 scores **22.2%**.

The difference is the reader prompt. `bench.rs` states the rule outright — *"If the memories do not
contain the answer, reply exactly: I don't know."* — and LongMemEval-V2's vendored prompt does not.

That is the entire spread, and it is worth being precise about what it does and does not imply.

It does **not** mean we can fix LongMemEval-V2 by changing a prompt: that prompt is vendored and
pinned by the protocol, and editing it would invalidate the comparison. This measurement is
therefore not a route to G1.

It also does **not** mean the reader can be steered to abstain through the evidence channel. That
hypothesis was the one measurement this finding motivated, and
`docs/measurements/m6-abstention-gate.md` records the result: telling the reader through its
evidence that nothing was found made things **28.3 points worse**, and the reader demonstrably read
the statement, restated it, and answered from pretraining anyway.

What it does mean is that our abstention numbers are a property of the *harness prompt*, not of the
memory system, and that a system's abstention score is not portable between benchmarks. Reading the
LoCoMo 69.96% and the LongMemEval-V2 22.2% as facts about myelin would be wrong in both directions.

## Scope of the claim

These are our numbers under our documented scorer. No LoCoMo harness is vendored here, so nothing
claims protocol identity with the published table, and no leaderboard comparison is drawn.
`per_question.jsonl` uses the field names `adapters/paired_ci.py` reads, so any two LoCoMo runs can
be compared with paired confidence intervals the same way the LongMemEval-V2 runs are.
