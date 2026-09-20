# M25 — retrieval width, and the end of "retrieval-limited" on LoCoMo

**Measured.** A pre-registered null on the defaults, and a finding that moves where the next
mechanism should aim. 997 questions, six cells, **no reader and no judge** — the first milestone
since M22 to produce a number, because it needed no generation model at all.

```
MYELIN_EMBED__URL=http://127.0.0.1:5814/v1 \
  myelin-eval ablate --width
```

Artifact: `runs/width_locomo/width.json`.

---

## 1. Why width, and why it had never been measured

Six retrieval mechanisms since M12, six nulls. **Not one of them varied width.** Every arm in
`ablate::arms` runs at `RetrieveConfig::default()`'s `prefetch_limit = 50`, `rerank_depth = 25`;
M21 and M22 each declared width out of scope, and M23 added the CLI knobs without an arm.

M24 then measured the consequence by accident: decomposition grew the fused pool 61 → 62 while
`admitted` stayed **25 → 25**, because `rerank_depth` binds. A mechanism that widens the candidate
pool cannot pay off through a window that does not widen with it. That made width the one
untested knob standing between every future retrieval arm and its own effect.

## 2. The rule, fixed before the first cell ran

`RetrieveConfig::default()`'s `(50, 25)` changes **only if** a cell improves emitted `recall@k` by
**≥ 0.02 absolute** *and* costs **< 2×** the shipped cell's p50. Evaluated in code
(`ablate::width_verdict`), not read off the table by eye — a threshold a human applies after
seeing the numbers is not a pre-registration, and the function is unit-tested against a cell one
ulp under the margin, a cell over the latency budget, and a cell that raises only pool recall.

Stated in the same doc comment, in advance: *a cell that raises `pool_recall` without raising
`recall` changes nothing about the defaults — it has moved the loss from retrieval to truncation,
which is a finding about where to aim next and not a reason to pay for a wider pool the emitter
cannot use.* That is exactly what happened.

## 3. The grid

LoCoMo dev split (5 conversations, 997 questions with gold `dia_id` evidence), k = 6.

| prefetch | depth | recall@6 | pool | trunc | miss | p50 ms |
|---|---|---|---|---|---|---|
| **50** | **25** | **0.9092** | 0.9590 | 0.0498 | 0.0410 | 419 |
| 50 | 50 | 0.9071 | 0.9801 | 0.0730 | 0.0199 | 850 |
| 100 | 25 | 0.9085 | 0.9575 | 0.0490 | 0.0425 | 414 |
| 100 | 50 | 0.9046 | 0.9789 | 0.0743 | 0.0211 | 803 |
| 200 | 50 | 0.9033 | 0.9786 | 0.0753 | 0.0214 | 702 |
| 200 | 100 | 0.9113 | **0.9947** | 0.0834 | **0.0053** | 954 |

- `recall` — gold-turn recall of the **emitted** evidence, what a reader would see.
- `pool` — the same arithmetic over the **reranked pool**, before `compose` truncates to k.
- `trunc = pool − recall` — gold retrieval found and `compose` dropped. A selector's to win.
- `miss = 1 − pool` — gold that never entered the pool. Only retrieval can win it.

**VERDICT: defaults stay.** Best cell is 0.9113 against the shipped 0.9092 — **+0.0021**, an
order of magnitude under the +0.02 bar. Width is a measured null on what the reader sees.

## 4. The finding the null is hiding

Emitted recall is flat across a 4× prefetch and 4× depth range, but the pool is not:

**`miss` collapses from 4.10% to 0.53% — a 7.7× reduction.** At `(200, 100)` retrieval puts
**99.47%** of LoCoMo's gold turns in front of `compose`.

And `trunc` rises monotonically with depth, 0.0498 → 0.0834. Widening does not create evidence and
lose it; it **converts retrieval loss into truncation loss** at very nearly 1:1, which is why the
emitted column does not move.

So at the shipped cell the loss is already split 5.0% truncation / 4.1% retrieval, and at the
widest cell it is **8.3% truncation / 0.5% retrieval — 94% of what remains is `compose` throwing
away gold it was handed.**

## 5. What this contradicts, and how far

`docs/sota/research-brief.md` §4 states as a structural finding: *"We are retrieval-limited, not
reader-limited… No reader-side or prompt-side change can close these gaps."* That is measured, and
it stands — **on LME-V2**, where M16 and M22 put S = P(sufficient | wrong) at 7.4% and 12.1%.

On **LoCoMo it does not hold at the retrieval layer.** Gold-turn recall of the pool is 95.9% at the
shipped cell. Retrieval is very nearly solved on this corpus; the binding constraint at k = 6 is
selecting six of twenty-five.

Three limits on how far that generalises, all of them real:

1. **Corpus-shaped, and the other corpus disagrees.** M21's coverage on LongMemEval_S
   temporal-reasoning was 0.662 emitted against a 0.852 pool — trunc 0.190, miss 0.148. Both halves
   are three to thirty times larger than LoCoMo's. M21 already concluded "selection's value is
   corpus-shaped"; this is the retrieval-side twin of that.
2. **Gold-turn coverage is not answer-bearing content.** `Coverage` maps a record to the turns it
   can testify to through I4 lineage. A consolidated record can cover the gold turn and still have
   summarised away the detail the question needs — that is M3's extraction-quality concern, and it
   is not measurable from this instrument. So "91% of gold turns reach the reader" is an upper
   bound on what the reader was given, not a claim that it was given the answer.
3. **The obvious cash-out was already measured and was a null.** If truncation is the constraint,
   raise k. M19 measured k = 25 on LongMemEval temporal at **+3.8 with a CI spanning zero**. More
   records in the evidence set is not the same as more answer in it.

Take (2) and (3) together and the read is: the remaining loss is not *which records* — retrieval
has them — it is **what a record says once it arrives**. That is granularity, not width.

## 6. Cost, and why this one ran

The whole sweep needs an embedder, a reranker and Qdrant. **No generation model.** The reader on
`big` has been held by a household voice assistant for three milestones; this measured anyway, by
serving bge-m3 from ollama's own blob under llama.cpp on **CPU** (`-ngl 0`, port 5814) and pointing
`MYELIN_EMBED__URL` at it. Verified before use: cosine **1.000000** against three vectors already
stored in `myelin_locomo`, so retrieval against that store is comparable to every prior run.

72 minutes wall, of which ~13 is the CPU warm-up. The warm-up is outside every timed cell, so the
p50 column is Qdrant + rerank and not the CPU embedder — confirmed by the embedder logging no
request after warm-up finished. Rerank cost scales with depth, which is the grid's whole latency
story: 419 ms at depth 25, 954 ms at depth 100.

## 7. What changes

Nothing in the defaults — the rule said so in advance and the number did not clear it.

What changes is the target. `PLAN.md` §15's standing claim that the next win is *wider or better
retrieval* is now false on LoCoMo and unproven on LongMemEval_S. The next mechanism should make
each of the six emitted slots carry more of the answer, not add a seventh — which is the
granularity item M23 already half-built (`build --pools`, `--typed-probes`) and never ran against
a store that has the pools.
