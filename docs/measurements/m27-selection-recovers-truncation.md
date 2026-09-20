# M27 — selection recovers 41% of the truncation loss, and the guard that nearly wasn't there

**Measured.** M25 and M26 localised the bottleneck to `compose`'s top-k truncation on both corpora
carrying per-turn gold. This is the first mechanism aimed at that loss, measured on a full
population, with no answer generation and no judge:

| cell | recall@6 | pool | trunc | miss | p50 |
|---|---|---|---|---|---|
| (50, 25) plain | 0.8251 | 0.9665 | 0.1414 | 0.0335 | 410 ms |
| (50, 25) **+ selector** | **0.8836** | 0.9665 | **0.0829** | 0.0335 | 1385 ms |

**+0.0585 emitted recall** — nearly 3× the pre-registered 0.02 bar — recovering **41.4%** of the
truncation loss, at +975 ms/query. LongMemEval_S, 478 questions.

Artifact: `runs/select_longmemeval_s/width.json`.

---

## 1. Why this, and what the rule said in advance

M25/M26 measured retrieval as effectively solved on both annotated corpora — pool recall 96–99.8%
— with 5–22% of gold discarded by `compose` to fit `k = 6`. The pre-registered rule
(`ablate::SELECT_GRID`, committed before the first cell ran) said:

> Selection **recovers** the truncation loss if emitted `recall@k` rises by ≥ 0.02 absolute at the
> shipped cell with `pool_recall` unchanged. Recovering it does **not** flip a default — M21
> measured this same mechanism at exactly +0.0 judged inside `investigate` — it opens a branch
> that only a bench arm can close. Failing to recover it closes the selection branch on evidence.

Both conditions are met, decisively. `pool_recall` is **byte-identical** (0.9665 → 0.9665), which
is the mechanism's own signature: `Selector` stable-partitions the reranked pool and drops
nothing, so it cannot move the ceiling and can only move what is emitted against it. That
identity is the internal-validity check — a selector that changed `pool_recall` would be a bug.

The VERDICT line reports `Keep`, as pre-registered: a selector cell's model call cannot come in
under 2× a pure-retrieval p50, so the latency clause is dispositive by construction and the
recall delta in the table is the finding.

## 2. What this does and does not license

**Does:** the selection branch is open. It is the largest retrieval-side gain this project has
measured on a full population, on the corpus carrying 56% of the remaining headroom.

**Does not:** flip any default. Recall gain is not answer gain, and this project has the
counter-example in its own history — M21 measured the same switch at recall 0.658 → 0.838 on the
temporal stratum and **exactly +0.0 judged** inside `investigate`, because that loop unions its
probes into a 60-record pool and re-composes. `PLAN.md` §7.1 also forbids an LLM in `recall`
whatever this measures. Only a bench arm with a judged column can close it.

Also unmeasured: LoCoMo. Its truncation loss is 0.0498 against LongMemEval_S's 0.1414, so the
recoverable quantity is a third the size and the transfer is not obvious — M21's verdict that
"selection's value is corpus-shaped" applies here too.

## 3. The defect this milestone nearly shipped

The pre-registered second question was the **interaction**: M26 found the cross-encoder's
precision at the top 6 *degrades* as its pool widens (emitted recall fell monotonically
0.8251 → 0.7791 while pool recall rose to 0.9976). If selection is what recovers that, the gain at
`(200, 100)` should exceed the gain at `(50, 25)`.

That cell did not run, and the reason is the finding:

A 100-candidate selector prompt over **real** LongMemEval records is **8,298 tokens**
(`CANDIDATE_CHARS` = 400 × 100 ≈ 36 KB) against a reader serving **8,192 per slot**. llama.cpp
answers `HTTP 400 exceed_context_size_error` on every one. And `Selector::select` degraded any
non-`EmptyCompletion` failure to `Ok(fallback())` — the unmodified rank order — so the cell would
have reported **a recall identical to the unselected cell** and been read as
*"selection does not help at depth 100"*.

A pre-registered question, answered by a mis-sized server, indistinguishable from a real null.
That is the failure class M12, M14 and M20 each lost a run to.

It was caught by pre-flighting a synthetic depth-100 prompt against the live reader before the
cell ran — and only caught properly on the *second* attempt: the first pre-flight used repetitive
filler text, which tokenised at ~6 chars/token and fitted in 6,841 tokens. Real records tokenise
at ~4.4 and do not fit. **Pre-flight with the corpus, not with lorem.**

### What was fixed, not worked around

- `Selector::select` now returns `Selected { keep, degraded }`. A fallback is reported as one.
- `RecallTrace::select_degraded` carries it per query.
- `WidthPoint::select_degraded` carries the rate per cell.
- `width_verdict` returns `WidthVerdict::Degraded` — **before** any recall comparison — when a
  selecting cell fell back on more than `MAX_DEGRADED` (2%) of its queries, so such a cell is
  refused rather than reported.

Five tests pin it, including the exact shape that nearly shipped: identical recall plus a total
fallback rate must read as `Degraded`, not as a null. The 2% floor is not zero on purpose — one
refused request in five hundred is noise, and failing a 90-minute sweep on it would be its own
kind of unreliability — but a systematically mis-sized server degrades on *every* query and can
never pass.

## 4. G3: offline gates pass, the live sweep did not finish

The gate with the only quantified miss (ASR 12.50% [5.5, 26.1] vs ≤10%) was launched concurrently
with the selector sweep. Its offline half passed:

- **E3** quarantine efficacy: **100.0% (15/15)**, 0/12 benign false positives — gate ≥90%. PASS.
- **E3b** (reported, not gated): paraphrased poison **0/8** caught — the pattern gate catches
  templates it was written against and nothing else, which is why M15 built the content
  adjudicator.
- **E5** confidence-is-not-safety: 15/15 quarantined at the highest trust tier, **0 admitted**. PASS.

The live sweep then died at `write legit` with a Qdrant `Timeout expired`, and host `big` went
fully offline — no SSH, no ping, across three probes two minutes apart. **No ASR number.**

I take this one: the host is a 31 GB box with no useful swap, and I had the reader, the reranker,
a 12-thread CPU embedder holding bge-m3 in RAM, Qdrant, and **two** sweeps running against it at
once. Running the two sweeps concurrently was a deliberate call to use a GPU window that has
historically closed without warning; it was the wrong call on this host. The lesson is in
`PLAN.md` §13 now: concurrency on `big` is bounded by host RAM, not by GPU VRAM.

## 5. Reproducing

```
# Reader must serve >= ~9k tokens per slot for a depth-25 selector, and
# ~12k for depth 100. -np 2 -c 16384 is NOT enough for depth 100.
myelin-eval ablate --width --select --corpus longmemeval-s
```

The base cell reproduced M26's measurement **exactly** — 0.8251 / 0.9665 / 0.1414 / 0.0335 —
across a different run, a different binary and a restarted embedder, which is the same-code
baseline check M22's lesson demands.
