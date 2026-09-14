# M3 — write path end to end

`PLAN.md` M3 asks for three numbers per corpus: **records/unit, tokens, wall time**. Measured
2026-09-14, commit `dadf60c`, reader `qwen3.5-9b` UD-Q4_K_XL (`enable_thinking: false`, 4 slots),
embedder `bge-m3` via ollama (1024-d), Qdrant 1.19.1, all on one shared RTX 3090.

## Result

| | |
|---|---|
| conversations | 10 / 10 |
| turns | 5,882 |
| episodes | 550 |
| records (ledger) | **5,036** — 4,168 semantic, 550 episodic, 318 procedural |
| live records | 4,871 (165 superseded by a later `update` or `delete`) |
| **records/unit** | **503.6** |
| wall | **99.8 min** (3,042 s + 2,944 s across two runs) |
| ~9.7 min/conversation | |

Per-conversation records range from 269 (`conv-30`, 369 turns) to 612 (`conv-43`, 680 turns) —
1.6× the spread of the turn counts, so record yield is not a linear function of length.

## Four-op deltas actually fire

The whole point of a four-op delta over an append-only log is that later facts revise earlier ones.
They do:

| event | n |
|---|---|
| `add` | 4,876 |
| `noop` | 761 |
| `update` | 166 |
| `dedup` | 166 |
| `consolidate_rejected` | 38 |
| `delete` | 5 |

`update` and `delete` together superseded **165** records, which is why the ledger holds 5,036 rows
and the hot index holds 4,871. A design where consolidation never fired would show `add` ≈ total
and no supersessions; this one revises about 3.4% of what it writes.

`noop` is the second-largest bucket at 761. Those are candidates the judge explicitly declined —
recorded rather than dropped, because "we considered this and declined" is the audit trail C10 asks
for.

## Where the time goes

From the second run's instrumented totals (5 conversations, 2,944 s):

| stage | seconds | share |
|---|---|---|
| consolidate | 1,979.7 | 67% |
| extract | 843.8 | 29% |
| index | 116.8 | 4% |

Consolidation dominates, and has since the first probe (68% of a 768 s conversation). It is one
model call per candidate against its neighbours, and it is *ordered* — episode N must be able to
supersede what episode N−1 wrote — so the only lever is per-episode concurrency, which is already
at the server's slot count.

Indexing is 4%. The embedder is not the bottleneck and never was, which is why swapping the 8B
embedder for bge-m3 cost nothing in throughput.

## One crash, and what it taught

The first run died at `conv-44` on an unparseable judgement: the model wrote a 2,255+ character
`reason` and ran out of completion budget mid-token. Three fixes followed, and a fourth came from
cleaning up after it:

1. An unparseable judgement now **quarantines the candidate** instead of aborting the corpus.
2. The judge's budget went 512 → 1024 tokens, with `reason` instructed under 25 words.
3. `build` **resumes** from a `unit_complete` audit event — not a row count, because `conv-44` died
   holding 79 records and all 62 of its episodes, so any count-based test would have declared it
   finished.
4. The crash left **two live records in the ledger with no vector** (`Pixie is a small white dog.`
   and one sibling). A ledger row with no vector is the worse of the two possible drifts: it is
   exported, counted and reported, and invisible to every read path. The reverse — a point with no
   row — is harmless, because `recall` re-checks the ledger and drops it.

   `WritePath::insert` now indexes whatever reached the ledger **before** propagating an error, and
   every `build` ends with a `reconcile` that fails the build on unrepaired drift. The two orphans
   were found by diffing the stores by hand; nothing should ever have to be found that way twice.

## Verification

```
ledger all 5036   ledger live 4871   qdrant 4871
live records missing from qdrant:                 0
qdrant points with no live record:                0
```

The 165 superseded records are **deliberately** absent from the hot index. `reconcile` classifies
them as `stale_points` and evicts them: a retracted record that stays retrievable is how I3 leaks.
A first pass at this diff compared *all* ledger rows against Qdrant and reported 165 "missing"
records — that was a bad diff, not a bad store, and it is recorded here because the same mistake is
easy to repeat.

---

# LME-V2-Small

Same commit, same card, 2026-09-14. Ingested **episodically** (`extract_facts: false`) — see
`WritePath::extract_facts` for the ~250 GPU-hour measurement that rules out fact extraction on this
corpus, and `datasets/lmev2.rs` for why the accessibility trees are chunked rather than dropped.

| | |
|---|---|
| trajectories | 200 / 200 (100 `web`, 100 `enterprise`) |
| turns | 89,928 |
| **episodes** | **85,589** |
| records/unit | 427.9 |
| approx input tokens | **39,514,686** |
| wall | **27.1 min** |
| throughput | **52.6 episodes/s, 24,289 tokens/s** |
| reconcile | clean |

Qdrant: 85,589 points, status `green`, ledger and store agree exactly.

Two numbers worth keeping.

**39.5M tokens against the 43M projected** from a 20-trajectory sample. The projection method —
measure the composition of a sample, extrapolate by trajectory count — was accurate to 8%.

**24,289 tokens/s against 8,279 measured earlier.** That earlier figure was taken *while the LoCoMo
ingest was competing for the card*, and it drove a 1.44 h estimate for this pass. The real number
on a quiet card is 2.9× higher and the pass took 27 minutes. A throughput measured under contention
is a lower bound, not an estimate, and planning against it overstates cost by roughly the
contention factor.

Time is 97.6% index (1,587.7 s of 1,626.8 s) — all embedding, no reader. Extraction and
consolidation are exactly zero, which is the episodic path working as designed.
