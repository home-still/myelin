# M37 — the second stage was never selecting

## Pre-registration

Written before the arm ran.

**Claim under test.** `rerank_factor = 4` raises answer-string recall over the
shipped `rerank_factor = 1`, at an unchanged `k`.

**Instrument.** `adapters/recall_sweep.py`. Retrieval only — no reader, no
judge. For each question it asks whether every gold phrase appears verbatim in
the composed evidence. Rows whose gold cannot appear verbatim
(`mc_choice_match`, boolean golds) are excluded, not counted as misses.

**Population.** LME-V2 small, both domains, answerable rows only
(`is_abstention_problem == false` in `runs/m34_pools_{web,ent}`), string
checkable. n = 255.

**Arms.** One knob. Both at `k = 25`, `budget_tokens = 200000`,
`prefetch_limit = 50` (the default), `mode = recall`.

| arm | `rerank_factor` | head depth | fused candidates discarded unscored |
| --- | --- | --- | --- |
| base | 1 | `max(25, 25 × 1)` = 25 | ~54–75 of ~79–100 |
| deep | 4 | `max(25, 25 × 4)` = 100 | ~0 |

Factor 4 is chosen so the head covers the whole fused list at the default
prefetch (50 per channel, so ≤ 100 fused). It is the largest selection
available without also moving `prefetch_limit`, which keeps this a
single-knob arm.

**Decision rule.**

- Ship `rerank_factor = 4` as the default if recall improves by **≥ +3.0
  points** and the paired 95% CI excludes zero.
- Otherwise keep the default at 1 and report the null.

**Why a recall arm and not a judged one.** The mechanism's target quantity is
recall; judged accuracy is recall composed with a reader whose accuracy given
present evidence is 62.9%. Measuring recall directly is both cheaper and a
sharper test of the claim. A judged arm only becomes meaningful if this one
passes.

## Why this knob

`pipeline::retrieve` truncates the fused candidate list to `depth` *before*
reranking:

```rust
let depth = rerank_head_depth(self.config.rerank_depth, self.config.rerank_factor, query.budget.k);
let head: Vec<(uuid::Uuid, f32)> = fused.into_iter().take(depth).collect();
```

At the shipped operating point `rerank_depth` is 25 and `k` is 25, so `depth`
is 25. The cross-encoder receives exactly the set that will be emitted. It can
reorder that set; it cannot exclude anything from it, and it never sees the
~54–75 fused candidates ranked below 25, which are dropped on RRF rank alone.

`rerank_depth`'s own doc comment claims it "gets a deeper pool than `compose`
will finally emit". At `k = 25` that was false.

This also explains a null measured earlier in M37: raising `prefetch_limit`
from 50 to 400 moved recall by **+0.6 / +0.0 / +0.8** points at k = 25 / 50 /
100. Widening the first stage cannot pay while the second stage is
degenerate — the extra candidates were fused and then discarded at `take(25)`.

## Safety argument

Shuster et al. (2021, EMNLP Findings, *Retrieval Augmentation Reduces
Hallucination in Conversation*) measured the cost of simply showing a reader
more:

> increasing the number of retrieved documents yields improvements in
> perplexity and F1 measures. However, we see substantial dropoffs in
> Knowledge F1 measures, which might imply that the models begin to
> hallucinate more and more

So the emitted width must not move. `rerank_factor` changes *which* records
are emitted, never *how many*; `a_deeper_rerank_head_does_not_widen_what_the_reader_sees`
pins that at k=5 for factors 1 and 4.

## Results

**Null. The default stays at 1.**

| arm | `rerank_factor` | head | recall (n=255) | mean items emitted |
| --- | --- | --- | --- | --- |
| base | 1 | 25 | 57.65% | 16.8 |
| deep | 4 | 100 | 58.43% | 23.9 |

**+0.78 points, 95% CI [−3.14, +4.71]** (paired bootstrap, 10,000 resamples,
seed 20250921). 14 rows gained, 12 lost — churn, not signal. The
pre-registered bar was +3.0 with a CI excluding zero.

Verified at the wire before the run: the deep server reports `reranked`
100 / 79 / 81 against the base's flat 25, and logs
`retrieval: prefetch=50 rerank_depth=25 rerank_factor=4` at startup.

The null is stronger than the headline suggests. The deep arm emitted **42%
more evidence** and still did not move recall — and the reason it emits more
is itself part of the defect: at `factor = 1` the 25-candidate head is thinned
by ledger admissibility and the token budget until only ~16.8 records survive,
so the shipped configuration was not even satisfying the `k = 25` it was asked
for. Fixing that, and handing the cross-encoder four times the candidates,
buys +0.78 points.

A cross-encoder given 4× the candidates returns essentially the same set. It
is not the bottleneck.

## What this milestone actually establishes

Three independent widening knobs, all measured, all null or near-null:

| knob | change | effect on recall |
| --- | --- | --- |
| `prefetch_limit` | 50 → 400 | +0.6 / +0.0 / +0.8 at k = 25 / 50 / 100 |
| `k` | 25 → 100 | +9.1 (52.9% → 66.7%), at 4× the reader's context |
| `rerank_factor` | 1 → 4 | +0.78, CI [−3.14, +4.71] |

Against a **corpus ceiling of 88.2%** — the fraction of answerable,
string-checkable rows whose gold phrases appear verbatim in *some* record of
the tenant — and a shipped emitted-evidence recall of **59.2%**.

So ~29 points of recall are lost between "the answer is in the store" and
"the answer is in the evidence", and none of the three ways of looking at more
candidates recovers it.

### Why: the index is mostly not text

| record shape | count (web tenant) | share |
| --- | --- | --- |
| `page:` AXTree dumps | 37,731 | 98.5% |
| `step:` agent reasoning | 292 | 0.8% |
| `goal:` trajectory goals | 100 | 0.3% |
| `note:` / `event:` pools | 197 | 0.5% |

A page record is `[720] link '\ue607', clickable, visible` — bid numbers,
ARIA roles, and Unicode private-use icon glyphs. Median 1,838 chars, so the
records are already passage-sized; the problem is not chunking, it is that
there is almost no natural language for either channel to match a question
against. Entity extraction agrees: **all 37,731 page records carry exactly one
entity, the literal token `page`**, taken from the text's own prefix.

That single fact explains every null above. More candidates cannot help when
the candidates are indistinguishable.

### The direction that is not a widening knob

The natural-language layer that *does* discriminate is already in the store
and unused. Ranking the 100 `goal:` records against each question by embedding
cosine locates the trajectory that contains the gold:

| gold trajectory in… | top-1 | top-3 | top-5 | top-10 | top-20 |
| --- | --- | --- | --- | --- | --- |
| share of 112 web rows | 41.1% | 53.6% | 62.5% | **76.8%** | 83.9% |

Median rank 3, out of 100. This is the session-level-vs-turn-level retrieval
granularity axis, not an LME-V2 quirk.

Acting on it needs a scope predicate the store does not currently carry: the
trajectory id lives in `prov_source.doc` and is not in the Qdrant payload, and
`record.scope.session` — which *is* indexed and filterable — was left unset by
the LME-V2 build. Setting it is a write-path change, and scope is immutable by
I1, so it costs a re-ingest rather than a payload backfill. That is the next
milestone's work, not this one's.

## Instrument

`adapters/recall_sweep.py`, added here. Retrieval-only recall measurement
against any served store: no reader, no judge, so a full 451-question sweep
costs embedding and search only. This is the instrument that made all of the
above visible; judged accuracy alone cannot separate "the reader got it wrong"
from "the reader was never shown it".
