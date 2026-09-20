# M26 — the same width split on LongMemEval_S: truncation dominates on both corpora

**Measured.** M25 found retrieval nearly solved on LoCoMo and flagged its own biggest caveat: the
result is one corpus's, and M21 said LongMemEval_S disagrees. It does not. On the harder corpus
the same split is **stronger in the same direction** — truncation loss is 4.2× retrieval loss at
the shipped cell, against 1.2× on LoCoMo.

478 questions, six cells, 48 minutes, **no reader and no judge**.

```
MYELIN_EMBED__URL=http://127.0.0.1:5814/v1 \
  myelin-eval ablate --width --corpus longmemeval-s
```

Artifact: `runs/width_longmemeval_s/width.json`.

---

## 1. The grid

LongMemEval_S, all 500 questions less 22 with no `has_answer` turn over the matcher's 30-character
floor, k = 6.

| prefetch | depth | recall@6 | pool | trunc | miss | p50 ms |
|---|---|---|---|---|---|---|
| **50** | **25** | **0.8251** | 0.9665 | 0.1414 | 0.0335 | 629 |
| 50 | 50 | 0.7993 | 0.9902 | 0.1908 | 0.0098 | 1006 |
| 100 | 25 | 0.8246 | 0.9698 | 0.1452 | 0.0302 | 450 |
| 100 | 50 | 0.7982 | 0.9907 | 0.1925 | 0.0093 | 922 |
| 200 | 50 | 0.8006 | 0.9876 | 0.1869 | 0.0124 | 865 |
| 200 | 100 | 0.7791 | **0.9976** | 0.2184 | **0.0024** | 2070 |

**VERDICT: defaults stay.** The shipped cell is not merely the winner — it is the *best cell on
the grid*, at +0.0000. Every wider cell is worse on what the reader sees.

## 2. Side by side with M25

At the shipped `(50, 25)` cell:

| corpus | n | recall@6 | pool | trunc | miss | trunc ÷ miss |
|---|---|---|---|---|---|---|
| LoCoMo | 997 | 0.9092 | 0.9590 | 0.0498 | 0.0410 | **1.2×** |
| LongMemEval_S | 478 | 0.8251 | 0.9665 | 0.1414 | 0.0335 | **4.2×** |

And at the widest cell, `miss` on both corpora is essentially zero — 0.53% and **0.24%**. Pool
recall reaches 99.47% and **99.76%**.

Three things follow, and none of them were true of the roadmap before this pair of milestones:

1. **Retrieval is not the binding constraint on either corpus that carries per-turn gold.** The
   reranked pool holds 96–97% of gold at the shipped cell and 99.5–99.8% when widened.
2. **Truncation is, and it dominates harder where the corpus is harder.** LongMemEval_S loses
   14.1% of gold to `compose`'s top-k against 3.3% never retrieved.
3. **Widening is worse than neutral here.** On LoCoMo it converted retrieval loss into truncation
   loss at ~1:1 and netted +0.0021. On LongMemEval_S it converts at **worse than 1:1** — emitted
   recall falls monotonically with depth, 0.8251 → 0.7791, while pool recall climbs to 0.9976. The
   cross-encoder's precision at the top 6 *degrades* as it is given more to choose from.

Point 3 is the sharpest, because it is a fact about the reranker rather than about the corpus, and
it explains M24's null shape in advance: a mechanism that widens the candidate pool does not merely
fail to help through a fixed `k`, it can actively cost emitted recall.

## 3. Reconciling with M21, which said the opposite

M21's coverage reported **0.662** emitted against a **0.852** pool on LongMemEval_S — far worse
than the 0.825 / 0.967 here. Both are right; they are different populations:

- M21 measured `temporal-reasoning` alone, n = 133 — the corpus's hardest stratum, and one of the
  two the research brief identifies as carrying 170 of its 218 available answers.
- This measures all 478 matchable questions.

The stratum is much harder than the mean, which is exactly why it is a target. The *shape* agrees
in both: `trunc` (0.190 for M21, 0.141 here) is larger than `miss` (0.148 / 0.034). M21 already
had the finding in its numbers; it was reported as a coverage figure rather than as a split.

## 4. What this does not say

- **LME-V2 is untouched and may genuinely be retrieval-limited.** M16 and M22 measured
  S = P(sufficient | wrong) at 7.4% and 12.1% there. That is a third corpus, with no per-turn gold
  annotation, graded by an LLM on sufficiency rather than by set arithmetic. `ablate --width`
  cannot score it — `coverage` hard-errors on it for the same reason — so the "retrieval-limited"
  finding stands exactly where it was measured and nowhere else.
- **Gold-turn coverage is an upper bound on what the reader was given**, not proof it was given
  the answer. `Coverage` resolves a record to the turns it can testify to through I4 lineage, and
  on LongMemEval_S a record matches when it contains the gold turn's 80-character prefix — a
  consolidated record can satisfy that and still have summarised away the detail the question
  needs. That is M3's extraction concern and this instrument is blind to it.
- **22 questions were dropped, and the drop is reported rather than scored zero.** They carry no
  `has_answer` turn longer than 30 characters; `coverage::is_found` refuses short units because
  "Thanks!" occurs in almost any evidence set, so counting them as misses would report the
  corpus's filler as retrieval failure.

## 5. The instrument

One code path, two gold annotations (`ablate::GoldSource`). LoCoMo resolves record ids to `dia_id`
turns through I4 lineage; LongMemEval_S matches the `has_answer` turn's text prefix through
`coverage::is_found` — **verbatim**, not a second matcher, so these numbers are comparable with
every coverage figure since M21. Both recalls in a `WidthPoint` go through one function over one
shape, so their difference is a quantity and not a comparison of two implementations.

Scope mirrors `bench` exactly (tenant `lme_s/{question_id}`, namespace `longmemeval_s`): a sweep
that retrieved from a different scope would measure a different store.

Reader-free, again, and that is why it ran. bge-m3 served from ollama's own blob under llama.cpp
on **CPU** while a household voice assistant held the GPU; verified at cosine 1.000000 against
vectors already in the store. The CPU warm-up sits outside every timed cell, so p50 is Qdrant +
rerank — which is why depth 100 costs 2070 ms and depth 25 costs 450.

## 6. What changes

Nothing in the defaults, on either corpus — the rule said so in advance, twice.

What changes is that the granularity thesis now has two corpora behind it instead of one. The next
mechanism should make each of the six emitted slots carry more of the answer. `PLAN.md` §15 keeps
M27's pre-registered rule and adds the second corpus to it.
