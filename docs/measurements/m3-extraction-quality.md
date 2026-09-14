# M3 — extraction precision/recall on 50 hand-labelled episodes

`PLAN.md` M3 exit criterion. Measured 2026-09-14 against the LoCoMo corpus built by
`myelin-eval build --corpus locomo`, commit `985554a`, reader `qwen3.5-9b` (Q4_K_XL) with
`enable_thinking: false`.

## Result

| | |
|---|---|
| episodes labelled | 50 |
| facts extracted | 327 (6.5 per episode) |
| gold facts enumerated | 365 |
| **precision** | **0.917** (300 / 327) |
| **recall** | **0.797** (291 / 365) |
| F1 | 0.853 |

## Method

Sample: the first 50 episodes by `ORDER BY id` across the five conversations complete at the
time (`conv-26`, `-30`, `-41`, `-42`, `-43`). Ids are UUIDv5 hashes of the episode's natural key,
so hash order is deterministic *and* uncorrelated with content, position or length — a
reproducible pseudo-random sample with no seed to record and no cherry-picking available:

```sql
SELECT id, tenant, text FROM record
WHERE kind='episodic' AND tenant IN (...)
ORDER BY id LIMIT 50;
```

Each episode's facts were recovered through `prov_derived_from`, then judged by hand against the
episode text.

**Precision** counts a fact correct when it is both *supported* by the episode and *non-degenerate*.
The second clause matters and is not free: "Maria shared an image of a microphone on a table" is
literally true and carries no knowledge, and it competes for one of six evidence slots at read
time. Counting it correct would report a precision the read path does not experience.

**Recall** is against gold facts enumerated by hand per episode — distinct, checkable, standalone
propositions a question could target. Near-duplicate extractions (`sat with her` / `talked with
her` / `listened to her`) collapse to one gold fact, so over-extraction cannot inflate recall.

## Error taxonomy — 27 errors, and 74% of them are one bug

| cause | n | share of extracted |
|---|---|---|
| image-caption restatement | 20 | 6.1% |
| wrong speaker / over-attribution | 6 | 1.8% |
| garbled image read | 1 | 0.3% |

**Image-caption restatement is the dominant failure and it is ours, not the model's.**
`myelin-eval`'s LoCoMo loader appends `[shared an image: <caption>]` to a turn's text so the
caption is not lost — LoCoMo questions are scored against it. The extractor then treats the
sharing *event* as a fact and emits rows like:

```
(sema) Maria shared an image of a microphone, a charger, and a charger on a table.
(proc) John shared an image of a man holding a stick and a giant cartoon figure.
(sema) John has a desk with a laptop and a lightbox.
```

The third is the interesting one: no literal "shared an image" substring, so a grep-based estimate
undercounts. Corpus-wide the literal phrasing appears in **82 of 2,209** non-episodic records
(3.7%); the hand count puts the true rate at 6.1%.

Two of the six misattributions are the same bug wearing a different hat — the model assigns the
image to whichever speaker is discussing it rather than whichever speaker attached it
(`19.1`: Maria credited with a photo John shared).

Genuine content errors are rare: one relation invented outright ("Melanie and Caroline are family
members" — they are friends), two possessives widened ("Max was in John **and Maria's** family"),
one garbled caption ("The fire-fighting brigade loaded a truck with a fire truck in the back").

## What recall misses

Recall failures are not random. The extractor reliably captures *elaborated* facts — anything a
speaker spends two clauses on — and drops facts stated once in passing, including several that
LoCoMo questions plainly target:

- `#25` "John joined the fire-fighting brigade" — the headline event of the episode, stated in its
  first sentence, not extracted. Five downstream details about the brigade were.
- `#10` "Gina lost her job at Door Dash" — the employer is exactly the kind of span a
  single-hop question asks for. `10.1` captured "Jon lost his job"; Gina's, with the employer,
  was dropped.
- `#41` "John got married" — inferable only from "your new wife", and missed.
- `#42` "Tim is learning German" — mentioned twice, extracted zero times.

The pattern: **first-sentence context-setting statements are under-extracted relative to
mid-episode elaboration.** Worst episodes were `#6` (3/8), `#16` (2/6), `#31` (4/8), `#45` (4/8).

## Not fixed yet, on purpose

The obvious prompt fix — forbid facts whose subject is the act of sharing an image, and pin
attribution to the turn that carries the caption — targets 74% of precision errors directly.

It is **not** applied yet because the LoCoMo ingest was mid-run when this was measured. Changing
the extraction prompt between `conv-43` and `conv-44` would produce a corpus whose halves were
built by different extractors, and every downstream number — the M4 ablation, G2 — would be
measured on a mixture. The fix lands after the corpus is complete, and re-ingestion is then a
cost/benefit decision against ~2 GPU-hours, not a reflex.
