# M50c — events beside turns, not instead *(route A chosen 2026-09-24; building)*

## What M50 and M50b measured

Events rescue the rows the reader would have declined, on both benchmarks:
+15.8 on LongMemEval_S's 19 base declines (M50 pilot) and +11.6 on LoCoMo's
121 (M50b). But both arms put the events into the same store as the turns,
so at k = 6 they compete for the same slots. On LoCoMo the events displaced
a gold turn on 74 rows and restored one on 17, and the arm netted −0.13.

## What Chronos actually does

Chronos (Sen et al., `10.48550/arXiv.2603.16862`) never makes events and
turns compete:

- **Two indexes.** Events go to an event calendar, raw turns to a turn
  calendar, "enabling independent retrieval over each representation"
  (§3.1). §5 names the separate indexes as the design's storage cost.
- **Turns first.** Initial retrieval is over the turn calendar only: the top
  100 by cosine, reranked, top 15 kept and widened by a turn each side
  (§3.3).
- **Events on demand.** The agent reaches events through its own
  `search_events` / `grep_events` tools (§3.4).

The ablation matters for sizing: removing events costs Chronos Low 34.5
points but Chronos High only 2.6 (Table 3). The weaker the reader, the more
the events carry. If M55 moves the system to Bonsai 27B, expect a smaller
gain than the 9B's decline-rescue numbers suggest.

## Routes — to be chosen before building

Events are stored today as `semantic` records whose source document ends
in `@event<k>`. There is no event kind, and they sit in a copy of the turn
store (`myelin_*_events`).

| route | what it is | for | against |
|---|---|---|---|
| **A. Dual index** (Chronos) | An events-only collection. Read k = 6 from the base store exactly as today, plus the top m events by the same query, reranked, in their own `[events]` block with their own token budget. | Faithful to §3.1. The turn evidence is the base's by construction, which gives an exact control for attribution. | A second collection and a second query in `recall`/`investigate`. |
| **B. Reserved slots** | Keep the combined store. k = 6 + m, with events capped at m by a quota keyed on the `@event` source suffix. | No new collection. Reuses `compose`'s quota. | One shared candidate pool: events still crowd the prefetch and rerank depth, so the six turn slots are not guaranteed to be the base's. The quota keys on a naming convention, not a type. |
| **C. An event kind** | Add `RecordKind::Event`, rebuild both events stores (extraction is cached, so embedding and writes only), and give events a `kind_quota` allocation. | Typed, and the quota machinery exists. | A schema change touching the ledger. Same shared-pool caveat as B. |

Recommended: **A**, with m = 3 events in a separate 512-token budget. It is
the only route where the turns the reader sees are the base's turns, so a
null would be attributable, and it is what the paper built.

Pilot, once a route is chosen: the M50 100-question LongMemEval_S population
against whichever model is shipped by then. Gate +5.

## Decision and build plan *(2026-09-24 ~15:45)*

**The user chose route A, the dual index.** It is also the first
code-first build aimed at LoCoMo, the benchmark furthest from its
finish-line row (69.87 against 77.85).

**The events-only index needs no new write code.** `events-build` writes
each event as a `semantic` record whose lineage points at its session's
episodic turns. Those turns must already be in the ledger (I4). It indexes
the events into whatever collection it is given. So the build is:
1. copy `data/locomo.ledger` to `data/locomo_evonly.ledger`, so the turn
   records are there for lineage;
2. run `events-build` from the cache (`data/events/locomo.jsonl`, no
   extraction) into a new, empty collection, `myelin_locomo_evonly`.

That collection then holds only events. The base store and ledger are not
touched.

**The read path is the new code:**
- A second retrieval over the events index with the same question.
- It is reranked like the turns, and its top m = 3 go into their own
  `[events]` block under their own 512-token budget.
- Everything it adds comes after the base's k = 6 turns, which stay
  exactly the base's. Measured against the unchanged base, the arm's delta
  is the events alone.
- One mechanism in `myelin-core`, used by `bench` and by the MCP `recall`
  / `investigate` tools alike, and switched by naming the events index.

**Checklist**
- [x] Events-only index built: `myelin_locomo_evonly` and
      `data/locomo_evonly.ledger`. 939 events from 272 cached sessions in
      219 s. Qdrant holds 939 points, the ledger gained exactly 939
      semantic records (5,108 − 4,169), and M50b's build wrote the same
      939.
- [x] Core: `pipeline/events_block.rs`. A second retriever runs over the
      events index (reranked, no second `[timeline]`), appends the top 3
      events after the base evidence under a 512-token budget, and opens
      them with an `[events]` header view. Tested against scratch Qdrant:
      the base items are byte-identical with the block on, the block holds
      only events, and nothing is appended when no event is found.
- [x] `bench --events-collection/--events-ledger`, in both LoCoMo and
      LongMemEval_S paths. It refuses a half-given pair, a non-`myelin_*`
      collection, or a missing or empty index. The run records both, and
      standing treats a run with the block as an arm (test). The MCP switch
      comes with a ship.
- [x] Pilot pre-registered (below); it runs on big after the M62 pilot, before M60 (user, 2026-09-24).

## Pilot — pre-registered *(2026-09-24 ~15:40, before any row)*

**The arm.**
- LoCoMo, the full 1,986, at the shipped LoCoMo settings: the Qwen3.5-9B
  reader, `recall`, k = 6, `max_steps` 2, plain reader, store
  `myelin_locomo`.
- Plus the events block:
  `bench --corpus locomo --mode recall --k 6 --max-steps 2 --events-collection myelin_locomo_evonly --events-ledger data/locomo_evonly.ledger`
  → `runs/m50c_locomo_events`. That is m = 3 events and a 512-token block
  budget.
- Two shards on the 9B, merged and closed.
- Judged by the 9B with `judge --seed runs/m19_locomo_full`, which reuses a
  verdict only on a byte-identical answer.

**Comparisons, paired over 1,986:**
1. **To ship:** against `runs/m19_locomo_full` (69.87). The bar is +3.0 on
   judge 1–4 with the 95% CI excluding zero. **Veto:** adversarial below
   69.96.
2. **Attribution:** against `runs/m50b_locomo_events` (events in the same
   store, −0.13). This is the dual index against the combined one.

**Predictions.**
- Judge 1–4 **+1 to +3**.
- Declines on 1–4 fall from 121 to about 90: M50b rescued +11.6 on the
  121, and here the turns are kept too.
- Temporal gains the most, because the events carry resolved dates.
- Gold turns held are unchanged, by construction.

**Diagnostics reported either way:**
- how often the six base items match m19's, as a drift check on the
  base retrieval;
- rows with at least one event appended;
- block tokens per row.

**Falsifier.** A clear loss on questions the base answered (base-answered
stratum down more than 2): the extra block distracts the 9B.

---

## Pilot result *(2026-09-24 19:07)* — **vetoed: adversarial 62.11**

`runs/m50c_locomo_events`, 1,986 rows. It was judged by the 9B unseeded,
because m19's verdicts predate the answers map `--seed` needs.

| paired over 1,986 | m19 | M50c | Δ | 95% CI |
|---|---|---|---|---|
| judge 1–4 | 69.87 | 69.87 | +0.00 | [−1.56, +1.56] |
| adversarial | 69.96 | **62.11** | **−7.85** | [−11.43, −4.26] |
| declines on 1–4 | 121 | 112 | | |

- **The veto fires.** The pre-registered falsifier's twin is what
  happened: the dated events make the 9B answer the traps.
- The temporal gain seen early (temporal scorer +11 of 180) does not
  survive into judge 1–4 overall.
- The drift check: the base items matched m19 on ~75% of rows, so part of
  any delta against m19 is base drift. M63's fresh base (below, 70.52)
  measures that drift at +0.65 over m19.

**Verdict:** events beside turns do not help this reader on LoCoMo, and
they cost the adversarial rows. Chronos's own ablation predicted a smaller
gain for a stronger reader; here the reader is the weak one, and the gain
still did not come.

---

## Correction *(2026-09-26)*: every event carried a second, wrong date

The round-4 code review found that compose re-resolved each event's quoted
phrase against the event's own, already-resolved date. The M50 pilot shows it
(`runs/m50_pilot_s1`):

```
got the Air Fryer [2023-05-20 — "yesterday", said 2023-05-21] … (yesterday = 2023-05-19)
```

So every relative-dated event in M50, M50b and this pilot reached the reader
with two dates that disagree by the phrase's offset. The adversarial veto
above was measured with that defect in place. The measurement stands as a
record of what ran; it is not a measurement of events with correct dates.

The events index now composes with `resolve_relative` off
(`events_block::events_retrieve_config`). The test
`an_event_reaches_the_reader_with_exactly_one_date` fails on the old config
and passes on the new one.
