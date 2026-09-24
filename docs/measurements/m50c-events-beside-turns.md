# M50c — events beside turns, not instead *(design; route not yet chosen)*

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
