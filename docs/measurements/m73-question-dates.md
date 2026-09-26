# M73 — the question's own date phrase *(retrieval measured first; 2026-09-26)*

## Why

The shipped LongMemEval_S run (`runs/m57_bonsai_premise_s1`) loses 30
temporal-reasoning questions under LongMemEval's own grader (M70). **11 of
the 30 never had a gold turn in their evidence.** Most of those ask about a
day the question names relative to itself:

- "What kitchen appliance did I buy **10 days ago**?"
- "Who did I go with to the music event **last Saturday**?"
- "What did I do with Rachel on the Wednesday **two months ago**?"

myelin resolves relative dates inside stored turns (M19, M64). It has never
resolved them in the **question**, so retrieval reads "10 days ago" as three
more words.

LongMemEval's authors measured the fix on their own benchmark: *time-aware
query expansion*. A model turns the question into a date range and retrieval
is narrowed to it (Wu et al. 2024, arXiv 2410.10813, §5.4, Table 4).
- With GPT-4o extracting the range, round-level recall@10 on the temporal
  subset rose from 0.550 to 0.722.
- With Llama 3.1 8B extracting it, the gain vanished. The small model
  "struggles to generate accurate time ranges, often hallucinating or
  missing temporal cues".

TempAgent reports the same for temporal KGQA: removing its time-filtered
retrieval tools "led to a marked drop" (Hu et al. 2025,
`10.18653/v1/2025.findings-naacl.334`).

myelin needs no model for the range. `time::resolve_relative` is M19's
closed grammar, and it fails closed: a phrase outside it resolves to
nothing, never to a guess.

## The instrument

Run `resolve_relative` on each question's text, anchored at its
`question_date`. Then read three things from the shipped store
(`data/longmemeval_s.ledger`) and the shipped run:
1. Is a gold session dated inside the resolved window?
2. How many records does the window hold?
3. Where does the gold turn rank inside the window?
   - (a) among every in-window record, under the cross-encoder, against
     the question;
   - (b) among the in-window items of the question's content-ranked
     recall pool (k = 50, the shipped retrieval config).

No reader is involved. The reranker (bge-reranker-v2-m3 Q8_0) ran on big
beside the M54 controller, under myelin's lease.

## Stage 0 — turns *(measured 2026-09-26)*

**Coverage.** 38 of the 500 questions carry a phrase the grammar resolves.

| phrase kind | n | gold session inside the window |
|---|---|---|
| a past point ("10 days ago", "last Saturday", "past weekend") | 18 | **13** (16 within ±3 days) |
| a period ("last month", "this year", "last week") | 15 | 6 |
| the future ("this weekend", "next week") | 5 | 0 |

The period phrases are read two ways by the corpus. Of the 9 "last month"
and "past month" questions, **1** has its gold inside the calendar month;
the rest mean "the past 30 days". A calendar window would point those
questions at the wrong month. Only past-point phrases are candidates.

**The target.** In **8 lost questions**, the window holds the gold session
while the shipped evidence holds none of it. All 8 are temporal-reasoning
questions. A ninth (`71017277`, "last Saturday") held its gold and declined.

**The gold does not rank inside the window.**

| question | phrase | records in window | gold's rank among them (a) | gold's rank among the pool's in-window items (b) |
|---|---|---|---|---|
| `gpt4_d6585ce9` | last Saturday | 190 | 3 | 11 |
| `gpt4_1e4a8aec` | two weeks ago | 50 | 20 | 11 |
| `gpt4_e414231f` | past weekend | 99 | 23 | 4 |
| `eac54add` | four weeks ago | 112 | 90 | 13 |
| `4dfccbf8` | two months ago | 53 | 6 | 4 |
| `0bc8ad93` | two months ago | 74 | 23 | 7 |
| `6e984302` | four weeks ago | 19 | 15 | 2 |
| `gpt4_8279ba03` | 10 days ago | 99 | 36 | 13 |

- **Promoting the top 2 in-window pool items would recover 1 of the 8.**
  The top 4 would recover 3. That is at most +0.6 overall, before the
  reader.
- **Why:** LongMemEval plants its facts as asides. The gold turn for "What
  kitchen appliance did I buy 10 days ago?" is about BBQ sauce, and says
  "By the way, I just got a smoker today". The cross-encoder scores it
  −10.6 against the question. The window's best record, a waste-reduction
  reply, scores −6.2. A day window holds several sessions (19 to 190
  records), and the aside is one sentence in one of them.

**Verdict for turns: no build.** The window is the right filter, and the
turn is the wrong unit to rank inside it.

## Stage 0b — events *(measured 2026-09-26)*

The unit that fits is the event. M50 extracts one short record per thing
the user did, dated by the same closed grammar. For example: "the user
started ukulele lessons with friend Rachel [2023-02-01, 'today']". That
follows Chronos's events calendar (Sen et al., `10.48550/arxiv.2603.16862`)
and LongMemEval's own timestamped-event index (Wu et al. 2024, Fig. 12).

**Setup.**
- Events existed for M50's 100 pilot haystacks only; two of the targets
  were among them.
- For the other eight haystacks (six of the 8 targets, plus `71017277` and
  `gpt4_59149c78`), the events were extracted with M50's extractor
  unchanged:
  - Qwen3.5-9B UD-Q4_K_XL, M50's prompt and schema;
  - 455 sessions on `bmb` while big ran M54, 24.7 s per session;
  - one failed session was retried before the build, which refuses a
    partial cache.
- They were written into the events store copy
  (`myelin_longmemeval_s_events`) as 583 records.
- Each question's in-window events were ranked by the same cross-encoder,
  against the question.

**Result.** For each of the 10 target questions, the top-ranked event in
the window:

| question | events in window | top event | states the answer? |
|---|---|---|---|
| `gpt4_8279ba03` kitchen appliance, 10 days ago | 17 | "the user got a smoker" | **yes** |
| `gpt4_1e4a8aec` gardening, two weeks ago | 12 | "the user planted 12 new tomato saplings" | **yes** |
| `eac54add` business milestone, four weeks ago | 24 | "the user signed first client" | **yes** |
| `4dfccbf8` with Rachel, two months ago | 9 | "the user started ukulele lessons with friend Rachel" | **yes** |
| `6e984302` competition investment, four weeks ago | 1 | "the user got a set of sculpting tools …" | **yes** |
| `gpt4_e414231f` bike, past weekend | 10 | "the user upgraded Shimano Ultegra pedals" (alias: "replaced road bike pedals") | **yes** |
| `71017277` jewelry, last Saturday | 17 | "the user got crystal chandelier" (alias: "got chandelier from aunt"; gold: "my aunt") | **yes** |
| `0bc8ad93` museum, two months ago | 12 | "the user attended a lecture at the History Museum" | the right visit; who went is not in the event |
| `gpt4_d6585ce9` music event, last Saturday | 18 | "the user attended a music festival in Brooklyn" | no: extraction missed the concert with the user's parents |
| `gpt4_59149c78` art event, two weeks ago | 0 | none | no: the gold is 3 days before the window |

**7 of 10 top-ranked events state the answer, against 0 of 8 top-ranked
turns.**

The same aside that scored −10.6 as a turn ("By the way, I just got a
smoker today") is its own record as an event. It ranks first in its window
at −7.1.

## What it says

- **The lever is real, but it cannot ship alone.** Its ceiling here is
  about 7 temporal questions, +1.4 on the 500, before the reader.
- **The combination with events is what it points to.**
  - M50's LongMemEval_S pilot put events in the fused ranking and gained
    +4.0, below its +5 gate.
  - On LoCoMo the same design displaced gold turns and netted −0.13
    (M50b).
  - M50c, events beside turns and never displacing them, is the untested
    design. This measurement says what M50c should do when the question
    names a past day: take its events from that window.
- **It needs events for every haystack.** About 14,000 more distinct
  sessions:
  - on big after M54, roughly the ~9 GPU-hours M50 priced;
  - on bmb at the measured 24.7 s per session, about four days, so bmb
    is not the host for it.
- **That is a scope decision for the user**, recorded in `BACKLOG.md`.
  So is M50c's route, which was already waiting on the user.

## Reproduce

The analysis scripts are in the session scratchpad, not in the tree:
- `qanchor`: `resolve_relative` over every question;
- `qpool`: the k = 50 recall pool;
- `m73_window_rank.py`: turns;
- `m73_event_rank.py`: events.

Every step can be rerun from this description with the shipped binary's
`events-extract` and `events-build`, plus the `resolve_relative` function.

```
myelin-eval events-extract --corpus longmemeval-s --questions <the 10 ids> \
  --concurrency 4 --out data/events/longmemeval_s.m73.jsonl
myelin-eval events-build --corpus longmemeval-s --cache data/events/longmemeval_s.m73.jsonl \
  --questions <the 8 ids without events> \
  --collection myelin_longmemeval_s_events --ledger data/longmemeval_s_events.ledger
```
