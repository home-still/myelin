# M46 — the timeline never said how long ago

## What the rows say

The backlog framed M46 as "temporal arithmetic belongs in `compose`" and
proposed signed day-deltas between events. Before building that, the rows
were read. On the shipped M43 operating point (`runs/m43_dated_judged`,
67.80), the questions that ask for an elapsed time split like this:

| subset (regex over the question) | n | base | declines |
| --- | --- | --- | --- |
| any duration cue (`how many days … / how long …`) | 118 | 61.0 | — |
| duration *arithmetic* (`… before / after / since / ago`) | 83 | 54.2 | 19 |
| **anchored to the question's day** (`ago`, `since`, `how long have …`) | 62 | 56.5 | 14 |
| `temporal-reasoning` rows with no duration cue | 65 | 47.7 | — |

The wrong answers in the anchored subset are not composition failures; they
are arithmetic against *today*:

| question | gold | answer | what the timeline showed |
| --- | --- | --- | --- |
| How many weeks ago did I meet my aunt …? | 4 | *declined* | `2023-03-04 +19d`; question date 2023-04-01 |
| How many months have passed since I last visited a museum …? | 5 | 2 | `2022-10-22 +0d`; question date 2023-03-25 |
| How many weeks ago did I attend the Nordstrom sale? | 2 | 1 week | `2022-11-18 +26d`; question date 2022-12-01 |
| How many months ago did I attend the photography workshop? | 3 | 2 months | — |
| How many weeks had passed since I recovered from the flu when …? | 15 | 10 weeks | — |

Day-level *ago* questions mostly succeed (`26 days ago` → 26, `17` → 17).
The failures begin where a **unit conversion joins the subtraction**, and on
the relative-lookup shape (*what did I do two weeks ago?*) where the reader
has to find the event 14 days before a date it is only told in the prompt.

The `[timeline]` view M19 shipped states each event's offset from the
**earliest** event (`+19d`) and the span. It never states the offset from
the day the question is asked. The reader gets `<today>` in the prompt and is
left to subtract, then convert.

## Why the reader will not do it

M19 already measured the asymmetry on this codebase: resolving dates *for*
the reader was worth **+37.6** on LoCoMo's temporal stratum, telling it to
resolve them **+14.3**. Test of Time (Fatemi et al.,
`10.48550/arxiv.2406.09170`) measures the same shape at the frontier — GPT-4
at 16.00% on duration arithmetic, Claude-3-Sonnet 15.00, with off-by-one in a
fifth to a quarter of responses — so a 9B was never going to close this by
being asked nicely. Allen (`10.1145/182.358434`, §VII.2) is the older
statement: on a date line the relation between two dated events is a cheap
comparison that should be computed, not searched for.

## The mechanism

`Recall::as_of` carries the day the question is asked. It has **no default**:
a memory backend that substituted the wall clock would emit offsets wrong by
exactly the age of the corpus, so absence means no anchor. The bench sets it
from LongMemEval_S's `question_date` (the same value the reader sees in
`<today>`) and from LoCoMo's last session; the MCP `recall` and `investigate`
tools accept `as_of: "YYYY-MM-DD"` and **refuse** a malformed one.

`ComposeConfig::timeline_ago` (default **off**, pending this arm) anchors the
existing view:

```
[timeline] as of 2023-04-01: 2023-02-13 +0d (47 days ago; 6 weeks; 1 month) · …;
           2023-03-04 +19d (28 days ago; 4 weeks) · …  (span 19 days)
```

Every unit a question might ask in is stated, each as the floor — which is
how the corpus's gold counts them — and the day count is exclusive of both
ends, fixed once here so the off-by-one Test of Time found is never the
reader's to make. Zero model calls. The `+Nd` offsets M19 measured are
untouched, and with the switch off or no anchor the item is **byte-identical**
to the M19 form (pinned by
`the_anchored_timeline_states_each_entrys_distance_from_today`).

Chronological order is kept. Test of Time also reports fact *ordering*
moving Claude-3-Sonnet 45.71 → 73.57 when the target comes first; that is a
second mechanism and, per M31, it gets its own arm if this one moves.

## Pre-registration

Written before the arm ran.

**Base.** The shipped operating point, or M44's winner if M44 ships first —
whichever is the default when the arm runs, named in the results. As of
writing: `runs/m43_dated_judged`, **67.80**.

**Arm.** The base command plus `--timeline-ago` → `runs/m46_ago`, judged with
`--seed <base>`.

**Population.** All 500 LongMemEval_S rows.

**Primary metric.** `longmemeval_s.judge_score.n500`, paired bootstrap.

**Decision rule.** Ship on at **≥ +3.0** with a paired 95% CI excluding
zero; **veto** on any drop across the 30 abstention rows.

**Predicted, specifically.**

1. The anchored subset (n = 62, base 56.5, 14 declines) moves up, and the
   move is concentrated on the weeks/months rows and the relative-lookup
   rows; day-level *ago* rows, already right, move ~0.
2. Rows that get no `[timeline]` — every question that is not an interval
   question in `recall`, and single-record sets in `investigate` — move
   **exactly +0.0**. That is the control.
3. Abstention rows do not fall: an anchored timeline adds no claim about
   events that are not there.
4. The headline will not clear +3.0 on its own — 62 rows at, say, +15 is
   +1.9 overall. This arm is measured on its numerator and reported that
   way; if the subset moves and the control is exact, the switch ships on
   the subset's CI, which is the rule M40's stratum prediction established.

**Falsifier.** If the weeks/months rows do not move with the arithmetic on
the page, the reader is not arithmetic-limited on them; it is failing to
pick the right event, and the next arm is ordering (target first), not
computation.

**Cost.** One bench run, no extra model calls per row.

## Results

*(pending — queued behind M44 R1 on the reader)*
