# M50 — one `build` with Chronos-style event tuples *(implemented and pre-registered 2026-09-23)*

## Why this is the only write-path arm worth a GPU window

Every read-path lever this project has pulled since M32 has now been
measured, and the ledger is unambiguous:

| lever | milestone | result |
| --- | --- | --- |
| select what is sufficient | M32 | **+5.8**, shipped |
| digest every memory, dated | M40/M41/M43 | **+5.8**, shipped |
| filter or type the digest | M43 B, M48 | −4.4, +0.0 |
| let the reader reason / think | M44 R1 (R2 pending) | −0.8, veto |
| force, or agree on, a commit | M42, M45 | +2.2 veto, −0.4 veto |
| anchor the timeline | M46 | −0.4, veto |

The reader is at **67.80** with retrieval solved (M38: every gold session
delivered 88.6% of the time) and every attempt to make it *use* what it
holds either null or vetoed. The remaining gap to MemPro's 80.80 is not
in the reader's confidence, its labels or its arithmetic; it is in what
the memory hands it. That is the write path, and it has not been touched
since M20.

Chronos (Sen et al., `10.48550/arxiv.2603.16862`) is the strongest
published result on exactly this benchmark: **92.60 (Low) / 95.60 (High)**
on LongMemEval_S, +7.67 over the best prior system, and its ablation
attributes **58.9% of the gain to the events calendar** — raw dialogue
decomposed into *subject–verb–object event tuples with resolved datetime
ranges and entity aliases*, indexed beside a turn calendar that keeps the
full conversational context. Every other component (dynamic retrieval
guidance, the tool-calling loop) is worth 15.5–22.3%. The backbones are
frontier models; the number is not ours to match, the *allocation* is the
finding.

REALM (`2609.16053`) is the caution: reconsolidation of memories at write
time is +1.31 over Zep on a GPT-4o-mini backbone, and our own graph channel
measured as a loss in M12. One re-ingest, with the one mechanism the
ablation says matters, not three.

## What already exists

`myelin-eval build --pools` (M23 D1, M34) is the template: a second pass
over a unit's episodic records that calls the reader once per unit,
parses a strict schema, and writes `RecordKind::Semantic` records whose
`prov_derived_from` names the episodic records they abstract — the
lineage the I4 invariant requires and M34 fixed. It exists only for
LME-V2's trajectories (UI transitions); LongMemEval_S's episodic store is
raw turns, 162,181 of them, written with **no LLM at ingest**.

`time::resolve_relative` (M19) already turns "last Tuesday" into a date
against a record's `t_valid`; the same resolver gives an event tuple its
datetime range. `ComposeConfig::kind_quota` (M35, implemented, unmeasured)
already reserves slots per kind at compose time, AgentRunbook-R's
allocation; `select_sufficient` and the digest treat any record kind alike.

## The mechanism, as proposed

A `build --events` pass for conversational corpora:

1. **Per session** (not per turn), one call: *list the events this session
   establishes* as `{subject, verb, object, when}` with `when` a phrase the
   session states (`last Tuesday`, `next month`, `2023-05-06`) or empty.
   `minItems: 0`, `maxItems: N` — the count is forced by the schema only at
   the top; a session with no events is allowed to say so.
2. **Resolve `when`** against the session's date with `resolve_relative`;
   an unresolvable phrase leaves `t_valid` at the session date and the
   phrase in the text.
3. **Write each tuple as a `Semantic` record**: text
   `"{subject} {verb} {object}"` (with the date range appended when it is
   a range), `t_valid` the resolved date, `prov_derived_from` the session's
   episodic records, same tenant and namespace. Same collection; the
   dense, BM25 and rerank channels see them beside the turns.
4. **Read path unchanged.** Events compete in the fused ranking and pass
   through selection and the digest like any record. `kind_quota` is a
   separate, later arm.

Falsifiable at the retrieval layer before any judged number: M38's
coverage instrument reports whether the gold *sessions* are still
delivered, and a new column reports how many of the k slots events take.

## The cost, recomputed

The backlog priced this at "one ~57-minute re-ingest". That figure is
`reindex` throughput (embed-only, 38 records/s). An events pass is one
reader call per session: LongMemEval_S has 500 haystacks × ~50 sessions =
**~25,000 calls** at 2–4 s each — **14–28 GPU-hours** on one slot, half
that on two. The LME-V2 pools pass (M34) covered 190 trajectories.

So the scope is a decision, not a default, and it is the user's:

- **(a) Pilot on a question subset.** Build events for 100 of the 500
  haystacks (~5,000 calls, 3–6 h), pre-register on that n = 100 with the
  same bar. A +10 effect shows at n = 100; a +3 does not. The 400 untouched
  haystacks are the control.
- **(b) Full corpus, overnight.** ~25,000 calls, 14–28 h, then the arm at
  n = 500 against the shipped base. The only version that can move the
  standing row.
- **(c) LoCoMo first.** 10 conversations × ~25 sessions = ~250 calls, under
  half an hour, then the 1,986-question arm. Cheap, and the M43 read-path
  gain has never been measured on LoCoMo either — but Chronos's number is
  a LongMemEval_S number, and LoCoMo's temporal stratum is where M19
  already resolved dates.

Whichever scope, the pass is resumable per session (the `unit_complete`
audit event, as `build` already does) and ships with its `reindex` path.

## What was built (2026-09-23)

Two passes, so the expensive half runs where a reader is free and the cheap
half runs anywhere:

- **`myelin-eval events-extract`** — one schema-constrained call per
  *distinct* session (Chronos's 25-turn windows with a 5-turn overlap),
  written to a JSONL cache keyed by the SHA-256 of the session text. It
  touches no store, resumes from the cache, and `--shard i/n` splits a
  corpus across hosts. `myelin_core::pipeline::events` holds the prompt,
  schema and resolution.
- **`myelin-eval events-build`** — resolves each event's `when` against the
  date of *that copy* of the session and writes one `Semantic` record per
  event into a **copy** of the store, derived from the session's episodic
  records (I4). No reader.

Three decisions, each measured before it was made:

| | measured | decision |
| --- | --- | --- |
| who does date arithmetic | M46: the reader subtracts badly at answer time | the model copies `when` verbatim; M19's closed grammar resolves it, failing closed |
| cache key | LongMemEval_S: 25,112 session slots, **18,821 distinct contents**, 5,283 slots re-dated | extract date-free, once per content; resolve per copy |
| the empty list | first prompt ended on the literal `{"events": []}` → **16 of 19** LoCoMo conv-26 sessions returned nothing, including "I went to a LGBTQ support group yesterday" | the literal is gone; the same sessions give **70 events, one empty**, every `when` verbatim |

Stores: `myelin_longmemeval_s_events` and `myelin_locomo_events` are Qdrant
snapshot restores of the shipped collections (5 s and 2 s, against ~57 min
to re-embed), with ledgers copied to `data/*_events.ledger`. `events-build`
refuses the shipped collection and ledger by name.

## Scope, decided 2026-09-23

The user's instruction was to bundle what is validated elsewhere, keep one
arm per new mechanism, and keep the GPU busy. So:

1. **LongMemEval_S pilot, n = 100** — the gate. Chronos is a LongMemEval_S
   result, and on LoCoMo the store already holds 4,169 extracted facts, so
   the mechanism is least distinct there. The population is
   [`m50-pilot-questions.txt`](m50-pilot-questions.txt): every 5th question
   by file index, proportional across all six types (14 / 27 / 6 / 27 / 15 /
   11; 6 abstention rows). `--limit 100` would have been 70
   `single-session-user` and 30 `multi-session` with no temporal row at all.
2. **Extraction where the reader is free.** 4,765 distinct sessions, eight
   shards: `bmb` takes shard 7 while `big` runs the M47 pair alone, `big`
   takes shards 0–6 after it. bmb decodes ~10 tok/s per slot (4 slots) and
   prefills slowly, so it carries an eighth; every extraction process runs
   on the workstation and writes a local cache, so whichever host finishes
   first resumes the other's shard where it stopped.
3. **LoCoMo** extracts in full on bmb (272 sessions, ~50 min) and its arm
   waits for M51, whose operating point it will run at.

## Pre-registration — LongMemEval_S pilot

**Base.** `runs/m44_r2_s1_judged` restricted to the 100 pilot ids: **70.00**
(full 500: 78.40).

| stratum | n | base |
| --- | --- | --- |
| single-session-user | 14 | 100.00 |
| multi-session | 27 | 48.15 |
| single-session-preference | 6 | 0.00 |
| temporal-reasoning | 27 | 81.48 |
| knowledge-update | 15 | 66.67 |
| single-session-assistant | 11 | 100.00 |
| abstention (`_abs`) | 6 | 83.33 |

**Arm.** The shipped command, seed 1, on the events store:

```
myelin-eval events-extract --corpus longmemeval-s --questions docs/measurements/m50-pilot-questions.txt --shard <i>/8 --out data/events/longmemeval_s.s<i>.jsonl
myelin-eval events-build --corpus longmemeval-s --questions docs/measurements/m50-pilot-questions.txt \
  --cache data/events/longmemeval_s.s0.jsonl,…,data/events/longmemeval_s.s7.jsonl \
  --collection myelin_longmemeval_s_events --ledger data/longmemeval_s_events.ledger
myelin-eval bench --corpus longmemeval-s --mode investigate --k 6 --budget-tokens 4096 --max-steps 2 \
  --select-sufficient --item-digest --digest-dates --reader-thinking --reader-seed 1 \
  --questions docs/measurements/m50-pilot-questions.txt \
  --collection myelin_longmemeval_s_events --ledger data/longmemeval_s_events.ledger --out runs/m50_pilot_s1
myelin-eval judge --run runs/m50_pilot_s1 --seed runs/m44_r2_s1
```

Run **alone** on `big`, reader served as for `m44_r2_s1` (2 slots, 32k,
no projector, 1,024-token thinking budget).

**Amended before any arm ran (2026-09-23 06:20):** `m44_r2_s1` co-ran beside
M48 for its whole length (operational debt: *two arms sharing the reader's
slots do not get an exact control*), so its rows cannot be reproduced
byte-for-byte. The pilot therefore re-runs its own base first — the same
command on the shipped store with `--out runs/m50_pilot_base_s1`, alone,
~25 minutes — and pairs the arm against **that**. A row whose composed
evidence holds no event must then come back byte-identical: the pilot's
exact control, reported as a count. The delta between the re-run and
`m44_r2_s1` on the same 100 rows is reported too; it is the co-running
noise the M48 note warned about, measured.

**Schedule amended 2026-09-23 06:35.** The extraction co-runs with M47, M51
and M52 on an 8-slot unified-KV reader (a build pass needs no control). The
two pilot *runs* keep their exact control: base and arm run alone, back to
back, on one serve configuration, after everything else has finished.

**Extraction moved to `bmb` (2026-09-23 06:58).** Measured: concurrency buys
only ~1.3× on this reader, so the ~6 GPU-hours of pilot extraction would
push M47, M51 and M52 past the evening on `big`. `bmb` extracts all eight
shards (~12 s/session, 4 slots) while `big` runs those three; the pilot
runs when the cache is complete. The extractor's backend does not enter the
comparison: base and arm read the same built store.

**Reported before judging.** Rows with ≥ 1 event in the composed evidence;
mean event slots per row; events written, and how many `when`s resolved,
stayed verbatim, or were empty.

**Predictions.**

- Pilot overall: **+5 or better.** Chronos Low lost 34.5 without events
  (GPT-4o), Chronos High 2.6 (Opus); a 9B sits nearer the first.
- multi-session (27): **+10 or better** — aggregation over events is the
  cross-session count Chronos reports at 91.73.
- temporal-reasoning (27): **+5 or better** — resolved dates, and the gain is
  capped by the base's 81.48.
- knowledge-update (15): **no prediction** — M49's finding (the reader picks
  the older value with the newer one in hand) is not an events problem.
- single-session-user / -assistant: **no drop** (already 100.00).
- abstention (6): **the veto** — any drop fires it. Events state what
  happened; they should not invent an answer to a question about what did not.

**Gate, not ship.** n = 100 resolves only effects near ±9, so the pilot
decides whether to spend ~9 more GPU-hours extracting the other 14,056
distinct sessions, not whether the switch ships:

- point estimate ≥ +5 and abstention not lower → **full extraction**,
  co-running with M51 (whose comparison has no exact control to lose), then
  the n = 500 arm on the ship bar: +3.0 with the paired CI excluding zero
  and the abstention veto.
- point estimate in [0, +5) → stop; record; LoCoMo arm only.
- negative → stop; the falsifier below decides what is recorded.

**Falsifier.** Events crowd gold turns out of the k = 6 slots: rows whose
base evidence held every gold session and whose arm evidence does not are
counted (M38's coverage instrument), and a loss concentrated there means the
events belong in a separate quota (`kind_quota`, M35), not in the fused
ranking.

## Results

*(pending)*
