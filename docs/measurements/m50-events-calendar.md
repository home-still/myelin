# M50 — one `build` with Chronos-style event tuples *(plan — not yet pre-registered)*

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

## Pre-registration

*(written once the scope is chosen; bar +3.0 with the CI excluding zero,
abstention veto, coverage reported before judging)*
