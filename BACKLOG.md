# Backlog

Open work, ranked by **answers ÷ cost**. Finished items move to
[`BACKLOG_DONE.md`](BACKLOG_DONE.md) with their measured number and the commit
that carries it.

Every item ships on a bar that is **pre-registered before the arm runs**:
≥ +3.0 on the reported population with a paired 95% CI excluding zero, and
M42's **abstention veto** — any drop on the abstention stratum ships it off
whatever the headline says. Switches default **off** until a measurement says
otherwise.

Literature lives in home-still; the guidance this backlog was built from is
`docs/sota/2026-09-22-agentic-memory-sota-guidance.md`.

---

## Where we stand

See [`BACKLOG_DONE.md`](BACKLOG_DONE.md#sota-standing) for the full table and
its history. Short version: **one gate closed** (MINJA 7.50% ≤ 10%), one
claimable row (LoCoMo beats Mem0's published 66.88 by +2.99), and three open
gaps — LoCoMo −7.98, **LongMemEval_S −2.40** (M43 closed 5.80, **M44 R2
closed 10.60**: 62.00 → 67.80 → 78.40 in one day), LME-V2 −20.02 at a
pre-M43 configuration. M44 R1, M46 and M48 measured null with the veto
firing; M48 closed the negation story; M45 closed decline recovery (AUROC
0.59). Then R2 — native thinking, 1,024 tokens, two seeds at 78.40 —
moved the two-fact stratum +18 and temporal reasoning +30, with abstention
*up*. The reader was compute-limited all along.

---

## M44b — the thinking reader, tightened *(follow-ups to a shipped win)*

R2 ships at **78.40** with a 1,024-token budget and no budget message. Two
cheap arms, both against `runs/m44_r2_s1_judged` under the same seed:

- **R2b — budget message.** *Seed 1 measured 2026-09-23: 78.4 → 80.2, +1.8
  [−0.4, +4.0]; spill rows 56 → 1 and +14.3 on them; control +0.9. Misses
  the bar on one seed; seed 2 running, ship decided on the pooled two-seed
  delta.* llama.cpp's `--reasoning-budget-message` with Qwen's own wording
  closes the trace cleanly. One serve flag.
- **R2c — budget 2,048.** The same 165 rows, given room. Costs ~1.5× per
  row. Predicted: `temporal-reasoning` and gold ≥ 3 move; the preference
  stratum (−20.0 on seed 2, the one cost) does not recover — it is not a
  budget problem.
- **LoCoMo under the thinking reader.** The pinned 69.87 is a plain-reader
  `recall` run; `shipped_reader_thinking` is off for LoCoMo until this is
  measured. 1,986 rows × ~12 s ≈ 6.5 h, then the ratchet floor moves there
  too.

**Cost.** R2b/R2c ~1.5 h each; LoCoMo an overnight.

---

## M47 — presupposition verification, contradiction only

M35's `premise_analysis` tripled declines on answerable rows (8.3% → 26.8%)
and moved abstention only 1.3×, because it fired on *unsupported* and a 9B
says "unsupported" whenever the store is merely silent. LME-V2's abstention
rows are **wrong-premise** questions; silence is not a false premise,
contradiction is.

Kim et al. (`2101.00391`) give the pipeline — presupposition generation from
linguistic triggers, verification, explanation — and ~21% of Natural
Questions' unanswerable items are explained by unverifiable presuppositions.
(QA)² (`2212.10003`) and FalseQA (`2307.02394`) show models *hold* the
knowledge to rebut false premises but need the rebuttal step activated; we
cannot fine-tune, so the activation must be structural.

Schema: `{claim, status: supported|contradicted|absent, evidence_index}`,
`minItems ≥ 1` (M40's forcing), `status` after `claim` (M42/M43's ordering).
Emit a `[premise]` line **only** on `contradicted`; emit **nothing** on
`absent`. That one line is the whole difference from M35, and it makes M35's
damage unreachable by construction.

**Predicted.** LME-V2 abstention +10 or better from 17.97%; answerable −1.0 or
better. **Falsifier:** if true premises get marked `contradicted` often enough
to cost answerable rows, verification is the bottleneck exactly as Kim et al.
found, and an NLI model should do it instead of the 9B.

**Cost.** ~1 hour.

**Status.** *Implemented and pre-registered* —
`docs/measurements/m47-presupposition-contradiction.md`.
`InvestigateConfig::premise_check` (off), `--premise-check` on `bench` and
`run_myelin.py`, `premise_check` on the MCP `investigate` tool. Field order
`claim → evidence_index → status`; only a contradiction naming a real memory
emits the `[premise]` item; the view carries the weakest trust it cites. The
arm needs an LME-V2 base at the shipped defaults first (none exists since
M43 flipped the digest on), so it is two LME-V2 pairs, queued behind M44 R1
and M46 on the reader.

---

## M49 — REPLAY and supersedes routing, as read-path views

JustMem (`2609.19877`) reports three access modes, of which **REPLAY**
recovers the original conversation "for fidelity-sensitive evidence" — which
is M41's finding in one word: compressed evidence loses the recency signal.
RD-Forget (`2609.10263`) "separates what an agent stores from what it uses",
routing current-state questions to newest-in-slot and historical ones to the
full archive.

We already own both halves: `prov_source.doc` links records to sessions, and
`supersedes` edges exist. So this is a compose-time view, **not a re-ingest**.

**Cost.** Minutes. **Numerator.** `knowledge-update` (78) and temporal.

> JustMem's full text was rate-limited; §6 of the guidance rests on the
> abstract. Read it via `markdown_read` once home-still's scribe is back up.

**Checked 2026-09-22 (measured):** the `link` table of
`data/longmemeval_s.ledger` is **empty** — 0 rows over 162,181 records — so
there are no `supersedes` edges to route on for this corpus; the write path's
`update` verdict never minted one here. On the M43 base `knowledge-update`
delivers every gold session on 65 of 72 annotated rows and still scores 0.800
on them (the reader answers the *older* value with the newer one in hand:
three sessions for five, 1250 followers for 1300, 8:30 for 7:30). The slot
signal has to come from somewhere other than links — same-tenant records
ordered by date with a "latest" marker, or a write-time slot label — before
this arm can be built.

---

## M50 — one `build` with Chronos-style event tuples

Chronos (`2603.16862`) attributes **58.9%** of its gain to the events
calendar. Unknown at 9B, frontier-backbone result. The only write-path arm
worth a GPU window, and worth **one** re-ingest, not three — REALM's
reconsolidation (`2609.16053`) is +1.31 over Zep on a GPT-4o-mini backbone and
our graph channel already measured as a loss in M12.

**Cost — recomputed 2026-09-22.** "57 minutes" was `reindex` throughput
(embed-only). An events pass is one reader call per *session*: LongMemEval_S
is ~25,000 sessions → **14–28 GPU-hours** on one slot. The scope is the
user's call; `docs/measurements/m50-events-calendar.md` lays out three:
(a) a 100-haystack pilot (~5,000 calls, 3–6 h, n = 100, the other 400 as
control), (b) the full corpus overnight (the only scope that can move the
standing row), (c) LoCoMo first (~250 calls, under an hour, 1,986-question
arm). The plan reuses `build --pools`' shape — one call per unit, strict
schema, `Semantic` records with `prov_derived_from` lineage — and M19's
`resolve_relative` for the datetime ranges.

---

## Do not re-run

`typed_probes`; `premise_analysis` as built (M35, **−8.75**); `select_coverage`
(M38, 500/500 byte-identical, verified at the wire); `answerability_gate`
(M36, zero `supported` verdicts on the pilot); trajectory routing (M38, worse
than flat); entity-extraction repair (`entity_ids` is written, indexed, and
never read); one-step `self_ask` (M39, **−4.6**); MMR or any model-free
diversity term over the reranked pool (M21, gold recall **0.658 → 0.550** —
co-evidence resembles itself 1.60× more than the rest of the set, so every
such term is aimed squarely at the answer); widening (M37, three nulls).

Do not revise the adjudicator prompt (M15/M23). Do not re-run decline
recovery on this reader in any form — forced schema (M42), reasoning field
(M44 R1) and sampled agreement (M45) are measured; the declines are not a
confidence problem.

---

## Operational debt

- **Qdrant is shared and unprotected.** All five `myelin_*` collections were
  deleted through the dashboard mid-run on 2026-09-22. M41's `reindex` is the
  recovery; the prevention is `QDRANT__SERVICE__API_KEY` +
  `QDRANT__SERVICE__READ_ONLY_API_KEY`, or our own instance on a separate
  port. Snapshot after every `build` either way.
- **The LME-V2 index was 95% missing until 2026-09-22.** M41's `reindex`
  recovered `myelin_longmemeval_s` (162,181 of 162,181) and nothing else:
  `myelin_lme_v2_small` held **4,608 points against 85,979 live records** —
  every LME-V2 read since the deletion incident, including the standing
  row's own candidates, ran against a twentieth of the store. Found while
  snapshotting; `reindex --corpus lme-v2-small` rebuilt it (ids preserved).
  `myelin_locomo` (4,875) and `myelin_lme_s_pref` (13,536) were verified
  complete against their ledgers' *live* counts — the `record` table's total
  (5,040 / 13,687) includes rows `count_live` excludes, so compare against
  the reindex header, not `select count(*)`. Snapshots of all four
  collections now exist on `big` (`…-2026-09-22-19-15-*.snapshot`); a
  Qdrant point count should be checked against the ledger's live count
  before any arm is launched.
- **GPU tenancy on `big`.** M44 R2 and M45's N-sample decode need more of the
  card than any arm so far. No-sudo levers: `-ctk q8_0 -ctv q8_0` with
  `-fa on`; two server profiles (LongMemEval_S at k=6/4,096 plus a 1,024-token
  thinking budget fits 8k per slot; only LME-V2 needs the large window); `n`
  parallel samples off one prefill for M45.
- **Two arms sharing the reader's slots do not get an exact control.**
  M48 ran beside M44 R2 for its whole length; llama.cpp batches concurrent
  slots, and the same greedy request can decode a different token when its
  batch-mate changes. Result: 375/500 byte-identical (M46 alone: 429), and
  the 28 rows with no note in either run moved +3.6 instead of +0.0. Either
  run one arm at a time, or take untouched rows from the base by
  construction as `commit-arm` does. The throughput gain of co-running was
  real (both arms progressed); the exact control was the price.
- **`investigate` never narrows `[timeline]` to interval questions.** Only
  `recall` applies `is_interval_question`; `investigate` composes with the
  view on every row. Every judged number since M19 was measured that way, so
  it is the operating point, not a bug to fix silently — but the switch that
  was meant to be question-conditioned is unconditional on the path that
  scores everything, and M46's control (rows with no timeline) did not exist
  because of it. Decide whether to narrow it (an arm, since it changes the
  evidence on ~380 rows) or document it as the default.
- **The reader can be killed from outside mid-arm, and `bench` did not
  resume.** 2026-09-22 15:0x: `/tmp/myelin-reader.log` on `big` ended with
  *"Received second interrupt, terminating immediately"* — two SIGINTs from
  something other than this loop, `tenant=none` throughout — 249 rows into
  the M46 arm. `run-with-reader.sh` re-served the reader (1 slot, 16k) and
  retried, and `bench` truncated `per_question.jsonl` and started over.
  `bench --resume` now keeps the finished rows (recorded as `resumed_rows`
  on the run) and the wrapper passes it on every retry. Who sends the
  interrupts is unknown; `serve-models.sh` is the only thing on this side
  that retires the port, and it was not run.
- **A partially-offloaded reader is a different measurement, not a slower
  one.** When the card is full the reader falls back to CPU layers and
  throughput drops ~5× (2.9 → 14.5 s/row). Such runs should be recorded
  `Degraded`, as `WidthVerdict` already does.
- **home-still scribe is down**, so the 22 papers catalogued on 2026-09-22 are
  not yet converted or indexed; `distill_search` will not surface them.
