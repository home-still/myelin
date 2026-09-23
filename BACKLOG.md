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
gaps — LoCoMo −7.98, LongMemEval_S **−13.00** (M43 closed 5.80 of it),
LME-V2 −20.02 at a pre-M43 configuration. M44 R1, M46 and M48 all measured
null with the abstention veto firing; M48 closed the negation story M42
opened, and M45 closed decline recovery itself — a forced schema, room to
reason and agreement across samples all convert this reader's declines at
30–42% while surrendering adversarial rows (AUROC 0.59 for agreement).
The reader does not have those answers; the remaining levers are M47's
contradiction check for the adversarial half and write-time composition
(M50) for the answerable half.

---

## M44 — the reader has never been allowed to reason

**The finding.** `with_thinking(true)` exists in `llm/mod.rs`, is unit tested,
and has **no production call site**. The server pins
`{"enable_thinking":false}`. Every reader call is capped at 160 tokens and
`READER_SYSTEM` says *"Answer in as few words as possible … Do not explain."*
Meanwhile `vendor/longmemeval-v2/evaluation/harness.py:186` sets
`reader_enable_thinking=True` **by default**, and our own
`adapters/run_myelin.py:201` overrides it to `False`.

Two consequences:

1. **The standing row is wrong.** AgentRunbook-R's 58.60 came from a thinking
   Qwen3.5-9B with a 20,000-token budget; our 38.58 from the same weights with
   thinking off and 160 tokens. It is labelled "same reader" and is not one.
   It must carry `caveat-reader-mode` until an equal-configuration number
   exists.
2. **Every diagnosis since M38 was taken under that configuration.** The
   2-fact collapse (79.9 → 56.7 → 40.0), the ignored instructions, the 63
   false declines — all textbook behaviour for a small model denied reasoning
   tokens. Tam et al. (`10.18653/v1/2024.emnlp-industry.91`) measured it: JSON
   mode put the answer key before the reason key in **100%** of responses,
   producing direct answering instead of chain-of-thought, and LLaMA-3-8B
   loses **38.15%** on Last Letter. `READER_SYSTEM` is that failure mode with
   no reason field at all.

**Arms.**

- **R1 — structured reasoning, thinking off.** `{reasoning, answer,
  evidence_absent}`, field order is the mechanism (third application of M42's
  and M43's rule). Ceiling 160 → 480. Temperature 0 stays, isolating "let it
  reason" from "sample". **Measured 2026-09-22: −0.8 [−3.8, +2.2], null;
  abstention 90.0 → 43.3, veto fires; gold=2 +3.1 [−0.4, +7.0];
  `multi-session` exactly +0.0. Ships off.** 273 byte-identical rows, control
  exact. On the 49 answerable rows the base declined it answers 27 and is
  right on 11 (40.7%) — M42's conversion rate to the point. Seven of the 14
  lost abstention rows turn absence into `0` / `Never` / `Nothing`.
- **R2 — thinking on.** `enable_thinking: true`, bounded thinking budget
  (1,024 tokens first), temp 0.6 / top_p 0.95 / top_k 20 per the Qwen3
  Technical Report (`2505.09388`), two seeds so the CI carries sampling noise.
  *Implemented (`--reader-thinking --reader-seed`, PR #26); the run records
  the trace per row. Needs the reader restarted with
  `MYELIN_READER_THINK_BUDGET=1024`; next on the GPU after M46.*
- **LME-V2 corollary.** Re-run the shipped operating point at the harness's
  own default. Not a mechanism arm — it is the *comparable* number.

**Predicted.** gold=2 (n=217, 56.7%) and gold≥3 (n=31, 35.5%) move; gold=1
(n=169, 79.9%) does not regress by more than 1.0; `temporal-reasoning` and
`multi-session` carry the gain; abstention on the 30 `_abs` rows does not
fall. If R2 ≫ R1 the gain is deliberation; if R1 ≈ R2 the cheap ship is the
bounded field.

**Falsifier.** If neither arm moves gold=2, the reader is not compute-limited,
M38's "reading is the gap" theory is refuted at the cheapest possible point,
and the project redirects to write-time aggregation with far more confidence.

**Cost.** R1 minutes. R2 2–6 GPU-hours. LME-V2 hours.

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

**Cost.** One ~57-minute re-ingest plus arms.

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
