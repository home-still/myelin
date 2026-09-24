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
undated configuration: **38.80** (web 43.75, enterprise 33.18), measured 2026-09-23 at today's memory defaults on the rebuilt store. M52 tested the harness's thinking reader on it: at
a 1,024-token budget thinking *loses* on LME-V2 (web −2.08, abstention
−9.72), so the gap is not our reader declining to think. M44 R1, M46 and M48 measured null with the veto
firing; M48 closed the negation story; M45 closed decline recovery (AUROC
0.59). Then R2 — native thinking, 1,024 tokens, two seeds at 78.40 —
moved the two-fact stratum +18 and temporal reasoning +30, with abstention
*up*. The reader was compute-limited all along.

---

## M55 — Ternary Bonsai 2 27B as myelin's model *(LongMemEval_S +3.8 vetoed; LoCoMo −3.18; the lever is when to answer)*

Every open gap points at the 9B: thinking was worth +10.6 on LongMemEval_S
(compute-limited), LME-V2 misreads 36% of the answers it is handed, and
LoCoMo's declines rose 121 → 197 at the shipped settings. The comparable SOTA
row answers with Qwen3-30B-A3B. Bonsai 27B (ternary, 5.95 GB, local) is on
big today. Plumbing merged first: bench records the served model
(`llm_served_model`), `standing` treats a non-9B run as an arm, and
`serve-models.sh` serves Bonsai with `MYELIN_READER_MODEL=bonsai-27b`.
The pilot on the M50 100-question population measured **69.0 → 79.0,
+10.0 [+3.0, +17.0]**, past its +5 gate: +36.8 on the 9B's declines, +20.5
on two-session questions, at 1.4× the time per row. The full 500 measured **78.40 → 82.20, +3.8 [+1.2, +6.6]**:
+4.7 on answerable. But abstention fell from 28/30 to 25/30, so the veto
fired. Two of the lost rows are premise corrections ("You see Dr. Smith, not
Dr. Johnson") that the string-rule abstention scorer cannot see. One is a
real false-premise answer. LoCoMo on Bonsai (M55b) measured **−3.18
[−4.87, −1.49]** with the same evidence. Declines on answerable rows went
121 → 292 (118 with the gold turn in hand) and adversarial went +22.87. On
both benchmarks Bonsai knows more; what is left is calibrating when it
answers. The fix for the next arm is waiting on the user's choice
(`docs/measurements/m55-bonsai-27b-model.md`).

---

## M56 — an external checker on when to answer *(Step 0 failed its gate; recorded)*

Jev (TypeSafe's cloud decision model) replayed over existing answers:
LongMemEval_S 82.20 → 59.80 simulated, LoCoMo 66.69 → 62.99. AUROC was
0.65–0.74, better than self-agreement but not enough to gate on. A gate can
only earn on abstention rows, and false alarms cost every time.
`docs/measurements/m56-verified-answers.md`. Next: M57, the reader-prompt fix.

---

## M54 — a local file-reading controller for LME-V2 *(pilot +40.4 [+25.5, +55.3]; full pair running)*

M53 (`docs/measurements/m53-state-completion.md`) measured where LME-V2
loses its answers — 45% of wrong phrase/list answers are in the haystack but
never delivered, 36% delivered and misread — and probed three cheap
retrieval fixes: reranked state completion recovered **1 of 6** live, a
change view over delivered states would cover 19 of 65 misses (~+2 to +3),
a step-anchored change view 3–4. None is the lever. The answer is a specific
UI string in a specific transition, and the LME-V2 paper's best system finds
it by *acting*: a coding agent over trajectory files, 72.5 against RAG's
48.5 (`10.48550/arXiv.2605.12493`).

Local-only (hard requirement): an `investigate` tool set over the stored
trajectories — `grep` over page text and thoughts, `open(traj, state)`,
`diff(traj, state)` against the previous state — driven by a local
controller (big's 27B coding model, or the 9B), returning the evidence it
gathered to the unchanged reader. Pilot on a stratified LME-V2 subset before
any full pair. Design doc first.

**Pilot measured 2026-09-24: 42.55 → 82.98, +40.43 [+25.53, +55.32]** on 47
questions, with the paper's AgentRunbook-C driven by Bonsai 27B through Codex
and the protocol's 9B reader. The full 451-question pair is running in
resumable chunks (~27 GPU-hours)
(`docs/measurements/m54-local-file-controller.md`).

---

## M44b — the thinking reader, tightened *(follow-ups to a shipped win)*

R2 ships at **78.40** with a 1,024-token budget and no budget message. Two
cheap arms, both against `runs/m44_r2_s1_judged` under the same seed:

- ~~**R2b — budget message.**~~ *Measured on two seeds 2026-09-23: +1.8 and
  +1.4; pooled **+1.6 [−0.1, +3.2]**, spill rows +5.5 [+0.5, +10.6], control
  +0.0 exactly. Real, a point and a half, below the bar; ships off. The
  serve default stays without the message.*
- **R2c — budget 2,048.** The same 165 rows, given room. Costs ~1.5× per
  row. Predicted: `temporal-reasoning` and gold ≥ 3 move; the preference
  stratum (−20.0 on seed 2, the one cost) does not recover — it is not a
  budget problem.
- **LoCoMo split (after M51).** M51 measured the LongMemEval_S bundle on
  LoCoMo at **−4.81** [−7.01, −2.66] on categories 1–4 and +14.13 on
  adversarial: the reader declines 121 → 197 answerable rows. Split, as
  pre-registered: (a) `recall` + `--reader-thinking` alone; (b) `investigate`
  + select + dated digest with the plain reader. ~2–3 h each on big.

**Cost.** R2b/R2c ~1.5 h each; LoCoMo an overnight.

---

## M52b — the LME-V2 reader thinks and stops *(follow-up to M52, off)*

M52 measured thinking at a 1,024-token budget on LME-V2 web: −2.08, abstention
−9.72 (veto), with the loss on the 172 of 240 answers that hit the budget
and spilled (up to 12,556 tokens). M44 R2b's budget message cut the same
spill from 56 to 1 on LongMemEval_S. So the arm is thinking **plus** the
budget message (`MYELIN_READER_THINK_MESSAGE`), on the base's replayed
memory (`--reuse-prompts-from`), reading only: ~1 h per domain. Pre-register
before running; the bar and the veto are M52's.

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

## M50c — events beside turns, not instead *(M50 and M50b's shared lesson)*

Events rescue declines on both benchmarks: +15.8 on LongMemEval_S's base
declines and +11.6 on LoCoMo's. But at k = 6 they displace turns. On LoCoMo,
74 rows lost a gold turn the base held against 17 that gained one, and the
arm netted −0.13 (M50b). The untested design keeps k = 6 turns and adds the
top events in a separate budget. Reader-side only, so the stores already
built serve it. Pilot on the M50 population first. Chronos keeps events in a
separate index; three routes are written up, and one needs choosing before
any code (`docs/measurements/m50c-events-beside-turns.md`).

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
- **LME-V2 needs the vision projector.** The harness attaches question
  screenshots as `image_url` content; a reader served with `MYELIN_MMPROJ=0`
  (the LongMemEval_S/LoCoMo saving) answers HTTP 500 on the first reader
  call, after the ~45-minute prompt build, which the harness cannot resume.
  2026-09-23 05:00: one such loss. `ops/big/README.md` says so now, and
  `run_myelin.py` preflights it: when any selected question carries a
  screenshot (web small: 15 of 240; enterprise: none) it sends that image
  through the harness's own `to_data_url` as one `max_tokens: 1` request
  before building anything, and a refusal ends the run at the door with
  the fix named. Verified 2026-09-23: a text-only selection sends nothing,
  the projector-served reader accepts in one request, a dead endpoint is
  refused.
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
