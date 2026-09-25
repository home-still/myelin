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
its history. Short version:
- **One gate closed:** MINJA 7.50% ≤ 10%.
- **LongMemEval_S is past the same-size SOTA row:** **83.40** against
  MemPro-15 (Qwen3-30B) at 80.80, since M57 (Bonsai 27B plus the premise
  clause). It is `caveat-judge`, so `standing` keeps that gate open. The climb
  since M32: 62.00 → 67.80 → 78.40 → 83.40.
- **LoCoMo: the gate closes under MemPro's own judge (M68, 2026-09-25).**
  - Graded the way the 77.85 row was graded (gpt-4o-mini, LightMem's
    prompt), we score **78.18**, a `comparable` row, +0.33.
  - Under the strict 9B judge, which still decides every arm, we score
    70.52.
  - The lead is inside reader variance. Multi-hop (−4.6) and open-domain
    (−34.4) still trail MemPro under its own judge.
- **LME-V2 38.80**, behind the 74.90 AgentRunbook-C gate. M54's local
  controller piloted at 82.98, and its full 451-question pair is running.

---

## SOTA push *(plan approved 2026-09-24; round 2 approved 2026-09-25)*

The checklist lives in `docs/measurements/sota-push-checklist.md`. The
research behind the next builds is `docs/research/sota-catalog-2026-09-24.md`
and its addendum `docs/research/sota-catalog-2026-09-25.md`.

**Round 2, measured 2026-09-25 on `runs/m63_locomo_base`** (454 of 1,540
lost under the strict judge):
- **Multi-hop loses 114:** 63 hold only part of their gold (missing ~2 of
  3.6 turns), 43 hold none, and 8 hold it all. The correct-rate is 87% with
  all gold held, 59% with part, and 36% with none.
- **The cause is granularity.** An episode is a 512-token segment of ~13
  turns, so k = 6 reaches ~2.3 episodes plus ~3.7 facts. The pool holds the
  gold (0.959) far more often than the kept set does (0.863).
- **Single-hop loses 147.** About 60 of those pick a distractor inside the
  13-turn chunk.
- **Temporal loses 127**, of which ~45 are date handling.
- **Open-domain loses 66.** 41 are speculative (reader capability) and 18
  are entity→name mappings.
- **About 40–60 rows are judge or label disagreements** that no memory change
  recovers.

**Next, in order:**
1. **M66: turn windows** *(the user's first build)*. Each selected episode
   is shown as its best turn ±2 neighbours, scored by the cross-encoder, and
   k rises so that tokens per question stay flat.
   - Measure retrieval first, with a new turn-level all-gold metric.
   - Then one reader arm, paired vs `m63_locomo_base`.
   - Research: QueryLink (±c turns), JustMem (fine units beat packs by
     +10.7), MemPro's focused snippets, RECOMP, Du 2025.
2. **M67: list questions, rewrites unioned.** A JustMem-style COMPOSE that
   reuses `decompose.rs` (M24, built and never measured). The planner emits
   an operation plus ≤2 answer-free rewrites.
3. **Temporal residuals** in `time.rs`: 5 relative phrases left unresolved in
   chunk turns, plus duration rendering. This lands before the next fresh
   base.
4. **Open-domain entity enrichment at write time** (QueryLink's implicit
   view). It needs a rebuild, so it runs only if OD stays the largest
   stratum.
5. **LME-V2:** M54's full pair. 180/404 questions are done. sib runs; big
   waits for room without evicting anyone (1 slot × 96K, `--cache-ram 0`
   after the 09:31 OOM).

**Bottlenecks measured 2026-09-24 PM:**
- **LoCoMo.** The 9B loses 464 answerable rows, split evenly:
  - *reading with the gold in hand* (233 rows): relative dates are
    resolved ~2,000 chars from the phrase; ISO ranges are judged wrong;
    Bonsai refuses 292;
  - *retrieval* (~231): facts duplicating a selected episode waste a
    slot on 72% of rows; multi-hop holds all its gold on only 22%.
- **Measured 2026-09-24 evening (none ships alone):**
  - M61 grounded override: +3.38 on Bonsai, adversarial untouched, +0.19
    vs the 9B;
  - M64 dates in place: +0.78;
  - M63 dedupe: **−4.81** (falsified);
  - M50c events: vetoed (adversarial 62.11);
  - M65 k = 10: retrieval all-gold +3.1 (reader arm pending);
  - fresh LoCoMo base 70.52.
- **LME-V2's native controller** (M62b 36.17, M62c 25.53) pauses; the GPU
  goes to the M54 full pair (13/32 chunks).
- **M60 and M58 are deferred** (user, 16:40), so big goes to the M54
  full pair after M62b. sib went offline at 16:32.
- **LongMemEval_S:** M57 **shipped 83.40** (past the 80.80 row); M58 (the preference clause on top) runs next.
- **Code first (user, 2026-09-24 ~14:15).** Every LongMemEval_S gain since
  M43 came from the model (thinking, Bonsai, a prompt clause), and LME-V2's
  +40 from the authors' method. From here the building effort goes into
  myelin's memory code, and no new per-benchmark prompt clauses are added.
  The model arms already queued (M60, M58) finish.
- **LoCoMo:** M59 (best-guess clause) **did not ship: −1.95**. The clause
  moved only 27 of 292 refusals, though 70% of those were right. M60 (+
  thinking) is queued.
- **M62, myelin's own agent-history reader: built** (PRs #94–#98, #100).
  It stores trajectories state by state, has bounded tools and a
  schema-constrained controller. **Pilot: 31.91 against AgentRunbook-C's
  82.98.** 36 of 47 answers were forced by the step budget, 19 named no
  span (0/19), and answers with spans scored 54%. M62b gives it the rule
  that naming a span is enough, and a forced answer that names the spans it
  found.
- **LME-V2:** M54's full pair on big (2 slots) plus sib, then adopt AgentRunbook-C as
  myelin's agent-history mode (labelled) and build a native version (M62).

---

## M56 — an external checker on when to answer *(0b, used per the manual: safe; LoCoMo 70.65, short of the gate)*

Jev (TypeSafe's cloud decision model) replayed over existing answers:
LongMemEval_S 82.20 → 59.80 simulated, LoCoMo 66.69 → 62.99. AUROC was
0.65–0.74, better than self-agreement but not enough to gate on. A gate can
only earn on abstention rows, and false alarms cost every time.
Step 0 misused Jev against its own manual. Step 0b follows TypeSafe's per-passage
recipe: LongMemEval_S 81.80, abstention 26/30 (flat). LoCoMo 66.69 → 70.65 by
overriding refusals that had evidence in hand, with the 9B stand-in right 42% of
the time. Next: Bonsai answering those refusals itself under a best-guess clause,
with an adversarial guard (`docs/measurements/m56-verified-answers.md`).

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
