# M34 — AgentRunbook's three pools, completed and measured: a null

**Verdict: completing the three knowledge pools is worth +0.22 points, and the
CI includes zero.** LongMemEval-V2 tier-small, both domains, identical
operating point, the only difference being that the store now holds all three
of AgentRunbook-R's pools instead of one:

| | combined | web | enterprise |
|---|---|---|---|
| episodic only (M33) | 38.36 | 42.50 | 33.65 |
| **three pools (M34)** | **38.58** | 44.58 | 32.70 |
| Δ | **+0.22** | +2.08 | −0.95 |

Paired over all 451 questions: **+0.22, 95% CI [−3.77, +4.21], p = 1.0000**.
Web alone: +2.08, CI [−3.33, +7.50], p = 0.5585 — 45 of 240 rows changed
verdict, 25 gained and 20 lost. Churn, not a shift.

The pools are not being ignored. They are **33.3% of the evidence the reader
sees**:

| emitted record kind | count | share |
|---|---|---|
| episodic (raw states) | 2,292 | 66.7% |
| procedural (notes) | 778 | **22.7%** |
| semantic (events) | 364 | **10.6%** |

3,434 items over 240 web questions, 14.3 per question, and **96.7% of
questions receive at least one pool record**. A third of the reader's context
was handed to two new pools and the answer did not move.

## 1. Why this was worth doing

`standing` now reads the LME-V2 gap as **−20.02 against AgentRunbook-R**
(58.60), which is the only apples-to-apples row in the table: LME-V2's reader
is Qwen3.5-9B, the same open-weights model myelin serves. So the question
"what does AgentRunbook do that we do not" has an exact answer available, and
`docs/research/11-frontier-2026.md` §1.6 gives its method:

> **AgentRunbook-R** — 3 knowledge pools: raw-state slice pool; state-transition
> event pool (LLM-generated `{overview, state_transition}`); procedure + hint
> note pool (per trajectory, `{procedure_note, hint_note}`).

A census of our stores said we had one of the three, and said something
sharper than that:

| store | episodic | semantic | procedural | gate gap |
|---|---|---|---|---|
| **LoCoMo** | 552 | **4,012** | **310** | **−7.98** |
| LongMemEval_S | 162,181 | 0 | 0 | −18.80 |
| LME-V2 | 85,589 | 0 | 0 | −20.02 |

The one corpus with all three pools is 8 points off the bar. The two with raw
episodes only are 20 points off. That is a correlation across three corpora,
and it was the best available hypothesis for the gap. **It is now tested and
it does not hold** — at least not through content alone (§5).

## 2. The mechanism existed and had never run

`build --corpus lme-v2-small --pools` was written in M23 (D1) and is a close
replication: the note prompt is verbatim from the vendored AgentRunbook-R
(`memory_modules/support.py`, `NOTE_GENERATION_SYSTEM_PROMPT`, prompt version
`qwen_v6_retrieval_safe`), and the event pass takes the same top-6.

It had never successfully run. The ledger contained zero pool-minting audit
events and the store zero records of either kind. Two defects, both in code
that had never been executed:

### 2.1 I4: the events pool could not be written at all

`RecordKind::Semantic` triggers `MemoryRecord::requires_lineage`, and the
ledger enforces I4 — a semantic record must name what it was abstracted from,
and every ancestor must exist. The pool pass set `record_kind` and nothing
else, so the first insert died:

```
Error: store: I4: semantic record a5c28e0a-… has empty derived_from
```

Fixed with real lineage rather than by relaxing the invariant: an event
abstracts over the trajectory's raw states, and those are exactly the
trajectory's episodic records. `WritePath::derived_from` carries them, and
`Ledger::ids_from_source_docs` resolves them — `SourceRef::doc` serialises as
`{"doc":"<traj>:<state>"}`, so a trajectory's states are the documents under
`<traj>:`. `visible_of_kind` could not serve this: it is scoped, not
source-filtered, and one LME-V2 tenant holds ~38k records, so 200
trajectories would have been 200 full-tenant scans.

The lookup runs **before** the model call, so a store without the episodic
pass now fails with a sentence naming the cause instead of spending 400 LLM
calls and dying inside the ledger.

### 2.2 The notes pool asked for a grammar llama.cpp will not compile

Every notes call returned HTTP 400:

```
Failed to initialize samplers: failed to parse grammar
```

`maxLength` is expanded by llama.cpp's json-schema-to-grammar into that many
optional character repetitions, and the parser rejects the result. Bisected
against the live reader: **1999 compiles, 2000 does not.** The note's
`content` was capped at exactly 2000; the event schema's 1000 was fine, which
is why events "worked" and notes did not.

The failure mode is worse than an error. `build_lmev2_pools` catches a failed
extraction per trajectory and logs `skipping pool`, so the pass **reported
success** with notes at 0.0% coverage — and the coverage line it prints was
the only evidence, on stderr, in a 30-minute build.

Fixed at 1500, with margin against a limit that is a llama.cpp implementation
detail rather than a specified one. `MAX_SCHEMA_MAX_LENGTH` pins it and
`no_schema_asks_for_a_maxlength_llama_cpp_cannot_compile` walks every schema
we send. Verified to fail on the reintroduced 2000:

```
maxLength 2000 >= 2000: llama.cpp will reject this grammar and the pool it
belongs to will report 0% coverage while the build says it succeeded
```

The test is arithmetic over the schemas, not a live call, so it runs in the
hermetic suite and cannot be skipped when no server is up.

### 2.3 The build, after both fixes

```
pool extraction coverage: events 100.0%  notes 100.0%
```

190 event records and 200 note records over 200 trajectories, 30m30s. Both
pools segment to roughly one record per trajectory, because the six event
turns and two note turns each merge through `segment` like any other episode.

## 3. The instrument this milestone needed first

M33 shipped a per-query trace as a **sidecar file**, and it could only report
aggregate counts, because a sidecar has no question identifier and
`harness.py` builds prompts across four worker threads — so its line order is
not the question order and the rows cannot be joined to outcomes at all.

Replaced by `Memory.post_query_hook`, which the harness calls immediately
after `query()` on the same thread and writes to `per_question.jsonl` as
`memory_post_query_metadata` — **keyed to the question id by the harness
itself**. Present on 240/240 and 211/211 rows.

The per-query state is **thread-local**, for the same reason the base class's
own `_query_context_local` is: a plain `self._last_trace` would let one
question's trace be reported against another's, and a join over
mis-attributed rows looks valid and is wrong.
`test_concurrent_workers_never_report_each_others_traces` was verified to
fail against a shared attribute and pass against the thread-local.

`clear_query_context` is overridden to drop the trace too, because
`harness.py` clears in a `finally` and a query that raised must not leave its
trace for the next question on that thread to claim.

### It also found a broken test file

M33 made the adapter send `select` unconditionally and three tests in
`test_query_privacy_myelin.py` started raising `AttributeError` on a field the
constructor sets and the hand-written fixture did not. They are `unittest`, so
`cargo test` never runs them and nothing in the landing checklist caught it —
**M33 landed with them broken.** The fixture now constructs through the real
`__init__` with only the transport swapped, so a new operating-point field
cannot make it stale again. The privacy test also did its job: it flagged
`select` as a new outbound wire key, now recorded deliberately with the reason.

## 4. The defect this milestone created, and the fix

Minting pools changed `myelin_lme_v2_small` **without renaming it**. M33's
LME-V2 runs and M34's therefore share an identical operating point and read
materially different stores, and `pair_metrics` pairs by operating point — by
design, and stated as such: "Pairing is by operating point, never by directory
name."

It formed a cross pair. `standing` published **39.47**, from M34's web run
(44.58) and M33's enterprise run (33.65): a combined accuracy that no
configuration ever produced and nobody could reproduce.

Two changes:

- `run_myelin.py --ledger` censuses the ledger by record kind and records it
  as `store_fingerprint` in `memory_config.json`. The collection *name* is not
  the store's identity; the counts are what differ.
- `pair_metrics` refuses to pair runs whose `store_fingerprint` differs, and
  **absence is not a match for presence** — an artifact that does not name its
  store cannot be shown to have read the same one.
  `a_pair_must_have_read_the_same_store` pins all four corners.

`store_fingerprint` is deliberately **not** in `PAIR_KEYS`: adding it there
would mark every historical artifact `stale-config` for a field that did not
exist, which is a different claim from the one being made.

### The M33 LME-V2 pair is removed

`runs/m33_web` and `runs/m33_ent` measured a store that no longer exists. They
cannot be reproduced, they carry no `store_fingerprint`, and keeping them
invites exactly the cross pair above. Removed — the same reasoning
`lafs_gain` already applies to superseded arms, and M33's number stays on the
record in its own measurement doc and PR.

`standing` now reads the pair it can reproduce: **38.58, −20.02 to
AgentRunbook-R.**

## 5. What the null actually says, and what is left

The pools are present, retrieved, and take a third of the reader's evidence
for no measurable gain. That rules out the content hypothesis and leaves the
**routing** difference, which §1.6 states precisely:

> Query: controller emits JSON `{"raw_state_queries":[…≤5], "event_query":…,
> "note_query":…}`; **top-6 events, top-3 notes, top-m raw states**
> (m = min(2, 6//n_queries)).

AgentRunbook does not fuse the three pools into one ranking. It issues a
**separate typed query per pool** and gives each pool a **reserved quota**.
Our `compose` takes top-k from a single fused ranking, so which pool a slot
goes to is decided by cross-encoder score against the whole question — and
the note that surfaces is the one that scores high on the question, not the
one that answers the note-shaped part of it.

Note that our emitted share (33.3% pools) is already close to theirs
(6 events + 3 notes of ~19 items ≈ 47%), so the *proportion* is not the
defect. The typed query is the untested part, and `InvestigateConfig::typed_probes`
— built in M23, never measured — is exactly it. Before this milestone it had
nothing to aim at: every LME-V2 record was episodic, so a probe tagged
"event" or "note" searched a pool that did not exist. **That is why it was not
measured here**; running it would have been M27's failure class, a
pre-registered question answered by an empty store and indistinguishable from
a real null.

## 6. The selector's declines are not yet an abstention signal

§15 asked whether `ModelDeclined` — the selector answering "nothing here
jointly answers this" — predicts a wrong answer. Over the combined 451, now
joinable because of §3:

| | correct | wrong | accuracy |
|---|---|---|---|
| no decline | 166 | 256 | 39.3% |
| **declined** | 8 | 21 | **27.6%** |

29 declines, 6.4%. Fisher exact two-sided **p = 0.2409**. The sign is right
and the effect is not resolvable at this n, so it does not support wiring
`abstain_on_insufficient` to it.

One process note: the first Fisher value computed for the web subset alone was
**0.0352 from a hand-rolled implementation that was wrong**; the correct
exhaustive value is 0.4658. It was caught by re-deriving rather than by
review, and it would have been a published false positive.

## 7. Reproduction

```sh
ssh big gpu-tenant claim coding
ssh big "MYELIN_READER_SLOTS=1 MYELIN_READER_CTX=65536 bash -s" \
    < ops/big/serve-models.sh
ssh -N -o ServerAliveInterval=15 -L 5810:127.0.0.1:5810 \
    -L 5813:127.0.0.1:5813 big &
export MYELIN_QDRANT__URL=http://192.168.1.110:6334

# Mint the two missing pools INTO the episodic store (30m30s).
# --repair because adding records is drift until reconcile accepts it.
myelin-eval build --corpus lme-v2-small --pools --lmev2-dir data/lmev2 \
  --repair --collection myelin_lme_v2_small --ledger data/lme_v2_small.ledger

myelin-mcp --serve 127.0.0.1:7446 \
  --collection myelin_lme_v2_small --ledger data/lme_v2_small.ledger &

cd crates/myelin-eval
export OPENAI_API_KEY=local
for domain in web enterprise; do
  PYTHONPATH=vendor/longmemeval-v2:adapters ../../.venv/bin/python \
    adapters/run_myelin.py --data-root ../../data/lmev2 \
    --domain "$domain" --tier small \
    --k 25 --budget-tokens 10000 --mode investigate --max-steps 2 \
    --select --undated --ledger ../../data/lme_v2_small.ledger \
    --evaluator-base-url http://127.0.0.1:5810/v1 \
    --evaluator-model Qwen/Qwen3.5-9B --reader-model Qwen/Qwen3.5-9B \
    --output-dir "../../runs/m34_pools_$domain"
done
```

Pass `--ledger` on every future LME-V2 run: without it the artifact does not
name its store and `pair_metrics` will refuse to pair it.

Wall clock: pools 30m30s, web 48m50s, enterprise 47m18s.
