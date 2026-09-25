# M62 — myelin's own file-reading agent for agent histories *(design, 2026-09-24)*

## Why

On 2026-09-24 the user adopted the LME-V2 authors' agent, AgentRunbook-C
(`10.48550/arXiv.2605.12493` §4.2), as myelin's agent-history mode. Standing
prints the method beside every number it produces (C3, PR #86), so nobody
mistakes it for myelin's own memory. It scored 82.98 against myelin's 42.55
on M54's 47-question pilot. M62 builds the same capability natively, so that
myelin's memory reaches that point without shelling out to Codex over an
exported sandbox.

## What AgentRunbook-C actually does

Read from the vendored implementation
(`vendor/longmemeval-v2/memory_modules/assets/agentrunbook_c/`):

1. **Shortlist from summaries.** A rendered summary of every trajectory:
   goal, start URL, action sequence and final reward. The agent picks a few
   likely sessions before opening any raw state.
2. **Verify with one small tool.** `inspect_trajectory.py <id>` returns a
   compact summary; `--state N` one state; `--span a:b` a short span;
   `--match "A|B"` the states of one trajectory matching a pattern. The
   instructions forbid `rg`/`find` over raw trajectories.
3. **Answer with spans.** The output is `trajectory_spans`, each a trajectory
   plus a start and end state, with at most 20 states in total, and a short
   "support analysis" in prose. The harness renders the spans into the
   reader's context.
4. **Strict rules.**
   - Reject near-but-not-exact matches.
   - If the evidence contradicts the question, say so in the support
     analysis rather than extrapolating.
   - Keep the package small.

The controller is the only model-driven part. Everything else is file access
and formatting.

## The native design

Built into `investigate` (`crates/myelin-core/src/pipeline/investigate.rs`),
whose loop already has a model deciding steps (`Investigator::investigate`,
`max_steps`) and a pool-level selector.

- **Trajectory index** (write path, once per store). One summary record per
  trajectory: goal, start URL, action sequence, outcome. It is derived from
  the records the LME-V2 build already writes, keyed by trajectory id in the
  ledger. No new store.
- **Tools the controller may call**, served from the ledger, never from raw
  files:
  - `shortlist(query) -> [summary]`: the trajectory index, reranked.
  - `state(traj, i)`, `span(traj, a, b)`: rendered states, with the
    accessibility tree (`axtree`) as the evidence mode, as M54's
    pre-registration fixed.
  - `match(traj, pattern)`: state indices whose text matches, within one
    trajectory only.
- **Output**: the same span contract, with at most 20 states. Spans become the
  `EvidenceSet` items the reader already receives, so the reader, the judge
  and the harness adapter stay unchanged.
- **Controller**: Bonsai 27B through llama.cpp's native tool calls. That
  avoids the Codex shim and its two rewrites, which then stop being needed
  for this path.
- **Rules**: AgentRunbook-C's four, as the controller's system prompt, with
  citations in the code.

## How it is measured

1. The M54 pilot population (47 questions), against M54's own pilot rows
   (82.98). This is a replication gate: within 5 points, and above the
   74.90 bar.
2. Then the full pair, run the same way as M54's, to replace the adopted
   mode. The standing constant `SHIPPED_LME_V2_MEMORY` moves from
   `agentrunbook_c` back to `myelin` only if the native pair is at least as
   good.

## Open before building

- **Summary depth.** AgentRunbook-C has a concise and a full summary file.
  Start with the full one; it is what the instructions say to shortlist
  from.
- **Tool-call reliability.** Bonsai's native tool calls through llama.cpp
  are documented to work on big (they back the omp coding agent), but they
  are unmeasured on this task. Before anything else, a smoke test of 5
  questions, counting tool-call parse failures.

## Code map *(2026-09-24, before building)*

Where each piece of the design lands today, so the build starts from facts.

| piece | today | what M62 adds |
|---|---|---|
| step loop | `Investigator::investigate` (`myelin-core/src/pipeline/investigate.rs`): one action, *search*; the model decides by `reflect()`, a JSON-schema call returning `{sufficient, next_query, …}` | a trajectory-agent branch that fills `EvidenceSet.items` with spans and skips `compose` |
| evidence | `EvidenceSet { items, tokens, trace, … }` (`model/evidence.rs`); `to_wire()` sends only `{type, value}` | nothing: one `Text` item per rendered span, with a real `record_id` and source, keeps the reader, harness and wire unchanged |
| trajectories in the store | `datasets/lmev2.rs::turns_for` writes a goal turn (`{id}:goal`), step turns (`{id}:{state}`) and page chunks (`{id}:{state}:{chunk}`, 1,800 chars) per trajectory; the trajectory id lives only in `provenance.source.doc` | the index (goal, start URL, actions, outcome) is derived from these records, with no new store |
| fetching one trajectory | `Ledger::ids_from_source_docs(scope, "{traj}:")` returns ids by UUID order, then one `get` per id | `records_by_source_prefix`, returning rows in state order in one query |
| LLM tool calls | `ToolSpec` / `ToolCall` / `CompletionRequest.tools` exist and `openai.rs` sends and parses them, but nothing has ever called them, and `Message` is only `{role, content}` (no `tool_call_id`, no assistant `tool_calls`) | depends on the controller choice below |
| span format | the authors' `format_span_header` / `format_state_text`, at most 20 states (`vendor/longmemeval-v2/memory_modules/codex.py`) | the same rendering, ported, with the source cited |
| adapter | `MyelinMemory.query` returns `result["items"]` verbatim; `mode` is checked against an allowed set in `adapters/myelin.py` | a new `investigate` switch, not a new mode, so the adapter is untouched |

**Two things the map changed.**
- One steps record can cover several states, because step turns merge
  into episodes of up to ~512 tokens. So `state(traj, i)` reads the
  page-chunk records, not the steps record.
- Native tool calls are unexercised plumbing in myelin. Two ways to drive
  the controller:
  - **Native tool calls**, as the design says. This needs `tool_call_id`
    and assistant `tool_calls` on `Message`, plus the 5-question
    parse-failure smoke test.
  - **One schema-constrained action per step**, e.g. `{"tool": "span",
    "traj": …, "a": 3, "b": 7}`, the path `reflect()` already uses on
    every `investigate` call.

  **Decided by the user, 2026-09-24:** one schema-constrained action per
  step. Each step's output is forced to match the action form (grammar-
  constrained decoding, Geng et al., EMNLP 2023,
  `10.18653/v1/2023.emnlp-main.674`), in a reason-then-act loop (ReAct, Yao
  et al., ICLR 2023, `10.48550/arXiv.2210.03629`). This is the path
  `reflect()` already runs on every `investigate` call.

## Build plan *(2026-09-24, after the two decisions)*

The user put the SOTA push on **code first** the same day. Model- and
prompt-side arms stop at M58–M60, and the building goes into myelin's own
memory. M62 is the first of those builds.

**Second decision (user): agent histories get their own table.** Today the
store keeps a trajectory only as merged search chunks. Step turns merge into
episodes of up to ~512 tokens, and provenance keeps only the first turn's
source, so a state's action cannot be read back exactly. M62 adds
`trajectory` and `trajectory_state` tables to the ledger: one row per
trajectory (goal, environment, start URL, outcome, and the id of its goal
episode record, for lineage) and one row per state (index, step, URL,
action, thought, accessibility tree). It sits beside the episodic records,
which stay as they are for search. It is filled at build time, and existing
stores are backfilled from the dataset with no re-embedding.

- [x] **PR 1 — the store.** Schema (`CREATE TABLE IF NOT EXISTS`, so
      existing ledgers gain it on open), model types, `Ledger` write and
      scoped reads (list a tenant's trajectories; read one state or a
      range), and tests.
- [x] **PR 2 — the write path.** `build_lmev2` fills the table;
      `build --trajectories` fills already-built stores (no GPU, no
      embeddings). Row counts are checked against the dataset. *Done:* 200
      trajectories and 5,095 states on the small tier in 14 s, matching the
      release field for field (checked independently in Python); a re-run is
      idempotent.
- [x] **PR 3 — tools and rendering.** `pipeline/trajectory_tools.rs`,
      with every output bounded. On LME-V2 a state's page averages ~34 KB
      (~8k tokens), so the controller searches and reads windows while the
      reader gets whole spans.
      - `list`: every trajectory, sorted by start URL.
      - `summary`: numbered actions and the state each led to.
      - `grep`: case-insensitive, at most 40 lines and 400 states, and it
        says when it was cut.
      - `read`: a 150-line window of one state.
      - `evidence`: the authors' layout (notes, the span list,
        `### Trajectory span k`, per-state `State i (step s)` with the
        AXTree) and at most 20 states. A bad span is refused, so the
        controller can correct it; the authors silently drop it.
      - Each state item cites its anchor record and `SourceRef::span`. The
        controller's notes are a view (nil record, mechanism source,
        weakest trust).
- [x] **PR 4 — the controller.** `pipeline/trajectory_agent.rs`, its own
      module rather than a branch inside `investigate`'s search loop.
      - It opens with the trajectory list and the question. Each step is
        one JSON action forced into `action_schema` (thought plus one of
        `summary` / `grep` / `read` / `answer`).
      - The system prompt is the authors' INSTRUCTION.md rules, rewritten
        for these tools.
      - A misused tool, or an answer the tools refuse, comes back as the
        observation for the model to correct.
      - Past 16 steps or a 48,000-token transcript, the next step allows
        only `answer`. An answer still refused twice is an error, never an
        invented evidence set.
      - Tested against a scripted model (5 tests).
- [x] **PR 5 — wiring.** The MCP tool `trajectories` runs the controller on
      the server's model (for the pilot, the server's LLM URL points at
      Bonsai, and the harness reader stays the 9B). `run_myelin.py --mode
      trajectories` sends only the arguments that tool declares. `--max-steps`
      now defaults per mode: 2 for `investigate` (the M7 step-curve peak), 16
      for `trajectories`.
- [x] **Pilot.** **31.91 against 82.98: fails** (result below). The 47 questions of M54's pilot, against its 82.98, when a
      GPU is free. Phase one builds the prompts with Bonsai as controller;
      phase two reads them on the 9B as one 204,800-token slot, as M54's
      reader did.

## Pilot — pre-registered *(2026-09-24 ~15:05, before any row)*

The user scheduled the pilot on big right after M59, ahead of M60 and M58
(code first).

**Population.** M54's pilot, 47 questions: web 24 and enterprise 23
(`docs/measurements/m54-pilot-{web,enterprise}.txt`).

**The arm.**
- **Store:** `data/lme_v2_small.ledger` with the M62 tables (200
  trajectories, 5,095 states). The pre-M62 copy is kept beside it.
- **Controller:** Ternary Bonsai 2 27B PTQ1_0 on big, through
  `serve-models.sh` (`MYELIN_READER_MODEL=bonsai-27b`, 2 slots x 65,536,
  no projector). It is the MCP server's LLM.
- **Controller settings:** temperature 0, thinking off, at most 16 steps,
  a 48,000-token transcript budget, and `axtree` evidence.
- **Phase one:** `run_myelin.py --mode trajectories --prompts-only`, two
  questions at a time.
- **Phase two:** `--reuse-prompts-from`, read by Qwen3.5-9B exactly as
  M54's reader. That is one 204,800-token slot plus the projector
  (`MYELIN_READER_SLOTS=1 MYELIN_READER_CTX=204800 MYELIN_MMPROJ=1`),
  one request at a time, temperature 0.6, top-p 0.95, top-k 20, 20,000
  completion tokens and a 200,000-token memory context. The harness judge
  is the 9B.

**Comparisons**, paired over the 47:
1. **Replication:** against AgentRunbook-C with the same local controller,
   M54's pilot (`runs/m54_arc_{web,ent}`, 82.98). M62 passes if it scores
   at least **77.98** (within 5 points) and at least **74.90** (the LME-V2
   gate row).
2. **Against myelin's shipped LME-V2 memory** on the same 47
   (`runs/m47_base_{web,ent}`, 42.55): +5 with the CI excluding zero,
   M54's own gate.

**Predictions.**
- Overall 70–85.
- Abstention at least 50% (M54's pilot: 64.29%).
- Fewer than 30% of answers forced by the budget.
- Zero schema failures, meaning no reply outside the action form.

**Diagnostics reported either way:**
- steps per question and the forced-answer rate, by domain;
- tool errors fed back to the model;
- states per answer;
- phase-one minutes per question, against M54's controller.

**Known differences from M54's AgentRunbook-C.**
- **Tools:** bounded tools here, where M54's controller had a shell and
  scripts.
- **Sampling:** temperature 0 and thinking off, where M54 used the Codex
  defaults and reasoning effort "medium".
- **Budget:** a 16-step budget.
- **Images:** neither controller sees images. M54's shim replaced every
  image with a text note; the reader still gets the question image here.

## Pilot result *(2026-09-24 16:07)* — **fails the replication gate: 31.91 against 82.98**

`runs/m62_pilot_{web,ent}` (phase one `runs/m62_pilot_*_prompts`). The code
is main at `21398c3`, plus the MCP timeout fix (#100) that landed before any
row: the controller is Bonsai PTQ1_0 and the reader is the 9B at
1 × 204,800. Both served models are confirmed from their endpoints.

| paired over 47 | AgentRunbook-C (M54 pilot) | M62 | Δ | 95% CI |
|---|---|---|---|---|
| **combined** | 82.98 | **31.91** | **−51.06** | [−68.09, −31.91] |
| answerable | 90.91 | 39.39 | −51.52 | [−72.73, −30.30] |
| abstention | 64.29 | 14.29 | −50.00 | [−78.57, −14.29] |

Against myelin's shipped LME-V2 memory on the same 47 (42.55), it is
−10.64 [−27.66, +6.38], a null.

**Gates.** Replication needs at least 77.98 and at least 74.90: it failed.
The +5 gate over the base also failed.

**Predictions.**
- ✗ 70–85.
- ✗ Abstention ≥ 50%.
- ✗ Under 30% forced: **36 of 47 (77%)** were.
- ✓ Zero schema failures: every reply parsed.

**Where it lost:**
- **19 answers named no span.** They scored 0/19. The controller's own
  notes explain several of them. It had found the right trajectory and
  state and not finished reading it: "the exploration budget was
  exhausted before I could read the accessibility tree of state 2". It
  believed it had to read a page whole before citing it, and a page is
  ~34 KB while `read` shows 150 lines.
- **Answers with spans scored 15/28 (54%)**, against 83% for
  AgentRunbook-C. It chose worse spans as well as too few.
- **Answers given before the budget scored 3/11.** Stopping early did not
  mean it was sure.
- **Some notes report "tool errors".** The adapter did not keep the action
  list, so which errors they were cannot be told from this artifact.
- **Speed was the one strength:** 80 s per question on web and 129 s on
  enterprise (median), against about 5 minutes for M54's controller.

**Next: M62b**, held on branch `m62b-controller` until this write-up:
- the rules say that naming a span is enough, because the reader sees
  every named state in full;
- the forced-answer message asks for the spans of the states it
  identified, read or not;
- `AgentRun` reports whether the answer was forced and how many tool
  errors there were, and the adapter keeps the action list.

It targets the 19 empty answers directly. If answers with spans still trail
AgentRunbook-C, the next lever is thinking: M54's controller ran with
reasoning effort "medium" and this one with thinking off.

## M62b — pre-registered *(2026-09-24 ~16:15, before any row)*

**One change of mechanism** (commit on `m62b-controller`):
- the rules add "Naming a span is enough: the reader receives every state
  you name in full; read only to choose between candidate states";
- the forced-answer message asks for the spans of the states the
  controller identified, read or not.

The instrumentation that lands with it changes nothing the model sees:
`AgentRun.forced` and `tool_errors`, and the action list in the artifact.

**Run.** Identical to the M62 pilot in every other respect:
- the same 47 questions;
- Bonsai controller at 2 × 65,536, temperature 0, thinking off, 16
  steps;
- the same 9B reader at 1 × 204,800 plus the projector;
- the same judge.

**Comparisons:**
1. Against M62 (31.91): the attribution of the rule.
2. Against AgentRunbook-C (82.98): the replication gate is unchanged, at
   least 77.98 and at least 74.90.

**Predictions.**
- Answers with no span fall from 19 to 5 or fewer.
- The combined score rises by at least 10 over M62.
- Replication is **not** expected: the answers with spans scored 54%
  against AgentRunbook-C's 83%, and this arm does not address span choice.

**Falsifier.** Empty answers stay at 10 or more, which would mean the rule
does not reach the forced step.

## M62c — pre-registered *(2026-09-24 ~17:10, before any row)*

**The finding that motivates it.** M62b's first half recorded its actions.
The web half issued 268 `read` actions, and **267 came back "read needs a
trajectory and a state"** (e.g. `read: e9db8fea state ? from 0`, 15 times
in a row on one question). The flat action schema made every field
optional and allowed extra keys. The grammar let the model write `read`
without `state` (or under another name), and it kept doing so. So **M62's
reads almost never worked**, which is the "tool errors" in its notes, and
its 31.91 partly measured a schema defect, not the design.

**The change (commit on `m62c-strict-schema`):**
- **A strict object per tool** under `anyOf`, each with its own required
  fields and `additionalProperties: false`. A `read` without a state
  cannot be written.
- **grep hits say where they are:** `t1 state 2 line 41: …`, and `action`
  / `thought` / `url`, so `read` can open at the hit.
- **`read` shows the recorded thought.** The bottleneck review found the
  thoughts were never surfaced.
- **Every observation ends with the steps left** before the answer is
  forced (BATS, Liu et al. 2025, arXiv 2511.17006).

These are bundled because each fixes a defect or adds information, not
because they are separate mechanisms. The attribution is against M62b.

**Run.** Identical to M62 and M62b otherwise: the same 47 questions,
Bonsai controller at 2 × 65,536 with temperature 0 and thinking off, 16
steps, the 9B reader at 1 × 204,800, the same judge.

**Comparisons:**
1. Replication against AgentRunbook-C (82.98): at least 77.98 and at least
   74.90.
2. Attribution against M62b.

**Predictions.**
- Tool errors fall from ~11 per question to under 1.
- Forced answers fall below 50%.
- Combined score **≥ 55**, above AgentRunbook-R's local 58.6 being the
  stretch.
- Replication (≥ 77.98) is possible but not expected.

**Falsifier.** Tool errors stay above 3 per question, which would mean the
server did not apply the `anyOf` schema.

## M62b and M62c results *(2026-09-24 18:24)* — **36.17 and 25.53; the native controller stays far behind**

| paired over 47, vs AgentRunbook-C 82.98 | score | Δ | forced | empty answers | tool errors | actions |
|---|---|---|---|---|---|---|
| M62 | 31.91 | −51.06 | 36 | 19 | (not recorded) | (not recorded) |
| **M62b** (naming a span is enough) | **36.17** | −46.81 | 34 | 6 | 459 | grep 119, read 455, summary 29 |
| **M62c** (strict per-tool schema) | **25.53** | −57.45 | 0 | 18 | 0 | **summary 248, answer 47** |

- **M62b** did its job on empty answers (19 → 6) and scored +4.3 over M62.
  Its instrumentation found the flat-schema defect: 459 of its reads
  failed for want of a `state`.
- **M62c** removed the errors, but the controller then **never used `grep`
  or `read`**. It answered from summaries alone, and 18 answers named no
  span. M62c vs M62b: −10.64 [−25.53, +4.26].
  - Zero uses of two of the four tools, in 47 questions, looks like the
    `anyOf` grammar not really offering those branches rather than a
    choice. This is unverified: the server's grammar was not inspected.
- **Every M62 prediction failed.** No variant came within 45 points of
  replication.

**What it means for LME-V2.** The path to 74.90 is AgentRunbook-C itself:
the M54 full pair, 13 of 32 chunks done. The native controller's gap is
not one fix away; Cao 2026 (arXiv 2603.20432) predicts that bespoke
schema tools lose to a shell and files. M62 work pauses here, and the
GPU goes to M54.
