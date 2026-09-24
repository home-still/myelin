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
- [ ] **PR 5 — wiring and the pilot.** An MCP / adapter switch, then the
      47-question pilot against M54's 82.98 when a GPU is free.
