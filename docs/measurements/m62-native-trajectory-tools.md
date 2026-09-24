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
