# M54 — a local file-reading controller for LME-V2 *(design, 2026-09-23)*

## Why

M53 located the LME-V2 gap: 45% of wrong phrase/list answers are in the
haystack but never delivered, and the missing answer is a specific UI string
in a specific transition (a banner after "Save", a dropdown's options). Three
cheap retrieval fixes were probed and none reaches enough rows (1 of 6 live;
19 of 65 and 3–4 of 65 offline). The LME-V2 paper's result is the direction:
a **coding agent over trajectory files** reaches **72.5** against the best
RAG system's **48.5** with the *same* Qwen3.5-9B reader
(`10.48550/arXiv.2605.12493` §4.2, Table 2) — the controller, not the reader,
is what differs.

## What AgentRunbook-C is (from the paper, Tables 10–11)

A per-question sandbox: `question.json`; `INSTRUCTION.md` (a workflow: act
as a quick memory module, shortlist from the manifests, inspect, do not
over-explore); `trajectories/<id>/trajectory.json` (full goal, actions,
states); two manifests (`TRAJECTORY_SUMMARY_CONCISE.md` for triage,
`…_FULL.md` with thoughts and actions for shortlisting); and
`scripts/inspect_trajectory.py` (view a trajectory, a state, a span, or match
text within one). The agent writes `memory_module_output.json`:
`{memory_markdown, trajectory_spans: [{trajectory_id, start_state_index,
end_state_index}]}`, rendered into the reader's context. Controller: Codex +
GPT-5.4-mini, ~108 s/query. The vendored harness ships it
(`memory_modules/agentrunbook_c.py`, `codex.py`), and `codex.py` passes
arbitrary `-c` config to the Codex CLI.

## Local-only constraints (hard requirement)

- Controller models available: Qwen3.5-9B (big, the reader); Ternary Bonsai 2
  27B (big `:8081`, llama-swap, tool calls, 96k); qwen3-32b / qwen3.8-27b on
  bmb (Metal, ~10 tok/s).
- llama.cpp on big serves `/v1/responses` (verified 2026-09-23), the API the
  OpenAI Codex CLI speaks.
- Not installed: the OpenAI Codex CLI (the `codex` on this machine is a
  different tool). `omp`, the home cloud's coding agent, is installed and
  configured for the local models.
- Cost: ~100–300 s/query locally; a 451-question pair is 13–40 GPU-hours, so
  every option pilots on a stratified subset first (~40 questions).

## Options

1. **Vendored AgentRunbook-C, local model.** Install the OpenAI Codex CLI and
   point `codex.py`'s `extra_config` at a local provider. Least code; replicates
   the paper's system exactly with a local controller — the fastest answer to
   "can a local controller close the gap?". But it measures AgentRunbook-C,
   not myelin, and Codex's prompts are tuned for GPT models.
2. **omp over an exported sandbox.** Same sandbox layout, driven by the coding
   agent the home cloud already runs locally. No new install; integration
   unknown.
3. **Native tools in myelin's `investigate`.** `grep` over page text and
   thoughts, `open(traj, state[, span])`, `diff(traj, state)` against the
   previous state, as tools the loop's controller calls; spans rendered into
   the evidence set. The most engineering, and the only option that makes
   myelin itself better. Option 1 or 2 first would tell us whether it is worth
   building before it is built.

The choice is the user's (see the session note of 2026-09-23).
