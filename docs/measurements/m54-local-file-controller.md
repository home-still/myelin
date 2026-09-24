# M54 — a local file-reading controller for LME-V2 *(pilot measured 2026-09-24: +40.4, full pair running)*

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

## Decided (2026-09-23)

**Route 1**, with **Ternary Bonsai 2 27B** as the controller, served on big's
3090 through the PrismML fork (re-enabled in big's llama-swap as
`qwen3.8-27b` at the user's request; ~10.3 GiB with 96k context, 57.7 tok/s).

What made it run locally, measured:
- The OpenAI Codex CLI (0.156.1) with an isolated `CODEX_HOME` whose provider
  is local; it speaks the Responses API, which llama.cpp serves.
- A shim (`responses_shim.py`, scratch) that folds Codex's extra developer
  messages into `instructions`: Qwen-family chat templates reject any system
  message that is not first ("System message must be at the beginning").
- `codex exec` must get `< /dev/null`; it otherwise waits for stdin forever.
- Mechanics verified with the 9B: a shell command run and the right answer,
  12,891 input tokens for a one-step task.

Driver: `adapters/run_agentrunbook_c.py` runs the vendored AgentRunbook-C
unmodified in two phases, because the 27B controller and the 9B reader do not
fit on the card together: `--prompts-only` (controller builds and saves every
prompt) then `--reuse-prompts-from` (reader answers, harness scores).

## Pre-registration — pilot

**Population.** `m54-pilot-web.txt` (24) + `m54-pilot-enterprise.txt` (23):
every 11th question within each category by question id, 47 questions, 14
abstention (30%, against 28% in the tier). Base on these rows,
`runs/m47_base_{web,ent}` (today's shipped myelin, undated): **42.55** (web
45.83, enterprise 39.13; whole tier 38.80).

**Arm.** AgentRunbook-C, controller Bonsai 27B via Codex (reasoning effort
medium), evidence `axtree` (the controller has no vision projector), reader
Qwen3.5-9B non-thinking, judge Qwen3.5-9B — the base's reader and judge.

**Reported.** Paired difference on the 47 with bootstrap CI; the abstention
14 separately; controller wall time per question; how often the controller
fails or times out (a failure returns no context and is scored as the reader
does with none).

**Gate, not ship.** n = 47 resolves only large effects. AgentRunbook-C with a
frontier controller sits 13.9 above AgentRunbook-R and 34 above our 38.80
tier score; a local 27B is expected to land well short of 72.5. **≥ +10 on
the pilot** → a full pair (or a larger pilot, given ~2–5 min/question);
**below +5** → the local controller is not the lever at this size, recorded.

**Falsifier.** If the controller mostly fails to produce spans (errors,
timeouts, empty output), the result measures the harness under a small model,
not file-based retrieval — report the failure rate first.

## Run log

- **2026-09-23 17:01.** First controller attempt: every question refused at
  the door. The harness resolves trajectory screenshots even for `axtree`
  evidence, and the official screenshot bundles were not downloaded. Fixed
  by downloading them (all 5,095 references in the pilot's 200 trajectories
  resolve).
- **2026-09-23 23:14.** Second attempt: 5 of the first 9 web questions died
  with HTTP 400 `Output of tool call should be 'Input text'`. Codex's
  `view_image` tool returns its image inside the tool output, and llama.cpp's
  `/v1/responses` accepts only text there. The harness then hands the reader
  an empty memory context, which would have measured the plumbing, not the
  controller. Stopped at 23:41. `adapters/responses_shim.py` now replaces an
  image in a tool output with a text note, which keeps the controller
  text-only as pre-registered. A captured failing request returns 200
  through it, and the question that died completes with 12 memory items.
- **2026-09-23 23:50.** Relaunched from clean run directories.
- **2026-09-24 03:53.** The first reader pass sent empty answers for two
  web rows. Their controller contexts ran 162,125 and 198,651 tokens, over
  the reader's 64k slots. The harness's own memory budget is 200,000 tokens
  and Qwen3.5-9B's window is 262k, so the reader was re-served as one slot:
  262,144 tokens ran out of VRAM beside big's other tenants, and 204,800 fit.
  The 198,651-token prompt was probed through it, and both reader passes
  then ran with zero overflows. The controller phase was not re-run: its
  prompts were reused byte-identical (`--reuse-prompts-from`).

## Results — pilot, measured 2026-09-24

**42.55 → 82.98, +40.43 [+25.53, +55.32]** on the 47 pre-registered
questions (web 24 + enterprise 23), against `m47_base_web` +
`m47_base_ent` on the same questions. The gate was +5, so it passes by 35
points.

| stratum | n | myelin (M47 base) | AgentRunbook-C, local | Δ | 95% CI |
|---|---|---|---|---|---|
| combined | 47 | 42.55 | 82.98 | +40.43 | [+25.53, +55.32] |
| answerable | 33 | 54.55 | 90.91 | +36.36 | [+18.18, +54.55] |
| abstention | 14 | 14.29 | 64.29 | +50.00 | [+21.43, +78.57] |
| static | 13 | 46.15 | 92.31 | +46.15 | [+7.69, +76.92] |
| dynamic | 9 | 55.56 | 88.89 | +33.33 | [+0.00, +66.67] |
| procedure | 7 | 85.71 | 100.00 | +14.29 | [+0.00, +42.86] |
| gotchas | 4 | 25.00 | 75.00 | +50.00 | [+0.00, +100.00] |

**The falsifier did not fire.** After the shim fix the controller returned
a memory context for all 47 questions, with no refused requests.

| controller | web | enterprise |
|---|---|---|
| questions with empty memory | 0 / 24 | 0 / 23 |
| memory items per question, median | 7.5 | 8 |
| minutes per question, median | 4.4 | 3.6 |
| minutes per question, max | 12.0 | 17.0 |

**No answer leakage.** The controller's sandbox holds the question text,
`INSTRUCTION.md`, the inspection script and the question's haystack
trajectories (102 for the first web question). The question record's
`answer` field is not copied in.

**What it means, with its caveats.** The paper's own finding reproduces
locally, and more strongly: a coding agent that searches the trajectory
files beats retrieval on this benchmark. The paper reports 72.5 against
RAG's 48.5 with GPT-5.4-mini; here it is 82.98 against our RAG's 42.55
with a local 27B. The comparison is a stratified pilot of 47 (the CI spans
30 points) and is not the tier population, so it is not quotable against
the 58.60 standing row. The full pair is the measurement that is.

Artifacts: `runs/m54_arc_web`, `runs/m54_arc_ent` (reader phase, scored)
and `runs/m54_arc_{web,ent}_prompts` (controller phase).

## Full pair — running

The remaining 404 questions (web 216, enterprise 188) run through the
controller in 17 chunks of 24. A chunk writes its own prompt rows, so the
run resumes chunk by chunk and can yield big between chunks. At ~4 minutes
per question that is ~27 GPU-hours. Then one reader pass over all 451 (the
pilot's 47 prompt rows merged with the chunks'), on the 204,800-token
reader, judged by the 9B, paired against `m47_base_{web,ent}`.

## Amendment — a second controller host *(pre-registered 2026-09-24, before any of its chunks)*

At about 4–7 minutes per question, the full pair is ~36 GPU-hours on big
alone. The user approved bmb (M4 Pro) as a second controller:
- **Model:** `bonsai-2-27b`, the same Ternary Bonsai 2 27B in its PQ2_0 pack
  (2.13 bpw; big serves PTQ1_0 at 1.75 bpw), through bmb's llama-swap.
- **Context:** 64K (big's: 96K). Codex is told so
  (`model_context_window = 64000`).
- **Path:** a second Responses shim on :5822.

**Rules, fixed now:**
1. Chunks are claimed atomically: one `mkdir` per chunk under
   `scratchpad/m54_full/claims/`. big works forward and bmb backward, so no
   chunk runs twice.
2. Every chunk writes `controller.json`: host, model alias, pack. The two
   chunks big finished before this amendment (`ent_c00`, `ent_c01`) are
   recorded as big / `qwen3.8-27b` / PTQ1_0.
3. The full pair is scored as one run. Accuracy is also reported **by
   controller host**, with a 95% bootstrap CI for each. If the hosts differ
   beyond their CIs, that is reported next to the headline; it is not
   averaged away.
4. The transport-failure rule applies to both hosts: a chunk with any
   transport failure is discarded and re-run. A context-window overflow on
   bmb is **not** transport. It is that controller's behaviour and stays in
   the measurement, counted per host.

**Smoke test** before bmb's first chunk: one question, run end to end
through bmb, must produce a non-empty memory context with no refused
requests.

