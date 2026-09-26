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
- **Context:** Codex is given the same budget as on big
  (`model_context_window = 96000`). bmb's llama-swap turned out to serve
  196,608 tokens, not the 64K its docs listed (read from its `/running`
  at the first chunk), so the budget is Codex's and identical on both hosts.
  The one difference between the controllers is the pack.
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
requests. **Passed 2026-09-24 11:30:** `edea0219` took 717 s and returned
5 memory items, with 0 failures and 0 refused requests. That is about 2.7×
slower than big (median ~265 s). Each chunk also records the host's serving
command (`controller_running.json`).

**Withdrawn 2026-09-24 11:45, before any bmb chunk completed.** At about 12
minutes per question, bmb added only ~1.5× throughput. Its Bonsai server
held ~21 GB of that daily-driver Mac's memory, leaving 20% free. The user
chose to stay local on big alone. bmb's one in-progress chunk (`web_c08`)
was discarded and its claim released. **No bmb-controlled chunk is in the
measurement**, so the controller is Bonsai PTQ1_0 on big throughout, exactly
as first pre-registered. The claim-based worker stays; with one host it is
the old sequential runner.

## Amendment 2 — more controllers, same model *(pre-registered 2026-09-24, before any of their chunks)*

The user asked for every machine that can help. The controller stays
**the identical model file** on every host:
`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, sha256 prefix `53107f530aa52eb0`,
checked on each host after copying. The flags are also identical: 96K
context, q8_0 KV cache, temperature 1.0, top-p 0.95, top-k 20, min-p 0,
repeat penalty 1.0, thinking budget 8,192, reasoning effort "medium", and
the patched chat template, all copied from big's `run-qwen3.8.sh`.

| host | hardware | runtime | measured decode | Codex budget |
|---|---|---|---|---|
| big | RTX 3090 | PrismML fork `1a07bfa` (CUDA), llama-swap `qwen3.8-27b` | 57.7 tok/s | 96,000 |
| **sib** | RTX 3060 12 GB | the same CUDA build, copied from big | 27.1 tok/s | 96,000 |
| **big_mac** | M1 Max | PrismML fork build 10709 (Metal) | 17.5 tok/s | 96,000 |
| **big, 2 slots** (after the quick fixes) | RTX 3090 | the same build, our own server with `-np 2 -c 131072` | ~1.3× in total (measured on the 9B) | **64,000** |

**Not on the new hosts:**
- The vision projector is left off on sib and big_mac. Its only input is
  images, and the Responses shim strips every image before it reaches a
  controller.
- sib's GPU held another agent's Ollama `qwen3:8b` (9.8 GB, pinned). The
  user approved unloading it for this run.

**Rules, unchanged from amendment 1:**
- Chunks are claimed atomically. big works forward; sib and big_mac work
  backward.
- `controller.json` records host, slots and Codex budget for every chunk.
- Transport failures are discarded and re-run. Anything else is the
  controller's behaviour and is kept.
- The pair is scored once, with accuracy also reported by host and
  configuration, each with a 95% CI.

The 2-slot mode is the one departure in configuration: 64K per question
instead of 96K. It is reported separately.

**Smoke tests:** one question per new host before its first chunk.


**big_mac dropped, 2026-09-24 12:49, before any big_mac chunk.** Its smoke
question hit the 1,800-second controller timeout with **no memory written**.
Its real decode on long agent contexts was 6.2–7.3 tok/s, with ~85 tok/s
prefill, well below the 17.5 tok/s short test. A 24-question chunk would take
most of a day, and one slow host holding a tail chunk would delay the finish.
The server is stopped. **No big_mac-controlled chunk is in the measurement.**

**sib passed its smoke question, 2026-09-24 12:58.** One question end to end
in 916 seconds, with 9 memory items written, no failed attempts and no
transport errors. Its decode on long agent contexts is 20–23 tok/s, with
~240 tok/s prefill. The sib worker started at 12:58.

**Chunks halved, 12:58, before sib's first chunk and big's 2-slot restart.**
At ~15 minutes per question, a 24-question chunk on sib takes ~6 hours, and
the last chunk still running sets the finish time. Each unstarted chunk was
split into two halves of 10–12 questions (`web_c08` → `web_c08a` + `web_c08b`),
so 15 chunks became 30. The question set is unchanged: the sorted
question-id lists hash identically before and after the split. `ent_c00`
and `ent_c01`, already done by big, are untouched. Chunking is only how the
work is shared out; the pair is still scored once, over all 451 questions (the 47-question pilot plus these 404).

**Merge and report tooling, 2026-09-24, before any merge.**
- `adapters/merge_arc_chunks.py` writes each domain's merged prompt set:
  every source's `prompt_rows.jsonl` plus a `controller_hosts.json` giving the
  mixture and the configuration of every question (`big/1x96000`,
  `big/2x64000`, `sib/1x96000`). It refuses a source without
  `controller.json`, a repeated question, or a different model file. The
  reader phase then refuses any merge that does not cover exactly the domain.
- `adapters/arc_by_controller.py` prints the pre-registered by-configuration
  breakdown, reading each question's configuration from the arm's own
  `memory_config.json`. On the pilot, with a stand-in manifest, its "all"
  row reproduces the pilot's result exactly (+40.43 [+25.53, +55.32]).
- The pilot's two prompt directories gained a `controller.json` (big,
  llama-swap `qwen3.8-27b`, 1 slot, 96,000), taken from their recorded Codex
  parameters, so all 451 questions name their controller.

## Amendment 3 — big rejoins at 1 slot, with a RAM cap *(2026-09-25, before any of its chunks)*

- **What happened.** Overnight, big ran 10 half-chunks at 2 × 64K. At 09:31
  a host-RAM OOM stopped both workers; `ent_c07a/b` were discarded under the
  transport rule. Our Bonsai server held 10 GB of swap. Most of that was
  llama-server's host-RAM prompt-state cache (`--cache-ram`, default
  8 GiB; 183 evictions in its log).
- **The 2 × 64K timeouts.** That setup produced all 11 of the full run's
  1,800-second timeouts, 8 of them with empty memory. Each followed a Codex
  history compaction; 1 × 96K compacted in only 2 of 95 questions. Those
  rows stay in the measurement and are reported under their own
  configuration, as amendment 2 fixed.
- **big returns at 1 × 98,304**, the configuration first pre-registered
  (Codex budget 96,000, worker `big1`, `codex_home_big1`), plus two
  operational changes:
  1. `--cache-ram 0` switches off only the host-RAM cache of saved prompt
     states. The slot's own KV reuse within a question is untouched.
  2. The server runs under `gpu-tenant run --mem 14G`, so an OOM kills our
     server, not big.
  Throughput at 1 slot matched 2 slots (~10 vs 10.3 q/h).
- **When it starts.** big's GPU is held by other tenants. The user's call:
  wait, and evict no one. `scratchpad/m54_big_wait.sh` polls
  `gpu-tenant status` every 5 minutes and starts only once ≥ 11,000 MiB of
  VRAM and ≥ 12 GiB of RAM are free.
- **Progress at this amendment:** 180 of 404 full-pair questions done. sib
  runs from the back.

## Amendment 4 — the cloud Bonsai finishes the run *(pre-registered 2026-09-25, before its smoke question and any of its chunks)*

- **Why.** At 18:00 the local run was at 0 questions per hour. sib (the
  kids' PC) went offline at 17:52. big had ~9.6 GB of VRAM free against the
  ~11 GB the controller needs, with six other tenants' processes holding
  ~14.6 GB. Measured levers could not rescue it:
  - PTQ1_0 decoding is compute-bound, so 2 slots give 1.1×;
  - no small draft model shares Bonsai's vocabulary, and Bonsai has no MTP
    layers;
  - mac_air is too slow.

  **The user's call (2026-09-25):** OpenRouter's Bonsai finishes the
  remaining questions.
- **The controller.** `prism-ml/ternary-bonsai-2-27b` on OpenRouter, pinned
  to its one provider (`Darkbloom`, no fallbacks). OpenRouter lists that
  endpoint as **int4**, 262K context. It is Ternary Bonsai 2 27B, but **not
  the local PTQ1_0 file**. Whether int4 is a lossless repacking of the
  ternary weights is not stated anywhere we can check.
- **The same settings, sent per request.** The Responses shim
  (`adapters/responses_shim.py`, cloud mode) injects the pre-registered
  sampling that llama-server carried as defaults: temperature 1.0, top-p
  0.95, top-k 20. Codex sends reasoning effort "medium" as before.
- **What differs:**
  - the 8,192-token thinking budget is not enforced by this endpoint;
  - the chat template is the provider's.
- **Unchanged:**
  - Codex budget 96,000;
  - the harness and prompts;
  - the 1,800-second per-question timeout;
  - the claim directory, so no chunk runs twice.
- **Transport rule.** A chunk with any transport failure is discarded and
  re-run. For the cloud this also counts HTTP 429, 502, 503 and 504, which
  are rate limits and gateway errors, not controller behaviour.
- **Provenance.** Every cloud chunk's `controller.json` names host
  `openrouter`, the served model, the provider and `int4`.
  `merge_arc_chunks.py` accepts exactly the two pre-registered controllers
  (the local PTQ1_0 file, and this endpoint) and refuses anything else.
- **Scoring.** The pair is scored once over all 451 questions. It is also
  reported by controller configuration, each with a 95% CI, as amendment 2
  fixed.
- **Caveat on the headline:** *controller partly cloud-served*. The 180
  local questions stay as they are.
- **Smoke test first.** One question (`edea0219`, as in amendment 1) run end
  to end through the cloud must produce a non-empty memory context with no
  refused request. The smoke output is not part of the measurement.
- **Parallelism:** up to 4 cloud workers, sized to rate limits after the
  smoke. The local workers (big, sib) may still claim chunks whenever their
  hardware is free.
- **Cost:** ~$8 for the remaining 224 questions, estimated from ~410K
  cumulative input tokens and ~9.5K output tokens per question.

**Amendment 4 smoke test passed, 2026-09-25 18:30.** `edea0219` went end
to end through the cloud controller:
- status finished;
- 5 memory items (8,303 tokens), one span (`2e8f6477:6-6`);
- no refused requests, no transport failures;
- 1,239 s, under the 1,800-second timeout.

That is slow for a provider that answers a 13K-token, tool-bearing,
streamed request in ~12 s and decodes at ~63 tok/s (measured directly). A
Codex turn took 83–210 s. The shim now logs per-request timing
(`SHIM_TRACE_LOG`: first byte and total, never content) to find where the
time goes. The four cloud workers started at 18:36 on `ent_c07a`,
`ent_c07b`, `web_c00a` and `web_c00b`.

**The reader pass stays local (user, 2026-09-25).** The pre-registered
reader (the 9B at 1 × 204,800 + projector) needs ~23 GB on big. The user
chose to wait for big rather than use bmb or a cloud reader, and to evict no
one. `scratchpad/m54_finish_full.sh` runs the end unattended:
1. wait for all 32 chunks;
2. merge each domain (the pilot plus its chunks) with `merge_arc_chunks.py`;
3. poll big's own `gpu-tenant status` until ≥ 23,000 MiB of VRAM are free;
4. read and judge both domains under a renewed lease;
5. write the pre-registered pair (`lmev2_strata.py` vs `m47_base`) and the
   by-controller breakdown (`arc_by_controller.py`).

**Cloud throughput as measured.** Four workers share the one provider.
Per-request latency is ~30 s at the median with four in flight, against
~12 s alone, so the provider is throughput-bound and more workers do not
help. That is ~20 questions an hour.

## Amendment 4b — the cloud controller could search the web; its chunks are discarded *(2026-09-25 21:35, before any cloud chunk was accepted)*

- **What happened.** Codex sends its built-in hosted `web_search` tool with
  every request. Our llama-server never exposed it to the model. OpenRouter
  executes it server-side as `openrouter:web_search`.
  - The first cloud chunk to finish, `web_c00a`, failed on eight such calls
    ("Server tool "openrouter:web_search" failed: upstream returned 502").
  - So the cloud controller *could* reach the web: outside knowledge on a
    memory benchmark.
- **What is kept and what is not.**
  - None of the 15 accepted chunks, and neither pilot, contains a single
    `web_search` mention in its traces. All ran on local controllers.
  - The four cloud chunks in flight (`ent_c07a`, `ent_c07b`, `web_c00a`,
    `web_c00b`) were stopped and moved aside as `*.discarded.websearch`.
  - Their claims were released. No cloud row is in the measurement.
- **The fix.** The shim's cloud mode forwards only `function` tools, exactly
  what llama-server offers the model. Each request's trace line records the
  tool types it dropped (`dropped_tools`).
- **Before the cloud workers resume,** a new smoke question must show
  `web_search` dropped from its requests and no web search in its trace.
