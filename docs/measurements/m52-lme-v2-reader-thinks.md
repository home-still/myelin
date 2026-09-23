# M52 — the LME-V2 reader thinks, as the harness intends *(pre-registered 2026-09-23)*

## What was found

Every LongMemEval-V2 number this project has published ran the harness
reader with thinking **off**, and the harness's own default is **on**:

| | reader thinking | completion ceiling |
| --- | --- | --- |
| vendored `evaluation/run_eval.py` | `--reader-enable-thinking` default **True** | 20,000 |
| vendored `evaluation/harness.py` | `set_defaults(reader_enable_thinking=True)` | — |
| our `adapters/run_myelin.py` | default **False** | 20,000 |

The adapter turned it off on a measurement: without a cap, Qwen3.5-9B
spent the whole completion on `reasoning_content` and returned
`content: ""`. That was the right call for a reader with no budget. Since
M44 R2 the server caps the trace (`--reasoning-budget 1024`), and on
LongMemEval_S the same reader, thinking under that cap, went **67.80 →
78.40** on two seeds.

It is also a comparability problem, not only a lever. AgentRunbook-R's
**58.6** — the registry's comparable row, "Qwen3.5-9B controller and
reader" (`10.48550/arXiv.2605.12493`) — was produced by a harness whose
reader thinks by default. Our −20.02 gap compares a non-thinking reader
against a thinking one.

And turning it on was broken. `harness.build_extra_body` sends
`chat_template_kwargs: {enable_thinking: false}` when thinking is off and
**nothing** when it is on, trusting a vLLM server that thinks by default.
Ours is llama.cpp with thinking off server-wide (so the evaluator, which
shares the endpoint and never sends the flag, stays a plain judge). So
`--reader-enable-thinking` reached our reader as no flag and ran with
thinking off, silently. Verified on the live server: the harness's request
body was `{'top_k': 20}`.

## The fix

`run_myelin.py --reader-enable-thinking` now:

1. **Preflights** one `enable_thinking: true` request before building
   anything: the reply must carry a non-empty `reasoning_content` and a
   non-empty answer, or the run ends at the door. Verified 2026-09-23 on
   bmb's reader: a 2,765-character trace and the correct answer (`35` days).
2. **States the flag explicitly** by wrapping the harness's own
   `build_extra_body` (the vendored file stays unmodified, `PLAN.md` §3.3):
   every field it computes is kept and `chat_template_kwargs:
   {enable_thinking: true}` is added. The evaluator builds its requests
   elsewhere and is untouched — it stays a non-thinking judge in base and
   arm alike.
3. Records `reader_thinking` in `memory_params`, which `standing` reads as
   an arm (`PAIR_SWITCH_DEFAULTS`, absent = false).

## Pre-registration

**Population.** LME-V2 tier-small, web 240 + enterprise 211 = 451;
abstention 128, answerable 323 (the harness's split).

**Base.** `runs/m47_base_{web,ent}` — today's memory defaults on the
rebuilt store, reader thinking off. Shared with M47.

**Arm.** The same pair with `--reader-enable-thinking` →
`runs/m52_think_{web,ent}`; reader served with the projector
(`MYELIN_MMPROJ=1`, the harness sends screenshots) and the shipped
1,024-token budget, no budget message (M44 R2b measured it below the bar).

**Deviation, stated.** The harness intends up to 20,000 completion tokens
of thinking; we cap the trace at 1,024. That is the configuration that
works on this server and the one LongMemEval_S ships. A larger budget is a
separate arm (as R2c is on LongMemEval_S).

**No exact control exists on either side.** The harness samples its reader
(temperature 0.6, top-p 0.95, top-k 20, no seed), so no row is
byte-identical between any two LME-V2 runs; the paired bootstrap carries
that noise. Consequence for scheduling: this arm may share the reader with
a build pass without losing a control it never had.

**Primary metric.** `lme_v2_small.overall_full_set.combined`, paired
bootstrap over `web+enterprise` (`adapters/paired_ci.py`), with the two
strata.

**Decision rule.** Ship `reader_enable_thinking` on for LME-V2 at **≥ +3.0**
combined with the paired 95% CI excluding zero, **and** no drop on the
abstention stratum (M42's veto).

**Predicted.**

1. Combined **+5 or better**. LongMemEval_S moved +10.6; LME-V2's reader
   reads 10,000 tokens of trajectory evidence, where a trace has more to
   organise, but its questions are less arithmetic.
2. Abstention stratum: **no drop, and a gain** — M44 R2 raised abstention on
   LongMemEval_S; thinking is where a 9B notices a premise does not hold.
3. Answerable: **+3 or better**.
4. Per-row latency roughly doubles in the reading stage; prompt building
   (the memory side) is unchanged.

**Falsifier.** A gain concentrated in rows where the budget was not hit
and a loss where it was means the cap is binding on LME-V2's long evidence
— the next arm is the budget, not the reader.

**Cost.** One LME-V2 pair: ~1 h of prompt building and ~30–60 min of
thinking reads per domain.

## Results

*(pending)*
