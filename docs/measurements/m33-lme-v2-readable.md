# M33 — LME-V2 becomes readable, and two more inverted arm verdicts

**Verdict: `lme_v2_small.overall_full_set.combined` is a comparison again.**
It has been `stale-config` — literally unquotable — since M16. Measured at the
shipped operating point, both domains, every operating-point key recorded:

| domain | full set | non-abstention | abstention | n |
|---|---|---|---|---|
| web | 42.50 | 48.81 | 27.78 | 240 |
| enterprise | 33.65 | 40.65 | 14.29 | 211 |
| **combined (micro)** | **38.36** | — | — | **451** |

`standing` now reports it as `caveat-judge(open_weights_local vs frontier_api)`
with real gaps instead of a refusal:

| reference | bar | ours | gap |
|---|---|---|---|
| AgentRunbook-C | 74.90 | 38.36 | −36.54 (gate) |
| Codex (vanilla) | 69.90 | 38.36 | −31.54 |
| **AgentRunbook-R** | **58.60** | **38.36** | **−20.24** |
| RAG slice+notes | 51.00 | 38.36 | −12.64 |

AgentRunbook-R is the row that matters: LME-V2's reader is Qwen3.5-9B
(2605.12493 L518/L525/L952), the same open-weights model myelin serves, so
this is the only apples-to-apples headroom in the standing table (§11.5).

Unsupported gates stay at **4** — the count does not move, because this row's
gate is AgentRunbook-C's 74.90 and we are behind it. What changed is that
"behind by 36.54" is a fact, where "cannot be quoted" was an absence.

## 1. The `stale-config` was not a plumbing gap

The obvious hypothesis was that the adapter could not record the M23/M24
switches. It can: `run_myelin.py` has written all 12 of `PAIR_KEYS`
unconditionally since M24, with a comment explaining why a key that appears
only when its flag is set would split a pair over a schema difference. The
artifacts were stale because they were **old** — `runs/m22_*` predate the
switches — and no run since had been made on the LME-V2 path.

So this milestone is a measurement, not a migration. The pilot confirmed the
recording before the 2-hour commitment:

```
recorded: mode=investigate k=25 budget_tokens=10000 max_steps=2
          prefetch_limit=None rerank_depth=None select=True dated=False
          pool_rerank=False premise=False typed_probes=False decompose=None
MISSING: none -- all 12 PAIR_KEYS present
```

## 2. Two more inverted arm verdicts, both caused by M32

M32 flipped `InvestigateConfig::select_sufficient` to the `investigate`
default and fixed `Ours::arm` in `bench_metrics` to compare against the
default **for the run's mode**. It missed two places, and every LME-V2 run is
`mode: investigate`, so both were live.

### 2.1 `harness_arm` tested `select` against a constant

`PAIR_SWITCH_DEFAULTS` carried `("select", false)`. With the default now on
for `investigate`, that inverts both verdicts at once: the shipped
configuration reads as an arm and is excluded from "where we stand", while a
`select: false` run — an override — is published as it. This is `Ours::arm`'s
own motivating defect (M21's `m21_full_sel` at 60.40 displacing the shipped
56.60; `standing` publishing 39.91 for four milestones) for the third time.

`select` is now absent from that constant and tested separately against
`shipped_select_sufficient(mode)`, the same helper M32 added, which reads the
library defaults rather than hardcoding them. Pinned by
`harness_select_is_an_arm_only_against_its_own_modes_default` (all four
corners) and `an_unset_select_is_never_an_arm`.

The observable consequence, before the fix: `standing` published **39.02**
for this row — `runs/m22_nodate_web`, an arm carrying a switch M22 measured as
a null — because after M32 every artifact on disk was either an arm or stale
and the ordering fell through to value.

### 2.2 The adapter wrote a `select` it had not sent

`MyelinMemory.query` sent the key only when true:

```python
if self.select:
    arguments["select"] = True
```

Before M32 that was equivalent to sending `false`, because the server's
default was off. After M32 an omitted key means the server turns the selector
**on**, while `memory_config.json` records `select: false`. The artifact would
have described a run that did not happen — and `standing` reads exactly that
key to decide which number gets published.

Now sent unconditionally as `bool(self.select)`. This is the failure mode
`PAIR_KEYS`' own comment warns about, arriving from the opposite direction:
not a key that is absent, but a key that is present and wrong.

## 3. The harness path could not see its selector either

`bench` was blind to its own mechanism until M32. The LME-V2 path — the
benchmark G1 is scored on — still was: it drives retrieval over MCP, and

- `RecallTraceJson` reported everything about the fusion (`dense_hits`,
  `lex_hits`, `fused`, `reranked`, `admitted`, four timers, `top_score`,
  `abstained`) and **nothing** about the one stage that can silently do
  nothing;
- `investigate` returns its whole `InvestigateTrace`, which has carried
  `select_degraded` since M32, and the adapter discarded it.

Both fixed. `RecallTraceJson` now carries `selected`, `select_ms` and
`select_degraded`; `MyelinMemory` appends one JSON line per query to
`trace_path`, which `run_myelin.py` points at `<run>/myelin_trace.jsonl`.
`trace_path` is deliberately **not** in `PAIR_KEYS`: it changes nothing about
what the server does, only whether the run can prove what it did, so it must
not affect a pair's fingerprint.

Written per query rather than aggregated so a run that dies mid-way still
says what happened up to that point — which is what the first attempt at this
run did (§5).

### What it shows, first time on this path

| | none | model_declined | **call_failed** |
|---|---|---|---|
| enterprise (n = 211) | 181 | 30 | **0** |

Zero failed calls: the selector ran on every query. That certification did not
exist on the LME-V2 path before this milestone, and it is what lets 38.36 be
quoted as the selector's number rather than assumed to be.

`selected` spreads across 1–20 of a `k = 25` window with a median of 3, and
`steps` is 2 on 185 of 211 questions.

## 4. The M32 cause-split is load-bearing, and this run proves it

M32 split `Degradation` into `ModelDeclined` and `CallFailed` because the
first version of `DegradationGuard` counted both and refused a healthy
LongMemEval_S arm at 11/298. That looked like a close call. On LME-V2 it is
not close:

| run | declines | rate | union-gated | cause-gated |
|---|---|---|---|---|
| LongMemEval_S `investigate` | 15/500 | 3.0% | `wilson_lower = 0.0183` → accepted, by 0.0017 | 0.0000 → accepted |
| **LME-V2 enterprise** | **30/211** | **14.2%** | **`wilson_lower = 0.1014` → REFUSED** | **0.0000 → accepted** |

A guard on the union of causes would have **refused this measurement
outright**, at five times the floor — and LongMemEval_S cleared it by less
than two thousandths, meaning one further decline would have refused that one
too. The decline rate is a property of the corpus (14.2% vs 3.0%), not of the
system's health, which is exactly why it cannot be the gated quantity.

`ablate::width_verdict` carried the same conflation from M27 until M32.

## 5. What is still unreadable, and why it is not a code fix

`lme_v2_small.lafs_gain.small` remains `stale-config`. This is correct
behaviour, not a residual bug. `pair_metrics` computes `lafs_unrecorded` as
the **union** over every pair feeding the LAFS point:

> The LAFS point is computed over every pair, so it inherits the drift of
> every pair that fed it: one unrecoverable operating point in the submission
> set makes the frontier it was scored against unreadable.

LAFS is a latency–accuracy *frontier* over submitted points, so a point that
cannot be described poisons the frontier rather than merely being omitted
from it. `runs/m22_{base,sel,nodate}_{web,ent}` are in that set and predate
the M23/M24 keys.

Making it readable therefore requires a fully-recorded submission set, not an
instrument change: either re-run the M22 arms on this code, or decide that
superseded exploratory arms are not submission points and separate them from
`runs/`. That is a decision about what "our submission" means, and this
milestone does not make it silently.

## 6. The incident

The first enterprise attempt died at 38 minutes. `big` reached load average
**177** with 467 MB available, SSH timed out during banner exchange, and the
adapter went into its retry loop (`investigate attempt 4/6 failed (empty
response body)`).

Cause was mine and it was PLAN §13 by proxy. §13 says concurrency on `big` is
bounded by host RAM, not VRAM — a rule added after M27, when I took the host
down running two sweeps at once. This time I told a documentation subagent
that the scribe server was free and to convert five papers with it, while I
held the GPU for this run. `scribe_convert` launches olmOCR under vLLM.
Delegating work onto the contended host is the same mistake as running it
myself.

Two operational notes worth keeping:

- **`| tee` masks the exit code.** The earlier M32 certification run reported
  `exit=0` while the binary had failed, because `tee` succeeded. Every run
  wrapper in this milestone uses `set -o pipefail`.
- **`free -g` on `big` now shows 32 GB of swap.** The note in this project's
  operating memory that the host has "no useful swap" is stale. What bit was
  `MemAvailable` at 467 MB against an RSS total of only 10.8 GB — 5.1 GB of
  it `Shmem` from the graphics/CUDA stack, with `/dev/shm` empty at 84 KB.

The web result was already on disk and survived; only enterprise was re-run.

## 7. Reproduction

```sh
# One GPU job at a time (PLAN §13). Check `free -g`, not `nvidia-smi`.
ssh big gpu-tenant claim coding

# The harness's SCORING stage embeds the reader's full raw response AND the
# parsed prediction, so its judge prompts run ~2x --max-completion-tokens.
# At 20000 that is ~40k tokens; a 400 there destroys every generation in the
# run, because harness.py holds them in memory until scoring.
# 1 slot x 65536 and 2 slots x 32768 are the SAME total KV, so re-splitting
# costs no VRAM and buys the per-slot headroom.
ssh big "MYELIN_READER_SLOTS=1 MYELIN_READER_CTX=65536 bash -s" \
    < ops/big/serve-models.sh
ssh -N -o ServerAliveInterval=15 -L 5810:127.0.0.1:5810 \
    -L 5813:127.0.0.1:5813 big &

# Pre-flight the scoring stage with REAL records, not filler: filler
# tokenises at ~6 chars/token and real records at ~4.4 (M27).
# Measured: 52,289 prompt tokens, HTTP 200.

MYELIN_QDRANT__URL=http://192.168.1.110:6334 myelin-mcp --serve 127.0.0.1:7446 \
    --collection myelin_lme_v2_small --ledger data/lme_v2_small.ledger &

cd crates/myelin-eval
export OPENAI_API_KEY=local
for domain in web enterprise; do
  PYTHONPATH=vendor/longmemeval-v2:adapters ../../.venv/bin/python \
    adapters/run_myelin.py --data-root ../../data/lmev2 \
    --domain "$domain" --tier small \
    --k 25 --budget-tokens 10000 --mode investigate --max-steps 2 \
    --select --undated \
    --evaluator-base-url http://127.0.0.1:5810/v1 \
    --evaluator-model Qwen/Qwen3.5-9B --reader-model Qwen/Qwen3.5-9B \
    --output-dir "../../runs/m33_$domain"
done
```

`--select` is passed explicitly because it is now the server's default for
`investigate`: the flag makes the artifact record what the server was going to
do anyway, which is the point of §2.2.

Wall clock: web 1h09m (240 questions), enterprise 1h04m (211).
