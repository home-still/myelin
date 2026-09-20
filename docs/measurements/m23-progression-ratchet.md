# M23 — the progression ratchet, and three published numbers it was wrong about

**Instrument milestone.** No mechanism ships on. What lands is the check that was missing
when the shipped defaults lost 3.3 points between M16 and M22 without anyone noticing, plus the
wiring that makes M23's mechanism arms reachable from the harness that measures the strata they
target.

Everything below is reproducible offline, with no GPU and no network:

```
cargo run -p myelin-eval -- standing
cargo run -p myelin-eval -- ratchet
```

---

## 1. The defect this exists for

M22 re-measured the shipped configuration on LME-V2-Small and got **36.59**. `standing` had been
publishing **39.91** since M16 — from `runs/myelin_inv2_web_small`, an artifact written before M19
changed what ships.

Nothing caught it, and nothing *could* have, because every instrument in the project points
outward. `standing` joins our artifacts against `docs/sota/registry.json` — a table of other
people's numbers. A system can lose three points against its own past while every row in that
table stays exactly as red as it was. `Ours::arm` (M21) catches "this run turned on a switch that
ships off"; it has nothing to say about a run produced by code whose *defaults* were different.

## 2. What was actually wrong, in the order the fixes landed

Each number below is what `cargo run -p myelin-eval -- standing` printed for
`lme_v2_small.overall_full_set.combined` at that commit stage, against the same `runs/` tree.

| stage | published | why |
|---|---|---|
| before M23 | **39.91** | `runs/myelin_inv2_web_small`, pre-M19, selected on value |
| + drift detection | 39.91, now `stale-config` | flagged, but every candidate was stale so value still chose |
| + drift-size ordering | **39.02** | least-unrecoverable artifact wins — `runs/m22_nodate_web` |
| + harness-path `arm` | **36.59** | `m22_nodate_web` carries `dated: false`, which is an arm |

Three distinct defects, and only the first was known:

**(a) No era signal.** An artifact that does not record a switch cannot be checked against
today's defaults. `runs/myelin_inv2_web_small` records `mode`, `k`, `budget_tokens`, `max_steps`
and nothing else; `runs/m22_base_web` records eleven keys. The drift check fires on **absence**,
not on a value — the point is not that the old run was configured differently, it is that nobody
can check. `Ours::unrecorded` names the missing keys and `Verdict::StaleConfig` refuses the claim.

**(b) Ordering among stale artifacts.** Once every candidate is partly unrecoverable, "prefer the
better number" picks the *oldest* artifact, because the oldest code was measured before the
mechanisms that cost latency and accuracy were added. Ordering by drift size instead picks the
artifact closest to today's code.

**(c) `arm` was never computed on the LME-V2 path at all.** `harness_metrics` hardcoded
`arm: false` from the day it was written. `dated` ships **on** — `apply_operating_point`
suppresses the date machinery only for an explicit `false` — so `runs/m22_nodate_web` is an arm,
and M22 measured that arm as a **null** (+2.4, CI [−1.1, +6.0]). It was being published as where
we stand, at 39.02 against the shipped configuration's 36.59. This is exactly the defect
`Ours::arm` was added in M21 to prevent, unfixed on the benchmark G1 is scored on.

`standing`'s LME-V2 rows now read **36.59** and carry `stale-config`, because no run on disk
records M23's three new operating-point keys. That is the correct state and it is self-clearing:
one fresh pair at the shipped defaults removes it.

## 3. The ratchet

`myelin-eval ratchet` compares today's artifacts against a checked-in floor,
`docs/sota/progression.json`, and exits non-zero on a regression.

Rules, all tested:

- **Only a quotable row may be pinned** — complete, not an arm, no config drift. Pinning an arm is
  the trap the instrument exists to avoid: M21's `runs/m21_full_sel` at 60.40 would pin a floor the
  shipped 56.60 can never clear, every honest run afterwards reads as a regression, and the
  apparent fix is "turn the arm on" — which is how a measured null becomes a default by accident.
- **Direction comes from the metric.** `minja.asr.*` is lower-is-better. A ratchet that compares
  with `<` in both directions inverts silently on exactly the metrics where a regression matters
  most.
- **`--update` only ever raises.** Accepting a regression has to be a deliberate edit to a
  checked-in file, visible in review, not a flag someone reached for to make the build green.
- **Unverifiable is not a pass and not a regression.** A pinned metric whose best artifact is an
  arm, is incomplete, or does not record its operating point is reported loudly and fails under
  `--strict`. That is the M22 state, named.
- **A deleted run does not retire its floor.**

### The floor as pinned on this commit

Eight metrics. The LME-V2 rows are deliberately absent: every artifact for them is stale, and the
ratchet refuses to pin a number it cannot verify.

| metric | pinned | from |
|---|---|---|
| `locomo.judge_score.n1540` | 69.87 | `runs/m19_locomo_full` |
| `locomo.temporal.n1540` | 59.98 | rescored |
| `locomo.abstention_accuracy.n446` | 69.96 | `runs/locomo_recall` |
| `longmemeval_s.judge_score.n500` | 56.60 | `runs/m21_full_base` |
| `longmemeval_s.token_f1.n500` | 47.06 | — |
| `minja.asr.k6_prepopulated` | 77.50 | `runs/attack_live_m18` |
| `minja.asr.k6_prepopulated_defended` | 12.50 | `runs/attack_live_m18` |
| `minja.injection_success.k6_prepopulated` | 92.50 | `runs/attack_live_m18` |

## 4. Measurement legs closed

M23's mechanism arms existed but could not be measured where the headroom is. The strata with the
largest gaps — LoCoMo multi-hop (n=282, +49 answers at the gate profile) and LongMemEval
multi-session (n=133, +50) — are only reachable through `myelin-eval bench`, and four switches
were reachable only from the MCP server or the attack harness.

`bench` now carries `--rerank-pool`, `--premise`, `--typed-probes` and `--untrusted-max`; all four
are recorded in `BenchRun` and read back by `rescore`, and all four make a run an arm.

Two of those matter beyond plumbing:

- **`--premise` implies the insufficiency gate**, exactly as the MCP server does. The analysis
  rewrites the statement the gate emits, so the switch alone is inert — the failure class M12, M14
  and M20 each lost a run to.
- **`--untrusted-max` is reachable from `bench` and not only from `attack --live`.** A defence
  measured for attack success and never for utility is half a measurement. M15 paid for its
  adjudicator's false-positive column over 550 real LoCoMo episodes; a quota that quietly drops
  real evidence would be invisible from the attack harness alone.

## 5. What is NOT measured here

The G3 sweep (`attack --live`, ten conditions including the three quota arms) and every read-path
arm need the reader on `big`. During this milestone the card held a live household voice assistant
(ollama `qwen3:8b`, 10 GB, rolling 30-minute keep-alive) and a 10.5 GB llama-swap model, leaving
~4 GB against the reader's ~7 GB. The run was deferred rather than taken by evicting someone
else's work.

Consequently **no M23 mechanism has a number and every one of them ships off**, which is the
project's standing rule anyway. What this milestone claims is the instrument and the three
corrections above, all reproducible without a GPU.
