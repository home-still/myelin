# M32 — the pool-level selector earns the `investigate` default

**Verdict: `InvestigateConfig::select_sufficient` ships ON.** Judged on all 500
LongMemEval_S questions, both arms `--mode investigate --k 6 --max-steps 2`,
differing only in this switch:

| | base | selector | Δ | 95% CI | p |
|---|---|---|---|---|---|
| **overall (n = 500)** | **56.2** | **62.0** | **+5.8** | **[+2.8, +8.8]** | **0.0001** |
| non-abstention (n = 470) | 54.0 | 60.0 | +6.0 | [+3.0, +9.1] | 0.0001 |
| abstention (n = 30) | 90.0 | 93.3 | +3.3 | [+0.0, +10.0] | 0.7185 |

`62.00` is the highest judged LongMemEval_S score this project has recorded.
`ratchet` reads it as **IMPROVED** on two pins at once —
`longmemeval_s.judge_score.n500` 56.60 → 62.00 and
`longmemeval_s.token_f1.n500` 47.06 → 48.89 — with nothing regressed.

**Replicated byte-for-byte.** The selector arm was then re-run end to end
under the new instrumentation (`runs/m32_inv_sel_certified`, 43m13s) and all
500 rows came back **identical in both the composed evidence set and the
reader's raw response** — 500/500 on each — giving the same 62.0, the same
+5.8 [+2.8, +8.8], and the same 282/407 judge verdicts with `0 from cache`.
That is the tightest reproduction this project has on a path with a model call
in it, and it is what lets the published artifact be the self-certifying one:
`select_call_failed_rate = 0.0000` over 500 queries.

### What the selector actually does

It concentrates far harder than "reordering" suggests. Records kept, over the
500 queries of the certified run:

| kept | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| queries | 119 | 194 | 72 | 38 | 19 | 58 |

Median **2**, and 63% of queries keep only one or two records out of a pool of
up to `max_pool = 60`. Nothing is dropped — `select_pool` stable-partitions,
so those records are promoted to the head and `compose`'s `k = 6` window still
emits six — but the mechanism's decision is a sharp one about which *one or
two* memories jointly answer the question, not a gentle re-sort. The 58 at
`kept = 6` include the 15 declines, whose fallback is `0..k` by definition, so
the true "kept six" count is 43.

## 1. The pre-registered condition, and that it was already in the code

`InvestigateConfig::select_sufficient`'s doc comment carried its own release
criterion before this milestone ran:

> **Default off until the pool-level arrangement is measured.** This project
> ships a default on a number, never on a mechanism's plausibility.

That is the whole of M32. The number is above; the switch moved because the
number cleared the bar the code had already set, not because a rule was
written after the data arrived.

## 2. Why this does not contradict M21's `+0.0`

M21 measured **the same switch** inside `investigate` at **exactly +0.0 (95%
CI [−3.8, +3.8], p = 1.0000)** and that null is why the default was off. Two
things separate it from the number above, and both were already on record.

**It measured a different arrangement.** M21 selected *per probe*, inside each
step's `recall` over `step_k = 10` candidates. Since M22 the selection happens
**once over the accumulated pool** (`max_pool = 60`), after the last step and
before `compose` truncates to `k`. The code says so at
`investigate.rs`: *"That arrangement is gone; the selection now happens once,
over the union, where it is the only ranking decision that sees the whole
candidate set."* M21's own diagnosis explains why the per-probe form could not
work — probe results are unioned across steps, so reordering one probe's list
"only changes which items enter a pool that was going to hold them anyway."

**It measured one stratum.** M21's investigate pair is `n = 133`, category 5
only. The +0.0 was never a full-population result. M32's base arm reproduces
M21's temporal cell **exactly** — 36.84 on both — which is the same-code
baseline check M22's lesson demands, and then the selector moves that cell to
42.11.

| | M21 (per-probe, cat 5) | M32 (pool-level, cat 5) |
|---|---|---|
| base | 36.84 | **36.84** |
| selector | 36.84 | **42.11** |
| Δ | +0.0 | **+5.26** |

## 3. Where the gain is, and why that is the right shape

| category | n | base | selector | Δ |
|---|---|---|---|---|
| single-session-user | 70 | 91.43 | 91.43 | **+0.00** |
| single-session-assistant | 56 | 96.43 | 96.43 | **+0.00** |
| preference (`_abs`) | 30 | 26.67 | 33.33 | +6.67 |
| multi-session | 133 | 36.09 | 48.12 | **+12.03** |
| temporal-reasoning | 133 | 36.84 | 42.11 | +5.26 |
| knowledge-update | 78 | 74.36 | 79.49 | +5.13 |

The two single-session strata move by **exactly zero**, on 126 questions. That
is the mechanism's signature rather than a disappointment: a question answered
by one turn has no second hop to select across, so a decision about *which
records jointly answer* has nothing to decide. The gain concentrates on
multi-session (+12.03), the stratum defined by needing evidence from more than
one place. A switch that had moved the single-session strata would have been
measuring something other than joint sufficiency.

## 4. The literature's warning, tested

`docs/research/11-frontier-2026.md` §D.1 records AgentRunbook-R's failure mode:
it *"reduces retrieval+reading errors vs RAG but does NOT improve abstention
(presents evidence that misleads reader into using it instead of rejecting)"*.
A mechanism that surfaces more plausible-looking evidence is exactly the shape
that should trip this.

It did not fire. The 30 `_abs` rows go **90.0 → 93.3**; the CI includes zero,
so the honest reading is "no detectable harm" rather than a gain, which is the
result this switch needed on that stratum. HiGMem's complementary finding —
*"the failure mode is bloated evidence sets: extra superficially similar turns
add little recall but erode precision"* — is the loss this mechanism removes,
and MemPro's base pipeline (`{Retrieve → Integrate → Reflect} until
sufficient`, §3/§A.5) is the SOTA reference for the same idea. MemPro-15 on
Qwen3-30B is the 80.80 in the standing table.

## 5. Cost

| path | base p50 / avg | selector p50 / avg | cost |
|---|---|---|---|
| LongMemEval_S `investigate` | 2.42 / 2.38 s | 4.73 / 4.42 s | **+2.31 / +2.04 s** |

One model call per query over at most `max_pool = 60` records. `PLAN.md` §7.1
pins `recall` at "no LLM in the loop", which is why the +3.8 measured on that
path (§6) cannot ship; `investigate` already spends a model call per step on
the reflect gate, so this is an increment to a path that was never latency-
bound, not a category change.

## 6. The `recall` arm, for the record

The same switch on the forbidden path, same store, same 500 questions:

| | base | selector | Δ | 95% CI | p |
|---|---|---|---|---|---|
| overall (n = 500) | 56.6 | 60.4 | +3.8 | [+1.0, +6.6] | 0.0087 |

This replicates M21's `recall` pair **to the digit** (+3.8, [+1.0, +6.6],
p = 0.0087) on a different binary, and the base arm's token F1 came back at
0.4706 against a pinned floor of 0.47056. It is reported because it is the
cleanest available check that the two milestones measured the same mechanism —
and it stays unshipped, because §7.1 is a latency contract and this costs a
model call per query.

Worth flagging rather than acting on: §7.1's "no LLM in the loop" protects a
**p95 < 100 ms** target, and plain `recall` measures **0.42 s** average today.
The rule may be outliving its reason. That is an architectural decision, not a
measurement one, and this milestone does not make it.

## 7. The defect this found

`bench` is the path that produces every judged number in this repository, and
until M32 it **discarded the retrieval trace entirely** — `.0` on both the
`recall` and `investigate` branches, `RecallTrace` unmentioned anywhere in the
file. Separately, `InvestigateTrace` never carried `select_degraded` at all:
`select_pool` received `Selected { keep, degraded }` and returned
`keep.keep.len()`, dropping the flag.

That is M27's failure class, still live in the highest-stakes path. Its shape:
a degraded selector call falls back to `0..k`, so a **fully degraded selecting
arm emits the unselected arm's evidence set**, scores what the base arm scores,
and the pair reads as a tight, entirely credible null for a mechanism that
never ran. M27 measured the cause — a 100-candidate prompt over real
LongMemEval records is **8,298 tokens** against a reader serving 8,192 per slot
— gated it on the offline instrument, and wrote down that M12, M14 and M20 each
lost a run to the same class. The judged path was never gated.

Fixed rather than noted:

- `InvestigateTrace::select_degraded`, threaded out of `select_pool`.
- `ScoredQuestion::{selected, select_degraded}`, persisted per row, `serde(default)`
  so the 28 historical artifacts `standing` reads still parse.
- `BenchRun::{select_call_failed_rate, select_declined_rate}`, derived from the
  rows rather than plumbed, so a rescored artifact reports them too.
- `DegradationGuard`: a selecting run whose **call-failure** rate is
  incompatible with `ablate::MAX_DEGRADED` at 95% confidence **fails**. Not
  warns.

### The guard's threshold is derived, not chosen

The rule is `wilson_lower(call_failed, seen) > MAX_DEGRADED`, reusing the
Wilson helper the G3 ASR numbers already report with. `MAX_DEGRADED` is 2%
because M27 judged one refused request in five hundred to be noise rather than
a broken run, so the only free parameter is how many queries must be seen
before a failure can abort. That is the smallest `n` at which a *single*
failure is still compatible with 2%:

| | `wilson_lower(1, n)` | |
|---|---|---|
| n = 8 | 0.0224 | would abort a healthy run |
| **n = 9** | **0.0197** | inside the floor |

Nine. A systematically mis-sized server fails *every* call, reaches
`wilson_lower(9, 9) = 0.70`, and aborts **nine queries in** rather than after
the 44 minutes a full investigate arm costs. Both sides are pinned by
`the_minimum_sample_is_the_smallest_that_tolerates_one_failure`, so the
constant cannot be nudged without the arithmetic that justifies it failing
first.

### The guard's first version was itself the bug, and measuring caught it

The first version counted *every* fallback, and on its first live run it
**refused a healthy arm**: 11 of 298 LongMemEval_S `investigate` queries fell
back — 3.7%, Wilson lower 2.1% against the 2% floor — while the reader and the
reranker both answered `/health` = `{"status":"ok"}` for the whole run, before
it and after it.

`Selector::select` had one `degraded: bool` for two unrelated events, which
M27's own comment described without drawing the consequence: *"Anything else is
the model producing something unusable, or the server refusing the request.
Both degrade to rank order, and both are reported as degraded."* They must not
be gated alike:

| cause | what happened | response |
|---|---|---|
| `CallFailed` | server refused, timed out, or answered unparseably | **abort** — a mis-sized server fails *every* call, which is M27's class |
| `ModelDeclined` | the call succeeded, the answer parsed, and it named no usable candidate | **count and report** — "none of these jointly answer it" is a position the selector is allowed to take |
| `None` | a usable selection | — |

M27's 2% floor was calibrated for refusals alone. Applying it to the union of
two unrelated causes gates on the wrong quantity, and the observed decline rate
sits just above it — so the first guard would have made the shipped default
unmeasurable. `Degradation` is now a three-valued enum, the guard gates on
`CallFailed` only, and `model_declines_are_never_fatal_however_many` asserts a
**10%** decline rate still passes, which is the only assertion that would have
caught this.

The same conflation was live in `ablate::width_verdict`, which has gated
`WidthVerdict::Degraded` on the union since M27; it now gates on `CallFailed`
too.

### The published run predates its own instrument

`runs/m32_inv_sel` was produced before these fields existed, so its artifact
cannot self-certify. Two things close that gap:

1. **The effect size is incompatible with a non-running mechanism.** A fully
   failed arm emits the base arm's evidence set and therefore scores the base
   arm's score. +5.8 at p = 0.0001 is not a number it can produce.
2. **The arm was re-run over the same 500 questions under the instrument** —
   `runs/m32_inv_sel_certified` reports `select_call_failed_rate = 0.0000` and
   `select_declined_rate = 0.0300` (15/500), and reproduces the original run's
   evidence and responses on 500/500 rows. The published number is now the
   self-certifying artifact rather than an argument from effect size.

## 8. The second defect: `standing` would have published the wrong arm

`standing`'s `Ours::arm` test is a hand-maintained OR chain over switches that
ship off, and it carried `|| run.select_sufficient` — "a run with this set is
measuring something the defaults do not do." The moment the default flips for
`investigate`, that line inverts both verdicts at once: `runs/m32_inv_sel`, the
shipped configuration, reads as an arm and is excluded from "where we stand",
while `runs/m32_inv_base`, which is *not* the shipped configuration, gets
published as it.

This is `Ours::arm`'s original defect — M21's `runs/m21_full_sel` at 60.40
displacing the shipped 56.60, and `standing` publishing 39.91 for four
milestones — one switch later. `select_sufficient` is the first switch whose
shipped value depends on the mode, so it cannot be tested by value:

```rust
|| run.select_sufficient != shipped_select_sufficient(&run.mode)
```

`shipped_select_sufficient` reads the library defaults rather than hardcoding
the mapping, so flipping either default moves `standing` with it instead of
silently disagreeing. All four corners are pinned by
`select_sufficient_is_an_arm_only_against_its_own_modes_default`.

## 9. Reproduction

```sh
# Serve the reader wide enough for the selector prompt (2 slots x 32768).
ssh big gpu-tenant claim coding
ssh big "MYELIN_READER_SLOTS=2 MYELIN_READER_CTX=32768 bash -s" < ops/big/serve-models.sh
ssh -N -L 5810:127.0.0.1:5810 -L 5813:127.0.0.1:5813 big &

export MYELIN_QDRANT__URL=http://192.168.1.110:6334

# The pair. 27m35s and 44m21s respectively.
myelin-eval bench --corpus longmemeval-s --mode investigate --k 6 --max-steps 2 \
  --out runs/m32_inv_base
myelin-eval bench --corpus longmemeval-s --mode investigate --k 6 --max-steps 2 \
  --select-sufficient --out runs/m32_inv_sel

for d in m32_inv_base m32_inv_sel; do
  myelin-eval judge --run runs/$d
  myelin-eval rescore --run runs/$d --scorer judge --out runs/${d}_judged
done
python3 crates/myelin-eval/adapters/paired_ci.py \
  runs/m32_inv_sel_judged runs/m32_inv_base_judged

# The self-certifying re-run of the selector arm (43m13s).
myelin-eval bench --corpus longmemeval-s --mode investigate --k 6 --max-steps 2 \
  --select-sufficient --out runs/m32_inv_sel_certified
# -> select_call_failed_rate 0.0000, select_declined_rate 0.0300
myelin-eval judge --run runs/m32_inv_sel_certified
myelin-eval rescore --run runs/m32_inv_sel_certified --scorer judge \
  --out runs/m32_inv_sel_certified_judged
```

Judge verdict counts: base 254/377 correct (67.4%), selector 282/407 (69.3%);
the denominators differ because the judge is asked only about rows that
produced an answer.
