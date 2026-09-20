# M22 — pool-level selection at G1's operating point

M16 measured that G1's gap is retrieval, not the reader. M21 built the mechanism that fixes it and
shelved it, because the one path it was allowed to default on neutralised it. This milestone put the
selector where the pool actually is, added the gate an undated corpus needs, and re-measured G1 on
the full tier-small pair.

## Verdict

**Both switches ship off. Both are measured nulls at n = 451, and the third arm — the one nobody
planned to run — is the finding: the shipped configuration's own number has been falling since M19,
and 2.4 of the 3.3 points it lost on this corpus come from two date mechanisms annotating records
that carry no dates.**

| arm | flags | combined | vs base | vs nodate | memory latency | S |
|---|---|---|---|---|---|---|
| base | *(shipped default)* | **36.59** | — | — | 18.31 s | 12.1% |
| nodate | `--undated` | **39.02** | +2.4 [−1.1, +6.0] p=0.21 | — | 10.95 s | 8.6% |
| select | `--undated --select` | **37.92** | +1.3 [−2.4, +5.1] p=0.52 | −1.1 [−4.7, +2.4] p=0.58 | 14.45 s | 11.2% |

All intervals from `adapters/paired_ci.py`, paired over the same 451 question ids. Neither switch
clears §1's bar, so by the rule fixed before any arm ran:

- `dated` stays at its default (`true` — no suppression): **measured null**, +2.4 with a CI spanning
  zero. The base configuration is carried forward.
- `select` stays off as an operating point: it does not beat the better of {base, nodate} — against
  `nodate` it is **negative**.
- `InvestigateConfig::select_sufficient` stays **`false`**. The library-default clause required the
  select arm to clear the bar first; it did not, so §6g's LongMemEval_S off-target check was never
  triggered.
- `RetrieveConfig::select_sufficient` stays off regardless, per `PLAN.md` §7.1.
- Winning arm 39.02 against a break-even of **51.0**: G1's break-even is **not** claimed. The gap to
  AgentRunbook-C's 74.90 is **−34.99** and has not moved.

Five things were measured or fixed that are worth more than the two nulls:

1. **The LME-V2 store has no dates in it at all.** 85,589 records, every one stamped inside the
   2.5-hour build window of 2026-09-14, `t_valid` equal to `t_ingested` to the nanosecond, one
   distinct calendar day. `build_lmev2` never sets `t_valid`, because LME-V2 trajectories are agent
   task logs with no timestamps.
2. **M19's two shipped compose defaults have been harming this corpus ever since.** Measured
   directly: the base arm emitted a `[timeline]` block on **240 of 240** web questions and 600
   date-stamped evidence items across the first 40, all anchored to the build date. Suppressing them
   is +2.4 (null) — but it is also **−7.4 s of latency per query**, 18.31 s → 10.95 s, which is not
   noise.
3. **The fresh base is 3.3 points below the number `standing` still publishes.** `runs/m22_base_web`
   is 40.42 where the pre-M19 `runs/myelin_inv2_web_small` was 45.00 (paired −4.6, CI [−9.6, +0.4]),
   and enterprise 32.23 where it was 34.12 (−1.9, CI [−8.1, +4.3]). The published 39.91 was produced
   by code that no longer exists.
4. **M16's central statistic reproduced exactly on new code and a new run.** S = P(sufficient |
   wrong) = **12.1% [8.1, 17.6]** on the base arm against M16's 12.2% [8.1, 17.9]. Retrieval is
   still the binding constraint in all three arms: the reader-fix ceiling is 41.5–42.4, *below* the
   51.0 break-even, and the retrieval-fix ceiling is 76.9–77.8.
5. **Four defects fixed**, three of which would have corrupted this measurement and one of which was
   found by the artifacts it produced (§2, §5).

## 1. The rule, fixed before any arm ran

On the full tier-small pair (web 240 + enterprise 211 = **n = 451**), scored by the vendored
harness's own scorer:

- `dated=false` becomes the LME-V2 default operating point if its combined accuracy beats the
  same-store fresh base with a 95% paired CI excluding zero. If the interval spans zero it is
  reported as a measured null and the base configuration is carried forward — in which case the
  finding that all 85,589 records are undated is still recorded, because it invalidates any future
  temporal claim on this corpus.
- `select=true` becomes the G1 submission's operating point if its combined accuracy beats the
  better of {base, `dated=false`} with a 95% paired CI excluding zero.
- `InvestigateConfig::select_sufficient` flips to `true` as a library default only if that same
  arm clears the bar **and** does not regress LongMemEval_S multi-session or temporal-reasoning
  (the two strata M21 measured) by more than 1.0 point.
- If the winning arm's combined accuracy exceeds **51.0**, G1's break-even is claimed and LAFS is
  recomputed with the new operating point. Otherwise the number and the remaining gap are reported.

`RetrieveConfig::select_sufficient` remains **off** as a `recall` default regardless of outcome:
`PLAN.md` §7.1 pins that path at "no LLM in the loop". It is reachable only as a query-time
operating point.

At n = 451, a paired bootstrap CI on a 0/1 score excludes zero at roughly **±4.5 points**. Every
interval below is inside that, which is the whole verdict.

### Why this was the right target

`docs/measurements/m16-evidence-sufficiency.md` §Verdict: S = P(sufficient | wrong) is 7.4% at the
`recall` point and 12.2% at `investigate`. A perfect reader tops out at 38.8% / 44.6% and cannot
clear 51.0; perfect retrieval reaches 67.6% / 66.0%. §4 below re-measures S on all six new runs and
the conclusion is unchanged — which is the useful part, because it says the mechanism that failed
here failed *for a reason other than the diagnosis being wrong*.

### Why the latency objection did not transfer

M21 shelved the selector over `recall`'s p95 < 100 ms product SLO. G1's operating point is
`mode=investigate, k=25`, and the selector cost **+3.5 s/query** (10.95 s → 14.45 s) against a 40 s
budget where the reference frontier's own best point (AgentRunbook-C) runs at 108.3 s.
`PLAN.md:47-49` states the p95 figure is "a **product SLO**, not a scoring objective, and must never
be traded against accuracy in pursuit of G1." The selector was not rejected on latency. It was
rejected on accuracy.

## 2. Four defects, fixed

1. **The selector was inert on the MCP path.** `MyelinServer::retriever` wired `with_config` and
   `with_reranker` and never `with_llm`, so `RetrieveConfig::select_sufficient` could not fire there
   whatever a caller asked for — the failure class M12, M14 and M20 each lost a run to. Now wired
   unconditionally, like `with_reranker`.
2. **`investigate` never selected over its pool.** It unioned up to `max_pool` records across probes
   and sorted them by `score` — cross-encoder logits produced by *different probe queries*, which
   are not mutually comparable, the same category error `RetrieveConfig::tau_abstain` documents for
   RRF scores — then truncated to `k`. M21 measured per-probe selection underneath that at exactly
   +0.0. `select_pool` replaces it: one question-conditioned judgement over the whole pool, stable
   partition, nothing dropped.
3. **Every LME-V2 record carries the build date as `t_valid`** (§3).
4. **A `--limit` pilot could publish a tier-population number.** `--limit` is not an operating point,
   so a 40-question web pilot has the same fingerprint as the 240-question arm and paired with the
   211-question enterprise run — publishing a "combined" accuracy over 251 questions against a bar
   defined on 451. All three M22 pilots did exactly this and entered the candidate list; the only
   thing between one of them and the published number was its value. `pair_metrics` now requires
   each side to carry its domain's full population.

Two more were found and deliberately **not** acted on:

- **`investigate` does not apply the `is_interval_question` gate that `recall` does.** `recall`
  composes `timeline` only for a question that asks for an elapsed time or an ordering
  (`retrieve.rs`); `investigate` copies the compose config straight through, so the dated index is
  emitted on *every* question in agentic mode — 40/40 in the measured pilot. Changing it would
  invalidate comparability with every `investigate` number measured since M19, so it needs its own
  arm.
- **`investigate`'s pool is structurally too small for `k` to bind.** `step_k` is 10 and `max_steps`
  is 2, so the pool is at most 20 records and `k = 25` never truncates. Selection can only change
  the composed set through the 10,000-token budget; otherwise it reorders. That is a ceiling on the
  mechanism's effect here that the `recall` path does not have, and §7 names it as the next lever.

## 3. The corpus has no dates in it

```
sqlite3 data/lme_v2_small.ledger \
  "select count(*), min(t_valid), max(t_valid) from record;"
85589|2026-09-14T17:20:44.112722000Z|2026-09-14T19:55:54.032512000Z

select count(distinct substr(t_valid,1,10)) from record;   -> 1
select count(*) from record where t_valid <> t_ingested;   -> 7423
```

The 7,423 "differences" are nanoseconds: `2026-09-14T17:20:44.112722000Z` against
`...112695000Z`. Every record carries the moment it was written, spread across the build's own
2.5-hour window. `build_lmev2` never sets `t_valid` and never increments `sessions_without_date`,
because LME-V2 trajectories are agent task logs with no timestamps at all.

`ComposeConfig::stamp_valid_time`, `::resolve_relative` and `::timeline` have shipped **on** since
M19, measured on LoCoMo and LongMemEval_S, which do carry dates. On this corpus they annotate
against a meaningless anchor. Proven live against the served store:

```
recall "what did the user configure last tuesday" tenant=lme_v2_small/web k=3
  (default)    '[2026-09-14] step: [0] (initial state)\nTo find the price configuration…'
  dated=false  'step: [0] (initial state)\nTo find the price configuration…'
```

and over the 40-question pilot pair, `memory_context` differs on **40 of 40** rows: 600 date-stamped
items and 40 `[timeline]` blocks in the base arm, zero of each with `--undated`.

## 4. S, per arm — retrieval is still the binding constraint

`myelin-eval evidence-audit` over all six runs: for every answerable question the harness scored
wrong, a reader-only judge decides whether the evidence the reader was actually shown contained the
answer. 537 judgements.

| arm | acc | wrong & answerable | sufficient | S | 95% Wilson | reader-fix ceiling | retrieval-fix ceiling |
|---|---|---|---|---|---|---|---|
| base | 36.59 | 182 | 22 | **12.1%** | [8.1, 17.6] | 41.46 | 76.94 |
| nodate | 39.02 | 175 | 15 | **8.6%** | [5.3, 13.7] | 42.35 | 77.83 |
| select | 37.92 | 178 | 20 | **11.2%** | [7.4, 16.7] | 42.35 | 77.38 |

Per domain: base 19.3% / 5.3%, nodate 9.6% / 7.6%, select 14.8% / 7.8% (web / enterprise).

Two readings:

- **M16 reproduced.** Its `investigate` figure was 12.2% [8.1, 17.9]; the base arm here is 12.1%
  [8.1, 17.6], on a different code version and a fresh run. The diagnosis is not an artefact of the
  run it was measured on.
- **Selection did not move it.** The plan's expected direction was that the *insufficient + wrong*
  cell falls and S rises. Web went the other way (19.3% → 14.8%), enterprise moved +2.5, and every
  interval overlaps every other. The reader-fix ceiling stays **below** the 51.0 break-even in all
  three arms: a perfect reader over today's evidence still fails G1.

## 5. What `standing` says now

```
lme_v2_small.overall_full_set.combined   ours 39.91  bar 74.90  gap −34.99   [GATE]
lme_v2_small.lafs_gain.small             ours  0.00  bar  0.00  tie         [GATE]
```

Unmoved, and the 39.91 is `runs/myelin_inv2_web_small + _enterprise_` — pre-M19 code. The candidate
list after the population guard is five legitimate n=451 pairs and no pilots:

```
myelin_inv2_web_small  39.91      m22_sel_web    37.92      myelin_k25_web_small  35.70
m22_nodate_web         39.02      m22_base_web   36.59
```

The honest statement is therefore two numbers, not one: **the best LME-V2 combined accuracy this
project has ever recorded is 39.91, and the best one it can reproduce today is 39.02.** LAFS gain is
0.00 at every point — the frontier dominates all of them, so no operating point earns a gain and
recomputing it with a slower arm cannot help.

## 6. Reproduction

```bash
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
ssh big gpu-tenant claim coding
ssh big "MYELIN_READER_SLOTS=2 MYELIN_READER_CTX=65536 bash -s" < ops/big/serve-models.sh
ssh -N -L 5810:127.0.0.1:5810 -L 5813:127.0.0.1:5813 big &

# dataset (7.6 GB; trajectory_screenshots are NOT needed — myelin is text-only
# and the prior G1 run produced 0 non-text memory items — so validate with
# --no-check-screenshots and delete the 6.5 GB of tarballs)
cd crates/myelin-eval/vendor/longmemeval-v2
python data/download_data.py --data-root ../../../../data/lmev2
python data/validate_data.py --data-root ../../../../data/lmev2 --tier small --no-check-screenshots

cargo run --release -p myelin-mcp -- --serve 127.0.0.1:7447 \
  --collection myelin_lme_v2_small --ledger data/lme_v2_small.ledger &

bash ops/m22_full_arms.sh          # six arms, ~6.5 h, resumable

python3 crates/myelin-eval/adapters/paired_ci.py \
  runs/m22_nodate_web+runs/m22_nodate_ent runs/m22_base_web+runs/m22_base_ent
python3 crates/myelin-eval/adapters/paired_ci.py \
  runs/m22_sel_web+runs/m22_sel_ent runs/m22_nodate_web+runs/m22_nodate_ent
for d in m22_base_web m22_base_ent m22_nodate_web m22_nodate_ent m22_sel_web m22_sel_ent; do
  cargo run --release -p myelin-eval -- evidence-audit --run "runs/$d"
done
cargo run --release -p myelin-eval -- standing
```

Three arms were lost to infrastructure before any of this produced a number, and the fixes are part
of the milestone: the MCP session is now re-established when the server drops it (`rmcp` evicts an
idle session, and the harness idles >5 min loading its 1.1 GB `trajectories.jsonl`), the call retry
budget went from 6 s to ~3 min, `--openai-max-retries` raises the reader/judge budget past the
vendored 55 s, and `ops/m22_full_arms.sh` blocks until every dependency answers before starting an
arm. `big` is shared: another tenant's vLLM start drove it to load 151, sshd stopped answering, and
the host later rebooted mid-sweep.

## 7. What this says about the next lever

M21's conclusion was "selection works and there is nowhere to put it". M22's is narrower and less
comfortable: **selection's win is corpus-shaped, and LME-V2 is not the shape.** It was +3.8 over all
500 LongMemEval_S questions and a null on LoCoMo multi-hop; here it is −1.1 against the better
baseline. The common factor is that LongMemEval_S rewards picking *which* of many similar sessions
answers the question, while LME-V2's evidence is a page dump whose relevant span the reader must
still find — reordering 15 page dumps does not change what the reader has to read.

Two candidates, in order:

1. **Pool width.** `step_k = 10`, `max_steps = 2`, so `investigate` composes from at most 20 records
   and `k = 25` never binds. Every selection mechanism measured here was choosing the order of a set
   it was going to emit anyway. `prefetch_limit` / `rerank_depth` / `step_k` were deliberately out of
   scope for M22 and M21's argument against breadth was measured on LongMemEval_S, not here.
2. **Sub-record granularity.** The retrieval-fix ceiling is 77 and the reader-fix ceiling is 42, so
   the answer is in the store and not in the evidence. On a corpus whose records are whole rendered
   pages, the unit of retrieval may simply be too large for `k` items to carry the answer.
