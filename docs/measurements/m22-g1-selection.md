# M22 — pool-level selection at G1's operating point

M16 measured that G1's gap is retrieval, not the reader. M21 built the mechanism that fixes it and
shelved it, because the one path it was allowed to default on neutralised it. This milestone puts
the selector where the pool actually is, adds the gate an undated corpus needs, and re-measures G1.

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

At n = 451, a paired bootstrap CI on a 0/1 score excludes zero at roughly **±4.5 points**.

### Why this is the right target

`docs/measurements/m16-evidence-sufficiency.md` §Verdict: S = P(sufficient | wrong) is 7.4% at the
`recall` point and 12.2% at `investigate`. A perfect reader therefore tops out at **38.8% / 44.6%**
and cannot clear the 51.0 break-even; perfect retrieval reaches **67.6% / 66.0%**. The headroom is
all on the retrieval side, and M21 measured the mechanism that exploits it at +3.8 judged points
over all 500 LongMemEval_S questions (CI [+1.0, +6.6], p = 0.0087).

### Why the latency objection does not transfer

M21 shelved the selector over `recall`'s p95 < 100 ms product SLO. G1's measured operating point is
`mode=investigate, k=25` at p50 11.01 s (web) / 12.40 s (enterprise) against a 40 s budget, and
`compute_lafs.py:36-41` integrates from T_min = 1 s. The reference frontier's own best point
(AgentRunbook-C) runs at 108.3 s. `PLAN.md:47-49` states the p95 figure is "a **product SLO**, not a
scoring objective, and must never be traded against accuracy in pursuit of G1."

## 2. Three defects fixed before measuring

1. **The selector was inert on the MCP path.** `myelin-mcp/src/server.rs::retriever()` wired
   `with_config` and `with_reranker` and never `with_llm`, so `RetrieveConfig::select_sufficient`
   could not fire there whatever a caller asked for. This is the failure class M12, M14 and M20 each
   lost a run to. Fixed unconditionally, like `with_reranker`.
2. **`investigate` never selected over its pool.** It unioned up to `max_pool = 60` records across
   probes and sorted them by `score` — cross-encoder logits produced by *different probe queries*,
   which are not mutually comparable — then truncated to `k`. M21 measured per-probe selection
   underneath that at exactly +0.0 for +1.87 s/query. Pool-level selection had never been tried.
3. **Every LME-V2 record carries the build date as `t_valid`.** `select count(*), min(t_valid),
   max(t_valid) from record` on `data/lme_v2_small.ledger` returns **85,589 rows, all
   `2026-09-14T17:20:44Z`**: `build_lmev2` never sets `t_valid` because LME-V2 trajectories are agent
   task logs with no timestamps. `ComposeConfig::resolve_relative` and `::timeline` have shipped ON
   since M19 and were annotating the reader's context against a meaningless anchor on this corpus.

<!-- §3 onward is written after the arms run. -->
