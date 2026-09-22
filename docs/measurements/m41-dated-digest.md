# M41 — dating the digest's lines

> **Status: mechanism built, arm not yet run.** Two attempts died mid-flight —
> one to a dropped SSH tunnel, one to the `myelin_*` collections being deleted
> through the Qdrant dashboard while the run was live. Both causes are fixed in
> `m41-durability.md`. The pre-registration below is unchanged and was written
> before any arm; the arm is M42's first job.

## What M40 left

`item_digest` fixes the digest's entry count at the number of memories, which
took the count away from a model that had been under-producing. It worked as
designed — rows emitting two or more contributions went **20.4% → 86.4%** —
and the pre-registered prediction held for the first time since M32:

| stratum | n | base | digest | delta | 95% CI |
| --- | --- | --- | --- | --- | --- |
| gold = 1 | 169 | 79.9 | 78.7 | −1.2 | [−5.3, +3.0] |
| **gold = 2** | 217 | 56.7 | **62.7** | **+6.0** | **[+0.9, +11.1]** |
| gold ≥ 3 | 31 | 35.5 | 45.2 | +9.7 | [−3.2, +22.6] |
| `multi-session` | 121 | 44.6 | 53.7 | **+9.1** | [+0.0, +18.2] |
| `knowledge-update` | 72 | 77.8 | **73.6** | **−4.2** | [−11.1, +2.8] |

Headline **+2.40 (95% CI [−0.60, +5.60])**, short of the +3.0 bar, so the
switch stayed off.

One category paid, and the cause was visible in the artifact. The digest
flattens a dated evidence set into an undated fact list:

```
[notes] User mentions recently getting a new 50mm prime lens.;
        User mentions using a new Canon EF lens for portrait work.;
        User mentions getting great shots with a new 70-200mm zoom lens.
```

The question was which lens was bought **most recently**. Base answered
"70-200mm zoom lens" (correct); with the digest the reader answered "50mm
prime lens" — the first line. Same shape on the coffee-ratio question,
answered backwards.

`knowledge-update` is defined by recency. `ComposeConfig::stamp_valid_time`
ships on, so every composed item already carries `[YYYY-MM-DD]`, and M19
measured resolving dates *for* the reader at **+37.6** on LoCoMo category 2
against **+14.3** for telling it to resolve them itself. The digest was
discarding the larger half of that.

## The mechanism

`InvestigateConfig::digest_dates`. Each digest line is prefixed with the
`(YYYY-MM-DD)` of the memory it came from, read off the composed text with
`stamped_date` so the digest can only ever date a line with what the reader is
actually shown. A wrong-shaped prefix yields no date, never a wrong one.

Costs **no extra model call** — the stamp is already there.

A separate switch rather than a change to `item_digest`, for the reason M20
kept its profile block and its reader clause apart: one combined flag cannot
produce the marginals, and the undated arm is already measured. With
`digest_dates` off the digest is byte-identical to M40's, which is pinned by
`the_dating_switch_off_reproduces_the_undated_digest`.

### The subtle part: dedup and dating meet

M40 de-duplicates contributions on exact string match. Dating changes what
"exact" means, and getting it wrong would delete the very thing this fixes:
on a `knowledge-update` question the repetition of a statement **across
dates** is the signal. So the dedup key is the **dated** line — the same
sentence on two different days survives twice, while the same sentence twice
on one day collapses to one. Pinned by
`the_same_fact_on_two_days_is_not_deduplicated`.

## Pre-registration

Written before the arm ran.

**Arms.** `myelin-eval bench --corpus longmemeval-s --mode investigate --k 6
--budget-tokens 4096 --max-steps 2 --select-sufficient --item-digest`, with
and without `--digest-dates`. Base is `runs/m32_inv_sel_certified_judged`
(62.00); the undated digest arm is `runs/m40_digest_judged` (64.40), same
store, same operating point, so both the absolute effect and the marginal over
M40 are computable.

**Population.** All 500 LongMemEval_S rows.

**Primary metric.** `longmemeval_s.judge_score.n500`, judged.

**Decision rule.**

- Ship `item_digest` + `digest_dates` on if the judged score improves by
  **≥ +3.0** over base with a paired 95% CI excluding zero.
- Report per stratum and per contribution count regardless.

**Predicted, specifically.** `knowledge-update` recovers most of its −4.2;
the multi-fact strata are roughly unchanged, since those questions do not turn
on recency; `temporal-reasoning` (+3.9, CI [−2.4, +10.2] in M40) tightens.

**Falsifier.** If dating does not move `knowledge-update`, the M40 regression
was the digest's confident phrasing rather than its missing dates, and the
next attempt is about hedging, not timestamps. That is a real possibility:
one of the five regressed rows had the digest assert "implying no prior gadget
in this context", which no timestamp fixes.

## Results

Filled in after the run.
