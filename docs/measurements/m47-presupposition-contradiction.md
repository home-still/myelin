# M47 — presupposition verification, contradiction only

## What M35 measured, and why it was the wrong verb

`premise_analysis` (M35) replaced the insufficiency statement with a
premise analysis whenever the investigate loop stopped unsatisfied. It
tripled declines on answerable rows (8.3% → 26.8%) to move abstention by
1.3× and cost **−8.75 judged (95% CI [−15.00, −2.92])** on LME-V2 web. The
mechanism fired on *unsupported*, and a 9B says "unsupported" whenever the
store is merely silent.

LME-V2's abstention rows are **wrong-premise** questions: the question
assumes something the trajectory contradicts. Silence is not a false
premise. Contradiction is.

## The literature

Kim et al., *Which Linguist Invented the Lightbulb?* (ACL 2021,
`10.18653/v1/2021.acl-long.304`): ~21% of Natural Questions' unanswerable
items are explained by unverifiable presuppositions; the proposed pipeline
is presupposition generation → verification → explanation, and "the biggest
bottleneck is the verification component … even transfer from the best
entailment models currently falls short." (QA)² (`2212.10003`) and FalseQA
(`2307.02394`) show models *hold* the knowledge to rebut a false premise but
need the rebuttal step activated. We cannot fine-tune, so the activation is
structural.

## The mechanism

`InvestigateConfig::premise_check` (default **off**), one model call per
query after the digest, strict schema:

```json
{ "presuppositions": [ { "claim": "…", "evidence_index": 1, "status": "contradicted" } ] }
```

- `minItems: 1`, `maxItems: 4` — M40's forcing; a question always assumes
  something and a model allowed to return nothing returns nothing.
- Field order is the mechanism, for the fourth time (M42, M43, M44 R1):
  `claim`, then `evidence_index` — the memory it was checked against, or −1
  — then `status`. Asked for the verdict first, a model rules on a memory
  it has not located; asked for the memory first, an `absent` has to follow
  a −1 and a `contradicted` has to follow a real index the caller verifies.
- **Only `contradicted` with a checkable index emits anything**: one
  `[premise]` item at the tail — *the question assumes "X", but memory [i]
  says otherwise: …* — carrying the weakest trust among the memories it
  cites. `supported` and `absent` emit **nothing**, so on those rows the
  evidence is byte-identical to the switch being off. That one rule is the
  whole difference from M35, and it makes M35's damage unreachable by
  construction.
- Fail-open: refused, unparseable or empty → nothing appended. Views
  (`[timeline]`, `[notes]`) are neither checked nor citable.

`InvestigateTrace::premise_contradictions` counts what fired, so an inert
arm is tellable from a null. Exposed as `premise_check` on the MCP
`investigate` tool, `--premise-check` on `bench` and on
`adapters/run_myelin.py` (written unconditionally to `memory_params`).

## Pre-registration

Written before any arm ran.

**Population.** LongMemEval-V2 tier-small, web 240 + enterprise 211 = 451,
the harness's own split; abstention stratum 128, answerable 323.

**Base.** No LME-V2 run at the shipped configuration exists (M43 flipped the
digest on; every LME-V2 artifact on disk ran without it). The base is
therefore a fresh pair at today's defaults, `runs/m47_base_{web,ent}`, run
first, and it becomes the quotable LME-V2 row whatever the arm does.

**Arm.** The same pair with `--premise-check` → `runs/m47_check_{web,ent}`.

**Primary metric.** `lme_v2_small.overall_full_set.combined`, paired
bootstrap over `web+enterprise` (`adapters/paired_ci.py`). Reported with
the two strata.

**Decision rule.** Ship on at **≥ +3.0** combined with a paired 95% CI
excluding zero, **and** answerable accuracy falling by no more than 1.0.
The second clause is this milestone's veto: the mechanism is *for* the
abstention rows, and buying them with answerable rows is M35 again.

**Predicted, specifically.**

1. Abstention accuracy moves **+10 or better** from the shipped 17.97%.
2. Answerable accuracy moves −1.0 or better.
3. The item fires (`premise_contradictions > 0`) on a minority of rows,
   concentrated on the abstention stratum; rows where it does not fire
   move **exactly +0.0**.

   *Amended 2026-09-23, before the arm ran:* "exactly" cannot hold on this
   benchmark. The harness samples its reader (temperature 0.6, top-p 0.95,
   top-k 20, no seed), so no LME-V2 row is byte-identical between two runs
   whatever the memory does. The testable form is: rows where the item does
   not fire move **+0.0 within their paired CI**, and the composed evidence
   on those rows is identical between base and arm (the memory side is
   greedy and deterministic; only the harness reader samples).
4. Nothing on LongMemEval_S or LoCoMo is measured here; those corpora have
   too few wrong-premise rows to carry a CI (30 and 446 adversarial-by-
   silence rows respectively, which is the *other* abstention shape).

**Schedule amended 2026-09-23 06:35, before this arm ran.** The user asked
for every queued arm to run now rather than in series. `big`'s reader was
re-served with 8 slots on one unified 128k KV pool (`MYELIN_READER_KV_UNIFIED=1`)
plus the projector, and M47, M51, M52 and the M50 extraction share it. The
cost is reproducibility, not validity: llama.cpp batches concurrent slots,
so the same greedy request can decode differently when its batch-mate
changes (M48's note). No decision rule here rests on byte-identity. Concretely for M47: `m47_base_web` ran alone on 1 slot × 32k before the
change; `m47_base_ent` and both arm domains run co-scheduled. Prediction 3's
composed-evidence identity is therefore *reported* as a diagnostic, not
predicted; the decision rule (paired bootstrap, answerable veto) is
unchanged.

**Falsifier.** If true premises get marked `contradicted` often enough to
cost answerable rows, verification is the bottleneck exactly as Kim et al.
found, and the next step is an NLI model doing the verification instead of
the 9B — not another prompt.

**Cost.** Two LME-V2 pairs (base and arm), each ~1–2 h on the reader at
k = 25 / 10,000 tokens, plus one model call per query in the arm.

## Results

*(pending — queued behind M44 R1 and M46 on the reader)*
