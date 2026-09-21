# M36 — no recorded signal discriminates abstention, so build one that decides

**Verdict: the abstention trigger this project has been using is worthless,
and that is why M35's premise arm lost 8.75 points.** The mechanism that
replaces it is built, tested and wired; its arm is **not run**, and the two
pilot measurements that shaped it are the other results here.

## 1. The trigger, measured

`abstain_on_insufficient` fires when the investigate loop stops for any reason
other than `"sufficient"`. As a "this question should be declined" classifier,
over all 451 LME-V2 tier-small questions at the shipped operating point:

| signal | recall | precision | lift over 28.4% base |
|---|---|---|---|
| **`stopped_because != "sufficient"`** | 86.7% | **32.8%** | **1.16×** |
| `steps == 2` (budget exhausted) | 86.7% | 32.8% | 1.16× |
| selector declined | 5.5% | 24.1% | **0.85×** |
| `pool < median` | 32.8% | 23.9% | 0.84× |

It fires on **70% of questions that have an answer**. `premise_analysis` only
runs *after* that gate has fired, so M35's −8.75 was never about the premise
prose — it was the trigger, making a bad decline persuasive on two-thirds of
the answerable set.

The selector's own decline scores **below base rate**. Every numeric signal in
the trace is flat between strata:

| | abstention | answerable | diff |
|---|---|---|---|
| selected | 7.258 | 6.746 | +0.512 |
| pool | 15.203 | 14.687 | +0.516 |
| steps | 1.867 | 1.703 | +0.164 |
| evidence items | 14.898 | 14.505 | +0.394 |
| context tokens | 7150.9 | 6969.2 | +181.7 |
| **gate abstained** | **0.000** | **0.000** | — |

`abstained` is zero on both strata: in the shipped configuration the
insufficiency gate never fires at all, because `abstain_on_insufficient` is
off. The only signal with any separation is the loop's own verdict —
`sufficient` is reached on 30% of answerable questions and 13% of abstention
ones — and 1.16× lift is not a gate.

**The decision has to be made, not recovered.**

## 2. What the literature says the mechanism must look like

Corrective RAG (2401.15884 §4.3) ablated exactly M35's design and reports the
result:

> Preliminary experiments of employing only the **Correct and Incorrect**
> actions show that the efficacy of CRAG was easily affected by the accuracy
> of the retrieval evaluator. The reason might be the distinct knowledge
> switch for all input cases, regardless of the level of confidence in their
> judgment. **The design of the Ambiguous action significantly helps to
> mitigate the dependence on the accuracy of the retrieval evaluator.**

`premise_analysis` is a binary gate. A binary gate's damage scales with its
evaluator's error rate, which is precisely what −8.75 with a 3.2× false-fire
rate looks like. CRAG's fix is a third, soft action.

A second warning from the same paper, which applies harder here: their
evaluator is a **fine-tuned T5-large (0.77B)**, and they note that prompting
a general model for the same judgement "underperforms". We have no fine-tuned
evaluator, so the soft branch matters more for us, not less.

## 3. The mechanism

`InvestigateConfig::answerability_gate` judges the **composed** evidence —
what the reader will actually see — and acts on a graded verdict:

| verdict | action |
|---|---|
| `Supported` | evidence emitted **untouched** |
| `Ambiguous` | evidence emitted, plus one line permitting a decline |
| `Unsupported` | evidence replaced by the insufficiency statement |

### The property that bounds the damage

Under `Supported` the evidence set is **byte-identical** to the gate-off arm.
A question the evaluator classifies correctly therefore cannot be harmed at
all, and an arm measures the evaluator's error rate rather than a
prompt-contamination effect. `premise_analysis` had no such property: it
rewrote the evidence channel on every question it fired on.

### Calibration the schema cannot enforce

`missing` is a required field, and an `Unsupported` verdict that cannot name
the absent fact is demoted to `Ambiguous`. A model asked for a boolean says
"no" on thin-looking evidence; a model that must also say *what* is missing
has to look for it first. This is the only calibration available without
CRAG's fine-tuned evaluator, and it is enforced in code because a model can
always emit an empty string.

### Fail-open

A refused, timed-out or unparseable call returns `Supported` and changes
nothing. The alternative is a server hiccup silently declining on answerable
questions — M32's silent-degradation class, except corrupting answers rather
than a measurement.

Six tests pin all of it, including the byte-identity property and the
demotion rule.

## 4. Two things the pilot measured, and one it refuted

### 4.1 A truncated evidence head manufactures false refusals

The first implementation gave the evaluator a 600-character head of each item,
copying the selector's approach. On a 12-question pilot:

| stratum | verdicts |
|---|---|
| abstention (3) | 3 × `unsupported` |
| **answerable (9)** | 3 × `ambiguous`, **6 × `unsupported`** |

A **67% false-refusal rate** — M35's failure reproduced in two minutes.

The cause is not the prompt. LME-V2 records are page dumps: median **1,642
characters**, p90 1,916, so a 600-char head keeps **39.4%** of the content.
The evaluator was asked whether a fact was present while being shown a
minority of each memory.

The general rule this establishes: **selection may truncate, judgement may
not.** Selection is a *relative* decision — which of these ranks highest — and
a head suffices. Answerability is *absolute*, and a head is how you
manufacture a "no".

### 4.2 Untruncated is correct and unaffordable

Removing the truncation makes each gate call ~10k tokens at `k = 25`. A
20-question pilot **exceeded 1,000 seconds** against a reader serving one slot
— and produced nothing, because `harness.py` holds generations in memory until
the scoring stage, so the timeout discarded them (the M17 trap, again).

At >50 s/query a 240-question domain is over three hours, which is not a
mechanism that can be measured, let alone shipped.

`EVIDENCE_CHARS = 2000` is the compromise: above p90, so essentially all of
90% of records survive, with the worst case bounded. **Its effect on verdict
quality is unmeasured.** The 600-character figure is measured; 2,000 is chosen
from the length distribution.

## 5. Status and the pre-registered rule

`answerability_gate` ships **off**. Wired through MCP → adapter → runner
(`--answerability-gate`), and the graded verdict is recorded per question in
`memory_post_query_metadata.support`, so an arm can report the verdict
distribution by stratum rather than inferring it.

**Pre-registered, unchanged from `PLAN.md` §15:** ships on for `investigate` at
**≥ +3.0 combined over the 451** with a paired CI excluding zero, **and
reported per stratum** — a gain on abstention bought with a loss on answerable
is M35 repeating and must be visible.

Two prerequisites the pilots established for whoever runs it:

1. Serve the reader so both the gate (~10k tokens) and the harness's scoring
   stage (~2× `--max-completion-tokens`) fit *per slot*. At
   `--max-completion-tokens 20000` that is 40k, so `MYELIN_READER_SLOTS=1
   MYELIN_READER_CTX=65536` is required and the run is serialised.
2. Budget for it. The gate adds one ~10k-token call per query on a path that
   already spends a selector call and a reflect call per step.

## 6. What this milestone does not claim

No accuracy number moved. `kind_quota` (M35) and `answerability_gate` (M36)
are both built, tested and unmeasured, and the standing table is unchanged:
−20.02 to AgentRunbook-R on the only apples-to-apples row.

What did move is the account of *why* the last two arms failed. M35 said the
premise mechanism was anti-selective; M36 shows the anti-selectivity was in
the trigger it inherited, measures that trigger at 1.16× lift, and shows that
nothing else currently recorded does better.
