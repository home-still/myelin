# M79 — the typed premise pass: answer, then say how the question fits *(user decision 2026-09-28; pre-registered before any row)*

## Why

The reader's own declines are the largest loss on LongMemEval_S. 44 of M57's
107 official losses are "I don't know" on answerable questions, and 20 of
those hold every gold turn. M57's premise clause makes the reader decline
first and correct after. Classifying M57's 78 declines by what the correction
says (regular expressions over the text; `m79` section of the round-5 plan):

| decline says | answerable | abstention |
|---|---|---|
| nothing after "I don't know." | 18 | 8 |
| a contradiction ("guitar, not violin") | 6 | 9 |
| an **unstated detail** ("7 shirts … don't specify the trip was 5 days") | **13** | **7** |
| something else | 12 | 5 |

The 13 answerable "unstated detail" declines are the target. The 7
abstention rows phrased the same way are the risk. One is "only state you
planted 5 tomato plants", for a question about tomatoes *and chili*. That is
LongMemEval's trap: the question adds a thing the memories never mention.

## Mechanism

`commit-arm --typed` (M79; `bench::commit_typed`). It is a greedy second
call, one per declining row, over any run's declines. Like M71b it is a
post-pass, so it can also stack on top of the grounded pass.

- **Schema:** `{answer, mismatch}`, answer first. That is M42's order and
  also the wire order (`defect-2026-09-28-schema-field-order.md`). `mismatch`
  is one of `none`, `detail unstated`, `contradicted` or `never mentioned`,
  and each is defined in the instruction appended to the base's own system
  prompt.
- **The decision is structure.** Code commits the answer only if it is
  non-empty, is not itself a decline, and `mismatch` is `none` or
  `detail unstated`. On `contradicted` or `never mentioned` the decline
  stands. So does every failure: a model error, or content that does not
  parse.
- **Grounds:**
  - Wagner (2026, arXiv 2607.08456): answer confidence "tracks whether an
    answer is right but is nearly blind to whether the question is
    answerable", so the two are asked for as separate axes.
  - CREPE (Yu et al. 2023, `10.18653/v1/2023.acl-long.583`) on false
    presuppositions, and (QA)² (Kim et al. 2023,
    `10.18653/v1/2023.acl-long.472`) on questionable or unverifiable
    assumptions. Both separate the kinds of assumption a question can fail
    on.
  - Sufficient Context (Joren et al., arXiv 2411.06037): smaller models
    "abstain often, even with sufficient context".

## Measurement

**Gate arm:** the typed pass over each base replicate's declines:
- `m57_bonsai_premise_s1` → `m79_base_s1`;
- `r5_base_s2` → `m79_base_s2`;
- `r5_base_s3` → `m79_base_s3`.

These are the round-5 base replicates (`r5-bundle-seeds.md`). The pass is
greedy, so every row that does not decline is untouched by construction.
Paired against the three base replicates, seed-averaged.

**Gate** (the user's stratum rule, plus the veto in its strict form):
1. On M57's 49 answerable declines (`m71b_answerable_declines`), the strict
   paired difference has a 95% CI excluding zero, *and*
2. the official difference on the same rows is positive, *and*
3. **the abstention veto:** the seed-mean abstention score under both
   readings is not below the base's. A single abstention row talked out of
   its decline fails M79, as it failed M42.

**Use:** if the gate passes, the typed pass stacks after the grounded pass on
the round-5 bundle's replicates. Those are `r5_bundle_s*_grounded`, or
`r5b_bundle_s*_grounded` if M78b replaced M77c. The bundle's bar is then read
on those, as `r5-bundle-seeds.md` pre-registered.

**Cost:** about 80 greedy calls per replicate, around 10 minutes on big, plus
judging. It runs after the round-5 and M78b units, from the same queue.

## Predictions

- Commits on 12–20 of about 78 declines per replicate. M42 committed 34%
  and M71b 9%; the enum should land between them.
- Answerable declines: 5–9 fixed per replicate, with strict and official
  about +10 to +18 on the 49.
- Overall: strict +1.0 to +1.8.
- Abstention: 0 flips expected. **This is the riskiest prediction in the
  doc.** The 7 "unstated detail" abstention declines are where it breaks.

## Falsifiers

- **Any abstention flip:** the enum does not separate "a detail left
  unstated" from "a thing never mentioned" on LongMemEval's traps. Report
  the flipped rows' `mismatch` value.
- **Commits right under 50% of the time:** the enum commits, but on the
  wrong rows. Report accuracy by `mismatch` value.
- **Commits under 5 per replicate:** the reader types almost everything as
  `contradicted` or `never mentioned`, so the typed pass is as inert as
  M71b's grounding.

## Amendment before any row *(2026-09-28, round-5 catalog)*

Wagner (2026, arXiv 2607.08456) reports that instructing a model to check
premises "backfires, because it then disputes sound and false premises alike
(57% false challenges)". M79's `mismatch` field is that instruction. The gate
is unchanged, and the readout will also report Wagner's two error rates
separately:
- **false challenges:** of M57's 49 answerable declines, the share typed
  `contradicted` or `never mentioned`. Each is a recovery missed, never a
  new loss.
- **false fits:** of the abstention declines, the share typed `none` or
  `detail unstated` with a non-empty answer. Each breaks a correct
  abstention, and it is what the veto guards.
- The full `mismatch` distribution, split into answerable and abstention.

If the label is noisy, the catalog's next step is to state the premise
finding upstream, in code, and hand it to the reader. That is
LongMemEval-V2's AgentRunbook-C pattern, which "improves abstention" where
handing over raw evidence does not.

## Result — the target moves, and the veto fires on exactly the predicted rows *(measured 2026-09-29, 00:27)*

The typed pass ran over each base replicate's declines (`m79_base_s{1,2,3}`)
and was paired against the base replicates.

| stratum | strict Δ [95% CI] | W/L | official Δ [95% CI] | W/L |
|---|---|---|---|---|
| M57's 49 answerable declines | **+19.7 [+9.5, +31.3]** | 10/0 | **+10.9 [+0.7, +21.1]** | 7/1 |
| non-abstention (470) | +2.1 [+0.9, +3.5] | 11/0 | +1.2 [+0.1, +2.3] | 8/1 |
| **abstention (30)** | **−20.0 [−36.7, −6.7]** | 0/6 | **−20.0** | 0/6 |
| overall | +0.8 [−0.7, +2.4] | 11/6 | −0.1 | 8/7 |

**Verdict:** criteria 1 and 2 pass, and **criterion 3, the abstention veto,
fails.** Six abstention rows are answered in every replicate. M79 does not
ship and does not stack.

**Wagner's two error rates (the pre-data amendment):**
- **False challenges:** of the answerable declines, 28/49, 29/51 and 30/52
  are typed `contradicted` or `never mentioned`. That is **~57%, Wagner's
  own figure** ("57% false challenges"), reproduced on a different model and
  benchmark.
- **False fits:** 6 per replicate. They are the same six abstention
  declines, typed `detail unstated` every time. This is the risk the
  pre-registration named: LongMemEval's traps add a thing the memories never
  mention, and the reader calls it a detail.

**The label distribution is stable across seeds** (answerable: none 5,
detail unstated 16–17, never mentioned 24–25, contradicted 4–5;
abstention: never mentioned 20, detail unstated 6, contradicted 3). The
typed label is a consistent judgement, not noise, and on these six rows it
is consistently wrong.

**The stack on the round-5 bundle** (`r5_bundle_s*_typed`) shows what the
veto is for:
- strict **+3.2 [+1.1, +5.5]**, which clears the +3.0 bar;
- abstention **23.3/30**, and official **80.40**, *below* 80.80.

The abstention losses cost the official grader more than the recovered
answers gain.

**Next (plan step 3, M80):** the premise finding moves to code and is handed
to the reader. That is LongMemEval-V2's AgentRunbook-C pattern. The six
traps all name a thing (chili, a 30-gallon tank, a university…) that no
memory contains, and a code check can see that where the reader's label
cannot. M80 is a reader-input change, so it goes to the user first.
