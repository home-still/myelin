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
