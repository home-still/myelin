# M80 — the premise finding, computed in code and shown to the typed pass *(user decision 2026-09-29; pre-registered before any row)*

## Why

M79's typed pass asked the reader to judge each declined question's premise
itself (`m79-typed-premise.md`):
- It recovered answerable declines: strict +19.7 [+9.5, +31.3], 10 wins and
  0 losses.
- It also answered **six unanswerable traps** in every replicate, typing
  each as a "detail unstated". Abstention fell −20.0, and the veto fired.
- Its false-challenge rate was ~57%, Wagner's (2026, arXiv 2607.08456) own
  figure for a model asked to check premises itself. The label is
  consistent across seeds, and consistently wrong on those six.

The six traps all ask about something no memory contains: chili peppers, a
30-gallon tank, an undergraduate course's university. Code can see that
absence directly. LongMemEval-V2's best method, AgentRunbook-C (Wu et al.
2026, arXiv 2605.12493), is the only one there whose abstention improves,
and its memory module is "instructed to explicitly identify the
inconsistencies and wrong question premises and present them to the
downstream model". Handing over raw evidence alone "can be misled".

## Mechanism

`commit-arm --typed-premise` (M80; `bench::{unmentioned_terms,
premise_finding, commit_typed_with}`). It is M79's greedy typed pass, with
one line appended after the question:
- `<premise_check>No memory contains these words from the question: "chili",
  "peppers".</premise_check>`, or
- `<premise_check>Every word the question uses appears in the
  memories.</premise_check>`.

**The words:**
- the question's words of three or more letters, and numbers of any length
  (the "30" of a "30-gallon tank");
- minus 44 function words and M71b's connectives;
- matched against every shown memory up to one plural ending, the same
  rule as M71b's `memory_states`.

The commit rule is M79's, unchanged. The reader still decides, but it
decides knowing what the memories lack.

**Honest caveat:** M80's design came after seeing M79's six failures. Its
rule, "content words absent from every memory", is generic and was not
tuned to those rows, but it is measured on the same questions, and there is
no held-out split. A LoCoMo run is the transfer check if it passes.

## Measurement (a post-pass; minutes of GPU)

- **Gate arm:** the typed-premise pass over the base replicates
  (`m57_bonsai_premise_s1`, `r5_base_s2`, `r5_base_s3`), writing
  `m80_base_s{1,2,3}`. Paired against them, seed-averaged.
- **Stack:** the same pass over the shipped bundle's replicates
  (`r5_bundle_s*_grounded`), writing `r5_bundle_s*_m80`, and read against
  the base the way round 5 was.
- **Graders:** the strict 9B (preference rubric), seeded from each source,
  and LongMemEval's official grader.

## Gate (M79's, including the strict veto)

On the base replicates:
1. on M57's 49 answerable declines (`m71b_answerable_declines`), the strict
   paired Δ has a 95% CI excluding 0, *and*
2. the official Δ on the same rows is > 0, *and*
3. **the abstention veto:** the seed-mean abstention score under both
   readings is not below the base's.

If M80 passes, the stack is read against the shipped bundle's bar: it
ships only if it beats the shipped configuration without an abstention
loss. The +3.0 bar is from the round-5 pre-registration.

## Predictions

- False fits (abstention commits) fall from 6 to ≤ 1 per replicate.
- Answerable commits fall from ~20 to 12–18 per replicate. The finding
  also names details like the "5" of "5-day trip", and the reader will
  decline some of those.
- Strict on the 49 answerable declines: +10 to +18, with CI > 0.
- Stack: strict ≥ +3.0 over the base, with abstention at the base's level.

## Falsifiers

- **False fits stay ≥ 3 per replicate:** the reader ignores the finding.
  That is the one law again: information handed over as text is not
  structure.
- **Answerable commits fall below 8:** the finding scares the reader off
  true answers as well.

## Result — the veto fires again; the reader ignores the finding, and the finding is too coarse *(measured 2026-09-29, 06:18–07:04)*

The runs are `m80_base_s{1,2,3}` (the typed-premise pass over the base
replicates) and `r5_bundle_s*_m80` (the same pass over the shipped
bundle). They were graded by the strict 9B and the official grader.

| stratum | strict Δ [95% CI] | W/L | official Δ | W/L |
|---|---|---|---|---|
| M57's 49 answerable declines | **+19.7 [+9.5, +31.3]** | 10/0 | +8.8 [+0.0, +19.0] | 6/1 |
| **abstention (30)** | **−16.7 [−30.0, −3.3]** | 0/5 | −16.7 | 0/5 |
| overall | +1.0 [−0.5, +2.5] | 11/5 | −0.1 | 7/6 |

**Verdict:** criterion 1 passes. Criterion 2 is borderline, since the
official CI touches 0. **Criterion 3, the abstention veto, fails.** M80 does
not ship or stack.

**Predictions:**
- false fits 6 → ≤ 1: **wrong, 5 per replicate**;
- answerable commits 12–18: 21–24 per replicate;
- strict on the 49 +10 to +18: +19.7.

The falsifier "false fits stay ≥ 3: the reader ignores the finding"
**fires.**

**The stack on the shipped bundle:** strict 81.60 (+3.3 [+1.2, +5.5]), but
abstention 24/30 and official **80.60**, below the shipped 81.13.

**Why, row by row.** What the code showed the reader on the five abstention
rows it still answered:

| question (trap) | finding | reader's label |
|---|---|---|
| "bake for my **uncle's** party" (it was a niece) | `uncle` | detail unstated |
| "tomatoes and **chili peppers**" | *(empty)* | detail unstated |
| "which **university** … undergrad course" | `present`, `undergrad` | detail unstated |
| "collecting vintage **films**" (cameras) | `long` | detail unstated |
| "engineers … as Software Engineer **Manager**" | `just`, `started` | detail unstated |

**Two failure modes:**
1. **The reader ignores a correct finding.** "uncle" was named, and the
   reader still called it an unstated detail and answered. This is the one
   law again (M38–M43): information handed over as text does not move this
   reader's decision.
2. **Absence across all memories is too coarse.** "chili", "peppers" and
   "films" do occur somewhere in the evidence, just not about the thing
   asked. The traps swap a detail within a context where the word itself
   appears elsewhere, and a bag-of-words test cannot see that. Noise words
   ("long", "just", "started") dilute the rest.

**What this rules in and out:**
- A premise *finding* is not enough on this reader. A premise *decision*
  would have to be made in code, where the reader never gets a vote.
- It would also need to be finer than word absence, for example checking
  that a flagged word is absent from the memories the answer comes from,
  not from all of them.
- Declines stay the largest recoverable pool: 44 answerable declines.
  M79/M80's answerable gain (+19.7, 10/0) is real and repeatable, but it
  cannot ship while the same pass answers the traps.
