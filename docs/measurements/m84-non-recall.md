# M84 — a declined request for advice or an inference, asked again without the recall rule *(user decision 2026-09-29: "post-pass on those declines"; pre-registered before any row)*

## Why

The reader's system prompt says: *"If the memories do not contain the answer,
reply exactly: I don't know."* That rule is for recall. A request for advice,
or for a judgement about what is likely, has no recorded answer to find. So
the rule turns it into a refusal, and those refusals are where the largest
gaps to SOTA now sit:
- **LongMemEval_S preference:** 52.2% against MemPro-15's 80.0%, a gap of
  −8.3 questions under the official grader.
  - The shipped bundle declines 5–8 of these advice requests per seed ("Any
    tips?" → "I don't know."), several with the user's preference in hand.
  - M76b traced those declines to this line.
- **LoCoMo open-domain:** 36.5% under LightMem's grader.
  - 33 of its 61 losses are declines.
  - Many are "Would X likely…?" questions whose gold answer is an inference
    ("Likely no").

Refusing a request that calls for a response is over-abstention (Wen et al.
2024, "Know Your Limits", TACL, `10.1162/tacl_a_00754`; Brahman et al. 2024,
"The Art of Saying No", `10.52202/079017-1573`).

## Mechanism

`commit-arm --non-recall` (`bench::commit_non_recall`, and
`myelin_core::pipeline::query_shape::{is_non_recall_request,
is_inference_question}`).

1. **Only declined rows whose question has a non-recall shape are touched.**
   A non-recall shape is either:
   - an advice request (`is_advice_request`, M20b's cue list) made in the
     first or second person, or
   - an inference question: a modal (likely, might, would, could, probably)
     that is not addressed to "you".

   Every other decline, which includes every recall question, is kept.
2. **Those rows are asked again, greedy.** The call sends the base's own
   system prompt plus one instruction: the "I don't know" rule is for recall
   questions and does not apply here. Answer from what the memories say
   about the people involved, in at most three sentences.
   - The schema is `{answer}`, with no field to decline in.
   - An error, unparseable content, or a decline-shaped answer keeps the
     decline.
3. **The raw content goes into the trace** under `[m84:non-recall]`.

**The shape, measured on every question before any row.** Python's count
was reproduced by the Rust function, row for row:

| corpus | fires on | never fires on |
|---|---|---|
| LongMemEval_S | all 30 preference questions, 1 multi-session | **any of the 30 abstention traps** |
| LoCoMo | 42 of 96 open-domain, 1 of 841 single-hop | **any of the 446 adversarial**, multi-hop or temporal |

- **Why the advice cue needs a person:** LoCoMo asks recall questions
  *about* advice in the third person ("What advice did Calvin receive…?").
  12 adversarial questions carry an advice cue. Requiring "I", "me", "my" or
  "you" removes all 12, and keeps 29 of LongMemEval_S's 30 preference
  questions.
- **Rows it will act on:**
  - the shipped bundle's replicates hold **7, 5 and 8** shaped declines (0
    abstention);
  - `m63_locomo_base` holds **16** (0 adversarial).

**Honest caveats:**
- This is a reader-prompt change for one question shape. The user approved
  it as an exception to the code-first rule.
- The strict and official graders agree least on exactly these preference
  answers (κ 0.474, `judge-agreement-kappa.md`). So the strict reading may
  disagree with the official one.

## Measurement (post-passes; minutes of GPU)

1. **LongMemEval_S stack:** the shipped bundle's replicates
   `r5_bundle_s{1,2,3}_grounded` → `r5_bundle_s*_m84` (Bonsai).
   - The pairing is against the shipped bundle itself, on the 31 shaped
     rows (`--ids`), seed-averaged.
2. **LoCoMo:** `m63_locomo_base` → `m84_locomo_base` (the 9B), paired on
   the 43 shaped questions and overall.

Graders:
- the strict 9B judge (seeded from each source; LongMemEval_S preference
  rows use its preference rubric);
- LongMemEval's official grader, and LoCoMo's LightMem grader.

## Gate

**LongMemEval_S:**
1. On the 31 shaped rows: paired CI > 0 under the official grader **and**
   under the strict judge.
2. Abstention seed mean unchanged under both readings.
3. **Ships** (into the shipped LongMemEval_S point) only if 1 and 2 hold
   and the stack's seed means beat the shipped 81.13 official and 80.80
   strict.

**LoCoMo:**
1. At most 3 of 303 adversarial declines flip.
2. On the 43 shaped questions: paired CI > 0 under LightMem's grader and
   under the strict judge.
3. Overall strict Δ ≥ 0.
4. **Ships** (into the shipped LoCoMo point) only if 1–3 hold.

## Predictions

- **LongMemEval_S:**
  - 4–8 commits per replicate, 40–70% of them right under the official
    grader;
  - on the stratum, official +6 to +15 and strict +3 to +12;
  - stack seed means: official 81.6–82.2, strict 81.0–81.8.
- **LoCoMo:**
  - 12–16 commits, 5–10 right under LightMem;
  - the stratum +12 to +23;
  - LightMem overall +0.3 to +0.65, strict overall +0.2 to +0.5;
  - 0 adversarial flips.

## Falsifiers

- **Fewer than 30% of commits are right:** the answers are generic advice
  that the preference rubric rejects, so the gap is the reader's
  personalisation, not the decline.
- **Strict negative where official is positive:** the two rubrics part on
  these answers, as κ warned. Report both, and ship on neither.
- **Any adversarial flip:** the shape test leaks recall questions.
