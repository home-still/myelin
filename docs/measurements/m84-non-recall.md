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

## Result — ships for LoCoMo (78.18 → **78.64**); fails its LongMemEval_S gate narrowly *(measured 2026-09-29, 12:39–13:47, main db4f504)*

**LoCoMo** (`m84_locomo_base`, the 9B): every criterion holds.

| reading | the 43 shaped questions | all 1,540 non-adversarial |
|---|---|---|
| LightMem's grader | 30.2 → 46.5, **+16.3 [+7.0, +27.9]** (7/0, sign p 0.016) | 78.18 → **78.64** |
| MemPro's repo judge | | 80.26 → **80.78** |
| strict 9B judge | 27.9 → 39.5, **+11.6 [+2.3, +20.9]** (5/0) | 70.52 → **70.84** |

- 14 of the 16 shaped declines were answered.
- Adversarial flips: **0 of 303**. Every adversarial row is byte-identical
  to the base's.
- The answers are inferences with their reason. Examples:
  - "Would Melanie go on another roadtrip soon?" → "No, Melanie recently had
    a traumatic roadtrip accident…" (gold "Likely no; since this one went
    badly");
  - "What would Caroline's political leaning likely be?" → "…likely liberal
    or progressive…" (gold "Liberal").

**Shipped for LoCoMo** (`bench::shipped_commit_non_recall`):
- the matched gate row is now **78.64** against MemPro-15 (Qwen3-30B)'s
  77.85, **+0.79** (it was +0.33);
- the shipped LoCoMo run is `runs/m84_locomo_base`.

**LongMemEval_S** (`r5_bundle_s*_m84`, stacked on the shipped bundle,
Bonsai): criterion 1 fails.

| reading | the 31 shaped rows | stack seed mean | shipped |
|---|---|---|---|
| official grader | +6.5 [**+0.0**, +15.1] (3/0) | 81.53 | 81.13 |
| strict 9B judge | +4.3 [**+0.0**, +11.8] (2/0) | 81.07 | 80.80 |

- Abstention is unchanged (official 28.0, strict 29.33 of 30).
- Both stratum CIs touch 0, so M84 **ships off for LongMemEval_S**, even
  though the stack's means beat the shipped ones.

**Why LongMemEval_S moved so little: Bonsai declined inside the answer.**
- Of 20 re-asked rows over the three seeds, 12 came back as "I don't know…"
  written into the `{answer}` field, even though the schema has no decline
  field and the instruction says the rule does not apply.
- Several of them held the preference in hand. The battery question got
  "the memories only cover portable power banks… not phone battery
  troubleshooting", and the power bank *is* the preference the rubric
  wants.
- The 8 that were answered were right 6 times under the official grader.
- The 9B on LoCoMo followed the instruction; Bonsai on LongMemEval_S mostly
  did not.

**Predictions:**
- LoCoMo:
  - 12–16 commits: **14, held**;
  - 5–10 right under LightMem: **7, held**;
  - stratum +12 to +23: **+16.3, held**;
  - LightMem overall +0.3 to +0.65: **+0.46, held**;
  - strict overall +0.2 to +0.5: **+0.32, held**;
  - 0 flips: **held**.
- LongMemEval_S:
  - 4–8 commits per replicate: 3, 1 and 4, **wrong**;
  - 40–70% right: 75%, above;
  - stratum official +6 to +15: +6.5, held;
  - stratum strict +3 to +12: +4.3, held;
  - stack official 81.6–82.2: 81.53, just under;
  - stack strict 81.0–81.8: 81.07, held.

**Also found while shipping (fixed in the same PR):**
- **The standing tool's arm check did not know `commit_non_recall`.** Any
  M84 run, including the LongMemEval_S ones that fail, would have been
  quoted as the shipped system.
  - A test fixture that should have caught it passed vacuously: its
    "shipped" LoCoMo base was already an arm.
  - The test now asserts that the shipped base is not an arm.
- **`commit_answer`** ("a second pass ran") is now expected wherever any
  second pass ships (`shipped_commit_answer`), not only where M71b's does.
- **The LoCoMo abstention pin moved 69.96 → 67.94 by hand.**
  - The old pin was `runs/m19_locomo_full` (M19, 2026-09-20). M84 makes that
    run an arm.
  - The shipped LoCoMo base has read 67.94 on the 446 adversarial questions
    since M63, and M84 changes none of those rows.
  - So the old pin was a number from an older configuration, not a loss.
- **serde_json now parses floats exactly** (`float_roundtrip`). The
  ratchet's own pin 60.484971494365276 read back one ULP high and failed
  `ratchet --strict` against the value it had just written.
