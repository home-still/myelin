# M83 — the speaker contrast: an entailed claim must not also hold for the other speaker *(pre-registered 2026-09-29, before any row)*

## Why

M82 put an NLI model in charge of the premise. On LongMemEval_S it was
safe: 0 false fits, and the stack beat the shipped bundle (official 81.40,
strict 81.20). On LoCoMo it failed: **13 of 303 adversarial declines
flipped**, against a limit of 3 (`m82-nli-premise.md`).

Each flip is a person swap. The cited turn is Melanie's ("I ran a charity
race…") and the statement is about Caroline ("Caroline realized self-care is
important after her charity race"). The NLI model entailed it: it does not
bind a turn's first person to the name in front of it.

## The screen (exploratory, 2026-09-29; big_mac, MPS, M82's NLI model)

The data is every typed candidate M82 produced that passed M79's label rule:
- 249 in all;
- LongMemEval_S: 69 answerable and 9 traps over three replicates;
- LoCoMo (`m82_locomo_base`): 64 answerable and 107 adversarial.

A Python copy of M82's statement guard reproduces the Rust decisions on all
249 (0 mismatches), so the variants below differ from M82 only in the rule
tested.

| variant | LME answerable commits | LME traps | LoCoMo answerable commits | **LoCoMo flips** |
|---|---|---|---|---|
| M82 as run | 13 | 0 | 5 | **13** |
| speaker-resolved premise (turns rewritten in the third person) | 12 | 0 | 5 | 10 |
| contrast, P(ent) above the swap's | 13 | 0 | 4 | 4 |
| **contrast, the swap not entailed (argmax)** | **13** | **0** | **3** | **0** |
| + the question's head word exempt from the guard | 16 | 0 | 3 | 2 |

**What each row taught:**
- **The recorded fix, a speaker-resolved premise, is not enough.** Rewriting
  "I ran" as "Melanie ran" still left 10 of 13 swaps entailed. The model
  reads "Hey Caroline" in the same turn and entails the claim about her
  anyway.
- **Swapping the name in the statement separates them.** Replace the named
  speaker with the other one ("Melanie realized…") and ask again. The four
  swaps that survive a probability comparison do so by hair-widths (0.43 vs
  0.42, 0.99 vs 0.98, 1.00 vs 0.99, 1.00 vs 1.00). The model entails both
  claims, which means the memories do not say whose claim it is.
- **Using the model's own decision on the swap closes all 13** and keeps
  every LongMemEval_S commit. It costs 2 of LoCoMo's 5 answerable commits,
  one of them right ("2 weeks").
- **The head-word exemption was screened and rejected.** It exempts the noun
  a *what/which* question asks for, so "What **brand** of shampoo…" can be
  answered "Trader Joe's". It adds 3 LongMemEval_S commits (one question on
  three seeds). But it lets through "What **cult** did Tim join?", answered
  "travel club": there the head noun *is* the trap.

## Mechanism

`commit-arm --typed-nli --speaker-contrast --nli-url …`
(`bench::{dialogue_speakers, speaker_swap, commit_typed_nli}`).

1. **M82 unchanged up to its decision:**
   - the typed pass writes `{answer, mismatch, statement, supporting}`;
   - code checks the label, the citations, and that the statement keeps
     every question word;
   - the NLI model must entail the statement from the cited memories.
2. **The dialogue's speakers:**
   - these are the names that speak in the shown memories;
   - a memory counts when its first line is a turn (`[date] Name: text`) and
     it holds turns by at least two different names;
   - a name is one capitalised alphabetic word.
3. **The contrast:**
   - it applies when there are exactly two speakers and the statement names
     exactly one of them, as a whole word;
   - every mention of that speaker becomes the other, and the NLI model
     judges the swapped statement against the same premise;
   - **the row commits only if the swap is not entailed.**
   - Both decisions are the model's argmax, so there is no threshold.
4. **The trace records** both probability triples (`nli=…` and `swap=…`).
   So the readout can say what M82 alone would have done on every row.
5. **The contrast can only block a commit, never create one.**

Grounds:
- the entity swap is FactCC's canonical transformation for making a
  factually inconsistent claim (Kryściński et al. 2020,
  `10.18653/v1/2020.emnlp-main.750`);
- here it is used as a contrast set at decision time (Gardner et al. 2020,
  `10.18653/v1/2020.findings-emnlp.117`): a claim the evidence supports
  equally about either speaker is not attributed by that evidence;
- NLI over dialogue turns needs the speaker bound to the claim (Welleck et
  al. 2019, Dialogue NLI, `10.18653/v1/P19-1363`).

**Honest caveats:**
- The rule was chosen on `m82_locomo_base`'s rows, so its 0 of 303 there is
  in-sample. It is reported, not gated.
- **The contrast cannot fire on LongMemEval_S.** Its turns are `user:` and
  `assistant:`, which are not names. Checked on all 3,000 rows of the six
  replicates M82 read: 0 have dialogue speakers. **M83's LongMemEval_S
  result is M82's by construction**, so LongMemEval_S is not rerun.

## Measurement (post-passes on big; the 9B reader, LoCoMo's shipped reader)

1. **Design run:** `m63_locomo_base` → `m83_locomo_base` (419 declines, 303
   adversarial).
2. **Held out:** two LoCoMo runs whose evidence and statements M83 never saw.
   - `m50c_locomo_events` → `m83_locomo_m50c`: recall over the events ledger;
     389 declines, 277 adversarial.
   - `m51_locomo_s1` → `m83_locomo_m51`: investigate mode; 572 declines, 375
     adversarial.

   Their adversarial questions overlap the design run's. The readout also
   reports flips on questions outside the 107 design candidates.

Graders:
- the strict 9B judge, seeded from each base;
- LoCoMo's official LightMem grader (gpt-4o-mini) on the design run.

## Gate

1. **Held-out flips at most 1% of each run's adversarial declines (rounded
   down), the same rate as M82's 3 of 303:**
   - `m83_locomo_m50c` ≤ 2 of 277;
   - `m83_locomo_m51` ≤ 3 of 375.
2. **No LoCoMo loss:** on each of the three runs, strict Δ ≥ 0 against its
   base (right commits at least match flips).

## What passing means (the ship rule stays M82's)

M82's ship rule needs:
- its LongMemEval_S criterion: stratum CI > 0 on the 49 answerable declines;
- the LoCoMo transfer;
- the stack beating the shipped bundle under both readings.

M83's LongMemEval_S rows are M82's, and M82's stratum CI was
+6.1 [+0.0, +14.3]: it touches 0.

- **If the gate passes,** M83 clears the LoCoMo criterion M82 failed, and
  only the narrow LongMemEval_S stratum miss remains.
- **Whether that ships goes to the user,** with:
  - the stack: official 81.40, strict 81.20;
  - overall +0.7 [+0.1, +1.5] (4/0);
  - 0 false fits.
- **If the gate fails, M83 does not ship,** and the swaps it missed are
  recorded.

## Predictions

- Design run: 0–1 flips and 2–4 answerable commits.
- Held out: 0–2 flips per run and 2–8 answerable commits per run.
- On held-out rows, M82 alone would have flipped 5–15 per run. The contrast
  removes most of them.
- LongMemEval_S: the contrast fires on 0 rows.

## Falsifiers

- **Held-out flips over the limit:** the contrast does not generalise
  beyond the rows it was chosen on. Report the statements it missed. A
  likely cause is a statement that names neither speaker ("She realized…"),
  where no swap exists.
- **Held-out Δ < 0:** the right commits the contrast keeps are fewer than
  the swaps it misses.
- **Answerable commits collapse to 0 on the held-out runs:** the model
  entails both speakers almost always, and the rule blocks everything.

## Result — the contrast removes 30 of 34 swaps, but one held-out run fails narrowly; does not ship *(measured 2026-09-29, 11:13–11:56, main 4a48fb8)*

| run | adversarial declines | **flips** (limit) | M82 alone would have flipped | answerable commits (right) | strict Δ, all 1,986 rows |
|---|---|---|---|---|---|
| `m83_locomo_base` (design) | 303 | **0** (reported only) | 13 | 3 (1) | +0.1 [+0.0, +0.2] (1/0) |
| `m83_locomo_m50c` (held out) | 277 | **0** (≤ 2) ✅ | 4 | 4 (3) | +0.2 [+0.0, +0.4] (3/0) ✅ |
| `m83_locomo_m51` (held out) | 370 | **4** (≤ 3) ❌ | 17 | 8 (3) | −0.1 [−0.3, +0.2] (3/4) ❌ |

- On the design run, LoCoMo's LightMem grader gives +0.1 [+0.0, +0.3] (2/0).
- **"M82 alone"** is read from each row's trace: the NLI decision before the
  contrast.
- **The adversarial count on `m51`:** the pre-registration's 375 came from a
  looser decline match. The readout's `I don…` prefix gives 370. The limit
  is 3 either way.

**Verdict:**
- **Gate criterion 1 fails on `m51`** (4 flips, limit 3). **Criterion 2
  also fails there** (3 wins, 4 losses).
- `m50c` passes both criteria.
- **M83 does not ship,** so the ship question for M82 plus M83 does not
  arise.

**What the contrast did:**
- It is the mechanism it was built to be. Across the three runs it took
  M82's person-swap flips from **34 to 4** (13 → 0, 4 → 0, 17 → 4).
- It kept 15 answerable commits, 7 of them right.

**Why the four got through: the statement disputed the premise instead of
asserting it.** In each, the reader's `statement` is a correction, which the
NLI model rightly entails:

| question | the reader's statement |
|---|---|
| What did Nate take a picture of near Fort Wayne? | "Nate did **not** take a picture of a sunflower…; **Joanna** did." |
| What inspired James to create his painting? | "Witcher 3 inspired James to create his game, **not** a painting." |
| What did Sam share a photo of…? | "Sam did **not** share a photo of a kayak…; **Evan** took that photo." |
| What game has Joanna been playing…? | "Joanna mentioned that **Nate** has been playing… Cyberpunk 2077." |

- Three name both speakers, so there is no swap to test.
- The pass commits `answer` anyway, because M79's label said `none`.
- This is the reader's premise vote failing once more (M79 measured it at
  ~57% false challenges). NLI read these statements correctly.
- The pre-registered falsifier fired, but not for the predicted cause (a
  statement naming neither speaker).

**Predictions:**
- design run 0–1 flips: **0, held**;
- design run 2–4 answerable commits: **3, held**;
- held out 0–2 flips per run: **0, held**, and **4, wrong**;
- held out 2–8 answerable commits: 4 and 8, held;
- M82 alone 5–15 flips per run: 4 and 17, wrong at both ends.

**Recorded, not queued: M83b, the statement must assert the question.**
- The rule, in code: reject a statement that
  - adds a negation the question lacks ("not", "never", "n't"), or
  - names a dialogue speaker the question does not name.
- *Post hoc* on these three runs, it blocks all 4 flips and none of the 7
  right answers (exploratory; found on `m51`).
- A real test needs LoCoMo runs M83 never saw, such as
  `m55b_locomo_bonsai` or `m50b_locomo_events`.
- The LongMemEval_S side stays M82's: stack official 81.40 against the
  shipped 81.13, with a stratum CI that touched 0.
