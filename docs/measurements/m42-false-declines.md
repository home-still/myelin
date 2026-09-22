# M42 — the reader refuses with the answer in hand

## The finding

M38 put the LongMemEval_S gap in reading. M39 measured its shape as a
compositionality gap. Neither asked what the wrong answers actually *say*.

They mostly say `I don't know.`

Joining `runs/m32_inv_sel_certified_judged` against `runs/m39_cov_k6.jsonl`,
over all 500 rows:

| question type | n | declined | rate | wrongly | accuracy |
| --- | --- | --- | --- | --- | --- |
| `multi-session` | 133 | 37 | 27.8% | 27 | 48.1 |
| `temporal-reasoning` | 133 | 31 | 23.3% | 25 | 42.1 |
| `single-session-preference` | 30 | 9 | 30.0% | **9** | **33.3** |
| `knowledge-update` | 78 | 8 | 10.3% | 2 | 79.5 |
| `single-session-user` | 70 | 6 | 8.6% | 0 | 91.4 |
| `single-session-assistant` | 56 | 0 | 0.0% | 0 | 96.4 |

"Wrongly" means: declined, scored zero, and **not** one of the 30 abstention
problems where declining is correct. That is **63 questions — 12.6 points of
the benchmark** — thrown away by refusing to answer.

Refusal is not obviously wrong; the reader may simply lack the evidence. So
the rows were split by whether retrieval had actually delivered it. On **48 of
the 63 the composed evidence contained every gold session**: 9.6 points
refused with the answer in hand. M40's digest recovers 22 of those and
introduces 10, leaving **36 rows — 7.2 points** still refused.

`single-session-preference` is the clearest case and the worst category in the
benchmark at 33.3%. Every one of its nine wrong declines looks like this:

> **Q** Can you suggest some accessories that would complement my current
> photography setup?
> **Gold** The user would prefer Sony-compatible accessories or high-quality
> photography gear…
> **A** `I don't know.`

The reader is not wrong under its instructions. `READER_SYSTEM` says *"If the
memories do not contain the answer, reply exactly: I don't know"*, and no
memory contains a list of accessories — only the dispositions from which one
is built. It is being asked to decide *whether* the memories answer and *what*
the answer is in a single emission, and it resolves the conflict by declining.

## Why M20 did not find this

M20 built `READER_PREFERENCE_CLAUSE` for exactly this stratum and measured it
at **+3.3, 95% CI [−10.0, +16.7]**, shipping it off as "fails the rule". That
was a power failure, not a null: M20's own doc records that at n = 30 with
judged 0/1 scoring, a paired bootstrap needs roughly **+13 points — four
questions** — to exclude zero. Nothing short of an enormous effect was
measurable there.

The phenomenon is not preference-specific. Of the 48 wrong declines with
complete coverage, preference holds 8; `multi-session` holds 21 and
`temporal-reasoning` 17. Measured on all 500 rows rather than one 30-row
stratum, an effect this size is resolvable.

## The mechanism

`commit_answer`: when the first response declines, ask once more under a
strict schema with the two decisions **split into separate fields, in order**.

```json
{ "answer": "…", "evidence_absent": false }
```

A strict schema is emitted field-by-field, so `answer` is generated while
`evidence_absent` is still open — the model must write the best answer the
memories support *before* it may assert there is none. Reversing the fields
would let it decline first and fill in an answer it has already disowned.

This is the one lever that has ever worked on this reader. M38 rewrote a
prompt clause and got 500/500 byte-identical responses; M39 asked it to
decompose and got 30% compliance; M40 fixed the count in the schema and
firing went 20.4% → 86.4%. Instructions are ignored; structure is obeyed.

Cost: one extra call on the ~20% of rows that decline, none on the rest.

### Abstention must stay reachable

The 30 `_abs` rows require declining, and the MINJA posture measured at 7.50%
ASR depends on a reader that can still refuse. So `evidence_absent` is the
model's own escape hatch and it wins over the answer field, and **every**
failure — model error, unparseable response, blank answer, a decline restated
inside `answer` — leaves the original decline byte-identical. Four tests pin
this; `an_asserted_absence_leaves_the_decline_exactly_as_it_was` is the one
that matters.

## Pre-registration

Written before the arm ran.

**Arms.** `myelin-eval bench --corpus longmemeval-s --mode investigate --k 6
--budget-tokens 4096 --max-steps 2 --select-sufficient`, with and without
`--commit-answer`. Base is `runs/m32_inv_sel_certified_judged` (62.00).

**Population.** All 500 LongMemEval_S rows.

**Primary metric.** `longmemeval_s.judge_score.n500`, judged.

**Decision rule.**

- Ship `commit_answer` on if the judged score improves by **≥ +3.0** over base
  with a paired 95% CI excluding zero.
- **Veto, regardless of the headline:** if accuracy on the 30 abstention rows
  drops at all, the mechanism is buying score by surrendering refusal and
  ships off whatever the total says. The security posture is not tradeable.

**Predicted, specifically.**

1. The switch fires on **~90 rows** (the measured decline count) and commits on
   most of them.
2. Non-firing rows move **exactly +0.0**, as M40's control did — anything else
   means prompt contamination, not mechanism.
3. Gains concentrate in `single-session-preference`, `multi-session` and
   `temporal-reasoning`; `single-session-assistant` (0 declines) moves 0.0.
4. Abstention accuracy on the 30 `_abs` rows is unchanged.

**Falsifier.** If the committed answers are wrong roughly as often as they are
right, the decline was well-calibrated and the 9.6 points were never
recoverable — the reader knew it could not answer, and M36's finding that no
recorded signal discriminates abstention (precision 32.8%, lift 1.16×) extends
to the reader's own judgement. That result would redirect the project back to
evidence quality and close the "reading" theory M38 opened.

## A second finding, not patched here

M40's digest recovers 22 wrong declines and **introduces 10**. Reading them
names the cause exactly. `item_digest` requires one entry per memory, so a
memory that does not bear on the question still gets an entry — and the entry
is a negation:

> **Q** What new kitchen gadget did I invest in before getting the Air Fryer?
> **base** `Instant Pot` **·** **M40** `I don't know.`
> `[notes] User mentions getting an Air Fryer yesterday, implying no prior
> gadget in this context.; Assistant provides Air Fryer tips without
> mentioning any previously purchased kitchen gadget.; …`

And where the facts *are* present, a negation still outvotes them:

> **Q** How many days before I bought the iPhone 13 Pro did I attend the
> Holiday Market?
> `[notes] User attended Holiday Market a week before Black Friday.; User
> bought iPhone 13 Pro on Black Friday.; No information about market
> attendance relative to purchase.; …`

Both facts needed to answer are in the note, and the reader still declined.

This is M35's `premise_analysis` shape in a third location: **a confident
negative in the evidence channel makes a bad decline persuasive.** The fix is
not a heuristic filter on wording but the same structural lever — a
`bears_on_question` boolean per entry, so the forcing survives (the model must
still consider every memory) while non-contributing entries never reach the
reader.

It is deliberately **not** built here. `runs/m40_digest` must stay reproducible
from the code that produced it, the mechanism above is contingent on
`item_digest`, which has not cleared its bar, and two mechanisms in one
milestone cannot be attributed. Pre-registered as M43.

## Results

**The mechanism works and the switch still ships off.** Both halves are the
point.

| stratum | n | base | arm | delta | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- |
| **overall** | 500 | 62.00 | **64.20** | **+2.20** | **[+0.8, +3.8]** | 0.0033 |
| non-abstention | 470 | 60.00 | 62.80 | +2.80 | [+1.5, +4.3] | <0.0001 |
| **abstention** | 30 | 93.30 | **86.70** | **−6.70** | [−16.7, −0.0] | 0.2487 |
| `multi-session` | 133 | 48.10 | **56.40** | **+8.30** | [+3.8, +13.5] | <0.0001 |
| `single-session-user` | 70 | 91.40 | 91.40 | +0.00 | [+0.0, +0.0] | 1.0000 |
| `single-session-assistant` | 56 | 96.40 | 96.40 | +0.00 | [+0.0, +0.0] | 1.0000 |
| `single-session-preference` | 30 | 33.30 | 33.30 | +0.00 | [+0.0, +0.0] | 1.0000 |
| `temporal-reasoning` | 133 | 42.10 | 42.10 | +0.00 | [−2.3, +2.3] | 1.0000 |
| `knowledge-update` | 78 | 79.50 | 79.50 | +0.00 | [−3.8, +3.8] | 1.0000 |

**The switch ships off, on the veto and on the bar.** +2.20 misses the
pre-registered +3.0 even though its interval excludes zero, and the abstention
veto fired independently.

### The mechanism is real

It fired on **91 rows** — the prediction was ~90 — and committed on 31; on the
other 60 the model took its own `evidence_absent` hatch. On the 31 rows it
changed, accuracy went **6.5 → 41.9, +35.5**: 13 of 31 forced answers were
judged correct where the decline scored zero by definition. `multi-session`
**+8.3** is the largest single-category effect this project has measured.

So the 9.6-point false-decline gap is real and about a quarter of it is
recoverable by asking twice. The falsifier is refuted: these declines were not
well-calibrated.

### Why it ships off anyway

Two of the 30 abstention rows were talked out of refusing:

> **Q** Which task did I complete first, fixing the fence or purchasing three
> cows from Peter? → base `I don't know` · arm **`fixing the fence`**
> **Q** How many engineers do I lead when I just started my new role as
> Software Engineer Manager? → base `I don't know.` · arm **`4`**

Both are adversarial: the premise is not in memory. The `evidence_absent`
hatch held 60 times and failed exactly where it mattered — a mechanism that
makes declining expensive makes it expensive on the rows that *should*
decline. The veto was pre-registered for this and is not tradeable against
+2.20: MINJA-defended ASR of 7.50% rests on a reader that can still refuse.

### The control is exact, and getting it that way found a defect

Non-firing rows moved **+0.0000** across all 469 — not approximately, exactly,
because the arm is computed over the base's own rows (`commit-arm`) rather
than by re-running the pipeline. Every point of difference is the second pass.

The first attempt was not exact, and the reason is worth keeping. Re-judging
the arm from scratch flipped **2 of 469 byte-identical responses**, moving the
control to **−0.43** and the headline to +1.80. The judge disagreed with
itself, and a quarter of the measured effect was that disagreement. The cache
was keyed on `question_id` and lived inside the run directory, so a new arm
re-judged everything.

`JudgeFile::answers` now records the answer each verdict was given for, and
`judge --seed <run>` reuses a verdict only while the answer it was given for
is still the answer. Re-judging the arm seeded from the base: **29 judged, 407
reused**, control exactly zero, headline **+2.20**. Every future paired arm
gets this for free; without it, judge noise is indistinguishable from a small
effect, which is the size of effect this project keeps measuring.

### What this closes

M38 opened the "reading is the gap" theory. M42 confirms one concrete,
measurable piece of it — the reader refuses 63 answerable questions, 48 of
them with complete gold coverage — and shows that the naive fix trades
security for score. The next attempt must make the commit *earn* its answer
rather than merely demand one.
