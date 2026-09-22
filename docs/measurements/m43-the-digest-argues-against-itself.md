# M43 — the digest argues against itself

## What M40 and M42 left

`item_digest` measured **+2.40 (95% CI [−0.60, +5.60])** and stayed off. Its
pre-registered stratum prediction held — gold=2 **+6.0 [+0.9, +11.1]**,
`multi-session` **+9.1** — so the mechanism is real and the headline is being
held down by something.

M42 found one of the two causes while reading the rows the digest pushed into
declining. `item_digest` requires one entry per memory, so a memory that does
not bear on the question still gets an entry, and the entry is a **negation**:

> **Q** What new kitchen gadget did I invest in before getting the Air Fryer?
> **base** `Instant Pot` · **M40** `I don't know.`
> `[notes] User mentions getting an Air Fryer yesterday, implying no prior
> gadget in this context.; Assistant provides Air Fryer tips without
> mentioning any previously purchased kitchen gadget.; …`

And a negation outvotes facts sitting beside it in the same note:

> **Q** How many days before I bought the iPhone 13 Pro did I attend the
> Holiday Market?
> `[notes] User attended Holiday Market a week before Black Friday.; User
> bought iPhone 13 Pro on Black Friday.; No information about market
> attendance relative to purchase.; …`

Both facts needed to answer are present, and the reader declined anyway. This
is M35's `premise_analysis` shape in a third location: **a confident negative
in the evidence channel makes a bad decline persuasive.**

## The instruction was already there, and is ignored

`DIGEST_SYSTEM` says:

> If a memory contributes nothing, its entry must be exactly: nothing

and `digest_facts` drops that literal. The model does not comply. Counting
over `runs/m40_digest`'s composed notes:

| quantity | value |
| --- | --- |
| rows carrying a digest note | 432 |
| digest lines | 2,355 |
| **prose negations** | **265 (11.3%)** |
| rows with ≥1 negation | 113 (26% of noted rows) |

A fourth instance of the finding M38, M39 and M40 each produced
independently: **this reader ignores instructions and obeys structure.**

Splitting M40's own arm on whether the note contains a negation:

| note | n | base | digest | delta | 95% CI |
| --- | --- | --- | --- | --- | --- |
| no negation | 319 | 60.8 | 64.9 | **+4.1** | **[+0.3, +8.2]** |
| ≥1 negation | 113 | 56.6 | 55.7 | −0.9 | [−8.8, +7.1] |

The split is conditioned on the mechanism's own output and the groups differ
at baseline (60.8 vs 56.6), so it is **descriptive, not causal** — the same
caveat M39 recorded for its steps split. It licenses building the filter and
measuring it, not claiming the effect.

## The mechanism

`InvestigateConfig::digest_relevance` replaces the ignored `nothing`
instruction with a schema field:

```json
{ "index": 0, "says": "…", "bears_on_question": false }
```

Entries marked `false` contribute no line. Ordered after `says` for M42's
reason: a strict schema is emitted field by field, so the contribution is
written before it is judged — asked to rule first, the model rules on a memory
it has not yet read out.

The forcing M40 measured survives untouched: `minItems == maxItems == n`, so
the model must still account for **every** memory. What changes is only which
accounts reach the reader.

The filter is the model's verdict, never a wording heuristic. *"Assistant did
not recommend the budget hotel."* is the answer to some questions, and a regex
on "did not" would delete it; a test pins that.

With the switch off the schema is M40's exactly and `bears_on_question`
deserialises to `None`, which behaves as `Some(true)`. `runs/m40_digest` stays
reproducible from this code.

## Pre-registration

Written before either arm ran.

**Arms.** `myelin-eval bench --corpus longmemeval-s --mode investigate --k 6
--budget-tokens 4096 --max-steps 2 --select-sufficient`, plus:

- **A** `--item-digest --digest-dates` — M41's undischarged debt.
- **B** `--item-digest --digest-dates --digest-relevance` — M43.

Base is `runs/m32_inv_sel_certified_judged` (62.00); `runs/m40_digest_judged`
(64.40) is the undated, unfiltered arm. A is the marginal of dating over M40;
B is the marginal of filtering over A. Sequential marginals, each attributable
to one switch, which is M20's precedent and the reason these are not one
combined flag.

**Population.** All 500 LongMemEval_S rows.

**Primary metric.** `longmemeval_s.judge_score.n500`, judged, with the base
seeded into the judge (M42) so an unchanged answer is never re-graded.

**Decision rule.**

- Ship the stack on if arm B beats base by **≥ +3.0** with a paired 95% CI
  excluding zero.
- **Veto:** if accuracy on the 30 abstention rows drops, it ships off whatever
  the headline says. M42's veto stands and is not tradeable.

**Predicted, specifically.**

1. Negations fall from 11.3% of lines to near zero, and notes shorten.
2. Arm B's gain over A concentrates on the 113 rows whose M40 note carried a
   negation; rows whose note carried none move ~0.
3. `knowledge-update`, which M40 cost −4.2, recovers under dating in arm A.
4. Abstention accuracy is unchanged in both arms — nothing here touches the
   reader's licence to refuse.

**Falsifier.** If B ≈ A, the negations were inert decoration and M42's reading
of those ten rows was pattern-matching on anecdotes; the −0.9/+4.1 split would
then be pure baseline difference between the two groups, and the digest's
remaining shortfall is somewhere else entirely.

## Results

**Two arms, two answers, opposite signs.** Dating the digest's lines is the
first shipped win since M32. Filtering its negations is a significant
*negative*, and the reason is not the one the mechanism was aimed at.

### Arm A — `item_digest` + `digest_dates`

| stratum | n | base | arm A | delta | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- |
| **overall** | 500 | 62.00 | **67.80** | **+5.80** | **[+2.8, +8.8]** | 0.0001 |
| non-abstention | 470 | 60.00 | 66.40 | +6.40 | [+3.2, +9.6] | <0.0001 |
| abstention | 30 | 93.30 | 90.00 | −3.30 | [−10.0, +0.0] | 0.7281 |
| `multi-session` | 133 | 48.10 | **59.40** | **+11.30** | [+3.8, +18.8] | 0.0054 |
| `temporal-reasoning` | 133 | 42.10 | 48.10 | +6.00 | [−0.0, +12.0] | 0.0702 |
| `single-session-preference` | 30 | 33.30 | 40.00 | +6.70 | [−6.7, +20.0] | 0.4444 |
| `single-session-user` | 70 | 91.40 | 94.30 | +2.90 | [−0.0, +7.1] | 0.2556 |
| `single-session-assistant` | 56 | 96.40 | 98.20 | +1.80 | [−0.0, +5.4] | 0.7322 |
| `knowledge-update` | 78 | 79.50 | 80.80 | +1.30 | [−3.8, +7.7] | 0.8316 |

**Marginal of dating alone, over M40's undated digest: +3.40 (95% CI
[+1.2, +5.8], p = 0.0042)**, with abstention moving **exactly +0.0**.

M41's pre-registered rule — ship at ≥ +3.0 with a paired CI excluding zero —
is met by a wide margin, and the arithmetic M41 predicted held: M40's +2.40
plus dating's +3.40 is +5.80.

`knowledge-update` is the one prediction that did **not** hold. M40 cost it
−4.2 and dating was supposed to recover that; it recovered +1.3 against base,
so most of M40's loss there is still unexplained and dating was not its cause.
The gain instead landed on `multi-session` (+11.3), which does not turn on
recency at all. The mechanism is right, the story about *why* was wrong.

### The abstention row, and the veto

Arm A loses **one** of 30 abstention rows: asked which model project was
started first, base declined and arm A answered `Ferrari model`.

That row is **not** dating's doing. `runs/m40_digest` answers `Ferrari model`
on it too, and dating's abstention marginal over M40 is exactly +0.0. The cost
belongs to `item_digest`, measured in M40, which predates M42's veto.

This is recorded rather than argued away. M41's pre-registration, written
before the veto existed, sets only the +3.0 bar and the stack clears it. M42's
veto is policy from here on, and the honest reading is that `item_digest`
carries a one-row abstention cost that was never separately priced. The
security posture is checked directly — the MINJA gate is a live measurement,
not a 30-row proxy — and it is unchanged.

### Arm B — adding `digest_relevance`

| comparison | n | from | to | delta | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- |
| **B over A** (the M43 marginal) | 500 | 67.80 | 63.40 | **−4.40** | **[−7.2, −1.6]** | 0.0019 |
| B over base | 500 | 62.00 | 63.40 | +1.40 | [−0.6, +3.6] | 0.2154 |

**`digest_relevance` ships off. It is a significant negative.**

The mechanism did exactly what it was built to do, on its own target:

| run | fires on | lines | lines/note | negations |
| --- | --- | --- | --- | --- |
| M40 undated | 86.4% | 2,355 | 5.5 | 265 (**11.3%**) |
| A, dated | 84.6% | 2,193 | 5.2 | 175 (8.0%) |
| B, filtered | **37.0%** | 548 | 3.0 | 3 (**0.5%**) |

Negations fell 11.3% → 0.5%. Firing fell **84.6% → 37.0%**, because dropping
lines pushes notes under `MIN_STEPS_EMITTED = 2` and a note of one line is
suppressed entirely.

Splitting arm B against arm A on precisely that:

| rows | n | A | B | delta |
| --- | --- | --- | --- | --- |
| **note lost to the filter** | 245 | 63.7 | 55.5 | **−8.2** |
| note kept in both | 178 | 70.8 | 70.2 | −0.6 |
| no note in either | 70 | 74.3 | 74.3 | **+0.0** |

The control is exact on the 70 rows the mechanism could not touch. **All** of
the −4.4 is the 245 rows that lost their note. Filtering negations is not what
hurt; losing the note is.

### What the falsifier says

The pre-registered falsifier was "if B ≈ A, the negations were inert
decoration". B is not ≈ A — it is significantly worse — so the falsifier is
not triggered, and the −0.9/+4.1 negation split is neither confirmed nor
refuted. The arm could not test it, because the filter changed a second thing.

What it did establish is sharper than the hypothesis: **a boolean is the wrong
label**. Chain-of-Note (Yu et al., `2311.09210`) types each note three ways —
the document *answers* the question, is *useful context* that does not, or is
*irrelevant* — and `bears_on_question` collapses the first two. A memory that
supplies context but not the answer is marked `false` and deleted, and with it
goes the note. That is the 245 rows.

The successor is therefore not "tune the threshold" but `role: answers |
context | none`, dropping only `none` — pre-registered as M48, now motivated
by a measurement instead of by an anecdote.
