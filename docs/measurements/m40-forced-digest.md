# M40 — taking the count away from the model

## What M39 left

M39 built self-ask against a measured compositionality gap and got a null:
**+0.80 (95% CI [−1.60, +3.20])** over 500 LongMemEval_S rows, and **exactly
+0.00 [−4.15, +4.15]** on its own target stratum. Splitting that stratum by
how many follow-ups the model actually produced explained the flatness:

| steps produced | n | base | self-ask | delta | 95% CI |
| --- | --- | --- | --- | --- | --- |
| ≥ 2 | 65 (30%) | 67.7% | 78.5% | +10.8 | [+1.5, +21.5] |
| < 2 | 152 (70%) | 52.0% | 47.4% | −4.6 | [−8.6, −1.3] |

The binding constraint was **the model's choice of how much to produce**. On
questions needing two gold sessions it asked two or more follow-ups only 30%
of the time — while holding **7 or 8 memories**. Every row in the run had 7–8
evidence items and 398 of 500 got zero or one follow-up.

That is the same reader that ignored M38's rewritten parsimony clause and
returned 500/500 byte-identical selections. It does not change behaviour on
instruction.

## The mechanism

`InvestigateConfig::item_digest`. One model call states what **every**
composed memory contributes to the question; the contributions are appended as
a single additive `[notes]` item.

The difference from `self_ask` is one line of schema:

```rust
"entries": { "type": "array", "minItems": n, "maxItems": n, … }
```

`n` is the number of real memories. Eight memories, eight entries, or the
response does not parse. **The count is taken out of the model's hands**,
which is the whole mechanism — asking nicely was measured twice and failed
twice.

Everything else is inherited from M39 and re-tested here: additive and never
destructive, one model call, fail-open, entries bound to memories **by index**
so a reordered response cannot misattribute a fact, and the note is a *view*
carrying the weakest trust among the items it draws on. `view_item` now
factors those last three invariants into one constructor so a third mechanism
cannot get them subtly wrong.

Still subject to `MIN_STEPS_EMITTED = 2`: M39 measured a one-line note costing
−4.6 points, so a digest where one memory contributes is not shown.

## Two defects the free pilot caught

A 24-question pilot, before any judged arm.

**1. The digest was digesting its own siblings.** `compose` appends a
`[timeline]` view before this runs, and the pilot's very first row restated
one fact three times — from the user turn, from the assistant's reply, and
from the timeline's gist of the same record:

```
[notes] Congratulations on your degree in Business Administration;
        I graduated with a degree in Business Administration;
        2023-05-30 +1d · assistant: Congratulations on your degree…
```

A view of a view. Fixed by offering only records with a non-nil `record_id`,
which is the invariant `view_item` exists to keep. Pinned by
`a_synthetic_view_is_not_itself_digested`.

**2. Exact duplicate facts.** A LongMemEval user turn and the assistant's
reply often carry the identical sentence, so the digest emitted "You graduated
with a degree in Business Administration." twice. Now de-duplicated on exact
match, case- and trailing-period-insensitive.

Exact match **only**. A similarity penalty here would be aimed at
near-duplicates, and M21 measured MMR over the reranked pool destroying gold
recall (0.658 → 0.550) precisely because co-evidence for one question
resembles *itself* 1.60× more than the rest of the set. Identical strings are
the one case where nothing can be lost.

## Does forcing the structure work?

That was the pilot's actual question, and the mechanism is pointless if the
answer is no.

| | rows with ≥ 2 contributions |
| --- | --- |
| M39 `self_ask`, all 500 rows | **20.4%** |
| M39 `self_ask`, two-fact rows | **30.0%** |
| M40 `item_digest`, pilot (n=24) | **62.5%** |

Roughly a threefold firing rate. The distribution is bimodal — `{0: 9, 2: 4,
6: 11}` — so the model either declines entirely or finds a contribution in
every memory; it does not produce the middle. That is a change in kind from
M39's "one and stop", and whether the over-inclusive half helps or merely
costs tokens is what the arm measures.

## Pre-registration

Written before the arm ran.

**Arms.** `myelin-eval bench --corpus longmemeval-s --mode investigate --k 6
--budget-tokens 4096 --max-steps 2 --select-sufficient`, with and without
`--item-digest`. One knob. Base is `runs/m32_inv_sel_certified_judged`
(62.00), same store, same operating point.

**Population.** All 500 LongMemEval_S rows.

**Primary metric.** `longmemeval_s.judge_score.n500`, judged. Coverage cannot
move — the mechanism adds no records — so only a judged number can test it.

**Decision rule.**

- Ship `item_digest` on if the judged score improves by **≥ +3.0** with a
  paired 95% CI excluding zero.
- Report the split by gold-fact count and by contributions emitted, whatever
  the headline does.
- **The pre-registered prediction is specific**, because M39's was and it
  failed: gains should concentrate on rows needing ≥2 facts, and rows needing
  one fact should not regress. M39's gains landed on `temporal-reasoning` and
  preference while its target stratum sat at +0.00; if that happens again the
  result is not evidence for the composition story whichever way the headline
  goes.

**What falsifies the M39→M40 reasoning rather than this mechanism.** M39's
+10.8 on its ≥2-step rows was conditioned on the mechanism's own output, so it
was descriptive and explicitly not a promise. If firing rate triples and the
score still does not move, that conditional gain was selection: the model
decomposed the rows it already answered. That would close off "make it fire
more often" as a direction, which is most of what remains after M25–M39.

## Results

**The headline misses the bar, so the default stays off. The pre-registered
prediction is confirmed — the first time since M32.**

| | base | digest | delta | 95% CI |
| --- | --- | --- | --- | --- |
| all 500 | 62.00 | **64.40** | **+2.40** | [−0.60, +5.60] |

38 rows gained, 26 lost. The bar was +3.0 with a CI excluding zero; +2.40
with an interval spanning zero meets neither, and a near miss is exactly what
a pre-registered rule is for. **Default stays off.**

### Forcing the structure worked

| | rows with ≥ 2 contributions |
| --- | --- |
| M39 `self_ask` | 20.4% |
| M40 `item_digest` | **86.4%** |

### The prediction, and it holds

| stratum | n | base | digest | delta | 95% CI |
| --- | --- | --- | --- | --- | --- |
| gold = 1, complete | 169 | 79.9 | 78.7 | −1.2 | [−5.3, +3.0] |
| **gold = 2, complete** | **217** | **56.7** | **62.7** | **+6.0** | **[+0.9, +11.1]** |
| gold ≥ 3, complete | 31 | 35.5 | 45.2 | +9.7 | [−3.2, +22.6] |
| **`multi-session`** | 121 | 44.6 | 53.7 | **+9.1** | **[+0.0, +18.2]** |
| `temporal-reasoning` | 127 | 39.4 | 43.3 | +3.9 | [−2.4, +10.2] |
| `knowledge-update` | 72 | 77.8 | **73.6** | **−4.2** | [−11.1, +2.8] |

Gains concentrated on the multi-fact strata, no regression on single-fact.
That is what was pre-registered and what M39 failed to produce — M39's target
stratum sat at +0.00 and `multi-session` went −2.26. The same stratum is now
**+6.0 with an interval excluding zero**.

The control is exact: on the **68 rows where the digest did not fire, the
delta is +0.0 with a CI of [+0.0, +0.0]** — byte-identical outcomes. The
mechanism is additive, and the arm measures it rather than prompt
contamination.

So M39's conditional +10.8 was **not** pure selection. Tripling the firing
rate did move the target stratum, which keeps "make it fire" alive as a
direction rather than closing it off.

### Why the headline is lower than the stratum

The target stratum is 217 of 500 rows. The rest dilute it, and one category
pays: `knowledge-update` at −4.2 over 72 rows.

Five knowledge-update rows regressed and the cause is visible in the notes.
**The digest discards the dates.** `41698283` asks which camera lens was
bought *most recently*; the composed evidence carries `[YYYY-MM-DD]` stamps
(`stamp_valid_time` ships on) and the note flattens them away:

```
[notes] User mentions recently getting a new 50mm prime lens.;
        User mentions using a new Canon EF lens for portrait work.;
        User mentions getting great shots with a new 70-200mm zoom lens.
```

Base answered "70-200mm zoom lens" (correct). With the digest the reader
answered "50mm prime lens" — the *first* line. Same shape in `6071bd76`
(coffee ratio, answered backwards) and `0977f2af`, where the digest asserted
"implying no prior gadget in this context" and the reader declined.

Knowledge-update is defined by recency. Flattening a dated evidence set into
an undated fact list destroys exactly the signal M19 measured at **+37.6** on
LoCoMo category 2.

**This was not patched before publishing.** The fix is obvious and the
evidence for it is five qualitative rows, not an interval; changing the
mechanism now would leave `runs/m40_digest` unreproducible from the code for
a change that has to be measured anyway. It is M41, pre-registered below.

## M41, pre-registered

Carry each contribution's date into its digest line, taken from the
`[YYYY-MM-DD]` prefix the composed item already carries.

- **Predicted:** `knowledge-update` recovers its −4.2 and `temporal-reasoning`
  (+3.9, CI [−2.4, +10.2]) tightens; the multi-fact gain is unchanged, since
  those questions do not turn on recency.
- **Bar:** unchanged — ≥ +3.0 over the 500 with a CI excluding zero, plus the
  same per-stratum report.
- **Falsifier:** if dating the lines does not move `knowledge-update`, then
  the regression is the digest's confident phrasing rather than its missing
  dates, and the next attempt is about hedging, not timestamps.
