# M39 — the compositionality gap, measured and attacked

## The diagnosis

M38 established that LongMemEval_S retrieval is solved. This milestone asks
what the reader does with it.

**Measured at the operating point that actually scored 62.00** — k=6,
`budget_tokens` 4096, `investigate`, `max_steps` 2, `select`. That
qualification is load-bearing and was nearly got wrong: the first version of
this table joined M38's coverage sweep, run at k=25 / 10,000 tokens, against
M32's judged rows at k=6 / 4,096. The filter "the reader had every gold
session" would then have been computed at a configuration the reader never
ran. Re-measured at k=6 the coverage is **any 93.0%, complete 83.4%, mean
coverage 89.2%** at 7.0 emitted items.

Keeping only rows where retrieval delivered **every** gold session:

| gold sessions needed | n | accuracy | n (incomplete) | accuracy |
| --- | --- | --- | --- | --- |
| 1 | 169 | **79.9%** | 1 | 0.0% |
| 2 | 217 | **56.7%** | 12 | 0.0% |
| 3 | 25 | **40.0%** | 14 | 7.1% |
| 4+ | 6 | 16.7% | 26 | 46.2% |

Accuracy collapses from one fact to two and keeps falling. 417 of the 470
answerable rows have complete coverage even at k=6, so this is the common
case and not a corner.

The incomplete column is the metric's own sanity check: when a gold session
is missing, accuracy is 0.0% at one and two facts. Coverage is measuring
something real. (The 4+ row inverts — n=6 complete against n=26 incomplete,
and the incomplete ones score higher — which is why it is reported and not
leaned on.)

The distractor load is flat across these rows, so this is not distraction; it
is composition.

Two framings were tested and rejected first:

- **Not arithmetic.** Aggregation questions ("how many", "how much",
  "total", numeric gold) are *not* the failing set: within
  `temporal-reasoning` they score 37.1% against 41.5% for the rest, and
  within `multi-session` they score **better** (45.5% vs 40.0%). The fact
  count separates the failures; the question's surface form does not.
- **Not the shipped temporal machinery being off.** `resolve_relative`,
  `timeline` and `stamp_valid_time` are all `true` by default, so M19's wins
  are already inside these numbers.

## What this is

Press et al., *Measuring and Narrowing the Compositionality Gap in Language
Models* (2210.03350), name the quantity:

> We measure how often models can correctly answer all sub-problems but not
> generate the overall solution, a ratio we call the compositionality gap …
> as model size increases we show that the single-hop question answering
> performance improves faster than the multi-hop performance does, therefore
> **the compositionality gap does not decrease**.

Two consequences. A larger reader will not fix this, so it is our problem and
not a hardware budget. And their remedy is specific:

> We present a new method, self-ask, that further improves on chain of
> thought. In our method, the model explicitly asks itself (and answers)
> follow-up questions before answering the initial question.

## The mechanism

`InvestigateConfig::self_ask`. After the evidence is composed, one model call
decomposes the question into at most four follow-ups, answers each **from the
composed evidence only**, and the resolved pairs are appended as a single
`[notes]` evidence item.

Done in the memory layer rather than by instructing the reader, because M19
measured that asymmetry on this codebase: resolving dates *for* the reader was
**+37.6** on LoCoMo category 2, while telling the reader to resolve them
itself was **+14.3** — a third of the effect for the same information.

Four properties, each a test:

1. **Additive and never destructive.** Existing items are not reordered,
   rewritten or dropped. A question the reader already answers sees its
   evidence unchanged but for one appended line. This is the property M36's
   `Supported` branch had and `premise_analysis` lacked, and it is what makes
   the arm measure the mechanism instead of prompt contamination.
2. **Unresolved follow-ups are dropped, not shown.** A `[notes]` line reading
   "how much was the helmet — unknown" is an argument for declining delivered
   through the evidence channel, which is the shape that cost
   `premise_analysis` −8.75 by tripling declines on answerable questions.
3. **The note is a view and carries the weakest trust it saw.** `record_id` is
   nil, source is the literal doc `self-ask`, and trust is the weakest tier
   among the items it draws on. Restating an `Untrusted` memory's claim at
   `Verified` would hand the M11 attack suite a free promotion — the poison
   arrives twice, the second time wearing better credentials.
   `compose::weakest_trust` became `pub(crate)` so both synthetic items
   (`[timeline]`, `[notes]`) owe the same guarantee.
4. **Fail-open.** A refused, unparseable or empty response appends nothing.

Truncation note: the decomposer sees 1,200 characters per item. A head is
legitimate here and was not for M36's answerability gate, by M36's own rule —
*selection may truncate, judgement may not*. That gate asked an absolute
question (is the answer present?) and a 2,000-char head produced a 54.5%
false-refusal rate. This asks a relative one: which memories bear on the
question and what do they say.

## Pre-registration

Written before the arm ran.

**Arms.** `myelin-eval bench --corpus longmemeval-s --select-sufficient`
against the same plus `--self-ask`. One knob. Both `investigate`,
`max_steps 2`, k=25.

**Population.** All 500 LongMemEval_S rows.

**Primary metric.** `longmemeval_s.judge_score.n500`. This arm **must** be
judged — coverage cannot move, because the mechanism adds no records and
retrieval already delivers the gold. M38's central finding is that the reader
fails with the evidence in hand, so only a judged number can test it.

**Decision rule.**

- Ship `self_ask` on if the judged score improves by **≥ +3.0 points** over
  the 500 with a paired 95% CI excluding zero.
- Report the split by gold-fact count (1 / 2 / 3 / 4+) whatever the headline
  does. The mechanism predicts gains concentrated on the multi-fact rows and
  **no change on the single-fact rows**; a uniform shift, or a gain on
  single-fact rows, would mean something other than composition moved and the
  result should not be read as confirming the diagnosis.
- Report `asked_steps` distribution. A mechanism that resolved nothing is a
  null for a mechanism that never ran, not a null for self-ask.

**What would falsify the diagnosis rather than the mechanism.** If
`asked_steps` is healthy (≥2 on most multi-fact rows) and accuracy still does
not move, then handing the reader resolved sub-answers is not enough, and the
next attempt has to answer the question in the memory layer rather than
pre-digesting it.

## Results

**Null on the headline. Default stays off. And the pre-registered split
contradicts the mechanism's own prediction, which is the result worth
keeping.**

| | base | self-ask | delta | 95% CI |
| --- | --- | --- | --- | --- |
| all 500 | 62.00 | **62.80** | **+0.80** | [−1.60, +3.20] |

20 rows gained, 16 lost, against a pre-registered bar of +3.0. Paired
bootstrap, 10,000 resamples, seed 20250922.

### The mechanism ran

Required by the pre-registration, because "a mechanism that resolved nothing
is a null for a mechanism that never ran". **286 of 500 rows carry a
`[notes]` item**, with 1–4 resolved steps. It is live; this is a null for
self-ask.

### The prediction was that gains concentrate on multi-fact rows

| stratum | n | base | self-ask | delta | 95% CI |
| --- | --- | --- | --- | --- | --- |
| gold = 1, complete | 169 | 79.9% | 81.1% | +1.2 | — |
| **gold = 2, complete** | **217** | **56.7%** | **56.7%** | **+0.00** | **[−4.15, +4.15]** |
| gold ≥ 3, complete | 31 | 35.5% | 45.2% | +9.7 | [−3.23, +22.58] |
| `temporal-reasoning` | 133 | 42.11 | 45.86 | +3.76 | [−0.75, +9.02] |
| `multi-session` | 133 | 48.12 | 45.86 | **−2.26** | [−7.52, +3.01] |
| `single-session-preference` | 30 | 33.33 | 40.00 | +6.67 | [−6.67, +20.00] |

The largest multi-fact stratum moved **exactly zero**, and `multi-session` —
the category most defined by cross-session composition — went **down**. Every
interval straddles zero. The pre-registration said a pattern other than
"gains on multi-fact, no change on single-fact" must not be read as
confirming the diagnosis, and this is such a pattern.

### Why it is flat: the decomposer mostly does not decompose

Splitting the target stratum by how many steps the model actually produced:

| steps produced | n | base | self-ask | delta | 95% CI |
| --- | --- | --- | --- | --- | --- |
| **≥ 2** | 65 (30%) | 67.7% | **78.5%** | **+10.8** | **[+1.5, +21.5]** |
| **< 2** | 152 (70%) | 52.0% | **47.4%** | **−4.6** | **[−8.6, −1.3]** |

Both intervals exclude zero, in opposite directions, and they cancel to the
stratum's +0.00. On questions needing two gold sessions the decomposer asked
two or more follow-ups only **30%** of the time (mean 1.08 steps).

A one-step note is a confident **partial** answer arriving through the
evidence channel: on a two-fact question it anchors the reader on one fact.
That is `premise_analysis`'s failure shape — content in the evidence channel
arguing for a conclusion — which is why the effect is a measured harm rather
than merely neutral.

**This split is conditioned on the mechanism's own output and is therefore
descriptive, not causal.** The two groups differ at baseline (67.7% vs
52.0%), so the rows the model chose to decompose were already the easier
ones. It does **not** license "+10.8 once the decomposer is fixed". What it
licenses is refusing to emit the note that measurably costs 4.6 points.

### Acted on

`MIN_STEPS_EMITTED = 2`: no `[notes]` item unless at least two follow-ups
resolved. The measured arm permitted one-step notes and `runs/m39_selfask`
was produced by that version; **this revision is itself unmeasured**, and it
is off by default either way.

### The falsification branch, resolved

The pre-registration named the fork: if `asked_steps` were healthy and
accuracy still did not move, then handing the reader resolved sub-answers is
not enough. Steps were healthy on 57% of rows and the headline did not move —
but the sub-split locates the failure one step earlier than that. It is not
that the reader cannot use resolved sub-answers; it is that **a 9B model asked
to decompose mostly does not**, and when it does the notes appear to help.

That is the same shape as M38's selector result, where the same model ignored
a parsimony instruction and 500/500 rows came back byte-identical. Two
milestones, two prompts, one finding: **this reader does not change its
behaviour on instruction.** Anything that depends on it deciding to do
something differently should be assumed inert until measured at the wire.
