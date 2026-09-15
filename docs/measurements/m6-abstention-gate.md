# M6 — the insufficiency gate, measured and rejected

Authorised as a deliberate exception to M6's three-attempt cap, on the strength of three
converging measurements:

- `m5-reference-baselines.md` — the reader fabricates an answer on **97.2%** of unanswerable
  questions when given *no evidence at all*, so it will not abstain unprompted.
- `m7-step-value-curve.md` — abstention accuracy collapses **35.3% → 11.8%** as steps go 2 → 4
  (+23.5, 95% CI [+5.9, +47.1], p = 0.021), i.e. loop persistence manufactures false confidence.
- Abstention is the entire G1 gap.

`Investigator` already judges sufficiency with a model on every step and records `stopped_because`,
then hands the reader the accumulated pool regardless. The change applies the gate where that
judgement was actually formed.

## Design

Not an empty set. Empty context is what `no_retrieval` measures, and that produces 2.8% abstention
— an invitation to guess, not a signal. Instead, one evidence item carrying a plain statement:

> No stored memory answers this question. The search was run and returned nothing sufficient.

Phrased as a fact about the store, never as an instruction. A read path that speaks imperatively
through its own evidence channel has exactly the shape of the injection attacks in
`m11-attack-suite.md`, and could not honestly claim that memories are data and never commands.
It rides in the evidence channel because the LongMemEval-V2 reader prompt is vendored and must not
be edited; R1 fixes that channel at `{type, value}`, which carries a sentence fine.

## Result: it failed, decisively

60-question `web`/`small` subset, `k = 25`, `max_steps = 2` — identical to the best prior operating
point, only the gate changes.

| variant | overall | non-abstention | abstention | avg |
|---|---|---|---|---|
| `max_steps=2` | **43.3%** | 46.5% | 35.3% | 11.54 s |
| `max_steps=2` + gate | **15.0%** | 18.6% | 5.9% | 13.09 s |

Paired bootstrap, 20,000 resamples:

| stratum | n | A−B | 95% CI | p |
|---|---|---|---|---|
| overall | 60 | **−28.3** | [−41.7, −15.0] | **< 0.0001** |
| non-abstention | 43 | −27.9 | [−44.2, −11.6] | 0.0009 |
| abstention | 17 | −29.4 | [−58.8, +0.0] | 0.058 |

Abstention accuracy went **down**, from 35.3% to 5.9%. That is the opposite of the intended effect,
and it is the finding.

## Why: the reader agrees with the statement and answers anyway

**50 of 60 questions (83%)** were gated — `stopped_because == "sufficient"` is rare at
`max_steps = 2`, because the loop reflects once and then stops on `step budget`. So the gate was
not a rare safety net; it was the common path. That alone is a design error: the trigger is mostly
"ran out of steps", not "a model judged this insufficient".

But the deeper result is what the reader did with it. Given the statement as its *entire* context:

> Based on standard web design patterns for forum software (like the custom Postmill/Reddit-based
> implementation described) and the specific context that **the search returned no sufficient
> memory**: 1. …

It reads the statement. It **restates it correctly**. Then it answers from pretraining.

This is stronger than the `no_retrieval` finding. There the reader had nothing and guessed; one
could argue it simply lacked a cue. Here it was handed an explicit, unambiguous cue, demonstrated
comprehension of it in its own output, and overrode it. The failure is not that the signal was too
quiet. It was heard, acknowledged, and ignored.

## What this rules out

**Abstention cannot be fixed from the memory side with this reader.** Every channel available to a
memory system is the evidence channel — the reader model and its prompt are both pinned by the
benchmark, and the vendored prompt must not be edited. We have now tried the three distinguishable
things that channel permits:

1. withhold weak evidence (`tau_abstain`) — +0.4 points, inside noise;
2. return nothing (`no_retrieval`, as the natural limit) — 2.8% abstention;
3. state insufficiency explicitly (this) — **−28.3 points**.

That exhausts the design space reachable from where a memory system sits. The remaining lever is
the reader's disposition, which the protocol pins.

Left in the tree as `InvestigateConfig::abstain_on_insufficient`, **defaulted off**, with the
killing number in its doc-comment — the same treatment as the pattern gate, provenance labelling,
and `tau_abstain`. Four defences now, each kept as a switch, each carrying the measurement that
killed it, none of them on.

## Not iterated

One authorised attempt, one measurement, one answer. M6's cap stands.
