# M9 — paired confidence intervals: `investigate` buys nothing measurable

`PLAN.md` G2 requires paired CIs excluding zero. They do not exclude zero, and the interesting
part is how precisely they fail to.

Method: `adapters/paired_ci.py`, 20,000 paired resamples of the per-question score difference. Two
runs over the same questions are paired, and a question's difficulty is shared nuisance variance;
comparing independent marginal CIs discards the pairing and is needlessly conservative.

## LoCoMo — full corpus, n = 1,986

`investigate` (`max_steps = 2`) against `recall`, same store, same `k = 6`, deterministic scorer.

| stratum | n | investigate | recall | diff | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 1,986 | 56.7% | 56.9% | **−0.1** | **[−0.7, +0.4]** | 0.659 |
| non-abstention | 1,540 | 53.0% | 53.1% | −0.1 | [−0.6, +0.4] | 0.714 |
| abstention | 446 | 69.7% | 70.0% | −0.2 | [−1.8, +1.3] | 0.881 |

Latency: **1.81 s against 0.28 s** — 6.5×.

This is a *precise* null, not an underpowered one. The interval rules out any effect larger than
about ±0.7 points. Six and a half times the latency buys nothing measurable.

### The hypothesis it was built to test, tested directly

Both corpora say multi-hop is the weak axis (`m9-longmemeval-s.md`), and `investigate` exists to
fix it by searching again with a refined query. It does not:

| question type | n | diff | 95% CI | p |
|---|---|---|---|---|
| multi-hop (cat 3) | 96 | +1.07 | [−2.30, +4.64] | 0.523 |
| temporal (cat 2) | 321 | −0.94 | [−2.40, +0.38] | 0.172 |
| single-hop (cat 1) | 282 | +0.43 | [−0.78, +1.78] | 0.537 |

## LongMemEval-V2 — the comparison I had not run

`m7-step-value-curve.md` reported `investigate max_steps=2` at 43.3% against `recall` at 36.7% and
I treated +5.0 as the best operating point. It was never significance-tested against `recall`:

| stratum | n | investigate | recall | diff | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 60 | 43.3% | 38.3% | +5.0 | [−8.3, +18.3] | 0.547 |
| non-abstention | 43 | 46.5% | 44.2% | +2.3 | [−11.6, +16.3] | 0.887 |
| abstention | 17 | 35.3% | 23.5% | +11.8 | [−17.6, +41.2] | 0.532 |

Not significant, and the interval is wide enough at n = 60 to contain both a large gain and a
large loss. The LoCoMo run is the one that settles it, because n = 1,986 makes the interval tight.

## Conclusion

**`investigate` has no measured accuracy benefit over `recall` on either corpus.** On LoCoMo the
effect is bounded within ±0.7 points at 6.5× the latency; on LongMemEval-V2 the point estimate is
positive but indistinguishable from zero.

This supersedes the reading in `m7-step-value-curve.md`. That document established two things that
still hold — abstention accuracy collapses as steps increase (p = 0.021), and `max_steps = 4`
crosses the LAFS latency cliff — but its framing of `max_steps = 2` as "the best operating point
found" implied a gain over `recall` that the data does not support. The step curve compares
`investigate` settings against each other; it never compared the winner against not doing it at
all.

What remains true about the loop: it is the only mechanism we have that can issue a second query,
and the G2 per-type tables show multi-hop and temporal are where the accuracy is missing. The
measurement says this implementation of that idea does not convert the opportunity. One reflect
step refines the query; refining the query does not find the second fact.

Recorded rather than tuned: M6 is at its cap and `investigate`'s parameters have had their sweep.
