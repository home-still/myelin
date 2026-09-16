# M7 — marginal value of an investigate step

60-question `web`/`small` subset, `k = 25`, 10k evidence budget, identical store. Only `max_steps`
varies. `recall` is the same subset through the non-agentic path.

| mode | overall | non-abstention (n=43) | abstention (n=17) | `memory_query` avg |
|---|---|---|---|---|
| `recall` | 36.7% | 46.5% | 11.8% | 1.83 s |
| `investigate` `max_steps=1` | 33.3% | 37.2% | 23.5% | 2.66 s |
| `investigate` `max_steps=2` | **43.3%** | 46.5% | **35.3%** | 11.54 s |
| `investigate` `max_steps=3` | 41.7% | 48.8% | 23.5% | 24.87 s |
| `investigate` `max_steps=4` | 38.3% | 48.8% | 11.8% | 28.91 s |

## More steps make the system worse

The curve is not monotone and does not saturate — it **peaks at two steps and then declines**.
Best operating point found so far: **43.3% at 11.54 s**, which is both more accurate *and* 13.3 s
faster than the `max_steps=3` point measured earlier.

The two columns move in opposite directions:

- **Non-abstention rises** with steps — 37.2 → 46.5 → 48.8 → 48.8 — and saturates at three.
  Extra search genuinely helps questions that have an answer.
- **Abstention collapses** past two steps — 35.3 → 23.5 → 11.8. By four steps it is back to the
  `recall` baseline, i.e. the entire abstention benefit of the agentic loop has been spent.

The mechanism is the obvious one: a loop instructed to keep searching until it is satisfied will,
given enough steps, always surface *something*, and that something reads to the reader as evidence.
Persistence manufactures false confidence. The loop talks itself out of the correct "I don't know".

Combined with `m5-reference-baselines.md` — where the reader fabricated answers on 97.2% of
abstention questions given *no evidence at all* — the picture is consistent: this reader treats an
empty or weak evidence set as licence to guess, and the agentic loop's persistence makes the set
look less empty without making it more informative.

## `max_steps=4` is on the wrong side of the latency cliff

At 28.91 s it crosses the 26.9 s frontier breakpoint (see `m6-g1-breakeven.md`), so its bar rises
from 51.0 to **58.6**. It is the slowest, least accurate, and hardest-graded point measured. It
should never be shipped.

LAFS gain against the released `small` frontier, all still zero:

| point | accuracy | latency | bar | LAFS gain |
|---|---|---|---|---|
| `max_steps=2` | 43.3% | 11.54 s | 51.0 | `+0.0000` |
| `max_steps=3` | 41.7% | 24.87 s | 51.0 | `+0.0000` |
| `max_steps=4` | 38.3% | 28.91 s | 58.6 | `+0.0000` |

## Sample size

17 abstention questions, so that column moves in 5.9-point increments: 35.3% is 6/17, 23.5% is
4/17, 11.8% is 2/17. Individual differences are two or three questions and are **not** significant
on their own. What carries weight is the monotone decay across three consecutive settings combined
with a mechanism that predicts exactly that decay. Treat the *shape* as real and the *levels* as
provisional until run on the full 240.

## Consequence

`max_steps` default should be **2**, not 4. The remaining gap to the 51.0 bar is **7.7 points**
with **15.4 s** of latency headroom before the cliff — a better position than any previously
measured, and it argues for spending that headroom on something other than more steps.

---

## Paired bootstrap CIs — what actually survives

The caveat above said to treat the shape as real and the levels as provisional. Running
`adapters/paired_ci.py` (20,000 paired resamples over per-question scores; two runs on the same
question set are paired, and resampling the per-question *difference* removes the shared-difficulty
nuisance variance that independent CIs would leave in) settles which parts hold.

**`max_steps=2` vs `max_steps=4`**

| stratum | n | A | B | A−B | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 60 | 43.3% | 38.3% | +5.0 | [−5.0, +16.7] | 0.442 |
| non-abstention | 43 | 46.5% | 48.8% | −2.3 | [−14.0, +9.3] | 0.846 |
| abstention | 17 | 35.3% | 11.8% | **+23.5** | **[+5.9, +47.1]** | **0.021** |

**`max_steps=2` vs `max_steps=3`**

| stratum | n | A | B | A−B | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 60 | 43.3% | 41.7% | +1.7 | [−8.3, +11.7] | 0.866 |
| abstention | 17 | 35.3% | 23.5% | +11.8 | [−11.8, +35.3] | 0.430 |

**`max_steps=2` vs `max_steps=1`**

| stratum | n | A | B | A−B | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 60 | 43.3% | 33.3% | +10.0 | [−5.0, +23.3] | 0.213 |

### Correction to the section above

Two claims must be separated, and the earlier text ran them together.

1. **The abstention collapse is real.** 35.3% → 11.8% from two steps to four is +23.5 points with a
   95% CI of [+5.9, +47.1], p = 0.021. The mechanism — persistence manufacturing false confidence —
   is supported.
2. **"Two is the peak" is *not* established.** `max_steps=2` vs `max_steps=3` on overall accuracy is
   +1.7 points with a CI spanning [−8.3, +11.7]. The ordering among 1, 2, and 3 is unresolved at
   n = 60. Even `max_steps=2` vs `max_steps=4` on *overall* accuracy (+5.0, p = 0.442) fails to
   reach significance, because the abstention gain on 17 questions is partly cancelled by a
   non-significant non-abstention loss on 43.

So the default of 2 does **not** rest on being measurably the most accurate. It rests on:

- **`max_steps=4` is excluded on two independent grounds** — significantly worse abstention
  (p = 0.021), and 28.91 s crosses the 26.9 s frontier breakpoint, raising our own bar from 51.0 to
  58.6. The latency argument is arithmetic, not statistics, and does not depend on n.
- **2 over 3 is a latency tie-break**: 11.54 s vs 24.87 s for a difference of +1.7 ± 10 points, i.e.
  less than half the latency at no measurable accuracy cost, leaving 15.4 s of headroom instead of
  2.0 s before the cliff.

That is a sound basis for the default and an unsound basis for the claim that the curve has a peak
at two. Confirming a peak needs the full 240-question set; 17 abstention questions cannot resolve
5.9-point increments.

---

## Settled at full set by M16

The caveat above — *"treat the shape as real and the levels as provisional until run on the full
240"* — is now measured. `docs/measurements/m16-evidence-sufficiency.md` ran `max_steps=2` on the
full set of **both** domains against the same `recall` k=25 baseline, paired:

|domain|stratum|n|`investigate` 2|`recall` k=25|Δ|95% CI|p|
|---|---|---|---|---|---|---|---|
|web|overall|240|45.0%|36.7%|**+8.3**|**[+2.9, +13.8]**|**0.0026**|
|web|non-abstention|168|51.2%|42.9%|**+8.3**|**[+2.4, +14.9]**|**0.0104**|
|web|abstention|72|30.6%|22.2%|+8.3|[−2.8, +19.4]|0.1625|
|enterprise|overall|211|34.1%|34.6%|−0.5|[−6.6, +5.7]|0.9423|
|enterprise|non-abstention|155|41.9%|39.4%|+2.6|[−5.2, +10.3]|0.5621|
|enterprise|abstention|56|12.5%|21.4%|**−8.9**|**[−17.9, −1.8]**|**0.0095**|

**The web level holds** — 43.3% on the 60-question subset becomes **45.0%** at n=240, and the gain
over `recall` is significant on the overall and answerable strata, which n=60 could not establish.

**It does not transfer.** On enterprise the same configuration is −0.5 points overall with a
**significant −8.9-point abstention loss**: the persistence-manufactures-false-confidence mechanism
diagnosed here at `max_steps` 3–4 on web is already in force at `max_steps=2` on enterprise. So the
default of 2 is defensible **for web** and is not a global default; the domain is a free parameter
this curve never varied.

`memory_query` average at full set: **11.06 s** (web, p50 11.01, p95 16.39) and **14.69 s**
(enterprise, p50 12.40, p95 32.60) — both inside the 26.9 s breakpoint, so the 51.0 bar still
applies to both.
