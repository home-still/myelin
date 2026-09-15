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
