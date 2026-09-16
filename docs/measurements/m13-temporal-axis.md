# M13 — the temporal axis: chronological evidence and a question date

Temporal was the axis where retrieval is *not* the bottleneck. On LoCoMo the gold turns for category
2 come back 96.3% of the time (`m12-graph-route.md` §8, `hybrid_rerank` per-category block) while
end-to-end token F1 is 0.2825 (n=321, `m9-locomo.md`); LongMemEval_S `temporal-reasoning` scores
0.2218 (n=133, `m9-longmemeval-s.md`). Two read-layer defects were fixed and measured
independently: the LoCoMo reader got no reference date, and composed evidence was never in time
order.

**Verdict: both defaults stay `false`.** `ComposeConfig::chronological` and `bench --question-date`
ship as switches; neither earns its default. The decision rule was fixed before the runs and the
target stratum fails for both mechanisms — LoCoMo category 2 moves +0.2 points under
`--question-date` (95% CI [−1.8, +2.2]) and −0.3 under `--chronological` ([−1.9, +1.3]), and
LongMemEval_S `temporal-reasoning` moves −1.1 under `--chronological` ([−5.2, +3.1]).

Both mechanisms have real reach — 27.2% and 16.9% of LoCoMo answers change — and both produce one
off-target effect large enough to name: `--question-date` raises LoCoMo abstention accuracy by
**+3.1 points, CI [+0.7, +5.6], p = 0.020**, and `--chronological` raises LongMemEval_S
`knowledge-update` by **+8.0 points, CI [−1.7, +17.6]**. Neither is the stratum the rule was written
about, and neither flips a default.

## What was built

| piece | where |
|---|---|
| ascending-`t_valid` ordering of the emitted evidence, after selection | `ComposeConfig::chronological`, `pipeline::compose::compose` |
| a `<today>` block in the LoCoMo reader prompt, from the conversation's last session date | `bench::bench_locomo`, reusing `build::parse_locomo_time` |
| both switches on the CLI, both recorded in the run artifact | `myelin-eval bench --chronological --question-date`, `BenchRun::{chronological, question_date}` |
| per-switch run directories, so no arm can clobber another | `runs/<slug>_<mode>[_graph][_chrono][_qdate]` |

Nothing in `ablate`, deliberately: it scores *retrieval* by gold-turn coverage and both switches
leave the retrieved record set identical, so an ablation arm for either would be a copy of its
comparator.

`--question-date` on `--corpus longmemeval-s` is rejected rather than silently ignored — that prompt
already carries `item.question_date`.

Retrieval behaviour is unchanged. `chronological` re-orders *after* selection, dedup and the token
budget, so `EvidenceSet::tokens` and the item set are identical between arms; `question_date` never
touches retrieval at all. No re-ingest, no backfill.

`myelin-mcp` is untouched: `Backend` builds `RetrieveConfig::default()`, so the MCP surface behaves
exactly as before.

## The reference date

LoCoMo has no per-question date. Inspecting `data/locomo10.json` directly, the QA keys across the
whole corpus are exactly `adversarial_answer`, `answer`, `category`, `evidence`, `question` — so the
plan's contingency ("prefer a per-question date if one exists") does not apply. The conversation's
last session date is used, and all ten conversations yield one:

| conversation | sessions with a date | `<today>` |
|---|---|---|
| conv-26 | 35 | 2024-01-04 |
| conv-30 | 19 | 2023-07-23 |
| conv-41 | 32 | 2023-08-16 |
| conv-42 | 29 | 2022-11-11 |
| conv-43 | 29 | 2024-01-12 |
| conv-44 | 28 | 2023-11-22 |
| conv-47 | 31 | 2022-11-07 |
| conv-48 | 30 | 2023-09-20 |
| conv-49 | 25 | 2024-01-11 |
| conv-50 | 30 | 2023-11-17 |

Every session timestamp in the corpus parses, so the "no parseable date → unmodified two-block
prompt" path never fires on LoCoMo-10. It is still there, because an invented date is worse than
none.

## Both switches, observed at the prompt

A prompt change that silently fails to apply is the obvious way for this measurement to be
worthless, so it was checked at the wire rather than inferred. The reader was pointed at a logging
proxy (`MYELIN_LLM__URL`) for three one-question runs over `conv-26`:

- **base**: `<memories>…</memories><question>…</question>`, evidence dates in bookend order
  `2023-07-20, 2023-05-08, 2023-07-20, 2023-07-20, 2023-07-17, 2023-05-08`.
- **`--question-date`**: identical, plus `<today>\n2024-01-04\n</today>` between the two blocks —
  which matches conv-26's last session date in the table above.
- **`--chronological`**: same six records, dates
  `2023-05-08, 2023-05-08, 2023-07-17, 2023-07-20, 2023-07-20, 2023-07-20`, verified ascending and
  verified an identical multiset to the base run's six.

The unit test `chronological_emits_oldest_first_over_the_same_selection` pins the same property
hermetically: four items whose `t_valid` order reverses their score order compose to oldest-first
under the switch and to `[1st, 3rd, 4th, 2nd]` without it, with identical `items.len()` and
`tokens`. Deleting the `sort_by_key` branch fails the first assertion; changing selection instead of
ordering fails the second.

Retrieval is deterministic across repeated runs (three consecutive one-question runs produced
byte-identical evidence), with one caveat worth recording: the *first* request after a cold
reranker start broke a score tie differently from every subsequent one. A two-question warm-up run
therefore preceded the first arm, and the reranker stayed up across all four, so no cold-start
tie-break lands inside a measured arm.

## The four runs

`--mode recall --k 6`, deterministic scorer, reader `qwen3.5-9b`, reranker `bge-reranker-v2-m3` on
`big`. Baselines are the M9 artifacts on disk.

| | LoCoMo base | `--chronological` | `--question-date` | both |
|---|---|---|---|---|
| token F1, answerable | 0.5307 | 0.5290 | 0.5303 | 0.5293 |
| exact match | 0.2870 | 0.2877 | 0.2818 | 0.2825 |
| abstention accuracy | 0.6996 | 0.7063 | **0.7309** | 0.7242 |
| `memory_query` p50 | 0.25 s | 0.24 s | 0.22 s | 0.22 s |
| cat 1 single-hop | 0.4252 | 0.4219 | 0.4310 | 0.4166 |
| **cat 2 temporal** | 0.2825 | 0.2795 | **0.2849** | 0.2864 |
| cat 3 multi-hop | 0.2007 | 0.1674 | 0.1941 | 0.1570 |
| cat 4 open-domain | 0.6985 | 0.7014 | 0.6957 | 0.7022 |
| cat 5 adversarial | 0.6996 | 0.7063 | **0.7309** | 0.7242 |

| | LME_S base | `--chronological` |
|---|---|---|
| token F1, answerable | 0.4369 | **0.4473** |
| exact match | 0.3489 | 0.3638 |
| abstention accuracy | 0.9667 | 1.0000 |
| `memory_query` p50 | 0.41 s | 0.50 s |
| cat 1 single-session-user | 0.9114 | 0.9351 |
| cat 2 single-session-asst | 0.7927 | 0.7981 |
| cat 3 single-session-pref | 0.0461 | 0.0500 |
| cat 4 multi-session | 0.3420 | 0.3343 |
| **cat 5 temporal-reasoning** | 0.2218 | 0.2111 |
| cat 6 knowledge-update | 0.6386 | **0.7187** |

`chronological` is free: a stable sort over at most six items, and LoCoMo p50 went 0.25 s → 0.24 s.
The LongMemEval_S p50 difference (0.41 → 0.50 s) is GPU sharing, not the switch — there is no
retrieval work in either mechanism.

Each `aggregated_metrics.json` records the flags that produced it: `{chronological, question_date}`
is `{true,false}`, `{false,true}`, `{true,true}`, `{true,false}` for the four arms in order.

## Paired bootstrap CIs, per category

20,000 paired resamples of the per-question difference (`adapters/paired_ci.py --by-category`).
`A − B` with the switch as A, so positive = the switch scores higher.

### LoCoMo `--chronological` − base, n = 1,986

| stratum | n | A | B | diff | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 1,986 | 56.9% | 56.9% | +0.0 | [−1.0, +1.0] | 0.974 |
| non-abstention | 1,540 | 52.9% | 53.1% | −0.2 | [−1.1, +0.7] | 0.693 |
| abstention | 446 | 70.6% | 70.0% | +0.7 | [−2.5, +4.0] | 0.726 |
| 1 single-hop | 282 | 42.2% | 42.5% | −0.3 | [−2.4, +1.7] | 0.744 |
| **2 temporal** | 321 | 28.0% | 28.3% | **−0.3** | **[−1.9, +1.3]** | 0.726 |
| 3 multi-hop | 96 | 16.7% | 20.1% | −3.3 | [−8.2, +1.2] | 0.153 |
| 4 open-domain | 841 | 70.1% | 69.9% | +0.3 | [−1.0, +1.6] | 0.655 |
| 5 adversarial | 446 | 70.6% | 70.0% | +0.7 | [−2.5, +4.0] | 0.726 |

### LoCoMo `--question-date` − base, n = 1,986

| stratum | n | A | B | diff | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 1,986 | 57.5% | 56.9% | +0.7 | [−0.1, +1.5] | 0.097 |
| non-abstention | 1,540 | 53.0% | 53.1% | −0.0 | [−0.8, +0.7] | 0.911 |
| abstention | 446 | 73.1% | 70.0% | **+3.1** | **[+0.7, +5.6]** | 0.020 * |
| 1 single-hop | 282 | 43.1% | 42.5% | +0.6 | [−0.3, +1.6] | 0.228 |
| **2 temporal** | 321 | 28.5% | 28.3% | **+0.2** | **[−1.8, +2.2]** | 0.803 |
| 3 multi-hop | 96 | 19.4% | 20.1% | −0.7 | [−3.0, +0.8] | 0.734 |
| 4 open-domain | 841 | 69.6% | 69.9% | −0.3 | [−1.3, +0.7] | 0.571 |
| 5 adversarial | 446 | 73.1% | 70.0% | **+3.1** | **[+0.7, +5.6]** | 0.020 * |

### LoCoMo both switches − base, n = 1,986

| stratum | n | A | B | diff | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 1,986 | 57.3% | 56.9% | +0.4 | [−0.7, +1.5] | 0.436 |
| non-abstention | 1,540 | 52.9% | 53.1% | −0.1 | [−1.2, +0.9] | 0.765 |
| abstention | 446 | 72.4% | 70.0% | +2.5 | [−1.1, +5.8] | 0.185 |
| 1 single-hop | 282 | 41.7% | 42.5% | −0.9 | [−2.8, +1.0] | 0.375 |
| **2 temporal** | 321 | 28.6% | 28.3% | +0.4 | [−2.1, +2.8] | 0.749 |
| 3 multi-hop | 96 | 15.7% | 20.1% | −4.4 | [−9.6, +0.4] | 0.074 |
| 4 open-domain | 841 | 70.2% | 69.9% | +0.4 | [−1.0, +1.7] | 0.597 |
| 5 adversarial | 446 | 72.4% | 70.0% | +2.5 | [−1.1, +5.8] | 0.185 |

### LongMemEval_S `--chronological` − base, n = 500

| stratum | n | A | B | diff | 95% CI | p |
|---|---|---|---|---|---|---|
| overall | 500 | 48.1% | 46.9% | +1.2 | [−1.2, +3.5] | 0.325 |
| non-abstention | 470 | 44.7% | 43.7% | +1.0 | [−1.4, +3.5] | 0.399 |
| abstention | 30 | 100.0% | 96.7% | +3.3 | [+0.0, +10.0] | 0.728 |
| 1 single-session-user | 70 | 93.5% | 91.1% | +2.4 | [+0.0, +5.7] | 0.092 |
| 2 single-session-asst | 56 | 79.8% | 79.3% | +0.5 | [−1.9, +3.5] | 0.927 |
| 3 single-session-pref | 30 | 5.0% | 4.6% | +0.4 | [−0.9, +1.8] | 0.581 |
| 4 multi-session | 133 | 33.4% | 34.2% | −0.8 | [−5.4, +3.8] | 0.739 |
| **5 temporal-reasoning** | 133 | 21.1% | 22.2% | **−1.1** | **[−5.2, +3.1]** | 0.611 |
| 6 knowledge-update | 78 | 71.9% | 63.9% | **+8.0** | [−1.7, +17.6] | 0.104 |

\* = 95% CI excludes zero. Category codings differ between the corpora — LoCoMo 2 is temporal,
LongMemEval_S 2 is single-session-assistant — so only ever compare like against like.

Across the four arms these tables report **30 distinct strata**. Exactly one excludes zero. At
α = 0.05 that is 1.5 expected by chance, so the single hit carries no more weight than its own
interval, which is why the rule was fixed in advance and names one stratum per switch.

## The interaction arm

Read against both single-switch arms rather than only the baseline:

| comparison | overall | cat 2 | cat 3 | cat 5 |
|---|---|---|---|---|
| both − base | +0.4 [−0.7, +1.5] | +0.4 [−2.1, +2.8] | −4.4 [−9.6, +0.4] | +2.5 [−1.1, +5.8] |
| both − `--chronological` | +0.4 [−0.4, +1.2] | +0.7 [−1.5, +2.8] | −1.0 [−4.8, +2.4] | +1.8 [−0.7, +4.5] |
| both − `--question-date` | −0.2 [−1.3, +0.8] | +0.1 [−2.0, +2.3] | −3.7 [−8.6, +0.7] | −0.7 [−4.0, +2.7] |

The two switches are additive within noise and there is no interaction to exploit: combining them
lands between the two single-switch arms on every stratum, and the abstention gain that
`--question-date` buys alone (+3.1, CI excluding zero) becomes +2.5 with a CI that includes zero
once `--chronological` is added. Nothing here would change a default either.

## The decision rule, applied

Fixed in advance: each switch is decided independently, and a default flips to `true` **iff** (a)
its target stratum improves with a paired 95% CI excluding zero on the positive side — LoCoMo
category 2 for either switch, plus LongMemEval_S category 5 for `chronological`, the only switch
measured on both corpora — **and** (b) no `overall` paired CI for that switch has a lower bound
below −1.0 points.

| switch | condition | required | measured | verdict |
|---|---|---|---|---|
| `chronological` | LoCoMo cat 2 CI > 0 | lower bound > 0 | −0.3, [**−1.9**, +1.3] | fail |
| `chronological` | LME_S cat 5 CI > 0 | lower bound > 0 | −1.1, [**−5.2**, +3.1] | fail |
| `chronological` | LoCoMo overall lower bound ≥ −1.0 | ≥ −1.0 | −1.0 | pass |
| `chronological` | LME_S overall lower bound ≥ −1.0 | ≥ −1.0 | **−1.2** | fail |
| `question_date` | LoCoMo cat 2 CI > 0 | lower bound > 0 | +0.2, [**−1.8**, +2.2] | fail |
| `question_date` | LoCoMo overall lower bound ≥ −1.0 | ≥ −1.0 | −0.1 | pass |

**`ComposeConfig::chronological` stays `false` and `question_date` stays off by default** — the same
discipline `tau_abstain`, `label_untrusted` and `RetrieveConfig::graph` already got. The switches
stay, the defaults do not move, and the intervals are the record.

One wrinkle in the rule worth recording rather than relitigating: condition (b) was written as a
harm guard, and on LongMemEval_S it trips on a *positive* point estimate (+1.2) whose interval is
merely wide ([−1.2, +3.5]). It costs `chronological` nothing here, because condition (a) already
fails on both corpora. A future rule of this shape should guard on the *upper* half of a harm
interval, not on any lower bound below a threshold, or it will veto wide-but-favourable results.

The bound this puts on the two mechanisms as implemented: on LoCoMo temporal, any true effect is
inside [−1.9, +1.3] for `chronological` and [−1.8, +2.2] for `question_date` at n = 321 — CI
half-widths of ±1.6 and ±2.0 points, 2.6× and 2.1× tighter than the ±4.2 M12 had on multi-hop at
n = 96. These are bounded nulls, not underpowered ones. On LongMemEval_S temporal-reasoning the
effect is inside [−5.2, +3.1] at n = 133.

## Why they fail, measured rather than guessed

Both switches reach a large share of the corpus, so neither verdict is "the flag did nothing".

| switch | corpus | answers changed | better | worse | same score | net |
|---|---|---|---|---|---|---|
| `chronological` | LoCoMo | 540 / 1,986 (27.2%) | 173 | 182 | 185 | +0.0 pts |
| `question_date` | LoCoMo | 335 / 1,986 (16.9%) | 103 | 95 | 137 | +0.7 pts |
| `chronological` | LME_S | 128 / 500 (25.6%) | 35 | 29 | 64 | +1.2 pts |

**`chronological` is a coin flip: 173 better against 182 worse.** That is the signature of a
perturbation that changes the prompt without adding information. The lost-in-the-middle result the
`bookend` default is built on (`10.48550/arxiv.2307.03172` Table 6) is measured at **20 documents**;
`compose` emits at most **six**, and the one LoCoMo prompt captured at the wire measured 4,457
chars / 1,114 `approx_tokens` against the bench evidence budget of 4,096. At that size the position
of the gold item is not the binding constraint, so trading `bookend`'s relevance interleave for
time order buys nothing and costs nothing. This is also the strongest defence of the `bookend`
default that exists: it was tested against its most plausible rival on 2,486 questions and neither
order wins.

The one place time order pays is LongMemEval_S **knowledge-update** (+8.0 points, 13 better / 6
worse of 24 changed). Those questions ask for the *current* value of a fact that changed over time,
and oldest-first puts the newest state last — adjacent to the question. That is `bookend`'s tail
slot, assigned by recency instead of relevance. It does not generalise to `temporal-reasoning`
(−1.1), which asks about *relations between* times rather than the latest value.

**`question_date` reaches the reader and changes the form of its answers, and the metric cannot see
it.** On LoCoMo category 2 the reader's answers shift from relative to absolute exactly as intended:

| run | relative-time answers | absolute-date answers | cat 2 F1 |
|---|---|---|---|
| base | 68 (21.2%) | 125 (38.9%) | 0.2825 |
| `--question-date` | 65 (20.2%) | 142 (44.2%) | 0.2849 |
| `--chronological` | 72 (22.4%) | 125 (38.9%) | 0.2795 |
| both | 66 (20.6%) | 142 (44.2%) | 0.2864 |

17 more answers name an absolute date, and 0.2 points of F1 come of it. The reason is the gold
strings. Shapes of the 321 category-2 gold answers:

| gold shape | n | share | example |
|---|---|---|---|
| anchored-relative | 77 | 24.0% | `The sunday before 25 May 2023` |
| full date | 73 | 22.7% | `7 May 2023` |
| month (+ year) | 53 | 16.5% | `June 2023` |
| bare year | 25 | 7.8% | `2022` |
| relative only | 8 | 2.5% | `10 years ago` |
| duration | 3 | 0.9% | `4 years` |
| other | 82 | 25.5% | `Tokyo`, `first week of April 2022`, `three months` |

Only 22.7% of the stratum is a plain absolute date that an ISO answer can score against. A quarter
of it is phrased as a relative expression *anchored* to an absolute date, against which even a
semantically right answer caps out low on token overlap — observed directly:

```
Q  When did Caroline go to a pride parade during the summer?
   gold   The week before 3 July 2023
   base   Last Friday          → 0.00
   qdate  2023-07-15           → 0.25
```

```
Q  When did Melanie run a charity race?
   gold   The sunday before 25 May 2023
   base   Last Saturday               → 0.00
   qdate  Last Saturday, May 20, 2023 → 0.40
```

Of the 68 baseline answers containing a relative expression, 5 became absolute dates under the
switch; the mean F1 change on those 5 was +0.043. 38.6% of category 2 scores exactly 0.00 at
baseline, and 49 of those 124 zeros are the reader declining outright against a gold answer as
simple as `2022` — an abstention problem, which `m5-reference-baselines.md` already named this
reader's dominant failure mode, not a reference-date problem.

**LoCoMo category 2 is also not uniformly a date-answering stratum.** Splitting it by whether the
*gold answer* names a time at all (post-hoc, so it decides nothing):

| sub-stratum | n | base F1 | `--question-date` | `--chronological` |
|---|---|---|---|---|
| gold names a time | 278 | 0.2664 | +1.0 [−1.1, +3.0] | +0.0 [−1.7, +1.7] |
| gold names no time | 43 | 0.3868 | −4.7 [−11.6, +0.0] | −2.3 [−7.0, +0.0] |

The 43 items with no time in the answer are *"Who did Maria have dinner with on May 3, 2023?"* →
`her mother` — the date is in the question, not the answer. Supplying a second, different date
(`<today>` = the conversation's end) measurably hurts them, and that loss cancels most of the gain
on the 278 that do want a date. Any future version of this mechanism should be conditioned on the
question rather than applied to every one of them. Both switches already allow that:
`ComposeConfig` is query-time (R4), so `chronological` can be set per `recall` call, and the
`<today>` block is built by whoever builds the prompt.

**The one asymmetric win is abstention.** `--question-date` scores 24 better / 10 worse of 49
changed answers on LoCoMo's 446 adversarial items: +3.1 points, CI [+0.7, +5.6]. A reader that can
see the current date can tell that the memories do not cover the period the question asks about, and
declines. That is a real effect of the mechanism and it is off-target — the rule names category 2,
and a rule that gets rewritten after the numbers arrive is not a rule.

## What this says about the temporal axis

Retrieval was never the bottleneck here, and after this milestone neither is the read layer's
plumbing: the date is present, the order is controllable, and both were verified at the wire. What
remains is two things the measurement now separates cleanly.

- **A reader-capability gap.** Converting `[2023-05-08] … yesterday` plus `<today>` into `7 May
  2023` is date arithmetic over retrieved text, and a 9B reader gets it wrong often enough that the
  supplied date does not pay. 49 of 124 category-2 zeros are outright abstentions.
- **A scorer ceiling.** Under this repository's deterministic token-F1 scorer, at most 22.7% of
  LoCoMo category 2 can reward a correct absolute date. The remaining 77% needs either date-aware
  answer matching or an LLM judge, and `bench.rs` refuses a judge for pinning reasons that still
  hold. **Any future temporal work should fix the scorer before it fixes the reader**, or it will
  keep measuring string overlap against `The sunday before 25 May 2023`.

Three things this milestone establishes independent of the verdict:

- `bookend` survives a direct test against time order on 2,486 questions: 208 better, 211 worse,
  net zero. The default is now measured against its rival rather than only against its citation.
- Emitted-evidence order is a free query-time parameter (stable sort over ≤ 6 items, p50 unchanged),
  so the knowledge-update finding is available to any caller that wants it.
- A `<today>` block is worth +3.1 points of abstention accuracy on LoCoMo, which is the one number
  from this milestone that a future abstention-focused milestone should start from.

## Reproduce

```bash
cargo test --workspace                       # 127 tests
cargo clippy --workspace --all-targets       # warning-free
export MYELIN_QDRANT__URL=http://192.168.1.110:6334
myelin-eval bench --corpus locomo        --mode recall --k 6 --chronological
myelin-eval bench --corpus locomo        --mode recall --k 6 --question-date
myelin-eval bench --corpus locomo        --mode recall --k 6 --chronological --question-date
myelin-eval bench --corpus longmemeval-s --mode recall --k 6 --chronological
python3 crates/myelin-eval/adapters/paired_ci.py runs/locomo_recall_chrono       runs/locomo_recall --by-category
python3 crates/myelin-eval/adapters/paired_ci.py runs/locomo_recall_qdate        runs/locomo_recall --by-category
python3 crates/myelin-eval/adapters/paired_ci.py runs/locomo_recall_chrono_qdate runs/locomo_recall --by-category
python3 crates/myelin-eval/adapters/paired_ci.py runs/lme_s_recall_chrono        runs/lme_s_recall  --by-category
```

Reader and reranker on `big` per `ops/big/README.md`; ports 5810/5813 are firewalled and need the
SSH tunnel, and LongMemEval_S needs `MYELIN_READER_SLOTS=2 MYELIN_READER_CTX=65536` on the remote
side of the `ssh` command. Warm the reranker with a throwaway `--limit 2` run after serving it and
before the first measured arm; the four arms then run sequentially against one warm service, never
concurrently — concurrent runs share the reranker and make `query_p50_seconds` meaningless.
