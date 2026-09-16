# M14 — the temporal scorer: date-aware deterministic scoring, and every run on disk re-read

M13 closed with a directive: *"Any future temporal work should fix the scorer before it fixes the
reader, or it will keep measuring string overlap against `The sunday before 25 May 2023`"*
(`m13-temporal-axis.md`, "What this says about the temporal axis"). This milestone does that, and
only that.

**Verdict: `Scorer::Temporal` becomes the reported default for LoCoMo.** The decision rule was fixed
before the judge ran, and both conditions pass. On the 272 judged LoCoMo category-2 items the
date-aware scorer agrees with the judge **96.7%** of the time against token F1's **84.9%** — **+11.8
points, 95% CI [+7.7, +15.8], p < 0.0001** — and Cohen's κ moves 0.6009 → 0.9139. Off the stratum it
targets the change is a near-no-op: **−0.021 points** over the 1,219 pooled category-1/3/4 items.

**The direction of the fix is the opposite of the intuition, and that is the headline.** Token F1 was
not under-crediting correct dates; it was handing out **partial credit for date-shaped near-misses**.
Sixteen category-2 answers score ≥ 0.50 under token F1 and 0.00 under the temporal scorer, and the
judge sides with the temporal scorer on fifteen of them:

```
gold 'The week before 27 June 2023'       resp '2023-06-27'      F1 0.50 -> temporal 0.00  judge 0
gold 'week before August 7, 2023.'        resp 'August 7, 2023'  F1 0.75 -> temporal 0.00  judge 0
gold 'Thursday before December 17, 2023.' resp '2023-12-17'      F1 0.50 -> temporal 0.00  judge 0
gold '19 days'                            resp '11 days'         F1 0.50 -> temporal 0.00  judge 0
```

So the LoCoMo temporal number was **inflated by 8.03 points** (0.2825 → 0.2022) and the overall
answerable figure by **1.69 points** (0.5307 → 0.5138). Twenty answers move the other way — correct
dates token F1 could not reward because the gold is coarser — and the judge sides with the temporal
scorer on nineteen of those. Across all 36 items where the two scorers disagree after binarisation,
the judge sides with the temporal scorer on **34**.

**M12's and M13's verdicts all survive the re-read.** `graph`, `chronological` and `question_date`
each stay `false` under the new scorer, by the same pre-registered rules. That is itself the useful
result: the metric was wrong and the conclusions held.

Total GPU cost: one reader-only window, **272 judge calls, 23 seconds**. No bench run was needed —
every input this milestone consumed was already on disk.

## What was built

| piece | where |
|---|---|
| the grammar: temporal expressions → closed day intervals and canonical durations | `crates/myelin-eval/src/temporal.rs` |
| the scoring formula, precision against the gold interval | `temporal::temporal_score` |
| one shared per-question decision for both corpora and the offline path | `bench::score_one` → `bench::Scores` |
| both scorers' columns on every row, forever | `ScoredQuestion::{score_token_f1, score_temporal, temporal_kind}` |
| run provenance, so an artifact names its own scorer | `BenchRun::{scorer, rescored_from}`, `bench::RunSpec` |
| offline re-read of any finished run, no GPU | `bench::rescore_run`, `myelin-eval rescore` |
| the arbiter, reader-only and cached | `crates/myelin-eval/src/judge.rs`, `myelin-eval judge` |
| agreement, κ, confusion, paired CI, disagreement dump | `crates/myelin-eval/adapters/scorer_agreement.py` |
| per-corpus scorer default | `BenchDefaults::scorer` in `main.rs` |

`--scorer` is a flag on `bench` and on `rescore`; the run-directory suffix chain is now
`runs/<slug>_<mode>[_graph][_chrono][_qdate][_temporal]`, and the scorer suffix is keyed on the
scorer's *identity* rather than on whether it is the default, so flipping LoCoMo's default cannot
silently redirect a `--scorer token-f1` run onto the M9 baseline path.

## The scoring formula, and why it is precision and not F1

Every resolved expression becomes a closed interval of whole days (or an unanchored duration in
canonical days), and

```
score = |response ∩ gold| / |response|
```

which is **precision against the gold interval**. The asymmetry is load-bearing:

| gold | response | score | why |
|---|---|---|---|
| `June 2023` | `2023-06-15` | 1/1 = **1.0** | gold's granularity is the limit of what the corpus knows; a more precise answer consistent with it is correct |
| `7 May 2023` | `May 2023` | 1/31 = **0.032** | a vaguer answer than the question admits is penalised |
| `June 2023` | `2023` | 30/365 = **0.082** | vagueness cannot game it upward |
| `The week before 27 June 2023` | `2023-06-27` | 0/1 = **0.0** | the anchor is not the offset |

A symmetric F1 over days would score the first row 0.065 and reintroduce exactly the failure M13
found. Durations compare exactly on canonical days, so `three months` == `3 months` ==
`nearly three months` == 90, and `19 days` != `11 days`.

When the gold answer names a time and the response names none, the score is 0.0 with **no fallback
to string overlap** — the fallback is the thing being fixed. Token F1 is kept only when the gold is
not temporal at all.

## The decision rule, applied

Fixed in advance: `Scorer::Temporal` becomes the reported default for LoCoMo **iff** (a) on LoCoMo
category 2, answerable, judged rows, its agreement with the judge exceeds token F1's with a paired
95% bootstrap CI on the agreement difference excluding zero on the positive side, **and** (b) on
LoCoMo categories 1, 3 and 4 pooled its mean score differs from token F1's by less than 0.5 points.

| condition | required | measured | verdict |
|---|---|---|---|
| (a) cat-2 judge agreement, temporal − token F1 | CI lower bound > 0 | +11.765, [**+7.721**, +15.809], p < 0.0001 | **pass** |
| (b) cats 1, 3, 4 pooled, n = 1,219 | \|diff\| < 0.5 pts | **−0.021** pts | **pass** |

Both pass, so the default moves. **Per corpus, not globally**: the evidence is LoCoMo category 2 and
nothing else. LongMemEval_S keeps `token-f1`, because the grammar resolves only 26 of its 470
answerable golds — 11 of 127 even in its own `temporal-reasoning` stratum — so there is no docket
there to decide on.

Both columns ship on every row regardless, because a metric change that erases the old metric makes
every historical number unreadable.

## Judge agreement, LoCoMo category 2, n = 272

Judge: `qwen3.5-9b` — the same model that produced the answers. Self-grading is a real concern and
it is the **conservative** direction here: `m9-judge-panel.md` measured this model as *harsher* than
`gemini-3.1-flash-lite` (25.6% vs 27.9% marked correct, Fleiss κ 0.8813), so a scorer that agrees
with it is not being flattered. It marks **74 of 272 (27.2%)** answers correct. Zero replies were
unparseable.

Both scorers binarised at 0.5, fixed in advance.

| scorer | n | agreement | Cohen's κ | S1J1 | S1J0 | S0J1 | S0J0 |
|---|---|---|---|---|---|---|---|
| token-f1 | 272 | 84.9% | 0.6009 | 48 | **15** | **26** | 183 |
| temporal | 272 | **96.7%** | **0.9139** | 66 | **1** | **8** | 197 |

`S1J0` is the scorer crediting an answer the judge rejects; `S0J1` is the scorer denying one the
judge accepts. Token F1 makes 41 such errors, the temporal scorer 9.

The judge is not a metric and never enters a reported accuracy number (`judge.rs` says why). Its
verdicts are cached in `runs/locomo_recall/judge_verdicts.json` and the full disagreement dump below
is the audit trail that lets a reader check the judge rather than trust it.

### The 36 items where the two scorers' binary verdicts differ

Reproduce with:

```bash
python3 crates/myelin-eval/adapters/scorer_agreement.py \
  runs/rescored/locomo_recall_temporal \
  --judge runs/locomo_recall/judge_verdicts.json --category 2
```

| question | gold | answer | token F1 | temporal | judge |
|---|---|---|---|---|---|
| `conv-26#28` | The friday before 15 July 2023 | 2023-07-15 | 0.50 | 0.00 | 0 |
| `conv-26#29` | The Friday before 15 July 2023 | 2023-07-15 | 0.50 | 0.00 | 0 |
| `conv-26#31` | The week before 27 June 2023 | 2023-06-27 | 0.50 | 0.00 | 0 |
| `conv-26#36` | The weekend before 17 July 2023 | 2023-07-17 | 0.50 | 0.00 | 0 |
| `conv-26#57` | The week before 25 August 2023 | 2023-08-25 | 0.50 | 0.00 | 0 |
| `conv-30#1` | January, 2023 | 2023-01-20 | 0.40 | 1.00 | 1 |
| `conv-30#10` | February, 2023 | 2023-02-08 | 0.40 | 1.00 | 1 |
| `conv-30#14` | April, 2023 | 2023-04-03 | 0.40 | 1.00 | 1 |
| `conv-30#21` | May, 2023 | 2023-05-27 | 0.40 | 1.00 | 1 |
| `conv-30#22` | June, 2023 | 2023-06-13 | 0.40 | 1.00 | 1 |
| `conv-30#35` | July, 2023 | 2023-07-09 | 0.40 | 1.00 | 1 |
| `conv-41#34` | The week before 16 June 2023 | 2023-06-16 | 0.50 | 0.00 | 0 |
| `conv-41#46` | The weekend before 22 July 2023 | 2023-07-22 | 0.50 | 0.00 | 0 |
| `conv-42#20` | May 2022 | 2022-05-20 | 0.40 | 1.00 | 1 |
| `conv-43#44` | November, 2023. | 2023-11-06 | 0.40 | 1.00 | 1 |
| `conv-43#58` | August 2023. | 2023-08-02 | 0.40 | 1.00 | 1 |
| `conv-43#63` | summer 2023 | Last summer. | 0.50 | 0.00 | 0 |
| `conv-44#13` | June 2023 | 2023-06-26 | 0.40 | 1.00 | 1 |
| `conv-44#38` | the weekend before October 24, 2023 | 2023-10-24 | 0.50 | 0.00 | 0 |
| `conv-44#5` | first week of May 2023 | 2023-05-03 | 0.25 | 1.00 | 1 |
| `conv-44#6` | around April 2, 2023 | 2023-04-02 | 0.29 | 1.00 | 1 |
| `conv-47#31` | 19 days | 11 days | 0.50 | 0.00 | 0 |
| `conv-47#59` | On the night of October 30 to 31, 2022 | 2022-10-31 | 0.36 | 1.00 | 1 |
| `conv-47#65` | November 7, 2022 | 2022-11-07 | 0.33 | 1.00 | 1 |
| `conv-48#38` | Friday before 13 March, 2023 | 2023-03-13 | 0.50 | 0.00 | 0 |
| `conv-48#60` | a week before 24 August,2023 | 2023-08-24 | 0.50 | 0.00 | 0 |
| `conv-49#13` | first week of June 2023 | 2023-06-06 | 0.25 | 1.00 | 1 |
| `conv-49#22` | week before August 7, 2023. | August 7, 2023 | 0.75 | 0.00 | 0 |
| **`conv-49#53`** | Saturday after 11 September, 2023. | 2023-09-11 | 0.50 | 0.00 | **1** |
| `conv-49#73` | Summer 2024 | Next summer. | 0.50 | 0.00 | 0 |
| `conv-49#74` | Thursday before December 17, 2023. | 2023-12-17 | 0.50 | 0.00 | 0 |
| `conv-50#12` | last week of May 2023 | 2023-05-31 | 0.25 | 1.00 | 1 |
| `conv-50#14` | 8 June, 2023 | 2023-06-08 | 0.33 | 1.00 | 1 |
| **`conv-50#2`** | on the weekend before March 26, 2023 | 2023-03-26 | 0.44 | 1.00 | **0** |
| `conv-50#69` | November 2023 | 2023-11-17 | 0.40 | 1.00 | 1 |
| `conv-50#8` | May 1, 2023 | 2023-05-01 | 0.33 | 1.00 | 1 |

The two bolded rows are the only ones where the judge sides against the temporal scorer, and both
are worth naming rather than averaging away:

- **`conv-49#53`** — 11 September 2023 is a Monday, so *"Saturday after"* it is the 16th. The answer
  names the anchor and the judge credited it, which is the same anchor-vs-offset confusion token F1
  makes. Here the scorer is right and the judge is wrong.
- **`conv-50#2`** — 26 March 2023 is itself a **Sunday**. `weekend before <anchor>` takes the most
  recent Saturday strictly before the anchor and its Sunday, which for a Sunday anchor yields an
  interval *containing* the anchor, so the answer scores 1.0. The judge rejected it, and the judge is
  right: this is the documented soft edge of that rule, and it is the one place the grammar is
  measurably too generous.

## Grammar coverage, measured per stratum

Coverage is the share of rows whose **gold** answer the grammar resolves — determined by re-scoring
each run with every response replaced by a non-temporal token, so an abstaining reader cannot hide a
resolvable gold. A gold the grammar does not resolve keeps token F1; that is the safe direction, and
the count is reported rather than hidden.

### LoCoMo, `runs/locomo_recall`

| stratum | n | gold resolves | token F1 | temporal | change |
|---|---|---|---|---|---|
| 1 single-hop | 282 | 4 (1.4%) | 0.4252 | 0.4242 | −0.10 |
| **2 temporal** | 321 | **269 (83.8%)** | 0.2825 | **0.2022** | **−8.03** |
| 3 multi-hop | 96 | 0 (0.0%) | 0.2007 | 0.2007 | ±0.00 |
| 4 open-domain | 841 | 11 (1.3%) | 0.6985 | 0.6986 | +0.01 |
| 5 adversarial | 446 | 0 (0.0%) | 0.6996 | 0.6996 | ±0.00 |
| cats 1, 3, 4 pooled | 1,219 | 15 | 0.5961 | 0.5959 | −0.02 |
| answerable overall | 1,540 | 284 (18.4%) | 0.5307 | 0.5138 | −1.69 |

Adversarial items are identical under both scorers by construction: `score_one` grades them by
`is_abstention` and writes the same value into both columns, so no consumer of `score_temporal` sees
a surprise there.

Within category 2 the 321 rows split: **238** scored by the grammar, **31** whose gold is temporal
but whose reader declined (0.0 under both scorers by the shared abstention rule), and **52** whose
gold is not a resolvable temporal expression. Those 52 rows are 50 distinct golds and they are the
grammar's own coverage limit:

- **not temporal at all** (42 rows, 40 distinct) — `Boston`, `Brazil`, `Chicago`, `France`,
  `Miami`, `Phuket`,
  `San Francisco`, `Seattle`, `Tokyo`, `Toronto, Canada`, `UK` ×2, `Woodhaven`,
  `Banff, Rocky Mountains`, `Whispering Falls waterfall`, `Max`, `Susie`, `Seraphim`, `her mother`,
  `Yes`, `No`, `painting` ×2, `painting classes`, `photography`, `kayaking`, `bowling`, `sushi`,
  `Valorant`, `Lord of the Rings`, `Avalanche by Neal Stephenson`, `attending a car show`,
  `camping with girlfriend`, `seeking solitude`, `work-related stress`,
  `electricity engineering project`, `a tech-for-good convention`,
  `Attending a Weight Watchers meeting`, `dyed his hair purple`,
  `finished her screenplay and printed it`, `He fell in love with a Canadian woman`,
  `They decided to live together and rented an apartment not far from McGee's bar.`
- **no year, so unresolvable** (3 rows) — `13 August`, `August`, `The week of April 3rd to 9th`
- **bare number, no unit** (2 rows) — `one`, `three`
- **open-ended or unanchored** (4 rows) — `Since 2016`, `10 years ago`, `A few years ago`,
  `a few years before 2023`
- **coarse anchor on a day-offset form** (1 row) — `few days before November 2023`: the offset needs
  an absolute day to count from, and a month is not one

One further limit, visible in the duration stratum: gold `One month` against the answer `A month.`
scores 0.0, because an indefinite article is not a quantity. That is the only duration row of 17 on
`locomo_recall` where the grammar is stricter than a reader would be.

### LongMemEval_S, `runs/lme_s_recall`

| stratum | n | gold resolves | token F1 | temporal | change |
|---|---|---|---|---|---|
| 1 single-session-user | 64 | 2 | 0.9031 | 0.9031 | ±0.00 |
| 2 single-session-assistant | 56 | 1 | 0.7927 | 0.7927 | ±0.00 |
| 3 single-session-preference | 30 | 0 | 0.0461 | 0.0461 | ±0.00 |
| 4 multi-session | 121 | 7 | 0.2850 | 0.2740 | −1.10 |
| **5 temporal-reasoning** | 127 | **11 (8.7%)** | 0.1850 | 0.1771 | −0.79 |
| 6 knowledge-update | 72 | 5 | 0.6084 | 0.6043 | −0.41 |
| answerable overall | 470 | 26 (5.5%) | 0.4369 | 0.4313 | −0.56 |

**The grammar barely reaches LongMemEval_S's temporal stratum, and that is a finding about the
corpus, not the scorer.** LoCoMo category 2 asks *when* and its golds are dates; LongMemEval
`temporal-reasoning` asks questions whose answers are counts, orderings and entities
(*"how many times…"*, *"which came first…"*), so only 11 of 127 golds are a temporal expression at
all. This is why the default flip is LoCoMo-only.

## Every run on disk, re-read

All eleven bench-schema run directories were re-scored offline at zero GPU cost. `response_raw` and
`answer_gold` are persisted, so nothing had to be re-generated.

| run | n | answerable | gold resolves | token F1 | temporal | Δ pts | target stratum Δ |
|---|---|---|---|---|---|---|---|
| `locomo_recall` | 1,986 | 1,540 | 284 | 0.5307 | 0.5138 | −1.69 | cat 2: 0.2825 → 0.2022 (−8.03) |
| `locomo_recall_chrono` | 1,986 | 1,540 | 284 | 0.5290 | 0.5116 | −1.74 | cat 2: 0.2795 → 0.1968 (−8.27) |
| `locomo_recall_qdate` | 1,986 | 1,540 | 284 | 0.5303 | 0.5123 | −1.80 | cat 2: 0.2849 → 0.1991 (−8.58) |
| `locomo_recall_chrono_qdate` | 1,986 | 1,540 | 284 | 0.5293 | 0.5143 | −1.49 | cat 2: 0.2864 → 0.2155 (−7.09) |
| `locomo_recall_graph` | 1,986 | 1,540 | 284 | 0.5315 | 0.5164 | −1.51 | cat 2: 0.2736 → 0.1991 (−7.44) |
| `locomo_recall_probe` | 200 | 153 | 37 | 0.4451 | 0.4084 | −3.67 | cat 2: 0.2576 → 0.0965 (−16.11) |
| `locomo_investigate` | 1,986 | 1,540 | 284 | 0.5298 | 0.5124 | −1.74 | cat 2: 0.2731 → 0.1894 (−8.37) |
| `lme_s_recall` | 500 | 470 | 26 | 0.4369 | 0.4313 | −0.56 | cat 5: 0.1850 → 0.1771 (−0.79) |
| `lme_s_recall_chrono` | 500 | 470 | 26 | 0.4473 | 0.4439 | −0.35 | cat 5: 0.1738 → 0.1698 (−0.39) |
| `lme_s_recall_graph` | 500 | 470 | 26 | 0.4315 | 0.4273 | −0.42 | cat 5: 0.1706 → 0.1627 (−0.79) |
| `lme_s_recall_probe` | 120 | 114 | 7 | 0.5661 | 0.5544 | −1.17 | no cat-5 rows in the probe |

Per category, token F1 → temporal. These strata include each category's adversarial rows, matching
how `adapters/paired_ci.py` slices them, so LoCoMo cat 5 and the LongMemEval_S `_abs` items appear
inside their categories; adversarial rows are identical under both scorers by construction.

| LoCoMo run | cat 1 | cat 2 | cat 3 | cat 4 | cat 5 |
|---|---|---|---|---|---|
| `locomo_recall` | 0.4252 → 0.4242 | **0.2825 → 0.2022** | 0.2007 → 0.2007 | 0.6985 → 0.6986 | 0.6996 → 0.6996 |
| `locomo_recall_chrono` | 0.4219 → 0.4209 | **0.2795 → 0.1968** | 0.1674 → 0.1674 | 0.7014 → 0.7015 | 0.7063 → 0.7063 |
| `locomo_recall_qdate` | 0.4310 → 0.4300 | **0.2849 → 0.1991** | 0.1941 → 0.1941 | 0.6957 → 0.6957 | 0.7309 → 0.7309 |
| `locomo_recall_chrono_qdate` | 0.4166 → 0.4156 | **0.2864 → 0.2155** | 0.1570 → 0.1570 | 0.7022 → 0.7023 | 0.7242 → 0.7242 |
| `locomo_recall_graph` | 0.4408 → 0.4408 | **0.2736 → 0.1991** | 0.2161 → 0.2161 | 0.6963 → 0.6972 | 0.7175 → 0.7175 |
| `locomo_investigate` | 0.4295 → 0.4296 | **0.2731 → 0.1894** | 0.2114 → 0.2114 | 0.6978 → 0.6978 | 0.6973 → 0.6973 |
| `locomo_recall_probe` | 0.3011 → 0.3011 | **0.2576 → 0.0965** | 0.1523 → 0.1523 | 0.6672 → 0.6743 | 0.6596 → 0.6596 |

| LongMemEval_S run | 1 user | 2 asst | 3 pref | 4 multi | 5 temporal | 6 update |
|---|---|---|---|---|---|---|
| `lme_s_recall` | 0.9114 → 0.9114 | 0.7927 → 0.7927 | 0.0461 → 0.0461 | 0.3420 → 0.3320 | **0.2218 → 0.2143** | 0.6386 → 0.6347 |
| `lme_s_recall_chrono` | 0.9351 → 0.9351 | 0.7981 → 0.7981 | 0.0500 → 0.0500 | 0.3343 → 0.3243 | **0.2111 → 0.2073** | 0.7187 → 0.7213 |
| `lme_s_recall_graph` | 0.9114 → 0.9114 | 0.7946 → 0.7946 | 0.0481 → 0.0481 | 0.3351 → 0.3301 | **0.2005 → 0.1930** | 0.6386 → 0.6347 |
| `lme_s_recall_probe` | 0.9114 → 0.9114 | — | — | 0.1347 → 0.1080 | — | — |

Only category 2 on LoCoMo moves materially. Ten of the eleven runs move by less than 2 points
overall; the outlier is `locomo_recall_probe` at −3.67, where the resolvable temporal golds are
24.2% of a 153-answer denominator against 18.4% in the full run.

The rescored subtree is 5.7 MB against the 110 MB `runs/` already carries, and nothing under
`runs/rescored/` is gitignored, so these are committed artifacts like every other run.

The re-read is also a port check. Re-scoring `runs/locomo_recall` with `--scorer token-f1`
reproduced the historical `score` and `exact_match` on **0 of 1,986 rows differing** — the offline
path recomputes from the raw text rather than carrying the old number forward, so this verifies the
normalisation as well as the plumbing.

`runs/` also holds four directories written by the vendored LME-V2 harness whose rows share the name
`per_question.jsonl` but not the schema. All four are refused by name rather than by serde message:

```
$ myelin-eval rescore --run runs/myelin_k25_web_small --scorer temporal
Error: runs/myelin_k25_web_small is not a `bench` run directory (its per_question.jsonl has no
`exact_match`/`tenant`); the vendored LME-V2 harness writes a different row shape
```

### One deviation from the pre-registered target numbers, and why

The plan pinned a prototype's category-2 coverage at 259/321 and a rescored category-2 score of
0.2087. The shipped grammar resolves **269/321** and scores **0.2022** — 10 more golds and 0.65
points lower. The plan predicted this: it flagged `November 5-6, 2022`, `between October 19 and 24,
2023` and `It happened on the 7th of May 2023` as *the prototype's known bugs*, fixed by the
normalisation and filler rules, and required them to fail a naive port. Fixing them necessarily
resolves more golds than the prototype did. The forms that account for the difference are the digit
range (`November 5-6, 2022`), the `and` connective (`between … and …`, 2 rows), `night of …`,
ordinal suffixes (`October 13th`), hedges outside durations (`Around August 2022`,
`around April 2, 2023`, `approximately summer of 2022`) and `Last week before 13 October 2022.`
Every resolved form is a row of the pinned grammar table. All other pinned numbers reproduced
exactly: category 1 4/282 and 0.4252 → 0.4242, category 3 0/96, category 4 11/841, the 1,219-item
pooled guard at −0.02, the LongMemEval_S answerable 0.4369 → 0.4313, the 272-row judge docket and
the 36-item disagreement count.

## Do M12's and M13's verdicts survive the re-read?

Every paired CI recomputed on the rescored pairs and read against each milestone's own
pre-registered rule. Recomputing the token-F1 columns first reproduced both published tables exactly
(e.g. M12's LoCoMo cat 3 `+1.5, [−2.4, +6.0]` is `+1.541, [−2.403, +5.952]`), so the comparison is
like for like.

### M12 — `RetrieveConfig::graph`

Rule: `graph` → `true` iff LoCoMo cat 3 **and** LongMemEval_S cat 4 both have CIs excluding zero on
the positive side, **and** neither corpus's `overall` CI has a lower bound below −1.0.

| condition | token F1 (M12) | temporal (M14) | condition |
|---|---|---|---|
| LoCoMo cat 3 CI > 0 | +1.54, [−2.40, +5.95] fail | +1.54, [−2.40, +5.95] fail | unchanged |
| LME_S cat 4 CI > 0 | −0.69, [−1.88, +0.00] fail | −0.19, [−0.56, +0.00] fail | unchanged |
| LoCoMo overall ≥ −1.0 | −0.29 pass | −0.14 pass | unchanged |
| LME_S overall ≥ −1.0 | −1.54 fail | −1.36 fail | unchanged |

**Verdict unchanged: `graph` stays `false`.** Category 3 has no resolvable temporal golds at all, so
its interval is byte-identical; the LongMemEval_S intervals tighten without crossing a threshold.

### M13 — `ComposeConfig::chronological` and `bench --question-date`

Rule: each switch flips iff its target stratum improves with a CI excluding zero on the positive
side — LoCoMo cat 2 for either switch, plus LongMemEval_S cat 5 for `chronological` — **and** no
`overall` CI for that switch has a lower bound below −1.0.

| switch | condition | token F1 (M13) | temporal (M14) | condition |
|---|---|---|---|---|
| `chronological` | LoCoMo cat 2 CI > 0 | −0.30, [−1.94, +1.31] fail | −0.55, [−2.49, +1.32] fail | unchanged |
| `chronological` | LME_S cat 5 CI > 0 | −1.07, [−5.21, +3.07] fail | −0.70, [−4.80, +3.37] fail | unchanged |
| `chronological` | LoCoMo overall ≥ −1.0 | −0.996 pass | **−1.041 fail** | **flipped** |
| `chronological` | LME_S overall ≥ −1.0 | −1.152 fail | **−0.960 pass** | **flipped** |
| `question_date` | LoCoMo cat 2 CI > 0 | +0.24, [−1.75, +2.16] fail | −0.31, [−2.80, +2.18] fail | unchanged |
| `question_date` | LoCoMo overall ≥ −1.0 | −0.12 pass | −0.26 pass | unchanged |

**Verdicts unchanged: both defaults stay `false`.** Condition (a) fails for both switches under both
scorers, which is what decides them.

Two condition-(b) evaluations flip, in opposite directions, and both are within 0.05 points of the
−1.0 threshold. They cost nothing, and they are further evidence for the wrinkle M13 already
recorded: *"A future rule of this shape should guard on the upper half of a harm interval, not on
any lower bound below a threshold, or it will veto wide-but-favourable results."* A guard that can
be tripped or untripped by a 0.045-point shift in a bootstrap bound is not measuring harm.

One point estimate changes sign worth naming: M13's headline that `--question-date` moves LoCoMo
category 2 by **+0.2 points** becomes **−0.3 points** under a date-aware scorer. Both intervals
contain zero, so the verdict is the same bounded null — but the apparent direction of the effect was
an artifact of the metric, which is precisely what this milestone predicted would be found.

`--question-date`'s one significant off-target effect is unchanged and exactly reproduced:
**abstention accuracy +3.14 points, CI [+0.67, +5.61], p = 0.020**. Adversarial rows are
scorer-invariant by construction, so this number cannot move — a useful consistency check on the
re-read.

## Reproducing this

```bash
# the grammar and the formula, hermetic
cargo test -p myelin-eval --lib temporal

# re-read every run on disk (no GPU)
for d in locomo_recall locomo_recall_chrono locomo_recall_qdate locomo_recall_chrono_qdate \
         locomo_recall_graph locomo_recall_probe locomo_investigate \
         lme_s_recall lme_s_recall_chrono lme_s_recall_graph lme_s_recall_probe; do
  myelin-eval rescore --run "runs/$d" --scorer temporal
done

# the judge: reader only, 272 calls, ~23 s (needs an ssh -L 5810 tunnel to `big`)
myelin-eval judge --run runs/locomo_recall --category 2

# agreement, kappa, the paired CI and the disagreement dump
python3 crates/myelin-eval/adapters/scorer_agreement.py \
  runs/rescored/locomo_recall_temporal \
  --judge runs/locomo_recall/judge_verdicts.json --category 2

# did M12's and M13's verdicts survive
python3 crates/myelin-eval/adapters/paired_ci.py \
  runs/rescored/locomo_recall_graph_temporal runs/rescored/locomo_recall_temporal --by-category
```

Re-running `judge` judges 0 and reports 272 from cache. A verdicts file written by a different model
is a hard error naming both models rather than a silently mixed κ.

## What this says about the temporal axis

M13 measured two read-layer fixes against a scorer that could not tell a correct date from the date
it was asked to offset from, and found both to be bounded nulls. Under a scorer that can, both are
*still* bounded nulls — the intervals move by fractions of a point and no verdict changes. The
reader's temporal weakness was therefore never hidden by the metric; it was overstated by it. The
true LoCoMo temporal number is **0.2022, not 0.2825**, and the 8-point gap was partial credit for
naming anchors.

What remains open is a reader problem and no longer a measurement problem: of the 269 category-2
items with a resolvable gold, 31 are declined outright and the offset forms
(`The week before <date>`, `<weekday> before <date>`) are answered with the anchor. Both columns are
on every row from here on, so any future attempt at that can be read under either metric without a
single GPU-hour of re-running.
