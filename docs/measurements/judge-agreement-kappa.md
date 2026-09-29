# How far our two graders agree, as Cohen's κ *(measured 2026-09-29)*

## Why

Two graders decide what we report:
- **the strict 9B judge** decides every arm;
- **the benchmark's own grader** decides the SOTA comparison. That is
  LongMemEval's official gpt-4o-mini grader (M70), or LightMem's prompt on
  LoCoMo (M68).

We used to say how far they agree as a percentage. That flatters them when
one label dominates. Across 21 judges and ~541,000 judgments, raw agreement
exceeds Cohen's κ by 33–41 points (Norman, Rivera & Hughes 2026,
"Reliability without Validity", arXiv 2606.19544; round-5 catalog §e).

## Method

`crates/myelin-eval/adapters/judge_agreement.py`:
- **The strict judge** is each row's `score` in `<run>_judged`, the
  rescored run `paired_ci.py` reads.
- **The second grader** is a verdict file in the run.
- **κ** is `scorer_agreement.cohens_kappa`.
- **The 95% interval** uses the large-sample standard error of Fleiss, Cohen
  & Everitt 1969 (`10.1037/h0028106`). That standard error was checked
  against a bootstrap on the rows below, and they match: 0.0328 vs 0.0334,
  and 0.0274 vs 0.0274.
- **The two discordant cells** are printed as well, so a reader sees which
  way the graders disagree.

## Result

| runs | n | raw agreement | **κ [95% CI]** | strict right only | second right only |
|---|---|---|---|---|---|
| shipped round-5 bundle, seed 1 | 500 | 94.2% | **0.817** [0.752, 0.881] | 15 | 14 |
| shipped round-5 bundle, seed 2 | 500 | 94.8% | **0.828** [0.764, 0.892] | 12 | 14 |
| shipped round-5 bundle, seed 3 | 500 | 94.0% | **0.804** [0.736, 0.871] | 13 | 17 |
| base, seeds 1–3 (pooled) | 1,500 | 95.6% | **0.872** (0.868–0.874 per seed) | 36 | 30 |
| LoCoMo `m63_locomo_base`, strict vs LightMem | 1,540 | 91.3% | **0.774** [0.738, 0.809] | 8 | **126** |

LongMemEval_S rows are strict vs official.

**What it says:**
- **LongMemEval_S: the graders agree well, and neither leans.** κ is
  0.80–0.87, about 8–13 points under raw agreement. That is a far smaller
  gap than Norman et al.'s, because our labels are less lopsided (~80%
  right).
  - The disagreements are symmetric: 40 rows are right only under strict,
    and 45 only under official.
  - So the shipped 81.13 official and 80.80 strict differ by grader noise,
    not by a lenient grader.
- **The shipped bundle agrees less than the base (0.816 vs 0.872), and
  almost all of that is the preference stratum.**

  | stratum | bundle: disagree / κ | base: disagree / κ |
  |---|---|---|
  | 30 preference rows × 3 seeds | 24 of 90 / **0.474** | 8 of 90 / 0.798 |
  | the other 470 × 3 | 61 of 1,410 / 0.844 | 58 of 1,410 / 0.866 |

  The bundle's preference clause (M77c) changes how the reader answers
  advice requests, and the two graders grade those answers by different
  rubrics:
  - the strict judge uses LongMemEval's preference criterion
    (`judge-preference-rubric.md`);
  - the official grader uses its own prompt.

  On these answers they part.
- **LoCoMo: LightMem is the lenient grader, one-sidedly.**
  - 126 answers are right under LightMem and wrong under strict; only 8 go
    the other way.
  - This is M68's finding (134 disagreements), now with a κ.
  - It is why the LoCoMo row is `comparable` under MemPro's grader while
    the strict judge's 70.52 stays our headline.

## Reproduce

```
python3 crates/myelin-eval/adapters/judge_agreement.py \
  runs/r5_bundle_s1_grounded,runs/r5_bundle_s2_grounded,runs/r5_bundle_s3_grounded
python3 crates/myelin-eval/adapters/judge_agreement.py \
  runs/m57_bonsai_premise_s1,runs/r5_base_s2,runs/r5_base_s3
python3 crates/myelin-eval/adapters/judge_agreement.py runs/m63_locomo_base \
  --verdicts judge_verdicts_lightmem.json
```

`runs/m63_locomo_base_judged` comes from `myelin-eval rescore --run
runs/m63_locomo_base --scorer judge --out runs/m63_locomo_base_judged`.
