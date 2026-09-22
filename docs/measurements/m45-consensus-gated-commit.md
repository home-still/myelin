# M45 — consensus-gated commit

## What two arms have now measured

M42's forced commit was worth **+35.5 on the 31 rows it changed** and
shipped off because 2 of 30 adversarial rows were talked out of refusing.
M44 R1 re-measured the same trade with a different lever: on the 49
answerable rows the base declined it answered 27 and was right on 11
(**40.7%**, M42's 41.9%), and it talked **14 of 30** adversarial rows into
an answer — seven of them by turning absence into `0` / `Never` /
`Nothing`.

Two prompts, one number. When this reader is made to answer instead of
decline, it is right two times in five, and its own `evidence_absent` hatch
— the only discriminator either arm had — gives way on about half the rows
it exists to protect. The discriminator has to come from somewhere other
than the model's say-so.

## The literature

Farquhar et al., *Detecting hallucinations in large language models using
semantic entropy* (Nature 2024, `10.1038/s41586-024-07421-0`; Kuhn et al.,
`2302.09664`): sample several answers, cluster them by *meaning* rather than
by string, and treat the entropy over clusters as the signal. Semantic
entropy averages **0.790 AUROC** against 0.691 for naive entropy and 0.698
for P(True), and is **stable at 0.78–0.81 from 7B to 70B** — the size class
this project runs. Its *discrete* variant uses cluster counts and no
logprobs, which is what llama.cpp's OpenAI shim can give.

Yadkori et al. (`2405.01563`) turn the threshold into an error-rate
guarantee from ~100 calibration rows: conformal risk control picks the
smallest threshold whose empirical false-commit rate on a held-out split is
at most α.

## The mechanism

`myelin-eval commit-arm --samples N --seed S --agree τ` — M42's offline arm
with the second pass sampled instead of greedy. Omitting `--samples` is
M42's arm byte for byte.

On each row the base declined:

1. **N seeded samples** under M42's schema `{answer, evidence_absent}` at
   the Qwen3 Technical Report's non-thinking setting (temperature 0.7,
   top-p 0.8, top-k 20; thinking stays off). A sample that declines, fails
   to parse, or answers with a decline is `null`.
2. **One structured clustering call**: for every answer, `same_as` — the
   index of the earliest answer that gives the same value — with
   `minItems == maxItems == n` (M40's forcing). Farquhar's N² pairwise
   entailment calls folded into one forced assignment per answer.
   `resolve_clusters` follows links downward only; a forward, out-of-range
   or cyclic link leaves an answer alone, so a malformed assignment can only
   *split* clusters, never merge them. A failed call leaves every answer in
   its own cluster.
3. **Commit the majority cluster's representative iff its share of the N
   samples is ≥ τ.** Otherwise the row keeps its decline, byte for byte.

Every sample and the agreement land on the row (`commit_samples`,
`commit_agreement`), so τ can be re-applied offline: one sampling run serves
both calibration and the arm.

Fail-closed at every step. The reader server caps `n` at its slot count
(measured: `1 <= n <= 2` at `-np 2`), so the N samples are N seeded requests;
llama.cpp's prompt cache makes each one decode-only.

## Pre-registration

Written before any arm ran.

**Base.** The shipped operating point: `runs/m43_dated_judged` (67.80), 76
declining rows.

**Sampling run.** `commit-arm --run runs/m43_dated --samples 5 --seed 0
--agree τ` → `runs/m45_consensus`, judged with `--seed runs/m43_dated`. The
samples are recorded regardless of τ.

**Calibration — an open decision, not assumed.** The backlog names LME-V2's
128 wrong-premise abstention rows as the calibration set. No LME-V2 run at
the shipped configuration exists yet, and the sampler runs over
`ScoredQuestion` rows, which the LME-V2 harness does not produce; both are
work. Two concrete options, for the user to choose:

- **(a) LME-V2, as pre-registered.** Build the LME-V2 base at the shipped
  defaults (M47's base too), add a `ScoredQuestion` export for its rows,
  sample its 128 abstention rows, pick τ by conformal risk control at
  α = 0.05. Honest to the backlog; several GPU-hours and a plumbing PR
  before the first number.
- **(b) LoCoMo category 5.** 446 adversarial rows already on disk with a
  base run, questions whose answer is absent by construction — the
  *silence* shape of abstention rather than LME-V2's *wrong-premise*
  shape. Cheap (one sampling run over declines there), but calibrates
  against a different abstention shape than the one M47 targets.

Until one is chosen, τ = 0.6 (a strict majority of 5) is the **provisional**
value the sampling run is written at, labelled uncalibrated; the reported
number is the one at the calibrated τ, re-applied offline from the recorded
samples. The 30 LongMemEval_S `_abs` rows are never used to pick τ.

**Population.** All 500 LongMemEval_S rows.

**Primary metric.** `longmemeval_s.judge_score.n500`, paired bootstrap.

**Decision rule.** Ship at **≥ +3.0** with a paired 95% CI excluding zero;
**veto** on any drop across the 30 abstention rows.

**Predicted, specifically.**

1. Commits fall from M42's 31 to ~15–20 of the 76 declining rows.
2. Accuracy on committed rows rises above M42's 41.9%.
3. The adversarial rows do not flip: an absent premise produces
   *disagreement* — no cluster reaches τ — where a present one produces the
   same value five times. Agreement on the 27 correctly-declined
   adversarial rows is lower than on the answerable declines.
4. Non-firing rows move **+0.0 exactly** (the offline arm's control, by
   construction).

**Falsifier.** If agreement on the adversarial rows is as high as on the
answerable ones — the reader agrees with itself about an answer that is not
there — then self-consistency does not discriminate for this reader, and
the discriminator has to be external (M47's contradiction check, or an NLI
model), not sampled.

**Cost.** 76 rows × (5 samples + 1 clustering call), decode-only on the
prompt cache: minutes.

## Results

*(pending — the sampling run is queued behind M46 and M44 R2 on the reader;
the calibration choice above is open)*
