# M56 — a second opinion on when to answer *(Step 0 measured 2026-09-24: gate failed, recorded)*

## Why

M55 found the bigger model knows more but misjudges when to answer. On
LongMemEval_S it went from 78.40 to 82.20, but abstention fell 28 → 25/30
and the veto fired. On LoCoMo it went from 69.87 to 66.69: declines rose
121 → 292, and 118 of the new ones had the gold turn in hand. Every gate
this project tried was decided by the reader itself or by the
cross-encoder: M6, M35, M36/M37, M42, M44 R1, M45, M47. All failed, and
M45's falsifier concluded "the discriminator has to be external".

The literature says the same:
- **CRAG** (Yan et al., 2024, arXiv:2401.15884) gates generation on a
  separate lightweight evaluator.
- **Self-RAG** (Asai et al., 2023, arXiv:2310.11511) critiques whether an
  output is supported by the passages (`IsSup`).
- **"Don't Hallucinate, Abstain"** (Feng et al., ACL 2024,
  10.18653/v1/2024.acl-long.786) finds that abstention decided by other
  models beats self-evaluation.

**The checker.** The external checker is TypeSafe's **Jev**
(`typesafe/jev-1.13`, snapshot `jev-1.13-20260917`), a "System One"
decision model served through OpenRouter's Decisions API. It reads 32k
tokens, returns calibrated typed answers, and costs $0.042 per million
input tokens. The user allowed cloud models "anywhere it helps" on
2026-09-24, so every number this produces carries the caveat **"local AI +
cloud checker"**.

**Checkers ruled out.** Laya holds ~320 tokens of state, against evidence
of p50 2,834 tokens on LongMemEval_S and 1,147 on LoCoMo. von holds 2,048
tokens and is not deployed.

## Step 0 — replay existing answers through Jev *(pre-registered)*

No reader or judge calls. For every row that carries an answer, Jev
receives the evidence the reader saw, the question and the answer, in one
Decisions request with two questions:

- `support`: choice. Label definitions follow OpenRouter's Jev-verified
  cascade recipe.
  - supported: the answer addresses the question, and every fact in it
    appears in the memories.
  - unsupported: the answer states something the memories do not contain
    or contradict, or it answers a different question.
  - declined: the answer says the memories do not cover the question and
    adds no facts of its own.
- `false_premise`: noul, phrased declaratively, because Laya's noul head
  fails on interrogatives: "The question takes for granted something the
  memories do not say."

**Policy.** Argmax and 0.5, fixed before any verdict; nothing is tuned to
the test.

1. `false_premise` > 0.5 → decline: "I don't know." followed by the draft
   as the explanation.
2. else `support` = unsupported → decline.
3. else keep the draft.
4. LoCoMo only, rows the arm's reader declined: the 9B's own answer from
   `runs/m19_locomo_full` stands in for a commit pass (the evidence is near
   identical: gold turns held 1,732 vs 1,736). It is adopted only if its
   `support` = supported and `false_premise` ≤ 0.5.

**Scoring.** Exact, from existing verdicts, with no new judging:
- A kept or adopted answer keeps its judge verdict.
- A decline scores 0 on an answerable row and 1 on an abstention or
  adversarial row. That is the string rule `bench::is_abstention` already
  applies.
- Cross-check, verified: rebuilding each base from its own verdicts gives
  78.40 (`m44_r2_s1`), 82.20 (`m55_bonsai_s1`), 69.87 (`m19_locomo_full`)
  and 66.69 (`m55b_locomo_bonsai`).

**Runs replayed:**
- `runs/m55_bonsai_s1_judged` (Bonsai) and `runs/m44_r2_s1_judged` (9B),
  on LongMemEval_S.
- `runs/m55b_locomo_bonsai` with the `runs/m19_locomo_full` proxy, on
  LoCoMo.

**Reported.** Simulated score, abstention (30), adversarial (446),
declines, and the AUROC of P(supported) separating judged-correct from
judged-wrong answers. For comparison, the self-agreement signal in M45
scored 0.592.

**Build gate.** Either of:
- LongMemEval_S simulated ≥ **80.80** with abstention ≥ **28/30**;
- LoCoMo simulated ≥ **+3.0** over 69.87 with adversarial not lower than
  69.96.

Either one passing → build the verifier into the read path (Step 1) and
run real arms. Neither → recorded, and the next arm is the reader-prompt
fix ("I don't know" first, then explain).

**Falsifiers.**
- Jev's AUROC ≤ 0.65, meaning it does not know supported from unsupported
  on this data any better than the reader's self-agreement.
- The policy declines more answerable rows it had right than wrong ones it
  catches.

## Results — Step 0, measured 2026-09-24

**The build gate failed on both benchmarks.** Nothing is built into the
read path. The whole probe cost **$0.22**: 2,385 calls to `jev-1.13-20260917`.

| replay | real | simulated with Jev | should-decline rows | answerable declines |
|---|---|---|---|---|
| LongMemEval_S, Bonsai (`m55_bonsai_s1`) | 82.20 | **59.80** | 25 → 30/30 | 180 |
| LongMemEval_S, 9B (`m44_r2_s1`) | 78.40 | **57.00** | 28 → 30/30 | 188 |
| LoCoMo, Bonsai (`m55b_locomo_bonsai`) | 66.69 | **62.99** | adversarial 92.83 → 95.96 | 355 |

**The falsifier fired on LongMemEval_S.** The policy lost 117 rows the
reader had right and gained 5.

**Jev's discrimination.** AUROC of P(supported) against the judge's
verdict: 0.704 (Bonsai), 0.743 (9B), 0.653 (LoCoMo). That beats M45's
self-agreement (0.592), but it is not enough to gate on.

**What broke it** (diagnosis, not a re-fit). The false-premise question
fired on 100 answers the judge had right and 33 it had wrong. Its median was
0.30 on right answers and 0.53 on wrong ones, so the 0.5 cut sits inside
the right answers' spread. The support check alone did better, but still
lost. Exploratory only, never pre-registered, reported so nobody re-runs it:

| exploratory variant | simulated | real |
|---|---|---|
| LongMemEval_S, Bonsai, support only | 72.40 (abstention 27/30) | 82.20 |
| LongMemEval_S, 9B, support only | 69.80 (abstention 29/30) | 78.40 |
| LoCoMo, Bonsai, support only | 65.45 | 66.69 |
| LoCoMo, support + 9B stand-ins | 68.51 (adversarial 91.48) | 66.69 (9B: 69.87) |

On LongMemEval_S, 51 of the 64 answers Jev called unsupported were judged
correct. Those answers are counts, date arithmetic and syntheses across
sessions, and a zero-shot checker reads them as "not in the memories". The
stand-ins it approved on LoCoMo were right 47 times in 120.

**Why a checker has little to gain here.** On an answerable row, a wrong
answer and a decline both score 0. So a gate can only earn points on the
abstention rows (30 on LongMemEval_S), while every false alarm costs one.
The veto that stopped M55 is three rows. That is a job for the reader's own
wording, not a veto on every answer. CRAG's evaluator was fine-tuned for
its retrieval setting; a zero-shot decision model is not a substitute on
this data.

**Next arm.** As pre-registered: the reader-prompt fix, "I don't know"
first and then the explanation, on Bonsai (M57).

Artifacts:
- `runs/m56_jev_probe/summary.json`
- `runs/m56_jev_probe/jev_verdicts.jsonl`: every verdict, keyed by
  content, so a rerun is free.
- `crates/myelin-eval/adapters/jev_probe.py`

