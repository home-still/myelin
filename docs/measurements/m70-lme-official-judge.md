# M70 — LongMemEval_S graded the way LongMemEval grades *(pre-registered 2026-09-25, before any verdict)*

## Why

LongMemEval_S ships at **83.40** (M57, `runs/m57_bonsai_premise_s1`),
2.60 past MemPro-15 on Qwen3-30B (80.80). `standing` keeps that gate open:
the row is `caveat-judge`, because theirs was graded by gpt-4o-mini and ours
by a strict local 9B.

## Which grader "matched" means here

MemPro says its LongMemEval judge "follows GAM" (arXiv 2606.00619, L122).
None of the candidate prompts is a runnable reading of how MemPro graded
answers:
- **GAM** (VectorSpaceLab/general-agentic-memory, every branch) never
  published a LongMemEval judge.
- **MemPro's Fig. 11** is a hand-merged paraphrase of the official templates.
  It has no abstention path, so the 30 `_abs` rows would be graded against
  their explanation as if it were an answer.
- **MemPro's repo** (`eval/longmemeval_test.py`) grades research summaries,
  not answers.

The one runnable reading is the benchmark's own grader:
`xiaowu0162/LongMemEval@9e0b455:src/evaluation/evaluate_qa.py`
(`get_anscheck_prompt`; Wu et al. 2024, arXiv 2410.10813):
- **templates:** five per-type templates, byte for byte (checked with
  `--verify-upstream`), with abstention routed by `_abs` in the question id;
- **model:** `gpt-4o-mini-2024-07-18` (its `gpt-4o-mini` entry), pinned to
  OpenAI;
- **call:** `temperature=0`, `max_tokens=10`;
- **verdict:** correct iff "yes" is in the reply.

It judges all 500 rows, the 11 declines included. Its type counts reproduce
MemPro's 500-row Avg.

The verdicts go to `judge_verdicts_lme_official.json`. `standing` publishes
them as `longmemeval_s.judge_score_official.n500`, and under the user's
every-reading rule the LongMemEval_S gate moves to that row. The strict 9B
stays the headline and decides every arm.

## Predictions (before any verdict)

- Official ≥ strict: **84–88**. The official templates accept "equivalent" and
  "contains the answer". Our strict rubric counted 11 declines and was harsh
  on long answers (M9).
- Abstention stays ~29/30. The official abstention template asks whether the
  model identifies the question as unanswerable, and M57's reader says
  "I don't know." first.

**Falsifier.** If the official score is below 80.80, the LongMemEval_S lead
exists only under our own judge. This doc then says so, and the gate stays
open.
