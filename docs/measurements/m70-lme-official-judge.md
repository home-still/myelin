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

## Result *(2026-09-25, 18:16; $0.012)*

**78.60** (393/500) on `runs/m57_bonsai_premise_s1`, against MemPro-15
Qwen3-30B's 80.80: **2.20 behind**. The falsifier fired: the LongMemEval_S
lead existed only under our own judge.

| type | n | official |
|---|---|---|
| single-session-user | 64 | 84.38 |
| single-session-assistant | 56 | 89.29 |
| single-session-preference | 30 | 43.33 |
| multi-session | 121 | 74.38 |
| temporal-reasoning | 127 | 76.38 |
| knowledge-update | 72 | 84.72 |
| abstention (`_abs`, 30) | 30 | 93.33 |

**It also found a defect.** The disagreeing rows included decline-first
answers that our strict judge had "passed". Their verdicts belonged to
answers M57 no longer gives (a seed's). Fixed and corrected in PR #119
(`docs/measurements/defect-2026-09-25-stale-verdicts.md`):
- M57's strict score is **79.20**, not 83.40. The strict and official
  graders now agree within 0.6.
- The claim of 2026-09-24 that LongMemEval_S passed 80.80 was wrong, under
  both graders.

**Where the gap is under the official grader.**
- Single-session-preference scores 43.33 (13 of 30). The official
  preference template asks whether the response "recalls and utilizes the
  user's personal information". Our reader's short answers often do not.
- Multi-session is 74.38 and temporal-reasoning 76.38.

`standing` gate: `longmemeval_s.judge_score_matched.n500` = 78.60, comparable,
behind by 2.20.
