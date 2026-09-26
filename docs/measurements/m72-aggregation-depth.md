# M72 — counting questions need every mention: question-shape-aware depth *(retrieval measured first; 2026-09-25)*

## Why

Graded by LongMemEval's own grader (M70), the shipped LongMemEval_S run loses
31 of its 121 answerable multi-session questions.
- **26 of the 31 are counting or summing questions.** Most held only part of
  their gold turns and **undercounted**:
  - "How many pieces of furniture…": 1 of 5 mentions in the evidence,
    answered 2, gold 4.
  - "How many health-related devices…": 2 of 6, answered 2, gold 4.
- **101 of the 121** multi-session questions have this shape.

MemPro's failure-driven pipeline evolution found the same lever. Its
"adaptive retrieval depth" iteration gained +1.34 (Liu et al. 2026, arXiv
2606.00619, App. A.1). JustMem types an *aggregate* operation and composes
for it. Its LongMemEval_S aggregation questions went 65.96 → 78.72 (Chen
et al. 2026, arXiv 2609.19877). The risk is length: Du et al. 2025
(`10.18653/v1/2025.findings-emnlp.1264`), and M29, where a wider budget hurt
the 9B.

## The instrument

`query_shape::is_aggregation_question` is a deterministic cue list ("how
many", "how much", "in total", "total", "altogether", "combined"), minus
elapsed-time questions (`time::is_interval_question`). `recall` allows no
model in its loop (`PLAN.md` §7.1). `ablate --width` now reports the
aggregation stratum's own `all` and `turn_all` beside the whole set's.

## Stage 1 — retrieval only

`ablate --width --corpus longmemeval-s --grid budget --k K` for K ∈ {6, 12,
18}: the shipped width at budgets 2,048 to 16,384. A reranker only, on big.
Results follow when measured. The reader arm is pre-registered only after
this, as M65 was.

### Stage 1 result *(2026-09-25, 19:04–19:57; reranker only, on big under the lease M54's controller held)*

LongMemEval_S, recall mode with the selector off (a proxy for the shipped
`investigate` path), the shipped width. Gold is the `has_answer` turns. The
aggregation stratum is the 130 counting and summing questions with matchable
gold.

| k | budget | all gold held | **counting questions** | evidence tokens |
|---|---|---|---|---|
| 6 | 4,096 (shipped) | 0.6695 | **0.6385** | 2,931 |
| 12 | 4,096 | 0.8410 | 0.7462 | 4,149 |
| 12 | 8,192 | 0.7929 | 0.7462 | 5,999 |
| 18 | 8,192 | 0.8933 | **0.8385** | 7,732 |
| 18 | 16,384 | 0.8808 | 0.8308 | 8,233 |

- **The pool is not the limit.** It holds 0.9665. At k = 6, one question in
  five loses gold to truncation.
- **k = 18 at 8,192 tokens** holds every mention for 84% of counting
  questions, against 64%: **+20 points**. It costs 2.6× the evidence
  tokens, which is why depth is raised for counting questions only.

## Stage 2 — the reader arm *(pre-registered 2026-09-25, before any row)*

- **Switch.** `bench --aggregation-k 18 --aggregation-budget-tokens 8192`.
  - A question that `is_aggregation_question` flags is retrieved at k = 18
    and 8,192 tokens.
  - Every other question keeps the shipped k = 6 and 4,096 tokens.
  - Recorded on the run, and counted as an arm by `standing`.
- **Exact control.**
  - Only the 137 flagged questions are re-run (90 multi-session, 30
    knowledge-update, 12 single-session-user, 5 single-session-assistant),
    with the shipped LongMemEval_S settings (M57's command) plus the switch.
  - The other 363 rows are copied byte for byte from
    `runs/m57_bonsai_premise_s1`, because their pipeline is unchanged.
  - The copied rows are merged with `bench --resume`.
- **Judging.** The strict 9B, seeded from the base, plus LongMemEval's own
  grader (M70).
- **Bar:** +3.0 on the strict judge over all 500, with the paired 95% CI
  excluding zero.
- **Veto:** any drop on the 30 abstention rows. None of them is re-run
  unless flagged.
- **Predictions:**
  - multi-session +8 to +14 on its 121 answerable;
  - knowledge-update −3 to +2, because deeper retrieval surfaces older
    values of a changed fact;
  - **overall +2 to +3.5**;
  - official 78.60 → 80.5–82.
- **Falsifiers:**
  - Multi-session does not rise although counting coverage rose by 20
    points: the reader does not count what it is given (Du 2025's length
    effect).
  - Knowledge-update falls by more than 5: deeper evidence is stale
    evidence.
- **Order.** It queues behind M71 for big.

## Stage 2 result *(2026-09-25 23:55 → 09-26 01:26; Bonsai on big, then the 9B to judge)*

`runs/m72_lme_agg`: the 137 flagged questions were re-read at k = 18 and
8,192 tokens, and the other 363 rows are the base's own.

| | base (M57) | arm | Δ (95% CI) |
|---|---|---|---|
| strict 9B, all 500 | 79.20 | **80.20** | **+1.0 [−0.2, +2.2]** |
| abstention (30) | 29/30 | 29/30 | 0 |
| LongMemEval's own grader | 78.60 | 79.00 | +0.40 |

**By stratum (strict, correct rows):**

| stratum | n | base | arm |
|---|---|---|---|
| multi-session, counting | 90 | 69 | **74 (+5)** |
| knowledge-update, counting | 30 | 28 | 28 |
| single-session, counting | 17 | 15 | 15 |
| every unflagged question | 363 | unchanged | unchanged |

**It does not ship.** +1.0 is under the +3.0 bar, and the CI spans zero.

**Against the predictions:**
- **Multi-session +5 (strict), predicted +8 to +14.** Evidence per counting
  question grew from 7.9 to 13.4 items, and retrieval had shown 20 more
  points of every-mention coverage. The reader turned that into 5 more
  right answers: 58 counting answers changed, 7 for the better and 2 for
  the worse.
- **The first falsifier fired in part.** The reader does not count what it
  is given. Coverage was the smaller half of the problem.
- **Knowledge-update did not fall** (28 → 28), so the stale-evidence risk
  did not materialise.

**What it says about the next lever.** Retrieval now delivers most of the
mentions, and the loss is in turning them into a count. The candidates are:
- enumeration before counting (Chain-of-Note, Yu et al. 2024,
  `10.18653/v1/2024.emnlp-main.813`);
- counting in code over structured items (APEX-MEM's SQL COUNT, Banerjee
  et al. 2026, `10.18653/v1/2026.acl-long.749`).

Each would be its own pre-registered arm, and the enumeration one runs
into the code-first rule if it is a prompt clause.
