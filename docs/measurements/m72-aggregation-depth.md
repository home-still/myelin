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
