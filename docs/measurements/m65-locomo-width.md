# M65 — how much evidence LoCoMo gets: k *(L3; retrieval measured 2026-09-24; reader arm pre-registered)*

## Why

The bottleneck review found `k` binds and the token budget does not on
LoCoMo:
- evidence averages ~31% of the 4,096-token budget;
- every LoCoMo reader run has used k = 6;
- multi-hop (list aggregation across sessions) needs several turns and
  rarely got all of them.

M29's "a wider budget made the 9B worse" was measured on budget, not on k,
and on LongMemEval_S. Du et al. 2025 (`10.18653/v1/2025.findings-emnlp.1264`)
is the standing warning: length alone costs accuracy even when retrieval
is perfect.

## Retrieval, no reader *(measured 2026-09-24 17:31–17:58)*

`myelin-eval ablate --width --grid shipped --k K [--dedupe-lineage]`: the
LoCoMo dev split, 997 questions, the shipped width (50/25) and a
4,096-token budget. Gold is counted through I4 lineage, so a fact
abstracted from a gold turn counts as holding it. That is more generous
than the verbatim check in the bottleneck review. `all` is the new
instrument: the share of questions whose evidence holds **every** gold
turn.

| cell | mean gold recall | every gold turn held | truncation loss | pool recall |
|---|---|---|---|---|
| k = 6 (shipped) | 0.9056 | 0.8626 | 0.0536 | 0.9593 |
| k = 6 + dedupe (M63) | 0.9138 | 0.8716 | 0.0454 | 0.9593 |
| k = 10 | 0.9321 | 0.8937 | 0.0272 | 0.9593 |
| k = 10 + dedupe | **0.9395** | **0.9017** | 0.0197 | 0.9593 |

**Read:**
- M63's dedupe frees a little at either width: +0.9 points of all-gold.
- k = 10 recovers half of k = 6's truncation loss: +3.1 all-gold.
- Together they add +3.9.
- The pool ceiling (0.959) is untouched by both, as it must be: neither
  changes retrieval, only what is kept.
- No token drops of note: 0.26 per query at k = 10.

## The reader arm *(pre-registered, before any row)*

- **M65:** the shipped LoCoMo settings at **k = 10**
  (`bench --corpus locomo --mode recall --k 10 --max-steps 2`), full
  1,986, 9B.
- Paired against M63's fresh base (`runs/m63_locomo_base`), and judged by
  the 9B seeded from it.
- It runs in a window between M54 chunks, only after M61/M63/M64 report.
  If M63 clears, k = 10 is measured with dedupe on top, the
  bundle-validated rule.

**Bar and veto.** +3.0 on judge 1–4 with the 95% CI excluding zero, and
adversarial at or above the fresh base's.

**Predictions.**
- Multi-hop +2 to +4, the stratum that needs several turns.
- Temporal and single-hop flat to −1: more text around the same one gold
  turn.
- **Overall +0.5 to +1.5**, likely under the bar. The retrieval gain is
  3.1 points of all-gold, and length costs a 9B something.

**Falsifier.** Multi-hop does not rise although all-gold rose. That would
mean the reader does not use the added turns, which is Du 2025's length
effect winning.
