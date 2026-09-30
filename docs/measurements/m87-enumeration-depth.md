# M87 — list and count questions retrieved deeper *(round 7, planned 2026-09-30; pre-registered before any row)*

## Why

LoCoMo's bar moved on 2026-09-30 to LeanMem (Qwen3-8B), 84.41 under
SimpleMem's judge. We read 78.64 under the same judge (−5.77). Against the
per-category rows we have, multi-hop is the largest *retrieval* loss (loss
anatomy of `runs/m84_locomo_base`, LightMem's grader):
- multi-hop scores 70.57 against MemPro-15's 75.17 (−13 questions);
- **80 of its 83 losses miss at least one gold turn**, and 186 of the 192
  missing turns sit in chunks absent from the evidence altogether;
- the evidence holds 0.27 of the gold turns on a loss and 0.50 on a win;
- 58 of the 83 ask for a list or a count ("What activities has Melanie done
  with her family?", "How many tournaments has Nate won?").

Every LoCoMo question is read at k = 6. M65 measured k = 10 on LoCoMo
reader-free: every gold turn held went from 0.8626 to 0.8937 (+3.1), with
0.26 token drops per query. Its reader arm never ran. Length alone costs
accuracy even with perfect retrieval (Du et al. 2025,
`10.18653/v1/2025.findings-emnlp.1264`), so the depth stays question-gated,
as M72's aggregation depth is on LongMemEval_S.

## Mechanism

`query_shape::is_enumeration_question`, with the depth applied by
`bench::depth_for` under `--enumeration-k/--enumeration-budget-tokens`. It is
deterministic.

A question is a list when any of these holds:
- **A list cue:** "what are some", "what kinds of", "what types of", "what
  sorts of".
- **A plural head noun.** The words between its first "what"/"which" and the
  first auxiliary (has, have, had, did, does, do, are, were, is, was), within
  6 words, end on a plural head noun.
  - The head is the word before `of`. When that word is a kind word ("kind
    of place"), the head is the span's last word instead.
  - Plural: irregular (children, people, men, women), or at least 4 letters
    ending in `s` but not `ss`/`us`/`is`, and not in a short list of
    non-plural `-s` words (news, series, lens, always, …).
- **Plural agreement:** the auxiliary follows the wh-word directly, is "are"
  or "were", and the next word is not I/you/we/they ("What are Nate's
  hobbies?").

`depth_for` gives a question the deepest `(k, budget)` of the gates that
open on it (aggregation, enumeration), element by element, or the run's own
when none does. Shipped LongMemEval_S depth is unchanged: its aggregation
gate is the only one it sets.

Grounds:
- JustMem types list operations and composes for them (arXiv 2609.19877).
- MemPro's adaptive retrieval depth gained +1.34 (arXiv 2606.00619).
- CueMem expands from anchor turns (arXiv 2609.12354; the larger step, not
  this one).

## Measured before any row

`myelin-eval shapes` runs the same Rust functions `bench` calls.

| LoCoMo stratum | enumeration | enumeration ∪ count (the arm) |
|---|---|---|
| multi-hop | 134 / 282 | **158 / 282** |
| single-hop | 68 / 841 | 71 / 841 |
| temporal | 1 / 321 | 3 / 321 |
| open-domain | 3 / 96 | 4 / 96 |
| adversarial | 38 / 446 | 40 / 446 |
| **all** | 244 / 1,986 | **276 / 1,986** |

- The Rust counts reproduce the design pass's Python prototype row for row
  on LoCoMo.
- On LongMemEval_S the enumeration gate fires on 5 of 500, and on no
  abstention item.
- Of the shipped run's 83 multi-hop losses, the arm's gate covers **54**
  under LightMem's grader and **49** under SimpleMem's.

## Pre-registration

**Arm:** the shipped LoCoMo recipe on the 276 gated questions only (`bench
--corpus locomo --mode recall --k 6 --budget-tokens 4096 --questions
<ids>`, ids from `shapes --corpus locomo --shape enumeration,aggregation
--ids-out`), plus both gates at M65's width:
- `--enumeration-k 10 --enumeration-budget-tokens 4096`;
- `--aggregation-k 10 --aggregation-budget-tokens 4096`.

The run is spliced into the deterministic LoCoMo base by construction, so
the other 1,710 rows are the base's own. Then come the shipped `commit-arm
--non-recall` pass and every judge:
- the strict 9B, seeded from the base;
- LightMem's, MemPro's and SimpleMem's, each seeded from the base's
  verdicts, so an unchanged answer keeps its verdict.

**Control:** `det_locomo_base` → `det_locomo_m84`, the deterministic
re-measurement of the shipped recipe.

**The stratum gate:**
- **Stratum:** the 158 gated multi-hop questions. The paired 95% CI
  excludes 0 under SimpleMem's judge, the bar's judge.
- **The other readings agree:** LightMem's and the strict 9B's Δ on the
  stratum are ≥ 0.
- **Adversarial veto:** the 40 gated adversarial rows are not below the
  control under the strict judge.
- **No harm:** the full-1,540 SimpleMem Δ is ≥ 0.

**Ship rule:** as M86. A stratum pass joins round 7's bundle at +3.0; a pass
below that goes to the user.

**Predictions:**
- gated multi-hop +3 to +8 questions;
- gated single-hop −1 to +1;
- adversarial flat;
- overall +0.2 to +0.6.
- The gap to LeanMem is 89 questions, so this arm cannot close it alone. It
  measures whether coverage is the lever.

## Result

*(not yet run)*
