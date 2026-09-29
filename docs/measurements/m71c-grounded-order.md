# M71c — the grounded pass in the order it was designed *(pre-registered 2026-09-28, before any row)*

## Why

M61, M71 and M71b's grounded pass was designed to write
`{named, supporting, answer}`, in that order:
- name each thing the question names, and the memory that states it;
- cite the memories that state the answer;
- only then answer.

The model never received that order. serde_json sorted the keys, so every
grounded run wrote `{answer, named, supporting}`: it answered first and named
things afterwards (`defect-2026-09-28-schema-field-order.md`). M71b's numbers,
3 fixed and 0 broken on 49 answerable declines, and 5–7 commits per run, all
belong to the answer-first order.

The round-5 catalog (§d) says order matters here:
- Tam et al. (2024, `10.18653/v1/2024.emnlp-industry.91`): "100% of GPT 3.5
  Turbo JSON-mode responses placed the 'answer' key before the 'reason' key,
  resulting in zero-shot direct answering".
- CRANE (Banerjee et al. 2025, arXiv 2502.09061): a grammar that leaves room
  for reasoning preserves it.

## Mechanism

- The same pass, `commit-arm --grounded` (`bench::commit_grounded`), with
  the same code check (`accept_grounded`: every named thing's identifying
  words must appear in the memory cited for it) and the same prompt.
- **Only the schema's order changes:** `named → supporting → answer`, and
  `thing → memory` per named thing. `preserve_order` makes that the wire
  order (#166), and a test pins it.

## Measurement (a post-pass; minutes of GPU)

- **Arm:** the M71c binary runs `commit-arm --grounded` over the round-5
  bundle's pre-grounded replicates `r5_bundle_s{1,2,3}`, writing
  `r5_bundle_s*_grounded_c`.
- **Base:** `r5_bundle_s*_grounded`, the same rows through M71b's
  answer-first pass, written tonight by the round-5 binary.
- Both arms start from identical rows, so the grounded pass's order is the
  only difference.
- **Graders:** the strict 9B, seeded from each `_grounded` run, and
  LongMemEval's official grader.
- **Stratum:** the rows that decline in any of the three pre-grounded
  replicates, fixed from `r5_bundle_s*` before M71c runs.

## Rule (a variant of a bundled mechanism, so it is a replacement rule)

M71c replaces M71b's order only if, seed-averaged:
1. it commits at least as many declines per replicate, *and*
2. the strict and official differences on the decline stratum are both ≥ 0,
   *and*
3. **the abstention veto:** no abstention row answered where M71b declined.

Otherwise, the branch sets the grounded order back to alphabetical before it
merges, so M71b's measured behaviour stays the one path.

## Predictions

- Commits rise from 5–7 to 7–12 per replicate. Naming each thing first gives
  the in-code check something to match before the model commits to an
  answer.
- Accuracy on commits is at least M71b's, and abstention is unchanged. The
  check is the same, and it caught LongMemEval's "tomatoes and chili" trap
  either way.

## Falsifier

- **Fewer commits:** naming first makes the model name more things than the
  memories state, so the code check refuses more often. Report the named
  counts per row for both orders.

## Result — fewer commits; the replacement rule fails; M71b's order stays *(measured 2026-09-29, 00:33–01:08)*

The M71c branch (#167, stacked on #166) ran on big after tonight's queue.

**#166's control passed first:** with `preserve_order` on, the shipped path's
evidence on the 5 control rows is byte-identical to M57's. #166 is merged as
a verified zero-change refactor.

Then `commit-arm --grounded` in the designed order ran over the round-5
bundle's pre-grounded replicates:

| seed | declines | M71c commits | M71b commits |
|---|---|---|---|
| 1 | 73 | 5 | 8 |
| 2 | 72 | 5 | 8 |
| 3 | 76 | 5 | 6 |

Paired over the three replicates (M71c vs M71b), on the 80 rows that
decline in any pre-grounded replicate:
- strict −0.4 [−3.8, +2.5] and official −0.4 (2/2);
- abstention **−1.1**: one row answered where M71b declined.

**Verdict:** rule 1 (commits at least as many) fails, and the abstention veto
fires. As pre-registered, the grounded pass keeps M71b's measured order,
which is alphabetical, answer first. #167 is closed and not merged.

**The falsifier fires:** naming first made the model list more named things
than the memories state, so the in-code check refused more commits. So for
this pass, on this reader, answer-first is *better*: the model commits, and
the check then decides. That is the opposite of what Tam et al. and CRANE
suggest for reasoning tasks. Here the "reasoning" field is a list the code
checks, not a trace the model uses.
