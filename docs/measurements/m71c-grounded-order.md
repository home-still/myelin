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
