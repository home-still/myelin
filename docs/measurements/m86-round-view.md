# M86 — a reply is shown with the user turn it answers *(round 7, planned 2026-09-30; pre-registered before any row)*

## Why

The largest block of LongMemEval_S losses that memory code can still recover
is a segmentation split, not a reading failure. (Loss anatomy on the shipped
round-5 replicates `runs/r5_nli_s{1,2,3}`, 2026-09-30, official grader.)

- **Where the gap is.** Against MemPro-15 (Qwen3-30B, the clean bar), we
  trail on three types and lead on the rest:

  | type | Δ questions per seed |
  |---|---|
  | preference | **−8.3** |
  | single-session-assistant | −4.0 |
  | single-session-user | −2.0 |
  | multi-session | +8.7 |
  | temporal | +7.0 |
  | knowledge-update | +2.3 |

  The overall margin is +3.7 questions, against a seed spread of 7.
- **Why.** Ingest cuts each session into episodes of at most 512 approximate
  tokens (`ingest::SegmentConfig`, `segment()`). A LongMemEval_S assistant
  reply is long (median ~599 tokens), so the cap usually falls between a
  short user turn and the reply that answers it. The user's words end one
  episode, and the reply opens the next record.
- **What it costs:**
  - 487 of 500 rows' evidence holds at least one record that opens with an
    assistant turn.
  - 20 of 43 wrong preference rows hold the reply but not the user turn that
    states the preference.
  - Preference scores 44% in that state against 72% with the user turn held.
  - Temporal scores 68% against 93%.
- **What it is not.** Single-session-assistant and single-session-user losses
  are never retrieval or truncation: every gold turn is in the evidence, whole.
  Their 12 always-wrong questions break down as:
  - 5 reader misreads;
  - 3 typed-pass declines;
  - 4 grader false negatives.

## Mechanism

`bench --round-view` (`myelin_core::pipeline::round_view`).

When an episode chosen for the reader opens with an `assistant:` turn, the
view shows the user turn it answers as its own item directly before it. That
turn is the last turn of the preceding episode of the same session.

**Rules, fixed before any row:**
- **Lookup.**
  - A LongMemEval_S turn's source is `{session}#{turn index}`, and an episode
    carries its first turn's.
  - The predecessor is the same-session episode with the largest first index
    below this one, among the scope's visible episodes.
- **It fires only when all four hold:**
  - the predecessor's last turn is a `user:` turn;
  - the predecessor ends on the turn just before this episode's first (an
    empty turn skipped at ingest leaves a gap, and a gap is not bridged);
  - the predecessor is not already in the evidence;
  - the turn is at most `ROUND_PROMPT_MAX_TOKENS` = 256 approximate tokens.
    A longer one is left out whole and never cut. On LongMemEval_S's
    haystacks the prompting user turn is p50 46, p95 114 and p99 318 tokens,
    so 256 admits 98.7% of them and every gold user turn outside
    single-session-assistant.
- **Beside the budget, not inside it.** The added item is outside compose's
  packing budget, so no chosen record is displaced. That is M50b's lesson:
  events that took turns' slots cost more than they gave.
- **The same rendering as its neighbours.** The added turn goes through the
  run's own compose configuration, as a one-turn window of its record. It is
  dated, elided with `…`, and quotes the record it came from.
- **Session ids never reach the reader.** They are read only for the lookup.
  A test asserts that no added item contains the session id; LongMemEval_S
  names its evidence sessions `answer_*`.
- **A no-op elsewhere by construction.** LoCoMo's sources are dialogue ids
  (`D1:3`) and LME-V2's are `<trajectory>:events`/`:notes`. Neither has a
  `#`, so the view returns before any ledger read. Neither corpus has an
  `assistant:` speaker.
- **It refuses `--turn-windows`,** because a windowed item does not open
  with its record's first turn.
- **Counts on every row:** candidates, fired, and each skip reason. They are
  recorded on the row (`round_view`) and summed on the run
  (`round_view_stats`).

**The grounds:**
- LongMemEval (Wu et al. 2024, arXiv 2410.10813): the user-assistant round is
  the best value granularity.
- JustMem (Chen et al. 2026, arXiv 2609.19877): replaying the source session
  for fidelity-sensitive evidence took single-session-assistant from 32.14 to
  94.64.
- MemLoc (arXiv 2609.07093): keep the full memory context and add cues;
  showing only an extract cost 4.5–9.0.

The research catalog is `docs/research/sota-catalog-2026-09-30.md`.

## Measured before any row (offline replay, no reader)

The view was replayed in Python on the shipped replicates' evidence, matching
each `assistant:`-first item to its haystack turn. This is an approximation of
the Rust view (a text-prefix match, and "already present" means anywhere in
the evidence); the arm's own counts are the measured ones.

| | |
|---|---|
| seed-1 rows where it fires | **424 of 500** |
| fires over 3 seeds | 4,072; skipped 17 over the cap, 3,408 already present |
| row-seeds where a **missing gold user turn** is restored | **124**: preference 39 (of 90), temporal 50, multi-session 23, knowledge-update 12 |

## Pre-registration

**Arm:** the shipped LongMemEval_S recipe (round-5 flags → `commit-arm
--grounded` → `commit-arm --typed-nli --speaker-contrast --assert-statement`,
`docs/measurements/m83b-assert-statement.md`) plus `--round-view`.

**Control:** the deterministic re-measurement of the same recipe
(`runs/det_lme_s{1,2,3}`, Qdrant ties ordered by id). The switch-off path is
byte-identical by construction, and verified: an `--evidence-only` run with
the branch binary on the 5 `control_ids` must reproduce the control's
evidence byte for byte before the arm starts.

**Order and cost:**
1. **Seed 1, all 500 rows** (the view touches 424 of them), paired against
   `det_lme_s1`. Several hours on big.
2. **Seeds 2 and 3** only if seed 1 passes the gate below.

**The stratum gate (per seed 1, then on 3-seed means):**
- **Stratum:** preference ∪ temporal-reasoning (163 questions), official
  grader. The paired 95% CI excludes 0.
- **Abstention veto:** the 30 `_abs` rows are not below the control under
  the official grader or the strict 9B judge.
- **No harm:** the full-500 official Δ is ≥ 0.

**Ship rule (the user's bundle rule):**
- A mechanism that passes its stratum gate joins round 7's bundle arm, which
  ships at **+3.0** on all 500 with the CI excluding 0 and the veto.
- A pass below +3.0 goes to the user, as the NLI pass did on 2026-09-29.

**Predictions (from the anatomy; correlational, not causal):**
- preference +2 to +4 questions per seed;
- temporal +1 to +3;
- multi-session and knowledge-update flat to +1;
- single-session-assistant and single-session-user flat.
- Overall +0.6 to +1.6 official.
- The main risk is the lost-in-the-middle cost of about 250 extra tokens a
  row (Liu et al. 2024, `10.1162/tacl_a_00638`; Du et al. 2025,
  `10.18653/v1/2025.findings-emnlp.1264`).

## Result

*(not yet run)*
