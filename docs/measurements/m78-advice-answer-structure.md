# M78 — the advice answer as structure, grounded in the user's own words *(pre-registered 2026-09-28, before any row)*

## Why

LongMemEval_S's preference stratum is where the gap to MemPro-15 lives.
Everything before this is measured, seed-averaged over three seeds, on the
30 preference rows:

| what | measured by |
|---|---|
| The user's own sentence reaches the evidence on 28 of 30. | M76, stage 0 |
| The reader declines anyway, citing `READER_SYSTEM`'s "reply exactly: I don't know". | M76b |
| The preference clause alone lifts official 10.7 → 16.3 and strict 8.7 → 12.0 (the latter on the rubric-corrected judge, post hoc). | M77c; `judge-preference-rubric.md` |
| Declines barely move under the clause: 8.3 → 6.3–7.0 a seed. | M77 |

The clause changed what the advice *says*. It did not change *whether* the
reader answers. This project's one law is that the reader obeys structure
where it ignores instructions (M19, M40, M42, M43). So the next step is
structure: an answer shape with no place to decline, grounded so that it
cannot invent the user's words.

## Mechanism

`--advice-answer` (M78; `src/advice_answer.rs`, `bench::read_advice_answer`).
It applies only to advice requests (`query_shape::is_advice_request`): 29 of
the 30 preference rows, and none of the other 470.

1. **The thinking stays free.** The request is the shipped thinking request
   (same system prompt, seed, sampling and budget) plus a JSON schema on the
   content after the trace. Constraining the reasoning itself costs accuracy
   (Tam et al. 2024, arXiv 2408.02442).
2. **The schema, in wire order** (see
   `defect-2026-09-28-schema-field-order.md`; a test pins it):
   `{"preferences": [{"memory": n, "quote": "…"}] (1–4), "recommendation": "…"}`.
   Recall comes first, then the recommendation (*Attribute First, then
   Generate*, Slobodkin et al. 2024, arXiv 2403.17104). There is no decline
   field.
3. **The check is in code.** A preference is kept only if its quote occurs,
   case- and whitespace-insensitively, inside a *user* turn of the memory it
   cites. It must be at least 12 characters, so "I" cannot pass. The rendered
   answer opens "You told me", so a quote from the assistant would be a false
   statement. (ALCE, Gao et al. 2023, `10.18653/v1/2023.emnlp-main.398`, and
   M71b's `accept_grounded`.)
4. **The answer** is `You told me: "q1"; "q2". <recommendation>`. When no
   quote verifies, the row is the ordinary decline, "I don't know."
5. **Auditing:** the constrained JSON is appended to `reader_trace` after the
   thinking (`[m78:advice-answer]`), so every dropped quote can be read.

PrefEval (Zhao et al. 2025, `10.48550/arxiv.2502.09597`) is the ground: a
retrieved preference plus a reminder makes answers follow preferences, and
its "unhelpful" failure is the decline this removes.

## Smoke test (before any arm; big, under the lease)

One advice row at seed 1, through the real server. The llama.cpp fork must
apply the grammar *after* the reasoning block with Bonsai's template:
- `reader_trace` holds a non-empty thinking trace;
- the content after `[m78:advice-answer]` parses as the schema.

If either fails, the driver stops and nothing runs. The mechanism has no
second path.

## Arms (the 29 fired rows × reader seeds 1–3; the other 471 rows copied from M57)

1. **M78:** `--user-words --advice-profile-clause --advice-answer`, giving
   `runs/m78_answer_s{1,2,3}`.
2. **M78c (diagnostic):** `--advice-profile-clause --advice-answer`, giving
   `runs/m78c_answer_s{1,2,3}`. It separates the words from the structure.
   It enters no bundle unless it clears the same gate itself.

The clause is on in both because it is a bundle candidate (M77c). M78 is
measured as what the structure adds on top of it.

Judges:
- the strict 9B, with the preference rubric (`judge-preference-rubric.md`);
- LongMemEval's official grader, on the workstation.

## Gate (the user's stratum rule, seed-averaged)

On the 30 preference rows, against the three-seed base
(`m57_bonsai_premise_s1,m57_pref_s2,m57_pref_s3`):
1. the strict paired difference has a 95% CI excluding zero, *and*
2. the official difference is positive.

**Which switch enters the bundle:** M78 replaces M77c only if it passes the
gate *and* its seed-averaged official preference score beats M77c's. The
head-to-head is also paired: `m78_answer_s{1,2,3}` against
`m77c_clause_s{1,2,3}`.

## Predictions

- The smoke test passes.
- Declines fall to ≤ 2 per seed, from 6.3–7.0 under the clause.
- Official 10.7 → 18–21 (M77c: 16.3); strict 8.7 → 14–17 (M77c: 12.0).
- At least 80% of stated quotes verify.
- M78c ≈ M78 − 1: with the structure asking for quotes, the words block
  matters more than it did under the clause alone.

## Falsifiers

- **The smoke test fails:** grammar with thinking does not work on this
  server. Stop.
- **Declines stay ≥ 5 a seed:** quotes fail verification. Report the
  verification rate and the dropped quotes.
- **Declines fall and official does not rise over M77c:** the forced answers
  are wrong or generic, and structure removes declines without adding right
  answers. Report the rows that changed from a decline.
- **M78c matches M78:** the words do not matter even for quoting.

## Cost

About 100 minutes on big (two arms side by side × 3 seeds), plus about 10
minutes of judging and the official grader on the workstation (about $0.01).
