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

## Result — the gate fails, and M77c stays the bundle's advice switch *(measured 2026-09-28)*

The runs are `runs/m78_answer_s{1,2,3}` and `runs/m78c_answer_s{1,2,3}`, on
big at main `cc6844e` (unit `myelin-m78`). Each is 29 fired rows per seed,
with the other 471 copied from M57. They were graded by the strict 9B with
the preference rubric, and by LongMemEval's official grader.

**The smoke test passed.** It produced 1,105 characters of thinking, then
constrained JSON that parsed. The fork applies the grammar after the
reasoning block. Across all six runs no content failed to parse (0 of 174
rows).

**Per seed, on the 30 preference rows** (declines / official right / strict
right):

| arm | seed 1 | seed 2 | seed 3 | seed mean |
|---|---|---|---|---|
| base (M57) | 7 / 13 / 10 | 10 / 10 / 8 | 8 / 9 / 8 | 8.3 / 10.7 / 8.7 |
| M77c (clause) | 7 / 17 / 12 | 6 / 18 / 14 | 8 / 14 / 10 | 7.0 / **16.3** / 12.0 |
| **M78** (words + clause + structure) | 9 / 16 / 14 | 8 / 13 / 9 | 7 / 18 / 14 | 8.0 / 15.7 / 12.3 |
| M78c (clause + structure) | 11 / 14 / 11 | 14 / 11 / 10 | 11 / 16 / 16 | 12.0 / 13.7 / 12.3 |

**Paired and seed-averaged**, on the 30 preference rows:

| comparison | strict Δ [95% CI] | official Δ [95% CI] |
|---|---|---|
| M78 vs base | +12.2 [−1.1, +25.6] | +16.7 [+0.0, +33.3] |
| M78c vs base | +12.2 [−3.3, +27.8] | +10.0 [−5.6, +25.6] |
| **M78 vs M77c** (head-to-head) | +1.1 [−11.1, +13.3] | **−2.2 [−15.6, +11.1]** |

- Abstention is unchanged in every arm (no gate fires on an abstention row).
- **Verdict: criterion 1 fails** (the strict CI crosses zero). M78 also does
  not beat M77c. Under the pre-registered rule it enters no bundle, and the
  round-5 bundle carries M77c (`r5-bundle-seeds.md`).

**Predictions against the result:**

| prediction | result | |
|---|---|---|
| smoke test passes | passed | held |
| declines ≤ 2 per seed | 8.0 (M78), 12.0 (M78c) | wrong |
| official 18–21 | 15.7 | wrong |
| strict 14–17 | 12.3 | wrong |
| ≥ 80% of quotes verify | 64% (M78), 40% (M78c) | wrong |
| M78c ≈ M78 − 1 | official −2.0, strict equal | held |

**The falsifier "declines stay ≥ 5 a seed" fires.** The structure removed the
reader's own decline, but its quotes then failed verification. Every quote
across the three seeds, by where the verifier found it:

| where the quote was found | M78 (245 quotes) | M78c (221) |
|---|---|---|
| a user turn of the cited memory: **kept** | 156 (64%) | 89 (40%) |
| the cited memory, outside a user turn (the assistant's advice) | 43 (18%) | 87 (39%) |
| nowhere verbatim (paraphrased or cut at the length cap) | 34 (14%) | 28 (13%) |
| another memory than the one cited | 9 (4%) | 13 (6%) |
| user fragments joined by "…" (rejected, pre-registered) | 3 (1%) | 4 (2%) |

**What this teaches:**
- **Asked for the user's preferences, this reader quotes the assistant.**
  About a fifth of M78's quotes, and two fifths of M78c's, are the
  assistant's advice presented as what the user said. Without the words
  block there are few user turns to quote, and the model fills the field
  with what is there.
- **The verifier did its job.** A "You told me" built from the assistant's
  own advice would be a false statement. The rows it refused became
  declines, which is why declines did not fall.
- **The structure is not what failed; the field did.** A free-text quote asks
  the model to find and copy the user's words, and it cannot tell whose
  words they are. The structural next step is to make the *choice* the
  constraint: the answer selects user turns by index from a list that holds
  only user turns. This is the same shape as the selector's `{keep: [int]}`,
  which this reader follows (M40). It leaves no way to quote the assistant,
  and nothing to verify.
