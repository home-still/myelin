# M48 — a boolean is the wrong label

## What M43 measured

`digest_relevance` asked the digest one bit per memory — `bears_on_question`
— and dropped the line on `false`. It did what it was aimed at: prose
negations fell 11.3% → 0.5%. It cost **−4.40 (95% CI [−7.2, −1.6])** over
the dated digest, and the split said why:

| rows | n | A (dated) | B (filtered) | delta |
| --- | --- | --- | --- | --- |
| note lost to the filter | 245 | 63.7 | 55.5 | **−8.2** |
| note kept in both | 178 | 70.8 | 70.2 | −0.6 |
| no note in either | 70 | 74.3 | 74.3 | **+0.0** |

A memory that supplies *context* — the same people, events or things, but
not the answer — is not `true` under a boolean about the answer, so it was
marked `false`, its line dropped, and with enough lines gone the note fell
under `MIN_STEPS_EMITTED` and vanished. **All of the −4.4 is the 245 rows
that lost their note.** Filtering negations was not what hurt; losing the
note was.

## The literature

Chain-of-Note (Yu et al., `2311.09210`) is `item_digest` under another
provenance, and its gains land on exactly our two failure shapes: **+7.9 EM
under entirely noisy retrieval** (34.28 → 41.83) and **+10.5 rejection
rate**. CoN types each note three ways — the document *answers* the
question, is *useful context* that does not, or is *irrelevant* — and M43's
boolean collapsed the first two. CoN's cost warning also transfers: their
inference went 0.61 s → 12.02 s per query (19.7×) because notes are per-item
calls. Ours is one call and stays one.

## The mechanism

`InvestigateConfig::digest_role` (default **off**): the same forced digest
(`minItems == maxItems == n`, M40), the same field order (`says` written
before the judgement, M42/M43), and one enum instead of a bool:

```json
{ "index": 2, "says": "…", "role": "answers" | "context" | "irrelevant" }
```

Only `irrelevant` loses its line. `context` — the label the boolean had no
room for — keeps it. The prompt says what each role is for and, as M43's
did, never asks the model to write that a memory lacks something.

`digest_relevance` and `digest_role` are alternative arms: `digest_label`
refuses both, the CLI marks them `conflicts_with`, and the MCP tool returns
`invalid_params`. A precedence rule would have made one of them silently
inert.

## Pre-registration

Written before the arm ran.

**Base.** The shipped operating point, `runs/m43_dated_judged` (67.80): the
dated digest, no label.

**Arm.** The base command plus `--digest-role` → `runs/m48_role`, judged
with `--seed runs/m43_dated`.

**Population.** All 500 LongMemEval_S rows.

**Primary metric.** `longmemeval_s.judge_score.n500`, paired bootstrap.

**Decision rule.** Ship on at **≥ +3.0** with a paired 95% CI excluding
zero; **veto** on any drop across the 30 abstention rows.

**Predicted, specifically.**

1. The note survives: rows carrying a note fall from A's 423 by far fewer
   than B's 245. Fewer than 60 rows lose their note.
2. Negations stay near B's 0.5%, not A's 8.0%.
3. On the 113 rows whose M40 note carried a negation, the arm gains; on the
   rest it moves ~0. (Descriptive, as in M43: conditioned on the mechanism's
   own output.)
4. Rows with no note in either arm move **exactly +0.0**.
5. Abstention does not fall.

**Falsifier.** If the note survives (prediction 1 holds) and the headline
still does not move, the negations were inert decoration all along — M42's
reading of those ten rows was pattern-matching — and the digest's remaining
shortfall is somewhere else entirely.

**Cost.** One ~1-hour run; one call per query, unchanged.

## Results

*(pending — queued behind M46 and M44 R2 on the reader)*
