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

**Run.** `runs/m48_role` on the shipped M43 stack, judged with `--seed
runs/m43_dated` (87 judged, 327 reused) → `runs/m48_role_judged`. 500 rows
(68 inherited across a reader restart), 10.2 s/row while sharing the
reader's two slots with M44 R2. 375 answers byte-identical to the base.

**Headline: +0.0 exactly (95% CI [−2.2, +2.2], p = 1.00). Null. The
abstention veto fires on one row (90.0 → 86.7). `digest_role` ships off.**

| stratum | n | base | arm | delta | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- |
| **overall** | 500 | 67.8 | 67.8 | **+0.0** | [−2.2, +2.2] | 1.000 |
| answerable | 470 | 66.4 | 66.6 | +0.2 | [−2.1, +2.6] | 0.933 |
| abstention | 30 | 90.0 | 86.7 | −3.3 | [−10.0, +0.0] | 0.726 |
| gold = 1 | 170 | 82.4 | 80.6 | −1.8 | [−4.7, +0.6] | 0.245 |
| gold = 2 | 229 | 62.9 | 63.3 | +0.4 | [−3.1, +3.9] | 0.899 |
| gold ≥ 3 | 71 | 39.4 | 43.7 | +4.2 | [−4.2, +14.1] | 0.446 |
| `multi-session` | 133 | 59.4 | 63.2 | +3.8 | [−0.8, +9.0] | 0.162 |
| `temporal-reasoning` | 133 | 48.1 | 47.4 | −0.8 | [−6.8, +5.3] | 0.899 |
| `knowledge-update` | 78 | 80.8 | 78.2 | −2.6 | [−6.4, +0.0] | 0.261 |

### What the predictions did

| run | rows with a note | lines | lines/note | negations |
| --- | --- | --- | --- | --- |
| M43 A, dated (the base) | 423 | 2,193 | 5.2 | 266 (12.1%) |
| M43 B, boolean filter | 178 | 548 | 3.0 | 3 (0.5%) |
| **M48, three-way role** | **440** | 2,004 | 4.6 | **133 (6.6%)** |

1. **The note survived** — the prediction the arm was built on. 32 rows
   lost their note (M43's boolean lost 245), 49 gained one, 440 carry one.
   `context` gave "related but not the answer" somewhere to go.
2. **Negations halved, not vanished**: 12.1% → 6.6%. A `context` line is
   allowed to be a sentence, and some of those sentences are still "the
   assistant did not mention …".
3. **The rows whose base note carried a negation did not gain**: −1.6
   [−6.2, +3.1] over 129 rows. This is the prediction that mattered and it
   failed.
4. Splits: note lost −9.4 [−21.9, +0.0] over 32 rows (as in M43, losing the
   note costs); note kept +1.0 [−1.5, +3.6] over 391.

### The falsifier fired

*If the note survives and the headline still does not move, the negations
were inert decoration all along.* It survived; nothing moved; the 129
negation-bearing rows moved −1.6. **M42's reading of its ten rows was
pattern-matching on anecdotes**: a note saying *no information about X*
beside two useful facts was not what made the reader decline. The −0.9 /
+4.1 split M43 started from was baseline difference between the groups,
as its own caveat allowed. The digest's remaining shortfall is somewhere
other than its negations, and the two label arms (M43 B, M48) are closed.

### One more thing the run measured

Only 375 of 500 answers were byte-identical, and the 28 rows with no note
in *either* run — where the evidence should be identical — moved +3.6
[+0.0, +10.7]. This arm shared the reader's two slots with M44 R2 for its
whole length, and llama.cpp batches concurrent slots together: the same
greedy request can decode a different token when its batch-mate changes.
The control is exact only when the arm runs alone, or when the rows the
mechanism did not touch are taken from the base by construction (M42's
`commit-arm`). Recorded under operational debt.
