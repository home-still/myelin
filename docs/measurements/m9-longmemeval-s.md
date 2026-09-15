# M9 — LongMemEval_S

500 questions, each with its own ~122k-token haystack, ingested as **500 independent tenants**.
`recall` at `k = 6`, deterministic scorer, no LLM judge.

## Ingest

| | |
|---|---|
| turns | 246,918 |
| episodes stored | **162,254** |
| approx tokens in | 61,126,168 |
| wall | 2,311.5 s (38.5 min) |
| throughput | 70.2 episodes/s, **26,444 input tokens/s** |
| reconcile | clean |

Extraction off, as for LME-V2. Consistent with the 24,289 tok/s measured in
`m3-write-path.md` on a different corpus.

## Result

| metric | value |
|---|---|
| token F1, answerable (n=470) | **0.4369** |
| token F1, answerable excluding preference (n=440) | **0.4636** |
| exact match | 0.3489 |
| abstention accuracy (`_abs` items, n=30) | **0.9667** |
| `memory_query` p50 / avg | 0.41 s / 0.42 s |

**G2 gate: LongMemEval_S ≥ 80.80 → FAIL at 43.69**, short by 37.11 points.

| question type | n | mean |
|---|---|---|
| single-session-user | 70 | **0.9114** |
| single-session-assistant | 56 | 0.7927 |
| knowledge-update | 78 | 0.6386 |
| multi-session | 133 | 0.3420 |
| temporal-reasoning | 133 | 0.2218 |
| single-session-preference | 30 | 0.0461 — *see below, not scorable this way* |

## The shape is consistent across both corpora

Retrieval is strong when the answer sits in one session — 0.91 and 0.79 — and weak when it must be
assembled or dated:

| capability | LongMemEval_S | LoCoMo |
|---|---|---|
| single-hop | 0.91 | 0.43 |
| multi-hop / multi-session | 0.34 | 0.20 |
| temporal | 0.22 | 0.28 |

Two independent corpora agreeing that multi-hop and temporal are the weak axes is a real finding
about the system, not a quirk of one dataset. Our read path is a single round of hybrid fusion:
a question needing two joined facts retrieves one of them. `investigate` exists to fix exactly
that and its step curve is in `m7-step-value-curve.md`.

## `single-session-preference` is not scorable by token F1, and 0.0461 is not a system result

Its gold answers are **rubrics**, not answers — mean gold length **390 characters** against **30**
for every other type. For example:

> **Gold:** "The user would prefer responses that suggest resources specifically tailored to Adobe
> Premiere Pro, especially …"
>
> **Ours:** "Adobe Premiere Pro's official documentation, YouTube channels like Premiere Bro and
> Video Copilot, and online …"

That answer satisfies the rubric and scores 0.133. LongMemEval judges this type with an LLM
against the rubric, which is the only way it can work. Token overlap between a short answer and a
description-of-a-good-answer measures length mismatch.

So the honest headline is **two numbers**: 0.4369 over all answerable items, and **0.4636**
excluding the 30 items our scorer cannot evaluate. Neither clears 80.80; the distinction matters
for reading the per-type table, not for the gate.

## Abstention is 96.67% here, 22.2% on LongMemEval-V2

Third corpus, same pattern as `m9-locomo.md`: where the reader prompt states the abstention rule,
abstention is near-perfect (96.67% here, 69.96% on LoCoMo); where the vendored prompt does not, it
is 22.2%. Same reader, same store, same code. Abstention is a property of the harness prompt, and
a system's abstention score is not portable between benchmarks.

## Scope of the claim

Not protocol-identical LongMemEval_S numbers. The official protocol uses a GPT-4o judge with
type-specific prompts; we score deterministically, which is reproducible forever and — per
`m9-judge-panel.md`, where our local judge came out *harsher* than a frontier one at κ = 0.8813 —
the conservative direction. The 80.80 target is MemPro-15's Qwen3-30B-A3B row, measured under
their judge, so the comparison is indicative of the gap's size and not a like-for-like score.

Corpus pinned at sha256 `08d8dad4…7894`, 278,025,796 bytes.

## Two robustness bugs this run found

Both were real defects that only a long, unusual corpus surfaces, and both are fixed with tests:

1. **Embedder died on long input.** A 12,240-character turn of newline-separated phrases tokenizes
   to over 8192 — under 1.5 chars/token against our chars/4 estimate — and ollama returns HTTP 400
   rather than truncating. Killed a 44-minute ingest at question 490 of 500. `RemoteEmbedder` now
   splits over-long texts and mean-pools the chunks.
2. **Reranker died on long input.** `input (9771 tokens) is too large to process (current batch
   size: 8192)`. `CrossEncoder` now truncates each document for scoring — legitimate here, unlike
   the embedder, because the score only decides ordering and the reader still gets the full text.
