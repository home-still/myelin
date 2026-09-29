# M82 — the premise decided by an NLI model on the memories the answer cites *(user decision 2026-09-29; pre-registered before any row)*

## Why

The typed second pass over declines recovers answerable questions
repeatably: strict **+19.7, 10 wins and 0 losses** on M57's 49 answerable
declines. Three ways of guarding it against LongMemEval's unanswerable traps
have failed (`m81-cited-premise.md`):
- **M79, the reader's label:** 6 traps answered per replicate.
- **M80, a code finding shown to the reader:** 5 answered; the reader
  ignored it.
- **M81, a code word check on the cited memories:** 0 traps, but also only
  1–2 answers, and 5/303 LoCoMo person swaps passed it.

The separating signal is semantic. The user chose to add an entailment
model.

**The screen (2026-09-29, exploratory):**
- The data: M81's 31 unique commit candidates (6 traps, 25 answerable).
- The hypothesis: each question and answer rewritten as one first-person
  statement by the 9B.
- The premise: the cited memories.
- The model: Laurer et al.'s DeBERTa-v3-large (MNLI, FEVER-NLI, ANLI,
  LingNLI, WANLI; `10.1017/pan.2023.20`).

| signal | AUROC | answerable above every trap |
|---|---|---|
| P(entailment) | **0.873** | 15/25 |
| 1 − P(contradiction) | 0.867 | 8/25 |

- Every trap scored P(entailment) ≤ 0.38, and none had entailment as its
  argmax.
- The uncle, films, university and Seattle traps are near-certain
  contradictions (0.98–1.00).

## Mechanism

`commit-arm --typed-nli --nli-url http://127.0.0.1:5820` (M82;
`bench::{typed_nli_schema, prepare_typed_nli, commit_typed_nli, HttpNli}`,
with `ops/big/nli_server.py` and `ops/big/serve-nli.sh`).
1. **Schema:** `{answer, mismatch, statement, supporting}`. M79's fields keep
   their measured order. `statement` is the question and answer as one
   first-person sentence; `supporting` cites the memories.
2. **Checks in code, before the model:**
   - M79's label and answer rule;
   - valid citations;
   - **the statement keeps every content word of the question** (M81's
     stemmed matcher, applied to the statement). The reader cannot drop the
     trap's detail and have the NLI model entail a claim the question never
     made.
3. **The premise decision:** the NLI model on (cited memories → statement).
   It commits **only when entailment is the model's argmax**, which is the
   model's own decision rule, not a threshold fitted to our rows.
4. **An NLI failure is an error, not a kept decline:** a dead server must
   not read as "no commits".

Grounds:
- presupposition verification by textual entailment (Kim et al. 2021,
  "Which Linguist Invented the Lightbulb?", ACL, arXiv 2101.00391), which
  also warns that verification is "a challenging problem";
- factual-consistency checking by NLI (AlignScore, Zha et al. 2023,
  `10.18653/v1/2023.acl-long.634`).

**Honest caveats:**
- The argmax rule was checked on the same 31 candidates this measurement
  re-reads.
- In the screen the 9B wrote the statements; here the reader (Bonsai on
  LongMemEval_S, the 9B on LoCoMo) writes them.
- LoCoMo's person-swap traps, which M82's design never saw, are the
  independent check.

## Measurement (post-passes; minutes of GPU; the NLI model on big's GPU, about 1 GB)

1. **Gate:** base replicates → `m82_base_s{1,2,3}` (Bonsai).
2. **Stack:** the shipped bundle's replicates → `r5_bundle_s*_m82` (Bonsai).
3. **Transfer, LoCoMo:** `m63_locomo_base` → `m82_locomo_base` (the 9B,
   LoCoMo's shipped reader).

Graders: the strict 9B, seeded from each source, and LongMemEval's official
grader.

## Gate and ship rule (M81's)

1. **LongMemEval_S:**
   - strict CI > 0 on the 49 answerable declines, *and*
   - official Δ > 0, *and*
   - the abstention veto: seed-mean abstention not below the base's under
     either reading.
2. **LoCoMo transfer:** at most 3 of 303 adversarial declines flip.
3. **Ships** only if 1 and 2 hold *and* the stack's seed means beat the
   shipped bundle's (81.13 official, 80.80 strict) under both readings,
   without an abstention loss.

## Predictions

- False fits: 0–1 per replicate.
- Answerable commits: 8–15 per replicate.
- Strict on the 49 answerable declines: +8 to +16, CI > 0.
- LoCoMo: ≤ 3 adversarial flips.
- Stack over the shipped bundle: strict +0.8 to +2.0, official +0.4 to
  +1.6.

## Falsifiers

- **False fits ≥ 2 per replicate:** the reader-written statements soften the
  trap, so NLI entails them. Report the statements.
- **Answerable commits < 5:** statements that keep every question word
  overstate the memories; NLI calls them neutral.
- **LoCoMo flips > 3:** NLI misses person swaps in conversational text.
