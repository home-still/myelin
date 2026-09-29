# M81 — the premise decided in code, on the memories the answer cites *(user decision 2026-09-29; pre-registered before any row)*

## Why

The typed second pass over declines recovers answerable questions, and it
does so repeatably: strict **+19.7 [+9.5, +31.3], 10 wins and 0 losses** on
M57's 49 answerable declines, in both M79 and M80. It cannot ship, because
the same pass answers **5–6 unanswerable traps** per replicate, and the
abstention veto fires:
- **M79:** the reader's own premise label called each trap a "detail
  unstated".
- **M80:** a code-computed finding of the question's words absent from all
  memories did not help, for two reasons:
  - the reader ignored a correct finding ("uncle");
  - the trap words ("chili", "films") do occur elsewhere in the evidence,
    just not in the memory the answer comes from.

So the reader must not vote on the premise, and the check must look at the
memories the answer rests on.

## Mechanism

`commit-arm --typed-cited` (M81; `bench::{typed_cited_schema,
accept_typed_cited, uncited_terms, commit_typed_cited}`).
1. **The schema is M79's plus citations:** `{answer, mismatch,
   supporting: [1–4 memory indices]}`. The first two fields keep M79's
   measured order, and the citations come last.
2. **The commit rule is M79's** (a non-empty answer that isn't a decline,
   with `mismatch` = `none` or `detail unstated`), **and code decides the
   premise:** every content word of the question must appear in the cited
   memories.
   - "Content words" means three letters or more, or a number, minus 44
     function words.
   - They are matched on Snowball English stems (Porter 1980,
     `10.1108/eb046814`), so "bake" meets "baked".
   - The reader's label cannot overrule the check.
3. **Grounds:**
   - M71b's `accept_grounded` check, applied to the question's own words
     instead of a list the reader names;
   - ALCE's in-code citation check (`10.18653/v1/2023.emnlp-main.398`);
   - a lexical proxy for Sufficient Context's question (Joren et al.,
     arXiv 2411.06037): does the context the answer rests on cover what is
     asked?
4. **Nothing else changes.** M71b's plural-only matcher and M80's
   all-memories test stay as measured.

**Honest caveat:** as with M80, the design came after seeing M79's traps.
The rule is generic, not fitted to those rows, but the LongMemEval_S gate is
measured on the same questions. Hence the transfer check below, on LoCoMo,
whose traps M81's design never saw.

## Measurement (post-passes; minutes of GPU)

1. **Gate:** the pass over the base replicates' declines (`m57_bonsai_premise_s1`,
   `r5_base_s2`, `r5_base_s3`), writing `m81_base_s{1,2,3}`, served by Bonsai.
2. **Stack:** the pass over the shipped bundle's replicates
   (`r5_bundle_s*_grounded`), writing `r5_bundle_s*_m81`, served by Bonsai.
3. **Transfer, on LoCoMo:** the pass over the shipped LoCoMo base
   (`m63_locomo_base`: 303 adversarial declines, 116 answerable), writing
   `m81_locomo_base`, served by LoCoMo's own shipped reader, the 9B.

Graders: the strict 9B (preference rubric), seeded from each source, and
LongMemEval's official grader for the LongMemEval_S runs.

## Gate, and what ships

1. **LongMemEval_S stratum (M79's gate):**
   - strict CI > 0 on the 49 answerable declines, *and*
   - official Δ > 0 there, *and*
   - **the abstention veto:** seed-mean abstention not below the base's
     under either reading.
2. **LoCoMo transfer:** at most **3 of the 303 adversarial declines**
   (1%) turn into answers.
3. **Ships (joins the shipped LongMemEval_S configuration)** only if 1 and 2
   hold *and* the stack's seed means beat the shipped bundle's (81.13
   official, 80.80 strict) under both readings, with abstention not below
   the shipped bundle's.

## Predictions

- False fits (LongMemEval_S abstention commits): **≤ 1 per replicate**,
  against 5–6.
- Answerable commits: **6–14 per replicate**, fewer than M79's ~20; the
  cited-word check refuses paraphrases.
- Strict on the 49 answerable declines: +6 to +14, CI > 0.
- Stack over the shipped bundle: strict +0.8 to +2.0, official +0.4 to +1.6.
- LoCoMo: ≤ 3 adversarial flips.

## Falsifiers

- **False fits ≥ 3 per replicate:** the cited memories contain the trap
  words too, and the check does not separate traps from answers.
- **Answerable commits < 5 per replicate:** the check is too strict for
  real questions; paraphrase defeats word matching even with stems.
- **LoCoMo adversarial flips > 3:** LoCoMo's person-swap traps pass a word
  check, because both names appear in the cited turns.
