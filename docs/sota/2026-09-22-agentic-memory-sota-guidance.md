# Pushing myelin toward SOTA after M43 — literature-grounded guidance

**File:** `2026-09-22-agentic-memory-sota-guidance.md`
**Audience:** the coding agent working `PLAN.md` §15 onward.
**Method:** read the repo state through M43, then used home-still (`distill_search`, `paper_search`,
`paper_download`) to pull the literature that bears on the *post-M38* diagnosis. 22 papers were added
to the home-still catalog today (§9). Scribe is down on `big` (the same GPU-tenancy problem you are
fighting), so those PDFs are catalogued but **not yet converted or indexed**; the numbers below were
read from the arXiv/Nature full texts and should be re-verified against `markdown_read` once scribe
converts them.

The brief in `docs/sota/research-brief.md` still holds. This document does not restate it; it
answers the question M38–M43 opened: *retrieval is solved on LongMemEval_S, the reader fails with the
gold in hand, it ignores instructions and obeys structure, the digest and the forced commit are both
real but sub-bar — what does the literature say to do next?*

---

## 0. The verdict, in order of expected answers ÷ cost

1. **The reader is being run in a mode no published comparator uses, and the LME-V2 comparison
   you treat as "same reader" is not the same reader configuration.** Thinking is off on every call
   (`with_thinking(true)` is never called in production code), temperature is 0, the reader is capped
   at 160 completion tokens and told "Do not explain." The official LME-V2 harness defaults to a
   *thinking* reader with a 20,000-token completion budget, and AgentRunbook-R's controller also
   thinks by default. Every diagnosis since M38 — the 2-fact collapse, instruction-ignoring,
   byte-identical selections, false declines — is the textbook behaviour of a small model denied
   any reasoning tokens. This is one read-path arm and it is the highest-expected-value unmeasured
   variable in the project. §1.
2. **Make `commit_answer` earn its answer with agreement, not a second prompt.** M42 showed the
   forced commit is worth +35.5 on the rows it changes and loses 2/30 abstention rows. The
   selective-prediction literature has a training-free discriminator that works at 7B: sample N
   answers, cluster by meaning, commit only when the majority cluster clears a *calibrated*
   threshold. Semantic entropy is stable at 0.78–0.81 AUROC from 7B to 70B; conformal abstention
   turns the threshold into an error-rate guarantee with ~100 calibration rows. §2.
3. **LME-V2 premise awareness: verify presuppositions, decline only on contradiction.** M35's
   `premise_analysis` was anti-selective because it fired on *unsupported*, not *false*. The
   presupposition-verification framework (generate → verify → explain) separates the two, and LME-V2
   itself says the memory module must flag wrong premises for abstention to move. Schema-forced,
   per-presupposition, `contradicted` ≠ `absent`. §3.
4. **Temporal duration questions: do the date arithmetic in `compose`, not in the reader.** Even
   GPT-4 scores 16% on duration arithmetic; a 9B will not do "how many days between" from two
   stamps. The `[timeline]` view already has the stamps; emit the deltas. Zero model calls. §4.
5. **The digest is Chain-of-Note; use its three-way label, not a boolean.** M43's
   `bears_on_question` is the right structural move; CoN's evidence says the useful middle label is
   "context, not answer." §5.
6. Write-path leads from the last ten days of arXiv (JustMem, REALM, CueMem) converge on one idea
   you already half-have: **retrieve compact, then replay the source turn for fidelity-sensitive
   questions.** One `build`, not three. §6.
7. Operational: own your Qdrant, quantize the KV cache, and run thinking arms in the window the voice
   assistant is idle. §8.

---

## 1. The reader mode is the unmeasured variable

### 1.1 What the code does

| where | setting | source |
|---|---|---|
| `llm/mod.rs` `CompletionRequest::new` | `thinking: false`, `temperature: 0.0` | doc comment: "Default off, and that is load-bearing … `investigate` (§7.2) turns it back on" — **but no production call site ever does**; `grep with_thinking(` finds only the unit test |
| `ops/big/serve-models.sh` | `TEMPLATE_KWARGS='{"enable_thinking":false}'` server-wide | comment: to protect the LME-V2 evaluator path |
| `myelin-eval/src/bench.rs` reader | `.with_max_tokens(160)` | three call sites |
| `READER_SYSTEM` | "Answer in as few words as possible … Do not explain." | direct answering, no chain of thought |
| `adapters/run_myelin.py` | `--reader-enable-thinking` default **False** | comment: "spends the whole completion budget on `reasoning_content`" |

### 1.2 What the benchmark protocol does

`vendor/longmemeval-v2/evaluation/run_eval.py:81` — `--reader-enable-thinking … default=True`.
`evaluation/harness.py:186` — `parser.set_defaults(reader_enable_thinking=True)`; the reader's
`max_completion_tokens` default is 20,000; `--controller-disable-thinking` defaults to False, so
AgentRunbook-R's Qwen3.5-9B controller *also* thinks. The paper's reader sampling (temp 0.6, top_p 0.95,
top_k 20 — `docs/research/11-frontier-2026.md` §1.4) is exactly the Qwen thinking-mode
recommendation (Qwen3 Technical Report, `10.48550/arxiv.2505.09388`: thinking mode "temperature of
0.6, a top-p value of 0.95, and a top-k value of 20"; non-thinking "temperature = 0.7, top-p = 0.8,
top-k = 20, presence penalty = 1.5"). The Qwen3.5-9B model card states thinking is **on by default**
and is disabled only via `chat_template_kwargs: {"enable_thinking": false}`.

So: AgentRunbook-R 58.6 and RAG-slice+notes 51.0 were produced by a *thinking* Qwen3.5-9B with a
20k budget. myelin's 38.58 was produced by the same weights with thinking off, temperature 0 and (on
the LongMemEval_S path) 160 tokens. The registry row "AgentRunbook-R, same reader" and the LAFS
reference frontier are therefore not same-reader comparisons, and the −20.02 gap has an unmeasured
reader-mode component. The `run_myelin.py` default silently deviates from the vendored protocol.

Why it was turned off is documented and real: a two-line *extraction* call burned its 512-token
budget on `reasoning_content`. That is the write path. It does not follow that the *reader* should
be denied reasoning — and the harness gives the reader 20,000 tokens for a reason.

### 1.3 What the literature says this costs

- **Compositionality.** Press et al. (`10.48550/arxiv.2210.03350`, already in your corpus) name the
  quantity M39 measured and their remedy is reasoning *in the reader* (self-ask). M39/M40 moved the
  reasoning into the memory layer because the reader was forbidden to reason. M40's +6.0 on gold=2
  is the size of a partial workaround, not the size of the effect.
- **Format restriction kills direct-answer reasoning.** Tam et al., *Let Me Speak Freely?*
  (`10.18653/v1/2024.emnlp-industry.91`, indexed): "100% of GPT 3.5 Turbo JSON-mode responses placed
  the 'answer' key before the 'reason' key, resulting in zero-shot direct answering instead of
  zero-shot chain-of-thought reasoning"; LLaMA-3-8B shows "a substantial 38.15% performance gap" on
  Last Letter under JSON with a parse-error rate of 0.148%. Their prescription is the one you already
  discovered empirically in M42 — **field order is the mechanism**: a `reasoning` field emitted
  *before* `answer` preserves chain-of-thought under a strict schema. Your `READER_SYSTEM` is the
  "answer key first" failure mode with no reason key at all.
- **A reference point for the same reader class without any memory system.** HeadWiseKV
  (`10.48550/arXiv.2609.02029`, in your sweep) runs "Qwen3.5-9B Q4_K_M on the cleaned 500-question
  LongMemEval-S set … with thinking disabled, temperature 0, and seed 42", *full history in context*
  (inputs over 122,880 tokens truncated), and reports **70.2% reviewed accuracy for Full-KV** under a
  DeepSeek V4 Pro binary judge with audited labels. Different judge, audited labels, Q4_K_M — but it
  is the same reader class with *no retrieval at all* scoring ~8 points above your 62.0 with thinking
  still off. Read it as an upper bound on how much the reader has left when it sees everything.
- **Thinking helps abstention specifically.** LME-V2 §D.1 (`11-frontier-2026.md`): AgentRunbook-R
  does not improve abstention because it "presents evidence that misleads reader into using it
  instead of rejecting"; the reader has to *reason about* the premise. M35 measured your abstention
  stratum at 17.97%. A non-thinking reader told to answer in as few words as possible is the
  worst-case configuration for premise rejection.

### 1.4 The arm, pre-registered in your house style

**Arms** (LongMemEval_S, all 500, `investigate`, `max_steps 2`, `select`, k=6, 4096 — the operating
point that scored 62.00):

- **R1 — structured reasoning, thinking off.** Reader schema
  `{ "reasoning": string(≤600 chars), "answer": string, "evidence_absent": bool }` in that field
  order (M42's rule: the answer is written before the model may disown it; here the reasoning is
  written before the answer). `max_tokens` raised to fit. Temperature 0 stays. This isolates "let it
  reason" from "sample".
- **R2 — thinking on.** `enable_thinking: true`, thinking budget capped (Qwen3's "thinking budget"
  mechanism: halt the trace at N tokens and force the answer; N = 1,024 first), temp 0.6 / top_p 0.95
  / top_k 20 per the Qwen thinking recommendation, fixed seed, and the same `\boxed{}` / schema
  final-answer contract. Report two seeds so the CI carries sampling noise.

**Population.** All 500. Report by gold-fact count (M39's split) and by `question_type`.

**Predicted, specifically.** gold=2 (n=217, 56.7%) and gold≥3 (n=31, 35.5%) move; gold=1 (n=169,
79.9%) does not regress by more than 1.0; `temporal-reasoning` and `multi-session` carry the gain;
abstention accuracy on the 30 `_abs` rows does **not** fall (the veto stands). If R2 ≫ R1, the gain
is deliberation, not format; if R1 ≈ R2, a bounded structured-reasoning field is the cheap ship.

**Decision rule.** Ship on ≥ +3.0 over 500 with a paired CI excluding zero and the abstention veto
intact. The bar is the same as always; the prior for this one is higher than for anything since M32.

**Falsifier.** If neither arm moves gold=2, the reader is not compute-limited and M38's "reading is
the gap" theory is refuted at the cheapest possible point — which redirects the project to write-time
aggregation (§6) with much more confidence than today.

**Cost.** Thinking traces at 9B on a shared 3090 will be 10–40 s/question; 500 rows ≈ 2–6 GPU-hours,
i.e. a *read-path* arm that costs like a *write-path* arm. §8 says how to fit it in the tenancy you
have. R1 is minutes.

**LME-V2 corollary.** Re-run the shipped LME-V2 operating point with `--reader-enable-thinking` to
match `run_eval.py`'s default. That is not a mechanism arm; it is the *comparable* number, and the
`standing` row against AgentRunbook-R should carry a `caveat-reader-mode` verdict until it exists.
M16's "perfect reader over today's evidence tops out at 38.8–44.6" bound was computed on the M16-era
store with a non-thinking sufficiency judge; do not let it pre-empt this run. The strongest expected
effect on LME-V2 is on the 128 abstention rows, which that bound does not cover.

---

## 2. `commit_answer` v2: consensus-gated commit

### 2.1 What M42 established

Forced commit fires on 91 rows, commits on 31, and those 31 go 6.5 → 41.9. It ships off because 2 of
30 adversarial rows were talked out of refusing. The mechanism needs a discriminator between "the
model declined out of habit" and "the model declined because the premise is not in memory." The
falsifier in M42 was refuted — the declines were *not* well calibrated — so a discriminator has
headroom.

### 2.2 The literature's discriminator is agreement, and it works at 7B

- **Semantic entropy** — Farquhar et al., *Nature* 2024 (`10.1038/s41586-024-07421-0`). Sample
  "ten generations" at temperature 1.0, cluster by bidirectional entailment, take entropy over
  clusters. AUROC for hallucination detection averaged **0.790** vs naive entropy 0.691, P(True)
  0.698, embedding regression 0.687; "stable performance (between 0.78 and 0.81 AUROC) across the
  different model families (LLaMA, Falcon and Mistral) and scales (from 7B to 70B parameters)." The
  **discrete** variant "approximates P(Ci|x) directly from the number of generations in each
  cluster, disregarding the token probabilities" and "performs similarly well" — no logprobs needed,
  which matters through llama.cpp's OpenAI shim.
- **Conformal abstention** — Yadkori et al. (`10.48550/arxiv.2405.01563`). Same idea, with a
  guarantee: score = match count among k=10 samples (similarity judged by the model itself), threshold
  set by conformal risk control `E[R(λ̂n)] ≤ α`; calibration sets from 10 to 3,200 rows; on Temporal
  Sequences at α=0.1 the match-count score bounds test error at 0.073 ± 0.017 vs 0.084 ± 0.010 for
  log-prob, and the paper's finding is that match counts "perform well for both short and long answers,
  while the log-probability score is significantly worse for questions with long answers."
- **Abstain on the uncertain tail** — Tomani et al. (`10.48550/arxiv.2404.10960`): "By sacrificing
  only a few highly uncertain samples we can improve correctness by 2% to 8%, avoid 50%
  hallucinations via correctly identifying unanswerable questions."
- **P(True)** — Kadavath et al. (`10.48550/arxiv.2207.05221`): ask the model whether a proposed answer
  is true; calibration improves when the model "examine[s] multiple samples before judging any single
  response's validity." Semantic entropy beats it in the Nature comparison, but it is one call and
  fits your schema habit as a `p_true` boolean after `answer`.

None of this is instruction. It is structure plus sampling, which is the only lever this reader has
ever obeyed.

### 2.3 The mechanism

`InvestigateConfig::commit_consensus`. When the first response declines (the M42 trigger):

1. Sample **N = 5** candidates under M42's schema (`answer` before `evidence_absent`), temperature
   0.6, same evidence, same prompt. llama.cpp serves `n` samples off one prefill, so this is
   decode-only cost on ~20% of rows.
2. Normalize and cluster answers. Exact/normalized string match first; for free-text answers, a
   bidirectional-entailment check. Farquhar et al. used a DeBERTa-large MNLI model for this at
   paragraph length; one fits in ~0.4 GB of the VRAM budget. Whether `bge-reranker-v2-m3` (already
   resident) is an adequate "A ≈ B given Q" proxy is unmeasured — a 50-pair spot check against the
   DeBERTa verdicts settles it. The 9B with a boolean schema `{same_answer: bool}` is the fallback.
3. Commit the majority-cluster answer **only if** its share ≥ τ *and* no sample asserted
   `evidence_absent`. Otherwise the original decline stands byte-identical (M42's invariant).
4. τ is **calibrated, not tuned**: conformal risk control on a calibration split that is *not* the
   reported population. The 30 `_abs` rows are too few; LME-V2's 128 abstention rows (web +
   enterprise) are the calibration set for an LME-V2 arm, cross-validated by domain; for LongMemEval_S
   pre-register τ from that calibration and report it.

**Predicted.** Commits fall from 31 to roughly 15–20 of the 91 declines; accuracy on committed rows
rises above M42's 41.9; the two adversarial rows that flipped in M42 do not flip (an adversarial
premise produces disagreement, which is the whole point). Non-firing rows +0.0 exactly.

**Veto.** Unchanged: any drop on the 30 abstention rows ships it off.

**What this also buys on LME-V2.** The same discriminator is an abstention *score* for every row,
not only declines. M36 found no recorded signal discriminates abstention (precision 32.8%, lift
1.16×); agreement across samples is a signal you have not recorded.

---

## 3. LME-V2 premise awareness: verify presuppositions, decline only on contradiction

### 3.1 Why M35's gate was anti-selective

`premise_analysis` tripled declines on answerable rows (8.3% → 26.8%) and moved abstention rows only
1.3×. It asked "is the premise supported?" and a 9B answers "no" whenever the store is merely silent.
LME-V2's abstention rows are *wrong-premise* questions ("assumptions valid in another environment but
wrong in the current one"); silence is not evidence of a wrong premise, contradiction is.

### 3.2 The framework

Kim et al., *Presupposition Verification for Question-Answering* (`10.48550/arxiv.2101.00391`):
~21% of Natural Questions' unanswerable questions "can be explained based on the presence of
unverifiable presuppositions"; the pipeline is presupposition **generation** (from linguistic
triggers: wh-words, definite descriptions, factives, possessives, temporal adjuncts), **verification**
against the source, and **explanation**. Their own bottleneck was verification quality ("even
transfer from the best entailment models currently falls short" — 0.79 accuracy vs 0.78 majority).
Kim et al., *(QA)²* (`10.48550/arxiv.2212.10003`) and Hu et al., *FalseQA*
(`10.48550/arxiv.2307.02394`) establish that models *hold* the knowledge to rebut false premises but
need the rebuttal step activated — FalseQA gets there with 256 fine-tuning examples; you cannot train,
so the activation has to be structural.

### 3.3 The mechanism

One schema-forced call before the reader, memory-side (M19's rule: do the computation in memory):

```json
{ "presuppositions": [
    { "claim": "…", "status": "supported" | "contradicted" | "absent", "evidence_index": int|null }
  ] }
```

`minItems ≥ 1` so the model must enumerate (M40's forcing); `status` **after** `claim` (M42/M43's
field-order rule). Then:

- any `contradicted` with an `evidence_index` → prepend one `[premise]` line naming the contradiction
  (this is what AgentRunbook-C's memory module does, and it is the documented reason its abstention
  is better than -R's);
- `absent` → **nothing is emitted**. Silence is not a decline signal. This is the one-line difference
  from M35.

**Arm.** LME-V2 web n=240 first (M35's cell), then the 451. **Predicted:** abstention stratum
+10 or more from 17.97%; answerable stratum −1.0 or better (the `absent` branch is inert by
construction, so the M35 damage cannot recur through it). **Falsifier:** if the model marks true
premises `contradicted` at a rate that costs answerable rows, verification is the bottleneck exactly
as Kim et al. found, and the next lever is the NLI model from §2.3 doing the verification instead of
the 9B.

---

## 4. Temporal duration arithmetic belongs in `compose`

`temporal-reasoning` is 133 rows at 42.1 (M42 base) and the largest error stratum after
multi-session. Part of it is composition (§1); part of it is arithmetic the reader cannot do.

*Test of Time* (Fatemi et al., `10.48550/arxiv.2406.09170`), ToT-Arithmetic: **Duration** ("computing
differences between two dates/times") is the worst category for every frontier model tested —
Claude-3-Sonnet 15.00%, GPT-4 16.00%, Gemini 1.5 Pro 13.50% — with off-by-one errors in ~21–25% of
responses and directional errors "going back in time." A 9B at temperature 0 with 160 tokens will
not beat GPT-4 at this. The same paper measures fact *ordering*: presenting the target and start
time first ("TargetAndStartTime") vs shuffled moves Claude-3-Sonnet 45.71% → 73.57% and Gemini
63.04% → 75.00%. TempReason (Tan et al., `10.48550/arxiv.2306.08952`) makes the same point at the
benchmark level — time-dependent QA needs explicit temporal span extraction.

M19 already proved the pattern on this codebase: resolving dates *for* the reader was +37.6, telling
it to resolve them +14.3. Apply it one step further:

- `ComposeConfig::timeline_deltas`: for every pair of stamped events the `[timeline]` view already
  emits, append the signed delta in days when the question contains a duration cue ("how many days /
  weeks / long before / after / between"). Deterministic, no model call, I1-safe (it is a view).
- Order the `[timeline]` view target-event-first when the question names an event, per the ToT
  ordering result.

**Arm.** Read-path, minutes. LongMemEval_S `temporal-reasoning` (133) and LoCoMo temporal (321).
**Predicted:** the "how many days" subset moves; the rest of the stratum is §1's problem and should
not move. Report the two subsets separately so a null on the arithmetic subset is legible.

---

## 5. The digest is Chain-of-Note; take its label set

Yu et al., *Chain-of-Note* (`10.48550/arxiv.2311.09210`) is `item_digest` with a different
provenance: one reading note per retrieved document, then the answer. Their gains are on exactly the
two failure shapes you have measured — noisy evidence ("+7.9 in EM score given entirely noisy
retrieved documents", 34.28 → 41.83 at 100% noise) and unknown-scenario rejection ("+10.5 in
rejection rates"). Two things transfer:

- **The three-way label.** CoN notes are typed: the document *answers* the question, the document is
  *useful context* but does not answer it, or it is *irrelevant*. M43's `bears_on_question: bool`
  collapses the first two. The M42 failure rows (a negation outvoting two facts that were also in
  the note) are "useful context" entries phrased as negations. A three-way `role: answers | context |
  none` keeps context lines (dated, per M41) while dropping only `none`, and lets the reader weight
  them. Same forcing, same field order, one enum instead of a boolean.
- **The cost warning.** CoN inference went 0.61 s → 12.02 s per query (19.7×). Your digest is one
  call, which is the right design; do not let it grow into per-item calls.

CoN was fine-tuned (LLaMA-2 7B on 10k GPT-4 notes); zero-shot CoN on GPT-4 still gained +2.6. You
are in the zero-shot regime at 9B, which is why the count forcing (M40) was necessary at all.

---

## 6. Write-path leads from the last ten days, ranked

None of these is demonstrated at ≤9B; all are frontier-backbone results. They are worth one `build`
between them, not one each.

| paper | date | backbone / judge | claim | what transfers |
|---|---|---|---|---|
| **JustMem** `10.48550/arxiv.2609.19877` | 2026-09-17 | *not read — arXiv rate-limited the full text; abstract only* | "highest mean accuracy and retrieval recall" on LoCoMo + LongMemEval-S with fewer tokens; three access modes: LOOKUP (local), COMPOSE (distributed evidence), **REPLAY** ("increases reading fidelity for fidelity-sensitive evidence" — recovers the original conversation) | Your M41 finding in one word: compressed evidence loses the recency signal. Route `knowledge-update` and `temporal` to raw source turns, everything else to compact records. `prov_source.doc` already links records to sessions, so REPLAY is a read-path view, not a re-ingest. **Read this paper first when scribe is back.** |
| **REALM** `10.48550/arxiv.2609.16053` | 2026-09-13 (already indexed) | GPT-4o-mini for everything | LoCoMo 75.97, LongMemEval 65.11 (+1.31 over Zep); reconsolidation = add/strengthen/weaken graph edge weights after each retrieval; ablation −2.13 on LongMemEval (−6.66 on preference) | Small effect, frontier backbone, and your graph channel measured as a loss in M12. Low priority; note that edge-weight events are appendable under I1 if you ever revisit the graph. |
| **CueMem** `10.48550/arxiv.2609.12354` | 2026-09-11 (in sweep) | — | cues → source-turn anchors → expand over a turn graph of temporal proximity + semantic relatedness | Same REPLAY idea with an expansion step. |
| **RD-Forget** `10.48550/arxiv.2609.10263` | 2026-09-09 | frozen curator | "separates what an agent stores from what it uses": retained archive + query-conditioned view; same-slot replacement links suppress superseded values for current-state questions, intent-aware retrieval re-admits them for historical ones | This is your `supersedes` edge used at *read* time by question intent. RQ6's cheapest 6 answers: route knowledge-update questions to newest-in-slot, everything else unfiltered. Read-path. |
| **Chronos** `10.48550/arXiv.2603.16862` | (in corpus) | frontier | events calendar = 58.9% of the gain | Still the strongest write-time lead for temporal; unknown at 9B. One `build` with SVO event tuples beside episodes is the RQ4 test. |
| **LightMem** `10.48550/arxiv.2510.18866` | 2025-10 | GPT and Qwen backbones | up to +7.7 / +29.3 QA accuracy on LongMemEval / LoCoMo with 38×/20.9× fewer tokens; topic-grouped short-term memory + sleep-time consolidation | The Qwen rows need checking for size; if a ≤9B row exists it is the only such datapoint in this table. Unconverted. |

Order of operations: §1 first (it changes the denominator every write-path arm is judged against),
then the two read-path views (REPLAY routing, RD-Forget-style supersedes routing), then one `build`.

---

## 7. What not to re-run

Everything in the brief's §4 still stands. Add from M35–M43: `typed_probes`, `premise_analysis` as
built, `select_coverage`, trajectory routing, entity-extraction repair, one-step `self_ask`. Do not
revise the adjudicator prompt (M15/M23 rule). Do not tune τ in §2 on the population you report.

---

## 8. Operational blocks, and what the research implies for them

**GPU tenancy on `big`.** The thinking arms in §1 and the N-sample commit in §2 are the two things
in this document that need more of the card than today's arms. Three levers that need no sudo:

- Quantize the KV cache: `-ctk q8_0 -ctv q8_0` (requires `-fa on`) roughly halves KV memory at
  negligible quality cost. How much that is in absolute terms at `READER_CTX 32768` × 2 slots depends
  on Qwen3.5-9B's KV layout — read it off `llama-server`'s startup log (`KV self size`) before
  counting on it.
- Size the context to the arm. LongMemEval_S arms at k=6 / 4,096 evidence tokens plus a 1,024-token
  thinking budget fit in 8k per slot; only LME-V2 needs the large window. Two server profiles.
- Use `n` (parallel samples per request) for §2 rather than N requests: one prefill, N decodes.

For the 5× throughput cliff when the voice assistant's model is resident: schedule the long arms
(thinking, LME-V2) for the window it is idle, and treat any arm whose manifest records partial
offload as `Degraded` the way `WidthVerdict` already does — a 14.5 s/row arm is a different
measurement, not a slower one.

**Someone deleted the `myelin_*` collections from the dashboard.** M41's reindex path is the right
recovery; the prevention is to stop sharing a Qdrant you do not control. Two options, neither
needing sudo: run your own Qdrant binary (or rootless container) on a separate port with
`QDRANT__SERVICE__API_KEY` and `QDRANT__SERVICE__READ_ONLY_API_KEY` set — the dashboard then requires
the key, and a read-only key is what anyone browsing should have; or, if the shared instance must
stay, ask its owner to set those two keys, which is a config change and a restart. Either way
schedule `POST /collections/{name}/snapshots` after every `build`, because reindex at 38 records/s
is 71 minutes for LongMemEval_S and hours for LME-V2 (85,979 records still rebuilding as of today).

**home-still scribe is down** (`system_status`: both scribe instances unhealthy, "backend
unavailable"; distill is healthy on CUDA). The same tenancy problem. The papers in §9 are downloaded
and catalogued; `scribe_convert` times out. Run the conversions when the GPU is free; until then
`distill_search` will not surface them.

---

## 9. Papers added to home-still today (22)

All catalogued under `papers/10/`; **none converted or indexed yet** (scribe down). Stems are the
DOI with `/` → `_`.

| DOI | title | why |
|---|---|---|
| 10.48550/arxiv.2505.09388 | Qwen3 Technical Report | thinking/non-thinking modes, sampling, thinking budget (§1) |
| 10.48550/arxiv.2609.19877 | JustMem: Just-Enough Memory Access for Long-Term Conversations | REPLAY (§6) — full text not yet read |
| 10.48550/arxiv.2609.16053 | REALM: Retrieval-Driven Memory Reconsolidation (already indexed) | §6 |
| 10.48550/arxiv.2609.10263 | What Should an Agent Forget? Separating What Is Stored from What Is Used | supersedes routing (§6) |
| 10.48550/arxiv.2203.11171 | Self-Consistency Improves Chain of Thought Reasoning | majority vote (§2) |
| 10.1038/s41586-024-07421-0 | Detecting hallucinations in LLMs using semantic entropy (Nature) | §2 |
| 10.48550/arxiv.2405.01563 | Mitigating LLM Hallucinations via Conformal Abstention | §2 |
| 10.48550/arxiv.2404.10960 | Uncertainty-Based Abstention in LLMs Improves Safety and Reduces Hallucinations | §2 |
| 10.48550/arxiv.2207.05221 | Language Models (Mostly) Know What They Know | P(True) (§2) |
| 10.48550/arxiv.2303.08896 | SelfCheckGPT | sampling-based consistency (§2) |
| 10.48550/arxiv.2407.18418 | Know Your Limits: A Survey of Abstention in LLMs | taxonomy for RQ-abstention |
| 10.48550/arxiv.2101.00391 | Which Linguist Invented the Lightbulb? Presupposition Verification for QA | §3 |
| 10.48550/arxiv.2212.10003 | (QA)²: Question Answering with Questionable Assumptions | §3 |
| 10.48550/arxiv.2307.02394 | Won't Get Fooled Again: Answering Questions with False Premises | §3 |
| 10.48550/arxiv.2311.09210 | Chain-of-Note | §5 |
| 10.48550/arxiv.2309.11495 | Chain-of-Verification | verification-step design (§2/§5) |
| 10.48550/arxiv.2212.10509 | IRCoT: Interleaving Retrieval with Chain-of-Thought | multi-hop; your M19 `investigate` null is the counter-evidence at 9B |
| 10.48550/arxiv.2211.10435 | PAL: Program-aided Language Models | compute-not-prompt (§4) |
| 10.48550/arxiv.2406.09170 | Test of Time | duration arithmetic, fact ordering (§4) |
| 10.48550/arxiv.2306.08952 | TempReason | temporal span extraction (§4) |
| 10.48550/arxiv.2408.03314 | Scaling LLM Test-Time Compute Optimally | budget allocation for §1/§2 |
| 10.48550/arxiv.2510.18866 | LightMem | Qwen-backbone rows to check (§6) |
| 10.48550/arxiv.2509.25140 | ReasoningBank | procedural/strategy memory for LME-V2 workflow + gotchas strata |

Also downloaded by mistake and safe to ignore: `10.48550/arxiv.2509.25153` (an attention theory paper;
wrong ID for ReasoningBank).

**Not obtained:** JustMem full text (arXiv rate-limited both the HTML and PDF fetch; the abstract is
what §6 rests on). A thinking-vs-non-thinking table for Qwen3.5-9B specifically — neither the model
card nor the Qwen3 report gives one for the ≤9B dense models, which is exactly why §1 has to be
measured here.

---

## 10. Suggested milestone order

| milestone | mechanism | cost | numerator (answers at the gate profile) |
|---|---|---|---|
| M44 | §1 R1 (structured reasoning field) then R2 (thinking, budgeted); LME-V2 re-run with `--reader-enable-thinking` for the comparable row | R1 minutes; R2 2–6 GPU-h; LME-V2 hours | the whole 2-fact stratum (217 rows) and the LME-V2 abstention stratum (128) |
| M45 | §2 consensus-gated commit, τ calibrated on LME-V2 abstention rows | minutes + decode on ~20% of rows | up to the 48 wrong-with-gold declines, minus what M44 already recovers |
| M46 | §4 timeline deltas + target-first ordering | minutes | the duration subset of 133 + 321 |
| M47 | §3 presupposition verification on LME-V2 | ~1 h | 128 × (target − 17.97%) |
| M48 | §5 three-way digest label (only if the digest is still off after M44) | minutes | the 113 negation rows |
| M49 | §6 REPLAY / supersedes routing as read-path views | minutes | knowledge-update 78 + temporal |
| M50 | one `build` with Chronos-style event tuples (RQ4) | one 57-min re-ingest + arms | temporal +23 / +50 |

The rule from the brief applies: rank by answers ÷ cost, and every arm ships on its pre-registered
bar or not at all. The only change this document argues for is the order — the reader-mode arm goes
first because every number after it is measured against the reader it uses.
