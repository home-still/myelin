# Research catalog, round 5 *(2026-09-28)*

This adds to [`sota-catalog-2026-09-24.md`](sota-catalog-2026-09-24.md),
[`sota-catalog-2026-09-25.md`](sota-catalog-2026-09-25.md) and
[`sota-catalog-2026-09-26.md`](sota-catalog-2026-09-26.md). Those catalogs
chose *where* LongMemEval_S is lost: preference, and declines with the
evidence in hand. This one serves *how* the reader is asked. It covers the
advice answer (M78/M78b), the typed premise pass (M79), the strict-judge
preference rubric, seed-replicated measurement, and the JSON-schema
field-order defect.

Sources come from the home-still corpus (`distill_search`, `markdown_read`,
`catalog_read`). Papers not in the corpus were checked with `paper_search`
(arXiv/OpenAlex/Crossref/Semantic Scholar metadata and abstracts only).
Every entry is tagged **home-still: yes**, or **home-still: not in corpus**
with the metadata source. Numbers are as the papers report them. **derived**
marks our own arithmetic.

## 0. Citation check

These are the ten papers cited in M78, M78b and M79 and in the
seed-replicated method, all checked on 2026-09-28. Three have full text in
home-still; the other seven have metadata only.

| citation | verified id / venue | home-still | correction |
|---|---|---|---|
| PrefEval, Zhao et al. 2025 | arXiv 2502.09597 (`10.48550/arxiv.2502.09597`) | yes | none; venue not stated in corpus text |
| "Let Me Speak Freely?", Tam et al. 2024 | EMNLP 2024 Industry, `10.18653/v1/2024.emnlp-industry.91`; arXiv 2408.02442 | yes (indexed under the ACL DOI) | authors: Tam, Wu, Tsai, Lin, Lee, Chen |
| Attribute First, then Generate, Slobodkin et al. 2024 | ACL 2024, `10.18653/v1/2024.acl-long.182`; arXiv 2403.17104 | not in corpus (paper_search: arXiv+Crossref+OpenAlex) | none |
| ALCE, Gao et al. 2023 | EMNLP 2023, `10.18653/v1/2023.emnlp-main.398`; arXiv 2305.14627 | not in corpus (paper_search: Crossref+OpenAlex) | none |
| CREPE, Yu et al. 2023 | ACL 2023, `10.18653/v1/2023.acl-long.583`; arXiv 2211.17257 | not in corpus (paper_search: Crossref+OpenAlex) | first author listed as "Xinyan Velocity Yu" |
| (QA)², Kim et al. 2023 | ACL 2023, `10.18653/v1/2023.acl-long.472` | yes, as the arXiv version 2212.10003 (2022-12-20) | authors: Kim, Htut, Bowman, Petty |
| Sufficient Context, Joren et al. | arXiv 2411.06037 (2024-11-09) | not in corpus (paper_search: arXiv+OpenAlex) | year is 2024; the venue was not confirmed |
| Wagner 2026 | arXiv 2607.08456 (2026-07-09), single author Benedikt J. Wagner | not in corpus (paper_search: arXiv+OpenAlex) | full title: "Two Axes of LLM Abstention: Answer Correctness and Question Answerability"; the indexed abstract has a spliced sentence, so only clean clauses are quoted below |
| Miller 2024 | arXiv 2411.00640, Evan Miller | not in corpus (paper_search: arXiv+OpenAlex) | none |
| Bouthillier et al. 2021 | arXiv 2103.03098 | not in corpus (paper_search: arXiv+OpenAlex) | venue not confirmed by paper_search |

Findings sit in the section where each citation is used.

## a. Preference answers, speaker attribution, selecting versus generating

- **PrefEval.** Zhao et al. 2025, arXiv 2502.09597. **home-still: yes.**
  - Table 2, 10 turns / ~3k tokens, zero-shot → Reminder:
    - Claude-3.5-Sonnet 0.07 → 0.45;
    - Gemini-1.5-Pro 0.07 → 0.91;
    - o1-preview 0.50 → 0.98.
  - Its error taxonomy has a class for our M78 failure. *Preference
    Hallucination Violation*: "The response fabricates or misattributes
    preferences, diverging from the user's true preference."
  - RAG over turns: k = 5 does as well as k = 10. More turns "might serve as
    another form of distraction".
  - **For myelin:** M78's quotes in the assistant's words are PrefEval's
    misattribution class, not a new failure. Keep the candidate set small.
- **CUPID.** Kim et al. 2025, arXiv 2508.01674. **home-still: not in corpus**
  (paper_search: arXiv/Semantic Scholar).
  - 756 human-curated session histories.
  - Across 10 LLMs, models "fail to discern what previous context is relevant
    to a new request -- under 50% precision and 65% recall".
  - **For myelin:** open search over history is hard even for frontier
    models. Rank candidates in code, and have the reader pick among a few.
- **HorizonBench.** Li et al. 2026, arXiv 2604.17283. **home-still: not in
  corpus** (paper_search: arXiv+OpenAlex).
  - 4,245 items, 6-month histories of ~163K tokens.
  - The best of 25 frontier models reaches 52.8%; most are at or below the
    20% chance baseline.
  - On evolved preferences, "over a third of the time they select the user's
    originally stated value".
  - **For myelin:** give the reader the date of each preference candidate and
    order candidates by time, so a later statement can win.
- **PersonaMem-v2.** Jiang et al. 2025, arXiv 2512.06688. **home-still: not in
  corpus** (paper_search: OpenAlex). Frontier models reach 37–48% on implicit
  personalization; a 2k-token agentic memory reaches 55% "using 16x fewer
  input tokens". **For myelin:** a compact preference block beats a long
  episode, as PrefEval's k = 5 also says.
- **When Users Don't Ask (LOCOMO-CONV).** Chang & Chen 2026, arXiv 2609.03467.
  **home-still: yes.**
  - "strong retrieval does not fully translate into response quality".
  - Implicit queries show "silent grounding": memory improves grounding
    without surfacing the gold fact.
  - AlpsBench (09-26) agrees: explicit memory "do[es] not inherently
    guarantee more preference-aligned" responses.
  - **For myelin:** this is the shape of our 20 declines that hold every gold
    turn. The fix is on the reader side, not in retrieval.
- **Speaker attribution (user versus assistant): no direct measurement
  found.**
  - We searched home-still (6 distill queries) and paper_search (4 queries).
    No paper measures how often a reader credits the assistant's words to
    the user in memory QA.
  - The nearest evidence is LongMemEval itself (Wu et al., arXiv 2410.10813,
    **home-still: yes**, §4 setup): "When sessions or rounds are used as the
    key, we only keep the user-side utterances."
  - Its commercial-system study skips single-session-assistant because
    "the systems do not remember any information given by the assistant".
  - **For myelin:** enforce the speaker in code, as M78b already does.
    Our 18–39% assistant-quote rate (M78) fills a gap in the literature and
    is worth writing up.
- **Attribute First, then Generate.** Slobodkin et al. 2024, ACL,
  `10.18653/v1/2024.acl-long.182`. **home-still: not in corpus**
  (paper_search).
  - Generation is split into content selection, then sentence planning, then
    sequential generation. The selected spans become the attributions.
  - It gives "more concise citations" while it "maintains - and in some
    cases enhances - both generation quality and attribution accuracy".
  - It also "significantly reduces the time required for fact verification".
    The abstract gives no numbers.
  - **For myelin:** M78b's pick-by-index is this pattern. It holds only
    because `picks` is decoded before `recommendation` (§d).
- **ALCE.** Gao et al. 2023, EMNLP, `10.18653/v1/2023.emnlp-main.398`.
  **home-still: not in corpus.** "on the ELI5 dataset, even the best models
  lack complete citation support 50% of the time." **For myelin:** select by
  index and check in code; do not generate quotes.
- **When Does Selection Replace Extraction?** Sharma & Lall 2026-09-27,
  Zenodo `10.5281/zenodo.22985242`, pre-registered. **home-still: not in
  corpus** (paper_search: OpenAlex).
  - Raw turns selected by one Jev call are non-inferior to LLM-extracted
    memory on LoCoMo: the one-sided 95% bound is −3.0 against a −5 margin.
    Raw turns are 3,061× cheaper to write.
  - Reranking adds +17.4 (LoCoMo) and +9.1 (LongMemEval) when 3 of 30
    candidates are kept, but only +1.5 / +1.1 at generous budgets.
  - "Reranking lowers correct abstention."
  - **For myelin:** this supports selecting user turns over generating
    quotes. The abstention cost is the thing to watch.

## b. Over-abstention, false premises, selective answering

- **Sufficient Context.** Joren et al. 2024, arXiv 2411.06037. **home-still:
  not in corpus** (paper_search).
  - Smaller models (Mistral 3, Gemma 2) "hallucinate or abstain often, even
    with sufficient context".
  - Selective generation guided by sufficient context raises "the fraction of
    correct answers among times where the model responds by 2--10%".
  - **For myelin:** our 20 gold-held declines are this case. A sufficiency
    signal from code (the value is present in the evidence) can gate a
    re-ask. That is M71b.
- **Two Axes of LLM Abstention.** Wagner 2026, arXiv 2607.08456. **home-still:
  not in corpus** (paper_search).
  - Five instruction-tuned models, 2B–14B.
  - Answer-confidence "is nearly blind to whether the question is
    answerable", and "the blind spot does not shrink with scale".
  - On CREPE, asking the model outright whether a premise is false stays
    "near chance". A hidden-state probe reaches 0.69–0.77 AUROC.
  - "Instructing a model to check premises backfires, because it then
    disputes sound and false premises alike (57% false challenges)". Routing
    that instruction with the probe "roughly triples challenge precision".
  - A two-threshold policy certifies both budgets at 0.75 coverage of correct
    answers, against 0.31 for a single threshold.
  - **For myelin:** M79's `mismatch` field is the "instruct it to check
    premises" arm. It runs only on rows that already decline, and it commits
    on `none` or `detail unstated` (`commit_arm.rs`). Wagner's two error
    types then cost differently:
    - a false *challenge* on an answerable decline is a recovery missed,
      never a new loss;
    - a false *fit* on one of the 30 `_abs` rows breaks a correct
      abstention.
    - Pre-register both rates, recovered answerable declines and commits on
      `_abs` rows, and read the commit rule against Wagner's 57% before
      trusting the typed label.
- **CREPE.** Yu et al. 2023, ACL, `10.18653/v1/2023.acl-long.583`.
  **home-still: not in corpus.** 25% of natural questions carry false
  presuppositions; models "find presuppositions moderately well, but
  struggle when predicting whether a presupposition is factually correct".
- **(QA)².** Kim et al. 2023, ACL, `10.18653/v1/2023.acl-long.472`.
  **home-still: yes** (arXiv 2212.10003).
  - Questionable assumptions are "false or unverifiable".
  - The best models reach 56% human-judged acceptability end to end, and 64%
    and 72% on the detection and verification subtasks.
  - Zero-shot, "most models only achiev[e] half the performance when
    questionable assumptions were present".
  - Detection tracks end-to-end acceptability (Spearman ρ = 0.58).
  - **For myelin:** our ~13 "detail unstated" declines (the "7 shirts, trip
    not stated as 5 days" case) are (QA)²'s *unverifiable* class. There the
    acceptable answer addresses the assumption *and still answers*.
    LongMemEval's grader accepts an answer that contains the gold, so a
    caveat costs nothing.
- **LongMemEval-V2.** Wu et al. 2026, arXiv 2605.12493. **home-still: yes.**
  - "Premise awareness" is one of the five abilities. The judge grades a
    generic `UNKNOWN` as 0 on flawed-premise questions.
  - AgentRunbook-R "does not improve abstention as it directly presents the
    relevant evidence … the model can be misled".
  - AgentRunbook-C improves abstention because the memory module is
    "instructed to explicitly identify the inconsistencies and wrong question
    premises and present them to the downstream model". It is the best
    method, at 72.5% average.
  - **For myelin:** premise analysis works better *upstream*, as a stated
    finding handed to the reader, than as a reader self-check. This is the
    code-side form of M79's question.
- **Uncertainty-Based Abstention.** Tomani et al. 2024, arXiv 2404.10960.
  **home-still: yes.**
  - In-dialogue uncertainty filters "50% of unanswerable questions at the
    cost of incorrectly refusing 10% of answerable questions".
  - **For myelin (derived):** LongMemEval_S has 30 abstention rows and 470
    answerable ones. At that ROC point a gate gains ≤ 15 abstentions and
    costs ~47 answerable rows. Any abstention gate needs its answerable-row
    cost measured first.
- **AbstentionBench.** Kirichenko et al. 2025, NeurIPS, `10.52202/085713-5729`.
  **home-still: not in corpus.** 20 datasets, underspecified and
  false-premise questions included. "reasoning fine-tuning degrades
  abstention (by 24% on average)". A system prompt "can boost abstention"
  without fixing it. **For myelin:** in a thinking reader, abstention moves
  with the prompt in both directions.

## c. Evaluation variance, power, and non-determinism

- **Adding Error Bars to Evals.** Miller 2024, arXiv 2411.00640.
  **home-still: not in corpus.** Treats questions as draws "from an unseen
  super-population"; formulas for analysis, for two-model differences and
  for planning; no numbers in the abstract. **For myelin:** the 3-seed
  average cuts answer-sampling variance only. Question-sampling variance
  remains, and the paired bootstrap over questions estimates it. Keep both.
- **Accounting for Variance in ML Benchmarks.** Bouthillier et al. 2021,
  arXiv 2103.03098. **home-still: not in corpus.** Data sampling, init and
  hyperparameters "impact markedly the results"; randomizing more sources of
  variation approaches the ideal estimator "at a 51 times reduction in
  compute cost".
- **With Little Power Comes Great Responsibility.** Card et al. 2020, EMNLP,
  `10.18653/v1/2020.emnlp-main.745`. **home-still: not in corpus.**
  Underpowered comparisons are common; 2,000 MT sentences give
  "approximately 75% power to detect differences of 1 BLEU point".
- **Quantifying Variance in Evaluation Benchmarks.** Madaan et al. 2024,
  arXiv 2406.10229. **home-still: not in corpus** (paper_search). Seed
  variance is measured; item analysis "struggle[s] to meaningfully reduce
  variance".
- **Non-Determinism of "Deterministic" LLM Settings.** Atil et al.,
  arXiv 2408.04667 / Eval4NLP 2025. **home-still: not in corpus.** "accuracy
  variations up to 15% across naturally occurring runs"; best-to-worst gap
  up to 70%.
- **Greedy Decoding Is Not Precision-Invariant.** Du et al. 2026-09-22,
  arXiv 2609.26621. **home-still: not in corpus.** BF16 vs FP16: "49-100% of
  prompts diverge", decided by "the top-two logit margin at the LM head". A
  selective FP32 LM head gives +22–36 pp exact agreement at batch ≤ 4 and
  nothing at batch ≥ 8. Related, metadata only, no numbers quoted: Yuan et
  al. 2025, NeurIPS `10.52202/085713-5653`; He 2025, `10.64434/tml.20250910`.
- **The Price of Safety.** Bhowmik 2026-09-19, arXiv 2609.22818.
  **home-still: not in corpus.** A memory-agent eval run 3× per condition;
  noise "remains significant even at temperature zero"; CIs span ~±4.5
  points; a reranker costs −4.4 (CI [−9.0, −0.05]; McNemar p = 0.064).
- **Our arithmetic (derived)**:
  - One run at 79% over 500 questions has a 95% CI of **±3.6 points** from
    question sampling alone.
  - The 2.2-point gap to MemPro-15 is **11 questions**. A net +11 is
    significant (exact McNemar, two-sided α = 0.05) only if **≤ 25
    questions flip in total**: 18 won / 7 lost passes (p = 0.043), 19 / 8
    fails (p = 0.052).
  - On the 30 preference rows, a change needs at least **6–0, 8–1 or 10–2**
    (wins–losses; p = 0.031 / 0.039 / 0.039) to be significant. The 13/30 baseline
    carries a ±17.7-point CI.
  - **For myelin:**
    - Report discordant counts beside every paired delta.
    - Use the exact sign test on strata of ≤ 30 rows.
    - Hold llama-server batch and parallel settings identical across arms.
      This is inferred from Du and Atil: batch composition changes numerics.

## d. Structured decoding, reasoning, and JSON field order

- **Let Me Speak Freely?** Tam et al. 2024, EMNLP Industry,
  `10.18653/v1/2024.emnlp-industry.91`. **home-still: yes.**
  - "a significant decline in LLMs' reasoning abilities under format
    restrictions". "stricter format constraints generally lead to greater
    performance degradation in reasoning tasks."
  - JSON mode helps classification by "constraining possible answers".
  - The key finding for us: "100% of GPT 3.5 Turbo JSON-mode responses placed
    the 'answer' key before the 'reason' key, resulting in zero-shot direct
    answering instead of zero-shot chain-of-thought reasoning."
  - "The order of keys in structured outputs and the decoupling of reasoning
    from format adherence emerge as important factors."
- **CRANE.** Banerjee et al. 2025, arXiv 2502.09061. **home-still: not in
  corpus** (paper_search).
  - Grammars restrictive enough to allow only final answers provably reduce
    reasoning. Adding reasoning rules to the grammar preserves it.
  - Up to 10 points over baselines on GSM-symbolic and FOLIO.
- **Quantifying the Impact of Structured Output Format.** Yuan et al. 2026,
  Findings EACL, `10.18653/v1/2026.findings-eacl.91`. **home-still: not in
  corpus** (paper_search).
  - Causal analysis finds "no causal impact in 43 out of 48 scenarios" on
    GPT-4o.
  - o3 is "more resilient to output formats" than GPT-4o/4.1.
  - Our reading: the format per se rarely matters; instructions and, per
    Tam, key order do.
- **JSONSchemaBench.** Geng et al. 2025, arXiv 2501.10868. **home-still: not
  in corpus.** 10K real schemas; six frameworks, llama.cpp included; the
  abstract gives no numbers.
- **The repo side** is already measured, in
  `docs/measurements/defect-2026-09-28-schema-field-order.md`:
  - No crate enables serde_json's `preserve_order`, which
    `cargo tree -i serde_json -e features` confirms again today. Every schema
    therefore goes out alphabetically.
  - A llama.cpp wire test showed the model writes fields in the order they
    are sent.
- **For myelin:**
  - Tam's "answer key before the reason key → direct answering" is the
    mechanism behind the defect doc's reversed rows. M44 R1's −0.8 measured
    *answer first*, so a true reasoning-first reader is still untested. Tam's
    results predict that reasoning-first does better on reasoning-heavy rows.
  - The accident runs both ways. On the wire, the shipped `reflect` writes
    `sufficient` *last*, after `reason`, and `support` writes `missing`
    before `verdict`. That is the reasoning-before-decision order Tam and
    CRANE favour.
  - The intended source orders put the decision first. Turning on
    `preserve_order` as-is would move the shipped path to the order Tam
    predicts is worse.
  - **Derived recommendation:** set each source order deliberately, with
    reasoning before the decision, before flipping the feature. Then measure
    with the backlog item's `--evidence-only` control.
  - M78, M78b and M79 are unaffected. Their field names sort into the
    intended order (`memory` < `quote`, `picks` < `recommendation`,
    `answer` < `mismatch`), and a test pins the order on the wire.
  - Once `preserve_order` lands, those names stop carrying the order and the
    source order becomes the one path.

## e. LLM-as-judge with rubric references

- **LongMemEval's own judge protocol.** Wu et al., arXiv 2410.10813,
  App. A.4. **home-still: yes.**
  - The judge is prompt-engineered `gpt-4o-2024-08-06`, with ">97% agreement
    with human experts".
  - Meta-evaluation used 30 questions per type. The judge "slightly deviates
    from human experts for the single-session-preference and abstention
    problems", and stays at ≥ 90% in all settings.
  - The preference prompt reads: "The model does not need to reflect all the
    points in the rubric. The response is correct as long as it recalls and
    utilizes the user's personal information correctly."
  - **For myelin:**
    - The strict judge's preference rubric (`judge-preference-rubric.md`)
      already carries this clause as "does not need to reflect every point in
      the rubric".
    - The paper meta-evaluated gpt-4o, not the gpt-4o-mini behind our
      official column. The ≥ 90% is **not measured for our grader**.
- **Can LLM be a Personalized Judge?** Dong et al. 2024, Findings EMNLP,
  `10.18653/v1/2024.findings-emnlp.592`. **home-still: yes.**
  - Personalized judging shows "low and inconsistent agreement with human
    ground truth".
  - With verbalized confidence, accuracy is "above 80%" on high-certainty
    binary samples.
- **The Coin Flip Judge?** Yagubyan 2026, arXiv 2606.13685. **home-still:
  yes.**
  - Judges were GPT-4o-mini and GPT-4.1-mini, with 50 trials per question.
    Pairwise verdicts flip 13.6% of the time, and 28% of questions flip
    above 20%.
  - Cross-judge agreement is 76% (κ = 0.51). Equivalent templates change
    the majority verdict in 25% of cases.
  - Deterministic decoding "reduces but does not eliminate" inconsistency.
    11 trials recover the 50-trial verdict with 95% probability.
  - **For myelin:** the official grader is this model family. On the 30
    preference rows, judge 3× and take the majority, and report the judge's
    own flip rate.
- **Reliability without Validity.** Norman, Rivera & Hughes 2026, arXiv
  2606.19544. **home-still: yes.**
  - 21 judges, ~541,000 judgments.
  - Raw agreement overstates Cohen's κ by 33–41 pp. Judge rankings shift by
    up to 14 positions across benchmarks.
  - Test-retest above 0.95 coexists with position bias above 0.10.
  - **For myelin:** report strict-versus-official agreement as κ, not
    percent.
- **LoCoMo-Plus.** Li et al. 2026, arXiv 2602.10715. **home-still: yes.**
  "conventional string-matching metrics and explicit task-type prompting are
  misaligned" with latent-constraint memory; it grades constraint
  consistency. (EdgeMem's 17.7% lenient-judge flips: 09-25 catalog.)

## f. Landscape: LongMemEval_S rows since 2026-09-24

paper_search with `date >= 2026-09-15` and `>= 2026-09-24` (5 queries each),
plus corpus `distill_search`:

| system | date / id | what it reports | same-size LME_S ≥ 80.80? |
|---|---|---|---|
| EnSIMem (Meng et al.) | 2026-09-23, arXiv 2609.27279 | "high answer accuracy"; no numbers in the abstract; not in corpus | unknown, pull the full text |
| HasMem (He et al.) | 2026-09-25, arXiv 2609.30797 | LME-S *lexical F1* 3.4 → 8.9; not QA accuracy | no |
| Selection vs Extraction (Sharma & Lall) | 2026-09-27, Zenodo | reranking deltas on LongMemEval; no overall row in the abstract | no row |
| CARD + LOCI (Sanyal) | CORE 409229069, 2026 | claims 96.6% on "LongMemEval-500" with phi4-14B; only 4 categories (IE/KU/MSR/TR); credits the benchmark to "Zhang et al."; cites a "GPT-4 baseline (~82%)"; grader not stated; a defensive prior-art disclosure | **not admitted**: unverifiable and not LME_S as described |
| Agent Brain (Sritharan) | SSRN `10.2139/ssrn.6617298` | LongMemEval-**M** (cleaned) 71.7%, GPT-4o judge; notes Zep's 71.2% on LME-S with gpt-4o-mini | no |
| PSD; RPMem | arXiv 2609.23449; 2609.23466 | LoCoMo only; PERMA 85.52% (Qwen3-8B) | not LME |

**Verdict:** no verified same-size open-weight LongMemEval_S row at or
above MemPro-15's 80.80 appeared since 09-24. `docs/sota/registry.json`
needs no change. EnSIMem's full text is the one open item.

## g. Low-bit quantization (recorded only)

- **Quantization Hurts Reasoning?** Liu et al. 2025, arXiv 2504.04823.
  **home-still: not in corpus.** Lossless at W8A8 or W4A16; "lower bit-widths
  introduce significant accuracy risks"; size, origin and task difficulty
  decide the damage.
- **Exploring the Trade-Offs.** Lee et al. 2025, IJCAI,
  `10.24963/ijcai.2025/902`. **home-still: not in corpus.** Quantized models
  "often struggle with instruction-following and hallucination detection";
  "quantization magnifies a model's inherent weaknesses".
- **"Give Me BF16 or Give Me Death"?** Kurtic et al. 2025, ACL,
  `10.18653/v1/2025.acl-long.1304`. **home-still: not in corpus.** At 4 bits
  and above: FP8 lossless, INT8 1–3%, INT4 competitive.
- **BitNet b1.58 2B4T.** Ma et al. 2025, arXiv 2504.12285. **home-still: not
  in corpus.** Natively trained ternary reaches parity at 2B (not PTQ).
- **Gap:** nothing found that measures post-training ternary at ≤ 2 bpw
  (PrismML PTQ1_0 / PQ2_0 class) on instruction following, JSON adherence
  or long-context QA.
- **For myelin:** the Bonsai reader sits outside every measured regime.
  Measure format fragility (e.g. a schema-order A/B) rather than assume it
  either way.

## What we take from it

Ranked by what each unblocks, for a Bonsai 27B reader under the code-first
rule:

| # | build | target | research |
|---|---|---|---|
| 1 | **Schema field order** (BACKLOG item): set every source order deliberately, with reasoning before the decision (`reflect`: reason → sufficient; `support`: missing → verdict), *then* turn on `preserve_order` and measure with the `--evidence-only` control; re-test a true reasoning-first reader (M44 R1 measured answer-first) | shipped `reflect`/`support`/extract; M44 R1 conclusion | Tam 2024, CRANE, Yuan 2026 |
| 2 | **M78b** (queued): the research backs its design, with picks restricted in code to user memories, ≤ 3 of them, and no decline field. Worth adding: dates on the candidates and time order, so a later preference can win | preference (13/30 vs 24/30) | Attribute First, ALCE, PrefEval, CUPID, HorizonBench, Sharma & Lall |
| 3 | **M79** (queued): read it on two pre-registered rates, recovered answerable declines and commits on the 30 `_abs` rows. The next step, if the typed label is noisy, is a code-side premise finding handed to the reader (LongMemEval-V2's AgentRunbook-C pattern) | ~13 "detail unstated" declines | Wagner, (QA)², CREPE, LongMemEval-V2, Tomani |
| 4 | **Measurement**: keep 3 seeds with the paired bootstrap; add discordant counts and the exact sign test for strata ≤ 30; fix llama-server batch and parallel settings per arm; judge preference rows 3× by majority | every readout | Miller, Card, Bouthillier, Atil, Du, Coin Flip Judge |
| 5 | **Strict-judge preference rubric**: LongMemEval's "need not reflect all the points" clause is already in it; report κ against the official grader, not percent agreement | preference rows' strict column | LongMemEval App. A.4, Dong 2024, Norman 2026, LoCoMo-Plus |

Item 1 changes the shipped path, so it is its own measured arm. Items 2 and
3 do not wait for it (their names already sort correctly). They are validated
on their own strata, then go into the one bundle arm held to +3.0 (user,
2026-09-26).
