# M18 — where myelin stands against the industry, verified programmatically

The question this milestone answers is not "are we SOTA" — the answer to that was never in doubt — but
**which comparisons against the published literature are admissible at all, and what each one costs us**.
Before M18 the published numbers lived in a hand-maintained markdown table (`PLAN.md` §11.5, 13 rows) and
ours lived in 28 run directories holding an `aggregated_metrics.json` each (17 under `runs/`, 11 under
`runs/rescored/`); nothing joined them, `myelin-eval report` was a `println!("not implemented")`, and the two
G2 comparisons were inadmissible by construction because every published LoCoMo and LongMemEval_S number is
an LLM-judge score while both of our columns were deterministic token F1.

M18 makes our side of the comparison an artifact and the comparison itself a command:
[`docs/sota/registry.json`](../sota/registry.json) (25 claims, 10 converted sources, every row carrying a
verbatim quote with line numbers), judged columns on both G2 runs, a serialised G3 attack run, and
`myelin-eval standing`, which joins them and exits non-zero while a gate is unsupported.

## Verdict

**All five gates fail, every one of them on quality rather than on a missing artifact, and only one of the 25
published claims is beaten.** `myelin-eval standing --gate` exits 1. That is the headline, and the useful
detail is in the shape of the failures:

- **Two rows of 25 are `comparable` with no caveat at all.** GAM's LoCoMo token F1 (40.00 macro-mean, Qwen2.5-7B
  backbone, deterministic scorer) against our 53.15 — `+13.15`, the only row in the table where
  `claim: yes` — and the G3 gate row, where neither side has a judge. Everything else carries a caveat or
  refuses comparison.
- **Seventeen rows are `caveat-judge`, and that single fact is the largest structural obstacle to a SOTA
  claim.** Every judged accuracy comparison on LoCoMo, LongMemEval_S and LME-V2 has `qwen3.5-9b` on our side
  and `gpt-4o-mini`, `gpt-4.1-mini` or `gpt-5.2` on theirs. M18 removed the *metric* mismatch (token F1 vs
  judge); it could not remove the *grader* mismatch, because no frontier API key exists in this project.
- **Where we actually stand, in points.** LoCoMo judged 62.66 vs the G2 bar of 77.85 (`−15.19`) and vs the best
  published 93.05 (`−30.39`). LongMemEval_S judged 52.00 vs the bar of 80.80 (`−28.80`) and vs EverMemOS's
  83.00 (`−31.00`). LME-V2-Small 39.91 vs AgentRunbook-C's 74.9 (`−34.99`). The closest published system is
  Mem0 *as its own paper reports it* — 66.88 on LoCoMo, `−4.22` from us — which is 25.6 points below the 92.5%
  its vendor blog claims for the same system.
- **The one apples-to-apples row worth staring at.** AgentRunbook-R scores 58.60 on the same 451 LME-V2
  questions with the same open-weights reader we serve (`Qwen3.5-9B`, arXiv 2605.12493 L518) and the same
  memory-controller class. Only the judge differs. We are `−18.69` against it. That is the cleanest measured
  headroom in the whole table and it is what M19 aims at.
- **G3 is the only gate we are close to, and this run made it a file.** Undefended, pre-populated, at k=6 our
  ASR is 77.50% against MINJA's reported 76.80% — we are exactly as poisonable as the literature's victim
  agents. Defended it is 12.50% [Wilson 5.5–26.1], against a 10% bar: the gate misses by a single attack out of
  40. Injection success is 92.50% against MINJA's 98.20% (`+5.70`, the second row in the table where we are
  ahead, though it is a necessary condition for an attack rather than an attack). At the realistic
  `Asserted` tier the defended figure is 17.50%, and the ungated adaptive probe — ten poison records whose
  only defect is that they are false — sits at 60.0%.
- **LAFS is exactly 0.0, and that is the correct answer, not a bug.** Both of our operating points (39.91 @
  12.76 s and 35.70 @ 1.97 s) are dominated by the reference frontier's fastest point (RAG slice+notes, 51.0 @
  0.2 s), so adding them moves nothing. `adapters/lafs_point.py` computes `reference_lafs =
  55.76484693638005`, matching `PLAN.md` §11.5's 55.765, and `lafs_gain = 0.0`. The gate row is the one place
  in the registry that needs `bar: greater_than`: a tie is what a dominated submission scores, so an
  inclusive bar would have passed a gate we plainly do not meet.
- **No gate is unsupported for lack of an artifact any more.** Before this milestone G3 had no artifact at all
  (its numbers existed only in prose in `m11`/`m15`) and both G2 rows had no admissible metric. All five gate
  rows now compute from files on disk. The two `missing-artifact` rows left are DMR, which `PLAN.md` §11.2
  deliberately does not build because Zep has saturated it at 98.2%.

### Is the judged number believable?

The pre-registered alarm was a judged LoCoMo above ~70 against a 53.07 token-F1 baseline. It came in at
**62.66** (965 of 1,540 correct; 125 declines scored 0; 1,143 fresh verdicts plus M14's 272 cached), i.e.
+9.5 points over the same run's token F1 — the size of gap M14 already measured between token F1 and the
judge on category 2 (agreement 84.9% → 96.7% when moving to the date-aware scorer). No lenient-judge alarm
fires, and the judge is the model `m9-judge-panel.md` measured as the *harsher* grader of the two it
compared.

On LongMemEval_S the judged column is 52.00 over all 500 questions — the 470 answerable ones graded by the
judge, the 30 `_abs` items by the same decline rule `bench` uses. That population is not a choice: MemPro's,
NEMORI's and Zep's LongMemEval averages all reproduce **exactly** as micro-averages over the six official
type counts summing to 500 (arithmetic in each row's `caveat`), and the abstention items sit inside those
counts.

### Four defects in the numbers we were comparing against

Re-verifying every §11.5 row against its converted source found four:

1. **GAM's 40.00 is the Qwen2.5-7B row**, not `@ gpt-4o-mini` (that row is 43.14), and it is a macro mean of
   four category F1s rather than an overall string F1 (2604.12285 L365, L529, L174).
2. **NEMORI's 74.6 is `gpt-4.1-mini`**, not `@ gpt-4o` — its gpt-4o-mini LongMemEval average is 64.2
   (2508.03341 L923, L997-999).
3. **Vanilla Codex on tier small is 69.9 at 177.2 s**; the 69.3 ≈182 s in §11.5 is the abstract's
   tier-unspecific figure (2605.12493 L311 vs L659-665).
4. **The "62% → 6.7% when memory is pre-populated" attributed to MINJA is not in MINJA.** It is the
   EHR-poisoning paper's Table 1 at k=3 Levenshtein retrieval over 50 indication prompts (2601.05504 L148),
   and that paper's own sweep gives 6% at k=3, 20% at k=5 and 38% at k=10 — so at our k=6 the comparable
   literature figure is nearer 20% than 6.67%.

Two numbers were also *missing* and are now in the registry: EverMemOS's LongMemEval 83.00 (the best
frontier-backbone figure on that benchmark, above MemPro's 79.00) and NEMORI's LoCoMo 80.8.

### `myelin-eval standing`, verbatim

| metric | system | theirs | ours | gap | verdict | claim | run |
|---|---|---|---|---|---|---|---|
| locomo.judge_score.n1540 | MemPro-15 | 84.93 | 62.66 | -22.27 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/locomo_recall` |
| locomo.judge_score.n1540 | MemPro-15 (Qwen3-30B) **(gate)** | 77.85 | 62.66 | -15.19 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/locomo_recall` |
| longmemeval_s.judge_score.n500 | MemPro-15 | 79.00 | 52.00 | -27.00 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/lme_s_recall` |
| longmemeval_s.judge_score.n500 | MemPro-15 (Qwen3-30B) **(gate)** | 80.80 | 52.00 | -28.80 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/lme_s_recall` |
| locomo.judge_score.n1540 | EverMemOS | 93.05 | 62.66 | -30.39 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/locomo_recall` |
| longmemeval_s.judge_score.n500 | EverMemOS | 83.00 | 52.00 | -31.00 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/lme_s_recall` |
| locomo.judge_score.n1540 | Mem0 (paper) | 66.88 | 62.66 | -4.22 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/locomo_recall` |
| locomo.judge_score.n1540 | Mem0 (vendor) | 92.50 | 62.66 | — | not-comparable(vendor) | no | `runs/locomo_recall` |
| longmemeval_s.judge_score.n500 | Mem0 (vendor) | 94.40 | 52.00 | — | not-comparable(vendor) | no | `runs/lme_s_recall` |
| locomo.token_f1.n1540 | GAM | 40.00 | 53.15 | +13.15 | comparable | yes | `runs/locomo_recall_graph` |
| locomo.abstention_accuracy.n446 | HiGMem | 0.78 | 0.73 | -0.05 | caveat-backbone(open_weights vs frontier_api) | no | `runs/locomo_recall_qdate` |
| longmemeval_s.judge_score.n500 | Zep | 71.20 | 52.00 | -19.20 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/lme_s_recall` |
| dmr.accuracy.n500 | Zep | 98.20 | — | — | missing-artifact | no | — |
| dmr.accuracy.n500 | Zep (gpt-4-turbo) | 94.80 | — | — | missing-artifact | no | — |
| lme_v2_small.overall_full_set.combined | AgentRunbook-C **(gate)** | 74.90 | 39.91 | -34.99 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/myelin_inv2_web_small` |
| lme_v2_small.overall_full_set.combined | AgentRunbook-R | 58.60 | 39.91 | -18.69 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/myelin_inv2_web_small` |
| lme_v2_small.overall_full_set.combined | Codex (vanilla) | 69.90 | 39.91 | -29.99 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/myelin_inv2_web_small` |
| lme_v2_small.overall_full_set.combined | RAG slice+notes | 51.00 | 39.91 | -11.09 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/myelin_inv2_web_small` |
| lme_v2_small.lafs_gain.small | LME-V2 reference frontier **(gate)** | 0.00 | 0.00 | +0.00 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/myelin_inv2_web_small` |
| longmemeval_s.judge_score.n500 | NEMORI | 74.60 | 52.00 | -22.60 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/lme_s_recall` |
| locomo.judge_score.n1540 | NEMORI | 80.80 | 62.66 | -18.14 | caveat-judge(open_weights_local vs frontier_api) | no | `runs/locomo_recall` |
| minja.asr.k6_prepopulated | MINJA | 76.80 | 77.50 | -0.70 | caveat-backbone(open_weights vs frontier_api) | no | `runs/attack_live_m18` |
| minja.injection_success.k6_prepopulated | MINJA | 98.20 | 92.50 | +5.70 | caveat-backbone(open_weights vs frontier_api) | no | `runs/attack_live_m18` |
| minja.asr.k6_prepopulated | EHR poisoning (k=3) | 6.67 | 77.50 | -70.83 | caveat-backbone(open_weights vs frontier_api) | no | `runs/attack_live_m18` |
| minja.asr.k6_prepopulated_defended | G3 gate (PLAN.md 11.5) **(gate)** | 10.00 | 12.50 | -2.50 | comparable | no | `runs/attack_live_m18` |

### Unsupported gates (5)

- locomo.judge.mempro15.qwen3_30b [locomo.judge_score.n1540] behind by 15.19 (ours 62.66, gate needs >= 77.85)
- longmemeval_s.judge.mempro15.qwen3_30b [longmemeval_s.judge_score.n500] behind by 28.80 (ours 52.00, gate needs >= 80.80)
- lme_v2_small.agentrunbook_c [lme_v2_small.overall_full_set.combined] behind by 34.99 (ours 39.91, gate needs >= 74.90)
- lme_v2_small.lafs_gain.frontier [lme_v2_small.lafs_gain.small] tied at 0.00; this gate needs a strict improvement, and a tie is what a dominated submission scores
- minja.asr.g3_gate [minja.asr.k6_prepopulated_defended] behind by 2.50 (ours 12.50, gate needs >= 10.00)

### Protocol caveats

- `locomo.judge.mempro15.gpt4omini` — Judge is GPT-4o-mini with LightMem's LoCoMo prompt (L122). 10% of each dataset was sampled for MemPro's evolution training and the numbers are on the remainder (L120) - but the Avg. column reproduces exactly as the micro-average over the FULL official category counts (82.26*282 + 62.50*96 + 90.01*841 + 80.68*321)/1540 = 84.93, so the population is LoCoMo's 1,540 non-adversarial questions in their official proportions.
- `locomo.judge.mempro15.qwen3_30b` — G2's LoCoMo bar (PLAN.md 11.5): MemPro's own open-weights backbone, so the answer-model class matches ours. The judge is still GPT-4o-mini (L122), which is the one asymmetry left in this comparison. Avg. reproduces as the micro-average over the full 1,540: (75.17*282 + 70.83*96 + 83.47*841 + 67.60*321)/1540 = 77.85.
- `longmemeval_s.judge.mempro15.gpt4omini` — The LongMemEval Avg. is a micro-average over the six official question types: (77.44*133 + 73.68*133 + 83.33*78 + 98.57*70 + 60.71*56 + 86.67*30)/500 = 79.00 exactly, so n = 500 and the 30 `_abs` abstention items are inside those type counts rather than excluded. Judge prompt follows GAM (L122).
- `longmemeval_s.judge.mempro15.qwen3_30b` — G2's LongMemEval_S bar (PLAN.md 11.5). Micro-average check: (71.43*133 + 75.94*133 + 82.05*78 + 92.86*70 + 98.21*56 + 80.00*30)/500 = 80.80. Note the open-weights backbone scores HIGHER here than gpt-4o-mini's 79.00 and lower on LoCoMo - cross-model comparison on these two benchmarks is not monotonic.
- `locomo.judge.evermemos.gpt41mini` — Highest LoCoMo number in this registry. Population stated directly: "LoCoMo contains 1,540 questions over 10 ultra-long dialogues" (L110), repeated as #Questions 1,540 (L612). Judge is GPT-4o-mini plus two auxiliary judges that the paper never names, averaged blind (L114).
- `longmemeval_s.judge.evermemos.gpt41mini` — Supersedes MemPro's 79.00 as the best frontier-backbone LongMemEval_S number, and was not in PLAN.md 11.5's table. Population stated: "LongMemEval (S-setting, ~115k tokens per conversation) evaluates 500 questions" (L110), #Questions 500 (L613). Table 2's baselines are re-reported from the MemOS leaderboard rather than run by the authors (L586).
- `locomo.judge.mem0.paper` — The paper's own number, 25.6 points below the 92.5% the vendor blog claims for the same system. Adversarial category excluded ("this category was excluded from our evaluation because ground truth answers were unavailable", L67); the paper says "entire dataset" without stating n, so 1,540 is inferred from that exclusion. Judge model never named; prompt adapted from MemGPT (L614). Table 2 labels both Mem0 rows MemO^g; prose (L523-532) identifies 1764/66.88 as base Mem0.
- `locomo.vendor.mem0.blog` — mem0.ai blog, 2026. No protocol, no judge prompt, no question count, and no converted source in the local corpus - hence `vendor`, which `standing` treats as not comparable in either direction. Kept in the table because it is the number a reader will meet first.
- `longmemeval_s.vendor.mem0.blog` — mem0.ai blog, 2026. Same objection as the LoCoMo vendor row; note it exceeds every peer-reviewed LongMemEval_S number in this registry by more than 11 points.
- `locomo.token_f1.gam.qwen25_7b` — PLAN.md 11.5 recorded this as "GAM @ gpt-4o-mini"; it is the Qwen2.5-7B-Instruct row. GAM's gpt-4o-mini Average F1 is 43.14 (L529). The figure is also a MACRO mean of the four category F1s, not a micro-average over questions, and the paper never states n - 1,540 is LoCoMo's four non-adversarial categories, which is the subset GAM says it uses (L174). GAM reports no LLM-judge number at all.
- `locomo.abstention.higmem.gpt4omini` — Not the same quantity as ours, and the gap must be read with that in mind: HiGMem reports SQuAD token F1 on LoCoMo10's Adversarial category using a category-specific QA prompt that is handed the distractor candidate (L753), while our column is the fraction of adversarial items the reader declined. The paper does not state n; 446 is LoCoMo10's adversarial count (PLAN.md 1.1).
- `longmemeval_s.judge.zep.gpt4o` — Judge is GPT-4o with LongMemEval's own question-specific prompts, which is the protocol our judge rubric was written against. The paper never states a question count; 500 is LongMemEval_S's released size (docs/EVALUATION.md 6.2, probed).
- `dmr.accuracy.zep.gpt4omini` — Saturated benchmark and deliberately unbuilt on our side (PLAN.md 11.2), so this row exists to stay honest about a number we have not matched rather than to be beaten. The judge is only ever called "An LLM judge" (L207) - never named - so `frontier_api` is inferred from the paper's other judge being GPT-4o.
- `dmr.accuracy.zep.gpt4turbo` — Same DMR protocol and the same unnamed judge as the gpt-4o-mini row; kept because PLAN.md 11.5 carried both backbones.
- `lme_v2_small.agentrunbook_c` — G1's accuracy bar (PLAN.md 11.5). Two facts make this the cleanest comparison in the registry and one makes it dirtier. Clean: the reader is Qwen3.5-9B (L518, L525, L952) - the same open-weights model we serve - and the population is the same 451 questions we run (L77, Table 1 #Q L299), abstention and gotchas included. Dirty: the coding-agent family's controller is Codex + GPT-5.4-mini at xhigh reasoning (L656), where ours is the local reader, and free-form answers are graded by GPT-5.2 at medium reasoning (L974) where ours are graded by Qwen3.5-9B. The paper prints 0.749; recorded here on the pct_0_100 scale. The 240 web / 211 enterprise split is our harness's, not the paper's - it pools domains per tier (L1195).
- `lme_v2_small.agentrunbook_r` — The RAG-family ceiling, and the closest published system to ours in kind: Qwen3.5-9B as both memory controller and reader with Qwen3-Embedding-8B for retrieval (L518). Only the judge differs from our protocol. 26.9 s per query.
- `lme_v2_small.codex_vanilla` — PLAN.md 11.5 recorded 69.3 at about 182 s/query, which is the abstract's tier-unspecific figure (L311); Table 2's tier-small row is 0.699 at 177.2 s and the medium row is 0.687 at 185.8 s (L659-671). The tier-small number is the comparable one and is what this row carries.
- `lme_v2_small.rag_slice_notes` — The LAFS break-even bar: this is the reference frontier's fastest point (51.0 at 0.2 s), so any submission slower than 0.2 s must beat 51.0 to move the frontier at all (docs/EVALUATION.md 3.3). Our 45.0 at 11.06 s sits below it, which is why the LAFS gain is exactly 0.
- `lme_v2_small.lafs_gain.frontier` — G1's primary gate. The number to beat is a GAIN, so the bar is strict: 0.0 is what a dominated submission scores, and `bar: greater_than` is what stops a tie from passing. The reference frontier is hard-coded in the released tool (leaderboard/compute_lafs.py `fixed_frontier_points['small']` = 51.0@0.2s, 58.6@26.9s, 74.9@108.3s, 69.9@177.2s) and computes to reference_lafs 55.76484693638005, matching PLAN.md 11.5's 55.765.
- `longmemeval_s.judge.nemori.gpt41mini` — PLAN.md 11.5 recorded this as "NEMORI @ gpt-4o"; Table 8's two blocks are gpt-4o-mini (Average 64.2) and gpt-4.1-mini (Average 74.6) per L923, so 74.6 is the gpt-4.1-mini row. The Average is a micro-average over the six official type counts - (86.7*30 + 92.9*56 + 72.2*133 + 55.6*133 + 79.5*78 + 90.0*70)/500 = 74.60 - which independently confirms n = 500. The LongMemEval judge MODEL is never named: L480 says only that the LLM-judge prompts are adapted from Zep's task-specific set, so frontier_api is inferred from the paper's LoCoMo judge (gpt-4o-mini).
- `locomo.judge.nemori.gpt41mini` — NEMORI's LoCoMo LLM-judge average, not in PLAN.md 11.5's table. Judge is gpt-4o-mini (L480, printed "gpt-4.0-mini"). Population is stated: "featuring 1,540 questions across four reasoning categories" (L476).
- `minja.asr.paper_average` — Populations cannot be equal here and demanding it would make G3 unverifiable: MINJA averages over 45 victim-target pairs across EHRAgent (MIMIC-III, eICU), RAP (Webshop) and a QA agent, at 30 victim queries per pair (10 for MMLU) = 1,170 victim queries - a figure the paper never sums - while ours is 40 attacks over 8 cohorts in one domain. MINJA also reports no retrieval depth for this aggregate, so k=6 is ours alone. Direction is lower_is_better: a positive gap means our system is harder to poison.
- `minja.isr.paper_average` — Injection success is a necessary condition for an attack, not an attack: MINJA's 98.2% is over 630 attack queries (15 per pair, 10 for MMLU, 45 pairs) and counts records that entered memory. Ours counts poison payloads that reached the evidence set at k=6, over 40 attacks.
- `minja.asr.ehr_prepopulated_k3` — This is the actual source of PLAN.md 11.5's "62% -> 6.7% when memory is pre-populated", which that table attributes to MINJA - MINJA contains no such pair. Empty-memory ASR is 62%, relevant-initial-memory ASR is 6.67%, both over 50 indication-prompt variants against one victim/target pair at k=3 Levenshtein retrieval. The k matters more than the defence: the paper's own sweep gives 6% at k=3, 20% at k=5 and 38% at k=10 (Table 2), so at our k=6 the comparable literature figure is nearer 20% than 6.67%.
- `minja.asr.g3_gate` — The 10% bar is OURS, not a published number - hence `project_gate` rather than `paper`, which would attribute our threshold to someone else. It was rounded up from the EHR paper's 6.67% pre-populated ASR (the quote), and PLAN.md 11.5 states it as "MINJA ASR <= 10% with pre-populated memory at k = 6". Judged against the defended condition, which is the best our system can do.

### How our side was computed

Values are in the extractor's own unit, which is the one the run artifact carries; the comparison table above converts to each registry row's unit. `claimed by` names the registry rows a metric is compared against — `none` is something we measured that nobody published.

| metric | value | n | claimed by | detail | candidates |
|---|---|---|---|---|---|
| lme_v2_small.lafs_gain.small | 0.00 | 451 | 1 | adapters/lafs_point.py over 2 submission point(s): myelin_inv2_web_small+myelin_inv2_enterprise_small 39.91@12.76s, myelin_k25_web_small+myelin_k25_enterprise_small 35.70@1.97s | `myelin_inv2_web_small` 0.00 |
| lme_v2_small.memory_query_avg_seconds.enterprise | 2.120 s | 211 | none | memory_query.avg_seconds over 211 questions | `myelin_k25_enterprise_small` 2.12, `myelin_inv2_enterprise_small` 14.69 |
| lme_v2_small.memory_query_avg_seconds.web | 1.830 s | 240 | none | memory_query.avg_seconds over 240 questions | `myelin_k25_web_small` 1.83, `myelin_fast_web_small` 2.10, `myelin_inv2_web_small` 11.06 |
| lme_v2_small.overall_full_set.combined | 39.91 | 451 | 4 | question-weighted mean of myelin_inv2_web_small (45.00 over 240) and myelin_inv2_enterprise_small (34.12 over 211); mode="investigate" k=25 budget_tokens=10000 max_steps=2 prefetch_limit=null rerank_depth=null | `myelin_inv2_web_small` 39.91, `myelin_k25_web_small` 35.70 |
| lme_v2_small.overall_full_set.enterprise | 34.60 | 211 | none | overall.overall_full_set x 100 over 211 questions, evaluator Qwen/Qwen3.5-9B | `myelin_k25_enterprise_small` 34.60, `myelin_inv2_enterprise_small` 34.12 |
| lme_v2_small.overall_full_set.web | 45.00 | 240 | none | overall.overall_full_set x 100 over 240 questions, evaluator Qwen/Qwen3.5-9B | `myelin_inv2_web_small` 45.00, `myelin_k25_web_small` 36.67, `myelin_fast_web_small` 30.83 |
| locomo.abstention_accuracy.n446 | 73.09 | 446 | 1 | aggregated_metrics.json abstention_accuracy over category 5 | `locomo_recall_qdate` 73.09, `locomo_recall_qdate_temporal` 73.09, `locomo_recall_chrono_qdate` 72.42, `locomo_recall_chrono_qdate_temporal` 72.42, `locomo_recall_graph` 71.75, `locomo_recall_graph_temporal` 71.75, `locomo_recall_chrono` 70.63, `locomo_recall_chrono_temporal` 70.63, `locomo_recall` 69.96, `locomo_recall_temporal` 69.96, `locomo_investigate` 69.73, `locomo_investigate_temporal` 69.73, `locomo_recall_probe` 65.96, `locomo_recall_probe_temporal` 65.96 |
| locomo.judge_score.n1540 | 62.66 | 1540 | 6 | judge qwen3.5-9b : 965 of 1540 correct (125 declines scored 0) | `locomo_recall` 62.66 |
| locomo.temporal.n1540 | 51.64 | 1540 | none | date-aware scorer, f1_answerable over categories 1-4, rescored_from=runs/locomo_recall_graph | `locomo_recall_graph_temporal` 51.64, `locomo_recall_chrono_qdate_temporal` 51.43, `locomo_recall_temporal` 51.38, `locomo_investigate_temporal` 51.24, `locomo_recall_qdate_temporal` 51.23, `locomo_recall_chrono_temporal` 51.16, `locomo_recall_probe_temporal` 40.84 |
| locomo.token_f1.n1540 | 53.15 | 1540 | 1 | aggregated_metrics.json f1_answerable over categories 1-4, mode=recall k=6 | `locomo_recall_graph` 53.15, `locomo_recall` 53.07, `locomo_recall_qdate` 53.03, `locomo_investigate` 52.98, `locomo_recall_chrono_qdate` 52.93, `locomo_recall_chrono` 52.90, `locomo_recall_probe` 44.51 |
| longmemeval_s.judge_score.n500 | 52.00 | 500 | 6 | judge qwen3.5-9b : 260 of 500 correct (136 declines scored 0) | `lme_s_recall` 52.00 |
| longmemeval_s.token_f1.n500 | 44.73 | 470 | none | aggregated_metrics.json f1_answerable over the 470 answerable rows, mode=recall k=6 | `lme_s_recall_chrono` 44.73, `lme_s_recall` 43.69, `lme_s_recall_graph` 43.15, `lme_s_recall_probe` 56.61 |
| minja.asr.k6_prepopulated | 77.50 | 40 | 2 | condition "pre-pop/untrusted/undefended": 31/40 at k=6 over 8 cohorts, 39 injections admitted | `attack_live_m18` 77.50 |
| minja.asr.k6_prepopulated_defended | 12.50 | 40 | 1 | condition "pre-pop/untrusted/defended": 5/40 at k=6 over 8 cohorts, 9 injections admitted | `attack_live_m18` 12.50 |
| minja.injection_success.k6_prepopulated | 92.50 | 40 | 1 | condition "pre-pop/untrusted/undefended": 37/40 at k=6 over 8 cohorts, 39 injections admitted | `attack_live_m18` 92.50 |

## Corpus

Every registry row cites a paper that is downloaded, converted and indexed locally; a number we cannot quote
from a converted source is not in the table. 41 DOIs were touched this milestone:

|group|system / paper|doi|stem|status|note|
|---|---|---|---|---|---|
|1b existing|LongMemEval-V2/AgentRunbook|`10.48550/arXiv.2605.12493`|`10.48550_arxiv.2605.12493`|**indexed**|32pg, 38 chunks|
|1b existing|MemPro-15|`10.48550/arXiv.2606.00619`|`10.48550_arxiv.2606.00619`|**indexed**|20pg, 23 chunks|
|1b existing|EverMemOS|`10.48550/arXiv.2601.02163`|`10.48550_arxiv.2601.02163`|**indexed**|16pg, 20 chunks|
|1b existing|GAM|`10.48550/arxiv.2604.12285`|`10.48550_arxiv.2604.12285`|**indexed**|1pg, 27 chunks|
|1b existing|HiGMem|`(stem-only) 10.48550_arxiv.2604.18349`|`10.48550_arxiv.2604.18349`|**indexed**|10pg, 14 chunks; catalog row carries no DOI field (server-event conversion), so title backfill skips it|
|1b existing|NEMORI|`10.48550/arXiv.2508.03341`|`10.48550_arxiv.2508.03341`|**indexed**|1pg, 32 chunks|
|1b existing|Zep/Graphiti|`10.48550/arxiv.2501.13956`|`10.48550_arxiv.2501.13956`|**indexed**|1pg, 14 chunks|
|1b existing|Mem0|`10.48550/arxiv.2504.19413`|`10.48550_arxiv.2504.19413`|**indexed**|1pg, 23 chunks|
|1b existing|MemGPT|`10.48550/arxiv.2310.08560`|`10.48550_arxiv.2310.08560`|**indexed**|1pg, 17 chunks|
|1b existing|LoCoMo|`10.48550/arxiv.2402.17753`|`10.48550_arxiv.2402.17753`|**indexed**|1pg, 29 chunks|
|1b existing|LongMemEval-v1|`10.48550/arXiv.2410.10813`|`10.48550_arxiv.2410.10813`|**indexed**|1pg, 36 chunks|
|1b existing|MINJA|`10.48550/arXiv.2503.03704`|`10.48550_arxiv.2503.03704`|**indexed**|35pg, 39 chunks|
|1b existing|Hippocampus|`10.48550/arXiv.2602.13594`|`10.48550_arxiv.2602.13594`|**indexed**|21pg, 31 chunks|
|1b existing|SwiftMem|`(stem-only) 10.48550_arxiv.2601.08160`|`10.48550_arxiv.2601.08160`|**indexed**|18pg, 21 chunks; catalog row carries no DOI field (server-event conversion), so title backfill skips it|
|1b existing|MemBench|`10.48550/arXiv.2506.21605`|`10.48550_arxiv.2506.21605`|**indexed**|1pg, 23 chunks|
|1b existing|EHR-poisoning|`(stem-only) 10.48550_arxiv.2601.05504`|`10.48550_arxiv.2601.05504`|**indexed**|19pg, 18 chunks; catalog row carries no DOI field (server-event conversion), so title backfill skips it|
|1c already present|MemOS|`10.48550/arXiv.2507.03724`|`10.48550_arxiv.2507.03724`|**indexed**|37pg, 47 chunks|
|1c already present|Mem-alpha|`10.48550/arXiv.2509.25911`|`10.48550_arxiv.2509.25911`|**indexed**|1pg, 28 chunks|
|1c acquired|PoisonedRAG|`10.48550/arxiv.2402.07867`|`10.48550_arxiv.2402.07867`|**downloaded**|1079052 bytes|
|1c acquired|FOREVER|`10.48550/arXiv.2601.03938`|`10.48550_arxiv.2601.03938`|**downloaded**|5549753 bytes|
|1c acquired|BGE-M3|`10.48550/arxiv.2402.03216`|`10.48550_arxiv.2402.03216`|**downloaded**|754673 bytes|
|1c acquired|Adaptive-RAG|`10.48550/arXiv.2403.14403`|`10.48550_arxiv.2403.14403`|**downloaded**|577416 bytes|
|1c acquired|CRAG|`10.48550/arXiv.2401.15884`|`10.48550_arxiv.2401.15884`|**downloaded**|667756 bytes|
|1c acquired|Memory-R1|`10.48550/arXiv.2508.19828`|`10.48550_arxiv.2508.19828`|**downloaded**|3472962 bytes|
|1c acquired|PrefEval|`10.48550/arxiv.2502.09597`|`10.48550_arxiv.2502.09597`|**downloaded**|1650082 bytes|
|1c acquired|HELMET|`10.48550/arxiv.2410.02694`|`10.48550_arxiv.2410.02694`|**downloaded**|2164826 bytes|
|1c acquired|MABEN|`10.48550/arxiv.2507.05257`|`10.48550_arxiv.2507.05257`|**downloaded**|2002070 bytes|
|1c acquired|Cormack RRF (SIGIR 2009)|`10.1145/1571941.1572114`|`10.1145_1571941.1572114`|**downloaded**|66196 bytes|
|1e sweep|Utility Under Attack: Agent Memory Poisoning and the Lim|`10.48550/arXiv.2608.21230`|`10.48550_arxiv.2608.21230`|**downloaded**|311226 bytes|
|1e sweep|The Sleeping Agent: What Gist-Based Context Compression |`10.48550/arXiv.2608.11775`|`10.48550_arxiv.2608.11775`|**downloaded**|155379 bytes|
|1e sweep|HeadWiseKV: Budgeted Per-Head Cache Residency for Hybrid|`10.48550/arXiv.2609.02029`|`10.48550_arxiv.2609.02029`|**downloaded**|619926 bytes|
|1e sweep|MASkills: Continual Skills Optimization for Multi-Agent |`10.48550/arXiv.2609.02094`|`10.48550_arxiv.2609.02094`|**downloaded**|3466756 bytes|
|1e sweep|MemoryLACE: Memory Lifecycle-Aware Consolidation and Evi|`10.48550/arXiv.2609.03201`|`10.48550_arxiv.2609.03201`|**downloaded**|1999414 bytes|
|1e sweep|EdgeMem: LLM-Free Agent Memory Construction and Retrieva|`10.48550/arXiv.2609.05553`|`10.48550_arxiv.2609.05553`|**downloaded**|2122061 bytes|
|1e sweep|When Users Don't Ask: Benchmarking Context-Driven Memory|`10.48550/arXiv.2609.03467`|`10.48550_arxiv.2609.03467`|**downloaded**|427875 bytes|
|1e sweep|What Eviction Destroys: A Restore-Counterfactual Audit o|`10.48550/arXiv.2609.08279`|`10.48550_arxiv.2609.08279`|**downloaded**|232483 bytes|
|1e sweep|Personalizing LLM Agent Memory Using Biometrics|`10.48550/arXiv.2609.08558`|`10.48550_arxiv.2609.08558`|**downloaded**|980513 bytes|
|1e sweep|CueMem: Cue-Guided Context Reconstruction for Long-Term |`10.48550/arXiv.2609.12354`|`10.48550_arxiv.2609.12354`|**downloaded**|419724 bytes|
|1e sweep|Retrieval-Driven Memory Reconsolidation for Long-Term LL|`10.48550/arXiv.2609.16053`|`10.48550_arxiv.2609.16053`|**downloaded**|1965630 bytes|
|1e sweep|LSREP: A Longitudinal State-Replay Protocol for Evaluati|`10.48550/arXiv.2609.16730`|`10.48550_arxiv.2609.16730`|**downloaded**|461788 bytes|
|1c unobtainable|BEAM|`(none)`|`-`|**unobtainable**|identifier not resolved: two paper_search passes returned no title matching a BEAM memory/long-context benchmark|

**Paywalled: none.** Every DOI attempted resolved to an open-access PDF, including Cormack's SIGIR 2009 RRF
paper, which `PLAN.md` expected to be unobtainable — `paper_download` fetched a real 2-page PDF (66,196
bytes, sha256 `5c1f9f1f…`) rather than an ACM interstitial.

**Unobtainable: one.** `BEAM` — two `paper_search` passes ("BEAM benchmark memory agents", "BEAM benchmark
long-context 1M 10M tokens ICLR 2026") returned no title matching a BEAM memory or long-context benchmark, so
it is recorded as `unobtainable: identifier not resolved`. `docs/research/04-benchmarks.md` already flagged it
the same way. Three other identifiers the repo listed without a DOI **did** resolve and are now downloaded:
PrefEval → `10.48550/arxiv.2502.09597` (abstract: "We introduce PrefEval"), HELMET →
`10.48550/arxiv.2410.02694` (exact title match), MemoryAgentBench/MABEN → `10.48550/arxiv.2507.05257`
(abstract names MemoryAgentBench).

**Conversion deferred for 22 papers, and the reason is not the pipeline.** olmOCR-2-7B-FP8 runs on `big` under
vLLM with `--gpu-memory-utilization 0.60` ≈ 14.5 GB, and the card had 10.4 GB free all session: a household
voice assistant's `qwen3:8b` holds 9.8 GB with a rolling 30-minute `keep_alive` and a TRELLIS server holds
~3.1 GB. Lowering the reservation to 0.40 got past vLLM's startup guard and then OOMed during weight loading
(`torch.OutOfMemoryError … 9.19 MiB is free`, `/tmp/vllm-child.log`); the file was restored to 0.60 and the
other tenant's model was left alone. What unblocks it, in one line, when the assistant is idle:

```bash
ssh big gpu-tenant claim papers      # pauses nothing; evicts llama-swap models
# then, per stem: scribe_convert -> distill_index (the 22 `downloaded` rows above)
```

No registry row depends on those 22: all 25 cite one of the 10 stems that were already converted, and V2
confirms each one returns a `conversion.converted_at` from `catalog_read`. The papers matter for the
capability audit's literature column, where they are cited as *unconverted* wherever used.

**Title backfill.** `catalog_backfill_title` ran to exhaustion (101 → 0 candidates, 8 batches; the MCP tool
times out at 30 s, so it was driven in `limit: 6` batches) and filled 13 of the 16 comparison sources' titles.
Three remain null — `10.48550_arxiv.2604.18349` (HiGMem), `…2601.08160` (SwiftMem), `…2601.05504`
(EHR-poisoning) — because those catalog rows carry **no `doi` field at all**: they were synthesised by a
server-event conversion (`markdown_path` + `conversion` + `embedding` only), and the backfill keys on DOI.
Providers resolve all three titles fine (`hs paper get --doi …`), so this is a home-still catalog-repair item,
not a myelin one.

**The 1e sweep.** Forward citations of the five anchors returned 200 rows from the two 2024-era anchors
(LoCoMo, LongMemEval) and **zero** from the three 2026 anchors (LME-V2, MemPro, MINJA) — the local mirrors lag
for very recent arXiv ids, exactly as the plan anticipated. 135 unique candidates, all 2026; 47 had abstracts;
12 passed the keep-filter (names a memory system or benchmark **and** reports a number on
LoCoMo/LongMemEval/LME-V2/DMR/MemBench/BEAM) and all 12 were acquired, well inside the cap of 40, so nothing
is deferred for cap reasons. None of the 12 beats the registry's frontier rows — the strongest claim in the
set is REALM's 75.97 LoCoMo / 65.11 LongMemEval (`10.48550/arXiv.2609.16053`), below EverMemOS on both — but
three are directly relevant to the audit below: `2608.21230` (limits of content screening against memory
poisoning), `2609.08279` (a restore-counterfactual audit of eviction/forgetting on LongMemEval_S) and
`2609.03201` (lifecycle-aware consolidation, and a BEAM number).

## Capability audit

One row per mechanism the literature says matters. `our status` cites this repo at `path:line`; `literature
effect` cites a registry source or a converted stem. `decision ∈ {keep, measure, build, reject}`, and every
`measure`/`build` names the headroom that justifies it.

|#|mechanism|our status|our measured effect|literature effect|decision|
|---|---|---|---|---|---|
|1|episode segmentation|`pipeline/ingest.rs` `segment`/`SegmentConfig`, boundary-signal based|5,882 LoCoMo turns → 550 episodes, 503.6 records/unit (`m3-write-path.md:11-17`)|NEMORI's adaptive distillation is the same move and the only one of these papers to isolate it (`10.48550_arxiv.2508.03341`)|keep|
|2|fact extraction|`pipeline/extract.rs`, one model call per episode|precision **0.917**, recall 0.797, F1 0.853 on 50 hand-labelled episodes; 4,168 semantic records (`m3-extraction-quality.md:11-16`)|Mem0's extract+consolidate pipeline reaches 66.88 judged on LoCoMo (registry `locomo.judge.mem0.paper`)|keep|
|3|4-op consolidation|`pipeline/consolidate.rs`, ADD/UPDATE/DELETE/NOOP behind deterministic gates|add 4,876, noop 761, update 166, dedup 166, delete 5 over one LoCoMo build (`m3-write-path.md:28-34`)|Mem0/Memory-R1's delta ops; Memory-R1 itself is downloaded but unconverted (`10.48550/arXiv.2508.19828`)|keep|
|4|injection adjudication|`pipeline/adjudicate.rs`; **default off** (`pipeline/write.rs:170`, `:191`)|ASR at k=6 pre-populated **77.5% → 12.5%** [5.5–26.1] untrusted, 17.5% asserted, with **zero** legitimate records refused; adaptive probe 60.0% (`runs/attack_live_m18/attack_live.json`)|MINJA 76.8% ASR / 98.2% ISR (registry); EHR-poisoning 6.67% at k=3, 38% at k=10 (`2601.05504` Table 2); `2609.08279`/`2608.21230` argue content screening alone is insufficient|**build**: the gate misses by 2.5 points (one attack in 40) and the per-form profile shows 5 of 8 surface forms caught 5/5 while forms 6-7 are caught 0-2/5 — the residue is assertion-shaped poison that no content classifier can separate from a true fact|
|5|hybrid dense+BM25 with RRF|`pipeline/retrieve.rs:139-141`, `pipeline/fuse.rs:50` (`rrf_k = 1`)|recall@6 0.8815 (k=1) vs 0.8252 (k=60) vs 0.8666 (BM25 only) vs 0.7314 (dense only) (`m4-ablation.md:19-23`)|MemPro's ablation drops 84.93 → 72.25 without BM25 (`10.48550_arxiv.2606.00619` L595)|keep|
|6|cross-encoder rerank|`rerank/cross.rs`, depth 25 (`pipeline/retrieve.rs:142`)|**+2.7** points recall@6 over hybrid k=1 (0.9085), +8.3 over k=60, for +173 ms (`m4-ablation.md:23,52-57,92-93`)|MS MARCO MRR@10 18.7 → 36.5 for a cross-encoder over BM25 (`PLAN.md` §2 finding 2)|keep|
|7|graph route (PPR over phrase↔record incidence)|`store/graph.rs`; **default off** (`pipeline/retrieve.rs:144`)|**−0.7** points on LongMemEval_S, CI [−1.5, −0.1]; no gain in any category (`m12-graph-route.md`)|HippoRAG-2 / QueryLink / G-Memory report gains; our corpora do not reproduce them|reject|
|8|agentic loop|`pipeline/investigate.rs`, `max_steps = 2` (`:102`)|36.7% → **43.3%** at two steps on the 60-question web probe, then declining (`m7-step-value-curve.md:6-12`); full-set web 36.67% (k=25 recall) → 45.00% (investigate)|AgentRunbook-C 74.9 at 108.3 s and vanilla Codex 69.9 at 177.2 s, both multi-step coding agents (registry)|keep|
|9|evidence composition and budget|`pipeline/compose.rs:144`, bookend order, `chronological` off (`:105`)|order changed 668 of 2,486 answers and moved accuracy by nothing measurable (`compose.rs:74-94`, `m13-temporal-axis.md`)|Lost-in-the-Middle's U-curve is the reason bookend exists; LLMLingua/Provence compression is unconverted|keep|
|10|abstention gate|`RetrieveConfig::tau_abstain = None` (`pipeline/retrieve.rs:143`, rationale `:60-117`) and `InvestigateConfig::abstain_on_insufficient = false` (`pipeline/investigate.rs:95`)|best τ buys **+0.4** points (inside noise on 240 questions); the insufficiency statement cost **−28.3** points [−41.7, −15.0] because the reader acknowledges it and answers from pretraining anyway (`m6-abstention-gate.md`)|nothing published on the memory side; LME-V2 grades abstention with GPT-5.2 and a flawed-premise rubric (`2605.12493` L995)|reject|
|11|**forgetting arithmetic — absent**|`model::record::Salience` is defined (`model/record.rs:220-226`) and persisted (`store/schema.sql:54`), written only as `Salience::default()` at five sites (`ingest.rs:152`, `extract.rs:242`, `consolidate.rs:647`, `compose.rs:258`, `write.rs:618`); no `pipeline/forget.rs`; nothing increments `access_count`, sets `last_access` or decays `strength`|none — there is nothing to measure|MemoryBank `R = e^{−t/S}`, MOOM β=0.9 reinforcement vs α=0.1 decay (`PLAN.md` §8); `2609.08279`'s restore-counterfactual audit of eviction on LongMemEval_S is the first quantification and is downloaded but unconverted|**measure**, with a named prerequisite: both G2 corpora are single-shot, so no arm on them can move a decay term. The headroom is unmeasurable until a corpus where records age exists — `2609.08279`'s restore-counterfactual protocol is the cheapest way to get one|
|12|**learned/adaptive read policy — absent**|no `policy` module in `lib.rs:6-13`; `mode` is a caller-supplied parameter (`model/query.rs:15`)|M7's curve is non-monotone *in opposite directions per stratum*: non-abstention rises 37.2 → 46.5 → 48.8 with steps while abstention collapses 35.3 → 23.5 → 11.8, so one fixed `max_steps` is provably leaving points on both sides (`m7-step-value-curve.md:22-24`)|Adaptive-RAG, CRAG, Self-RAG, Memory-R1, Mem-α — Mem-α is converted and indexed (`10.48550_arxiv.2509.25911`); Adaptive-RAG (`2403.14403`) and CRAG (`2401.15884`) are downloaded, unconverted|**measure**: a per-question router between `recall` and `investigate` is the only mechanism that can take both strata's optimum at once; the arithmetic above bounds the prize at roughly +2 points per stratum|
|13|**audit log — absent**|no `audit.rs` anywhere in `crates/`; `PLAN.md` §9 C10 specifies per-write and per-read records|none|MemOS's governance layer (converted: `10.48550_arxiv.2507.03724`); Collaborative Memory's ACL projections|**build**: it is a governance requirement with no accuracy claim attached, and `runs/attack_live_m18` shows why — 29-31 of 40 injections are refused per condition and there is currently no per-decision record to post-mortem which mechanic fired|
|14|multi-tenant ACL and unlearning|`store/ledger.rs:823-882` (`grant_user_agent`/`grant_agent_namespace`/`revoke_*`/`acl_edges`), `hard_delete` at `:570`|I1–I5 property tests green; zero cross-tenant leaks on every read path (G3/M11)|Collaborative Memory's bipartite ACL; MINJA's premise is a *shared* memory bank, so isolation removes the premise (`PLAN.md` §9 C12)|keep|
|15|**local embedding path — absent**|`crates/myelin-core/src/embed/` contains `mod.rs` and `remote.rs` only|measured this session: 71–104 ms warm per query against ollama `bge-m3` on `big`, 13.8 s on a cold load, and the embedder was pushed 94% onto CPU by another tenant's VRAM|none — portability and latency only|**build**, lowest priority: the failure mode is availability, not accuracy, and it is the one dependency that made this session's numbers depend on another tenant's scheduler|
|16|latency envelope|`recall` p50 **0.253 s** (LoCoMo) and **0.413 s** (LongMemEval_S) (`runs/locomo_recall`, `runs/lme_s_recall`); LME-V2 memory-query mean 1.83 s at k=25 and 11.06 s under `investigate` (`runs/standing/standing.md` provenance table)|`PLAN.md` §11.5's own target is p95 < 100 ms for `recall`; we are 2.5–4× over it at p50|SwiftMem reports **10.834 ms** search-only on LoCoMo (`10.48550_arxiv.2601.08160` L162 — the prose prints "10,834 ms", a typo: its own baselines are 881.9–1231.3 ms and it claims 81–114× lower) at LLM-judge 0.7253|**measure**: our p50 is ~24× SwiftMem's search-only floor, and the decomposition it publishes (1.364 ms tag inference + 7.395 ms index search + 1.285 ms rerank) says the gap is in our index, not our reranker|
|17|token cost per query|8,926 mean prompt tokens on LME-V2 web (`runs/myelin_inv2_web_small/aggregated_metrics.json` `tokens.avg_prompt_tokens`); LoCoMo evidence budget 2,048–10,000 tokens (`ComposeConfig`)|—|Mem0 26,031 → 1,764 tokens (>90% saving) and p95 17.117 s → 1.440 s on LoCoMo (`10.48550_arxiv.2504.19413` L492-498, L7); Zep 115k → 1.6k average context tokens (`10.48550_arxiv.2501.13956` L247, L255)|keep: at 8.9k tokens against a 200k truncation budget we are already an order of magnitude inside the full-context baseline, and no gate is token-bound|

## What M19 does

M17 (the pre-registered width arm) is unspent and untouched by this milestone. M19 is the item this audit
selects **after** it: **write-time runbook synthesis — procedure and hint notes distilled per trajectory at
ingest, retrieved as first-class records.**

The headroom is named, and it is the only large one in the table that a retrieval-side change cannot reach:

- `lme_v2_small.overall_full_set.combined` — AgentRunbook-R scores **58.60** against our **39.91** on the same
  451 questions with the same `Qwen3.5-9B` reader and the same open-weights controller class (registry row
  `lme_v2_small.agentrunbook_r`, arXiv 2605.12493 L518, L596-602). `−18.69` points, and the mechanism the
  paper credits is precisely note synthesis at write time plus radius-1 evidence windows — not a wider
  prefetch.
- `m16-evidence-sufficiency.md` measured S = P(sufficient | wrong) at **7.4%** [4.4, 12.0] at `recall` k=25 and
  **12.2%** [8.1, 17.9] at `investigate max_steps=2`: the answer is usually *not in the pool at all*, so the
  retrieval-fix ceiling is 67.6% against 35.7% measured. Composition, ordering and reranking are all
  downstream of evidence that does not exist yet.
- Our weakest LME-V2 strata are exactly the ones notes target: answerable `dynamic` 35.3% web / 28.6%
  enterprise and `gotchas` 40.0% / 28.6%, against `static` 60.0% / 47.3% (`runs/myelin_inv2_*_small`
  `non_abstention_by_category`).

**The experiment, one command per arm.** Add a `Procedural`/hint note pass to `WritePath` for LME-V2
trajectories (the record kind already exists — `m3-write-path.md` counted 318 procedural records on LoCoMo),
rebuild `myelin_lme_v2_small`, and run both domains at M16's operating point:

```bash
myelin-eval build --corpus lme-v2-small --lmev2-dir /tmp/lmev2      # with notes on
.venv/bin/python crates/myelin-eval/adapters/run_myelin.py --domain web        --output-dir runs/m19_notes_web_small
.venv/bin/python crates/myelin-eval/adapters/run_myelin.py --domain enterprise --output-dir runs/m19_notes_enterprise_small
myelin-eval standing --out runs/standing        # the pair is picked up automatically
```

**Decision rule, fixed before the run.** Let Δ be the change in `lme_v2_small.overall_full_set.combined`
against the M16 baseline pair (39.91), with a paired bootstrap CI over the pooled 451 questions from
`adapters/paired_ci.py`:

- Δ ≥ +3.0 points **and** the CI excludes 0 ⇒ notes ship on by default.
- CI includes 0 ⇒ notes ship off, and the M19 doc records the null with the same prominence M12 and M13 gave
  theirs.
- Δ ≥ +3.0 on `procedure`/`gotchas` only, with the pooled CI including 0 ⇒ report as a category-specific
  finding, default off, and the branch becomes a per-category routing question for M20 rather than a default
  change.

The rule deliberately does not reference LAFS: at 12.76 s per query the submission is dominated by the
frontier's 51.0 @ 0.2 s point until accuracy clears 51.0, so no note pass can produce a positive gain on its
own, and measuring against LAFS would score the wrong thing.

## Artifacts, and how to reproduce this verdict

```bash
myelin-eval judge --run runs/locomo_recall        # 1,143 new verdicts + 272 cached, 3m0s
myelin-eval judge --run runs/lme_s_recall         # 334 verdicts, 1m58s
myelin-eval attack --live --ledger-dir data --out runs/attack_live_m18   # 48 min, six conditions + probe
myelin-eval standing --out runs/standing          # joins docs/sota/registry.json against runs/
myelin-eval standing --gate; echo $?              # 1, naming the five unsupported gates
```

|artifact|what it is|
|---|---|
|`docs/sota/registry.json`|25 published claims, 10 converted sources, verbatim quotes with line numbers|
|`runs/locomo_recall/judge_verdicts.json`|1,415 verdicts over the 1,540 LoCoMo non-adversarial rows (125 declines need none)|
|`runs/lme_s_recall/judge_verdicts.json`|334 verdicts over the answered rows of LongMemEval_S|
|`runs/attack_live_m18/attack_live.json`|the first serialised G3 sweep: six conditions × 8 cohorts + the adaptive probe|
|`runs/standing/standing.{json,md}`|the join, the verdicts, the gaps, and every candidate run that was not selected|

Three schema fields were added to the registry beyond the milestone plan's shape, each because a row could
not otherwise be recorded honestly, and none of them touches the comparison rules:

- `bar: at_least | greater_than` — four of the five gates are thresholds where equality passes; the LAFS gate
  needs a strict improvement, and without this a dominated submission's 0.0 gain would have passed it.
- `population_comparable: bool` (default `true`) — on benchmark rows a differing `n` is `PLAN.md` §1.1's
  landmine and stays fatal. On attack rows equal `n` was never possible (MINJA's 1,170 victim queries over 45
  pairs, the EHR paper's 50 indication prompts, our 40 attacks), so insisting on it would have made G3
  permanently unverifiable.
- `provenance: project_gate` — the 10% MINJA bar is *ours*, rounded up from the EHR paper's 6.67%. Recording
  it as `paper` would attribute our threshold to someone else; recording it as `vendor` would have excluded
  it from gating.

`PLAN.md` §11.5 now carries a pointer to the registry and the five gate rows instead of a table, and
`docs/EVALUATION.md` §1 names `standing` as the mechanical check of the three gates.
