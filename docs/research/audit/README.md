# Claim audit

Three independent adversarial verification passes over the factual claims in `PLAN.md`. Each verifier was
instructed to look for defects, not agreement, and to return `CONFIRMED | CORRECTED | UNVERIFIABLE |
MISATTRIBUTED` per claim with a verbatim quote, a `doc_id:line` pointer, and the experimental conditions
attached to each number.

| file | scope | claims | outcome |
|---|---|---|---|
| `retrieval-claims.md` | SwiftMem, ShardMemo, RocketQAv2, chunk size, hybrid α, Provence, HiGMem, lost-in-the-middle | 8 | 3 confirmed clean, 1 misattributed, 1 unverifiable, 3 condition-stripped |
| `systems-claims.md` | Mem0, Memory-R1, Zep/Graphiti, HippoRAG, HippoRAG-2, Generative Agents, MemoryBank, Mem-α, QueryLink | 8 (+2 sub-claims) | mostly confirmed; several condition-stripped; Zep DMR discrepancy resolved |
| `security-eval-claims.md` | MINJA, EHR poisoning study, Collaborative Memory ACL, SSGM Theorem 1, MOOM decay, EverMemOS judging, MEMAGENT budgets | 8 | 4 confirmed, 1 confirmed-with-semantics, 3 corrections |

## Defects found and fixed in PLAN.md

| # | defect | correction applied |
|---|---|---|
| 1 | MOOM decay coefficients inverted — the plan attached α = 0.1 to retrieval and β = 0.9 to time | Reversed: **β = 0.9 is retrieval reinforcement, α = 0.1 is temporal decay**. The qualitative conclusion survives (β > α) but the wiring would have been backwards. §8 |
| 2 | `36.5` MRR@10 credited to RocketQAv2 | It is **BERT-large** over BM25 top-1000; RocketQAv2-ERNIE is **40.1**; BM25-anserini is 18.7. All on MS MARCO dev. §2 finding 2 |
| 3 | ShardMemo "+3 / +6.8 F1" stated without conditions | **+2.9 / +3.1 F1** vs a simple learned router at S = 20/80; **+6.76 macro-F1 with GPT-OSS-20B** only (+6.34 Qwen3-32B; main backbone ≈5.9). Probe budget `B_probe = 3`, `K = 10`, `S = 40` now stated. §2 finding 3 |
| 4 | "hybrid +2.1 mAP over dense" | **Unverifiable** — not present in the Best-Practices-RAG paper, whose mAP deltas are ~23 points. Dropped; it never entered `PLAN.md`, and `03-retrieval.md` §1.2 should be read with this caveat. The α = 0.3 half of the claim is confirmed (BM25 + Contriever, TREC DL19/20) |
| 5 | 512-token chunk result cited to the EMNLP version | Table 3 is in the **arXiv preprint 2407.01219**, and 512 wins only on faithfulness — **256 wins on relevancy** (97.78 vs 97.41). It is a single-10-K eval judged by gpt-3.5-turbo. §6.1 now says "starting point, not settled result" |
| 6 | Zep numbers quoted without backbones; DMR 94.8 vs 98.2 looked contradictory | Both are real: **94.8% @ gpt-4-turbo** (the apples-to-apples row against MemGPT's 93.4% @ gpt-4-turbo) and **98.2% @ gpt-4o-mini**. LongMemEval_S 71.2% is the **gpt-4o** row; gpt-4o-mini is 63.8%. §2 finding 6, §11.5 |
| 7 | HippoRAG "2Wiki EM 33.4 → 46.6" without naming the baseline | Baseline is **ColBERTv2** on 2WikiMultiHopQA; AR@5 is the **all-recall** metric. §2 finding 7 |
| 8 | MINJA ASR figures spliced into one curve | Two distinct setups: **62% → 6.67%** (empty → pre-populated, k = 3) and **6% → 20% → 38%** (6 initial memories + 4 indication prompts, k = 3/5/10). Llama baseline is **52.94%**, not 53%. §2 finding 9 |
| 9 | "54 poisoned entries at trust = 1.0" without the model | It is **Gemini-2.0-Flash**; the GPT-4o-mini run rejected all 23 candidates. Trust was a guard-agent composite credibility score. §2 finding 9 |
| 10 | EverMemOS κ conflated with its 93.05 score | κ = **0.891 (LoCoMo) / 0.979 (LongMemEval)** vs 5 human annotators over 25 Q&A pairs; **93.05 is LoCoMo accuracy**, a different quantity. §11.3, §11.5 |
| 11 | SSGM `O(N·ε_step)` with `N` ambiguous | **`N` is the reconciliation interval**, not the total horizon — drift depends on cadence, which makes cadence a tunable budget. §9 |
| 12 | SwiftMem latency variants conflated | **10.834 ms vs 881.9–1231.3 ms** on LoCoMo; **11.7 ms vs 794–1264 ms** on LoCoMo **Refined** — different benchmarks. Also "10,834 ms" in the converted markdown is a decimal-separator artifact of 10.834 ms. §2 finding 3 |

## Confirmed without correction

MemPro BM25/embedding/PAGE-ID ablation (84.93 → 72.25 / 82.57 / 84.37, re-verified by the lead against
arXiv 2606.00619v1 directly); LongMemEval-V2 numbers (re-verified against 2605.12493v1 directly);
Mem0 and Memory-R1 both using exactly `{ADD, UPDATE, DELETE, NOOP}`; Zep bi-temporal separation and edge
invalidation; HippoRAG-2 synonym-edge counts; Generative Agents reflection threshold 150 and the
three-term retrieval score with unit weights; MemoryBank `R = e^{−t/S}`; Mem-α 0.592 vs 0.502;
MINJA 98.2% ISR / 76.8% ASR; Collaborative Memory bipartite-ACL formalism; MEMAGENT 1,024 / 5,000 / 8,192
budget split and RULER-HQA 81.25 @ 7k → 74.22 @ 896k; HiGMem 8.09 vs 99.84 turns and P@K 0.1909 vs 0.0101;
lost-in-the-middle 53.8% mid-context vs 56.1% closed-book on GPT-3.5-Turbo with 20 documents;
Provence ~49–82% compression at no accuracy cost with DeBERTa-large.

## Carry-over corrections for `README.md`

`README.md` (the prior literature survey, left as authored) contains two figures the audit reclassified.
They are not used by `PLAN.md`, but noting them here so the two documents are not read as agreeing:

- "SwiftMem 11.7 ms/query vs 794–1264 ms/query for Nemori/LightMem/EverMemOS" — those values are the
  **LoCoMo Refined** table (2601.08160 Table 4). On plain LoCoMo the figures are 10.834 ms vs
  881.9–1231.3 ms (Table 2). Same conclusion, different benchmark.
- "ShardMemo … +3 F1 on LoCoMo at fixed budget" — the controlled router lead is +2.9/+3.1 F1 at S=20/80;
  the larger +6.76 macro-F1 figure is GPT-OSS-20B-specific.
