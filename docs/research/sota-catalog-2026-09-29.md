# SOTA catalog, 2026-09-29 — the landscape after round 5

**Why:** the three gates closed today (LongMemEval_S 81.53, LoCoMo 78.64,
LME-V2 78.05). Before calling that "state of the art", a sweep for anything
newer, or anything we missed, that beats those numbers in the same class.

**How:**
- A research agent searched home-still (`paper_search`, `distill_search`)
  and read arXiv HTML where a paper was not converted.
- The main session then re-verified the one row that matters, **Hindsight**,
  in home-still's own converted text. The PDF was downloaded and converted
  into the corpus (stem `10.48550_arxiv.2512.12818`), and Table 3 is at
  lines 1346–1352.
- Both Hindsight rows are now in `docs/sota/registry.json`.

**Verdict:**
- **LongMemEval_S: one same-class row beats us.** Hindsight on gpt-oss-20b
  scores 83.6, against our 81.53 official / 81.20 strict.
  - Its judge is GPT-OSS-120B with LongMemEval's own per-type templates:
    the official prompts, on a different grader model.
  - The lead sits in the single-session strata.
  - That makes 80.80, the bar the user set, no longer the strongest
    same-class row.
- **LoCoMo:** two same-class rows score higher, Hindsight at 83.18 and
  AutoViewMem at 83.7–85.3. Neither is under a judge comparable to ours.
  - Under a gpt-4o-mini judge, the strongest same-class row is TiMem at
    74.74, which we beat.
- **LongMemEval-V2:** nothing new. Our 78.05 against 74.90 stands.

---

**Same class** means an open-weights backbone of about 32B parameters or fewer.

**Confidence tags.**
- **V-hs**: seen in the paper's full text in home-still (`markdown_read`/`distill_search` chunk text).
- **V-arx**: seen in the paper's full text on arXiv HTML, fetched with curl and grepped. The paper is not in home-still.
- **R-abs**: the paper's abstract only, from `paper_search`/`paper_get` metadata.
- **derived**: our own arithmetic.

| system | DOI | benchmark (population) | number | backbone | judge | class | beats us? |
|---|---|---|---|---|---|---|---|
| **Hindsight** (OSS-20B) | 10.48550/arXiv.2512.12818 (v1, 2025-12-14) | LongMemEval_S (500) | **83.6** (V-arx; derived 418/500 from per-type rows) | gpt-oss-20b for memory stack + answers | **GPT-OSS-120B**, temp 0, LongMemEval's per-type official prompts + "you may provide reasoning … \boxed{yes/no}" | **same class** | **YES, +2.07 over 81.53.** Judge prompt family matches; judge model differs (open 120B vs gpt-4o-mini). Single run, no CI. |
| **Hindsight** (OSS-20B) | 10.48550/arXiv.2512.12818 | LoCoMo (1,540; derived: category micro-average with 282/96/841/321 reproduces 83.18) | **83.18** (V-arx) | gpt-oss-20b | GPT-OSS-120B; the LoCoMo judge prompt is not given in the paper | **same class** | **Number is higher (+4.54), but the judge is not comparable.** Prompt unstated; not the LightMem prompt on gpt-4o-mini. |
| Hindsight (OSS-120B) | 10.48550/arXiv.2512.12818 | LongMemEval_S (500) / LoCoMo (1,540) | 89.0 / 85.67 (V-arx) | gpt-oss-120b | GPT-OSS-120B | open >32B | not same class |
| Hindsight (Gemini-3) | 10.48550/arXiv.2512.12818 | LongMemEval_S / LoCoMo | 91.4 / 89.61 (V-arx) | Gemini-3 Pro answers, OSS-120B memory | GPT-OSS-120B | frontier | n/a (frontier) |
| **AutoViewMem** | 10.48550/arXiv.2609.21940 (2026-09-18) | LoCoMo (1,540) | **83.7** Qwen3-8B; **85.3** Qwen3-14B (V-arx; derived 83.68 from category rows) | Qwen3-8B / Qwen3-14B (memory + generation) | **Qwen3-8B judge** ("unified"); prompt not quoted | **same class** | **Number is higher, but the judge is not comparable.** It is lenient: MemGAS scores J 0.783 at F1 0.183, and the Full-History oracle scores J 0.821. |
| TiMem | 10.48550/arXiv.2601.02845 | LoCoMo (1,540) | 74.74 (V-arx) | Qwen3-32B internal + Qwen3-32B answer | gpt-4o-mini (LLJ-G); Qwen3-32B judge gives 72.73 | same class | no (74.74 < 78.64) |
| TiMem | 10.48550/arXiv.2601.02845 | LoCoMo (1,540) | 66.04 (V-arx) | gpt-4o-mini internal + Qwen3-8B answer | gpt-4o-mini | mixed | no |
| TiMem | 10.48550/arXiv.2601.02845 | LoCoMo (1,540) | 75.30 (headline); 80.45 with a Qwen3-235B answerer (V-arx) | gpt-4o-mini (internal); answer gpt-4o-mini / Qwen3-235B-A22B | gpt-4o-mini, Mem0 QA prompts | frontier / mixed >32B | n/a |
| TiMem | 10.48550/arXiv.2601.02845 | LongMemEval_S (500) | 76.88 (gpt-4o-mini); 78.96 (gpt-4o) (V-arx) | gpt-4o-mini / gpt-4o | official LongMemEval-S prompt; judge model unstated in that line | frontier | n/a (below 81.53 anyway) |
| MemLoc | 10.48550/arXiv.2609.07093 (2026-09) | LongMemEval_S (500) | 73.00 Qwen3-8B-Think; 71.20 Qwen3-30B-A3B; 58.40 Qwen3-8B no-think (V-hs, **table OCR garbled; cell placement uncertain**) | Qwen3-8B / Qwen3-30B-A3B generator, 8B trained locator, top-3 | GPT-4o ("4o-J") | same class | no |
| MemLoc | 10.48550/arXiv.2609.07093 | LongMemEval_S (500) / LoCoMo (n unclear) | 68.40 / 57.40 (V-hs) | gpt-4o-mini generator | GPT-4o | frontier | n/a |
| MAGMA | 10.18653/v1/2026.acl-long.1709 | LoCoMo (**1,986**, adversarial included: 841+446+321+282+96) | 70.0 (V-hs) | gpt-4o-mini | gpt-4o-mini, own prompt (adversarial rule) | frontier | n/a; different population |
| MAGMA | 10.18653/v1/2026.acl-long.1709 | LongMemEval (>100k tokens, so S) | 61.2 "Average" (V-hs) | gpt-4o-mini | gpt-4o-mini | frontier | n/a |
| MRAgent | 10.48550/arXiv.2606.06036 | LoCoMo (1,540, adversarial excluded) | 84.21 Gemini; 88.32 Claude (V-hs) | Gemini-2.5-Flash / Claude-Sonnet-4.5 | GPT-4o-mini, own binary prompt, 3 runs | frontier | n/a |
| MRAgent | 10.48550/arXiv.2606.06036 | LongMemEval_S, **4 of 6 types** (multi-session, SSU, temporal, preference; no KU/SSA) | 72.95 Gemini; 86.76 MRAgent* (V-hs) | Gemini / Claude retrieval | GPT-4o-mini | frontier | n/a; subset |
| CueMem | 10.48550/arXiv.2609.12354 (2026-09-11) | LoCoMo (1,540) / LongMemEval_S (500) | 81.10 / 75.20 (V-hs) | Llama-3.3-70B-Instruct | Llama-3.3-70B-Instruct | open >32B | not same class (LoCoMo is higher but uses a self-judge at 70B) |
| JustMem | 10.48550/arXiv.2609.19877 (2026-09) | LoCoMo (1,540) / LongMemEval_S (500) | 79.61 / 83.40 (V-hs; mean of 5 runs) | GPT-4.1-mini | GPT-4.1-mini | frontier | n/a |
| REALM | 10.48550/arXiv.2609.16053 (2026-09-13) | LoCoMo / LongMemEval_S (unanswerable excluded) | 75.97 / 65.11 (V-hs) | gpt-4o-mini | gpt-4o-mini | frontier | n/a |
| Supra Cognitive Modes | 10.48550/arXiv.2607.19096 | LoCoMo factoid (1,540 of 1,986) / LongMemEval **oracle** (500; 948 haystack sessions) | 84.87 / 86.00 (V-arx) | Claude Sonnet 4.5 synthesis | gpt-4o-mini, Mem0 prompt / gpt-4o-2024-08-06 official prompts | frontier | n/a; the LME number is oracle, not S |
| Regimes / ActiveGraph | 10.48550/arXiv.2606.10241 | LongMemEval_S, held-out splits only | e.g. split 0.78→0.88 (V-arx) | claude-sonnet-4-6 reader | gpt-4o-2024-08-06, official | frontier | n/a |
| Hippocampus | 10.48550/arXiv.2602.13594 | LongMemEval_S per type; LoCoMo F1 | no overall; per-type accuracy 16.67–68.57; LoCoMo judge on a 1–5 scale (V-hs) | not stated | GPT-5, 1–5 scale | unknown | no |
| Retrieval Beats Cheap Structured Memory | 10.20944/preprints202608.1369.v1 | LoCoMo **n=160 subset**; LongMemEval subsets (KU n=72; n=254) | BM25 39.4 vs full-context 30.6; KU 73.6 (R-abs) | 8B extractor / 70B ablation; answerer unstated | "LLM-as-judge" | unknown | no; subsets |
| MemAudit / MEMPROBE | 10.48550/arXiv.2606.24595 | none (new benchmark MEMPROBE) | recovery ≈0.6 (R-abs) | — | — | — | not comparable |
| nox-mem | 10.5281/zenodo.22649268 | LongMemEval + LoCoMo **retrieval** (n=2,482) | nDCG@10 0.526 / 0.495 (R-abs) | Gemini embeddings; no QA reader | none (retrieval metric) | — | not comparable (no QA accuracy) |

## Checked with nothing comparable

- **LongMemEval-V2:** no new results. `paper_citations` on 2605.12493 lists 13 citing papers. The most likely four had their abstracts read:
  - SELF-INDEX 2609.19656;
  - Supra Cognitive Modes 2607.19096;
  - Regimes 2606.10241;
  - BudgetBench 2609.13149.

  None reports an LongMemEval-V2 small score. The registry's AgentRunbook-C (74.90) and our 78.05 stand.
- **RD-Forget** (2609.10263): LongMemEval *oracle* knowledge-update subset only. Backbones are Qwen3.5-flash (API), GPT-5.6-Luna, MiniMax, Kimi.
- **LSREP** (2609.16730): a protocol paper; its answerer is gpt-5.6-luna.

## Caveats on the Hindsight rows

- **Date:** Dec 2025 (v1). It is older than the new-paper sweep but absent from `docs/sota/registry.json` and from every file under `docs/`.
- **Code:** the evaluation code is public (github.com/vectorize-io/hindsight).
- **Backbone:** gpt-oss-20b is a 21B-total / 3.6B-active MoE reasoning model with open weights. It is in class.
- **Judge:** GPT-OSS-120B is allowed to reason before a boxed verdict. That is not byte-identical to LongMemEval's gpt-4o grader, so the +2.07 margin sits within plausible judge-swap variance.
- **Next step:** re-grade their released per-question outputs with our official gpt-4o-mini grader. That would settle it.
- **An error this check caught:** a WebFetch summary gave TiMem's gpt-4o LongMemEval_S score as 87.69. The paper text says 78.96. The table uses the text value.
