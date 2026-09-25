# Research catalog addendum *(2026-09-25)*

This adds to [`sota-catalog-2026-09-24.md`](sota-catalog-2026-09-24.md) and
serves the second bottleneck plan (BACKLOG "SOTA push", 2026-09-25). The
sources came from the home-still corpus (`distill_search`) and the ids were
checked with `paper_get` / `paper_search`. Numbers are as the papers report
them. Anything marked **derived** is our own arithmetic; **unverified** means
we could not check it.

## 0. The judge behind the LoCoMo row we chase

**MemPro's LoCoMo numbers are graded by gpt-4o-mini with LightMem's lenient
prompt** ("as long as it touches on the same topic… CORRECT"; arXiv
2606.00619, L122 and Fig. 10). LightMem's code is
`zjunlp/LightMem@8449d57:experiments/locomo/llm_judge.py`; Mem0's appendix
prompt (arXiv 2504.19413) is its ancestor.

- **EdgeMem** — Cui 2026, arXiv 2609.05553. Over 20,013 paired verdicts, the
  lenient judge flips 17.7% of strict "wrong" verdicts to correct and 0.1%
  the other way.
- **Penfield Labs LoCoMo audit** (2026 blog, not peer-reviewed; cited by
  Nous, arXiv 2606.22030). 99 of 1,540 gold answers (6.4%) are wrong, and
  the lenient judge accepts up to 63% of deliberately wrong answers.
- **SwiftMem** — Tian 2026, arXiv 2601.08160. On a label-cleaned LoCoMo with
  a stricter judge, EverMemOS falls from 89.9 to 58.3. The "LoCoMo Refined"
  source is **unverified**.
- **For myelin:** M68 re-graded our answers under LightMem's exact protocol.
  The strict 70.52 becomes **78.18** (`docs/measurements/m68-matched-judge.md`).

## 1. MemPro in detail (Liu 2026, arXiv 2606.00619)

- **Protocol.** The pipeline was evolved on 154 questions, with gpt-4o-mini
  as the task model and Codex (gpt-5.4-medium) as the editor. The Qwen3-30B
  row reuses the evolved pipeline unchanged, at temperature 0.7, averaged
  over 3 runs.
- **v0 (≈ GAM, arXiv 2511.18423).** Each session gets an abstract, and there
  is a page store. A research agent runs up to 5 rounds of retrieve
  (BM25 + bge-m3 + page id) → integrate → reflect.
- **Changelog on gpt-4o-mini (App. A.1).** Every step is code-side:

  | change | score |
  |---|---|
  | start | 77.33 |
  | direct-date temporal answers | 79.10 |
  | question-type-aware integration | 80.88 |
  | count and duration reasoning | 82.12 |
  | adaptive retrieval depth | 83.46 |
  | focused evidence snippets before integration | 84.93 |

- **On Qwen3-30B, MemPro-5 → 15:** multi-hop 74.36 → 75.17, open 67.94 →
  70.83. The strength was already present by iteration 5.
- **Retrieval ablation:** removing BM25 costs −12.41.

## 2. Evidence granularity and composition

- **QueryLink** — Hu 2026, `10.18653/v1/2026.findings-acl.765`.
  - Each hit is expanded by ±c neighbouring turns (c = 0 costs −11.36 avg).
  - Write-time events plus an implicit-information view raise open-domain
    from 57.29 to 65.63 (gpt-4o-mini judge).
  - k from 4 to 8 costs open-domain 4.17.
- **JustMem** — Chen 2026, arXiv 2609.19877.
  - Units are atomic subject-explicit cards (~14 per session), ranked
    globally.
  - That beats whole-session packs by +10.72 and one-per-session by +5.52.
  - The planner emits an operation plus ≤2 answer-free rewrites. COMPOSE
    unions them, reranks against the original question, and keeps K = 10.
  - Aggregation questions: LoCoMo 68.93 → 77.97; LME-S 65.96 → 78.72.
    Overall LoCoMo 79.61 (GPT-4.1-mini).
  - REPLAY recovers source text only when needed.
- **HyperMem** — Yue 2026, arXiv 2604.08256. Hyperedges over topical
  episodes: multi-hop 93.62. Facts plus a summary of their source episode
  beat facts alone by 3–4.
- **RECOMP** — Xu 2023, arXiv 2310.04408: extractive and abstractive
  compression to ~6% of tokens. **LongLLMLingua** — Jiang 2024,
  `10.18653/v1/2024.acl-long.91`. Neither was shown on a ≤10B reader.
- **Test of Time** — Fatemi 2024, arXiv 2406.09170: fact order matters,
  entity-then-time is best. **OP-RAG** — Yu 2024, arXiv 2409.01666: source
  order gives an inverted U in k (figures **unverified**).

## 3. Aggregation and list questions

- **BEAM / LIGHT** — Tavakoli 2025, arXiv 2510.27246. A running scratchpad
  of salient entities and facts beside episodic retrieval: +3.5 to +12.69%.
- **MemoryAgentBench** — Hu 2025, arXiv 2507.05257. Smaller chunks help
  retrieval and hurt long-range tasks. A larger top-k mostly helps.
- **SwiftMem co-consolidation.** Merging fragmented memories raised 64.3 to
  78.6 in a controlled appendix setup.

## 4. Open-domain (LoCoMo category 3)

- **PPRO** — Jiang 2026, arXiv 2607.00017. A profile prior at 0.2 weight in
  ranking, plus a GRPO query rewriter. The OCR of its table is partly
  garbled.
- **Nous** — Singh 2026, arXiv 2606.22030. Open-domain failures split into
  37% retrieval misses and 42% answer misses.
- **Memory-R1** — Yan 2025, arXiv 2508.19828. With a LLaMA-3.1-8B reader,
  open-domain reaches 68.78 (trained). The category mapping is **unverified**.
- **EverMemOS** (known source). Its LoCoMo run is Episodes-only, and the
  Profile module is never called.

## 5. What we take from it

Ranked for a 9B reader, and cross-checked against our own loss anatomy
(`runs/m63_locomo_base`; multi-hop loses 114, 63 of them with partial gold):

1. **Turn windows (M66).** Evidence at turn granularity with ±2 neighbours,
   and more items in the same tokens. (QueryLink, JustMem, MemPro's "focused
   snippets", RECOMP.)
2. **Rewrites unioned for list questions (M67).** (JustMem COMPOSE, MemPro's
   type-aware integration.) It reuses `decompose.rs` (M24, unmeasured).
3. **Aggregation and counting in code** before the reader. (MemPro,
   APEX-MEM.)
4. **A write-time implicit view** for open-domain. It needs a rebuild.
   (QueryLink, PPRO.)
5. **An iterative research loop** for multi-hop and open-domain only.
   (GAM, MemPro v0.) It has the highest cost.
