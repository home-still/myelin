# Research catalog for closing the SOTA gaps *(2026-09-24)*

Built from the home-still corpus (`distill_search`) and checked with
`paper_search` / `paper_get`. Ids were checked with those tools unless marked
otherwise. Numbers are as the papers report them, and marked **derived**
where we did the arithmetic. It serves the bottleneck plan of the same day,
which lives in the session plan and `BACKLOG.md`.

## Where the gaps are (our measurements, for context)

| benchmark | ours | same-size row | gap |
|---|---|---|---|
| LongMemEval_S | 83.40 (Bonsai + premise clause) | 80.80 (MemPro-15, Qwen3-30B) | +2.60 |
| LoCoMo judge 1–4 | 69.87 (Qwen3.5-9B) | 77.85 (MemPro-15, Qwen3-30B) | −7.98 |
| LME-V2 | 38.80 (myelin RAG memory) | 74.90 (AgentRunbook-C) | −36.10 |

**The LoCoMo gap by category** (derived, MemPro-15 Qwen3-30B-A3B per
category, weighted by n): multi-hop ~3.2 points (57.8 vs 75.17),
open-domain ~2.6 (29.2 vs 70.83), temporal ~1.5 (60.4 vs 67.60), and
single-hop ~0.7 (82.2 vs 83.47). The category-id mapping differs between
papers. Checked on our data: 1 = multi-hop (list aggregation across
sessions), 2 = temporal, 3 = open-domain (world-knowledge inference),
4 = single-hop, 5 = adversarial.

**LME-V2's 74.90 used a proprietary controller** (GPT-5.4-mini, xhigh
reasoning). The best same-size local-controller row in the paper is
AgentRunbook-R at 58.6.

## 1. Temporal reasoning

- **Chronos** — Sen 2026, arXiv 2603.16862.
  - Mechanism: subject-verb-object event tuples with resolved datetime
    ranges in an event calendar beside a turn calendar; per-question
    retrieval guidance; a tool loop with date filter, grep and vector
    search, and reranking.
  - Numbers: LongMemEval_S 92.60 (GPT-4o), 95.60 (Claude Opus 4.6).
  - Ablation: removing the events costs the weak variant 34.5 points and
    the strong one 2.6. Vector-only search drops the strong variant from
    94.8 to 83.6.
  - For us: the events help a weaker reader most. The hybrid search is
    load-bearing. *(M50c.)*
- **TReMu** — Ge 2025, `10.18653/v1/2025.findings-acl.972`.
  - Mechanism: timeline summaries with relative dates inferred at write
    time; the LLM writes and runs Python date code.
  - Numbers: LoCoMo-derived temporal multiple choice, 29.83 → 77.67
    (GPT-4o).
  - For us: hand date arithmetic to code, not to the reader.
- **Zep** — Rasmussen 2025, arXiv 2501.13956.
  - Mechanism: a temporal knowledge graph with validity intervals on
    edges; hybrid cosine + BM25 + graph search with reranking.
  - Numbers: LongMemEval_S 71.2 vs 60.2 full-context; temporal 36.5 →
    54.1.
- **LongMemEval time-aware query expansion** — Wu 2024, arXiv 2410.10813.
  - Mechanism: a time-range filter from the query.
  - Numbers: temporal R@10 0.550 → 0.722 with GPT-4o; a Llama-3.1-8B
    extractor hallucinates ranges and gains ~0.
  - For us: derive ranges deterministically.

## 2. Multi-hop, multi-session aggregation, counting

- **EviMem** — Li 2026, arXiv 2604.27695.
  - Mechanism: the evidence is graded EXACT / INFERRABLE / PARTIAL; the
    diagnosed gap drives the next query, with per-entity tracking and an
    abstention on insufficient evidence.
  - Numbers (LoCoMo): multi-hop 85.2 (single-pass 81.4); temporal 81.6
    (single-pass 58.8); 9.5 s per query.
- **APEX-MEM** — Banerjee 2026, `10.18653/v1/2026.acl-long.749`.
  - Mechanism: a property graph with time-anchored events, and a ReAct
    agent with read-only SQL.
  - Numbers: LoCoMo 88.88 (GPT-5). Adding SQL: temporal 72.9 → 82.3
    (Haiku 4.5).
  - For us: counting as SQL `COUNT`.
- **PREMem** — Kim 2025, `10.18653/v1/2025.findings-emnlp.1204`.
  - Mechanism: pre-storage reasoning, fragments linked across sessions at
    write time.
  - Numbers: exact figures unverified (garbled corpus copy).
- **MRAgent** — arXiv 2606.06036 (first author not captured).
  - Mechanism: an iterative cue-tag-episode graph walk.
  - Numbers: multi-hop recall +30% across turns; single-hop and temporal
    saturate by turn ~3.
  - For us: route iteration to multi-hop.
- **Chain-of-Note** — Yu 2024, `10.18653/v1/2024.emnlp-main.813`.
  - Mechanism: a note per document before answering.
  - Numbers: +7.9 EM with noisy docs, +10.5 correct rejections.
  - For us: the basis for enumerate-then-count.
- **HippoRAG 2** — Gutiérrez 2025, arXiv 2502.14802.
  - Mechanism: PPR over a passage+phrase graph.
  - Numbers: +7% associative retrieval.

## 3. When to answer: abstention and calibration

- **Sufficient Context** — Joren 2024, arXiv 2411.06037.
  - Finding: small models abstain or hallucinate even with sufficient
    context. A separate sufficiency autorater drives selective
    generation: +2–10% correct-among-answered.
  - For us: Bonsai's 132 refusals with every gold turn in evidence.
- **Trust-Align** — Song 2024, arXiv 2409.11242.
  - Finding: prompting fails to fix refusals; preference alignment on
    grounded refusals does (+12.6 ASQA, LLaMA-3-8B).
  - For us: matches M59.
- **Refusal Tokens** — Jain 2024, arXiv 2412.06748.
  - Mechanism: a threshold on the refusal-token probability sets the
    refusal rate without retraining.
- **Semantic entropy** — Farquhar 2024, `10.1038/s41586-024-07421-0`.
  - For us: M45 measured AUROC 0.59 on our 9B.
- **Self-Consistency Falls Short** — Byerly & Khashabi 2026,
  `10.1162/tacl.a.625`.
  - Finding: position bias on long context, worse for small models.
  - For us: do not vote over long evidence.
- **Know Your Limits** — Wen, `10.1162/tacl_a_00754`.
  - Contribution: abstention metrics. Report the answerable and
    adversarial splits separately.

## 4. Agent-trajectory memory and controllers

- **LongMemEval-V2** — Wu 2026, arXiv 2605.12493.

  | method | controller | Small | Medium |
  |---|---|---|---|
  | query→slice | Qwen3.5-9B | 42.8 | 38.1 |
  | query→slice + notes | Qwen3.5-9B | 53.1 | 45.9 |
  | AgentRunbook-R | Qwen3.5-9B | 58.6 | 57.0 |
  | vanilla Codex | GPT-5.4-mini xhigh | 69.9 (177 s) | 68.7 |
  | AgentRunbook-C | GPT-5.4-mini xhigh | 74.9 (108 s) | 70.1 |

  - AgentRunbook-R keeps three pools: raw state slices, transition events,
    and procedure/hint notes. Without the slices, static questions drop
    0.541 → 0.286.
  - AgentRunbook-C adds a workflow document, a manifest shortlist, and a
    span/search helper, and returns a note plus spans.
- **Coding Agents are Effective Long-Context Processors** — Cao 2026,
  arXiv 2603.20432.
  - Mechanism: text in a file system, worked with shell and code.
  - Numbers: +17.3% over published SOTA.
  - For us: why M62's bespoke schema tools lose.
- **BATS / Budget Tracker** — Liu 2025, arXiv 2511.17006.
  - Finding: agents lack budget awareness, so more tool calls plateau. A
    per-step remaining-budget signal moves the frontier.
  - For us: M62's 77% forced answers.
- **SMART** — Qian 2025, `10.18653/v1/2025.findings-acl.239`.
  - Mechanism: SFT on when a tool is needed.
  - Numbers: −24% tool use, +37% performance.
- **Agent Workflow Memory** — Wang 2024, arXiv 2409.07429.
  - Mechanism: induced reusable workflows.
  - Numbers: Mind2Web cross-domain 18.6 → 35.5.
- **ReasoningBank** — Ouyang 2025, arXiv 2509.25140.
  - Mechanism: strategies from successful and failed trajectories.
  - Numbers: unverified.

## 5. Retrieval and reading

- **LongMemEval (v1)** — Wu 2024, arXiv 2410.10813.
  - Round-level values plus fact-expanded keys: session R@10 0.783 →
    0.862; QA 0.676 → 0.714. Fact decomposition helps multi-session.
    Chain-of-Note plus JSON evidence, sorted by time.
- **SeCom** — Pan 2025, arXiv 2502.05589.
  - Topic-segment units plus compression: up to +11.98 on LoCoMo.
- **Context Length Alone Hurts** — Du 2025,
  `10.18653/v1/2025.findings-emnlp.1264`.
  - Accuracy drops 13.9–85% with input length even at perfect retrieval.
    Recite-then-answer recovers some. For us: the risk in raising k.
- **Lost in the Middle** — Liu, arXiv 2307.03172. A U-shaped position
  effect.

## 6. Top LoCoMo / LongMemEval systems

- **MemPro** — Liu 2026, arXiv 2606.00619.
  - Mechanism: failure-driven evolution of the whole pipeline over 15
    iterations, driven by a Codex agent.
  - Numbers: Qwen3-30B-A3B LoCoMo 77.85, LongMemEval 80.80. It is the only
    same-size open-model row.
- **EverMemOS** — Hu 2026, `10.18653/v1/2026.acl-long.2125`.
  - Mechanism: MemCells → MemScenes, hybrid retrieval, and a sufficiency
    check that rewrites 31% of LoCoMo queries.
  - SwiftMem (arXiv 2601.08160) reports the lead vanishes on a
    label-cleaned LoCoMo.
- **Nemori** — Nan 2025, arXiv 2508.03341.
  - Numbers: LoCoMo 73.0 (gpt-4o-mini), 80.8 (gpt-4.1-mini).
  - Weakest on open-domain, which needs world knowledge. That explains our
    29.2.
- **Mem0** — Chhikara 2025, arXiv 2504.19413. ADD / UPDATE / DELETE /
  NOOP. Independent reruns vary widely.
- **Also reported (mostly proprietary readers):**
  - LycheeMemory V2 (arXiv 2608.12990): 89.22 LoCoMo.
  - Memanto (arXiv 2604.22085): 87.1.
  - LiCoMemory (`10.18653/v1/2026.findings-acl.1835`).
  - SGMem (arXiv 2509.21212).
  - A-Mem (arXiv 2502.12110).
  - MIRIX (arXiv 2507.07957, id not independently verified).

## Ranked by evidence per effort, for LoCoMo and LME-V2

1. **Take answer/abstain away from the reader:** a sufficiency gate, and
   the reader answers when it says "sufficient". *(Sufficient Context,
   EviMem, Trust-Align.)*
2. **Gap-driven second retrieval for multi-hop only.** *(EviMem, MRAgent,
   EverMemOS.)*
3. **Event calendar plus date arithmetic in code.** *(Chronos, TReMu.)*
4. **AgentRunbook-C's scaffolding in the native controller**, or files and
   shell. *(LME-V2, Cao 2026.)*
5. **Budget awareness plus a forced commit.** *(BATS.)*
6. **AgentRunbook-R's three pools on the RAG side.** *(LME-V2, AWM,
   ReasoningBank.)*
7. **Fact-keyed index expansion plus enumerate-then-count, or SQL COUNT.**
   *(LongMemEval, APEX-MEM, Chain-of-Note.)*
8. **Preference-align the reader on its over-refusals, or distil the
   controller traces.** *(Trust-Align, SMART.)*

**Avoid:**
- self-consistency over long evidence;
- comparing per-category numbers without checking the category mapping;
- trusting LoCoMo gains without a label-cleaned check.
