# 08 — Context Engineering: Composing Retrieved Memory Into the Context Window
Design evidence pack for myelin (Rust agentic-memory backend). Line numbers cite local corpus markdown doc_id:line_start-line_end.

## Scope
How retrieved memory becomes a context window; how context degradation is measured. Covers: lost-in-the-middle / positional bias, effective-vs-advertised length (RULER/HELMET), context rot, prompt compression (LLMLingua family), KV-cache eviction (H2O, StreamingLLM, SnapKV), sliding-window/summarization compaction, progressive disclosure, tool-result clearing, subagent context isolation, and the production graduated-compaction paper.

## 0. DOI verification (MUST-REPORT)
VERIFIED PRESENT: 10.48550/arxiv.2604.14228 = Dive into Claude Code (Liu, Zhao, Shang, Shen; MBZUAI/UCL; 2026-04-14; 46pp; sha256 a4c3c920...). Catalog+markdown read. Documents the 5-layer graduated compaction pipeline. Numbers in Sec 9.
ABSENT (searched): LLMLingua-2, SnapKV, HELMET, standalone RULER paper (RULER numbers via MEMAGENT 2507.02259, cited as Hsieh et al 2024), no paper titled context rot (closest: LoCoBench-Agent 2511.13998, LoCoMo 2402.17753, LongMemEval 2410.10813).

## 1. Lost in the middle / positional bias (2307.03172)
- (3) Ordering, definitive: best at very start (primacy) or very end (recency), degrades middle, U-shaped (2307.03172:18-27, Fig1 :1-18). GPT-3.5-Turbo mid-context multi-doc QA BELOW closed-book 56.1% (2307.03172:98-153 Table1). Oracle 88.3%, closed-book 56.1%.
- 10/20/30 doc contexts tested (2307.03172:98-153). 7B Llama-2 recency-only; U-curve at >=13B; instruction FT cuts positional disparity 10%->4% (MPT-30B) (2307.03172:236-249). Extended context != better: 16K~4K GPT-3.5, 100K~8K Claude (2307.03172:98-153).
- (4) Reader accuracy saturates long before retriever recall in open-domain QA -> extra docs add noise (2307.03172:236-249 Fig11).

## 2. Effective vs advertised length; context rot
- RULER-HQA (via 2507.02259): Qwen2.5-Instruct-1M OK to 112K, ~0 at 896K, before 1M capacity (2507.02259:546-571). MEMAGENT 8K->3.5M <10% loss; >95% @512K NIAH (2507.02259:1-25). Default: memory 1,024 tokens, chunks 5,000, total <=8,192 (2507.02259:567-593); ablation 256-4096.
- LoCoBench-Agent (2511.13998): comprehension peaks ~15-20 turns; >12 turns only 1-2% gain for 50-60% more conversation (2511.13998:486-502). 1M windows don't beat 200K (2511.13998:486-502); cited LoCoBench 29%->3% for Claude 3.5 Sonnet @1M (2511.13998:649-669).
- LongMemEval (2410.10813): full ~115k history -> 30-60% drop vs oracle-evidence-only (2410.10813:260-365); ChatGPT/Coze on GPT-4o drop 37%/64% vs offline reading (2410.10813:260-365).
- LoCoMo (2402.17753): gpt-3.5-turbo adversarial 2.1% on long context vs 70.2% GPT-4-turbo @4K -> long context induces hallucination (2402.17753:486-569).

## 3. Prompt compression accuracy/token curves
- LLMLingua (2310.05736): coarse-to-fine, small-LM perplexity ranks tokens, budget controller. Up to 20x with little loss (2310.05736:1-17). Instruction+question keep more budget; demos redundant tail (2310.05736:61-97). tau=5: ~0.264*L*c ~ 4x savings (2310.05736:463-513). E2E latency (V100, GSM8K): 8.6s 1x -> 4.9s@2x(1.7x) -> 2.3s@5x(3.3x) -> 1.3s@10x(5.7x) (2310.05736:463-513 Table6).
- LongLLMLingua (2310.06839): question-aware + document reordering + dynamic ratio + subsequence recovery. +21.4pp NaturalQuestions (doc@pos10) with ~4x fewer tokens (2310.06839:161-169). 94.0% cost cut LooGLE (2310.06839:1409-1445). @2x-6x of ~10k prompts -> 1.4x-2.6x E2E latency (2310.06839:1-16). 3x/3000-token GPT-3.5: AVG 48.8 vs BM25 40.6, SBERT 41.4, OpenAI 41.7, SelectiveContext 32.0, LLMLingua 37.4; 3,283 tokens; 1.6x (2310.06839:265-397). (3) doc reordering mitigates lost-in-the-middle; helps all baselines incl timeline-heavy LooGLE (2310.06839:1409-1445;650-668). Cost/1k: -$3.3(71.7%) MDQA; -$28.5(90.5%) LongBench; -$88.0(94.0%) LooGLE (2310.06839:1409-1445 Table9).
- Selective Context (2310.06201): self-information pruning. 50% reduction -> 36% less mem, 32% less time, only -0.023 BERTscore / -0.038 faithfulness (2310.06201:1-33).

## 4. KV-cache eviction
- H2O (2306.14048): attention >95% sparse; heavy-hitters carry most. 20% HH+recent matches full cache -> 5-10x mem reduction (2306.14048:198-222). Throughput FlexGen/DeepSpeed/Accelerate 3x/29x/29x; latency -1.9x (2306.14048:1-14). Recency-only Local collapses at 60% budget; sparse baselines lose up to 35% @20%; H2 composes to parity (2306.14048:221-277).
- StreamingLLM (2309.17453): models dump attention onto initial sink tokens; 4 initial sinks + sliding window preserves perplexity, stable to 4M tokens, up to 22.2x speedup (2309.17453:1-21;324-343). Ablation: 1-2 sinks insufficient, 4 threshold (2309.17453:324-343). Positional encoding within cache, not original sequence (2309.17453:150-171).
- Implication: KV eviction must retain sink prefix + recent tokens, not blind recency; orthogonal to prompt-level compaction.

## 5. Sliding-window & summarization
- MEMWALKER (2310.05029): recurrence (carry fixed summary) loses query info after a few steps; retrieval beats recurrence; MEMWALKER (tree nav) best: QuALITY 67.4/73.6, SummScreenFD 67.3/64.5, GovReport 59.4/60.4 @4,096-window, max 8 nodes / 1,000-1,200-token segments (2310.05029:88-151). Left-vs-right truncation dataset dependent. Recurrence budget seg 2,500->summary <=500 (5:1) (2310.05029:88-151).
- Mem-alpha (2509.25911): RL compaction reward r3 = 1 - l_m/l_c (2509.25911:1361-1417).

## 6. Progressive disclosure / JIT retrieval
- CoALA (2309.02427): retrieval reads long-term->working at decision time; Generative Agents combine recency+importance+relevance for episodic recall (2309.02427:208-223). JIT frame: retrieve into working memory at compose step.
- Dive into Claude Code (2604.14228): CLAUDE.md + path-scoped rules lazily load when agent reads matching dirs; ToolSearch defers tool schemas; memory prefetch async (2604.14228 Sec4/7).

## 7. Tool-result clearing & subagent isolation (2604.14228)
Per-tool-result budget reduction caps oversized outputs -> content references (reconstructable on resume); runs before microcompact, composes cleanly (2604.14228 4.3). Subagents return only a summary to parent; full history in sidechain transcripts (sessionStorage.ts:247), never enters parent window (2604.14228 8.3). Agent teams ~7x tokens of standard session in plan mode -> summary-only return critical (2604.14228 8.3). Compaction append-only; boundary marker records headUuid/anchorUuid/tailUuid for chain re-patching (2604.14228 9).

## 8. Evidence-set bloat (HiGMem, VERIFIED, 2604.18349)
Verified verbatim: vector-only retrieval produces bloated evidence sets: once most relevant memories recalled, adding superficially similar fragments yields diminishing recall but steadily erodes retrieval precision, inflates answer-stage context, harder to inspect (2604.18349:1-18).
Numbers (2604.18349:99-163;451-593): A-Mem 99.84 turns P@K .0101 R@K .7502; HiGMem 8.09 turns P@K .1909 R@K .7241; A-Mem@8 .059/.385; @16 .038/.478; @32 .024/.580. ~19x higher precision, comparable recall. Adversarial F1 0.54->0.78; best F1 4/5 LoCoMo10 (2604.18349:1-18).
Cost hybrid GPT-4o-mini+GPT-5: $17.43->$6.43 (2.7x); answer tokens 25.4M->1.6M (~12.8K->0.8K/question) (2604.18349:254-304 Table10).
(4) count: k_turn=10, k_event=10 (2604.18349:99-163); 8.09 natural output is the before-degradation datum. Counterpoint: Cuconasu et al. (in 2312.10997:438-445) irrelevant docs can aid >30% in some settings -> task-conditional.
Corroboration: Shuster (2021.findings-emnlp.320) Knowledge-F1 drops as more docs retrieved (320:2608-2693); LoCoMo RAG peaks at top-5 observations (+5%), falters as more added (2402.17753:486-569).

## 9. Compaction triggers & fill ratio
Claude 5-layer graduated pipeline (2604.14228 4.3/7.3): 1 budget reduction (always on, per-tool caps); 2 snip (HISTORY_SNIP, trim older history, tokensFreed plumbed to auto-compact); 3 microcompact (CACHED_MICROCOMPACT, uses cache_deleted_input_tokens); 4 context collapse (CONTEXT_COLLAPSE, read-time virtual projection, no mutation); 5 auto-compact (default-on, full model summary, only if still over threshold after 4 cheaper layers) -> lazy cheapest-first.
Triggers: prompt_too_long -> context-collapse overflow recovery + reactive compaction (REACTIVE_COMPACT, at most once/turn) then terminate; max-output escalation up to 3 (MAX_OUTPUT_TOKENS_RECOVERY_LIMIT=3); pre-compact hooks first; cache-reuse false path 98% cache miss / 0.76% fleet (2604.14228 4.4/7.3).
Fill anchors: MEMAGENT total 8,192 with 5,000 chunk + 1,024 memory -> memory ~12.5% of window per chunk (2507.02259:567-593); Mem-alpha 1 - l_m/l_c; MEMWALKER 5:1 (2310.05029:88-151).

## 10. Composition policy (parameterized, cited)
Assemble memory block right before the model call at compose stage. Deterministic harness-side; models not trusted to self-manage context (2604.14228 3.2).
### 10.1 Ordering
- High-priority items to start AND end; never strand the single hardest fact mid-context (U-curve 2307.03172:18-27). Two primacy/recency bookend slots.
- Descending score then LongLLMLingua-style reordering pulling question-critical/answer-bearing items to head (2307.03172; 2310.06839:650-668). Query/instruction at very end (recency), system at very start (primacy/sink).
- 4-token attention-sink slot at absolute start where supported (2309.17453:324-343).
- Group by kind: system-sink -> semantic/facts (highest precision) -> episodic (temporal) -> procedural/instructions -> working/current-query (recency-heavy last).
### 10.2 Item count
- Hard cap MAX_ITEMS=10 per pass (HiGMem k_turn=k_event=10, 2604.18349:99-163). Natural ~8 (2604.18349:451-593). Do not grow set to chase recall (2604.18349:1-18; Shuster 320:2608-2693).
- Soft budget TOP_K_ANSWER=5 (LoCoMo SNR peak, 2402.17753:486-569); expand 8-10 only with rerank confirmation.
- Dedup near-duplicate pre-rank (HiGMem event layer -> 8.09 effective, 2604.18349:33-65).
### 10.3 Token budgets per kind
For window (model context limit): episodic 1,024 (MEMAGENT 2507.02259:567-593); working/current-chunk 5,000 (2507.02259:567-593); summary ratio 5:1 (2500->500, MEMWALKER 2310.05029:88-151); total composed memory <=12.5% of window (1,024/8,192, 2507.02259:567-593) unless task-verified. Remaining to system prompt/instructions/tail/headroom; never pack 100%, keep headroom for reactive compaction (2604.14228 4.4).
### 10.4 Dedup & provenance metadata
Each item MUST carry: kind, source_doc_id/stem (DOI where available), line/page pointer, created_at/event_id, embedding_id/vector key, retrieval score — modeled on HiGMem Turn/Event nodes (keyword/tag/timestamp/context; bidirectional links, 2604.18349:33-65) and Claude Code append-only boundary UUIDs for chain patching (2604.14228 9). Eval axes grounding/provenance and robustness/poisoning satisfiable only if every surfaced memory traces to a verifiable record and every compaction writes an auditable boundary marker (append-only invariant).

## 11. Highest-leverage findings for Rust
1. Ordering is a free accuracy lever: primacy+recency bookending + question-aware reorder beat naive score-order; mid-context deciding fact can drop below closed-book 56.1% — no harness should make that error by default (2307.03172; 2310.06839).
2. Evidence-set precision beats recall: 10 items ~ P .19/R .72; 100 items -> P .01. Cap k=8-10, never grow to chase recall (2604.18349; 2402.17753).
3. Graduated cheapest-first compaction (5 layers, no single-pass truncation) is the only production-validated pattern (2604.14228 4.3/7.3).
4. Effective context << advertised even at 2026 quality: 112K-of-1M usable, ~15-20 turns, 1M ~ 200K. Budget as if usable is 1/8th advertised (2507.02259; 2511.13998; 2410.10813).
5. Question-aware compression is cheap and lossy-tolerant: 3-5x reduction with a gain (LongLLMLingua +21.4pp @4x; 94% cost cut) — combine entropy-tier + question-tier + KV-tier (2310.06839/2310.05736/2306.14048).

## Sources table (all corpus-verified)
| DOI/doc_id | Title | Used for |
|---|---|---|
| 10.48550/arxiv.2307.03172 | Lost in the Middle | ordering |
| 10.48550/arxiv.2310.05736 | LLMLingua | ratios |
| 10.48550/arxiv.2310.06839 | LongLLMLingua | reorder/cost |
| 10.48550/arxiv.2310.06201 | Selective Context | ratios |
| 10.48550/arxiv.2306.14048 | H2O | KV |
| 10.48550/arxiv.2309.17453 | StreamingLLM | sinks |
| 10.48550/arxiv.2604.18349 | HiGMem | bloat (VERIFIED) |
| 10.48550/arxiv.2604.14228 | Dive into Claude Code | compaction (VERIFIED present) |
| 10.48550/arxiv.2507.02259 | MEMAGENT (RULER-HQA) | effective length/budgets |
| 10.48550/arxiv.2410.10813 | LongMemEval | rot |
| 10.48550/arxiv.2402.17753 | LoCoMo | rot/SNR |
| 10.48550/arxiv.2511.13998 | LoCoBench-Agent | rot |
| 10.48550/arxiv.2310.05029 | MEMWALKER | summary ratio |
| 10.48550/arxiv.2309.02427 | CoALA | progressive disclosure |
| 10.48550/arxiv.2509.25911 | Mem-alpha | compression reward |
| 10.48550/arxiv.2405.13792 | xRAG | embedding-compression pointer |
| 10.18653/v1/2021.findings-emnlp.320 | RAG hallucination | bloat corroboration |

Unverified/absent (searched, honest absence): LLMLingua-2, SnapKV, HELMET, standalone RULER, dedicated context rot paper.
