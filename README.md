Vocabulary of the field

 Forms/structure — agentic memory · long-term / short-term / working memory · episodic · semantic · procedural · declarative · autobiographical ·
 memory stream · memory bank · memory graph · knowledge graph · temporal knowledge graph · entity+relation · hierarchical / multi-level / tiered /
 layered memory · memory tier · virtual context · token-level vs parametric vs latent memory

 Operations — memory writing · memory reading · memory management · consolidation · abstraction · summarization / gist · reflection · compression ·
 compaction · distillation · forgetting / decay · eviction · pruning · salience / importance scoring · recency · deduplication · conflict
 resolution · merge-update-delete · memory evolution · self-evolving / self-organizing memory · lifelong / continual learning · catastrophic
 forgetting · memory governance

 Retrieval — semantic search · dense retrieval · embeddings · vector store / ANN / HNSW / IVF · sparse / lexical retrieval / BM25 · hybrid
 retrieval · reranking · cross-encoder · late interaction / ColBERT · rank fusion / RRF · multi-hop · query decomposition · query rewriting ·
 iterative retrieval · adaptive retrieval ("when to retrieve") · self-RAG / corrective RAG · just-in-time retrieval · agentic search ·
 scope-before-routing · query-aware indexing · admissibility vs relevance · shard-probe budget

 Context engineering — context window management · context budget / attention budget · context rot · lost-in-the-middle · positional bias · prompt
 compression / LLMLingua · KV-cache eviction · attention sink · sliding window · quantization · chunking / segmentation · progressive disclosure ·
 tool-result clearing · subagent context isolation · summary-only returns · context handoff · deferred tool schema loading · microcompact / snip /
 context collapse

 Evaluation & risk — token efficiency · latency / p95 · grounding · hallucination · faithfulness · provenance · memory poisoning / injection ·
 memory access control · LLM-as-a-judge · trajectory · skill library · experience replay · scratchpad · personalization / user profile ·
 metacognition · cognitive architecture · hippocampus-inspired · Zettelkasten

 Prevalence numbers (share of the 118 docs that scored as in-field) are in the report — e.g. hierarchical 47.5%, embeddings 53.4%, agentic memory
 28.8%, agentic search only 2.5%, progressive disclosure 0.8%. Note the last two: the terms your prompt uses are the newest vocabulary in the field
 and the least represented in the literature — they come from production agent harnesses, not papers.

 The 63 articles

 Grouped, all verified present: 6 surveys/position (memory-mechanism survey 2024, episodic-memory position 2025, CoALA, SSGM governance 2026), 15
 systems (MemGPT, Generative Agents, MemoryBank, A-MEM, Mem0, Zep, HippoRAG/HippoRAG2, MIRIX, G-Memory, GAM, HiGMem, H-MEM, MOOM, Collaborative
 Memory), 5 learned-memory-policy papers (Mem-α, NEMORI, Live-Evo, Meta-Cognitive Memory Policy, HAGE), 3 retrieval-infra (SwiftMem, ShardMemo,
 QueryLink) + 1 hardware (MemExplorer), 14 RAG/retrieval foundations, 10 context/compression, 6 benchmarks (LongMemEval, LoCoMo, MemBench,
 LoCoBench-Agent, Ragas), 1 security (memory poisoning).

 SOTA for concise context retrieval

 Hybrid retrieval as the primitive, agentic search as the control layer, graduated compaction for the window. The wins come from ordering, fusing,
 and pruning before injecting — not from any single technique.

 1. BM25 + dense in parallel, fused with RRF, then rerank (cross-encoder/late-interaction), pass only top 5–10 chunks. RRF is the default because
    it dodges cross-retriever score normalization; reranking is the highest-leverage stage for conciseness.
 2. Constrain before ranking. ShardMemo's scope-before-routing is the 2026 pattern: metadata predicates mask inadmissible memories first, then a
    learned router spends a bounded shard-probe budget. Post-filtering wastes budget on inadmissible memories (+3 F1 on LoCoMo at fixed budget).
 3. Lexical beats dense for exact-match domains (code symbols, filenames, config keys, logs); dense wins for paraphrase-heavy corpora; adaptively
    route.
 4. Just-in-time loading beats pre-indexing when the corpus churns. Claude Code is the reference: lightweight identifiers + runtime grep/glob/read
    instead of pre-embedding, and an LLM scan of file headers rather than a vector index for memory retrieval. (Source-verified in the corpus paper
    10.48550/arxiv.2604.14228.)
 5. Five-layer graduated compaction (Claude Code): budget reduction → snip → microcompact → context collapse → auto-compact. Clear/mask stale
    re-fetchable tool outputs before lossy summarization; trigger at 60–70% of the effective window and at task boundaries; pin head+tail verbatim,
    compact only the middle; structured summaries with exact identifiers preserved.
 6. Isolate the window, not just shrink it — deferred tool schemas, lazy instruction loading, summary-only subagent returns.

 Concrete corpus-verified numbers: SwiftMem 11.7 ms/query vs 794–1264 ms/query for Nemori/LightMem/EverMemOS at comparable judge score
 (query-agnostic full-space retrieval is the bottleneck, not the ANN index). HiGMem: the failure mode is bloated evidence sets — extra
 superficially similar turns add little recall but erode precision. Mem0: 91% lower p95 latency, >90% token cost vs full-context, +26% relative
 LLM-judge over OpenAI. Zep: 94.8% DMR vs MemGPT 93.4%.

 Still unsolved: multi-session long-horizon tasks; benchmark disagreement (LoCoMo/LongMemEval/BEAM leaders differ, hence "LoCoMo Refined");
 predictable code-quality degradation from lossy compaction; memory as an attack surface (>95% injection success under idealized conditions).
