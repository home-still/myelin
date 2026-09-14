# 06 — Consolidation & Forgetting (write-side lifecycle)

Evidence pack for designing `myelin`. Scope: what gets stored, how it is abstracted, how it decays, how conflicts resolve, and the temporal model. Every factual number carries a DOI/doc_id and line pointer into converted markdown (page-1 lines from `distill_search`). Values I could not verify in the corpus are marked `[UNVERIFIED]` with what I searched.

**Corpus papers actually read (DOI list):** 10.48550/arxiv.2304.03442 (Generative Agents); 10.48550/arxiv.2305.10250 (MemoryBank); 10.48550/arxiv.2502.12110 (A-MEM); 10.48550/arxiv.2501.13956 (Zep/Graphiti); 10.48550/arxiv.2405.14831 (HippoRAG); 10.48550/arxiv.2502.14802 (HippoRAG 2); 10.48550/arxiv.2310.08560 (MemGPT); 10.48550/arxiv.2501.07278 (continual/lifelong survey); 10.48550/arxiv.2508.03341 (NEMORI cost tables); 10.48550/arxiv.2303.11366 (Reflexion).

## 1. What gets stored: record kinds by system

| System | Kinds stored | Abstraction | Store |
|---|---|---|---|
| Generative Agents (2304.03442 L283-300) | working-adjacent observations + semantic reflections | raw observation leaf -> abstract reflection non-leaf (tree) | memory *stream* text + embedding + timestamp + importance; fill context window |
| MemoryBank (2305.10250 L27-44) | episodic dialogue logs + semantic event summaries + user portrait | 3 tiers: dialogue -> daily-event summary -> global summary | timestamped chronological store; hierarchical LLM distil (prompt "Summarize the events and key information in the content [dialog/events]") |
| A-MEM (2502.12110 L74-127) | note per interaction {keywords, context, tags} + embedding | flat atomic notes + self-organizing "boxes" (links) | LLM note; cosine retrieval; LLM link + evolution edges |
| Zep/Graphiti (2501.13956 L32-49,64-87) | episodic episodes (raw) + semantic entity/edge facts + community summaries | 3 subgraphs: episode G_e -> semantic entity G_s -> community G_c | raw episodes non-lossy (bidirectional index to semantic); LLM edges with bi-temporal stamps |
| HippoRAG (2405.14831 L46-59) | semantic KG triples (OpenIE) as hippocampal index | schemaless open KG | LLM OpenIE; encoder synonymy/cosine edges; PPR over graph |
| MemGPT (2310.08560 L58-85) | working-context block + archival storage + recall-storage messages | context tiers: main (prompt) vs external (disk) | FIFO queue + recursive summary at head; function-call paging |

**Takeaway:** a clean four-kind split maps directly: episodes (raw, non-lossy, Zep-style), semantic (facts/triples, HippoRAG), procedural (skills), working (MemGPT in-context block). Strongest lossless+derived precedent = Zep episode<->semantic bidirectional index (2501.13956 L49-64).

## 2. Consolidation & reflection

### 2.1 Generative Agents reflection tree (2304.03442)
(1) **Scoring.** Importance = LLM poignancy integer 1-10; worked examples give 2 for "cleaning up the room", 8 for "asking your crush out on a date"; assigned at creation (L283). Relevance = cosine similarity between memory embedding and query embedding. Full retrieval score (L283):

    score = a_recency*recency + a_importance*importance + a_relevance*relevance

All three min-max normalized to [0,1]; all alpha = 1 in reference impl.
(2) **Trigger:** periodic — when the **sum of importance scores of latest events exceeds threshold 150** -> ~2-3 reflections/day (L283-300). Accumulator/budget trigger: deterministic accumulation of LLM-produced ints.
(3) **Tree algorithm (L300-317):** query LLM with the **100 most recent records** -> "what are 3 most salient high-level questions we can answer about the subjects?"; use questions as retrieval queries (gather observations AND prior reflections); prompt "What 5 high-level insights can you infer from the above statements? (insight (because of 1,2,8,15))"; parse + store with **pointers to cited memory objects**. Leaves=observations, non-leaves=reflections, higher=more abstract.
(4) **Measured effect:** full arch beat all ablations + human crowdworkers on believability (Kruskal-Wallis H(4)=150.29, p<0.001; all pairwise p<0.001, L441-454). Reflection specifically aided cross-memory synthesis (Maria choosing Wolfgang gift, L454). End-to-end: info diffusion of 2 facts across 25 agents over 2 days, source dialogue located in memory stream (L454-466).
(5) **Failure modes (documented L441-454):** failed retrieval of correct instance; incomplete memory fragments (Tom knew what to do at party, not that it existed); hallucinated embellishments (Isabella: new announcement not discussed); world-knowledge contamination (Yuriko: neighbor = author of Wealth of Nations).
(5/3) **Cost:** per reflection = 1 question-gen call + up to 3 retrieval queries + 1 insight-gen call; ~2 reflections/day -> ~4-9 LLM calls/day per agent [INFERENCE from triggers; corpus states triggers/steps, not raw token counts]. Deterministic: importance accumulation, threshold compare, min-max, pointer store.

### 2.2 Reflexion (2303.11366)
Verbal reinforcement: binary/scalar feedback amplified to natural-language experience summaries in an episodic buffer (L5-13). AlfWorld +22 abs in 12 steps; HotPotQA +20%; HumanEval pass@1 91% (vs GPT-4 80%). Cost: 1 reflection call per failed trial; no weight updates. Failure: relies on LLM self-evaluation; "no formal guarantee for success."

## 3. Gist / abstraction hierarchies

### 3.1 MemoryBank hierarchical summary (2305.10250 L27-44)
Two-level LLM distil: daily dialogues -> "daily event summary"; events -> "global summary". **Trigger:** day-boundary (periodic). **Cost:** 2 LLM calls/day. Deterministic: day segmentation. Failure: info loss through repeated summarization (motivation for Zep non-lossy, 2501.13956 L32).

### 3.2 Zep/Graphiti community subgraph (2501.13956 L84-103)
Map-reduce summarization (GraphRAG lineage) -> community nodes, names embedded for cosine retrieval. **Trigger:** dynamic single-step label-propagation when node joins (assign to plurality community, update summary), + **periodic full refresh** to repair drift (L84-103). **Cost:** map-reduce multi-call; dynamic cheaper than full refresh ("reduces latency and LLM inference costs"). Failure: dynamic communities diverge from full label-propagation -> refresh needed.

### 3.3 HippoRAG vs RAPTOR/GraphRAG (2405.14831 L467-486)
RAPTOR/GraphRAG integrate by **summarizing** -> re-summarization on every add. HippoRAG integrates by **just adding edges** (non-lossy). Cost (2502.14802 table 12): HippoRAG 2 indexing input 9.2M / output 3.0M tokens on MuSiQue (11,656 passages); indexing 57.5 min (HippoRAG) / 99.5 (HippoRAG2); QA 0.9-1.2 s/query. RAPTOR 1.7M in/0.2M out but re-summarizes; LightRAG 68.5M in; GraphRAG 115.5M in (100% = HippoRAG). Failure: identifier/entity loss in summaries; HippoRAG's noun-phrase OpenIE chosen over summarization to preserve entities.

## 4. Decay & eviction

### 4.1 MemoryBank Ebbinghaus (2305.10250 L60-72)
(1) **Exact formula and constants:** R = e^(-t/S); R = retention fraction; t = time since last recall; e ~= 2.71828; S = memory strength, discrete, **initialized to 1** on first mention. On recall: **S += 1 and t := 0** (spacing effect: relearning resets curve, lowers forgetting probability).
(2) **Trigger:** decay continuous/time-driven (computed at retrieval); reinforcement event-driven (on recall).
(3) **Cost:** pure arithmetic — **zero LLM calls** for decay; summary is the only LLM cost.
(4) **Effect:** operationalizes Ebbinghaus retention + spacing effect; evaluated 10 days / 15 virtual users / 194 questions (L19-27).
(5) **Failure:** paper flags it "exploratory and highly simplified"; per-person/per-info variation; decay alone drops important-but-old memories vs relevance-based retrieval (survey 2404.13501 L521).

### 4.2 Generative Agents recency
Encoded in retrieval score's recency term, min-max normalized, alpha=1 (section 2.1). Decay is exponential in elapsed time [decay base commonly quoted 0.995 is NOT in converted markdown — [UNVERIFIED] (searched "0.995|decay factor|exponential minutes"); confirmed formula + alphas + threshold 150 only].

### 4.3 MemGPT queue eviction (2310.08560 L83-94)
(2) **Trigger: budget-driven thresholds.** prompt tokens > **warning count 70%** of context -> insert "memory pressure" system message so LLM self-stores important info to working/archival; prompt tokens > **flush count 100%** -> evict ~**50%** of context window, generate new recursive summary from old + evicted, move evicted to recall storage (kept indefinitely, retrievable).
(3) **Cost:** eviction deterministic; recursive summary 1 LLM call/flush; LLM prompted on pressure (may call funcs).
(1) No decay formula — token-budget constants, not scored decay.
(5) Failure: loss of important-but-recency-far content (MemBench 2506.21605 shows sharper accuracy decline with tokens for MemGPT-style, L660).

**Cost synthesis (NEMORI, 2508.03341 table 4, gpt-4o-mini, LoCoMo):** A-MEM response stage = **2614 tokens, search 947 ms, total 2867 ms E2E**; MemoryOS total 15220 ms; LangMem 22082 ms. Graph/link write-side + retrieval is cheapest at query time. NEMORI construction cuts LLM calls 59.5% and tokens 38.7% vs baselines (table 3) — write-side abstraction dominates token cost.

## 5. Deduplication, conflict resolution, contradiction

### 5.1 Zep/Graphiti (2501.13956)
**Entity resolution (L49-64):** current message + last n=4 messages (2 turns); speaker auto-extracted; Reflexion-style second pass reduces hallucination + boosts coverage; entity name embedded to **1024-dim**, cosine vs existing entity nodes; independent full-text search on names+summaries; candidates + episode -> LLM entity-resolution prompt -> duplicate => LLM emits updated name+summary. **Deterministic:** embedding + cosine + full-text candidates (SQL). **LLM:** duplicate decision + merge.
**Edge dedup (L64-87):** hybrid search for relevant edges **constrained to edges between the same entity pairs** as proposed new edge (prevents wrong cross-entity combinations; cuts search space).
**Contradiction:** LLM compares new edge vs semantically-related existing edges; on temporally-overlapping contradiction, invalidate affected edge by setting its `t_invalid := t_valid` of invalidating edge. New info always wins (transactional timeline T').
(5) Failure: weak models lag temporal categories (L335); dedup/contradiction is LLM-judgment-bound.

### 5.2 Relation typing grounding
The `contradict` edge type preserves both sides (bi-temporal supersession closes validity, never deletes) — pattern to adopt for provenance. LLM-issued confidence in [0,1] used directly as edge weight.

## 6. Bi-temporal temporal model (Zep/Graphiti)

Exact model (2501.13956 L49-64,64-87):
- **Two timelines.** T = valid time (when facts held true); T' = transactional/ingestion time (when Zep ingested; audit role).
- **Four timestamps per edge fact:** t_created, t_expired in T' (system lifecycle) and t_valid, t_invalid in T (period fact held true).
- Each message carries `t_ref` (reference timestamp) to resolve relative dates ("two weeks ago", "last summer") and ISO-8601 absolutes.
- **Invalidation rule:** new edge contradicting a semantically-related edge over an overlapping valid interval sets the older edge's `t_invalid := new edge's t_valid`. New info always wins.
- **Temporal-extraction contract (L456-511):** valid_at = when relationship "became true or was established"; invalid_at = when it "stopped being true or ended"; only set dates explicitly relating to formation/alteration (never infer from related events); present tense -> valid_at = reference timestamp; date-only -> 00:00:00; year-only -> Jan 1 00:00:00; always timezone (Z if unknown).

**Measured effect (L217-335, LongMemEval_s, avg ~115k tokens):**

| Memory | Model | Score | Latency | Context tokens |
|---|---|---|---|---|
| Full-context | gpt-4o-mini | 55.4% | 31.3s (IQR 8.76) | 115k |
| **Zep** | gpt-4o-mini | **63.8%** | **3.20s** (IQR 1.31) | **1.6k** |
| Full-context | gpt-4o | 60.2% | 28.9s (IQR 6.01) | 115k |
| **Zep** | gpt-4o | **71.2%** | **2.58s** (IQR 0.684) | **1.6k** |

+15.2% (mini) / +18.5% (4o) accuracy; latency -90%; context 115k->1.6k (~72x). Best categories: single-session-preference +77.7% (mini) and +184% (4o); temporal-reasoning +48.2%. DMR: Zep 94.8% vs MemGPT 93.4% vs recursive-summarization 35.3% (gpt-4-turbo, L132-203). Cost: top-20 edges + top entity nodes; BGE-m3 embed/rerank; gpt-4o-mini graph construction. Failure: weak-model temporal precision (L335); DMR trivial (60 msgs/conversation, L203-220).

## 7. Episodic->semantic transfer & sleep/offline consolidation
- **Episodic->semantic is the whole Zep design (2501.13956 L32-49):** raw episodic episodes (G_e) non-lossy substrate; semantic entities/facts (G_s) extracted; bidirectional indices keep citation (semantic->episode) and expansion (episode->facts). Mirrors episodic-vs-semantic psychology.
- **Sleep/offline consolidation:** corpus has neuroscience reviews (W4390974751 L28-39: NREM slow oscillations + spindles + REM replay relocate "new, unstable memories from hippocampus to neocortex"; synaptic-homeostasis hypothesis; CLS literature in 10.1038_nature14236). **None operationalizes sleep into an agent-memory algorithm in corpus.** CA3-CA1 computational consolidation model (10.21203/rs.3.rs-9584120) is external-only [UNVERIFIED]. myelin implication: "offline consolidation" must adapt - HippoRAG continual edge-add (no re-summarization) and Zep periodic community refresh with drift repair are the real precedents.

## 8. Catastrophic forgetting & continual
- **Survey (2501.07278 L287-316):** catastrophic forgetting = loss of prior knowledge on new tasks; stability-plasticity dilemma; four families: rehearsal/replay, regularization (EWC weight-importance penalties; LwF/distillation), architecture expansion, representation/prompt. LLM-era: continual pretraining / instruction tuning / knowledge editing.
- **HippoRAG as continual (2405.14831 L467-486, 2502.14802):** integrate by **adding edges**, not re-summarizing. PPR hyperparams: **damping 0.5**, synonym threshold 0.8, temperature 0.0.
- **PPR mechanics (2405.14831 L46-59,372-428):** query -> LLM NER entities -> link to KG nodes by retrieval-encoder similarity (query nodes) -> Personalized PageRank seeded from those nodes -> single-step multi-hop. Results: up to +20% R@2/R@5 on 2WikiMultiHopQA, ~3% MuSiQue; all-recall AR@5 avg 52.0 vs ColBERTv2 37.4; single-step on par/better than iterative IRCoT while **10-30x cheaper and 6-13x faster**; QA F1 up to +3% MuSiQue / +17% 2Wiki / +1% HotpotQA. Scale: MuSiQue 11,656 passages -> 91,729 nodes / 21,714 edges / 107,448 triples (+191,636 synonymy); 2Wiki 42,694 nodes; HotpotQA 82,157.
- **FOREVER (external 2601.03938):** model-centric time = optimizer-update magnitude; forgetting-curve replay intervals aligned to it; 0.6B-13B. [Not in corpus; [UNVERIFIED].]

## 9. Deterministic core vs LLM-required

| Mechanism | Pure Rust/SQL | LLM-required | Boundary |
|---|---|---|---|
| Importance (GA) | accumulator; threshold 150; store int | poignancy 1-10 at creation (1 call/record) | LLM emits 1 int; rest arithmetic |
| Recency decay | exponential in elapsed time; min-max -> [0,1] | none | deterministic; needs timestamp column |
| Retrieval score (GA) | a_rec*rec + a_imp*imp + a_rel*rel, a=1, min-max | embedding (encoder, not generative) | cosine in SQL; normalize; weight |
| Reflection tree | accumulator trigger; 100-recent select; query->retrieval->pointers | 3-question + 5-insight gen (2 calls/reflection) | LLM text in/out with cited IDs; edges SQL |
| MemGPT eviction | 70% warn / 100% flush / 50% evict bookkeeping; FIFO | recursive summary (1 call/flush); self-directed ops | thresholds+eviction deterministic; summary LLM |
| Ebbinghaus decay | R=e^(-t/S); S+=1, t:=0 on recall | none | fully deterministic; store (S, last_recall_ts) |
| Hierarchical summary | day segment; which dialogs -> which day | daily+global summary (2 calls/day) | summaries LLM; tiering SQL |
| A-MEM note build | embed note; slots | keywords+context+tags (P_s1); link (P_s2); evolve (P_s3) - ~3 calls/note [INFERENCE] | LLM JSON; indexing+links SQL |
| A-MEM retrieval | cosine top-k (k=10); box/link traversal | query embedding (encoder) | deterministic given embeddings |
| Zep entity resolution | 1024-d embed; cosine+full-text candidates; n=4 ctx | duplicate-or-not + merge (1 call) | deterministic narrowing; LLM final |
| Zep edge dedup | same-entity-pair constrained search | is_duplicate + existing uuid (1 call) | deterministic space; LLM verdict |
| Zep bi-temporal extract | 4 stamps; ISO-8601; relative->absolute w/ t_ref (datetime math) | which dates are relationship-forming (1 call) | LLM emits valid/invalid; arithmetic Rust |
| Zep contradiction/invalidation | overlap check; set t_invalid:=t_valid_new | identify contradiction (1 call) | **deterministic invalidation write once flagged**; pure range math |
| HippoRAG index | KG store; synonymy cosine (encoder) | OpenIE triples (1 call/passage, offline) | LLM triples -> node/edge tables |
| HippoRAG PPR | **Personalized PageRank (damping 0.5)** | query NER (1 call/query); encoder linking | **PPR pure graph math** - ideal Rust/petgraph; 1 small LLM call for query entities |

**Top Rust opportunities:** (a) PPR — full arithmetic, no LLM, multi-hop core; (b) bi-temporal invalidation + overlap — range arithmetic in SQL, LLM only flags; (c) Ebbinghaus decay — pure math; (d) cosine top-k, min-max, accumulators — SQL.

**LLM-call counts where stated:** GA reflection ~4-9 calls/day; MemoryBank ~2 calls/day + 0 for decay; A-MEM ~3 calls/note [INFERENCE]; Zep ~4-6/episode [INFERENCE from pipeline]; HippoRAG 1 OpenIE/passage (batch) + 1 query-NER. No mechanism's decay/eviction arithmetic needs an LLM; only content generation and identity/contradiction judgment do.

## 10. Synthesis for a Rust implementation
1. **Non-lossy episodes + derived semantic store, bidirectionally linked (Zep)** — citation/provenance backbone (grounding axis). Raw first, abstract second, never discard raw.
2. **Deterministic decay with reinforcement (MemoryBank)** — R=e^(-t/S), S+=1 on recall — auditable forgetting core (robustness/token-cost).
3. **PPR single-step multi-hop (HippoRAG)** — pure-Rust graph math replaces iterative reasoning calls: 10-30x cheaper, 6-13x faster, +17-20% on entity-centric multi-hop.
4. **Bi-temporal invalidation, new-info-wins (Graphiti)** — t_valid/t_invalid/t'_created/t'_expired, set t_invalid:=t_valid_new on contradiction — consistent temporal history in SQL.
5. **Budget-driven triggers beat fixed cadence**: GA importance accumulator (150) and MemGPT token-pressure (70%/100%) avoid wasted abstraction when little salient content; combine with Zep periodic community refresh for drift repair.

**Top failure modes to engineer against:** over-summarization (defeated by non-lossy episodic + bidirectional citation); identifier/entity loss (HippoRAG noun-phrase OpenIE over summarization); drift in periodic abstractions (dynamic extension diverges -> schedule refresh); weak-model temporal misread (verify temporal output schema, fall back to t_ref); retrieved-but-wrong / embellished recall (GA documented hallucination + fragment retrieval — provenance pointers mitigate).

**Eval axis mappings:** accuracy — Zep LongMemEval 63.8->71.2% (temporal +48.2%), HippoRAG +17-20%; token-cost — Zep 115k->1.6k, HippoRAG 10-30x cheaper, NEMORI -59.5% calls/-38.7% tokens; p95-latency — Zep 3.20->2.58s vs 31.3/28.9s, A-MEM 2.9s; grounding/provenance — Zep bidirectional index + GA cited pointers; robustness/poisoning — contradict preserves both sides, new-info-wins transactional rule.

## 11. Evidence gaps / [UNVERIFIED]
- GA recency decay base (0.995) not in markdown — searched "0.995|decay|exponential minutes"; confirmed formula + alphas=1 + threshold 150 only.
- A-MEM per-note LLM-call count (3) [INFERENCE from P_s1/P_s2/P_s3 prompts]; corpus publishes prompts, no per-note tally.
- Zep per-episode LLM-call count (~4-6) [INFERENCE from pipeline]; corpus reports E2E latency/cost, not per-phase calls.
- Sleep-based agent-memory consolidation: no algorithmic operationalization in corpus (only neuroscience reviews); CA3-CA1 external-only.
- FOREVER replay schedule external-only (2601.03938), not verified in corpus.
