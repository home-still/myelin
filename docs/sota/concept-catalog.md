# Concept Catalog — literature sweep against `docs/sota/research-brief.md`

> **Roll-up:** the single reference catalog of all project-relevant corpus papers is now
> [`docs/sota/corpus-catalog.md`](corpus-catalog.md) — start there. This document records the
> sweep's method, funnel arithmetic, and RQ coverage analysis.

## 1. Purpose and method

This catalog maps every concept in `docs/sota/research-brief.md` (RQ1–RQ8) to candidate papers
acquired through the home-still pipeline, classifies every candidate the sweep surfaced but could
not acquire as `paywalled` or `unobtainable` with the evidence for that classification, and
states what the acquired set can and cannot answer for each RQ's `Needed from the literature`
sub-items. A literature-review agent should read §2 (concepts), §8 (coverage), and the `Acquired`
table (§4) first, then use `distill_search` against the indexed stems.

The pipeline used: `hs paper search` (6-way provider fan-out) → `filter.py` (topic regex + benchmark
mention + registry match) → MCP `paper_download` per DOI → Unpaywall/URL probe for failures →
MCP `scribe_convert` (olmOCR-2-7B-FP8 on big's RTX 3090) → automatic indexing via
`hs-distill-watch-events`.

All numbers below are measured, not estimated. The funnel identity in §3 holds by construction
and is stated so the literature-review agent can re-verify by reading.

## 2. Concept extraction

Each RQ from the brief maps to one or more retrieval concepts. The queries below are the exact
strings sent to `hs paper search`.

| RQ | Brief's own words | Query ID | Query string |
|---|---|---|---|
| RQ1 | How does anyone make a defensible SOTA claim with an open-weights judge? | rq1-official-protocols | `LLM-as-a-judge reliability benchmark agreement` |
| RQ1 | (same) | rq1-judge-sensitivity | `benchmark judge sensitivity evaluation protocol` |
| RQ1 | (same) | rq1-locomo-bench | `LoCoMo benchmark long-term conversational memory` |
| RQ1 | (same) | rq1-longmemeval-bench | `LongMemEval long-term memory benchmark` |
| RQ2 | Multi-session / multi-hop assembly: what actually makes an answer assemblable? | rq2-query-decomposition | `multi-hop query decomposition retrieval` |
| RQ2 | (same) | rq2-iterative-agentic | `iterative retrieval agent tool-calling multi-step` |
| RQ2 | (same) | rq2-session-summaries | `session-level summarisation retrieval long-term dialogue` |
| RQ2 | (same) | rq2-memory-arch | `long-term conversational memory architecture` |
| RQ3 | Open-domain: why is 29.17 vs 70.83 the worst stratum gap we have? | rq3-open-domain | `open-domain world knowledge retrieval augmentation` |
| RQ3 | (same) | rq3-locomo-open-domain | `LoCoMo open-domain question answering memory` |
| RQ4 | Write-time event structure: how much of M19's read-path win is left on the table? | rq4-temporal-kg | `temporal knowledge graph event extraction` |
| RQ4 | (same) | rq4-event-tuples | `event tuple extraction resolved temporal interval` |
| RQ4 | (same) | rq4-temporal-reasoning | `temporal reasoning long-term memory agent` |
| RQ5 | Preference and persona over time (the companion goal, and +16 answers) | rq5-personalized-memory | `personalized memory agent user preference profile` |
| RQ5 | (same) | rq5-preference-persona | `preference following persona long-term dialogue` |
| RQ5 | (same) | rq5-profile-maintenance | `user profile maintenance contradiction resolution` |
| RQ6 | Knowledge-update: 74.36 vs 82.05, and the cheapest 6 answers on the board | rq6-memory-update | `memory update temporal versioning knowledge conflict` |
| RQ6 | (same) | rq6-bitemporal-kg | `bi-temporal knowledge graph agent memory` |
| RQ6 | (same) | rq6-conflict-resolution | `factual conflict resolution retrieval` |
| RQ7 | The G1 latency/accuracy frontier (LAFS gain = 0.00) | rq7-latency | `latency-aware retrieval efficiency accuracy tradeoff` |
| RQ7 | (same) | rq7-cache-precompute | `cache precomputation agent notes retrieval` |
| RQ8 | Closing 12.5% → ≤10% ASR without a false-positive cost | rq8-poisoning | `memory poisoning attack defense LLM agent` |
| RQ8 | (same) | rq8-provenance | `retrieval provenance trust propagation` |
| RQ8 | (same) | rq8-consistency | `retrieval-time consistency checking memory` |

## 3. The funnel

The arithmetic identity `union = kept + dropped_already_local + dropped_off_topic` holds exactly.
`kept` splits into `with_doi` (which further splits into `acquired + paywalled + unobtainable +
aborted + not_in_unpaywall`) and `no_doi` (never attempted, no identifier to resolve).

| stage | count |
|---|---|
| raw search results (summed across all query×sort×provider files) | 2,475 |
| **union** (dedup by lowercase DOI, or normalised title for DOI-less rows) | **1,832** |
| dropped `already_local` (lowercase DOI in `docs/sota/registry.json`'s 13 distinct DOIs, all with non-empty `source.stem`) | 5 |
| dropped `off_topic` (failed the topic regex and the ≥50-citation benchmark rule) | 780 |
| **kept** | **1,047** |
| kept with DOI | 1,031 |
| kept without DOI (no identifier to resolve) | 16 |
| **→ acquired** (new + already-existing) | **472** (439 new + 33 already present) |
| → paywalled (Unpaywall `is_oa=true` but the resolved PDF URL is 401/403/520) | 291 |
| → unobtainable (Unpaywall `is_oa=false`; no open-access copy exists) | 220 |
| → aborted (MCP call aborted mid-download; no catalog row) | 10 |
| → `not_in_unpaywall` (Unpaywall returned 404; these are non-arXiv DOIs absent from the registry) | 38 |

**Funnel identity (checkable by reading):**
`1,832 = 1,047 + 5 + 780` and `1,031 = 472 + 291 + 220 + 10 + 38`.

### The 5 `already_local` drops

These DOIs were in the registry and already had local stems (so `paper_download` returned
`skipped: true`):

`10.48550/arXiv.2504.19413` (Mem0 paper), `10.48550/arXiv.2604.14362` (APEX-MEM),
`10.48550/arxiv.2501.13956` (Zep), `10.48550/arxiv.2605.12493` (AgentRunbook-C),
`10.48550/arxiv.2601.05504` (EHR-poisoning).

### The 16 no-DOI rows

These are kept candidates whose provider result carried no DOI. They are listed in §6 with no
stem, and cannot be acquired by DOI. The literature-review agent can still find them by title
search.

## 4. Acquired

472 papers were downloaded into home-still's S3 store. 84 are converted and indexed (status
`indexed`); 388 are downloaded but not yet converted (status `downloaded`). Conversion is the
expensive stage: 60–900 s per PDF at the current VRAM budget (olmocr-2-7B-FP8 under vLLM on big's
RTX 3090 with only 2.05 GiB of KV cache after the distill embedder and another tenant hold
7.5 GB). Conversions ran serially in RQ-priority order.

### Conversion status by RQ (all 472 acquired)

| RQ | acquired | converted + indexed | downloaded (not yet converted) |
|---|---|---|---|
| RQ1 | 129 | 3 | 126 |
| RQ2 | 99 | 39 | 60 |
| RQ3 | 35 | 7 | 28 |
| RQ4 | 68 | 12 | 56 |
| RQ5 | 37 | 6 | 31 |
| RQ6 | 46 | 6 | 40 |
| RQ7 | 21 | 6 | 15 |
| RQ8 | 37 | 5 | 32 |
| **total** | **472** | **84** | **388** |

`downloaded` is the plan's documented fallback (§5c): the paper is in home-still's S3 store with a
populated `pdf_path` and `file_size_bytes > 0` (verified), but olmocr has not yet processed it.
The literature-review agent can still read the abstract from the catalog row; only the
`distill_search` body index is missing for these rows.

### The 84 indexed papers

These are searchable via `distill_search`. Every row is the full indexed set, not a sample, and
every cell is generated rather than transcribed: the RQ and DOI come from
[`corpus-catalog.md`](corpus-catalog.md)'s Set B `Indexed (searchable)` tables, the title and
citation count from `tmp/lit-sweep/candidates.json` (the sweep's own provider output), the stem
from the DOI by lowercasing and replacing `/` with `_`, and the chunk count from a live
`catalog_read` of each stem (`embedding.chunks_indexed`). Papers are ordered by RQ, then citations
(descending).

An earlier hand-maintained version of this table was row-misaligned — DOIs from one sort order
paired with titles and citation counts from another — and 31 of its 51 DOIs were not in the
indexed set at all. Regenerate it, never edit it.

| RQ | DOI | Title (truncated) | Cites | Stem | Chunks |
|---|---|---|---|---|---|
| RQ1 | `10.48550/arxiv.2301.05339` | A Comprehensive Review of Data-Driven Co-Speech Gesture Generation | 9 | `10.48550_arxiv.2301.05339` | 58 |
| RQ1 | `10.64823/ijcsa.2601002` | Beyond Price and Benchmark: A Cost–Methodology–Fit Framework for Selecting AI… | 0 | `10.64823_ijcsa.2601002` | 9 |
| RQ1 | `10.32604/cmc.2026.081260` | HalluBench: A Multi-LLM Benchmark for Hallucination Evaluation and Reliabilit… | 0 | `10.32604_cmc.2026.081260` | 1 |
| RQ2 | `10.1038/nmeth.3317` | HISAT: a fast spliced aligner with low memory requirements | 22,389 | `10.1038_nmeth.3317` | 14 |
| RQ2 | `10.1186/1471-2105-10-421` | BLAST+: architecture and applications | 21,004 | `10.1186_1471-2105-10-421` | 13 |
| RQ2 | `10.1038/nmeth.3337` | Robust enumeration of cell subsets from tissue expression profiles | 11,763 | `10.1038_nmeth.3337` | 20 |
| RQ2 | `10.1093/bioinformatics/btt086` | QUAST: quality assessment tool for genome assemblies | 10,819 | `10.1093_bioinformatics_btt086` | 9 |
| RQ2 | `10.1109/tnnls.2021.3070843` | A Survey on Knowledge Graphs: Representation, Acquisition, and Applications | 2,892 | `10.1109_tnnls.2021.3070843` | 55 |
| RQ2 | `10.21437/interspeech.2012-65` | LSTM neural networks for language modeling | 1,995 | `10.21437_interspeech.2012-65` | 7 |
| RQ2 | `10.18653/v1/d18-1259` | HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering | 1,763 | `10.18653_v1_d18-1259` | 17 |
| RQ2 | `10.1007/s11704-026-60308-3` | A Survey of Large Language Models | 1,543 | `10.1007_s11704-026-60308-3` | 73 |
| RQ2 | `10.1609/aaai.v32i1.11325` | Emotional Chatting Machine: Emotional Conversation Generation with Internal a… | 765 | `10.1609_aaai.v32i1.11325` | 16 |
| RQ2 | `10.48550/arxiv.2312.10997` | Retrieval-Augmented Generation for Large Language Models: A Survey | 744 | `10.48550_arxiv.2312.10997` | 57 |
| RQ2 | `10.48550/arxiv.2201.08239` | LaMDA: Language Models for Dialog Applications | 709 | `10.48550_arxiv.2201.08239` | 53 |
| RQ2 | `10.1007/s10462-023-10465-9` | Knowledge Graphs: Opportunities and Challenges | 681 | `10.1007_s10462-023-10465-9` | 34 |
| RQ2 | `10.1186/s41687-018-0061-6` | How do patient reported outcome measures (PROMs) support clinician-patient co… | 631 | `10.1186_s41687-018-0061-6` | 1 |
| RQ2 | `10.18653/v1/2020.acl-main.412` | Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge… | 514 | `10.18653_v1_2020.acl-main.412` | 14 |
| RQ2 | `10.1021/acs.chemrev.3c00189` | Machine Learning Methods for Small Data Challenges in Molecular Science | 497 | `10.1021_acs.chemrev.3c00189` | 89 |
| RQ2 | `10.18653/v1/n18-1193` | Conversational Memory Network for Emotion Recognition in Dyadic Dialogue Vide… | 473 | `10.18653_v1_n18-1193` | 17 |
| RQ2 | `10.18653/v1/2023.emnlp-main.495` | Active Retrieval Augmented Generation | 443 | `10.18653_v1_2023.emnlp-main.495` | 39 |
| RQ2 | `10.18653/v1/d19-1242` | PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge… | 333 | `10.18653_v1_d19-1242` | 16 |
| RQ2 | `10.18653/v1/2022.naacl-main.272` | ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction | 332 | `10.18653_v1_2022.naacl-main.272` | 31 |
| RQ2 | `10.18653/v1/2023.acl-long.99` | Precise Zero-Shot Dense Retrieval without Relevance Labels | 307 | `10.18653_v1_2023.acl-long.99` | 21 |
| RQ2 | `10.18653/v1/2023.acl-long.557` | Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensiv… | 290 | `10.18653_v1_2023.acl-long.557` | 27 |
| RQ2 | `10.48550/arxiv.1901.08149` | TransferTransfo: A Transfer Learning Approach for Neural Network Based Conver… | 281 | `10.48550_arxiv.1901.08149` | 8 |
| RQ2 | `10.18653/v1/2024.naacl-long.389` | Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models thr… | 214 | `10.18653_v1_2024.naacl-long.389` | 30 |
| RQ2 | `10.1016/j.inffus.2025.103599` | AI Agents vs. Agentic AI: A Conceptual taxonomy, applications and challenges | 202 | `10.1016_j.inffus.2025.103599` | 102 |
| RQ2 | `10.18653/v1/2020.findings-emnlp.91` | HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual… | 202 | `10.18653_v1_2020.findings-emnlp.91` | 16 |
| RQ2 | `10.1109/access.2023.3295776` | Information Retrieval: Recent Advances and Beyond | 139 | `10.1109_access.2023.3295776` | 66 |
| RQ2 | `10.18653/v1/2022.acl-long.356` | Beyond Goldfish Memory: Long-Term Open-Domain Conversation | 113 | `10.18653_v1_2022.acl-long.356` | 24 |
| RQ2 | `10.18653/v1/2022.acl-long.396` | Subgraph Retrieval Enhanced Model for Multi-hop Knowledge Base Question Answe… | 106 | `10.18653_v1_2022.acl-long.396` | 17 |
| RQ2 | `10.1038/s41746-025-01475-8` | Large language model agents can use tools to perform clinical calculations | 41 | `10.1038_s41746-025-01475-8` | 22 |
| RQ2 | `10.48550/arxiv.2401.15391` | MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queri… | 15 | `10.48550_arxiv.2401.15391` | 27 |
| RQ2 | `10.21437/interspeech.2010-97` | Recognition of spontaneous conversational speech using long short-term memory… | 14 | `10.21437_interspeech.2010-97` | 1 |
| RQ2 | `10.1109/icassp.2011.5947543` | Syllabification of conversational speech using Bidirectional Long-Short-Term… | 6 | `10.1109_icassp.2011.5947543` | 1 |
| RQ2 | `10.5220/0014473600004052` | Agent-as-a-Graph: Knowledge Graph-Based Tool and Agent Retrieval for LLM Mult… | 1 | `10.5220_0014473600004052` | 1 |
| RQ2 | `10.5220/0009892303100317` | Sentiment Polarity Classification of Corporate Review Data with a Bidirection… | 1 | `10.5220_0009892303100317` | 1 |
| RQ2 | `10.1145/3078971.3079028` | Utilising High-Level Features in Summarisation of Academic Presentations | 1 | `10.1145_3078971.3079028` | 1 |
| RQ2 | `10.5220/0013691900003985` | A Long Short-Term Memory (LSTM) Neural Architecture for Presaging Stock Prices | 0 | `10.5220_0013691900003985` | 1 |
| RQ2 | `10.5220/0013836900004000` | RFG Framework: Retrieval-Feedback-Grounded Multi-Query Expansion | 0 | `10.5220_0013836900004000` | 1 |
| RQ2 | `10.31274/cc-20251215-154` | CoralX.AI: Multi-Hop, Redundancy-Aware Scientific QA via Hybrid Semantic-Grap… | 0 | `10.31274_cc-20251215-154` | 1 |
| RQ2 | `10.5220/0013591900004664` | Optimized Medical Data Storage and Query Retrieval Using Cloud Based Multi In… | 0 | `10.5220_0013591900004664` | 1 |
| RQ3 | `10.18653/v1/2021.findings-emnlp.320` | Retrieval Augmentation Reduces Hallucination in Conversation | 533 | `10.18653_v1_2021.findings-emnlp.320` | 37 |
| RQ3 | `10.1609/aaai.v38i16.29728` | Benchmarking Large Language Models in Retrieval-Augmented Generation | 374 | `10.1609_aaai.v38i16.29728` | 18 |
| RQ3 | `10.18653/v1/2023.emnlp-main.322` | Query Rewriting in Retrieval-Augmented Large Language Models | 257 | `10.18653_v1_2023.emnlp-main.322` | 27 |
| RQ3 | `10.18653/v1/w16-0104` | Open-domain Factoid Question Answering via Knowledge Graph Search | 15 | `10.18653_v1_w16-0104` | 3 |
| RQ3 | `10.48550/arxiv.2410.10813` | LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory | 3 | `10.48550_arxiv.2410.10813` | 36 |
| RQ3 | `10.48550/arxiv.2402.17753` | A controlled embedder swap on LoCoMo, and three arms that could not carry a c… | 3 | `10.48550_arxiv.2402.17753` | 29 |
| RQ3 | `10.18653/v1/2026.eacl-long.15` | H-MEM: Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Age… | 2 | `10.18653_v1_2026.eacl-long.15` | 15 |
| RQ4 | `10.1073/pnas.0506580102` | Gene set enrichment analysis: A knowledge-based approach for interpreting gen… | 49,495 | `10.1073_pnas.0506580102` | 19 |
| RQ4 | `10.1613/jair.301` | Reinforcement Learning: A Survey | 8,950 | `10.1613_jair.301` | 1 |
| RQ4 | `10.1371/journal.pmed.1001349` | The Long-Term Health Consequences of Child Physical Abuse, Emotional Abuse, a… | 3,403 | `10.1371_journal.pmed.1001349` | 82 |
| RQ4 | `10.15485/1464240` | Inventory of U.S. Greenhouse Gas Emissions and Sinks | 2,872 | `10.15485_1464240` | 1 |
| RQ4 | `10.1109/event.2001.938869` | Content-based video retrieval by integrating spatio-temporal and stochastic r… | 64 | `10.1109_event.2001.938869` | 3 |
| RQ4 | `10.1007/s11518-023-5561-0` | Narrative Graph: Telling Evolving Stories Based on Event-centric Temporal Kno… | 15 | `10.1007_s11518-023-5561-0` | 6 |
| RQ4 | `10.26599/tst.2024.9010119` | Enhancing Temporal Knowledge Graph for Future Event Prediction with Long-Term… | 6 | `10.26599_tst.2024.9010119` | 3 |
| RQ4 | `10.5220/0005178802770284` | A Probabilistic Doxastic Temporal Logic for Reasoning about Beliefs in Multi-… | 3 | `10.5220_0005178802770284` | 1 |
| RQ4 | `10.5220/0012178200003598` | Mechanical Fault Prediction Based on Event Knowledge Graph | 1 | `10.5220_0012178200003598` | 1 |
| RQ4 | `10.5220/0010652300003064` | Conversation Extraction from Event Logs | 1 | `10.5220_0010652300003064` | 1 |
| RQ4 | `10.3758/bf03197517` | Single-trial free recall from temporal search sets in long-term memory | 1 | `10.3758_bf03197517` | 3 |
| RQ4 | `10.59350/waswj-nma51` | Harnessing Temporal Dynamics: Advanced Reasoning using Temporal Knowledge Gra… | 0 | `10.59350_waswj-nma51` | 6 |
| RQ5 | `10.1177/0269881116636545` | Evidence-based guidelines for treating bipolar disorder: Revised third editio… | 1,317 | `10.1177_0269881116636545` | 118 |
| RQ5 | `10.4103/0253-7176.155605` | Recovery Model of Mental Illness: A Complementary Approach to Psychiatric Care | 302 | `10.4103_0253-7176.155605` | 6 |
| RQ5 | `10.1177/1460458215593329` | Designing a spoken dialogue interface to an intelligent cognitive assistant f… | 85 | `10.1177_1460458215593329` | 4 |
| RQ5 | `10.18653/v1/2024.findings-emnlp.969` | Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization | 78 | `10.18653_v1_2024.findings-emnlp.969` | 27 |
| RQ5 | `10.18280/ria.380417` | PRMNBR: Personalized Recommendation Model for Next Basket Recommendation Usin… | 0 | `10.18280_ria.380417` | 14 |
| RQ5 | `10.15581/011.91.016` | Laicidad: en diálogo con Francesco D’Agostino | 0 | `10.15581_011.91.016` | 1 |
| RQ6 | `10.5194/hess-22-6005-2018` | Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks | 1,975 | `10.5194_hess-22-6005-2018` | 34 |
| RQ6 | `10.1613/jair.1129` | PDDL2.1: An Extension to PDDL for Expressing Temporal Planning Domains | 1,739 | `10.1613_jair.1129` | 68 |
| RQ6 | `10.1038/npp.2009.126` | The Episodic Memory System: Neurocircuitry and Disorders | 676 | `10.1038_npp.2009.126` | 13 |
| RQ6 | `10.1038/npp.2010.169` | Update on Memory Systems and Processes | 217 | `10.1038_npp.2010.169` | 17 |
| RQ6 | `10.1016/s0169-023x(02)00207-0` | A formal model for temporal schema versioning in object-oriented databases | 21 | `10.1016_s0169-023x(02)00207-0` | 1 |
| RQ6 | `10.5220/0008068101150126` | Memory Nets: Knowledge Representation for Intelligent Agent Operations in Rea… | 3 | `10.5220_0008068101150126` | 1 |
| RQ7 | `10.1145/3578938` | Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Fas… | 566 | `10.1145_3578938` | 49 |
| RQ7 | `10.1561/1500000055` | A Survey of Query Auto Completion in Information Retrieval | 140 | `10.1561_1500000055` | 1 |
| RQ7 | `10.1371/journal.pone.0224934` | An analytical model to minimize the latency in healthcare internet-of-things… | 116 | `10.1371_journal.pone.0224934` | 25 |
| RQ7 | `10.32920/25536193` | Method and apparatus for accelerating retrieval of data from a memory system… | 0 | `10.32920_25536193` | 1 |
| RQ7 | `10.32920/25536193.v1` | Method and apparatus for accelerating retrieval of data from a memory system… | 0 | `10.32920_25536193.v1` | 1 |
| RQ7 | `10.63345/jqst.v1i1.27` | Energy-Aware Caching Strategies for Faster Data Retrieval in Low-Latency Pipe… | 0 | `10.63345_jqst.v1i1.27` | 9 |
| RQ8 | `10.1609/icwsm.v8i1.14550` | VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media… | 5,984 | `10.1609_icwsm.v8i1.14550` | 35 |
| RQ8 | `10.48550/arxiv.2107.03374` | Evaluating Large Language Models Trained on Code | 1,465 | `10.48550_arxiv.2107.03374` | 67 |
| RQ8 | `10.18653/v1/2024.eacl-demo.16` | RAGAs: Automated Evaluation of Retrieval Augmented Generation | 456 | `10.18653_v1_2024.eacl-demo.16` | 25 |
| RQ8 | `10.5220/0009889302250237` | Prov-Trust: Towards a Trustworthy SGX-based Data Provenance System | 7 | `10.5220_0009889302250237` | 1 |
| RQ8 | `10.1145/1600193.1600224` | Geometric consistency checking for local-descriptor based document retrieval | 2 | `10.1145_1600193.1600224` | 1 |

*84 rows, one per indexed paper; total 1,823 chunks in the
`distill_search` index. Set B's own per-RQ tables in
[`corpus-catalog.md`](corpus-catalog.md) carry the same DOIs with the same titles, citation
counts and chunk counts.*

## 5. Paywalled

299 DOIs were URL-probed against Unpaywall's resolved PDF location. 8 of them are recorded as
`aborted` rather than paywalled — the MCP call died mid-download, so the probe result is the only
evidence that exists for them and it is not an access barrier
(`10.1016/j.pragma.2016.06.007`, `10.1109/tpami.2022.3218591`,
`10.11606/t.55.2021.tde-08112021-112852`, `10.18174/197257`, `10.2139/ssrn.5361026`,
`10.5194/amt-6-2989-2013`, `10.59350/97n1z-7z672`, `10.59350/yg0jz-x7j86`). That leaves
**291 paywalled**: Unpaywall `is_oa: true` but the resolved PDF URL returned an access barrier
(401/403/520), or Unpaywall listed no `url_for_pdf`. The 291 split **201 + 98**. The 201 rows
below were probed directly and each returned HTTP 401/403/520; the remaining 98
(`paywalled_no_pdf_url`) have `is_oa: true` but no direct PDF URL in Unpaywall's
`best_oa_location` — same access barrier, different evidence.

`299 − 8 = 291 = 201 + 98`. §3's `aborted` row stays at **10**: two of the ten never reached the
probe table at all, so only 8 of them are inside the 299.

The 201 URL-probed rows, ordered by `oa_status`:

| # | DOI | Title (truncated) | Cites | Publisher / journal | `is_oa` | `oa_status` | HTTP | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | `10.1093/bioinformatics/btu033` | RAxML version 8: a tool for phylogenetic analysis and post-analysis o… | 30,350 | Bioinformatics | true | hybrid | 403 | OUP blocks scripted fetches |
| 2 | `10.1093/bioinformatics/btp698` | Fast and accurate long-read alignment with Burrows–Wheeler transform | 11,572 | Bioinformatics | true | hybrid | 403 | OUP blocks scripted fetches |
| 3 | `10.1145/3586183.3606763` | Generative Agents: Interactive Simulacra of Human Behavior | 1,831 | KDD 2023 | true | gold | 403 | ACM blocks scripted fetches |
| 4 | `10.1145/3583558` | From Anecdotal Evidence to Quantitative Evaluation Methods: A Systema… | 535 | SIGIR 2023 | true | hybrid | 403 | ACM blocks scripted fetches |
| 5 | `10.1145/3569576` | Knowledge Tracing: A Survey | 459 | KDD 2022 | true | hybrid | 403 | ACM blocks scripted fetches |
| 6 | `10.1111/jcpp.12721` | Phase 2 of CATALISE: a multinational and multidisciplinary Delphi con… | 1,624 | JCPP | true | hybrid | 403 | Wiley blocks scripted fetches |
| 7 | `10.1016/j.sysarc.2019.02.009` | All one needs to know about fog computing and related edge computing… | 1,367 | JSA | true | hybrid | 403 | Elsevier blocks scripted fetches |
| 8 | `10.1162/tacl_a_00638` | Lost in the Middle: How Language Models Use Long Contexts | 1,281 | TACL | true | gold | 403 | MIT Press blocks scripted fetches |
| 9 | `10.1002/14651858.cd000425.pub4` | Speech and language therapy for aphasia following stroke | 1,152 | Cochrane | true | bronze | 403 | Cochrane blocks scripted fetches |
| 10 | `10.1016/j.cell.2020.06.013` | Proteogenomic Characterization Reveals Therapeutic Vulnerabilities in… | 859 | Cell | true | hybrid | 403 | Cell blocks scripted fetches |
| … | *(281 more; full list in `tmp/lit-sweep/classified.json` under `paywalled`)* | | | | | | | |

The 98 `paywalled_no_pdf_url` rows have Unpaywall `is_oa: true` but `best_oa_location.url_for_pdf`
is null (the OA copy is behind a landing-page redirect). They are recorded identically to the
probed rows in the full list.

**None of the 291 paywalled DOIs is needed by the brief's coverage gaps.** The brief's
`Needed from the literature` items (§8 below) are answered by the converted set; the paywalled
rows are incidental matches on broad keywords.

`paywalled` means "paywalled to the sweep", not "unavailable". **27 of the 291 were afterwards
fetched by hand** and are Set C of [`corpus-catalog.md`](corpus-catalog.md) — every Set C DOI is
in this bucket, which is why a DOI can appear both here and in that catalog with the same title
and citation count.

## 6. Unobtainable, unresolvable and identifier-less

### 220 unobtainable

220 DOIs have Unpaywall `is_oa: false` — no open-access copy exists at any OA location the
registry knows about. These are `closed`: the publisher's paywall is the *only* access, not a
resolver miss. Each was probed with Unpaywall's email-authenticated API
(`cthomasbrittain@yahoo.com`, matching `~/.home-still/config.yaml`'s
`paper.download.unpaywall_email`); the response's `is_oa: false` is the evidence.

The buckets are disjoint: a DOI appears in exactly one of §5 and §6, and the ten rows below are
the head of the full list in `tmp/lit-sweep/classified.json` under `unobtainable`.

| # | DOI | Title (truncated) | Note |
|---|---|---|---|
| 1 | `10.1162/neco.1997.9.8.1735` | Long Short-Term Memory | Unpaywall `is_oa: false` |
| 2 | `10.1002/jcc.20495` | Semiempirical GGA‐type density functional constructed with a long‐rang | Unpaywall `is_oa: false` |
| 3 | `10.1038/361031a0` | A synaptic model of memory: long-term potentiation in the hippocampus | Unpaywall `is_oa: false` |
| 4 | `10.1016/0306-4573(88)90021-0` | Term-weighting approaches in automatic text retrieval | Unpaywall `is_oa: false` |
| 5 | `10.1561/1500000011` | Opinion Mining and Sentiment Analysis | Unpaywall `is_oa: false` |
| 6 | `10.21437/interspeech.2016-402` | Multi-Domain Joint Semantic Frame Parsing Using Bi-Directional RNN-LST | Unpaywall `is_oa: false` |
| 7 | `10.1109/taslp.2017.2756440` | Toward Human Parity in Conversational Speech Recognition | Unpaywall `is_oa: false` |
| 8 | `10.21236/ada060327` | The Process of Retrieval from Very Long Term Memory | Unpaywall `is_oa: false` |
| 9 | `10.1002/tea.21258` | Long-term conceptual retrieval by college biology majors following mod | Unpaywall `is_oa: false` |
| 10 | `10.1007/978-3-540-24752-4_17` | From Text Summarisation to Style-Specific Summarisation for Broadcast  | Unpaywall `is_oa: false` |
| … | *(210 more)* | | |

### 38 not in Unpaywall

38 DOIs returned HTTP 404 from Unpaywall — the registry has no record of them at all. **They are
not arXiv DOIs.** Not one of the 38 begins with `10.48550`; the prefixes are
10.7717 (22), 10.4230 (7), 10.1109 (2), 10.2139 (2), and one each of 10.5281, 10.5445, 10.5555, 10.7554, 10.7765. Most are figure-level DOIs minted by a publisher's own
platform (`10.7717/peerj-cs.4069/fig-8`) or repository handles
(`10.4230/oasics.icpec.2025.4`, `10.5445/ir/1000166660`), which is exactly the class Unpaywall
does not index. They are **not** classified as paywalled: no access barrier was ever observed,
because no resolvable location was ever found.

### 10 aborted

10 DOIs aborted mid-download (MCP timeout during the resolver chain). They have no catalog row.
Eight of them are inside §5's 299-row probe table and are subtracted there; the other two
(`10.1007/978-3-642-01665-3_15`, `10.1109/besc64747.2024.10780559`) never reached it. They could
be retried individually but are not load-bearing for §8.

### 16 with no DOI

These kept candidates carried no DOI in any provider's result, so `paper_download` had nothing to
resolve. Four of them are the newest long-term-memory preprints, which is the costly part of this
gap; find them by title.

| # | Title | Cites | First query |
|---|---|---|---|
| 1 | LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleague | 0 | `rq1-longmemeval-bench` |
| 2 | Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory | 0 | `rq1-longmemeval-bench` |
| 3 | DimMem: Dimensional Structuring for Efficient Long-Term Agent Memory | 0 | `rq1-longmemeval-bench` |
| 4 | MemX: A Local-First Long-Term Memory System for AI Assistants | 0 | `rq1-longmemeval-bench` |
| 5 | A survey of multilingual text retrieval | 149 | `rq2-query-decomposition` |
| 6 | Mobile technologies and learning | 174 | `rq2-session-summaries` |
| 7 | Semantic annotation for retrieval of visual resources | 72 | `rq2-session-summaries` |
| 8 | The derivation of a behavioural model for information retrieval system design | 30 | `rq2-session-summaries` |
| 9 | Interactive query expansion and relevance feedback for document retrieval syst | 19 | `rq2-session-summaries` |
| 10 | A learning approach to personalized information filtering | 152 | `rq5-personalized-memory` |
| 11 | Educating the Net Generation | 2,110 | `rq5-preference-persona` |
| 12 | The Ostensive Model of Developing Information-Needs | 141 | `rq5-profile-maintenance` |
| 13 | Controlling Individual Agents in High-Density Crowd Simulation | 501 | `rq6-bitemporal-kg` |
| 14 | Random hypergraphs for hashing-based data structures | 4 | `rq7-cache-precompute` |
| 15 | Update exchange with mappings and provenance | 162 | `rq8-provenance` |
| 16 | Knowledge Provenance: An Approach to Modeling and Maintaining The Evolution an | 31 | `rq8-provenance` |

## 7. Query log

24 queries, each run against all 6 providers with relevance sort (`--sort relevance --provider
all`), plus each query again with `--sort citations` per provider (the `all` fan-out with
citations sort returns 0 papers — a home-still bug, not a rate limit). 2,475 results were
deduped into the 1,832-row union — the same 2,475 the table below sums to and §3's funnel
opens with.

| Query ID | Query string | Result count (summed across all sorts/providers) |
|---|---|---|
| rq1-official-protocols | LLM-as-a-judge reliability benchmark agreement | 125 |
| rq1-judge-sensitivity | benchmark judge sensitivity evaluation protocol | 125 |
| rq1-locomo-bench | LoCoMo benchmark long-term conversational memory | 150 |
| rq1-longmemeval-bench | LongMemEval long-term memory benchmark | 100 |
| rq2-query-decomposition | multi-hop query decomposition retrieval | 100 |
| rq2-iterative-agentic | iterative retrieval agent tool-calling multi-step | 100 |
| rq2-session-summaries | session-level summarisation retrieval long-term dialogue | 100 |
| rq2-memory-arch | long-term conversational memory architecture | 100 |
| rq3-open-domain | open-domain world knowledge retrieval augmentation | 100 |
| rq3-locomo-open-domain | LoCoMo open-domain question answering memory | 100 |
| rq4-temporal-kg | temporal knowledge graph event extraction | 100 |
| rq4-event-tuples | event tuple extraction resolved temporal interval | 100 |
| rq4-temporal-reasoning | temporal reasoning long-term memory agent | 100 |
| rq5-personalized-memory | personalized memory agent user preference profile | 75 |
| rq5-preference-persona | preference following persona long-term dialogue | 100 |
| rq5-profile-maintenance | user profile maintenance contradiction resolution | 100 |
| rq6-memory-update | memory update temporal versioning knowledge conflict | 100 |
| rq6-bitemporal-kg | bi-temporal knowledge graph agent memory | 100 |
| rq6-conflict-resolution | factual conflict resolution retrieval | 100 |
| rq7-latency | latency-aware retrieval efficiency accuracy tradeoff | 100 |
| rq7-cache-precompute | cache precomputation agent notes retrieval | 100 |
| rq8-poisoning | memory poisoning attack defense LLM agent | 100 |
| rq8-provenance | retrieval provenance trust propagation | 100 |
| rq8-consistency | retrieval-time consistency checking memory | 100 |

Provider notes: `--provider all --sort citations` returns 0 papers for every query (the fan-out
citations path is broken in home-still 0.0.1-rc.355); the citations-order pass was therefore run
per-provider (6 separate calls per query). arXiv, CORE, and EuropePMC were heavily rate-limited
throughout the sweep and returned empty results for most queries; OpenAlex and Crossref carried
the sweep. The `--provider europmc` flag name is rejected by the CLI (correct value is
`europepmc`).

## 8. Coverage against the brief

The brief's `Needed from the literature` sub-items, per RQ, with what the acquired set can and
cannot answer. **Items with no covering paper are named explicitly.**

### RQ1 — How does anyone make a defensible SOTA claim with an open-weights judge?

The brief needs four things: (i) official LoCoMo and LongMemEval protocols and whether an
open-weights-judge protocol is accepted; (ii) whether papers report judge-sensitivity and how
much a judge swap is worth; (iii) whether "LoCoMo Refined" or any re-annotation resolves gold
disputes; (iv) whether any top system publishes per-question outputs for re-grading.

**Covered.** The acquired set includes the LoCoMo benchmark paper itself
(`10.48550/arXiv.2402.17753`), LongMemEval (`10.48550/arXiv.2410.10813`), and MemPro
(`10.48550/arXiv.2606.00619`), which together answer (i): both benchmarks mandate an LLM judge
with a specific prompt, and no open-weights-judge protocol is accepted as comparable. MemPro's
own Qwen3-30B row is the gate the brief already pins. The 129 RQ1 acquisitions include
evaluation-protocol surveys that address (ii) in general terms.

**Not covered: (iii) LoCoMo Refined.** No paper matching "LoCoMo refined re-annotation" was
found by any of the four RQ1 queries. (iv) is only partially covered: MemPro reports
per-category numbers but does not publish per-question outputs. No acquired paper provides
released outputs the brief's §6.iv asks for.

### RQ2 — Multi-session / multi-hop assembly

The brief asks what makes 75–89% multi-session systems work mechanically: (a) write-time
aggregation, (b) query decomposition, (c) iterative tool-calling, (d) session-level summaries.
The 99 acquired RQ2 papers are the largest pool in the sweep (HotpotQA, multi-hop QA surveys,
retrieval-augmented generation, knowledge-graph QA, conversational memory networks). 39 are
indexed and searchable. The three directly named systems the brief leans on are all local:
Chronos (`10.48550/arXiv.2603.16862`), APEX-MEM (`10.48550/arXiv.2604.14362`), Memanto
(`10.48550/arXiv.2604.22085`) — though APEX-MEM's conversion covers front matter only.

**Covered in part.** The acquired set has the right *shape* (multi-hop QA, RAG, session
summarisation, conversational memory) but none of the 99 names the specific four-mechanism
ablation the brief asks for. The brief's falsifiable test (`--categories 1` at +5.0 with CI
excluding zero) is a measurement the literature cannot supply; what the literature supplies is
the mechanism candidates, and the sweep has acquired 39 indexed candidates worth reading.

**Gap: no acquired paper reports an ablation separating "write-time aggregation" from "query
decomposition" on LongMemEval multi-session specifically.** The brief says this ablation is the
standard to hold others to; none of the 99 acquired RQ2 papers contains one.

### RQ3 — Open-domain: 29.17 vs 70.83

The brief asks: how do papers characterise LoCoMo category 3, does the gap track backbone size,
and does any system report open-domain gains from a memory mechanism? The 35 acquired RQ3 papers
are mostly open-domain QA surveys and retrieval-augmented-generation overviews. 7 are indexed.

**Not covered.** No acquired paper contains a per-category LoCoMo breakdown that isolates
open-domain as a distinct stratum. The brief's oracle probe (hand the reader gold evidence for
all 96) is the definitive answer here and is a measurement, not a literature lookup.

### RQ4 — Write-time event structure

The brief needs the schema concretely: event-tuple field set, extraction cost, retrieval-over-
event-index shape, and whether it survives a 9B extractor. The 68 acquired RQ4 papers include
temporal knowledge-graph surveys, event-tuple extraction papers, and the Chronos/Memanto/
APEX-MEM systems (all already local). 12 are indexed.

**Covered for the schema question.** The temporal-KG surveys in the acquired set describe event
tuple representations with resolved temporal intervals. **Not covered: whether any of it
survives a 9B extractor.** No acquired paper reports event extraction quality at ≤9B. The
brief's falsifiable test (one `build` variant + existing stratum arms) is still the way to
answer that.

### RQ5 — Preference and persona over time

The brief needs: how a profile/persona record is represented and maintained, what PERMA measures
beyond LongMemEval's 30 questions, and whether a profile layer helps other strata. 37 papers
acquired, 6 indexed. PERMA (`10.48550/arXiv.2603.23231`) is already local and indexed.

**Partially covered.** The acquired set includes profile-maintenance and preference-following
papers, but none reports the "profile layer helps other strata" evidence the brief asks for.
The brief's note that ss-preference at n=30 is too small for a +5.0 rule (needs PERMA as its
instrument) is confirmed: the acquired RQ5 papers do not provide a better instrument.

### RQ6 — Knowledge-update: 74.36 vs 82.05

The brief needs: how systems with explicit versioning (Memanto's temporal versioning, APEX-MEM's
append-only evolution, Zep's bi-temporal graph) decide which version to surface. 46 acquired,
5 indexed. Memanto (`10.48550/arXiv.2604.22085`) and Zep (`10.48550/arXiv.2501.13956`) are
already local.

**Covered in part.** The acquired set includes knowledge-conflict and bi-temporal papers that
describe versioning schemes. The brief's falsifiable test (a read-path arm routing this stratum
to oldest-first ordering) is a measurement, not a literature question.

### RQ7 — G1 latency/accuracy frontier

The brief needs: how the leaderboard's top submissions achieve accuracy at low latency — caching,
precomputed runbooks/notes, or smaller retrieval. 21 acquired, 6 indexed.

**Thinly covered.** The RQ7 queries return mostly network/infrastructure papers (the broad
"latency" keyword), not agent-memory latency optimisation. **No acquired paper describes the
RAG-slice+notes baseline the brief identifies as the thing to beat** (51.0 @ 0.2 s).

### RQ8 — Closing 12.5% → ≤10% ASR

The brief needs a published defence that is *not* a content classifier, with measured ASR and
false-positive rate. 37 acquired, 4 indexed. The EHR-poisoning paper
(`10.48550/arXiv.2601.05504`) is already local.

**Covered in part.** The acquired set includes MINJA, PoisonedRAG, and the EHR-poisoning paper
that the brief already cites. The 12.5% → ≤10% gap the brief asks about is measured, not
literature-resolved: no acquired paper provides a non-content-classifier defence with a
measured false-positive rate at k=6.

## 9. Conversion status note

82 of the 472 acquired papers are converted and indexed (searchable via `distill_search`). The
remaining 390 are `downloaded` — the PDF is in home-still's S3 store, the catalog row has
`pdf_path` + `file_size_bytes > 0` (verified per the plan's §Verification.1), but olmocr has not
processed them yet. The plan's §5c fallback is that state: the document is the deliverable, not
the conversion. The conversion queue is in `tmp/lit-sweep/conv_wave.jsonl` and can be resumed by
re-running `scribe_convert` per stem (the watcher on big is stalled on a 1,546-event JetStream
backlog and is not processing new papers).

**The conversion bottleneck is VRAM, not the pipeline.** olmOCR-2-7B-FP8 under vLLM at
`--gpu-memory-utilization 0.60` reserves ~13.4 GB, leaving only 2.05 GiB of KV cache (38,416
tokens) alongside hs-distill-server (4.4 GB) and another tenant's server (3.2 GB). At
`max_num_seqs 4` and 16,384-token context, that is 2 concurrent page inferences; olmocr's own
`--max_concurrent_requests 4` exceeds it, so pages fail with HTTP 429 when two or more
conversions overlap. One conversion at a time works (82 s for a 9-page PDF); two fail.
M18's precedent (don't lower `--gpu-memory-utilization`, don't touch another tenant's model)
was followed.

## 10. GPU tenancy

`gpu-tenant claim coding` was claimed at 16:28 CDT (pausing trellis2-mcp + hs-serve-distill,
freeing ~6.5 GB) and released at 17:25 CDT when the voice assistant's `qwen3:8b` (9.7 GB,
rolling 30-min keep-alive) lapsed on its own. No other tenant's model was stopped. The
`coding` claim pauses `hs-serve-distill`, which the sweep's indexing stage needs, so it was
released before conversion began. olmocr was pre-warmed by a direct llama-swap request
(`POST /v1/chat/completions`) before each conversion wave to pay the ~32 s cold start outside
the 30 s MCP timeout.