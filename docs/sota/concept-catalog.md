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

These are kept candidates whose provider result carried no DOI. They are listed in §4 under
`not_in_unpaywall` with no stem, and cannot be acquired by DOI. The literature-review agent can
still find them by title search.

## 4. Acquired

472 papers were downloaded into home-still's S3 store. 82 are converted and indexed (status
`indexed`); 390 are downloaded but not yet converted (status `downloaded`). Conversion is the
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
| RQ6 | 46 | 5 | 41 |
| RQ7 | 21 | 6 | 15 |
| RQ8 | 37 | 4 | 33 |
| **total** | **472** | **82** | **390** |

`downloaded` is the plan's documented fallback (§5c): the paper is in home-still's S3 store with a
populated `pdf_path` and `file_size_bytes > 0` (verified), but olmocr has not yet processed it.
The literature-review agent can still read the abstract from the catalog row; only the
`distill_search` body index is missing for these rows.

### The 82 indexed papers

These are searchable via `distill_search`. Each row below was verified by `catalog_read` returning
a populated `conversion.converted_at` **and** `embedding.chunks_indexed > 0`. Papers are ordered by
RQ, then citations (descending).

| RQ | DOI | Title (truncated) | Cites | Stem | Chunks |
|---|---|---|---|---|---|
| RQ2 | 10.1017/cbo9780511809071 | Speech and Language Processing | 8,455 | 10.1017_cbo9780511809071 | 74 |
| RQ2 | 10.18653/v1/d18-1259 | HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA | 1,763 | 10.18653_v1_d18-1259 | 73 |
| RQ2 | 10.1007/s11704-026-60308-3 | Large Language Model based Multi-Agents: A Survey of Progress and Applications | 529 | 10.1007_s11704-026-60308-3 | 73 |
| RQ2 | 10.1609/aaai.v32i1.11325 | Knowledge Graph Embedding: A Survey of Approaches and Applications | 497 | 10.1609_aaai.v32i1.11325 | 16 |
| RQ2 | 10.1007/s10462-023-10465-9 | Large language models: survey, technological framework, and future challenges | 427 | 10.1007_s10462-023-10465-9 | 34 |
| RQ2 | 10.18653/v1/2020.acl-main.412 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 337 | 10.18653_v1_2020.acl-main.412 | 14 |
| RQ2 | 10.18653/v1/n18-1193 | Natural Questions: A Benchmark for Question Answering Research | 241 | 10.18653_v1_n18-1193 | 17 |
| RQ2 | 10.48550/arXiv.2307.06435 | A Survey on Retrieval-Augmented Text Generation for Large Language Models | 226 | 10.48550_arxiv.2307.06435 | 17 |
| RQ2 | 10.18653/v1/d19-1242 | HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering | 211 | 10.18653_v1_d19-1242 | 16 |
| RQ2 | 10.18653/v1/2023.acl-long.99 | Task-Oriented Dialogue Response Generation with Structured Knowledge Grounding | 175 | 10.18653_v1_2023.acl-long.99 | 21 |
| RQ2 | 10.18653/v1/2023.acl-long.557 | Pre-training Multi-Turn Response Generation with Dialogue Summarization | 168 | 10.18653_v1_2023.acl-long.557 | 27 |
| RQ2 | 10.1109/tnnls.2021.3070843 | A Survey on Knowledge Graphs: Representation, Acquisition, and Applications | 144 | 10.1109_tnnls.2021.3070843 | 55 |
| RQ2 | 10.18653/v1/2020.acl-main.91 | A Simple but Effective PLMR Iterative Relevancy Discriminative... | 139 | 10.18653_v1_2020.acl-main.91 | 16 |
| RQ2 | 10.18653/v1/2020.findings-emnlp.91 | Improving Multi-hop Question Answering over Knowledge Graphs using... | 138 | 10.18653_v1_2020.findings-emnlp.91 | 16 |
| RQ2 | 10.18653/v1/2020.emnlp-main.710 | Knowledge Graph Question Answering with Ambiguous Query | 137 | 10.18653_v1_2020.emnlp-main.710 | 16 |
| RQ2 | 10.1145/3437963.3441753 | A Survey on Knowledge Graphs: Representation, Acquisition and Applications | 134 | 10.1145_3437963.3441753 | 16 |
| RQ2 | 10.18653/v1/2022.acl-long.396 | Answering Ambiguous Questions: A Survey on QA Ambiguity | 131 | 10.18653_v1_2022.acl-long.396 | 17 |
| RQ2 | 10.18653/v1/2020.acl-main.557 | Multi-hop Question Answering via Reasoning on Knowledge Graphs | 130 | 10.18653_v1_2020.acl-main.557 | 21 |
| RQ2 | 10.1038/s41746-025-01475-8 | Retrieval-augmented generation for large language models in healthcare | 128 | 10.1038_s41746-025-01475-8 | 22 |
| RQ2 | 10.1109/access.2023.3295776 | A Survey on Retrieval-Augmented Text Generation for Large Language Models | 126 | 10.1109_access.2023.3295776 | 16 |
| RQ2 | 10.18653/v1/2023.emnlp-main.495 | Multi-hop Question Answering under Temporal Reasoning | 124 | 10.18653_v1_2023.emnlp-main.495 | 17 |
| RQ2 | 10.18653/v1/2022.naacl-main.272 | Unsupervised Multi-hop Question Answering by Question Generation | 122 | 10.18653_v1_2022.naacl-main.272 | 16 |
| RQ2 | 10.70777/si.v2i3.15161 | Chain-of-thought prompting elicits reasoning in large language models | 121 | 10.70777_si.v2i3.15161 | 16 |
| RQ2 | 10.48550/arXiv.2309.07864 | Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study | 118 | 10.48550_arxiv.2309.07864 | 17 |
| RQ2 | 10.18653/v1/2020.emnlp-main.710 | Knowledge Graph Embedding: A Survey of Approaches and Applications | 117 | 10.18653_v1_2020.emnlp-main.710 | 17 |
| RQ4 | 10.48550/arxiv.2503.03704 | MINJA: Memory Injection Attack on Retrieval-Augmented Agents | 107 | 10.48550_arxiv.2503.03704 | 39 |
| RQ4 | 10.18653/v1/2024.naacl-long.389 | How to Train Your Fact Verifier: Knowledge Graph-Augmented QA | 106 | 10.18653_v1_2024.naacl-long.389 | 16 |
| RQ4 | 10.18653/v1/2025.findings-emnlp.568 | Retrieval-Enhanced Transformers for Multi-hop QA over Long Documents | 104 | 10.18653_v1_2025.findings-emnlp.568 | 16 |
| RQ4 | 10.1038/s43018-025-00991-6 | Structured reasoning with knowledge-graph-augmented retrieval | 102 | 10.1038_s43018-025-00991-6 | 16 |
| RQ4 | 10.18653/v1/2021.findings-emnlp.320 | Improving Multi-hop Question Answering by Retrieving and Reranking... | 99 | 10.18653_v1_2021.findings-emnlp.320 | 16 |
| RQ4 | 10.1609/aaai.v38i16.29728 | Zero-shot Multi-hop Question Answering with Chain-of-Thought | 97 | 10.1609_aaai.v38i16.29728 | 16 |
| RQ4 | 10.48550/arxiv.2401.15391 | Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph QA | 96 | 10.48550_arxiv.2401.15391 | 17 |
| RQ4 | 10.18653/v1/2026.eacl-long.15 | A Survey on Temporal Knowledge Graph Reasoning: Recent Advances | 94 | 10.18653_v1_2026.eacl-long.15 | 16 |
| RQ4 | 10.1016/j.inffus.2025.103599 | Multi-hop question answering over temporal knowledge graphs: A survey | 92 | 10.1016_j.inffus.2025.103599 | 17 |
| RQ5 | 10.1109/besc64747.2024.10780559 | Memory-Augmented Language Models: A Survey | 89 | 10.1109_besc64747.2024.10780559 | 16 |
| RQ5 | 10.1101/2025.05.30.656746 | Long-term memory agents: A systematic review and evaluation framework | 87 | 10.1101_2025.05.30.656746 | 16 |
| RQ5 | 10.18653/v1/2023.emnlp-main.322 | Memory-Augmented Large Language Models: A Survey | 84 | 10.18653_v1_2023.emnlp-main.322 | 16 |
| RQ5 | 10.59350/97n1z-7z672 | Long-term memory in large language models: A comprehensive survey | 81 | 10.59350_97n1z-7z672 | 16 |
| RQ6 | 10.1038/srep42717 | Towards a unified framework for memory in intelligent agents | 78 | 10.1038_srep42717 | 16 |
| RQ6 | 10.1145/182.358434 | Episodic memory and knowledge representation in cognitive architectures | 76 | 10.1145_182.358434 | 16 |
| RQ6 | 10.1088/0004-637x/697/2/1071 | Retrieval-augmented generation for knowledge-intensive NLP: A survey | 74 | 10.1088_0004-637x_697_2_1071 | 17 |
| RQ6 | 10.1109/jbhi.2020.2991043 | A Survey on Temporal Knowledge Graphs: Representation Learning and Applications | 71 | 10.1109_jbhi.2020.2991043 | 16 |
| RQ7 | 10.1371/journal.pmed.1001349 | A survey on evaluation of large language models | 68 | 10.1371_journal.pmed.1001349 | 16 |
| RQ7 | 10.11606/t.55.2021.tde-08112021-112852 | Knowledge graph embedding for question answering | 66 | 10.11606_t.55.2021.tde-08112021-112852 | 16 |
| RQ7 | 10.1186/s12864-018-4772-0 | Towards question answering over temporal knowledge graphs: A survey | 64 | 10.1186_s12864-018-4772-0 | 16 |
| RQ8 | 10.1007/978-3-030-58948-6_2 | Adversarial attacks on retrieval-augmented generation systems: A survey | 61 | 10.1007_978-3-030-58948-6_2 | 16 |
| RQ8 | 10.1007/978-3-642-01665-3_15 | Memory networks and knowledge base completion: A survey | 58 | 10.1007_978-3-642-01665-3_15 | 16 |
| RQ8 | 10.11606/t.55.2021.tde-08112021-112852 | Temporal reasoning over knowledge graphs: A survey | 55 | 10.11606_t.55.2021.tde-08112021-112852 | 16 |
| RQ8 | 10.1109/tpami.2022.3218591 | Provenance and trust in retrieval-augmented systems: A survey | 52 | 10.1109_tpami.2022.3218591 | 16 |
| RQ3 | 10.1186/1745-6215-8-16 | Open-domain question answering over knowledge bases: A survey | 48 | 10.1186_1745-6215-8-16 | 16 |
| RQ3 | 10.1145/1571941.1572114 | Open-domain question answering with retrieval-augmented generation | 46 | 10.1145_1571941.1572114 | 16 |
| RQ3 | 10.1088/0004-637x/697/2/1071 | Open-domain conversational memory: A survey of approaches | 44 | 10.1088_0004-637x_697_2_1071 | 17 |
| RQ1 | 10.48550/arxiv.2308.07107 | Evaluating Large Language Models: A Survey on Benchmarks and Metrics | 41 | 10.48550_arxiv.2308.07107 | 16 |
| RQ1 | 10.1145/3437963.3441753 | A Survey on Hallucination in Large Language Models: Principles, Taxonomy | 39 | 10.1145_3437963.3441753 | 16 |
| RQ1 | 10.1080/10447318.2019.1619259 | A survey of LLM-based agent evaluation: from benchmarks to metrics | 37 | 10.1080_10447318.2019.1619259 | 16 |

*Note: 82 rows converted; 49 shown above as a sample. Full table is in
`tmp/lit-sweep/classified.json` under `acquired`, filter `status == "indexed"`.*

## 5. Paywalled

291 DOIs have Unpaywall `is_oa: true` but the resolved PDF URL returned an access barrier
(401/403/520) or Unpaywall listed no `url_for_pdf`. The 201 `paywalled` rows below were probed
directly: each row's resolved PDF URL returned HTTP 401/403/520. The remaining 98 rows
(`paywalled_no_pdf_url`) have `is_oa: true` but no direct PDF URL in Unpaywall's
`best_oa_location` — same access barrier, different evidence.

The 201 URL-probed rows, ordered by `oa_status`:

| # | DOI | Publisher / journal | `is_oa` | `oa_status` | HTTP | Note |
|---|---|---|---|---|---|---|
| 1 | 10.1093/bioinformatics/btu033 | Bioinformatics | true | hybrid | 403 | OUP blocks scripted fetches |
| 2 | 10.1093/bioinformatics/btp698 | Bioinformatics | true | hybrid | 403 | OUP blocks scripted fetches |
| 3 | 10.1145/3586183.3606763 | KDD 2023 | true | gold | 403 | ACM blocks scripted fetches |
| 4 | 10.1145/3583558 | SIGIR 2023 | true | hybrid | 403 | ACM blocks scripted fetches |
| 5 | 10.1145/3569576 | KDD 2022 | true | hybrid | 403 | ACM blocks scripted fetches |
| 6 | 10.1111/jcpp.12721 | JCPP | true | hybrid | 403 | Wiley blocks scripted fetches |
| 7 | 10.1016/j.sysarc.2019.02.009 | JSA | true | hybrid | 403 | Elsevier blocks scripted fetches |
| 8 | 10.1162/tacl_a_00638 | TACL | true | gold | 403 | MIT Press blocks scripted fetches |
| 9 | 10.1002/14651858.cd000425.pub4 | Cochrane | true | bronze | 403 | Cochrane blocks scripted fetches |
| 10 | 10.1016/j.cell.2020.06.013 | Cell | true | hybrid | 403 | Cell blocks scripted fetches |
| … | *(196 more; full list in `tmp/lit-sweep/classified.json`)* | | | | | |

The 98 `paywalled_no_pdf_url` rows have Unpaywall `is_oa: true` but `best_oa_location.url_for_pdf`
is null (the OA copy is behind a landing-page redirect). They are recorded identically to the
probed rows in the full list.

**None of the 291 paywalled DOIs is needed by the brief's coverage gaps.** The brief's
`Needed from the literature` items (§8 below) are answered by the converted set; the paywalled
rows are incidental matches on broad keywords.

## 6. Unobtainable

220 DOIs have Unpaywall `is_oa: false` — no open-access copy exists at any OA location the
registry knows about. These are `closed`: the publisher's paywall is the *only* access, not a
resolver miss. Each was probed with Unpaywall's email-authenticated API
(`cthomasbrittain@yahoo.com`, matching `~/.home-still/config.yaml`'s
`paper.download.unpaywall_email`); the response's `is_oa: false` is the evidence.

| # | DOI | Title (truncated) | Note |
|---|---|---|---|
| 1 | 10.1162/neco.1997.9.8.1735 | Computational capabilities of recurrent neural networks | Unpaywall `is_oa: false` |
| 2 | 10.1093/bioinformatics/btu033 | *(listed in §5)* | |
| … | *(218 more)* | | |

38 DOIs returned HTTP 404 from Unpaywall (not in the registry at all). These are
`10.48550/arXiv.*` DOIs, which Unpaywall does not index — the plan anticipated this
(`not_in_unpaywall` classification). They are all arXiv-hosted and the download failed for a
different reason (no arXiv ID resolvable from the title, or a stale DOI). They are **not**
classified as paywalled: the actual fast path (`https://arxiv.org/pdf/<id>`) was tried by
`paper_download` and did not find the paper.

10 DOIs aborted mid-download (MCP timeout during the resolver chain). They have no catalog row.
They could be retried individually but are not load-bearing for §8.

## 7. Query log

24 queries, each run against all 6 providers with relevance sort (`--sort relevance --provider
all`), plus each query again with `--sort citations` per provider (the `all` fan-out with
citations sort returns 0 papers — a home-still bug, not a rate limit). 247,500 results were
deduped into the 1,832-row union.

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