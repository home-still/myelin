# Corpus catalog — papers relevant to the myelin SOTA project

Single reference for agents working in this repo: everything in the home-still corpus
(`/Users/ladvien/mnt/home-still`, S3-backed) that bears on `docs/sota/research-brief.md`.
Generated 2026-09-17; conversion statuses verified against the live catalog that day.

## How to use the corpus

- **Search full text**: `hs distill search "<phrase>"` (CLI) or MCP `distill_search`.
  Only `indexed` papers hit. The index holds ~9,800 documents; only those catalogued here
  are known to bear on this project — do not treat other index hits as project-relevant
  without checking them against the brief.
- **Paths** (DOI with `/` → `_`, lowercased; shard = first two chars of the stem):
  - PDF: `papers/<shard>/<stem>.pdf` (occasionally `.html`)
  - Converted text: `markdown/<shard>/<stem>.md`
  - Metadata: `catalog/<shard>/<stem>.yaml` (created at conversion time)
- **Statuses**: `indexed` = converted + in the Qdrant index (full-text searchable);
  `pdf-only` = source stored, text not yet extracted — abstract only unless you read the PDF.
- **Numbers live in the repo, not the corpus**: our measurements in `runs/standing/standing.json`,
  published claims with verbatim quotes in `docs/sota/registry.json`, sweep funnel in
  `docs/sota/concept-catalog.md`.

## Population

| set | papers | indexed |
|---|---|---|
| A. Core competitors & mechanism papers (pre-existing corpus) | 40 | 40 |
| B. Literature-sweep open-access acquisitions | 472 | 84 |
| C. Anna's Archive paywalled acquisitions | 27 | 20 |
| **total** | **539** | **144** |

Sets are disjoint. A was found by semantic search over the pre-existing corpus; B came from the
sweep's provider searches via MCP `paper_download`; C was fetched manually from Anna's Archive
because paywalled.

---

## Set A — SOTA competitors and mechanism papers (40, all indexed)

These are the papers the brief's RQs name or directly imply. Every one is converted and
searchable. Read the markdown, don't re-download.

### RQ1 — judging and evaluation methodology

| DOI / stem | Title | Why it matters here |
|---|---|---|
| `10.48550_arxiv.2411.16594` | From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge | LLM-as-a-judge survey; RQ1 protocol comparability |
| `10.48550_arxiv.2403.18771` | CheckEval: A reliable LLM-as-a-Judge framework for evaluating text generation using checklists | CheckEval: reliability of LLM judges; RQ1 |
| `10.48550_arXiv.2306.05685` | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | MT-Bench: the canonical LLM-judge agreement paper; RQ1(i) |
| `10.48550_arxiv.2503.13657` | Why Do Multi-Agent LLM Systems Fail? | why multi-agent systems fail; RQ1(ii) sensitivity |
| `10.48550_arxiv.2507.02825` | Establishing Best Practices for Building Rigorous Agentic Benchmarks | rigorous agentic benchmark practice; RQ1(ii) |
| `10.48550_arXiv.2404.12272` | Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences | validator alignment; RQ1(ii) |
| `10.48550_arxiv.2601.01743` | AI Agent Systems: Architectures, Applications, and Evaluation | agent systems evaluation survey; RQ1 |

### RQ2 — multi-session / multi-hop assembly

| DOI / stem | Title | Why it matters here |
|---|---|---|
| `10.48550_arxiv.2508.03341` | What Deserves Memory: Adaptive Memory Distillation for LLM Agents | NEMORI: adaptive memory distillation; registry row source |
| `10.48550_arxiv.2504.19413` | Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory | Mem0: production memory pipeline; RQ2(b) candidate |
| `10.48550_arxiv.2502.12110` | A-MEM: Agentic Memory for LLM Agents | A-MEM: agentic memory, note linking; RQ2 mechanism |
| `10.48550_arxiv.2405.14831` | HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models | HippoRAG: hippocampal index; RQ2(c) analog |
| `10.48550_arxiv.2509.25911` | Mem-α: Learning Memory Construction via Reinforcement Learning | Mem-α: RL-learned memory construction; RQ2(a) |
| `10.48550_arxiv.2502.06975` | Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents | position: episodic memory missing piece; RQ2 framing |
| `10.18653_v1_2023.findings-emnlp.620` | Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy | iterative retrieval; RQ2(c) |
| `10.48550_arxiv.2309.02427` | Cognitive Architectures for Language Agents | CoALA: taxonomy of agent memory; RQ2 framing |
| `10.48550_arxiv.2305.16291` | Voyager: An Open-Ended Embodied Agent with Large Language Models | Voyager: skill library as memory; RQ2(c) analog |
| `10.48550_arxiv.2303.11366` | Reflexion: Language Agents with Verbal Reinforcement Learning | Reflexion: verbal RL self-improvement; RQ2(c) analog |
| `10.48550_arxiv.2604.12285` | GAM: Hierarchical Graph-based Agentic Memory for LLM Agents | GAM: hierarchical graph memory; registry comparison target |
| `10.48550_arxiv.2404.13501` | A Survey on the Memory Mechanism of Large Language Model based Agents | memory mechanism survey for LLM agents; RQ2 umbrella |
| `10.18653_v1_2024.emnlp-main.981` | Searching for Best Practices in Retrieval-Augmented Generation | RAG best practices; RQ2 |
| `10.48550_arxiv.2507.07957` | MIRIX: Multi-Agent Memory System for LLM-Based Agents | MIRIX: multi-agent memory types; RQ2 |
| `10.48550_arxiv.2310.11511` | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection |  |

### RQ3 — open-domain retrieval

| DOI / stem | Title | Why it matters here |
|---|---|---|
| `10.18653_v1_2021.naacl-main.466` | RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering | RocketQA: dense retrieval training; RQ3 |
| `10.48550_arxiv.2005.11401` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | the original RAG paper; RQ3 baseline |
| `10.18653_v1_2020.emnlp-main.550` | Dense Passage Retrieval for Open-Domain Question Answering | DPR: dense retrieval baseline; RQ3 |
| `10.18653_v1_2024.naacl-long.463` | REPLUG: Retrieval-Augmented Black-Box Language Models | REPLUG; RQ3 |
| `10.48550_arxiv.1811.01241` | Wizard of Wikipedia: Knowledge-Powered Conversational agents | Wizard of Wikipedia: knowledge-grounded dialogue; RQ3 |
| `10.18653_v1_2021.emnlp-main.168` | Neural Path Hunter: Reducing Hallucination in Dialogue Systems via Path Grounding | Neural Path Hunter: KG-path retrieval reduces hallucination; RQ3 |

### RQ4 — write-time event / temporal structure

| DOI / stem | Title | Why it matters here |
|---|---|---|
| `10.48550_arxiv.2603.16862` | Chronos: Temporal-Aware Conversational Agents with Structured Event Retrieval for Long-Term Memory | Chronos: event records + calendar; brief RQ4 lead (58.9% of gain from events) |
| `10.48550_arxiv.2604.14362` | APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI | APEX-MEM: multi-tool memory agent; brief RQ4 lead |
| `10.48550_arxiv.2502.16090` | Echo: A Large Language Model with Temporal Episodic Memory | Echo: temporal episodic memory; RQ4 |

### RQ5 — preference and persona

| DOI / stem | Title | Why it matters here |
|---|---|---|
| `10.18653_v1_2024.findings-naacl.229` | PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits | PersonaLLM: persona consistency; RQ5 |
| `10.48550_arxiv.2305.02547` | PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits | PersonaLLM (earlier version); RQ5 |
| `10.48550_arxiv.2408.10903` | BEYOND DIALOGUE: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model | profile-dialogue alignment; RQ5 |
| `10.48550_arxiv.2603.23231` | PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments | PERMA: the preference benchmark; brief RQ5 instrument |

### RQ6 — knowledge update / versioning

| DOI / stem | Title | Why it matters here |
|---|---|---|
| `10.48550_arxiv.2604.22085` | Memanto: Typed Semantic Memory with Information-Theoretic Retrieval for Long-Horizon Agents | Memanto: typed semantic memory, temporal versioning; brief RQ6 names it |
| `10.48550_arxiv.2501.13956` | Zep: A Temporal Knowledge Graph Architecture for Agent Memory | Zep: bi-temporal KG; brief RQ6 names it |

### RQ7 — latency frontier

| DOI / stem | Title | Why it matters here |
|---|---|---|
| `10.48550_arxiv.2407.01502` | AI Agents That Matter | AI Agents That Matter: cost/accuracy methodology; RQ7 |
| `10.48550_arxiv.2311.04934` | Prompt Cache: Modular Attention Reuse for Low-Latency Inference | Prompt Cache: KV reuse for low latency; RQ7 caching lead |

### RQ8 — poisoning defence

| DOI / stem | Title | Why it matters here |
|---|---|---|
| `10.48550_arxiv.2601.05504` | EHR memory poisoning: pre-populated record attacks on clinical retrieval agents | EHR-agent memory poisoning: the pre-populated-record threat model behind M15 |

## Set B — literature-sweep open-access acquisitions (472)

Acquired via MCP `paper_download` during the sweep (see `docs/sota/concept-catalog.md` §4 for
funnel arithmetic). `indexed` = converted + searchable; `pdf-only` = PDF in the store, not yet
converted — the abstract is still usable from the sweep records (`tmp/lit-sweep/downloads.jsonl`).

**Set B's count is an upper bound on relevant papers.** The sweep's topic filter matched on the
bare words "memory" and "retrieval", which admitted a long tail of bioinformatics and
astronomy hits that have nothing to do with agent memory — `10.1038/nmeth.3317` (HISAT, "low
memory requirements"), `10.1186/1471-2105-10-421` (BLAST+), `10.1093/bioinformatics/btt086`
(QUAST), `10.1073/pnas.0506580102` (GSEA) and `10.3847/1538-3881/aabc4f` (the Astropy Project)
are all still in the tables below. They really are in the corpus, so they are listed rather than
quietly dropped; treat a high-citation row whose title is about genomes or telescopes as filter
noise, not as a paper the brief needs.

Chunk counts are `embedding.chunks_indexed` read back from the live home-still catalog, one
`catalog_read` per stem.

### RQ1 — 129 papers (3 indexed, 126 pdf-only)

**Indexed (searchable):**

| DOI | Title | Cites | Chunks |
|---|---|---|---|
| `10.48550/arxiv.2301.05339` | A Comprehensive Review of Data-Driven Co-Speech Gesture Generation | 9 | 58 |
| `10.64823/ijcsa.2601002` | Beyond Price and Benchmark: A Cost–Methodology–Fit Framework for Selecting AI Developer Tools, with a Proposed Evaluation Protocol | 0 | 9 |
| `10.32604/cmc.2026.081260` | HalluBench: A Multi-LLM Benchmark for Hallucination Evaluation and Reliability Analysis | 0 | 1 |

**PDF-only (not yet converted):**

<details><summary>126 papers (click to expand)</summary>

| DOI | Title | Cites |
|---|---|---|
| `10.1371/journal.pmed.1000100` | The PRISMA Statement for Reporting Systematic Reviews and Meta-Analyses of Studies That Evaluate Health Care Interventions: Explanation and Elaboration | 28058 |
| `10.1136/bmj.b2700` | The PRISMA statement for reporting systematic reviews and meta-analyses of studies that evaluate healthcare interventions: explanation and elaboration | 17951 |
| `10.1038/s41586-023-06291-2` | Large language models encode clinical knowledge | 3807 |
| `10.2427/5768` | The PRISMA statement for reporting systematic reviews and meta-analyses of studies that evaluate health care interventions: explanation and elaboration | 3482 |
| `10.1186/s12880-015-0068-x` | Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool | 2920 |
| `10.1609/aaai.v32i1.11694` | Deep Reinforcement Learning That Matters | 1585 |
| `10.18653/v1/2021.emnlp-main.595` | CLIPScore: A Reference-free Evaluation Metric for Image Captioning | 1121 |
| `10.1038/s41591-024-03423-7` | Toward expert-level medical question answering with large language models | 930 |
| `10.1038/s41591-024-02855-5` | Adapted large language models can outperform medical experts in clinical text summarization | 753 |
| `10.18653/v1/2023.emnlp-main.153` | G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment | 736 |
| `10.48550/arxiv.2305.14314` | QLoRA: Efficient Finetuning of Quantized LLMs | 509 |
| `10.1038/s41592-019-0457-0` | Best practices and benchmarks for intact protein analysis for top-down mass spectrometry | 404 |
| `10.1038/s41746-024-01258-7` | A framework for human evaluation of large language models in healthcare derived from literature review | 370 |
| `10.1007/s10822-008-9196-5` | Recommendations for evaluation of computational methods | 353 |
| `10.1186/s13321-017-0232-0` | Beyond the hype: deep neural networks outperform established methods using a ChEMBL bioactivity benchmark set | 349 |
| `10.48550/arXiv.2410.12784` | JudgeBench: A Benchmark for Evaluating LLM-based Judges | 342 |
| `10.48550/arxiv.2305.09617` | Towards Expert-Level Medical Question Answering with Large Language Models | 335 |
| `10.1371/journal.pcbi.0020065` | A Community Resource Benchmarking Predictions of Peptide Binding to MHC-I Molecules | 293 |
| `10.2197/ipsjjip.17.242` | Proposal and Quantitative Analysis of the CHStone Benchmark Program Suite for Practical C-based High-level Synthesis | 286 |
| `10.18653/v1/2023.emnlp-main.397` | HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models | 276 |
| `10.48550/arxiv.2212.13138` | Large Language Models Encode Clinical Knowledge | 262 |
| `10.48550/arxiv.2309.01219` | Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models | 244 |
| `10.48550/arxiv.2402.06196` | Large Language Models: A Survey | 229 |
| `10.1186/s13059-019-1738-8` | Essential guidelines for computational method benchmarking | 228 |
| `10.6028/nist.ai.600-1` | Artificial intelligence risk management framework : | 207 |
| `10.18653/v1/2025.emnlp-main.138` | From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge | 102 |
| `10.48550/arxiv.2306.13063` | Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs | 54 |
| `10.48550/arXiv.2510.27246` | Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs | 50 |
| `10.48550/arxiv.2411.15594` | A Survey on LLM-as-a-Judge | 41 |
| `10.48550/arXiv.2507.00769` | LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing | 40 |
| `10.18653/v1/2024.findings-emnlp.79` | R-Judge: Benchmarking Safety Risk Awareness for LLM Agents | 39 |
| `10.48550/arxiv.2412.05579` | LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods | 38 |
| `10.48550/arXiv.2601.03515` | Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory for MLLM Agents | 37 |
| `10.48550/arXiv.2506.04078` | LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation | 37 |
| `10.18653/v1/2025.acl-long.1176` | HalluLens: LLM Hallucination Benchmark | 34 |
| `10.18653/v1/2024.emnlp-main.427` | Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment | 32 |
| `10.48550/arXiv.2509.21212` | SGMem: Sentence Graph Memory for Long-Term Conversational Agents | 31 |
| `10.48550/arxiv.2306.05087` | PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization | 29 |
| `10.1038/s42256-025-01169-6` | When large language models are reliable for judging empathic communication | 28 |
| `10.48550/arXiv.2510.09738` | Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement | 26 |
| `10.48550/arXiv.2503.04474` | Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges | 24 |
| `10.18653/v1/2024.findings-emnlp.592` | Can LLM be a Personalized Judge? | 20 |
| `10.48550/arXiv.2601.03444` | Grading Scale Impact on LLM-as-a-Judge: Human-LLM Alignment Is Highest on 0-5 Grading Scale | 18 |
| `10.48550/arXiv.2511.17208` | A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents | 16 |
| `10.48550/arXiv.2505.19549` | Towards Multi-Granularity Memory Association and Selection for Long-Term Conversational Agents | 16 |
| `10.18653/v1/2025.emnlp-main.1318` | Memory OS of AI Agent | 14 |
| `10.48550/arXiv.2506.13356` | StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns | 13 |
| `10.48550/arXiv.2604.08256` | HyperMem: Hypergraph Memory for Long-Term Conversations | 11 |
| `10.48550/arXiv.2602.10715` | Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents | 10 |
| `10.1101/2025.10.27.25338910` | Human Evaluators vs. LLM-as-a-Judge: Toward Scalable, Real-Time Evaluation of GenAI in Global Health | 10 |
| `10.48550/arXiv.2510.09011` | TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation | 9 |
| `10.18653/v1/2026.findings-acl.1330` | A Survey on Evaluation of LLM-based Agents | 8 |
| `10.48550/arXiv.2412.15524` | HREF: Human Response-Guided Evaluation of Instruction Following in Language Models | 8 |
| `10.48550/arXiv.2606.19544` | Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias | 8 |
| `10.48550/arXiv.2604.23478` | JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems | 7 |
| `10.1145/3805712.3808601` | Auto-Judge: A Cross-Task Benchmark for Comparing LLM Judges for Citation-Grounded RAG Systems | 6 |
| `10.18653/v1/2024.findings-emnlp.708` | Stark: Social Long-Term Multi-Modal Conversation with Persona Commonsense Knowledge | 5 |
| `10.48550/arXiv.2605.19196` | Time to REFLECT: Can We Trust LLM Judges for Evidence-based Research Agents? | 5 |
| `10.48550/arXiv.2604.23178` | Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines | 5 |
| `10.48550/arXiv.2603.05399` | Judge Reliability Harness: Stress Testing the Reliability of LLM Judges | 5 |
| `10.48550/arXiv.2602.04796` | LALM-as-a-Judge: Benchmarking Large Audio-Language Models for Safety Evaluation in Multi-Turn Spoken Dialogues | 4 |
| `10.48550/arxiv.2412.00804` | Examining Identity Drift in Conversations of LLM Agents | 4 |
| `10.48550/arXiv.2510.06538` | Auto-Prompt Ensemble for LLM Judge | 4 |
| `10.48550/arXiv.2504.05706` | SEVERE++: Evaluating Benchmark Sensitivity in Generalization of Video Representation Learning | 3 |
| `10.48550/arXiv.2601.06282` | Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning | 3 |
| `10.1609/aaai.v40i19.38679` | Personalize Before Retrieve: LLM-based Personalized Query Expansion for User-Centric Retrieval | 3 |
| `10.18653/v1/2025.findings-acl.1095` | Measuring What Makes You Unique: Difference-Aware User Modeling for Enhancing LLM Personalization | 3 |
| `10.48550/arXiv.2604.04532` | Multilingual Prompt Localization for Agent-as-a-Judge: Language and Backbone Sensitivity in Requirement-Level Evaluation | 2 |
| `10.48550/arXiv.2604.18164` | MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge | 2 |
| `10.48550/arXiv.2603.14597` | D-MEM: Dopamine-Gated Agentic Memory via Reward Prediction Error Routing | 2 |
| `10.48550/arXiv.2604.27695` | EviMem: Evidence-Gap-Driven Iterative Retrieval for Long-Term Conversational Memory | 2 |
| `10.18653/v1/2026.findings-acl.1337` | From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents | 2 |
| `10.18653/v1/2025.emnlp-main.1069` | Flexibly Utilize Memory for Long-Term Conversation via a Fragment-then-Compose Framework | 2 |
| `10.48550/arxiv.2412.15204` | LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks | 2 |
| `10.48550/arXiv.2606.15610` | LLM Judges Have Dark Current: A Psychometric Datasheet for LLM-as-a-Judge Evaluation | 1 |
| `10.48550/arXiv.2607.01153` | Adversarial Pragmatics for AI Safety Evaluation: A Diagnostic Framework and Seed Benchmark for Language-Mediated Control | 1 |
| `10.48550/arXiv.2606.22030` | When Does Belief-Based Agent Memory Help? Reliability-Conditional Updating and Provenance-Capped Poisoning Defense | 1 |
| `10.48550/arXiv.2605.25092` | AgentIR: A Workload-Adaptive Cascade Retrieval Substrate for Long-Term Conversational Memory | 1 |
| `10.48550/arXiv.2604.02431` | SelRoute: Query-Type-Aware Routing for Long-Term Conversational Memory Retrieval | 1 |
| `10.48550/arXiv.2607.00017` | Learning User-Aware Recall: Personalized Retrieval in Long-Term Conversational Memory | 1 |
| `10.48550/arxiv.2507.05257` | Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions | 1 |
| `10.1145/3805712.3808634` | AlpsBench: An LLM Personalization Benchmark for Real-Dialogue Memorization and Preference Alignment | 1 |
| `10.48550/arxiv.2512.16301` | Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills | 1 |
| `10.48550/arxiv.2504.04717` | Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models | 1 |
| `10.18653/v1/2025.emnlp-main.1214` | Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking | 1 |
| `10.48550/arXiv.2606.13685` | The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation | 1 |
| `10.1038/s41746-026-02992-w` | Human evaluators vs. LLM-as-a-Judge: toward scalable evaluation of GenAI in global health. | 1 |
| `10.48550/arXiv.2603.14732` | LLM-as-a-judge validity in physics assessment depends more on the task than the model | 1 |
| `10.48550/arXiv.2607.01103` | Clinician-Level Agreement Without Clinical Caution: LLM Evaluator Limits in Medical AI Benchmarking | 1 |
| `10.18653/v1/2026.gem-main.19` | An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability | 1 |
| `10.21203/rs.3.rs-10476655/v1` | From Benchmark Reuse to Benchmark Audit: A Version-Aware Re-Evaluation of a Public MODIS Wildfire Benchmark with Uncertainty, Sensitivity, and Calibration Analysis | 0 |
| `10.18653/v1/2026.gem-main.23` | MCJudgeBench: A Benchmark for Constraint-Level Judge Evaluation in Multi-Constraint Instruction Following | 0 |
| `10.7256/2454-0714.2026.3.80458` | Comparative testing of open-source LLMs in professional subject area tasks: a multi-domain benchmark with multi-judge and expert evaluation | 0 |
| `10.15446/rbct.n60.126975` | Evaluación comparativa de sensibilidad técnico-económica bajo unaenvolvente Benchmark fija en minería a cielo abierto | 0 |
| `10.2172/1831619` | Sensitivity and Uncertainty of the IFR-1 BISON Benchmark | 0 |
| `10.21203/rs.3.rs-10572455/v1` | A Validated Measurement Protocol for Comparable, Cost-Aware Software Testing Evaluation: A Reproducible Benchmark, an Oracle Sampling-Budget Guarantee, and a Real-Fault Validity Study, Instantiated for Quantum Programs | 0 |
| `10.21203/rs.3.rs-10884991/v1` | CleanScore: Black-Box Benchmark Audits with Negative Controls and Sensitivity Bounds | 0 |
| `10.18653/v1/2026.acl-long.488` | Beyond Word Boundaries: A Hebrew Coreference Benchmark and an Evaluation Protocol for Morphologically Complex Text | 0 |
| `10.1088/2053-2563/ab0823ch1` | A sensitivity benchmark for optical biosensors | 0 |
| `10.48550/arXiv.2605.06327` | Measuring Evaluation-Context Divergence in Open-Weight LLMs: A Paired-Prompt Protocol with Pilot Evidence of Alignment-Pipeline-Specific Heterogeneity | 0 |
| `10.48550/arXiv.2605.28848` | GPF-LiveNews: A Streaming Evaluation Protocol for Group-Conditioned Framing in Large Language Models | 0 |
| `10.48550/arXiv.2608.24258` | Beyond Accuracy: A Dual-Judge Evaluation Protocol for Vision-Language Models in Legally Grounded Tasks | 0 |
| `10.48550/arXiv.2608.02620` | JudgeArena: A Unified Framework for Reproducible LLM-Judge Evaluation | 0 |
| `10.18653/v1/2026.acl-srw.33` | Confidence as a Tie-Breaker: Reassessing Multilingual Hedging Bias in LLM-as-a-Judge Evaluation | 0 |
| `10.48550/arXiv.2608.01810` | RADAR: Rubric-Aware Dependency and Redundancy Analysis for LLM-as-Judge Evaluation | 0 |
| `10.48550/arXiv.2602.20379` | Case-Aware LLM-as-a-Judge Evaluation for Enterprise-Scale RAG Systems | 0 |
| `10.1922/ejprd.v34i7s.1683` | TG-HWBO: Transformer-Guided Hybrid Whale-Bee Optimization and a Reproducible Benchmark Protocol for Semantic-Aware MEC Offloading | 0 |
| `10.48550/arXiv.2606.01629` | Benchmarking LLM-as-a-Judge for Long-Form Output Evaluation | 0 |
| `10.48550/arXiv.2609.12439` | Debiasing as a Measurement Intervention: Calibrated Ties and Resolution Loss in LLM-as-a-Judge Evaluation | 0 |
| `10.48550/arXiv.2608.00009` | AgentMemBench: A Systematic Benchmark for Evaluating Long-Term Memory Management Strategies in Conversational AI Agents | 0 |
| `10.48550/arXiv.2609.05441` | When Does Memory Help? A Cost-Aware Evaluation of Long-Term Memory in Tool-Using LLM Agents | 0 |
| `10.48550/arXiv.2608.30508` | UTILMEM: Benchmarking Evidence Utilization in Long-Term Conversational Memory | 0 |
| `10.48550/arXiv.2609.12354` | CueMem: Cue-Guided Context Reconstruction for Long-Term Conversational Memory | 0 |
| `10.48550/arXiv.2609.07093` | Where to Look and What to Use: Retrieve-Localize-Generate for Long-Term Conversational Memory Question Answering | 0 |
| `10.18653/v1/2026.acl-long.370` | Mem2ActBench: A Benchmark for Evaluating Long-Term Memory Utilization in Task-Oriented Autonomous Agents | 0 |
| `10.21203/rs.3.rs-9801639/v1` | From Organizational Knowledge to AI Agent Memory: Empirical Validation of the SECI Model on the LongMemEval Benchmark | 0 |
| `10.22215/etd/1996-03435` | Hypnosis, hypermnesia and memory distortion in long term memory. | 0 |
| `10.31234/osf.io/gm6u2` | Pre-existing long-term memory facilitates the formation of visual short-term memory | 0 |
| `10.18653/v1/2026.findings-acl.1835` | LiCoMemory: Lightweight and Cognitive Agentic Memory for Efficient Long-Term Reasoning | 0 |
| `10.48550/arXiv.2609.12002` | Can We Trust LLM Judges: A Study of Capability-Dependent Biases and Multi-Judge Ensemble for Bias Calibration | 0 |
| `10.48550/arXiv.2608.24314` | Benchmarking LLM Judges for Voice-Agent Evaluation: Reliability, Calibration, and Human Oversight | 0 |
| `10.48550/arXiv.2605.06652` | When No Benchmark Exists: Validating Comparative LLM Safety Scoring Without Ground-Truth Labels | 0 |
| `10.48550/arXiv.2605.09610` | SmartEval: A Benchmark for Evaluating LLM-Generated Smart Contracts from Natural Language Specifications | 0 |
| `10.21203/rs.3.rs-10304807/v1` | Can LLMs Judge Legal Accuracy? Reliability of LLM Evaluators for High-Stakes Insurance QA in a Low-Resource Language | 0 |
| `10.21203/rs.3.rs-10047888/v1` | A Review of a Hybrid Evaluation Framework for Clinical AI: Integrating Interrater Reliability with the "LLM as a Judge" Methodology | 0 |
| `10.3156/jsoft.37.3_69_1` | LLM as a Judge | 0 |

</details>

### RQ2 — 99 papers (39 indexed, 60 pdf-only)

**Indexed (searchable):**

| DOI | Title | Cites | Chunks |
|---|---|---|---|
| `10.1038/nmeth.3317` | HISAT: a fast spliced aligner with low memory requirements | 22389 | 14 |
| `10.1186/1471-2105-10-421` | BLAST+: architecture and applications | 21004 | 13 |
| `10.1038/nmeth.3337` | Robust enumeration of cell subsets from tissue expression profiles | 11763 | 20 |
| `10.1093/bioinformatics/btt086` | QUAST: quality assessment tool for genome assemblies | 10819 | 9 |
| `10.1109/tnnls.2021.3070843` | A Survey on Knowledge Graphs: Representation, Acquisition, and Applications | 2892 | 55 |
| `10.21437/interspeech.2012-65` | LSTM neural networks for language modeling | 1995 | 7 |
| `10.18653/v1/d18-1259` | HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering | 1763 | 17 |
| `10.1007/s11704-026-60308-3` | A Survey of Large Language Models | 1543 | 73 |
| `10.1609/aaai.v32i1.11325` | Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory | 765 | 16 |
| `10.48550/arxiv.2312.10997` | Retrieval-Augmented Generation for Large Language Models: A Survey | 744 | 57 |
| `10.48550/arxiv.2201.08239` | LaMDA: Language Models for Dialog Applications | 709 | 53 |
| `10.1007/s10462-023-10465-9` | Knowledge Graphs: Opportunities and Challenges | 681 | 34 |
| `10.1186/s41687-018-0061-6` | How do patient reported outcome measures (PROMs) support clinician-patient communication and patient care? A realist synthesis | 631 | 1 |
| `10.18653/v1/2020.acl-main.412` | Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings | 514 | 14 |
| `10.1021/acs.chemrev.3c00189` | Machine Learning Methods for Small Data Challenges in Molecular Science | 497 | 89 |
| `10.18653/v1/n18-1193` | Conversational Memory Network for Emotion Recognition in Dyadic Dialogue Videos | 473 | 17 |
| `10.18653/v1/2023.emnlp-main.495` | Active Retrieval Augmented Generation | 443 | 39 |
| `10.18653/v1/d19-1242` | PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text | 333 | 16 |
| `10.18653/v1/2022.naacl-main.272` | ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction | 332 | 31 |
| `10.18653/v1/2023.acl-long.99` | Precise Zero-Shot Dense Retrieval without Relevance Labels | 307 | 21 |
| `10.18653/v1/2023.acl-long.557` | Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions | 290 | 27 |
| `10.48550/arxiv.1901.08149` | TransferTransfo: A Transfer Learning Approach for Neural Network Based Conversational Agents | 281 | 8 |
| `10.18653/v1/2024.naacl-long.389` | Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity | 214 | 30 |
| `10.1016/j.inffus.2025.103599` | AI Agents vs. Agentic AI: A Conceptual taxonomy, applications and challenges | 202 | 102 |
| `10.18653/v1/2020.findings-emnlp.91` | HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data | 202 | 16 |
| `10.1109/access.2023.3295776` | Information Retrieval: Recent Advances and Beyond | 139 | 66 |
| `10.18653/v1/2022.acl-long.356` | Beyond Goldfish Memory: Long-Term Open-Domain Conversation | 113 | 24 |
| `10.18653/v1/2022.acl-long.396` | Subgraph Retrieval Enhanced Model for Multi-hop Knowledge Base Question Answering | 106 | 17 |
| `10.1038/s41746-025-01475-8` | Large language model agents can use tools to perform clinical calculations | 41 | 22 |
| `10.48550/arxiv.2401.15391` | MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries | 15 | 27 |
| `10.21437/interspeech.2010-97` | Recognition of spontaneous conversational speech using long short-term memory phoneme predictions | 14 | 1 |
| `10.1109/icassp.2011.5947543` | Syllabification of conversational speech using Bidirectional Long-Short-Term Memory Neural Networks | 6 | 1 |
| `10.5220/0014473600004052` | Agent-as-a-Graph: Knowledge Graph-Based Tool and Agent Retrieval for LLM Multi-Agent Systems | 1 | 1 |
| `10.5220/0009892303100317` | Sentiment Polarity Classification of Corporate Review Data with a Bidirectional Long-Short Term Memory (biLSTM) Neural Network Architecture | 1 | 1 |
| `10.1145/3078971.3079028` | Utilising High-Level Features in Summarisation of Academic Presentations | 1 | 1 |
| `10.5220/0013691900003985` | A Long Short-Term Memory (LSTM) Neural Architecture for Presaging Stock Prices | 0 | 1 |
| `10.5220/0013836900004000` | RFG Framework: Retrieval-Feedback-Grounded Multi-Query Expansion | 0 | 1 |
| `10.31274/cc-20251215-154` | CoralX.AI: Multi-Hop, Redundancy-Aware Scientific QA via Hybrid Semantic-Graph Retrieval and RAG | 0 | 1 |
| `10.5220/0013591900004664` | Optimized Medical Data Storage and Query Retrieval Using Cloud Based Multi Indexing | 0 | 1 |

**PDF-only (not yet converted):**

<details><summary>60 papers (click to expand)</summary>

| DOI | Title | Cites |
|---|---|---|
| `10.1186/gb-2009-10-3-r25` | Ultrafast and memory-efficient alignment of short DNA sequences to the human genome | 20060 |
| `10.1038/srep42717` | SwissADME: a free web tool to evaluate pharmacokinetics, drug-likeness and medicinal chemistry friendliness of small molecules | 16413 |
| `10.1017/cbo9780511809071` | Introduction to Information Retrieval | 8652 |
| `10.1109/access.2021.3140175` | A Metaverse: Taxonomy, Components, Applications, and Open Challenges | 1830 |
| `10.18653/v1/2020.coling-main` | Proceedings of the 28th International Conference on Computational Linguistics | 1422 |
| `10.1136/bjsports-2016-096587` | Exercise interventions for cognitive function in adults older than 50: a systematic review with meta-analysis | 1394 |
| `10.18653/v1/2020.emnlp-main` | Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) | 1287 |
| `10.1109/jsac.2021.3126076` | Edge Artificial Intelligence for 6G: Vision, Enabling Technologies, and Applications | 805 |
| `10.1017/atsip.2013.9` | A tutorial survey of architectures, algorithms, and applications for deep learning | 741 |
| `10.1561/1500000066` | Explainable Recommendation: A Survey and New Perspectives | 714 |
| `10.1109/jbhi.2020.2991043` | AI in Medical Imaging Informatics: Current Challenges and Future Directions | 706 |
| `10.1371/journal.pone.0220116` | Helping patients help themselves: A systematic review of self-management support strategies in primary health care practice | 598 |
| `10.48550/arxiv.1909.09586` | Understanding LSTM -- a tutorial into Long Short-Term Memory Recurrent Neural Networks | 503 |
| `10.48550/arxiv.2307.06435` | A Comprehensive Overview of Large Language Models | 369 |
| `10.1038/lsa.2014.3` | Handheld high-throughput plasmonic biosensor using computational on-chip imaging | 356 |
| `10.1109/icassp.2015.7178826` | Constructing long short-term memory based deep recurrent neural networks for large vocabulary speech recognition | 334 |
| `10.48550/arxiv.2309.07864` | The Rise and Potential of Large Language Model Based Agents: A Survey | 259 |
| `10.1371/journal.pdig.0000877` | Retrieval augmented generation for large language models in healthcare: A systematic review | 219 |
| `10.18653/v1/2020.acl-main.91` | Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases | 206 |
| `10.1145/3437963.3441753` | Improving Multi-hop Knowledge Base Question Answering by Learning Intermediate Supervision Signals | 205 |
| `10.70777/si.v2i3.15161` | AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges | 171 |
| `10.18653/v1/2020.emnlp-main.710` | Hierarchical Graph Network for Multi-hop Question Answering | 161 |
| `10.1038/s43018-025-00991-6` | Development and validation of an autonomous artificial intelligence agent for clinical decision-making in oncology | 150 |
| `10.18653/v1/2025.findings-emnlp.568` | LightRAG: Simple and Fast Retrieval-Augmented Generation | 146 |
| `10.1101/2025.05.30.656746` | Biomni: A General-Purpose Biomedical AI Agent | 118 |
| `10.48550/arxiv.2210.02406` | Decomposed Prompting: A Modular Approach for Solving Complex Tasks | 100 |
| `10.48550/arxiv.2308.07107` | Large Language Models for Information Retrieval: A Survey | 97 |
| `10.18653/v1/2024.findings-acl.624` | InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents | 94 |
| `10.48550/arxiv.1705.01509` | Neural Models for Information Retrieval | 86 |
| `10.1002/aris.2008.1440420109` | Interactive information retrieval | 85 |
| `10.48550/arxiv.2402.19473` | Retrieval-Augmented Generation for AI-Generated Content: A Survey | 83 |
| `10.18653/v1/2024.acl-long.737` | CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges | 81 |
| `10.18653/v1/2020.emnlp-main.296` | Coarse-to-Fine Query Focused Multi-Document Summarization | 77 |
| `10.48550/arxiv.2307.16789` | Counterfactually Auditable Lifecycle Certification for Autonomous Agents | 72 |
| `10.18653/v1/2024.acl-long.747` | Evaluating Very Long-Term Conversational Memory of LLM Agents | 70 |
| `10.1145/1067268.1067296` | Using generative probabilistic models for multimedia retrieval | 70 |
| `10.48550/arxiv.1910.02610` | Multi-hop Question Answering via Reasoning Chains | 66 |
| `10.1007/s41019-025-00296-9` | LLM-Based Agents for Tool Learning: A Survey | 45 |
| `10.18653/v1/2024.acl-long.397` | Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering | 30 |
| `10.1007/s10506-023-09354-x` | Semantic matching based legal information retrieval system for COVID-19 pandemic | 28 |
| `10.1108/jd-11-2017-0154` | Pioneering models for information interaction in the context of information seeking and retrieval | 23 |
| `10.18653/v1/2025.findings-acl.97` | HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-Augmented Generation | 22 |
| `10.48550/arxiv.2404.00610` | RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation | 17 |
| `10.1109/wacv.2014.6836025` | Summarisation of short-term and long-term videos using texture and colour | 10 |
| `10.1007/s10791-006-9017-1` | Learning-based summarisation of XML documents | 6 |
| `10.18653/v1/2024.sighan-1.18` | PerLTQA: A Personal Long-Term Memory Dataset for Memory Classification, Retrieval, and Fusion in Question Answering | 6 |
| `10.1145/1076034.1076162` | Using query term order for result summarisation | 3 |
| `10.18653/v1/2026.findings-acl.1690` | HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents | 0 |
| `10.18653/v1/2026.acl-long.749` | APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI | 0 |
| `10.18653/v1/2026.findings-acl.2090` | Beyond Single-Shot: Multi-step Tool Retrieval via Query Planning | 0 |
| `10.54209/jatilima.v7i07.2716` | Gated, grounded, and governed: integrating large language models, authorised tool calling, and retrieval-augmented generation into NeoSiakad.com, a multi-tenant academic information SaaS | 0 |
| `10.66245/jyi.v1.i1.002` | The Impact of Query Decomposition and Cross-Encoder Reranking in Multi-Hop Retrieval-Augmented Generation | 0 |
| `10.21203/rs.3.rs-10725059/v1` | Beyond the Largest Gap: Multi-Boundary Ranked-List Truncation for Multi-Hop Retrieval | 0 |
| `10.18653/v1/2026.eacl-long.5` | GRITHopper: Decomposition-Free Multi-Hop Dense Retrieval | 0 |
| `10.21275/sr221231230330` | Enhancing Fashion Image Retrieval with Multi-Modal Query and Zero-Shot Learning for Cross-Domain | 0 |
| `10.21203/rs.3.rs-9148928/v1` | Controlled Multi-Hop RAG: A Deterministic Parallel Pipeline Architecture as an Alternative to Agentic Retrieval | 0 |
| `10.69987/jacs.2023.30802` | Controllable Long-Term User Memory for Multi-Session Dialogue: Confidence-Gated Writing, Time-Aware Retrieval-Augmented Generation, and Update/Forgetting | 0 |
| `10.7190/shu-thesis-00719` | Synthesising Summaries: A novel Retrieval-Augmented Generation-based pipeline for multi-document summarisation | 0 |
| `10.5121/csit.2022.120907` | Comparing Methods for Extractive Summarisation of Call Centre Dialogue | 0 |
| `10.2172/7352030` | Atlantic Richfield Hanford Company quarterly report, technology development for long-term management of Hanford high-level waste, October 1975 through December 1975. [Storage system; retrieval; immobilization; contaminated equipment] | 0 |

</details>

### RQ3 — 35 papers (7 indexed, 28 pdf-only)

**Indexed (searchable):**

| DOI | Title | Cites | Chunks |
|---|---|---|---|
| `10.18653/v1/2021.findings-emnlp.320` | Retrieval Augmentation Reduces Hallucination in Conversation | 533 | 37 |
| `10.1609/aaai.v38i16.29728` | Benchmarking Large Language Models in Retrieval-Augmented Generation | 374 | 18 |
| `10.18653/v1/2023.emnlp-main.322` | Query Rewriting in Retrieval-Augmented Large Language Models | 257 | 27 |
| `10.18653/v1/w16-0104` | Open-domain Factoid Question Answering via Knowledge Graph Search | 15 | 3 |
| `10.48550/arxiv.2410.10813` | LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory | 3 | 36 |
| `10.48550/arxiv.2402.17753` | A controlled embedder swap on LoCoMo, and three arms that could not carry a comparison | 3 | 29 |
| `10.18653/v1/2026.eacl-long.15` | H-MEM: Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents | 2 | 15 |

**PDF-only (not yet converted):**

<details><summary>28 papers (click to expand)</summary>

| DOI | Title | Cites |
|---|---|---|
| `10.1158/2159-8290.cd-12-0095` | The cBio Cancer Genomics Portal: An Open Platform for Exploring Multidimensional Cancer Genomics Data | 15341 |
| `10.48550/arxiv.1910.10683` | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | 8346 |
| `10.3847/1538-3881/aabc4f` | The Astropy Project: Building an Open-science Project and Status of the v2.0 Core Package | 7369 |
| `10.1007/s12525-021-00475-2` | Machine learning and deep learning | 2622 |
| `10.1186/s40537-021-00492-0` | Text Data Augmentation for Deep Learning | 1706 |
| `10.1109/icassp49357.2023.10095969` | Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation | 423 |
| `10.3389/fpsyg.2013.00440` | The role of locomotion in psychological development | 231 |
| `10.18653/v1/2021.acl-long.316` | Generation-Augmented Retrieval for Open-Domain Question Answering | 146 |
| `10.18653/v1/2023.findings-emnlp.691` | Self-Knowledge Guided Retrieval Augmentation for Large Language Models | 59 |
| `10.1371/journal.pbio.1002123` | Convergent Evolution of Mechanically Optimal Locomotion in Aquatic Invertebrates and Vertebrates | 50 |
| `10.1007/s00221-021-06049-0` | Perceptual-motor styles | 46 |
| `10.3389/fpsyg.2013.00273` | Choosing Actions | 32 |
| `10.18653/v1/2026.acl-long.583` | Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning | 5 |
| `10.18653/v1/2025.findings-acl.1014` | Evaluating the Long-Term Memory of Large Language Models | 4 |
| `10.31219/osf.io/srgpx` | Large Language Models with Knowledge Domain Partitioning for Specialized Domain Knowledge Concentration | 4 |
| `10.18653/v1/2024.findings-emnlp.794` | LLMs as Collaborator: Demands-Guided Collaborative Retrieval-Augmented Generation for Commonsense Knowledge-Grounded Open-Domain Dialogue Systems | 4 |
| `10.18653/v1/2025.findings-acl.989` | MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents | 3 |
| `10.18653/v1/2025.findings-acl.972` | TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues | 3 |
| `10.18653/v1/2026.acl-long.1709` | MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents | 2 |
| `10.48550/arxiv.2504.15965` | From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs | 2 |
| `10.48550/arxiv.2502.05589` | On Memory Construction and Retrieval for Personalized Conversational Agents | 2 |
| `10.48550/arxiv.2406.00057` | Toward Conversational Agents with Context and Time Sensitive Long-term Memory | 2 |
| `10.18653/v1/2025.findings-emnlp.1204` | Pre-Storage Reasoning for Episodic Memory: Shifting Inference Burden to Memory for Personalized Dialogue | 2 |
| `10.48550/arxiv.2412.15266` | On the Structural Memory of LLM Agents | 2 |
| `10.18653/v1/2023.dialdoc-1.2` | MoQA: Benchmarking Multi-Type Open-Domain Question Answering | 2 |
| `10.1145/3774904.3792089` | HingeMem: Boundary Guided Long-Term Memory with Query Adaptive Retrieval for Scalable Dialogues | 1 |
| `10.18653/v1/2026.acl-long.625` | HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents | 1 |
| `10.18653/v1/2021.findings-emnlp.286` | Distilling the Knowledge of Large-scale Generative Models into Retrieval Models for Efficient Open-domain Conversation | 1 |

</details>

### RQ4 — 68 papers (12 indexed, 56 pdf-only)

**Indexed (searchable):**

| DOI | Title | Cites | Chunks |
|---|---|---|---|
| `10.1073/pnas.0506580102` | Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles | 49495 | 19 |
| `10.1613/jair.301` | Reinforcement Learning: A Survey | 8950 | 1 |
| `10.1371/journal.pmed.1001349` | The Long-Term Health Consequences of Child Physical Abuse, Emotional Abuse, and Neglect: A Systematic Review and Meta-Analysis | 3403 | 82 |
| `10.15485/1464240` | Inventory of U.S. Greenhouse Gas Emissions and Sinks | 2872 | 1 |
| `10.1109/event.2001.938869` | Content-based video retrieval by integrating spatio-temporal and stochastic recognition of events | 64 | 3 |
| `10.1007/s11518-023-5561-0` | Narrative Graph: Telling Evolving Stories Based on Event-centric Temporal Knowledge Graph | 15 | 6 |
| `10.26599/tst.2024.9010119` | Enhancing Temporal Knowledge Graph for Future Event Prediction with Long-Term Dense Graph | 6 | 3 |
| `10.5220/0005178802770284` | A Probabilistic Doxastic Temporal Logic for Reasoning about Beliefs in Multi-agent Systems | 3 | 1 |
| `10.5220/0012178200003598` | Mechanical Fault Prediction Based on Event Knowledge Graph | 1 | 1 |
| `10.5220/0010652300003064` | Conversation Extraction from Event Logs | 1 | 1 |
| `10.3758/bf03197517` | Single-trial free recall from temporal search sets in long-term memory | 1 | 3 |
| `10.59350/waswj-nma51` | Harnessing Temporal Dynamics: Advanced Reasoning using Temporal Knowledge Graphs | 0 | 6 |

**PDF-only (not yet converted):**

<details><summary>56 papers (click to expand)</summary>

| DOI | Title | Cites |
|---|---|---|
| `10.1186/s40537-021-00444-8` | Review of deep learning: concepts, CNN architectures, challenges, applications, future directions | 7896 |
| `10.1109/tmi.2014.2377694` | The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS) | 6818 |
| `10.1186/1745-6215-8-16` | Practical methods for incorporating summary time-to-event data into meta-analysis | 5143 |
| `10.1145/182.358434` | Maintaining knowledge about temporal intervals | 4850 |
| `10.1016/j.ophtha.2016.01.006` | Global Prevalence of Myopia and High Myopia and Temporal Trends from 2000 through 2050 | 4842 |
| `10.1109/tac.2007.904277` | Event-Triggered Real-Time Scheduling of Stabilizing Control Tasks | 4406 |
| `10.1088/0004-637x/697/2/1071` | THE LARGE AREA TELESCOPE ON THE FERMI GAMMA-RAY SPACE TELESCOPE MISSION | 4168 |
| `10.1186/s12864-018-4772-0` | Slingshot: cell lineage and pseudotime inference for single-cell transcriptomics | 3529 |
| `10.1146/annurev-fluid-010719-060214` | Machine Learning for Fluid Mechanics | 2826 |
| `10.1017/s0140525x0999094x` | The myth of language universals: Language diversity and its importance for cognitive science | 2661 |
| `10.1109/jproc.2019.2918951` | Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing | 2380 |
| `10.3389/fncel.2019.00363` | Brain-Derived Neurotrophic Factor: A Key Molecule for Memory in the Healthy and the Pathological Brain | 1439 |
| `10.1088/1538-3873/aae8ac` | The Zwicky Transient Facility: Data Processing, Products, and Archive | 1251 |
| `10.18653/v1/p19-1470` | COMET: Commonsense Transformers for Automatic Knowledge Graph Construction | 889 |
| `10.1609/aaai.v33i01.33013656` | Spatiotemporal Multi-Graph Convolution Network for Ride-Hailing Demand Forecasting | 872 |
| `10.1371/journal.pone.0026752` | Temporal Patterns of Happiness and Information in a Global Social Network: Hedonometrics and Twitter | 834 |
| `10.18653/v1/d18-1516` | Learning Sequence Encoders for Temporal Knowledge Graph Completion | 569 |
| `10.1038/s41597-023-01960-3` | Building a knowledge graph to enable precision medicine | 507 |
| `10.14506/ca30.3.02` | Attuning to the Chemosphere: Domestic Formaldehyde, Bodily Reasoning, and the Chemical Sublime | 506 |
| `10.18653/v1/d18-1225` | HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding | 495 |
| `10.1109/access.2013.2260814` | Information Forensics: An Overview of the First Decade | 394 |
| `10.1609/aaai.v32i1.12039` | Graph Convolutional Networks With Argument-Aware Pooling for Event Detection | 393 |
| `10.1609/aaai.v34i04.5815` | Diachronic Embedding for Temporal Knowledge Graph Completion | 385 |
| `10.1109/access.2020.3030076` | Knowledge Graph Completion: A Review | 346 |
| `10.1037/0033-295x.114.1.38` | The simultaneous type, serial token model of temporal attention and working memory. | 311 |
| `10.1145/2528412` | A catalog of stream processing optimizations | 297 |
| `10.1109/comst.2023.3323344` | Toward Autonomous Multi-UAV Wireless Network: A Survey of Reinforcement Learning-Based Approaches | 283 |
| `10.23919/jcin.2021.9663101` | What is Semantic Communication? A View on Conveying Meaning in the Era of Machine Intelligence | 250 |
| `10.1109/access.2020.2973928` | Named Entity Extraction for Knowledge Graphs: A Literature Overview | 232 |
| `10.1145/3377455` | Blocking and Filtering Techniques for Entity Resolution | 201 |
| `10.14722/ndss.2017.23271` | ASLR on the Line: Practical Cache Attacks on the MMU | 198 |
| `10.1609/aaai.v36i4.20330` | TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs | 167 |
| `10.1609/aaai.v24i1.7512` | Temporal Information Extraction | 158 |
| `10.18653/v1/2020.emnlp-main.462` | TeMP: Temporal Message Passing for Temporal Knowledge Graph Completion | 148 |
| `10.3389/fnins.2015.00137` | On event-based optical flow detection | 104 |
| `10.1109/tkde.2016.2592527` | Scalable Daily Human Behavioral Pattern Mining from Multivariate Temporal Data | 99 |
| `10.18653/v1/d17-1092` | Temporal Information Extraction for Question Answering Using Syntactic Dependencies in an LSTM-based Architecture | 65 |
| `10.1109/access.2022.3168976` | Systematic Literature Review of Security Event Correlation Methods | 61 |
| `10.1162/tacl_a_00058` | Domain-Targeted, High Precision Knowledge Extraction | 54 |
| `10.26599/tst.2020.9010063` | Event temporal relation extraction with attention mechanism and graph neural network | 47 |
| `10.1007/s10579-021-09562-4` | SENTiVENT: enabling supervised information extraction of company-specific events in economic and financial news | 38 |
| `10.3758/bf03198542` | Long-term memory for temporal structure: | 28 |
| `10.18653/v1/2021.emnlp-main.815` | Utilizing Relative Event Time to Enhance Event-Event Temporal Relation Extraction | 18 |
| `10.1007/s41060-023-00428-2` | Graph-based feature extraction on object-centric event logs | 15 |
| `10.1109/bigdata50022.2020.9378471` | Knowledge Graph Enhanced Event Extraction in Financial Documents | 14 |
| `10.1613/jair.5431` | Resolving Over-Constrained Temporal Problems with Uncertainty through Conflict-Directed Relaxation | 12 |
| `10.3758/bf03195943` | The role of reminding in long-term memory for temporal order | 10 |
| `10.18653/v1/2023.findings-acl.490` | History repeats: Overcoming catastrophic forgetting for event-centric temporal knowledge graph completion | 3 |
| `10.14257/ijca.2013.6.6.06` | A New Spatio-temporal Event Model based on Multi-tuple for Cyber-Physical Systems | 1 |
| `10.3837/tiis.2015.01.010` | Anomalous Event Detection in Traffic Video Based on Sequential Temporal Patterns of Spatial Interval Events | 1 |
| `10.18653/v1/2024.findings-emnlp.47` | Temporal Cognitive Tree: A Hierarchical Modeling Approach for Event Temporal Relation Extraction | 1 |
| `10.5120/11974-7836` | Heuristic Event Filtering Methodology for Interval based Temporal Semantics | 0 |
| `10.21203/rs.3.rs-5802602/v1` | Document-level causal event extraction enhanced by temporal relation using dual-channel neural network | 0 |
| `10.18653/v1/2023.matching-1.3` | Toward Consistent and Informative Event-Event Temporal Relation Extraction | 0 |
| `10.15623/ijret.2016.0516011` | SEQUENTIAL TEMPORAL PATTERN MINING IN TIME-INTERVAL BASED EVENT DATA | 0 |
| `10.65286/icic.v20i2.13338` | Syntax-aware Event Temporal Relation Extraction Using Constraint Graph | 0 |

</details>

### RQ5 — 37 papers (6 indexed, 31 pdf-only)

**Indexed (searchable):**

| DOI | Title | Cites | Chunks |
|---|---|---|---|
| `10.1177/0269881116636545` | Evidence-based guidelines for treating bipolar disorder: Revised third edition recommendations from the British Association for Psychopharmacology | 1317 | 118 |
| `10.4103/0253-7176.155605` | Recovery Model of Mental Illness: A Complementary Approach to Psychiatric Care | 302 | 6 |
| `10.1177/1460458215593329` | Designing a spoken dialogue interface to an intelligent cognitive assistant for people with dementia | 85 | 4 |
| `10.18653/v1/2024.findings-emnlp.969` | Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization | 78 | 27 |
| `10.18280/ria.380417` | PRMNBR: Personalized Recommendation Model for Next Basket Recommendation Using User’s Long-Term Preference, Short-Term Preference, and Repetition Behaviour | 0 | 14 |
| `10.15581/011.91.016` | Laicidad: en diálogo con Francesco D’Agostino | 0 | 1 |

**PDF-only (not yet converted):**

<details><summary>31 papers (click to expand)</summary>

| DOI | Title | Cites |
|---|---|---|
| `10.1086/304888` | A Universal Density Profile from Hierarchical Clustering | 9019 |
| `10.1007/s10796-017-9810-y` | Advances in Social Media Research: Past, Present and Future | 1285 |
| `10.1023/a:1007369909943` | Learning and Revising User Profiles: The Identification of Interesting Web Sites | 1196 |
| `10.18653/v1/d15-1199` | Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems | 848 |
| `10.1140/epja/s10050-020-00141-9` | The joint evaluated fission and fusion nuclear data library, JEFF-3.3 | 710 |
| `10.1038/s41746-020-0288-5` | Sex and gender differences and biases in artificial intelligence for biomedicine and healthcare | 546 |
| `10.5334/csci.140` | Mapping Internet Celebrity on TikTok: Exploring Attention Economies and Visibility Labours | 545 |
| `10.1007/s11280-024-01276-1` | When large language models meet personalization: perspectives of challenges and opportunities | 335 |
| `10.1186/s40537-019-0219-y` | Effectiveness analysis of machine learning classification models for predicting personalized context-aware smartphone usage | 304 |
| `10.1007/s10648-020-09570-w` | Developing Personalized Education: A Dynamic Framework | 268 |
| `10.1007/s10551-019-04371-w` | Mapping the Ethicality of Algorithmic Pricing: A Review of Dynamic and Personalized Pricing | 194 |
| `10.1609/aaai.v38i17.29946` | MemoryBank: Enhancing Large Language Models with Long-Term Memory | 189 |
| `10.1109/access.2019.2944243` | A Survey of User Profiling: State-of-the-Art, Challenges, and Solutions | 159 |
| `10.1007/978-3-030-58948-6_2` | Personalized and Adaptive Learning | 158 |
| `10.48550/arxiv.1508.01745` | Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems | 126 |
| `10.1371/journal.pmed.0050234` | Health and Human Rights Concerns of Drug Users in Detention in Guangxi Province, China | 107 |
| `10.1609/aaai.v32i1.11938` | Personalizing a Dialogue System With Transfer Reinforcement Learning | 100 |
| `10.1007/s11257-011-9116-6` | Designing interfaces for explicit preference elicitation: a user-centered investigation of preference representation and elicitation process | 94 |
| `10.1609/aaai.v35i12.17287` | Personalized Adaptive Meta Learning for Cold-start User Preference Prediction | 59 |
| `10.18653/v1/d18-1284` | Learning Personas from Dialogue with Attentive Memory Networks | 32 |
| `10.18653/v1/2023.acl-long.544` | PAED: Zero-Shot Persona Attribute Extraction in Dialogues | 31 |
| `10.1609/aaai.v34i05.6503` | Learning Long- and Short-Term User Literal-Preference with Multimodal Hierarchical Transformer Network for Personalized Image Caption | 27 |
| `10.18653/v1/2025.naacl-long.272` | Hello Again! LLM-powered Personalized Agent for Long-term Dialogue | 16 |
| `10.5120/2903-3808` | Personalisation of User Profile: Creating User Profile Ontology for Tamilnadu Tourism | 7 |
| `10.12700/aph.12.8.2015.8.2` | User Preference Modeling by Global and Individual Weights for Personalized | 2 |
| `10.21203/rs.3.rs-755856/v1` | A User Preference Tree based Personalized Route Recommendation System for Constraint Tourism and Travel | 2 |
| `10.5391/ijfis.2012.12.4.270` | A Dynamic Ontology-based Multi-Agent Context-Awareness User Profile Construction Method for Personalized Information Retrieval | 2 |
| `10.1007/978-0-387-35175-9_83` | Where to locate user profiles of personalized applications? — A user profile management agent — | 0 |
| `10.9708/jksci.2015.20.1.029` | A Multi-Agent MicroBlog Behavior based User Preference Profile Construction Approach | 0 |
| `10.1007/978-981-92-3520-9_15` | EMP: Integrating Emotion Reasoning, Memory Structuring, and Persona Refinement for Long-Term Personalized Dialogue Generation | 0 |
| `10.22459/csy.2022.01a` | Controversial High-Profile Detention and Prosecution of Foreigners | 0 |

</details>

### RQ6 — 46 papers (6 indexed, 40 pdf-only)

**Indexed (searchable):**

| DOI | Title | Cites | Chunks |
|---|---|---|---|
| `10.5194/hess-22-6005-2018` | Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks | 1975 | 34 |
| `10.1613/jair.1129` | PDDL2.1: An Extension to PDDL for Expressing Temporal Planning Domains | 1739 | 68 |
| `10.1038/npp.2009.126` | The Episodic Memory System: Neurocircuitry and Disorders | 676 | 13 |
| `10.1038/npp.2010.169` | Update on Memory Systems and Processes | 217 | 17 |
| `10.1016/s0169-023x(02)00207-0` | A formal model for temporal schema versioning in object-oriented databases | 21 | 1 |
| `10.5220/0008068101150126` | Memory Nets: Knowledge Representation for Intelligent Agent Operations in Real World | 3 | 1 |

**PDF-only (not yet converted):**

<details><summary>40 papers (click to expand)</summary>

| DOI | Title | Cites |
|---|---|---|
| `10.1017/s0140525x01003922` | The magical number 4 in short-term memory: A reconsideration of mental storage capacity | 6910 |
| `10.1371/journal.pone.0169748` | SoilGrids250m: Global gridded soil information based on machine learning | 4843 |
| `10.1109/access.2019.2912200` | Review of Deep Learning Algorithms and Architectures | 1883 |
| `10.1613/jair.530` | AntNet: Distributed Stigmergetic Control for Communications Networks | 1585 |
| `10.3389/fnins.2018.00331` | Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks | 1211 |
| `10.1109/tnnls.2022.3160699` | Towards Personalized Federated Learning | 1142 |
| `10.1289/ehp.96104s4715` | Research needs for the risk assessment of health and environmental effects of endocrine disruptors: a report of the U.S. EPA-sponsored workshop. | 1137 |
| `10.1109/access.2014.2332453` | Toward Scalable Systems for Big Data Analytics: A Technology Tutorial | 1103 |
| `10.3758/cabn.1.2.137` | Interactions between frontal cortex and basal ganglia in working memory: A computational model | 923 |
| `10.3389/fpsyg.2011.00255` | The Spatial and Temporal Signatures of Word Production Components: A Critical Update | 781 |
| `10.3758/s13423-016-1191-6` | The many faces of working memory and short-term storage | 629 |
| `10.3389/frai.2021.654924` | Are We There Yet? - A Systematic Literature Review on Chatbots in Education | 537 |
| `10.1007/s10462-018-9646-y` | 40 years of cognitive architectures: core cognitive abilities and practical applications | 532 |
| `10.1038/npjscilearn.2016.11` | Learning and memory under stress: implications for the classroom | 491 |
| `10.35833/mpce.2021.000058` | A Review of Graph Neural Networks and Their Applications in Power Systems | 404 |
| `10.1007/s41019-020-00151-z` | A Survey of Traffic Prediction: from Spatio-Temporal Data to Intelligent Transportation | 387 |
| `10.3389/fnbeh.2013.00139` | The Influence of Prior Knowledge on Memory: A Developmental Cognitive Neuroscience Perspective | 290 |
| `10.1109/access.2023.3275789` | Graph Neural Networks for Intrusion Detection: A Survey | 220 |
| `10.18653/v1/2021.acl-long.365` | Search from History and Reason for Future: Two-stage Reasoning on Temporal Knowledge Graphs | 103 |
| `10.18653/v1/2024.naacl-long.219` | Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey | 103 |
| `10.48550/arxiv.2306.08302` | Unifying Large Language Models and Knowledge Graphs: A Roadmap | 101 |
| `10.3389/fnbeh.2013.00003` | Making the case that episodic recollection is attributable to operations occurring at retrieval rather than to content stored in a dedicated subsystem of long-term memory | 94 |
| `10.18653/v1/2024.emnlp-main.486` | Knowledge Conflicts for LLMs: A Survey | 72 |
| `10.1007/s00426-020-01417-x` | Virtual reality experiences promote autobiographical retrieval mechanisms: Electrophysiological correlates of laboratory and virtual experiences | 65 |
| `10.48550/arxiv.2212.14024` | Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP | 53 |
| `10.1007/s00521-025-11666-9` | A survey on retrieval-augmentation generation (RAG) models for healthcare applications | 47 |
| `10.3758/s13415-012-0096-8` | Neural correlates of metacognitive monitoring during episodic and semantic retrieval | 43 |
| `10.3389/fpubh.2025.1635381` | MEGA-RAG: a retrieval-augmented generation framework with multi-evidence guided answer refinement for mitigating hallucinations of LLMs in public health | 39 |
| `10.48550/arxiv.2211.02405` | Explainable Information Retrieval: A Survey | 35 |
| `10.3389/fnbeh.2013.00114` | Retrieval of Recent Autobiographical Memories is Associated with Slow-Wave Sleep in Early AD | 35 |
| `10.26481/dis.20000914rp` | Knowledge-based query formulation in information retrieval | 31 |
| `10.48550/arxiv.2501.00309` | Retrieval-Augmented Generation with Graphs (GraphRAG) | 28 |
| `10.24963/ijcai.2023/232` | Adaptive Path-Memory Network for Temporal Knowledge Graph Reasoning | 24 |
| `10.18653/v1/2025.findings-naacl.334` | Time-aware ReAct Agent for Temporal Knowledge Graph Question Answering | 3 |
| `10.21203/rs.3.rs-3144279/v1` | TKMBR: Temporal Knowledge Graph-based Multi-Behavior Recommendation for E-commerce | 0 |
| `10.18653/v1/2026.findings-acl.1587` | EvoMemKG: An Evolvable Memory Agent for Multi-hop Knowledge Graph Reasoning | 0 |
| `10.31224/5956` | KR-VLM: Enhancing Factual Reasoning in Vision-Language Models via Knowledge Retrieval and Self-Verification | 0 |
| `10.18653/v1/2026.acl-long.760` | LOKA: Conflict-Aware LLM Knowledge Update with Adaptive Knowledge Memory | 0 |
| `10.15406/iratj.2018.04.00088` | Ontology Optimization Based on Temporal Versioning | 0 |
| `10.1101/2022.12.02.518876` | Rhythmic temporal coordination of neural activity prevents representational conflict during working memory | 0 |

</details>

### RQ7 — 21 papers (6 indexed, 15 pdf-only)

**Indexed (searchable):**

| DOI | Title | Cites | Chunks |
|---|---|---|---|
| `10.1145/3578938` | Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Faster, and Better | 566 | 49 |
| `10.1561/1500000055` | A Survey of Query Auto Completion in Information Retrieval | 140 | 1 |
| `10.1371/journal.pone.0224934` | An analytical model to minimize the latency in healthcare internet-of-things in fog computing environment | 116 | 25 |
| `10.32920/25536193` | Method and apparatus for accelerating retrieval of data from a memory system with cache by reducing latency | 0 | 1 |
| `10.32920/25536193.v1` | Method and apparatus for accelerating retrieval of data from a memory system with cache by reducing latency | 0 | 1 |
| `10.63345/jqst.v1i1.27` | Energy-Aware Caching Strategies for Faster Data Retrieval in Low-Latency Pipelines | 0 | 9 |

**PDF-only (not yet converted):**

<details><summary>15 papers (click to expand)</summary>

| DOI | Title | Cites |
|---|---|---|
| `10.3389/fnins.2014.00150` | The speed-accuracy tradeoff: history, physiology, methodology, and behavior | 988 |
| `10.1109/access.2020.3031549` | Contrastive Representation Learning: A Framework and Review | 859 |
| `10.1109/comst.2018.2849509` | Survey on Multi-Access Edge Computing for Internet of Things Realization | 802 |
| `10.1109/tbme.2014.2309951` | Unobtrusive Sensing and Wearable Devices for Health Informatics | 727 |
| `10.1109/comst.2019.2933899` | A Survey on Security and Privacy of 5G Technologies: Potential Solutions, Recent Advancements, and Future Directions | 578 |
| `10.1371/journal.pone.0075992` | Ligand Pose and Orientational Sampling in Molecular Docking | 239 |
| `10.18653/v1/2021.repl4nlp-1.17` | In-Batch Negatives for Knowledge Distillation with Tightly-Coupled Teachers for Dense Retrieval | 124 |
| `10.5121/iju.2012.3202` | Content Based Video Retrieval Systems | 103 |
| `10.1007/s12599-025-00945-3` | Retrieval-Augmented Generation (RAG) | 72 |
| `10.1561/1500000071` | Efficient and Effective Tree-based and Neural Learning to Rank | 19 |
| `10.48550/arxiv.2504.09984` | On Precomputation and Caching in Information Retrieval Experiments with Pipeline Architectures | 1 |
| `10.48550/arxiv.2502.01960` | MPIC: Position-Independent Multimodal Context Caching System for Efficient MLLM Serving | 1 |
| `10.48550/arxiv.2012.11685` | Neural Methods for Effective, Efficient, and Exposure-Aware Information Retrieval | 1 |
| `10.70382/ajsitr.v12i9.088` | Contamination-Free LLM Routing on LiveBench Reasoning Tasks: Accuracy-Cost-Latency Tradeoff Learning | 0 |
| `10.56726/irjmets80949` | OPTIMIZING LATENCY AND ACCURACY TRADE-OFFS IN LARGE-SCALE RETRIEVAL-AUGMENTED GENERATION PIPELINES | 0 |

</details>

### RQ8 — 37 papers (5 indexed, 32 pdf-only)

**Indexed (searchable):**

| DOI | Title | Cites | Chunks |
|---|---|---|---|
| `10.1609/icwsm.v8i1.14550` | VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text | 5984 | 35 |
| `10.48550/arxiv.2107.03374` | Evaluating Large Language Models Trained on Code | 1465 | 67 |
| `10.18653/v1/2024.eacl-demo.16` | RAGAs: Automated Evaluation of Retrieval Augmented Generation | 456 | 25 |
| `10.5220/0009889302250237` | Prov-Trust: Towards a Trustworthy SGX-based Data Provenance System | 7 | 1 |
| `10.1145/1600193.1600224` | Geometric consistency checking for local-descriptor based document retrieval | 2 | 1 |

**PDF-only (not yet converted):**

<details><summary>32 papers (click to expand)</summary>

| DOI | Title | Cites |
|---|---|---|
| `10.1016/j.physd.2019.132306` | Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) network | 5037 |
| `10.3758/bf03196772` | Working memory span tasks: A methodological review and user’s guide | 2913 |
| `10.1371/journal.pone.0180944` | A deep learning framework for financial time series using stacked autoencoders and long-short term memory | 1133 |
| `10.3758/s13423-010-0034-0` | Does working memory training work? The promise and challenges of enhancing cognition by training working memory | 737 |
| `10.48550/arxiv.1704.00656` | Detection and Resolution of Rumours in Social Media: A Survey | 726 |
| `10.1006/cogp.1996.0011` | Templates in Chess Memory: A Mechanism for Recalling Several Boards | 625 |
| `10.1088/1748-9326/ab1b7d` | How can Big Data and machine learning benefit environment and water management: a survey of methods, applications, and future directions | 552 |
| `10.3758/bf03214334` | The effect of orthographic similarity on lexical retrieval: Resolving neighborhood conflicts | 537 |
| `10.1109/tkde.2015.2427795` | In-Memory Big Data Management and Processing: A Survey | 425 |
| `10.3758/bf03197611` | Memory metaphors in cognitive psychology | 389 |
| `10.14722/ndss.2020.24046` | Unicorn: Runtime Provenance-Based Detector for Advanced Persistent Threats | 314 |
| `10.3758/bf03213342` | Retrieval inhibition from part-set cuing: A persisting enigma in memory research | 254 |
| `10.1007/s11432-024-4337-1` | Overview of AI and communication for 6G network: fundamentals, challenges, and future research opportunities | 198 |
| `10.1038/s41591-024-03445-1` | Medical large language models are vulnerable to data-poisoning attacks | 185 |
| `10.1145/1807167.1807234` | Efficient querying and maintenance of network provenance at internet-scale | 134 |
| `10.3758/bf03196449` | Probability judgment and subadditivity: The role of working memory capacity and constraining retrieval | 102 |
| `10.1007/s11280-019-00746-1` | A survey on data provenance in IoT | 70 |
| `10.48550/arxiv.2209.02299` | A Survey of Machine Unlearning | 68 |
| `10.3390/s25061666` | Generative AI and LLMs for Critical Infrastructure Protection: Evaluation Benchmarks, Agentic AI, Challenges, and Opportunities | 61 |
| `10.1007/s10462-025-11248-0` | Deep learning model inversion attacks and defenses: a comprehensive survey | 54 |
| `10.48550/arxiv.2401.05459` | Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security | 31 |
| `10.1186/1471-2105-12-461` | A unified framework for managing provenance information in translational research | 24 |
| `10.1109/access.2023.3280928` | Security-Aware Provenance for Transparency in IoT Data Propagation | 16 |
| `10.3233/jcs-130487` | A core calculus for provenance | 14 |
| `10.3389/frai.2026.1737532` | An auditable and source-verified framework for clinical AI decision support: integrating retrieval-augmented generation with data provenance | 11 |
| `10.48550/arxiv.2407.12784` | AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases | 10 |
| `10.48550/arxiv.2410.02644` | Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents | 6 |
| `10.1109/time.2015.18` | Dynamic Consistency of Conditional Simple Temporal Networks via Mean Payoff Games: A Singly-Exponential Time DC-checking | 5 |
| `10.20935/acadai8122` | Retrieval-augmented generation for natural language art provenance searches in the Getty Provenance Index | 1 |
| `10.31144/si.2307-6410.2023.n22.p1-10` | Static Memory Consistency Constraints Checking | 0 |
| `10.1109/time.2016.16` | Instantaneous Reaction-Time in Dynamic-Consistency Checking of Conditional Simple Temporal Networks | 0 |
| `10.24963/ijcai.2023/212` | A Fast Algorithm for Consistency Checking Partially Ordered Time | 0 |

</details>

## Set C — Anna's Archive paywalled acquisitions (27)

Fetched manually because paywalled (MCP `paper_download` cannot reach them). The pipeline that
produced them is `tmp/annas/fetch_annas.py` and its log `tmp/annas/results.jsonl`; this table is
the canonical roll-up.

| DOI | Title | Cites | Status | Pages | Chunks |
|---|---|---|---|---|---|
| `10.1016/j.artint.2012.06.001` | YAGO2: A spatially and temporally enhanced knowledge base from Wikipedia | 1,257 | indexed | 34 | 47 |
| `10.1016/j.jbi.2013.08.010` | Extraction of events and temporal expressions from clinical narratives | 51 | indexed | 7 | 16 |
| `10.1016/j.jbi.2013.09.007` | TEMPTING system: A hybrid method of rule and machine learning for temporal relation extraction in patient discharge summaries | 45 | indexed | 9 | 17 |
| `10.1016/j.jbusres.2016.08.001` | Critical analysis of Big Data challenges and analytical methods | 2,056 | indexed | 24 | 46 |
| `10.1016/j.jksuci.2016.10.003` | A survey on Internet of Things architectures | 1,172 | indexed | 31 | 41 |
| `10.1016/j.patter.2024.100943` | Can large language models reason about medical questions? | 294 | indexed | 12 | 19 |
| `10.1109/t-affc.2012.16` | Affective Body Expression Perception and Recognition: A Survey | 607 | indexed | 19 | 40 |
| `10.1111/jcpp.12721` | Phase 2 of CATALISE: a multinational and multidisciplinary Delphi consensus study of problems with language development: Terminology | 1,624 | indexed | 13 | 21 |
| `10.1111/joa.12446` | A review of trabecular bone functional adaptation: what have we learned from trabecular analyses in extant hominoids and what can we apply to fossils? | 243 | indexed | 26 | 42 |
| `10.1145/3394486.3403305` | Embedding-based Retrieval in Facebook Search | 273 | indexed | 9 | 17 |
| `10.1145/3447772` | Knowledge Graphs | 1,803 | indexed | 37 | 46 |
| `10.1145/3450287` | Event Prediction in the Big Data Era | 107 | indexed | 37 | 49 |
| `10.1145/3569576` | Knowledge Tracing: A Survey | 459 | indexed | 37 | 43 |
| `10.1145/3916.3988` | Virtual time | 2,419 | indexed | 22 | 21 |
| `10.1152/jn.00005.2017` | The role of the hippocampus in navigation is memory | 442 | indexed | 37 | 24 |
| `10.1177/0269216318784474` | Advance care planning: A systematic review about experiences of patients with a life-threatening or life-limiting illness | 261 | indexed | 17 | 26 |
| `10.3390/electronics9050750` | A Survey on Knowledge Graph Embedding: Approaches, Applications and Benchmarks | 266 | indexed | 29 | 34 |
| `10.3390/g9030062` | Matrix Games with Interval-Valued 2-Tuple Linguistic Information | 11 | indexed | 19 | 20 |
| `10.3390/s120811113` | A Survey on Clustering Routing Protocols in Wireless Sensor Networks | 665 | indexed | 41 | 39 |
| `10.3390/s19030448` | Indexing Multivariate Mobile Data through Spatio-Temporal Event Detection and Clustering | 24 | indexed | 25 | 29 |
| `10.1002/14651858.cd000425.pub4` | Speech and language therapy for aphasia following stroke | 1,152 | pdf-only | 404 | — |
| `10.1002/hipo.22488` | Hippocampal sharp wave‐ripple: A cognitive biomarker for episodic memory and planning | 1,921 | pdf-only | 116 | — |
| `10.1016/j.sysarc.2019.02.009` | All one needs to know about fog computing and related edge computing paradigms: A complete survey | 1,367 | pdf-only | 50 | — |
| `10.1080/10447318.2019.1619259` | Seven HCI Grand Challenges | 516 | pdf-only | 42 | — |
| `10.1080/15622975.2016.1190867` | Biological markers for anxiety disorders, OCD and PTSD: A consensus statement. Part II: Neurochemistry, neurophysiology and neurocognition | 343 | pdf-only | 54 | — |
| `10.1152/physrev.00046.2019` | Brain mechanisms of insomnia: new perspectives on causes and consequences | 552 | pdf-only | 143 | — |
| `10.3390/electronics15061263` | A Query-Driven Graph Retrieval Framework with Adaptive Pruning for Multi-Hop Question Answering | 2 | pdf-only | 1048 | — |

---

## Appendix — known gaps

- **Oversized PDFs** (Set C): `hipo.22488` (116 pp), `physrev.00046.2019` (143 pp),
  `cd000425.pub4` (404 pp), `electronics15061263` (1048 pp), plus the 42–54 pp trio,
  cannot pass the scribe's 900 s conversion deadline. Split-PDF conversion is the remedy if
  a paper becomes load-bearing.
- **`10.1016/j.aiopen.2021.03.001`** was downloaded but lost to an rclone VFS flush failure;
  re-acquire if needed.
- **`10.1093/bioinformatics/btu033`** (Sci-Hub fallback) is an HTML error page, not a PDF.
- **Conversion is VRAM-bound**: olmOCR-2-7B-FP8 under vLLM on `big` needs ~14 GB free.
  The voice assistant's ollama (10 GB, 30-min rolling keep-alive) and other tenants block it;
  check `ssh big nvidia-smi` before batch conversion.
- The 388 pdf-only Set-B papers and 7 pdf-only Set-C papers can be promoted to `indexed`
  by running `scribe_convert` per stem (MCP; 30 s client abort is normal, poll `catalog_read`).

## Appendix — provenance & reproduction

- Sweep queries and funnel: `tmp/lit-sweep/` (`queries.json`, `downloads.jsonl`,
  `classified.json`, `filter.py`).
- Anna's Archive fetch: driver `tmp/annas/fetch_annas.py`, log `tmp/annas/results.jsonl`
  (plus `fetched.jsonl` and `scihub-fetched.jsonl` for the two fallback routes).
- Semantic-search harvest that produced Set A: 15 queries (3–4 per RQ theme) × top-30, log in
  `tmp/annas/semantic_hits.json`. Titles in Set A rows come from the search index; where a
  title is missing the stem is the identifier of record.
