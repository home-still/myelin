# 00 — Verified Environment & First-Hand Probe Results

Everything in this file was measured in this session (2026-09-14) against live systems, not read from docs.
It is the ground truth the plan builds on. Claims from vendor docs are marked `[doc]`; everything else is `[probed]`.

## 1. Hardware / hosts

| Host | Role | Facts |
|---|---|---|
| workstation (`darwin` arm64, Apple M5) | dev | `cargo 1.98.1`, `rustc 1.98.1` `[probed]` |
| `big` (CachyOS) | home cloud | 24 cores, 31 GiB RAM, RTX 3090 24576 MiB (4893 MiB resident at probe time), `/` on NVMe 3.7 TB with 1.5 TB free, NFS `/mnt/codex_fs` 1.8 TB (219 GB free, reads ~68 MB/s) `[probed]` |

`/mnt/codex_fs` is NFS at ~68 MB/s — never put hot weights or a database there (`skill://serve-gguf-on-big`).

## 2. Services on `big`

| Port | Service | Notes `[probed]` |
|---|---|---|
| 6333 / 6334 | Qdrant 1.19.1 (rootless podman), commit `6ab21cac` | HTTP / gRPC |
| 8081 | llama-swap (v229) | models: `qwen3.8-27b`, `qwen3-vl`, `olmocr`, `glm-ocr` |
| 11434 | ollama | `qwen3:4b`, `qwen3:8b`, `qwen3-vl:8b`, `qwen3-vl:8b-instruct`, `gpt-oss-20b-heretic`, `qwen2.5:7b`, `glm-ocr`, `bge-m3` |
| 7434 | `hs-distill-serve` 0.0.1-rc.352 | CUDA, `bge-m3`, collection `academic_papers` |
| 7435 | `hs-scribe-serve` 0.0.1-rc.352 | VLM PDF→markdown, 12 slots |
| 7445 | `hs-mcp` | MCP over streamable HTTP |

Existing Qdrant collections: `academic_papers`, `paper_abstracts`, `imessages`, `imessages_meta`,
`personal_docs`, `personal_docs_smoke`, `recall_test`, `voice_memory`, `voice_memory_probe`.
**`myelin` must not touch any of these.** Reserve the prefix `myelin_*`.

Local corpus state at probe time: 9,786 documents, 9,714 markdown, 9,770 embedded, 297,463 chunks, `bge-m3` on CUDA.

GPU tenancy is arbitrated by `gpu-tenant claim|release` on `big`; `hs-serve-distill` and llama-swap models are
evicted by a `coding` claim. Any myelin benchmark run that needs the GPU must claim it or accept eviction.

## 3. Qdrant 1.19.1 capability probes

Probe method: `qdrant-client` 1.19.0 over gRPC (`/tmp/qmvprobe`) plus raw REST. Scratch collections created and deleted.

### 3.1 One collection can hold all three retrieval channels `[probed]`

Created a single collection with **named dense** (`dense`, 4-d Cosine), **named multivector**
(`late`, 4-d Cosine, `multivector_config.comparator = max_sim`) and **named sparse** (`lex`) vectors.
`create`, `upsert` and both query forms succeeded over gRPC.

→ No second store is needed for hybrid + late-interaction retrieval.

### 3.2 Server-side RRF fusion works, but **k = 1**, not Cormack's k = 60 `[probed]`

```
prefetch[dense→rank1, lex→rank2] + query{fusion: rrf}  →  score 0.8333334 = 1/2 + 1/3
single-list ranks 1..6                                 →  0.5, 0.33333334, 0.25, 0.2, 0.16666667, 0.14285715
```

The observed series is exactly `1/(1 + rank)` for 1-based rank. Qdrant's RRF therefore uses **k = 1**.
Cormack/Clarke/Buettcher's canonical constant is 60, which would give ≈0.0164/0.0161/0.0159 — nearly flat.
**Consequence:** Qdrant's server-side RRF is far more top-rank-biased than the literature default.
If we want k = 60 semantics we must fuse client-side in Rust. `DBSF` fusion is also accepted (`200`).

### 3.3 Server-side late-interaction rerank works over gRPC, is broken over REST `[probed]`

`prefetch{dense, limit 10} + query{multivector matrix} using "late"` returns `max_sim` scores over gRPC.
The identical query over REST fails: `422 Validation error in JSON body: [internal.query.indices: must be unique]`
— the untagged `VectorInput` enum mis-parses a nested float matrix as a sparse vector. Upsert of a multivector
over REST is fine; only the *query* path is broken.

**Consequence:** `myelin-core` must talk gRPC (port 6334). This is also what `qdrant-client` does natively.

### 3.4 Qdrant computes BM25 **locally, server-side**, with no inference service `[probed]`

This is the single most useful finding for the build.

```
PUT /collections/X  { sparse_vectors: { lex: { modifier: "idf" } } }
upsert  vector.lex = { text: "the time machine by h g wells", model: "qdrant/bm25" }   → 200
scroll  → indices [242555882, …] values [1.6697302 ×5]      (stopwords the/of/by/h/g dropped, terms stemmed+hashed)
query   vector.lex = { text: "martian invasion", model: "qdrant/bm25" }               → 200, only the matching doc, score 2.3209305
```

Accepted option block (all `200`): `{k: 1.2, b: 0.75, avg_len: 64.0, language: "english",
stemmer: {type: "snowball", language: "english"}, stopwords: "english", lowercase: true}`.

The IDF modifier is applied at query time. Verified formula reproduction on a 3-point collection:

```
IDF(t) = ln( (N − n(t) + 0.5) / (n(t) + 0.5) + 1 )
N = 3, n(11) = 2, n(22) = 1  →  IDF(11) = ln 1.6 = 0.470004,  IDF(22) = ln 2.666… = 0.980829
point1 (1.2·IDF(11) + 0.8·IDF(22)) = 1.3486677   ← Qdrant returned 1.3486677
point2 (1.5·IDF(11))               = 0.7050055   ← Qdrant returned 0.70500547
```

Scoring is exactly `Σ_i q_i · d_i · IDF(i)` over the sparse index. Statistics are per shard `[doc]`.

**Consequence:** no `tantivy`, no client-side BM25 encoder, no separate lexical index. This matters because
BM25 is the highest-leverage retrieval channel in the 2026 evidence (MemPro ablation: LoCoMo 84.93 → 72.25
when BM25 is removed, vs → 82.57 when dense embedding is removed — `docs/research/11-frontier-2026.md` §3.2).

### 3.5 Model-based inference is **not** available locally `[probed]`

```
dense   model "sentence-transformers/all-MiniLM-L6-v2" → 500 InferenceService URL not configured
sparse  model "qdrant/minicoil-v1"                     → 500 InferenceService URL not configured
sparse  model "qdrant/bm25"                            → 200
```

Only `qdrant/bm25` is computed in-process (it is arithmetic, not a neural model). Dense vectors must be
produced by us: `fastembed`+`ort` in-process, or the existing `hs-distill` HTTP service, or ollama `bge-m3`.

### 3.6 Per-tenant IDF scoping — shape not yet resolved `[probed, partial]`

`params: {idf: {must: [...]}}` → `400 data did not match any variant of untagged enum IdfParams`.
The capability is documented `[doc]` but the exact payload shape needs to be read off the gRPC proto before use.
Relevant for multi-tenant memory, where global IDF would leak cross-tenant term statistics.

## 4. Benchmark data acquisition

| Dataset | Result `[probed]` |
|---|---|
| LoCoMo | `https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json` → 200, 2,805,274 bytes |
| LongMemEval | HF dataset `xiaowu0162/longmemeval`, not gated, configs `longmemeval_s`, `longmemeval_m`, `longmemeval_oracle` |
| `longmemeval_s` | downloaded, 278,025,796 bytes, 500 items |

### 4.1 LoCoMo `locomo10.json` — measured contents

10 conversations. Sessions and turns per conversation:

| sample_id | sessions | turns |
|---|---|---|
| conv-26 | 19 | 419 |
| conv-30 | 19 | 369 |
| conv-41 | 32 | 663 |
| conv-42 | 29 | 629 |
| conv-43 | 29 | 680 |
| conv-44 | 28 | 675 |
| conv-47 | 31 | 689 |
| conv-48 | 30 | 681 |
| conv-49 | 25 | 509 |
| conv-50 | 30 | 568 |

QA by category: `1: 282, 2: 321, 3: 96, 4: 841, 5: 446` — **total 1,986**.

**Protocol landmine, measured:** `1986 − 446 (category 5) = 1540`. The "1,540 questions" figure quoted by
every LoCoMo-reporting paper and vendor blog is the file **with the adversarial category dropped**. Any myelin
number must state which of `1986` / `1540` it used, or it is not comparable to anything.

Per-item schema: `{question, answer, evidence: ["D1:3", …], category}`, conversation keys
`{conversation, event_summary, observation, qa, sample_id, session_summary}`.

### 4.2 LongMemEval `longmemeval_s` — measured contents

500 items. Keys: `answer, answer_session_ids, haystack_dates, haystack_session_ids, haystack_sessions,
question, question_date, question_id, question_type`. Mean haystack sessions per item: **50**.

Question types: `temporal-reasoning 133, multi-session 133, knowledge-update 78, single-session-user 70,
single-session-assistant 56, single-session-preference 30`.

## 5. Rust crate versions — verified on crates.io, 2026-09-14 `[probed]`

| crate | max stable | released | license |
|---|---|---|---|
| `rmcp` | 3.3.0 | 2026-09-10 | Apache-2.0 |
| `qdrant-client` | 1.19.0 | 2026-08-04 | Apache-2.0 |
| `fastembed` | 6.1.0 | 2026-09-12 | Apache-2.0 |
| `ort` | 2.0.0-rc.13 | 2026-07-28 | MIT OR Apache-2.0 |
| `tantivy` | 0.26.2 | 2026-09-08 | MIT |
| `sqlx` | 0.9.0 | 2026-05-21 | MIT OR Apache-2.0 |
| `petgraph` | 0.8.3 | 2025-09-30 | MIT OR Apache-2.0 |
| `tokenizers` | 0.23.2 | 2026-09-03 | Apache-2.0 |
| `tiktoken-rs` | 0.12.0 | 2026-06-02 | MIT |
| `divan` | 0.1.21 | 2025-04-10 | MIT OR Apache-2.0 |
| `insta` | 1.48.0 | 2026-06-11 | Apache-2.0 |
| `proptest` | 1.11.0 | 2026-03-24 | MIT OR Apache-2.0 |
| `wiremock` | 0.6.5 | 2025-08-24 | MIT/Apache-2.0 |
| `hf-hub` | 1.0.0 | 2026-07-10 | Apache-2.0 |
| `axum` | 0.8.9 | 2026-04-14 | MIT |
| `ollama-rs` | 0.3.6 | 2026-07-24 | non-standard |
| `async-openai` | 0.42.0 | 2026-09-09 | MIT |
| `figment` | 0.10.19 | 2024-05-17 | MIT OR Apache-2.0 |
| `thiserror` | 2.0.20 | 2026-08-08 | MIT OR Apache-2.0 |
| `schemars` | 1.2.2 | 2026-07-27 | MIT |
| `rusqlite` | 0.40.2 | 2026-08-08 | MIT |
| `redb` | 4.2.0 | 2026-08-17 | MIT OR Apache-2.0 |
| `arrow` / `parquet` | 59.3.0 | 2026-09-01 | Apache-2.0 |
| `criterion` | 0.8.2 | 2026-02-04 | Apache-2.0 OR MIT |
| `bm25` | 2.3.2 | 2025-09-07 | MIT |

`/Users/ladvien/home-still/Cargo.lock` pins older versions (`rmcp` 1.3.0, `qdrant-client` 1.17.0,
`fastembed` 5.13.2, `ort` 2.0.0-rc.11). `rmcp` 3.x is a **breaking** change from 1.3 and its MSRV is Rust 1.88.

## 6. Corpus gap closed this session

Seven 2026 frontier papers were absent from the local corpus (`catalog_read` → "No catalog entry"), while their
PDFs were already on disk. `hs pipeline catch-up -y` republished 33 `papers.ingested` events; the scribe/distill
watchers are converting and indexing them:

`2602.13594` Hippocampus · `2605.12493` LongMemEval-V2 · `2606.00619` MemPro · `2606.06036` MRAgent/graph-memory ·
`2601.20352` AMA · `2605.23067` RL memory curriculum · `2606.24937` Hitchhiker's Guide to Agentic AI.

They were read directly from arXiv for `docs/research/11-frontier-2026.md`, so the plan does not depend on the
conversion finishing.

### 4.3 LongMemEval-V2 — acquisition verified `[probed]`

HF dataset `xiaowu0162/longmemeval-v2`, **ungated, Apache-2.0**, 4,408 downloads at probe time.
File manifest with real sizes from the HF blobs API:

| file | size |
|---|---|
| `trajectory_screenshots/enterprise_screenshots_base.tar.gz` | 3,354.2 MB |
| `trajectory_screenshots/web_screenshots.tar.gz` | 2,562.3 MB |
| `trajectories.jsonl` | 1,195.6 MB |
| `haystacks/lme_v2_medium.json` | 4.1 MB |
| `haystacks/lme_v2_small.json` | 0.8 MB |
| `questions.jsonl` | 0.3 MB |
| `DATA_CARD.md`, `SCHEMA.md`, `LICENSE`, `checksums.sha256` | < 10 KB each |
| `question_screenshots/*.png` | 29 files, 3.1 MB total |

Repo total 7.12 GB across 41 files. **A text-only evaluation needs ≈1.5 GB** (`trajectories.jsonl` +
haystacks + questions); the 5.9 GB of screenshot archives are only required for a multimodal reader.
`checksums.sha256` is shipped, so dataset integrity can be pinned rather than trusted.

Paper facts re-verified by the lead directly against `https://arxiv.org/html/2605.12493v1`:
451 manually curated questions; five abilities (static state recall, dynamic state tracking, workflow
knowledge, environment gotchas, premise awareness); Small = one shared 100-trajectory haystack ≈25M tokens,
Medium = question-specific 500-trajectory haystacks ≈115M tokens; AgentRunbook-C **74.9% Small / 70.1%
Medium / 72.5% overall**; AgentRunbook-R **58.6 / 57.0 / 57.8%**; plain state-slice RAG **40.1%**; strongest
external RAG baseline **48.5%**; vanilla Codex **69.9 / 68.7 / 69.3%** at **≈182 s/query, 6.9× slower than
AgentRunbook-R**, with AgentRunbook-C 32% faster than Codex.

### 4.4 MemPro ablation and dual-backbone results `[probed]`

Re-verified by the lead directly against `https://arxiv.org/html/2606.00619v1`:

> "BM25 is the most important retrieval channel: removing it drops accuracy from 84.93 to 72.25 with
> gpt-4o-mini and from 77.85 to 65.44 with Qwen3-30B-A3B-Instruct-2507."

Embedding removal: 84.93 → 82.57 (gpt-4o-mini), 77.85 → 75.67 (Qwen3-30B). PAGE-ID removal: 84.93 → 84.37,
77.85 → 77.12. Table 1 MemPro-15 by backbone:

| backbone | LongMemEval avg | LoCoMo avg |
|---|---|---|
| gpt-4o-mini | 79.00 | 84.93 |
| Qwen3-30B-A3B-Instruct-2507 | 80.80 | 77.85 |

The open-weights backbone scores **higher** on LongMemEval and **lower** on LoCoMo — cross-model comparison
on these benchmarks is not monotonic.

## 7. GPU tenancy is the binding constraint — and a corrected diagnosis

An earlier draft of this file asserted that the `olmocr` llama-swap entry was broken. **That was wrong, and
the correction is instructive.** What follows is the full evidence chain in the order it was obtained.

### 7.1 The symptom

`hs pipeline catch-up -y` queued 33 `papers.ingested` events; the scribe watcher dispatched them to the
healthy scribe at `192.168.1.110:7435` with the `olmocr` backend and a 1,800 s page-scaled timeout. No
conversion completed. Evidence `[probed]`:

- `curl http://localhost:8081/running` → `{"running":[]}`; `nvidia-smi` → 4,893 MiB used, 0% utilization.
- `llama-swap` journal: repeated `POST /v1/chat/completions … 200 0 "" 14m58s` — **HTTP 200 with a zero-byte
  body** after ~15 minutes, once per queued conversion.
- A direct `POST /v1/chat/completions` for `model: "olmocr"` hung and was killed at 120 s (`HTTP 000`).
- `gpu-tenant status` → `tenant=none`; scribe `/health` → `total_conversions: 149`,
  `last_conversion_at: 2026-09-13T15:30:49Z` (≈21 h stale). `pipeline_drift` was already 26 before any
  action in this session, so the stall predates it.

### 7.2 The root cause

`/tmp/qwen3.8-child.log` gave it away when the same empty-200 behaviour appeared for `qwen3.8-27b`
`[probed]`:

```
load_model: loading model '/home/ladvien/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf'
common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 999
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 16053.22 MiB on device 0: cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 16833026048
llama_model_load: error loading model: unable to allocate CUDA0 buffer
srv  llama_server: exiting due to model loading error
```

The card had 4,893 MiB held by `hs-serve-distill` (bge-m3 on CUDA). Claiming the GPU released it `[probed]`:

```
gpu-tenant claim coding  →  tenant=coding, paused= trellis2-mcp.service hs-serve-distill.service
nvidia-smi               →  343 MiB / 24576 MiB   (was 4893 MiB)
```

With the card free, **`olmocr` loaded and served immediately** — `{"running":[{"model":"olmocr",
"state":"ready",…}]}` at 13,701 MiB, and the llama-swap journal switched to successful conversions:
`200 12601 "" 1m49s`, `200 5592 "" 47s`, `200 5703 "" 33s`. Scribe's `total_conversions` moved 149 → 153 and
the markdown count 9,714 → 9,719, draining the queue this session created.

So the backend was never broken: it was **starved of VRAM**, and the failure was invisible because of a
genuine but secondary defect — **llama-swap returns HTTP 200 with an empty body when the upstream child
fails to spawn**, instead of surfacing the error. That masking is the thing worth fixing upstream.

This matches `skill://serve-gguf-on-big` exactly: Qwen3.8-27B UD-Q4_K_XL needs ≈22.7 GB of the 24 GB card
and is documented as loading only "under claim, distill stopped".

### 7.3 Consequences for `myelin`

- One RTX 3090 cannot host the paper pipeline and a benchmark run at the same time. `myelin-eval` MUST take
  a `gpu-tenant claim`, record tenancy in the run manifest, and fail fast if another tenant holds it —
  `memory_query_avg_seconds` is half of the LAFS score, so a contended run is not merely slow, it is invalid.
- Measured VRAM envelope: 343 MiB free-and-clear → 13,701 MiB with one VLM resident → ≈22.7 GB for a 27B
  Q4. The LongMemEval-V2 stack (`Qwen3.5-9B` + `Qwen3-Embedding-8B`) must be sized inside that envelope.
- **Local tool-calling is verified** and is the prerequisite for the agentic eval loop: ollama `qwen3:8b` at
  `localhost:11434/v1`, with `olmocr` concurrently holding 13.7 GB, returned
  `finish_reason: "tool_calls"` and a well-formed `recall({"k":1,"query":"cat"})` in 13.7 s `[probed]`.
- Treat an HTTP 200 with an empty body from llama-swap as a **load failure**, not a null answer. Any
  `myelin` LLM client must reject empty completions rather than propagating them as valid.
