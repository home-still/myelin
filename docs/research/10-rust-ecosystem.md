# myelin — Rust Crates: Ecosystem Survey & Dependency Decisions

Target: `myelin-core` (backend crate), `myelin-mcp` (MCP server crate), `myelin-eval` (eval harness). All versions verified against docs.rs / crates.io / GitHub on **2026-09-14**. Crate-version claims are sourced from the URLs given. All Qdrant server-side fusion/multivector claims carry a Qdrant docs URL.

Reference-convention anchor: the running **home-still** workspace (`/Users/ladvien/home-still/Cargo.toml`, `crates/hs-distill/Cargo.toml`, `crates/hs-mcp/Cargo.toml`, `crates/hs-distill/src/embed/onnx.rs`, `crates/hs-distill/src/config.rs`) already standardizes this house style — same `tokio`/`serde`/`anyhow`/`thiserror`/`tracing`/`clap`/`axum`/`reqwest`/`figment`/`qdrant-client`/`fastembed`/`ollama-rs` stack, `feature`-gated server/client split, and a `cuda` feature (`hs-distill/Cargo.toml` `cuda = ["server", "ort/cuda"]`). myelin should match exactly.

> **Version-surge warning (important).** The current crates.io/docs.rs metadata is far ahead of home-still's locked versions (which were pinned months ago). Adopt the latest here, not the pinned ones — the ecosystem moved: `rmcp` 1.3 → **3.3.0**, `qdrant-client` 1.17 → **1.19.0**, `fastembed` 5.13 → **6.1.0** (now ships a candle backend for qwen3/nomic-v2), `ort` rc.11 → **rc.13**, `ollama-rs` 0.3.4 → **0.3.6**, `axum` 0.8.8 → **0.8.9**, `tokenizers` 0.22 → **0.23.2**.

---

## 1. MCP — rmcp (official Rust SDK)

**Latest: `rmcp` 3.3.0** (published 2026-09-10) — https://docs.rs/crate/rmcp/latest — license MIT/Apache-2.0 dual (repo `LICENSE`). Actively maintained (official Anthropic/MCP SDK; 3.0.0 shipped 2026-07-28, then 3.1/3.2/3.3 through Sept 2026).

**1.3 is NOT the latest.** home-still pins `rmcp = "1.3"` (`crates/hs-mcp/Cargo.toml`), which is now 9 minor + 2 major releases behind. `rmcp` 1.3.0 = 2026-03-26; then 1.4–1.8, 2.0.0 (2026-06-29, wire-compatible), and **3.0.0 (2026-07-28)** — a **breaking** API release. `rmcp` MSRV: Rust 1.88 (migration guide PR #1034).

**Protocol revision implemented:** stable **MCP `2026-07-28`** spec, fully backward-compatible with `2025-11-25` and earlier — https://raw.githubusercontent.com/modelcontextprotocol/rust-sdk/main/README.md. `ProtocolVersion` gains `V_2026_07_28`; in 3.0.0 `LATEST` still defaults to `V_2025_11_25`, so select `V_2026_07_28` explicitly.

**Macro API:** `#[tool]`, `#[tool_router]` (with `#[tool_router(server_handler)]` to skip a separate `ServerHandler` impl), `#[tool_handler]`, plus `#[prompt]`, `#[prompt_router]`, `#[prompt_handler]`. JSON Schema 2020-12 via `schemars`. Example: `#[tool_router(server_handler)] impl Calculator { #[tool(description = ...)] fn add(&self, Parameters(AddParams{a,b}): Parameters<AddParams>) -> String }` (README § Tools).

**Transports:** stdio (`transport-io` server / `transport-child-process` client), Streamable HTTP (`transport-streamable-http-server` server / `transport-streamable-http-client[-reqwest]` client), generic `transport-async-rw`; pluggable `Transport` trait + `IntoTransport` for `(Sink,Stream)`/`(AsyncRead,AsyncWrite)`/`Worker`. TLS via `reqwest` (rustls) / `reqwest-native-tls`. Features: `server` (default), `client`, `macros` (default), `schemars`, `auth`, `auth-enterprise-managed`, `elicitation` (gated!). — https://docs.rs/rmcp/latest/rmcp/

**Feature support (all README-cited, spec 2026-07-28):**
- **resources** — `list_resources`, `read_resource`, `list_resource_templates`, ResourceContents text/blob, notify_resource_list_changed/updated.
- **prompts** — `#[prompt_router]`/`#[prompt]`, `PromptMessage` (text/image/embedded-resource), `GetPromptResult`.
- **sampling** — server→client `create_message` via `context.peer.create_message()`; **deprecated (SEP-2577)** but fully functional (README § Sampling).
- **elicitation** — form mode (`elicit::<T>()` + `elicit_safe!`) and URL mode (`elicit_url`); **must enable the `elicitation` feature**.
- **progress notifications** — `context.peer.notify_progress(ProgressNotificationParam::new(token, progress).with_total(n))` (README § Notifications).
- **roots**, **logging**, **completions** (`enable_completions()` + `complete()`), **subscriptions** (`subscriptions/listen` in 2026-07-28), **multi-round-trip requests** (`InputRequiredResult`/`input_required`), **Tasks Extension** (SEP-2663).

**Breaking changes since 1.3 (migration guide — https://github.com/modelcontextprotocol/rust-sdk/discussions/969):**
- `ServerHandler::call_tool`/`get_prompt`/`read_resource` return `Result<T, ErrorData>` where T is a response enum (not bare `*Result`); **new `InputRequiredResult` variant** requires a new match arm.
- `ToolResultContent.structured_content`: `Option<JsonObject>` → `Option<Value>` (SEP-2106).
- `Annotations.last_modified`: `DateTime<Utc>` → `String`.
- `Meta` split into `MetaObject`/`RequestMetaObject`/`NotificationMetaObject`; deprecated alias removed.
- Protocol message unions now `#[non_exhaustive]` — add a wildcard arm.
- Streamable HTTP: `stateful_mode` → **`legacy_session_mode`** (2026-07-28 requests always stateless).
- Tasks moved from experimental core API to the `io.modelcontextprotocol/tasks` extension (old `tasks/list`/`tasks/result`/`#[task_handler]` removed).
- OAuth: `discover_metadata` → `resolve_metadata`; unified `start_authorization(AuthorizationRequest)`.
- Client auto-drives MRTR rounds; opt out with `*_once`.

**Do NOT copy home-still's `rmcp = "1.3"`.** myelin-mcp should use `rmcp = "3.3"` with features `server`, `macros`, `schemars`, `transport-io`, `transport-streamable-http-server`, and `elicitation` if elicitation is wanted.

---

## 2. Qdrant — qdrant-client (Rust gRPC client)

**Latest: `qdrant-client` 1.19.0** — https://docs.rs/qdrant-client/latest/qdrant_client/ — matches **Qdrant server 1.19** (the version already running on `big`, Qdrant 1.19.1). License: Apache-2.0. Maintained (Qdrant official; tracks server releases).

**API shape (from the struct docs / Qdrant docs):**
- **gRPC builder connect** — `Qdrant::from_url("http://localhost:6334").api_key(..).build()?` (gRPC port 6334).
- **named vectors** — `CreateCollectionBuilder::new(name).vectors_config(VectorsConfigBuilder)`: `add_named_vector_params("image"/"text", VectorParamsBuilder::new(size, Distance))`; sparse via `sparse_vectors_config(SparseVectorsConfigBuilder)`. (Qdrant docs: https://qdrant.tech/documentation/manage-data/collections/index.md, https://qdrant.tech/documentation/manage-data/vectors/index.md.)
- **Filter construction** — `FilterBuilder::new().must([..]).should([..]).must_not([..]).min_should(..)` (docs.rs Filter page).
- **query_points / prefetch fusion** — `QueryPointsBuilder::new(name).add_prefetch(PrefetchQueryBuilder::default().query(Query::new_nearest(..)).using("dense"|"sparse").limit(..)).query(Query::new_rrf(RrfBuilder::default()))`. `RrfBuilder::with_k(u32)` + `.weights(Vec<f32>)` (docs.rs `RrfBuilder` page).
- **scroll** — `client.scroll(ScrollPointsBuilder...)`.
- **snapshots** — `create_snapshot`, `list_snapshots`, `download_snapshot`, `delete_snapshot`, `create_full_snapshot`, etc. (docs.rs `Qdrant` struct).
- **payload indexes** — `create_field_index(CreateFieldIndexCollectionBuilder::new(name).field_name(..).field_type(..))`.
- **named-vector adds** — `create_vector_name`, `delete_vector_name`.

**Server-side hybrid fusion — CONFIRMED for Qdrant 1.10+/1.11+/1.16/1.17:**
- Prefetch + RRF (`query: {rrf:{}}`) — https://qdrant.tech/documentation/search/hybrid-queries/index.md (RRF since v1.10; parameterized `k` since v1.16; weighted RRF since v1.17).
- **DBSF** (`query: {fusion:{fusion:"dbsf"}}` / `models.Fusion.DBSF`) — available since **v1.11.0** — hybrid-queries doc § Distribution-Based Score Fusion. Guidance: RRF = safe default; weighted RRF with an eval set; DBSF when you trust raw scores. Since big runs Qdrant 1.19, **both RRF and DBSF server-side fusion work natively.**

**Multivector comparators — CONFIRMED for v1.10+ (big's 1.19 supports it):**
- `multivector_config: {comparator: "max_sim"}` on a dense vector; Rust: `MultiVectorConfigBuilder::new(MultiVectorComparator::MaxSim)` — https://qdrant.tech/documentation/manage-data/vectors/index.md (`#multivectors`; `MultiVectorComparator` in `qdrant_client::qdrant`). MaxSim = sum over query vectors of max similarity against candidate vectors (late interaction / ColBERT).
- Rust client 1.19.0 ships these builders (docs.rs examples use them; `MultiVectorComparator`/`RrfBuilder`/`PrefetchQueryBuilder` present at 1.19.0).

So for myelin: one collection can hold named dense + sparse vectors and a **multivector** (ColBERT from BGE-M3), queried via prefetch + `rrf`/`dbsf`, and compared with MaxSim for late-interaction rerank — all server-side on big's Qdrant 1.19.

**Feature flag:** `qdrant-client = { version = "1.19", features = ["serde"] }` (home-still uses `serde` for payload serde).

---

## 3. Sparse / lexical retrieval

**`tantivy` — latest 0.26.2** — https://docs.rs/tantivy/latest/tantivy/ — license Apache-2.0/MIT. Actively maintained (Quickwit). Pure Rust, works on macOS arm64 and Linux CUDA (it is CPU-only by design — no GPU needed).
- API: `Index::create_in_dir`, `Schema::builder().add_text_field(.., TEXT|STORED)`, `index.writer(100_000_000)`, `writer.add_document(doc!(...))`, `commit`; `reader.searcher()`, `QueryParser::for_index(&index, vec![title, body])`, `searcher.search(&query, &TopDocs::with_limit(10).order_by_score())` → `(Score, DocAddress)`. Built-in **BM25 scoring** (Lucene-style, default similarity).

**`bm25` crate — latest 2.3.2** (2025-09-07) — https://docs.rs/bm25/latest/bm25/ — Apache-2.0/MIT. Small, focused. `EmbedderBuilder::with_fit_to_corpus(Language::English, &corpus).build()` → `embedder.embed(text)` produces sparse `(index, value)` vectors ready for Qdrant sparse fields; `Scorer` and in-memory `SearchEngine` included. Good fit if you want lexical weights as **Qdrant sparse vectors** fused server-side, rather than a separate tantivy index.

**`probly-search` — latest 2.0.1** — https://docs.rs/probly-search/latest/probly_search/ — MIT (rust-nostr author). Simple `Index::new(tokenizer, field_accessor)`; **no BM25 scores** (union + heuristic scoring, not BM25) — weaker retrieval relevance; not recommended here.

**BGE-M3 sparse from Rust — YES via fastembed.** `fastembed 6.1.0` provides `SparseTextEmbedding` (SPLADE-PP, **BGE-M3**, OpenSearch gte) and `Bgem3Embedding` whose `embed()` returns **dense + sparse (lexical) + ColBERT multivectors in a single pass** (`Bgem3EmbeddingOutput`). So myelin can produce BGE-M3 lexical weights and ColBERT from Rust, offline at query time. (docs.rs `fastembed` enum SparseModel, struct Bgem3Embedding.)

**Recommendation:** use `fastembed`'s BGE-M3 sparse vectors → Qdrant sparse field (fused via RRF/DBSF). Keep `tantivy 0.26` as the **dedicated in-process BM25 index** for exact-term recall on small/audit corpora, OR the `bm25 2.3` crate for pure sparse-vector generation without a second index store. `probly-search` is rejected (no BM25 scores).

---

## 4. Embeddings / reranking locally

**`fastembed` — latest 6.1.0** — https://docs.rs/fastembed/latest/fastembed/ — Apache-2.0. Local ONNX inference, synchronous (no Tokio), models cached once then offline. Actively maintained (Qdrant ecosystem).
- **Model lists (docs.rs 6.1.0):** `EmbeddingModel` (default `BGESmallEnV15`; also `BGEM3`), `SparseModel::{SPLADEPPV1, BGEM3, OpenSearchNeuralSparseDocV3Gte}`, `Bgem3Model::{BGEM3Q}` (gpahal/bge-m3-onnx-int8), `RerankerModel::{BGERerankerBase, BGERerankerV2M3, JINARerankerV1TurboEn, JINARerankerV2BaseMultiligual}`.
- **BGE-M3** = `EmbeddingModel::BGEM3` for dense, or `Bgem3Embedding` for dense+sparse+ColBERT together.
- **rerankers** = `TextRerank::new(RerankInitOptions)`, `RerankerModel::BGERerankerBase` / `BGERerankerV2M3` / `JINARerankerV1TurboEn` / `JINARerankerV2BaseMultiligual`.
- **ONNX + ort CUDA:** fastembed runs on `ort`; enable `fastembed` `cuda` feature → `ort::execution_providers::CUDAExecutionProvider` (home-still proves it: `crates/hs-distill/src/embed/onnx.rs:85-92` sets `CUDAExecutionProvider::default().build()` for `EmbeddingModel::BGEM3`, with a VRAM probe to confirm no silent CPU fallback). ort pins to `=2.0.0-rc.13` (fastembed feature page).
- **macOS arm64:** ort's CPU backend works on Apple Silicon (Accelerate via `accelerate`; Metal EP via `ort`/`metal` — for embeddings CPU/Accelerate suffices; CUDA never on macOS).
- **6.1.0 change:** candle backend added for `qwen3`/`nomic-v2-moe` models behind feature flags; the `cuda` feature pulls `candle-core`/`candle-nn` cuda for those. Dense/rerank/BGE-M3 path stays ort-ONNX.

**`ort` — latest 2.0.0-rc.13** — https://docs.rs/ort/latest/ort/ — MIT + ONNX Runtime license components. `ort = "=2.0.0-rc.13"` is what fastembed 6.1.0 requires (feature page: `ort = "=2.0.0-rc.13"`).

**`candle` — latest 0.11.0** (candle-core/candle-transformers 0.11.0, 2026-06-26) — https://docs.rs/candle-core/latest/candle_core/, https://docs.rs/crate/candle-transformers/latest — Apache-2.0/MIT, huggingface. In-process CPU/CUDA/Metal. **Cross-encoder reranker practical?** Yes in principle (candle-transformers has transformer models; load a bge-reranker onnx/safetensors and score pairs), but it is a DIY reranker — no turnkey `TextRerank`-style API, and wiring a BERT cross-encoder by hand is significant work. **Not worth it** given fastembed already ships a turnkey `TextRerank` with bge/jina models.

**Alternatives / tradeoffs (concrete):**
- **Local fastembed (chosen):** zero network at ingest/query, BGE-M3 dense+sparse+ColBERT + cross-encoder rerankers all in-process, ort CUDA on big (RTX 3090), Accelerate on Mac. Cost: manages the ort CUDA runtime (home-still's `lib_bootstrap`/VRAM probe pattern), pulls large models once.
- **Call `hs-distill` HTTP (7434) or ollama `bge-m3`: ** avoids bundling ort at all in myelin — myelin-core just POSTs text → embeddings. Tradeoff: couples myelin to a running home-still or ollama on the LAN; no ColBERT/reranker unless that service exposes it; extra hop. Good as a **feature-gated fallback** but not the primary path.
- **candle: ** rejected for rerank/embed because fastembed (ort) already covers it with less custom code; candle only attractive for qwen3/nomic-v2 MoE embeddings (behind fastembed features) — not core to myelin.

**Recommendation:** `fastembed = "6.1"` (features `cuda` on Linux big / default on macOS) via `ort = "=2.0.0-rc.13"`. Match home-still's feature split: `embed = ["fastembed", "ort/cuda"]` on big.

---

## 5. LLM clients

- **`ollama-rs` — latest 0.3.6** — https://docs.rs/ollama-rs/latest/ollama_rs/ — MIT, pekingduck/ollama-rs. Async; `Ollama::new(host, port)`, `ollama.generate(GenerationRequest)`, chat, streaming (`stream` feature). **Structured output:** `GenerationRequest::new(...).format(FormatType::Json)` (0.3.6 docs: `.format(FormatType)`), `.options(ModelOptions)` for temperature etc. **Tool calling:** `generation::tools` module exists but tool-call ergonomics are weaker than async-openai's; format is Ollama's own `format`, not full OpenAI `response_format={json_schema}`.
- **`async-openai` — latest 0.42.0** — https://docs.rs/async-openai/latest/async_openai/ — MIT, 64bit. Official-OpenAI typed API **and fully configurable for any OpenAI-compatible base URL**: `OpenAIConfig::new().with_api_base("http://big:8081/v1")`. First-class: `chat()` create with `ResponseFormat::JsonSchema { json_schema }` (structured output), typed `Tool`/tool-calling loop helpers, streaming, `Responses` API. `byot` feature for raw JSON passthrough.
- **plain `reqwest` — 0.13 (0.12 in home-still lock)** — https://docs.rs/reqwest/latest/reqwest/ — Apache-2.0/MIT. Minimal typed surface: hand-build the JSON body against llama-swap's OpenAI-compatible `/v1/chat/completions`; full control over `response_format`/`tools`/`stream`; no extra dependency, but you reimplement tool-call parsing.

**Recommendation: `async-openai` 0.42** as the LLM client inside myelin-core, pointed at **llama-swap's OpenAI-compatible endpoint** (`http://big:8081/v1`) — typed structured-output (`ResponseFormat::JsonSchema`) and typed tool-calling with the least glue, and the same code path works against ollama (base-url override) for eval/local runs. **Fallback:** plain `reqwest` for one-off embedding/rerank calls (see §4) where typed bodies buy nothing. **Rejected:** `ollama-rs` for generation — no native OpenAI `json_schema` response-format and weaker tool-calling; keep it (as home-still does) only for the ollama-only metadata path if needed.

---

## 6. Storage / state

- **`sqlx` — latest 0.9.0** — https://docs.rs/sqlx/latest/sqlx/ — Apache-2.0/MIT, LaunchBadge. Async (tokio/async-std). **SQLite driver is runtime-agnostic**; `SqlitePool` needs a runtime for timeouts/spawn. Postgres supported. **FTS5:** SQLx executes arbitrary SQL, so `CREATE VIRTUAL TABLE t USING fts5(...)` and `MATCH` queries run fine via `sqlx::query` — **SQLx exposes FTS5 because it is just SQL**, no dedicated API needed (FTS5 is built into libsqlite3). **Compile-time checked queries:** `query!`/`query_as!` need a live DB or `DATABASE_URL` + `sqlx-cli`-generated `.sqlx` offline metadata (`sqlx prepare --check` in CI). Tradeoff: adds a build-time DB/offline-metadata dependency for the checked macros; `query("...")` unchecked avoids that. **Recommend SQLx** for the relational/graph/audit side (async, future Postgres, one pool for tokens/graph/audit), with `sqlx = { version = "0.9", default-features=false, features=["runtime-tokio","sqlite","macros"] }` and offline metadata via `sqlx-cli`.
- **`rusqlite` — latest 0.40.2** — https://docs.rs/rusqlite/latest/rusqlite/ — MIT. Sync, zero-overhead. FTS5 via `bundled`/`fts5` feature. Simple for embedded-only SQLite. Rejected as primary because myelin-core is async; consider only for a sync eval shim.
- **`redb` — latest 4.2.0** — https://docs.rs/redb/latest/redb/ — Apache-2.0/MIT, cberner. Pure-Rust ACID embedded KV (B+tree), zero-copy, MVCC. Good for a fast KV store but no SQL/FTS/relational semantics.
- **`sled` — latest 1.0.0-alpha.124 (unstable) / 0.34.7 (2021 stable)** — https://docs.rs/crate/sled/latest — **rejected**: stable 0.34.7 is from 2021 (unmaintained), on-disk format not frozen until 1.0, docs say "use SQLite if reliability is your primary constraint."
- **`native_db` — latest 0.8.2** — https://docs.rs/crate/native_db/latest — Apache-2.0/MIT, vincent-herlemont. Drop-in embedded DB over redb with `#[native_db]` derive, multi-index, migrations; **API explicitly "not stable yet"** and only 18% documented. Rejected for audit/graph (schema evolves) — SQLite gives SQL/CTE/FTS.

**Recommendation:** `sqlx 0.9` (SQLite now, Postgres-ready). SQLite gives **FTS5 BM25** for lexical (`MATCH` + `bm25()`), **recursive CTEs** for graph/PageRank, JSON via `json1`, and the same file doubles as the audit store. FTS5 needs **no sqlx feature** — it is plain SQL on the virtual table. Compile-time query checking: enable only if you commit to `sqlx-cli` offline metadata; otherwise use `sqlx::query`.

---

## 7. Graph

- **`petgraph` — latest 0.8.3** — https://docs.rs/crate/petgraph/latest — Apache-2.0/MIT, bluss. In-memory `DiGraph`/`StableGraph`/`GraphMap`, algorithms: Dijkstra, BFS/DFS, min-spanning-tree. **page_rank / personalized PageRank is NOT built-in** (petgraph has no PPR algo).
- **Personalized PageRank in Rust:** **no first-class crate.** PPR implementations live in `petgraph`-adjacent community code or `sprs`/custom power iteration. Options: (a) implement power iteration over `petgraph::{DiGraph, visit}` — small, ~30 lines, deterministic; (b) **SQLite recursive CTE** for personalized walks (materialize adjacency in SQLite and run an iterative CTE) — no Rust graph lib at all, and queryable.
- **Recommendation:** `petgraph 0.8` for in-memory traversal + a ~30-line power-iteration PPR over it for per-memory-node importance, and SQLite recursive CTEs for persistence-heavy/neighbor-expansion queries. This uses both: petgraph in-process, CTE for SQL-side expansion.

---

## 8. Tokenization / budgeting

- **`tiktoken-rs` — latest 0.12.0** (2026-06-02) — https://docs.rs/tiktoken-rs/latest/tiktoken_rs/, https://docs.rs/crate/tiktoken-rs/latest — MIT, zurawiki. OpenAI tokenizers incl. `o200k_base`, `cl100k_base`, `p50k_base`. `o200k_base_singleton()` → `encode_with_special_tokens(text)`. MSRV Rust 1.85. Only OpenAI-family models — **not for qwen/llama**.
- **`tokenizers` (HuggingFace) — latest 0.23.2** — https://docs.rs/tokenizers/latest/tokenizers/ and https://docs.rs/crate/tokenizers/latest — Apache-2.0, HF. `Tokenizer::from_pretrained("Qwen/Qwen2.5-7B", None)` (needs `http` feature) or `from_bytes`/`from_file` on a local `tokenizer.json`; `encode(text, false)?.get_ids()`. **This is the exact tokenizer for qwen3/llama models** (BPE).
- **Recommendation:** `tokenizers 0.23` as the primary exact token-accounting path against a local model's `tokenizer.json` (qwen3, qwen2.5, gpt-oss); `tiktoken-rs 0.12` only if myelin ever drives OpenAI-hosted models directly. For deterministic token budgets before/after context assembly, `tokenizers` is what you want locally (it matches the served model's HF tokenizer).

---

## 9. Eval / benchmark infra

- **`divan` — latest 0.1.21** — https://docs.rs/divan/latest/divan/ — MIT/Apache-2.0, nvzqz. Macro-based `#[divan::bench]`, `divan::main()`, `harness = false`. Simpler/faster than criterion; recommended for micro-benchmarks.
- **`criterion` — latest 0.8.2** — https://docs.rs/criterion/latest/criterion/ — Apache-2.0/MIT. Statistical, plots, async bencher (`AsyncBencher`), regression detection. Heavier. **Use divan unless you want criterion's regression tracking.**
- **`insta` — latest 1.48.0** (2026-06-11) — https://docs.rs/insta/latest/insta/ and https://docs.rs/crate/insta/latest — Apache-2.0, mitsuhiko. `assert_snapshot!`/`assert_debug_snapshot!`/`assert_json_snapshot!`, inline snapshots, redactions, `cargo insta` review. For eval-trace/structured-output golden assertions.
- **`wiremock` — latest 0.6.5** — https://docs.rs/crate/wiremock/latest — Apache-2.0/MIT, LukeMathWalker. Async HTTP mock server, rich matchers, spying. Best for mocking the LLM/llama-swap endpoint and hs-distill HTTP in tests.
- **`mockito` — latest 1.7.2** — https://docs.rs/crate/mockito/latest — MIT, lipanski. Sync+async. Rejected in favor of wiremock (wiremock has extensible `Match` matchers + spying; mockito's matcher set is fixed).
- **`proptest` — latest 1.11.0** — https://docs.rs/proptest/latest/proptest/ — MIT/Apache-2.0. Property testing for fuzzy/token-budget/ranking invariants.
- **`cargo-mutants` — latest 27.1.0** — https://docs.rs/crate/cargo-mutants/latest — Apache-2.0 (source), mutants.rs. Mutation testing: injects bugs, sees which tests catch them. For eval-harness test-quality CI gate.
- **Structured trace capture:** `tracing` + `tracing-subscriber` (home-still pattern) with a test-layer that captures spans/events into memory for eval asserts; add `tracing-opentelemetry` only if a collector is wanted.
- **HF datasets offline (LoCoMo / LongMemEval):**
  - **`hf-hub` — latest 1.0.0** — https://docs.rs/hf-hub/latest/hf_hub/ — Apache-2.0 (HF). `HFClient::new()`, `client.dataset("owner","name").download_file().filename(..)` → cached path; `blocking` feature for sync. Downloads the LoCoMo/LongMemEval JSON/Parquet files once.
  - **`parquet` — latest 59.3.0** (arrow workspace) — https://docs.rs/parquet/latest/parquet/ — Apache-2.0. `ParquetRecordBatchReaderBuilder::new(path)` and `arrow::array::RecordBatch`; async via `ParquetRecordBatchStreamBuilder`.
  - **`arrow` — latest 59.3.0** (2026-09-01) — https://docs.rs/crate/arrow/latest — Apache-2.0, apache/arrow-rs. Columnar `RecordBatch`, `StringArray`, JSON reader. **59.3.0 pairs with parquet 59.3.0** (same workspace versioning).

**Recommendation:** `divan 0.1`, `insta 1.48`, `proptest 1.11`, `wiremock 0.6.5`, `cargo-mutants 27`; `hf-hub 1.0` + `parquet 59.3` + `arrow 59.3` for offline LoCoMo/LongMemEval loading; `tracing` test-layer (not opentelemetry) for eval trace capture unless a real collector is wanted.

---

## 10. Serving

- **`axum` — latest 0.8.9** (2026-04-14) — https://docs.rs/axum/latest/axum/ and https://docs.rs/crate/axum/latest — MIT, tokio-rs. 0.8 patterns: `Router::new().route("/", get(handler))`, extractors `Path/Query/Json`, `State`+`FromRef` substates, `IntoResponse`, `axum::serve(listener, app)`. **0.8 is the current line; do not use 0.7.** Middleware = `tower`/`tower-http` via `Router::layer`/`.route_layer`.
- **`tower`** — 0.5.2 (axum 0.8.9 dep). Timeouts, tracing, retry, compression layers; shared with hyper/tonic. Use `tower-http` (`TraceLayer`, `CorsLayer`, `CompressionLayer`) for myelin-mcp's HTTP service (rmcp's `StreamableHttpService` is a `tower::Service`).
- **`tracing-opentelemetry` — latest 0.33.0** (2026-05-18) — https://docs.rs/crate/tracing-opentelemetry/latest — MIT, tokio-rs; pairs with `opentelemetry 0.32`. `OpenTelemetryLayer` bridges `tracing` spans → OTLP. **Only if** myelin needs a real distributed trace collector (Jaeger/OTel). Otherwise plain `tracing` + `tracing-subscriber` (env-filter) matches home-still and is lighter.

**Recommendation:** `axum 0.8` (`features=["multipart"]` per home-still), `tower-http 0.6`, `tracing` + `tracing-subscriber`; add `tracing-opentelemetry 0.33` only behind a feature when a collector is present.

---

## Dependency decision table

| Capability | Chosen crate (version) | Why | Rejected alternative (reason) |
|---|---|---|---|
| MCP server/client | `rmcp` 3.3.0 (docs.rs/crate/rmcp/latest) | Official SDK; spec 2026-07-28; `#[tool_router]`/`#[tool]`/`#[prompt]` macros; stdio+Streamable HTTP; resources/prompts/sampling/elicitation/progress all supported | `rmcp` 1.3 in home-still (outdated, breaking → 3.0; 1→3 changes API) |
| Vector store client | `qdrant-client` 1.19.0 (docs.rs/qdrant-client/latest) | Matches big's Qdrant 1.19; named+sparse+multivector; `query_points` prefetch; server-side RRF/DBSF; scroll/snapshots/payload indexes | `qdrant-client` 1.17.0 in home-still (older builders; server 1.19 features need 1.19 client) |
| Sparse/lexical BM25 | `tantivy` 0.26.2 (docs.rs/tantivy/latest) | Real BM25 scores, Lucene-style index, pure-Rust, CPU | `probly-search` 2.0.1 (no BM25 scores — union/heuristic only) |
| BM25-as-sparse-vectors | `bm25` 2.3.2 (docs.rs/bm25/latest) | Emits `(index,value)` sparse vectors feedable to Qdrant for server-side fusion | `probly-search` (same rejection); tantivy kept for full index |
| Local embeddings + rerank | `fastembed` 6.1.0 (docs.rs/fastembed/latest) + `ort` =2.0.0-rc.13 | Local BGE-M3 dense+sparse+ColBERT + bge/jina rerankers, ort CUDA on big, Accelerate on Mac | `candle` 0.11 (DIY cross-encoder reranker, heavy wiring; fastembed already ships `TextRerank`) |
| Embedding fallback | plain `reqwest` 0.13 → hs-distill(7434)/ollama bge-m3 | Reuse home-still/ollama embedding without bundling ort | `ollama-rs` for embeddings (no ColBERT/rerank exposure) |
| LLM generation client | `async-openai` 0.42.0 (docs.rs/async-openai/latest) | OpenAI-compatible base URL → llama-swap /v1; typed `ResponseFormat::JsonSchema` + typed tool-calling | `ollama-rs` 0.3.6 (no native OpenAI json_schema, weaker tools) |
| Relational/graph/audit store | `sqlx` 0.9.0 (docs.rs/sqlx/latest) | Async; SQLite now + Postgres-ready; FTS5 & recursive CTE via plain SQL; compile-time checks (optional) | `sled` (stable unmaintained since 2021), `native_db` 0.8 (unstable API), redb+rusqlite (no SQL/CTE unification) |
| Embedded-KV (optional) | `redb` 4.2.0 (docs.rs/redb/latest) | Fast pure-Rust ACID KV if a raw KV cache is needed | `sled` (rejected above) |
| In-memory graph | `petgraph` 0.8.3 (docs.rs/petgraph/latest) | DiGraph + Dijkstra/BFS/DFS + ~30-line power-iteration PPR | no dedicated PPR crate exists → implement or use SQLite CTE |
| Token accounting | `tokenizers` 0.23.2 (docs.rs/tokenizers/latest) | Exact local `tokenizer.json` (qwen3/llama/gpt-oss) | `tiktoken-rs` 0.12 only for OpenAI-hosted (not local qwen/llama) |
| Micro-benchmark | `divan` 0.1.21 (docs.rs/divan/latest) | Macro ergonomics, fast, `harness=false` | `criterion` 0.8.2 (heavier; only if regression baselines needed) |
| Snapshot tests | `insta` 1.48.0 (docs.rs/crate/insta/latest) | Golden asserts for eval traces/structured output, `cargo insta review`, redactions | — |
| HTTP mocking | `wiremock` 0.6.5 (docs.rs/wiremock/latest) | Extensible `Match` matchers + spying; async LLM/endpoint mocking | `mockito` 1.7.2 (fixed matcher set, no extensible Match) |
| Property testing | `proptest` 1.11.0 (docs.rs/proptest/latest) | Invariant testing for token-budget/ranking logic | — |
| Mutation testing | `cargo-mutants` 27.1.0 (docs.rs/crate/cargo-mutants/latest) | Eval-harness test-quality CI gate (mutants.rs) | — |
| HF datasets offline | `hf-hub` 1.0.0 (docs.rs/hf-hub/latest) + `parquet`/`arrow` 59.3.0 | Downloads LoCoMo/LongMemEval once; reads JSON/Parquet as RecordBatches | Python HF loader (external runtime dependency) |
| Serving | `axum` 0.8.9 (docs.rs/axum/latest) + `tower`/`tower-http` 0.6 | 0.8 patterns; rmcp StreamableHttpService is a tower Service | axum 0.7 line (obsolete; 0.8 current) |
| Observability/trace | `tracing` + `tracing-subscriber` (0.1/0.3), optional `tracing-opentelemetry` 0.33.0 | Matches home-still; light; structured capture | opentelemetry-only path (overkill absent a collector) |

---

### Cross-platform notes

- **Works on both macOS arm64 and Linux CUDA:** `rmcp`, `qdrant-client`, `tantivy`, `bm25`, `sqlx`, `rusqlite`, `redb`, `petgraph`, `tokenizers`, `tiktoken-rs`, `divan`, `insta`, `wiremock`, `proptest`, `cargo-mutants`, `parquet`, `arrow`, `hf-hub`, `axum`, `tracing`-family, `reqwest`. All pure-Rust or with per-platform backends.
- **CUDA-specific (Linux big only; never macOS):** `fastembed`+`ort` `cuda` feature (RTX 3090). On macOS arm64, `ort` runs CPU/Accelerate; do not enable `fastembed/cuda` there. Gate via home-still-style `cuda = ["server", "ort/cuda"]`.
- **MSRV:** Rust 1.88 (rmcp 3.0), 1.85 (tiktoken-rs 0.12), 1.80 (divan 0.1), 1.64 (petgraph). Workstation Rust 1.98.1 satisfies all.

---

### Source URLs (crate versions)

- rmcp 3.3.0: https://docs.rs/crate/rmcp/latest ; spec + API: https://raw.githubusercontent.com/modelcontextprotocol/rust-sdk/main/README.md ; breaking: https://github.com/modelcontextprotocol/rust-sdk/discussions/969
- qdrant-client 1.19.0: https://docs.rs/qdrant-client/latest/qdrant_client/ ; fusion (RRF/DBSF/multivectors): https://qdrant.tech/documentation/search/hybrid-queries/index.md ; vectors/multivectors/max_sim: https://qdrant.tech/documentation/manage-data/vectors/index.md
- tantivy 0.26.2: https://docs.rs/tantivy/latest/tantivy/ ; bm25 2.3.2: https://docs.rs/bm25/latest/bm25/ ; probly-search 2.0.1: https://docs.rs/probly-search/latest/probly_search/
- fastembed 6.1.0: https://docs.rs/fastembed/latest/fastembed/ ; models: https://docs.rs/fastembed/latest/fastembed/enum.Bgem3Model.html , enum.SparseModel.html , enum.RerankerModel.html ; features/cuda: https://docs.rs/crate/fastembed/latest/features ; ort =2.0.0-rc.13: https://docs.rs/ort/latest/ort/ ; candle 0.11: https://docs.rs/candle-core/latest/candle_core/
- ollama-rs 0.3.6: https://docs.rs/ollama-rs/latest/ollama_rs/ ; async-openai 0.42.0: https://docs.rs/async-openai/latest/async_openai/ ; reqwest: https://docs.rs/reqwest/latest/reqwest/
- sqlx 0.9.0: https://docs.rs/sqlx/latest/sqlx/ ; rusqlite 0.40.2: https://docs.rs/rusqlite/latest/rusqlite/ ; redb 4.2.0: https://docs.rs/redb/latest/redb/ ; sled: https://docs.rs/crate/sled/latest ; native_db 0.8.2: https://docs.rs/crate/native_db/latest
- petgraph 0.8.3: https://docs.rs/crate/petgraph/latest
- tiktoken-rs 0.12.0: https://docs.rs/crate/tiktoken-rs/latest ; tokenizers 0.23.2: https://docs.rs/crate/tokenizers/latest
- divan 0.1.21: https://docs.rs/divan/latest/divan/ ; criterion 0.8.2: https://docs.rs/criterion/latest/criterion/ ; insta 1.48.0: https://docs.rs/crate/insta/latest ; wiremock 0.6.5: https://docs.rs/crate/wiremock/latest ; mockito 1.7.2: https://docs.rs/crate/mockito/latest ; proptest 1.11.0: https://docs.rs/proptest/latest/proptest/ ; cargo-mutants 27.1.0: https://docs.rs/crate/cargo-mutants/latest ; hf-hub 1.0.0: https://docs.rs/hf-hub/latest/hf_hub/ ; parquet 59.3.0: https://docs.rs/parquet/latest/parquet/ ; arrow 59.3.0: https://docs.rs/crate/arrow/latest
- axum 0.8.9: https://docs.rs/axum/latest/axum/ ; tracing-opentelemetry 0.33.0: https://docs.rs/crate/tracing-opentelemetry/latest

### Notes on method
- crates.io API returned HTTP 403 to direct fetches (user-agent-blocked), so all versions were taken from docs.rs crate/latest pages (which list published versions + dates) and docs.rs rustdoc JSON. Qdrant server-side claims are cited from Qdrant's own docs URLs (hybrid-queries + vectors pages).
- home-still conventions cited where relevant (workspace/Cargo.toml, hs-distill onnx.rs + config.rs, hs-mcp Cargo.toml) to anchor house style.
- Main agent confirmed this session's `write` tool is xd://-only, so the deliverable file target /Users/ladvien/myelin/docs/research/10-rust-ecosystem.md is represented by the `files` entry and the full body is here in `report` for Main to write to disk.
