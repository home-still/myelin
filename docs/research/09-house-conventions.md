# 09 — House Conventions (evidence pack for myelin)

Target: `/Users/ladvien/myelin/docs/research/09-house-conventions.md`. Source of truth: `/Users/ladvien/home-still` (Rust workspace, macOS arm64 workstation + Linux GPU host `big`). Secondary MCP idiom sources: `/Users/ladvien/bevy_brp/mcp`, `/Users/ladvien/auto_health/mcp`, `/Users/ladvien/apple_health_source/mcp`. Every version claim is verified against `/Users/ladvien/home-still/Cargo.lock`. All paths are relative to `/Users/ladvien/home-still` unless otherwise noted.

---

## 1. Workspace layout

### Members & resolver
`Cargo.toml` (workspace root):
```toml
[workspace]
members = ["crates/*", "hs-common", "paper"]
resolver = "2"
```
Cite: `Cargo.toml:3-4`. `hs-common` and `paper` are non-`crates/*` workspace members listed explicitly; every crate under `crates/*` is globbed in. `resolver = "2"`.

### `[workspace.dependencies]` (shared + pinned)
`Cargo.toml:7-30` (verbatim, features included):
```toml
[workspace.dependencies]
# Shared across workspace crates
reqwest = { version = "0.12", features = [
    "json",
    "rustls-tls-native-roots",
], default-features = false }
tokio = { version = "1", features = ["full"] }
clap = { version = "4.5", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
dirs = "6"
anyhow = "1"
thiserror = "2"
hs-common = { path = "hs-common" }
owo-colors = { version = "4.3", features = ["supports-colors"] }
serde_yaml_ng = "0.10"
figment = { version = "0.10", features = ["yaml", "env"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
dialoguer = "0.11"
async-trait = "0.1"
regex = "1"
axum = { version = "0.8", features = ["multipart"] }
qdrant-client = { version = "1", features = ["serde"] }
xxhash-rust = { version = "0.8", features = ["xxh3"] }
uuid = { version = "1", features = ["v4", "v5"] }
fastembed = "5"
chrono = { version = "0.4", features = ["serde"] }
```
Workspace crates reference these as `{ workspace = true }` (e.g. `crates/hs-distill/Cargo.toml:22-24`), and non-shared deps are pinned inline (e.g. `ort = { version = "=2.0.0-rc.11", optional = true }`, `crates/hs-distill/Cargo.toml:47`; `ollama-rs = { version = "0.3", features = ["stream"], optional = true }`, `crates/hs-distill/Cargo.toml:50`).

### Release profile
`Cargo.toml:32-36`:
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = "symbols"
```

### Edition / MSRV
Every crate uses `edition = "2021"` (e.g. `crates/hs-distill/Cargo.toml:4`, `crates/hs-mcp/Cargo.toml:4`, `hs-common/Cargo.toml:4`). **No `rust-version`/MSRV is declared anywhere** — a workspace-wide grep for `rust-version|MSRV` returns nothing in manifests; CI uses default stable via `actions-rs` toolchain (`ci.yaml`) and `components: rustfmt, clippy` (`ci.yaml:25`). Cargo.lock is `version = 4` (`Cargo.lock:3`).

### `build.rs` purpose
`crates/hs-distill/build.rs` (entire file, 31 lines) injects a `HS_VERSION` compile-time constant:
- `cargo:rerun-if-env-changed=GITHUB_REF_NAME` (`build.rs:4`) — forces the build script to re-run when CI sets a new tag, avoiding a stale `HS_VERSION` baked into dependent crates (documented in the comment, `build.rs:2-4`).
- Reads `GITHUB_REF_NAME`, strips a leading `v` (`build.rs:7-12`); else `git describe --tags --always` (`build.rs:12-21`); falls back to `CARGO_PKG_VERSION` (`build.rs:24`).
- Emits `cargo:rustc-env=HS_VERSION={version}` (`build.rs:29`).
Consumed via `env!("HS_VERSION")` in `crates/hs-distill/src/server.rs` health handler (`server.rs:75`).

---

## 2. Crate feature strategy

### hs-distill `[features]` (verbatim, `crates/hs-distill/Cargo.toml:6-17`)
```toml
[features]
default = ["client"]
client = []
server = [
    "dep:axum",
    "dep:fastembed",
    "dep:ort",
    "dep:qdrant-client",
    "dep:ollama-rs",
    "dep:regex",
    "dep:xxhash-rust",
    "dep:uuid",
    "dep:tokio-stream",
    "dep:tempfile",
    "dep:tracing-subscriber",
]
cuda = ["server", "ort/cuda"]
```
Pattern: `client` is the default (a pure HTTP client — no heavy server deps); `server` pulls every server-only optional dependency via the `dep:` syntax (auto-enabled when the matching `optional = true` dep is present). `cuda` is a thin feature that turns on `server` and forwards `ort/cuda`.

### Binary gating with `required-features`
`crates/hs-distill/Cargo.toml:18-22`:
```toml
[[bin]]
name = "hs-distill-server"
path = "src/server_main.rs"
required-features = ["server"]
```
The server binary only exists when the `server` feature is on (so `cargo build -p hs-distill` for the default client still compiles).

### Module-level gating in lib.rs
`crates/hs-distill/src/lib.rs:1-13`: always-on modules `chunker, cli, client, config, error, event_watch, quality, reconcile, types`; `#[cfg(feature = "server")]` modules `adaptive_batch, embed, metadata, pipeline, qdrant, server`.

### hs-common (shared, feature-gated)
`hs-common/Cargo.toml` shows the same `dep:` gating at larger scale: `cli`, `service`, `catalog`, `compose`, `auth`, `storage`, `storage-s3`, `events`, `events-nats`, `logging` — each feature name maps to `dep:<crate>,…` entries (e.g. `storage-s3 = ["storage", "dep:object_store", "dep:bytes", …]`, `hs-common/Cargo.toml:24-41`). Consumers enable exactly what they need, e.g. hs-mcp pulls `hs-common = { workspace = true, features = ["cli", "service", "catalog", "auth", "storage-s3", "logging"] }` (`crates/hs-mcp/Cargo.toml:30`) and hs-distill pulls `["service", "catalog", "storage-s3", "events-nats", "logging"]` (`crates/hs-distill/Cargo.toml:26`).

### Build with features
CUDA build for the GPU host: `GITHUB_REF_NAME=v0.0.1-rc.NNN cargo build --release -p hs-distill --features cuda` (`docs/deployment.md:527`).

---

## 3. Config: figment layering

`crates/hs-distill/src/config.rs` (read in full). Server load (`config.rs:96-110`):
```rust
pub fn load() -> Result<Self, Box<figment::Error>> {
    let home = dirs::home_dir().unwrap_or_default();
    let config_path = home.join(hs_common::CONFIG_REL_PATH);

    Figment::from(Serialized::defaults(Self::default()))
        .merge(Yaml::file(&config_path).nested())
        .merge(Env::prefixed("HS_DISTILL_"))
        .select("distill_server")
        .extract()
        .map_err(Box::new)
}
```
Layer order (lowest→highest precedence): `Serialized::defaults(Self::default())` → `Yaml::file(...).nested()` → `Env::prefixed("HS_DISTILL_")`. The YAML file is selected via `.nested()` and the whole config is one top-level key via `.select("distill_server")`. Client load (`config.rs:154-170`) selects `.select("distill")` and additionally extracts `.select("storage")` and `.select("events")` sub-configs by hand into `StorageConfig`/`EventBusConfig` fields marked `#[serde(skip)]`.

### Config file discovery path
`hs-common/src/lib.rs:9`: `pub const CONFIG_REL_PATH: &str = ".home-still/config.yaml";`. So the file is **`~/.home-still/config.yaml`**, resolved via `dirs::home_dir()`. `hs_common::resolve_project_dir()` (`hs-common/src/lib.rs:19-26`) reads `project_dir` out of that same file (default `~/home-still`). Example user config in `docs/deployment.md:459-483` (top-level keys `home:`, `storage:`, `distill_server:`, `scribe:`).

### Config struct shape (`config.rs:20-75`)
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DistillServerConfig {
    pub host: String,
    pub port: u16,
    pub qdrant_url: String,
    pub qdrant_data_dir: PathBuf,
    pub collection_name: String,
    pub embedding: EmbeddingConfig,
    pub chunk_max_tokens: usize,
    pub chunk_overlap: usize,
    pub qdrant_upsert_batch: usize,
    pub qdrant_upsert_parallelism: usize,
    pub llm_metadata: bool,
    pub metadata_model: String,
    pub ollama_url: String,
}
```
All structs carry `#[serde(default)]` + a `Default` impl so a partial file/env is fine. Server defaults (`config.rs:43-62`): `host "0.0.0.0"`, `port 7434`, `qdrant_url "http://localhost:6334"`, `qdrant_data_dir {project}/data/qdrant`, `collection_name "academic_papers"`, `chunk_max_tokens 1000`, `chunk_overlap 100`, `qdrant_upsert_batch 1000`, `qdrant_upsert_parallelism 4`, `llm_metadata false`, `metadata_model "llama3.2:latest"`, `ollama_url "http://localhost:11434"`.

`EmbeddingConfig` (`config.rs:77-117`): `model "bge-m3"`, `dimension 1024`, `batch_size None`, `pool_size None`, `adaptive_batch true`, `sparse_enabled true`. The `pool_size` doc comment (`config.rs:80-86`) explains `None` → per-device default (CUDA=1, CPU=min(concurrency,4)).

Env var prefix: **`HS_DISTILL_`** (server + client both). `HS_VERSION` is a separate compile-time env (from build.rs). S3 credentials use a different convention (`HS_S3_ACCESS_KEY` / `HS_S3_SECRET_KEY`, `docs/deployment.md:458,476-477`).

---

## 4. Error handling: thiserror vs anyhow

`crates/hs-distill/src/error.rs` (513 B, whole file):
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DistillError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Embedding error: {0}")]
    Embedding(String),
    #[error("Qdrant error: {0}")]
    Qdrant(String),
    #[error("Metadata extraction error: {0}")]
    Metadata(String),
    #[error("Config error: {0}")]
    Config(String),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}
```
Shape: one `thiserror` enum per crate (name `<Crate>Error`), `#[from]` for std/JSON/reqwest, plain `String` for domain errors (embedding/qdrant/metadata/config). thiserror is **2.x** (`workspace` dep `thiserror = "2"`, `Cargo.toml:19`; lock 2.0.18 at `Cargo.lock:5054`).

### Where each is used
- **thiserror** — any function returning a domain result: `Result<…, DistillError>` across `qdrant.rs`, `embed/*`, `chunker` callers. Library-layer typed errors.
- **anyhow** — the binary layer and the HTTP client: `server_main.rs` uses `anyhow::Result` / `anyhow::anyhow!` (e.g. `server_main.rs:34-52`), and `client.rs` returns `anyhow::Result` with `.context(...)` (e.g. `client.rs:107-121`). `cargo clippy --workspace --all-targets -- -D warnings` enforces no unused anyhow.

### Crossing the axum boundary
`server.rs` handlers return `impl IntoResponse` and convert errors to `(StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))` inline (e.g. `handle_exists` `server.rs:190-196`, `handle_status` `server.rs:99-105`, `handle_search` `server.rs:315-322`). Streaming NDJSON serializes errors as a `StreamLine::Error` JSON line (`server.rs:277-282`). There is **no custom axum `IntoResponse` impl for `DistillError`** — mapping is done per-handler via `format!("{e}")`.

---

## 5. Qdrant usage (`crates/hs-distill/src/qdrant.rs`, read in full 450 lines)

### Client construction
`server_main.rs:63-66`: `qdrant_client::Qdrant::from_url(&config.qdrant_url).timeout(Duration::from_secs(30)).build()` — `from_url(...)` then `.timeout(...)`, `.build()`; wraps `anyhow` on failure. gRPC port 6334 is the default URL.

### Collection creation (create-on-first-use)
`ensure_collection` (`qdrant.rs:47-102`): first `client.list_collections()` and check exists; if missing, `client.create_collection(
    CreateCollectionBuilder::new(name)
        .vectors_config(VectorParamsBuilder::new(dimension as u64, Distance::Cosine).on_disk(true))
        .hnsw_config(HnswConfigDiffBuilder::default().m(0))
        .on_disk_payload(true))` (`qdrant.rs:78-84`). Key params: **vector size** = embedding dimension (1024), **distance** = `Distance::Cosine`, **HNSW `m(0)`** to disable HNSW during bulk load (comment `qdrant.rs:72`), **on-disk vectors** (`on_disk(true)`) and **on-disk payload** (`on_disk_payload(true)`). **No quantization, no sparse/named vectors** on creation.

### Payload indexes
`create_indexes` (`qdrant.rs:104-132`): keyword fields `["doc_id", "authors", "topics", "keywords", "pdf_path"]` → `FieldType::Keyword`; integer fields `["year", "line_start", "page"]` → `FieldType::Integer`; `"title"` → `FieldType::Text` (full-text). Built with `CreateFieldIndexCollectionBuilder::new(collection, field, FieldType::…)`.

### Point id derivation (`deterministic_id`, `qdrant.rs:30-38`)
```rust
const NAMESPACE_UUID: Uuid = Uuid::from_bytes([0x6b,0xa7,0xb8,0x10,0x9d,0xad,0x11,0xd1,0x80,0xb4,0x00,0xc0,0x4f,0xd4,0x30,0xc8]);
pub fn deterministic_id(doc_id: &str, chunk_index: u32) -> String {
    let hash = xxhash_rust::xxh3::xxh3_64(format!("{}:{}", doc_id, chunk_index).as_bytes());
    Uuid::new_v5(&NAMESPACE_UUID, &hash.to_le_bytes()).to_string()
}
```
Deterministic: `xxhash-rust` **xxh3_64** over `"{doc_id}:{chunk_index}"`, then **UUID v5** under a fixed namespace. Stability is unit-tested (`qdrant.rs:397-411`). `uuid` is enabled with `v4 + v5` in workspace deps (`Cargo.toml:28`).

### Upsert batching
`upsert_chunks` (`qdrant.rs:135-173`): builds `Vec<PointStruct>` via `PointStruct::new(point_id, ec.embedding.dense.clone(), payload)`, then `client.upsert_points(UpsertPointsBuilder::new(collection_name, points))`. Batch sizing is configured: `qdrant_upsert_batch: 1000` chunks per request, `qdrant_upsert_parallelism: 4` concurrent requests per doc (config defaults `config.rs:59-60`; rationale comment `config.rs:31-40`). Payload is `serde_json::json!({...}).try_into::<qdrant_client::Payload>()` with keys `doc_id, chunk_index, chunk_text, title, authors, doi, year, topics, keywords, pdf_path, markdown_path, line_start, line_end, page, cited_by_count` (`qdrant.rs:151-162`).

### Filter construction
`build_filter(year, topic) -> Option<Filter>` (`qdrant.rs:169-181`) uses `Filter::must(conditions)` with `Condition::matches("topics", s)` (keyword) and `Condition::range("year", Range { gte/gt/lte/lt })`; exact year match uses `Condition::matches("year", i64)`. Doc existence filter: `Filter::must([Condition::matches("doc_id", id)])` (`qdrant.rs:300`).

### Search API shape (qdrant-client 1.17)
`search` (`qdrant.rs:147-168`):
```rust
let mut builder = QueryPointsBuilder::new(collection).query(qdrant_client::qdrant::Query::from(query_vector)).limit(limit).with_payload(true).params(SearchParamsBuilder::default().hnsw_ef(128));
if let Some(f) = filter { builder = builder.filter(f); }
let results = client.query(builder).await?;
Ok(results.result)  // Vec<ScoredPoint>
```
Consumers read `point.score` and `point.payload` via typed accessors `payload.get("...").as_str()/as_list()/as_integer()` (`server.rs:330-378`). Other builder surface used: `CountPointsBuilder` + `.filter().exact(true)` (`qdrant.rs:299-305`), `DeletePointsBuilder::new(c).points(filter)` (`qdrant.rs:311-312`), `FacetCountsBuilder::new(c, "doc_id").limit(n).exact(true)` for distinct-doc enumeration (`qdrant.rs:325-348`), `client.collection_info(name)` (`qdrant.rs:245-247`), `client.delete_collection(name)` (`qdrant.rs:262`).

### Version / API surface
qdrant-client **1.17.0** (`Cargo.lock:3757-3758`), workspace dep `qdrant-client = { version = "1", features = ["serde"] }`. Builder pattern names observed: `CreateCollectionBuilder`/`VectorParamsBuilder`/`HnswConfigDiffBuilder`/`CreateFieldIndexCollectionBuilder`/`PointStruct`/`UpsertPointsBuilder`/`QueryPointsBuilder`/`SearchParamsBuilder`/`CountPointsBuilder`/`DeletePointsBuilder`/`FacetCountsBuilder`.

---

## 6. Embedding

### Trait + device detection (`embed/mod.rs`)
`ComputeDevice { Cpu, Cuda }` with `Display` (`embed/mod.rs:8-24`). `Embedder` trait (`embed/mod.rs:29-36`): `async fn embed_batch(&[String]) -> Result<Vec<EmbeddingOutput>, DistillError>`, `fn dimension()`, `fn supports_sparse()`, `fn device()`. `detect_device()` (`embed/mod.rs:47-66`) shells out to `nvidia-smi --query-gpu=name --format=csv,noheader`; success + non-empty → Cuda else Cpu. `FallbackEmbedder` (`embed/mod.rs:71-121`) is **no-silent-fallback**: comment at `embed/mod.rs:71` and build logic — if CUDA is requested and the probe fails, startup errors (`“no silent CPU fallback”`), no substitute.

### fastembed/ort setup (`embed/onnx.rs`)
`build_text_embedding` (`onnx.rs:96-103`):
```rust
let mut opts = InitOptions::new(EmbeddingModel::BGEM3).with_show_download_progress(true);
if matches!(device, ComputeDevice::Cuda) {
    use ort::execution_providers::CUDAExecutionProvider;
    opts = opts.with_execution_providers(vec![CUDAExecutionProvider::default().build()]);
}
TextEmbedding::try_new(opts).map_err(|e| DistillError::Embedding(...))
```
- **Model**: `EmbeddingModel::BGEM3` (fastembed enum), config `model: "bge-m3"`, **dimension 1024** (`config.rs:117`).
- **CUDA EP**: `ort::execution_providers::CUDAExecutionProvider::default().build()`. ort is pinned `=2.0.0-rc.11` (`crates/hs-distill/Cargo.toml:47`; lock `Cargo.lock:3321-3322`).
- **CUDA verification probe** (`verify_cuda_probe`, `onnx.rs:106-127`): runs one embedding, measures wall-clock, then reads `nvidia-smi --query-gpu=memory.used`; if `< 200` MB VRAM used → `Err` (documented at `onnx.rs:117-121`).
- **Model pool** (`OnnxEmbedder::new`, `onnx.rs:34-66`): CUDA hosts build **1** instance; CPU hosts build `min(HardwareProfile::distill_concurrency, 4)`. Round-robin pick via `AtomicUsize` `next` (`onnx.rs:161-162`).

### Batch sizing strategy
Initial batch size: CPU=8, CUDA=32 (`onnx.rs:39-42`), then an **adaptive EWMA hill-climber** (`adaptive_batch.rs`) unless `adaptive_batch: false` pins it. `AdaptiveConfig::default_for_device` (`adaptive_batch.rs:45-69`): candidates **CUDA `[16,32,48,64,96,128]`**, **CPU `[4,8,12,16,24]`**; thresholds `improvement_threshold 1.05`, `regression_threshold 0.90`, `converge_after_stable 3`, `ewma_alpha 0.2`, `sample_interval 50`. `observe()` updates EWMA rate and steps/regresses/converges; `Decision::{Noop,Stepped,Reverted,Converged}` (`adaptive_batch.rs:83-95`). Embedding runs inside `tokio::task::spawn_blocking` because fastembed `embed` is sync (`onnx.rs:167-188`), looped `step_by(batch_size)`.

### Dense-only vs sparse/colbert
**Dense-only in hs-distill.** `embed_batch` returns `EmbeddingOutput { dense: Vec<f32>, sparse: None }` (`onnx.rs:193-198`), and `supports_sparse()` returns `false` (`onnx.rs:204`). Although `EmbeddingConfig.sparse_enabled` exists (`config.rs:114`) it is not wired into the ONNX path; the Qdrant `PointStruct` upserts only `ec.embedding.dense` (`qdrant.rs:158`). (`SparseVec` type exists in `types.rs:59-66` but is unused in the server path.) The separate `paper_abstracts` collection / other MCP tools are outside this crate.

---

## 7. Chunking (`crates/hs-distill/src/chunker.rs`)

`ChunkerConfig` (`chunker.rs:8-28`): `max_tokens 1000`, `overlap_tokens 100`, `chars_per_token 4`. So **max_chars = 4000**, **overlap_chars = 400** (`chunker.rs:53-54`).

Algorithm (`chunk_markdown`, `chunker.rs:55-126`):
1. Split the markdown into **page segments** at the separator `

---

` (`PAGE_SEPARATOR`, `chunker.rs:46`).
2. For each segment, `split_segment` (`chunker.rs:157-202`): if `len <= max_chars` → one trimmed chunk; else slide a window and **snap to sentence boundaries backwards** (`find_sentence_boundary`, `chunker.rs:205-224`): prefer `

` paragraph break, else `. `, `? `, `! `, searching back `(end-start)/5` (20%). Byte offsets snap to char boundaries (`snap_to_char_boundary`, `chunker.rs:139-149`).
3. Advance with overlap (`actual_end - overlap_chars`), forcing progress if stuck (`chunker.rs:192-202`).
4. Track a **CCH header** prepended for embedding quality (`chunker.rs:113-116`): `format!("{title} > chunk {n}\n\n{chunk_text}")` where `title` falls back to `doc_id`.

Metadata attached **per chunk** (`types.rs:23-50`, `Chunk`): `doc_id`, `chunk_index`, `total_chunks`, `text` (with CCH header), `raw_text` (without header), `span: ChunkSpan`, `page: Option<usize>`, `meta: DocumentMeta`. `ChunkSpan { line_start, line_end (1-based), char_start, char_end (byte offsets) }` (`types.rs:4-13`). Line numbers come from `build_line_offsets` + `byte_to_line` binary search (`chunker.rs:31-44`); page comes from `resolve_page` over catalog `PageOffset`s (`chunker.rs:37-43`).

---

## 8. MCP server (`crates/hs-mcp/src/main.rs`, ~120 KB — sampled)

### rmcp 1.3 idioms
- **Manifest** (`crates/hs-mcp/Cargo.toml:33-38`): `rmcp = { version = "1.3", features = ["server", "transport-io", "transport-streamable-http-server"] }`. Lock: **rmcp 1.3.0** (`Cargo.lock:4235-4237`).
- **Imports** (`main.rs:6-27`): `rmcp::handler::server::router::{prompt::PromptRouter, tool::ToolRouter}` and `wrapper::Parameters`; `rmcp::model::{ErrorData, …, ServerCapabilities, ServerInfo, RawResource, RawResourceTemplate, …}`; macros `prompt, prompt_handler, prompt_router, tool, tool_handler, tool_router`; `service::RequestContext`, `RoleServer`, `ServerHandler`; plus `schemars` for `JsonSchema`.
- **Parameter structs** (`main.rs:29-45` and throughout): `#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]` with `#[schemars(description = "…")]` per field (see `PaperSearchParams` `main.rs:29-58`).
- **Router + tools** (`main.rs:349-350`): `#[tool_router] impl HomeStillMcp {` … Each tool:
```rust
#[tool(description = "Search academic papers across 6 providers …",
      annotations(read_only_hint = true, …))]
async fn paper_search(&self, Parameters(p): Parameters<PaperSearchParams>) -> Result<String, String> {
…
}
```
(verbatim `main.rs:353-365`). Returns `Result<String, String>` (JSON text or an error string); rmcp maps `Err(String)` to a tool error. Representative tools: `paper_search` (`main.rs:353`), `paper_get` (`main.rs:408`), `paper_download` (`main.rs:431`), `catalog_list` (`main.rs:569`), `catalog_read` (`main.rs:749`), `distill_search` (`main.rs:1876`), `scribe_convert` (`main.rs:1724`), `system_status` (`main.rs:2402`).
- **`RequestContext`** is an optional extra param on mutating tools: `context: RequestContext<RoleServer>` (`main.rs:1736`).
- **Handler wiring** (`main.rs:2685-2688`):
```rust
#[tool_handler(router = self.tool_router)]
#[prompt_handler(router = self.prompt_router)]
impl ServerHandler for HomeStillMcp { … }
```
- **ServerInfo / capabilities / instructions** (`main.rs:2689-2742`): `get_info` returns `ServerInfo::new(ServerCapabilities::builder().enable_tools().enable_resources().enable_prompts().build()).with_instructions("home-still: Academic research pipeline server.\n\nFull pipeline workflow:\n1. DISCOVER…\n… Prompts: research_paper, summarize_document, compare_papers")` — the instructions are a long plain-prose string passed through `with_instructions`. (That is the string surfaced by the mounted home-still server in this session.)
- **Prompts** (`main.rs:2629-2682`): `#[prompt(description = "…")] fn name(&self, Parameters(p): Parameters<XPromptParams>) -> Vec<PromptMessage>` returning `vec![PromptMessage::new_text(PromptMessageRole::User, format!(…))]`.

### Transport selection via clap
`main.rs:2883-2895`:
```rust
#[derive(Parser)]
#[command(name = "hs-mcp")]
struct Args {
    /// Run as HTTP/SSE server on this address (default: stdio mode)
    #[arg(long)]
    serve: Option<String>,
}
```
- **stdio mode** (default when `--serve` absent): `let transport = rmcp::transport::io::stdio(); let ct = rmcp::service::serve_server(server, transport).await?; let _ = ct.waiting().await;` (`main.rs:2934-2938`). In stdio mode stdout is MCP protocol ⇒ logging is disabled (`StderrOutput::Disabled`, `main.rs:2971-2973`) and no human-readable lines go to stderr.
- **streamable HTTP mode** (`main.rs:2907-2930`): `rmcp::transport::streamable_http_server::{session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService}`; session `keep_alive = None` (dashboard keeps SSE open — `main.rs:2913-2914`); `StreamableHttpService::new(move || Ok(server.clone()), Arc::new(session_manager), StreamableHttpServerConfig::default())`; `axum::Router::new().fallback_service(service)`; `tokio::net::TcpListener::bind(&addr)`; `axum::serve(listener, router).with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); })` (`main.rs:2923-2926`). MCP-over-HTTP is port **7445** (`docs/deployment.md:101`).

### Resources (`catalog:///{stem}`, `markdown:///{stem}`)
- **list_resources** (`main.rs:2748-2796`): iterates catalog triples → `RawResource { uri: format!("catalog:///{stem}"), name: title, mime_type: Some("application/yaml"), … }.no_annotation()`, and markdown objects → `uri: format!("markdown:///{stem}"), mime_type: "text/markdown", size: Some(obj.size)`. `ListResourcesResult::with_all_items(resources)`.
- **list_resource_templates** (`main.rs:2799-2850`): `RawResourceTemplate` for `catalog:///{stem}`, `markdown:///{stem}`, and `markdown:///{stem}/page/{page}`.
- **read_resource** (`main.rs:2852-2940`): strips `catalog:///` → `hs_common::catalog::read_catalog_entry_via(…)`; strips `markdown:///`, optionally parses trailing `/page/{n}` (1-based, splits on `

---

`). Errors via `ErrorData::resource_not_found(...)`, `ErrorData::internal_error(...)`, `ErrorData::invalid_params(...)`, returning `ReadResourceResult::new(vec![ResourceContents::TextResourceContents { uri, mime_type, text, meta: None }])`.

### Error mapping to MCP
Tools return `Result<String, String>` (stringified JSON or error message — rmcp turns `Err` into a tool error). Resource handlers return `Result<_, ErrorData>` using `ErrorData::{resource_not_found, internal_error, invalid_params}` (see `read_resource`, `main.rs:2870-2920`).

### Secondary MCP idioms (different shapes worth noting)
- **bevy_brp** (`/Users/ladvien/bevy_brp/mcp`): manual `ServerHandler` with **no `#[tool]` macros** — overrides `list_tools`/`call_tool`, dispatches by name (`mcp_service.rs:120-140`), returns `Result<CallToolResult, McpError>`, errors via `McpError::invalid_params/internal_error`; builds tool schemas from a hand-rolled `ParameterBuilder` returning `Arc<Map>` (`tool/parameters.rs:75-150`); stdio-only via `rmcp::transport::stdio` + `ServiceExt::serve(stdio()).waiting()` (`main.rs:26-34`).
- **auto_health** (`/Users/ladvien/auto_health/mcp/src/main.rs`): rmcp transport selected from config `Transport::{Stdio,Http}`; stdio path `svc.serve(stdio())` then `service.waiting()` (`main.rs:45-50`); HTTP path mounts `StreamableHttpService` under `axum::Router::new().nest_service("/mcp", service)` with `CancellationToken` + `.with_cancellation_token()`, empty `allowed_hosts`, graceful shutdown `ctrl_c` → `ct.cancel()` (`main.rs:60-92`); logging via `EnvFilter::try_from_env("RUST_LOG")` writing to stderr in stdio mode.
- **apple_health_source** (`/Users/ladvien/apple_health_source/mcp/src/main.rs`): struct-based `#[tool]` derive: `#[tool(description = "…")] async fn name(&self, #[tool(aggr)] params: X) -> String` and thin `reqwest::Client` + Bearer token to a backend HTTP API; large prompt-injected schema in the param `description`.

For myelin, the house standard to clone first is **hs-mcp** (derive-macro style), with the manual `ServerHandler`/`nest_service("/mcp")` patterns available as needed.

---

## 9. Service / daemon patterns

### axum app (`crates/hs-distill/src/server.rs`)
`app(state: Arc<DistillServerState>) -> Router` (`server.rs:28-45`) builds:
- Routes: `POST /distill`, `POST /distill/stream` (NDJSON), `POST /search`, `GET /health`, `GET /readiness`, `GET /status`, `GET /exists/{doc_id}`, `DELETE /doc/{doc_id}`, `GET /docs`, `POST /collection/reset`.
- `.layer(DefaultBodyLimit::max(256 * 1024 * 1024))` (`server.rs:44`).
- `.with_state(state)` where `DistillServerState { embedder: Arc<FallbackEmbedder>, qdrant: Arc<qdrant_client::Qdrant>, config: DistillServerConfig, in_flight: Arc<AtomicUsize> }` (`server.rs:20-26`).

### Health / readiness
- **/health** (`server.rs:50-61`): calls `state.qdrant.health_check()`, returns `Json(HealthResponse { status: "ok", compute_device, collection, version: env!("HS_VERSION"), qdrant_version, embed_model, qdrant_url })`.
- **/readiness** (`server.rs:63-70`): `Json(ReadinessResponse { ready: true, in_flight })`. Client implements `ReadinessInfo` (`client.rs:76-84`) with `available_slots()`.
- In-flight guard: `hs_common::service::inflight::InFlightGuard` (RAII, `server.rs:7,157,217`).

### Streaming
NDJSON via `tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(16)`, `Body::from_stream(ReceiverStream::new(rx))`, `header::CONTENT_TYPE = "text/x-ndjson"` (`server.rs:216-290`); progress events serialized as `hs_common::service::protocol::StreamLine::{Progress, Result, Error}`.

### Graceful shutdown
MCP server uses `axum::serve(listener, router).with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); })` (`main.rs:2923-2926`). The distill server binary is simpler: `axum::serve(listener, server::app(state)).await` then shuts the logging handle (`.shutdown().await`) (`server_main.rs:75-79`).

### Entrypoint bootstrap
`server_main.rs:36-40`: `hs_common::service::lib_bootstrap::ensure_lib_paths_or_reexec()` **must run before any dlopen/tokio init** (re-execs self with augmented `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` so ort’s CUDA provider / pdfium load; `hs-common/src/service/lib_bootstrap.rs`, CUDA-12 libs at `~/.home-still/cuda12-libs/` per `docs/deployment.md:532-548`). Then `tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(async_main())` (`server_main.rs:41-44`). `hs_common::secrets::load_default_secrets()` at top of `main` loads `~/.home-still/secrets.env` (`hs-common/src/secrets.rs:1-17`).

### Systemd user/system units (how services are invoked)
`docs/deployment.md:501-516` — `hs serve <name> --install` drops a systemd unit (Linux) or LaunchAgent (macOS). System services run under the invoking user, e.g. `hs-serve-distill.service` (native bare binary, CUDA). Reference unit shape (`docs/deployment.md:203-218`):
```ini
[Unit]
Description=home-still cloud gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<your-user>
ExecStart=/home/<your-user>/.local/bin/hs-gateway --gateway-url https://cloud.example.com
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
User-scoped watcher services run under `systemctl --user` (`hs-scribe-watch-events.service`, `docs/deployment.md:510-511`; restarted via `systemctl --user restart hs-scribe-watch-events.service`, `deployment.md:897`). Build+install on the GPU host: `GITHUB_REF_NAME=v0.0.1-rc.NNN cargo build --release -p hs-distill --features cuda`, then `install -m755 target/release/{hs,hs-mcp,hs-scribe-server,hs-distill-server} ~/.local/bin/` (`deployment.md:524-528`).

---

## 10. Testing conventions

- **Unit tests** are inline `#[cfg(test)] mod tests` inside each source module, not a separate `tests/` tree: `chunker.rs:201`, `adaptive_batch.rs:255`, `event_watch.rs:250`, `metadata.rs:126`, `qdrant.rs:396`, `quality.rs:126`, `reconcile.rs:116`, `hs-mcp/src/main.rs:2993` (`provider_arg_tests`).
- **One integration test dir** exists: `crates/hs-scribe/tests/end_to_end_test.rs` (the only `tests/` dir in `crates/*`).
- **CI** (`.github/workflows/ci.yaml`): `cargo fmt … --check`, `cargo clippy --workspace --exclude hs-scribe --all-targets -- -D warnings`, `cargo clippy -p hs-scribe --features server --all-targets -- -D warnings`, `cargo test --workspace --exclude hs-scribe`, `cargo test -p hs-scribe --features server` (lines 29-36). No separate test crate; feature-gated server code is exercised via `--features server`.
- **E2E** (`.github/workflows/e2e.yaml`): weekly/manual workflow that boots the real scribe docker image + Ollama, POSTs a ghostscript-generated PDF to `/scribe`, and asserts non-empty markdown — a smoke/integration check, not unit.
- **Fixtures**: no checked-in fixture corpus for distill; tests build inputs inline (e.g. `"Page one content.\n\n---\n\nPage two content."` in `chunker.rs:221-226`). The scribe E2E generates a one-page PDF at CI time.
- **cargo mutants**: **used but not configured in-repo.** There is **no `mutants.toml` anywhere** (verified via glob), but `.gitignore` in the workspace root, `hs-common/`, and `paper/` each contain the RustRover-generated `**/mutants.out*/` block, and the myelin root `.gitignore` has the same `**/mutants.out*/` (`/Users/ladvien/myelin/.gitignore:13-14`). So `cargo mutants` is run locally with its output dir ignored, using defaults (no committed config). Follow the same convention: keep `**/mutants.out*/` in `.gitignore`, no `mutants.toml`.

---

## Template for myelin

Proposed manifests consistent with everything above. `myelin-core` mirrors `hs-distill` (feature-gated server/embed/qdrant), `myelin-mcp` mirrors `hs-mcp` (rmcp 1.3 derive + stdio/HTTP), `myelin-eval` is a client+CLI bin (anyhow/reqwest/clap). All shared versions are taken from `/Users/ladvien/home-still/Cargo.lock` (verified: rmcp 1.3.0 `Cargo.lock:4236`, qdrant-client 1.17.0 `:3758`, fastembed 5.13.2 `:1308`, ort 2.0.0-rc.11 `:3322`, figment 0.10.19 `:1366`, uuid 1.23.0 `:5677`, xxhash-rust 0.8.15 `:6314`, thiserror 2.0.18 `:5054`, anyhow 1.0.102 `:119`, tokio 1.50.0 `:5197`, reqwest 0.12.28 `:4167`, serde 1.0.228 `:4520`, serde_json 1.0.149 `:4561`, clap 4.6.0 `:581`, async-trait 0.1.89 `:218`, chrono 0.4.44 `:567`, tracing 0.1.44 `:5390`, axum 0.8.8 `:320`, schemars 1.2.1 `:4425`). `dirs 6`, `regex 1`, `tracing-subscriber 0.3`, `tokio-stream 0.1`, `tempfile 3` are the home-still declared minors (workspace/`crates/hs-distill/Cargo.toml`). Edition 2021, no MSRV, `resolver = 2`, fat-LTO release profile — matching the house exactly. `futures-util 0.3` for streaming. A `build.rs` like hs-distill’s injects `MYELIN_VERSION`.

### Workspace root `myelin/Cargo.toml`
```toml
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.dependencies]
# Shared across workspace crates (versions mirror /home-still Cargo.lock where shared)
reqwest = { version = "0.12", features = ["json", "rustls-tls-native-roots"], default-features = false }
tokio = { version = "1", features = ["full"] }
clap = { version = "4.5", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
dirs = "6"
anyhow = "1"
thiserror = "2"
figment = { version = "0.10", features = ["yaml", "env"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
async-trait = "0.1"
regex = "1"
axum = { version = "0.8", features = ["multipart"] }
qdrant-client = { version = "1", features = ["serde"] }
xxhash-rust = { version = "0.8", features = ["xxh3"] }
uuid = { version = "1", features = ["v4", "v5"] }
fastembed = "5"
chrono = { version = "0.4", features = ["serde"] }
futures-util = "0.3"
schemars = "1"
tokio-stream = "0.1"
tempfile = "3"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = "symbols"
```

### `crates/myelin-core/Cargo.toml` (backend — mirrors hs-distill)
```toml
[package]
name = "myelin-core"
version = "0.1.0"
edition = "2021"

[features]
default = ["client"]
client = []
server = [
    "dep:axum",
    "dep:fastembed",
    "dep:ort",
    "dep:qdrant-client",
    "dep:xxhash-rust",
    "dep:uuid",
    "dep:tokio-stream",
    "dep:tempfile",
    "dep:tracing-subscriber",
]
cuda = ["server", "ort/cuda"]

[[bin]]
name = "myelin-core-server"
path = "src/server_main.rs"
required-features = ["server"]

[dependencies]
clap = { workspace = true }
anyhow = { workspace = true }
thiserror = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tracing = { workspace = true }
tokio = { workspace = true }
reqwest = { workspace = true }
figment = { workspace = true }
async-trait = { workspace = true }
futures-util = { workspace = true }
chrono = { workspace = true }
dirs = { workspace = true }
regex = { workspace = true }

# Server only
axum = { workspace = true, optional = true }
fastembed = { workspace = true, optional = true }
ort = { version = "=2.0.0-rc.11", optional = true }
qdrant-client = { workspace = true, optional = true }
xxhash-rust = { workspace = true, optional = true }
uuid = { workspace = true, optional = true }
tokio-stream = { workspace = true, optional = true }
tempfile = { workspace = true, optional = true }
tracing-subscriber = { workspace = true, optional = true }

[dev-dependencies]
tempfile = { workspace = true }
```

### `crates/myelin-mcp/Cargo.toml` (MCP server — mirrors hs-mcp)
```toml
[package]
name = "myelin-mcp"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "myelin-mcp"
path = "src/main.rs"

[dependencies]
myelin-core = { path = "../myelin-core", features = ["client"] }
rmcp = { version = "1.3", features = [
    "server",
    "transport-io",
    "transport-streamable-http-server",
] }
serde = { workspace = true }
serde_json = { workspace = true }
schemars = { workspace = true }
tokio = { workspace = true }
anyhow = { workspace = true }
axum = { workspace = true }
chrono = { workspace = true }
clap = { workspace = true }
tracing = { workspace = true }
reqwest = { workspace = true }

[dev-dependencies]
myelin-core = { path = "../myelin-core" }
```

### `crates/myelin-eval/Cargo.toml` (eval harness — client + CLI)
```toml
[package]
name = "myelin-eval"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "myelin-eval"
path = "src/main.rs"

[dependencies]
myelin-core = { path = "../myelin-core", features = ["client"] }
clap = { workspace = true }
anyhow = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true }
reqwest = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }
```

### Layout notes for the myelin workspace
- Put `myelin-core` modules behind `#[cfg(feature = "server")]` in `src/lib.rs` exactly as hs-distill does (`crates/hs-distill/src/lib.rs:1-13`).
- Config: `~/.myelin/config.yaml` (new `CONFIG_REL_PATH`), figment `Serialized::defaults → Yaml::file(...).nested() → Env::prefixed("MYELIN_")`, one top-level `.select("core")` (server) + separate `storage`/`events` selects (client) — copy `config.rs:96-110,154-170` shape. Add a `build.rs` mirroring `crates/hs-distill/build.rs` to inject `MYELIN_VERSION`.
- Error: single `MyelinError` thiserror enum (`error.rs`), anyhow only in binaries + client.
- Qdrant: `Qdrant::from_url(url).timeout(30s).build()`, `CreateCollectionBuilder` + `VectorParamsBuilder::new(dim, Distance::Cosine).on_disk(true)` + `HnswConfigDiffBuilder::default().m(0)` + `on_disk_payload(true)`; deterministic `xxh3_64` + uuid-v5 point ids; payload indexes via `CreateFieldIndexCollectionBuilder`; `QueryPointsBuilder` + `Query::from(vec)` search; `FacetCountsBuilder` for doc enumeration.
- Embedding: bge-m3 / 1024-dim via `InitOptions::new(EmbeddingModel::BGEM3)` + `CUDAExecutionProvider` (dense-only, `supports_sparse() = false`); model pool (CUDA=1, CPU=min(concurrency,4)); EWMA adaptive batch (CUDA `[16,32,48,64,96,128]`, CPU `[4,8,12,16,24]`); `spawn_blocking`.
- MCP: rmcp 1.3 `#[tool_router]`/`#[tool]` with `Parameters<T>`/`Result<String, String>`, schemars param structs, resources `catalog:///{stem}` + `markdown:///{stem}` (+ `/{page}`), stdio default vs `--serve` streamable HTTP via clap, `ServerCapabilities` + `.with_instructions(...)` prose.
- Daemon: axum `Router` with `/health` + `/readiness` + `/status`, `DefaultBodyLimit`, in-flight guard, ctrl_c graceful shutdown; systemd unit `Type=simple`, `User=`, `Restart=on-failure`, `RestartSec=5`, `After=network-online.target`, `WantedBy=multi-user.target` installed via `myelin <name> --install`.
- Tests: inline `#[cfg(test)] mod tests` per module; CI `cargo test --workspace` + `cargo clippy --all-targets -- -D warnings`; keep `**/mutants.out*/` in `.gitignore`, run `cargo mutants` with defaults (no committed `mutants.toml`).
