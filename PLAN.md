# myelin — build plan

A Rust agentic-memory system for the home cloud, split into a backend crate and an MCP crate, with an
agentic evaluation harness that is the primary artifact of the project: **we do not get to say "SOTA"
unless the harness says it, on data we did not author, against numbers published by other people.**

- **Evaluation specification — read this first: [`docs/EVALUATION.md`](docs/EVALUATION.md).** It defines the
  gates below, the official LongMemEval-V2 harness integration, the LAFS metric and its arithmetic, the
  agentic loop, the statistics, and the CI tiers. The objective put evaluation above everything else, so it
  gets its own document and this one is answerable to it.
- Evidence base: [`docs/research/`](docs/research/) — 12 files, every claim DOI- or probe-cited.
- First-hand measurements (Qdrant capability probes, dataset shapes, crate versions, GPU envelope):
  [`docs/research/00-verified-environment.md`](docs/research/00-verified-environment.md);
  reproducible probe source in [`docs/probes/`](docs/probes/).
- Adversarial re-verification of every headline number, including the twelve defects it found in an earlier
  draft of this document: [`docs/research/audit/`](docs/research/audit/).
- Vocabulary: [`README.md`](README.md).

---

## 1. Success definition

Three gates. All three must pass on the same commit, or the claim is not made.

| Gate | Statement | Measured by |
|---|---|---|
| **G1 — agentic accuracy** | Positive **LAFS gain** against the released LongMemEval-V2 reference frontier (`small` = 55.765, `medium` = 51.074, both computed from the released tool), at **≥2 latency operating points**, on a leaderboard-valid run. Break-even is accuracy > 51.1 at ≤1 s on `small`; headline target `recall` ≥65 @ ≤1 s + `investigate` ≥80 @ ≤40 s ⇒ **+13.79** | official LME-V2 harness + `leaderboard/compute_lafs.py` |
| **G2 — conversational accuracy at open weights** | LongMemEval_S ≥ **80.80** and LoCoMo(n=1540) ≥ **77.85** — MemPro-15's own Qwen3-30B-A3B numbers, so the answer-model class matches ours — with paired CIs excluding zero | `myelin-eval bench` |
| **G3 — robustness** | MINJA-style ASR ≤ 10% with pre-populated memory at k=6, ≥90% templated-poison quarantine, and **zero** cross-tenant leaks on any read path | `myelin-eval attack` |

G1 is the primary claim: LME-V2 is the only benchmark of the three with a public leaderboard, a pinned
protocol, a released reference frontier, and a machine-enforced query-privacy rule. G2 is the claim that is
comparable to the existing memory-systems literature. Full derivations, sensitivity tables and the
statistical rules are in [`docs/EVALUATION.md`](docs/EVALUATION.md) §3 and §5.

A number without its protocol is not a number. Every result row records the full manifest of
`docs/EVALUATION.md` §9: dataset + subset + SHA, reader/controller/embedder/judge models, temperature,
seeds, repeat count, token counts, p50/p95 latency, GPU tenancy state, VRAM peak, and commit SHA.

**One correction the LAFS arithmetic forces on this plan:** sub-second latency earns *no* extra LAFS, because
the metric integrates from $T_{\min}=1\,$s. The p95 < 100 ms figure below is a **product SLO**, not a
scoring objective, and must never be traded against accuracy in pursuit of G1.

### 1.1 The protocol landmine we already stepped on

`locomo10.json` contains **1,986** QA pairs (measured: categories `1:282, 2:321, 3:96, 4:841, 5:446`).
Every paper and vendor blog reporting "1,540 questions" has silently dropped category 5 — the **adversarial /
unanswerable** set — because `1986 − 446 = 1540` exactly. Abstention is the hardest capability and it is the
one routinely excluded. `myelin-eval` therefore reports **three** LoCoMo columns: `n=1540` (comparable),
`n=1986` (honest), and `abstention-F1` on category 5 alone. See `00-verified-environment.md` §4.1.

---

## 2. What the literature actually forces

Ranked by how much each finding constrains the build. Each is load-bearing; the rest of this plan is downstream.

1. **BM25 is the highest-leverage single channel, not dense.** MemPro's retrieval ablation on LoCoMo:
   removing BM25 → 84.93 → **72.25** (−12.68); removing the dense embedder → 82.57 (−2.36); removing the
   structural channel → 84.37 (−0.56). `docs/research/11-frontier-2026.md` §3.2 (arXiv 2606.00619).
   Corroborated: lexical beats dense out-of-domain and on rare/exact tokens (`03-retrieval.md` §6d).
   → BM25 is a first-class index, and we get it free (§5.2).
2. **Reranking moves more quality than fusion.** MS MARCO dev MRR@10: BM25-anserini **18.7** → BERT-large
   cross-encoder over BM25 top-1000 **36.5** → RocketQAv2-ERNIE over the same candidates **40.1**
   (`10.18653/v1_2021.emnlp-main.224` Table 3). Fusion contributes ~1–2 points by comparison. Run both;
   rerank is the lever. `03-retrieval.md` §6c, audit `audit/retrieval-claims.md` claim 3.
3. **Scope before you route.** ShardMemo: metadata predicates mask inadmissible shards *before* the router
   spends its probe budget. Controlled comparison against a simple learned router: **+2.9 / +3.1 F1** at
   S=20 / S=80. Macro-averaged over query types: **+6.76 F1 with GPT-OSS-20B**, +6.34 with Qwen3-32B
   (appendix; the main GPT-OSS-120B delta is ≈5.9). Operating point `B_probe = 3` (swept 1/2/4/8),
   `K = 10`, `S = 40` (`10.48550/arxiv.2601.21545`).
   SwiftMem: query-agnostic full-space retrieval, not the ANN index, is the latency bottleneck —
   **10.834 ms vs 881.9–1231.3 ms** search-only on LoCoMo (81–114×, GPT-4.1-mini, HNSW dense baselines);
   the separately-quoted **11.7 ms vs 794–1264 ms** figure is LoCoMo **Refined**, a different benchmark
   (`10.48550/arxiv.2601.08160` Tables 2 and 4). `03-retrieval.md` §6e–f.
4. **Evidence-set bloat destroys precision.** HiGMem on LoCoMo10 (GPT-4o-mini, all-MiniLM-L6-v2) retrieves
   **8.09 vs 99.84** turns/query at **P@K 0.1909 vs 0.0101** (≈19×), adversarial F1 0.54 (A-Mem) → **0.78**,
   hybrid deployment cost \$17.43 → \$6.43 (`10.48550/arxiv.2604.18349`).
   Lost-in-the-middle: **GPT-3.5-Turbo** on 20-document NQ-Open multi-doc QA scores **53.8%** with the gold
   document in the middle — *below* its own **56.1% closed-book** score (`10.48550/arxiv.2307.03172` Table 6).
   `08-context-engineering.md` §1, §8.
   → Return few items, bookended, with question-aware ordering. Never "top-50 into the prompt".
5. **The write path is a 4-op delta, decided by a model.** `ADD | UPDATE | DELETE | NOOP` (Mem0
   `10.48550/arxiv.2504.19413`; Memory-R1 `10.48550/arXiv.2508.19828`). `DELETE` is what makes temporal
   consistency work — systems without it fail knowledge-update questions. `01-systems.md` §Cross-system.
6. **Bi-temporal validity is the grounding mechanism.** Zep/Graphiti separates *valid time* from *ingestion
   time* and invalidates superseded edges (sets `t_invalid`) rather than overwriting them. Numbers, each with
   the backbone that produced it (`10.48550/arxiv.2501.13956` Table 1 and §4.2–4.3): DMR **94.8%** on
   gpt-4-turbo — the apples-to-apples row against MemGPT's **93.4%**, also gpt-4-turbo — and **98.2%** on
   gpt-4o-mini; LongMemEval_S **71.2%** on gpt-4o, **63.8%** on gpt-4o-mini; average context
   **115k → 1.6k tokens** and **≈−90% latency** (3.20 s vs 31.3 s) on gpt-4o-mini.
   `06-consolidation-forgetting.md` §4, audit `audit/systems-claims.md` claims 2, 2b.
7. **Graphs earn their place only on multi-hop, and only if thin.** HippoRAG PPR on 2WikiMultiHopQA against
   its own ColBERTv2 baseline: EM **33.4 → 46.6**, all-recall **AR@5 25.1 → 75.7**; online retrieval cost
   10–30× lower than iterative IRCoT (`10.48550/arxiv.2405.14831` Tables 4, 6, Appendix G). But synonym edges
   are ~10× extracted edges (1,125,951 synonym vs 140,830 extracted on MuSiQue, HippoRAG-2 Table 10) and
   GraphRAG community re-summarization costs ≈14M tokens per insert. QueryLink beats Zep on LongMemEval
   (69.8 vs 63.8) with **no** graph.
   `07-graph-memory.md` §1, §5. → Route by query type; build the smallest graph that supports 1-hop PPR.
8. **2026 reframed the target: memory as *environment competence*, not chat recall.** LongMemEval-V2:
   451 curated questions over five abilities (static state, dynamic state, workflow, gotchas, premise
   awareness); `LME-V2-Small` = one shared 100-trajectory haystack ≈25M tokens, `LME-V2-Medium` =
   question-specific 500-trajectory haystacks ≈115M tokens; "context gathering" formulation — the memory
   system consumes history trajectories and returns **compact evidence**. Verified from arXiv 2605.12493v1:
   a coding-agent controller over raw trajectory files (**AgentRunbook-C, 74.9% Small / 70.1% Medium,
   72.5% overall**) beats their own RAG memory (**AgentRunbook-R, 58.6 / 57.0, 57.8% overall**) by
   **+16.3 / +13.1 points**, beats the strongest external RAG baseline (48.5%) by 24 points, and beats
   plain state-slice RAG (40.1%) by 32. The cost is latency: vanilla Codex spends **≈182 s/query, 6.9×
   slower than AgentRunbook-R**; AgentRunbook-C is 32% faster than Codex and still far slower than RAG.
   → `myelin` must expose **two retrieval modes on one store**: fast `recall` and agentic `investigate`,
   and the eval must report the accuracy/latency Pareto, not a single accuracy.
9. **Poisoning is real but conditional — and the conditions are the design brief.** MINJA: a *regular user*
   with query-only access poisons a **shared** memory bank — **98.2% ISR / 76.8% ASR** averaged over three
   agents and four datasets (RAP/Webshop, EHRAgent/MIMIC-III+eICU, QA/MMLU), **starting from empty memory**,
   retrieval k = 3–5 (`10.48550/arXiv.2503.03704`). Two separate follow-up experiments on the MIMIC-III EHR
   agent (`10.48550_arxiv.2601.05504`) bound the risk:
   (a) pre-populating legitimate memory drops ASR **62% → 6.67%** (GPT-4o-mini) and **52.94% → 0%**
   (Llama-3.1-8B-Instruct) at k = 3 (Table 1); (b) in a *different* configuration — 6 pre-existing memories
   plus 4 indication prompts — ASR climbs with retrieval breadth: **6% → 20% → 38%** at k = 3/5/10 for
   GPT-4o-mini, 0% → 13.33% → 27.27% for Llama (Table 2). These are two setups, not one curve.
   Separately, a Gemini-2.0-Flash guard agent accepted **54 malicious entries at trust = 1.0** (of 82 accepted
   from 151 candidates) while the GPT-4o-mini run rejected all 23 — a confidence score is not a safety filter.
   `05-security-governance.md` §3, audit `audit/security-eval-claims.md`.
   → Small k is a *security* parameter, not just a cost parameter. Per-tenant isolation is the primary
   defense, because isolation removes MINJA's shared-bank premise entirely.
10. **RL-learned write policies win but are not portable.** Mem-α (GRPO on Qwen3-4B, 3 days × 32 H100)
    reaches MemoryAgentBench avg 0.592 vs RAG-Top2 0.502 — and the policy is welded to the fine-tuned model
    (`10.48550/arXiv.2509.25911`). `02-learned-policies.md` §1.
    → v1 encodes the *rules* those papers learned (thresholds, gates, reward terms as heuristics); no training.

### 2.1 What the evidence says *not* to build

- One-shot vector-only RAG memory (`11-frontier-2026.md` synthesis).
- Static graph N-hop expansion: 2–3 hops degrade; 1-hop + PPR beats naive neighbor expansion R@5 72.5 vs 59.2
  (`07-graph-memory.md` §5).
- GraphRAG-style community summaries, unless whole-corpus sensemaking becomes a product requirement.
- A token-search substrate as the reasoning layer: Hippocampus is 31× faster and 14× cheaper with **0-token**
  construction, but scores 2.40/5 on LongMemEval-M — an efficiency substrate, not an accuracy play
  (`11-frontier-2026.md` §2).
- Single-GPU binary-reward RL over memory ops: EM reward collapses at G=4 (arXiv 2605.23067).
- DMR as a SOTA claim: it is saturated — Zep reaches 98.2% on gpt-4o-mini and 94.8% on gpt-4-turbo, against
  MemGPT's 93.4%. Regression smoke test only (`04-benchmarks.md` §2, `audit/systems-claims.md` claim 2b).

---

## 3. Crate split

Workspace at repo root; three members, two of them shippable crates. Matches the `home-still` house layout
(`crates/*` + shared, `resolver = "2"`, fat-LTO release profile, `build.rs` version injection) documented in
`09-house-conventions.md`.

```
myelin/
  Cargo.toml                  # [workspace] members = ["crates/*"]
  crates/
    myelin-core/              # the backend. library-first, feature-gated like hs-distill
    myelin-mcp/               # the MCP server + admin CLI. thin. rmcp 3.3
    myelin-eval/              # the harness. not published; the reason the other two are trustworthy
```

### 3.1 `myelin-core` — the backend

Library only; no binary. Everything below is a module, and the module boundary is the test boundary.

```
src/
  lib.rs
  error.rs          MyelinError (thiserror 2), one variant per failure domain, no anyhow in the lib
  config.rs         figment: Serialized::defaults → Yaml(~/.myelin/config.yml) → Env("MYELIN_")
  model/
    record.rs       MemoryRecord, RecordKind, Provenance, Validity, Trust
    delta.rs        Delta { Add, Update, Delete, Noop } — the only way the store mutates
    query.rs        Recall { scope, text, budget, mode }
    evidence.rs     EvidenceSet — what compose() returns; the MCP contract type
  store/
    qdrant.rs       collection schema, upsert, hybrid query, gRPC only (REST is broken, §5.1)
    ledger.rs       sqlx/SQLite append-only event log + bi-temporal validity + ACL edges
    graph.rs        phrase↔record incidence, petgraph PPR
    ids.rs          uuid v5 over (namespace, record_id) — same derivation as hs-distill
    export.rs       export_namespace / import_namespace + pinned MemoryConfigJson (R3)
  embed/
    mod.rs          Embedder trait; no silent CPU fallback (house rule, hs-distill/src/embed/mod.rs)
    local.rs        fastembed 6 + ort (CUDA feature) — bge-m3, 1024-d dense
    remote.rs       hs-distill HTTP / ollama bge-m3 — the default on macOS dev
  rerank/
    mod.rs          Reranker trait
    late.rs         Qdrant max_sim multivector rerank (server-side, zero extra hop)
    cross.rs        bge-reranker cross-encoder via fastembed or remote
  llm/
    mod.rs          Llm trait: complete(), complete_json::<T>()
    openai.rs       reqwest against llama-swap /v1 (qwen3.8-27b) and ollama
  pipeline/
    ingest.rs       segmentation into episodes; surprise/boundary detection
    extract.rs      episode → candidate semantic records + entities
    consolidate.rs  delta decision, conflict resolution, invalidation, dedup
    index.rs        embed + upsert + graph update, batched
    retrieve.rs     scope-filter → hybrid prefetch → fuse → rerank → (graph-expand)
    compose.rs      budgeted, bookended, provenance-carrying EvidenceSet
    forget.rs       decay, eviction, TTL, hard delete / unlearn
  policy/
    write.rs        thresholds & gates that stand in for RL-learned policies (§6.3)
    read.rs         mode selection, k selection, iterate-or-stop
    trust.rs        trust scoring, quarantine, poison-pattern filters
  audit.rs          append-only decision log (accept/reject + reason) for every op
```

Features, mirroring `hs-distill`'s `client`/`server`/`cuda` split:

| feature | pulls in | used by |
|---|---|---|
| `default = ["remote-embed"]` | reqwest | macOS dev, MCP on workstation |
| `local-embed` | `fastembed`, `ort` | in-process embedding |
| `cuda` | `local-embed`, `ort/cuda` | `big` |
| `graph` | `petgraph` | PPR multi-hop route |

### 3.2 `myelin-mcp` — the MCP surface

`rmcp` 3.3.0 (MCP spec `2026-07-28`, MSRV 1.88 — note: newer and **breaking** vs the `rmcp` 1.3 used by
`hs-mcp`). stdio by default, streamable-HTTP behind `--serve`, selected by clap exactly as `hs-mcp` does
(`09-house-conventions.md` §8). Contains no memory logic — it is a projection of `myelin-core`.

### 3.3 `myelin-eval` — the harness

A binary plus a library of scorers and runners. Owns dataset acquisition and checksum verification, run
manifests, the judge panel, the statistics, the attack suite, the ablation matrix, and the report. It must be
able to evaluate **any** memory backend behind one trait, so baselines (`FullContext`, `DenseOnly`,
`Bm25Only`, `HybridNoRerank`) are measured by the same code path as `myelin` itself. Without that, every
number is self-graded.

```
crates/myelin-eval/
  src/
    main.rs            subcommands: fetch | build | bench | attack | ablate | report | package
    datasets/          locomo.rs, longmemeval.rs, lme_v2.rs  — loaders + sha256 pinning
    backends/          trait MemoryBackend + Myelin / FullContext / DenseOnly / Bm25Only / HybridNoRerank
    judge/             panel, position-shuffle, κ, calibration, published prompt templates (hashed)
    scorers/           LoCoMo token-F1 / BLEU-1 / EM; abstention-F1   (LME-V2 scoring is delegated, never reimplemented)
    stats/             bootstrap CIs, McNemar, paired bootstrap, seed handling
    attack/            MINJA-style E1–E6 from `EVALUATION.md` §7
    manifest.rs        the run-manifest schema of `EVALUATION.md` §9
    report.rs          comparison tables, Pareto/LAFS rendering
  adapters/
    myelin.py          ~100-line `Memory` subclass registered with @register_memory; pure forwarder to myelin-mcp
  vendor/
    longmemeval-v2/    pinned checkout of the official harness (Apache-2.0); we call it, we do not fork it
```

The one hard rule: for LongMemEval-V2 we run the authors' `evaluation/run_eval.py`, their
`qa_eval_metrics.py`, and their `leaderboard/` builders. Reimplementing a benchmark's metric is how people
accidentally publish incomparable numbers.

---

## 4. Data model

One record type, four kinds, explicit time and explicit trust.

```rust
pub struct MemoryRecord {
    pub id: Uuid,                    // v5(namespace, natural key) — idempotent re-ingest
    pub kind: RecordKind,            // Episodic | Semantic | Procedural | Working
    pub scope: Scope,                // tenant + agent + session + namespace  → the scope-filter key
    pub text: String,                // the only thing a model ever reads
    pub entities: Vec<EntityRef>,    // graph seeds; phrase nodes
    pub validity: Validity,          // bi-temporal
    pub provenance: Provenance,      // immutable source pointer
    pub trust: Trust,                // tier + score + checks that passed
    pub salience: Salience,          // importance, access stats, decay state
    pub links: Vec<Link>,            // typed edges (supersedes, contradicts, derived_from, co_episode)
}

pub struct Validity {               // Zep/Graphiti, 10.48550/arxiv.2501.13956
    pub t_valid: DateTime<Utc>,      // when the fact became true in the world
    pub t_invalid: Option<DateTime<Utc>>, // when it stopped being true — set by UPDATE/DELETE, never overwritten
    pub t_ingested: DateTime<Utc>,   // when we learned it
    pub t_expired: Option<DateTime<Utc>>, // when we stopped believing our record
}

pub struct Provenance {             // C1, 05-security-governance.md
    pub source: SourceRef,           // doc/chunk/turn id + line or turn range
    pub contributed_by: ActorId,     // user
    pub written_by: ActorId,         // agent
    pub derived_from: Vec<Uuid>,     // consolidation lineage — required for grounding claims
}

pub enum TrustTier { Verified, Asserted, Untrusted, Quarantined }  // C4
```

Invariants, enforced in `consolidate.rs` and tested directly:

- **I1** A record is never mutated in place. `UPDATE` writes a new record and sets the predecessor's
  `t_invalid`; `DELETE` sets `t_invalid` only. The ledger is append-only (C9).
- **I2** Every record retrievable in `compose` has non-null `provenance.source` (C6b). No provenance → not
  admissible, regardless of score.
- **I3** `Quarantined` records are invisible to every read path except the explicit review tool (C4).
- **I4** A `Semantic` record's `derived_from` is non-empty and every ancestor exists — this is what makes
  "show me why you believe that" answerable, and it is what a grounding metric measures.
- **I5** Deleting a source record deletes or re-derives every descendant (C11); the harness asserts this.

### 4.1 Requirements imposed by the LongMemEval-V2 adapter ABI

These are not preferences. They come from the benchmark's released `Memory` base class and its
query-privacy test, read this session (`docs/EVALUATION.md` §2.3–2.4). Violating any one of them
disqualifies a leaderboard submission, so they are constraints on `myelin-core`, not on the harness.

- **R1 — evidence is a list of typed items.** `query()` must return `list[{type: "text"|"image", value: str}]`.
  `EvidenceSet` serialises to exactly that shape; no bespoke envelope.
- **R2 — ingest granularity is one whole trajectory.** `insert(trajectory)` hands over
  `{id, domain, environment, goal, outcome, start_url, states[]}` with each state carrying
  `{state_index, step, url, action, thought, accessibility_tree, screenshot}`. Segmentation into episodes is
  ours to do (§6.1), not the caller's.
- **R3 — a built memory must export and re-import byte-faithfully.** `reconcile_loaded_memory_config`
  *requires* the requested config to equal the saved config when loading a prebuilt artifact. So
  `myelin-core` needs `export_namespace` / `import_namespace`, and the emitted config must be a pure
  function of the build inputs. This is a first-class store feature, not a convenience.
- **R4 — operating points are query-time, never build-time.** `configure_runtime` exists for non-persisted
  overrides, and the leaderboard packager validates that every operating point shares one haystack and one
  method. Therefore `recall` vs `investigate`, `k`, `rrf_k` and step budgets **must all be parameters of a
  query against one identical store.** Any design where `investigate` needs its own index cannot produce a
  multi-operating-point submission — which, per the LAFS arithmetic, is where the score comes from.
- **R5 — concurrency-safe reads.** The harness drives queries from multiple threads against one built
  memory and attributes latency through a thread-local `query_invocation_id`. `myelin-core` reads must be
  `Send + Sync` and free of shared mutable retrieval state; per-query telemetry keys off that id.
- **R6 — metadata blindness.** A backend sees only the question text, an optional image path, and an opaque
  invocation id. It must never see `question_id`, `question_type`, the gold answer, the evaluator spec, or
  the original goal. So the read-policy router in §7.1 classifies a query from its **text alone**, and
  `myelin-eval` runs the benchmark's own privacy test against our adapter in CI so we cannot drift into
  cheating by accident.
- **R7 — empty completions are failures.** llama-swap returns HTTP 200 with a zero-byte body when an
  upstream model fails to load (`00-verified-environment.md` §7.2). The `Llm` trait must reject an empty
  completion as an error rather than propagating it as a null answer.

---

## 5. Storage

### 5.1 Qdrant — one collection, three channels

Verified working on the live Qdrant 1.19.1 via `qdrant-client` 1.19.0 over gRPC
(`00-verified-environment.md` §3.1–3.4):

```
collection "myelin_memory"
  vectors:
    dense : 1024-d Cosine                           # bge-m3 dense
    late  : 1024-d Cosine, multivector max_sim      # OPTIONAL — bge-m3 ColBERT output, see risk below
  sparse_vectors:
    lex   : { modifier: "idf" }                      # BM25, computed inside Qdrant
  payload indexes: tenant, agent, session, kind, trust_tier, t_valid, t_invalid, entity_ids
```

The `late` channel is deliberately optional. BGE-M3 emits dense, lexical-sparse and ColBERT multi-vector
heads from one forward pass, and its ColBERT vectors are **1024-d per token** — so a 200-token record costs
200 × 1024 floats, two orders of magnitude more storage than its dense vector. Two open risks, both to be
settled in M3 rather than assumed: (i) whether the Rust `fastembed` build we use exposes bge-m3's ColBERT
head at all (the dense head certainly; the multi-vector head is unverified), and (ii) whether late
interaction beats a `bge-reranker` cross-encoder at equal latency on our data. Until both are answered,
`rerank/cross.rs` is the default and `late` stays unpopulated.

Three probe results drive this design:

1. **BM25 runs inside Qdrant, locally, with no inference service.** Writing
   `vector.lex = {text: "...", model: "qdrant/bm25"}` returns 200 on the self-hosted instance; Qdrant
   tokenizes, stems (snowball), drops stopwords, hashes terms and stores TF weights, then applies
   `IDF(t) = ln((N − n(t) + 0.5)/(n(t) + 0.5) + 1)` at query time. Reproduced the scorer exactly:
   predicted 1.3486677 / 0.7050055, Qdrant returned 1.3486677 / 0.70500547.
   Options `{k, b, avg_len, language, stemmer, stopwords, lowercase}` are accepted.
   → **No `tantivy`, no client-side BM25 encoder, no second index.** Finding #1 of §2 is satisfied for free.
2. **Model-based inference is not available locally** (`500 InferenceService URL not configured` for dense and
   for `qdrant/minicoil-v1`). Dense vectors are ours to produce: `fastembed`+`ort` in-process on `big`,
   or the existing `hs-distill` service / ollama `bge-m3` from the workstation.
3. **gRPC only.** Multivector *queries* over REST fail with
   `422 internal.query.indices: must be unique` — the untagged `VectorInput` enum mis-parses a float matrix
   as a sparse vector. gRPC is unaffected.

### 5.2 Fusion: ours, not Qdrant's

Qdrant's server-side RRF uses **k = 1**, measured: single-list ranks 1..6 score
`0.5, 0.33333334, 0.25, 0.2, 0.16666667, 0.14285715` = `1/(1+rank)`. Cormack's canonical k = 60 would be
nearly flat by comparison. k = 1 is aggressively top-rank-biased, which is the wrong default when one channel
is systematically better than the other (and per finding #1, it is).

**Decision:** issue the dense and sparse prefetches as one `query_points` call with `limit` per branch, take
both ranked lists back, and fuse in Rust with configurable `k` (default 60). Server-side `rrf`/`dbsf` stay
available behind a config flag so the harness can A/B `k ∈ {1, 60}` as an ablation. One round trip either way.

### 5.3 SQLite — ledger, ACL, graph

`sqlx` 0.9 with SQLite (Postgres-ready types; no Postgres dependency in v1).

| table | purpose |
|---|---|
| `event` | append-only: every ingest/delta/read decision, with reason text (C9, C10) |
| `record` | current projection: id, kind, scope, validity, trust, salience — the source of truth for filters |
| `link` | typed edges; `supersedes` / `contradicts` drive invalidation |
| `incidence` | phrase ↔ record, the PPR bipartite structure (`07-graph-memory.md` §1.1) |
| `acl_ua`, `acl_ar` | bipartite ACL edges `G_UA(t)`, `G_AR(t)`; revocation = edge removal (C2) |
| `quarantine` | staged writes awaiting promotion (C4) |

Qdrant holds vectors and answers "which records are similar"; SQLite holds truth, time, permission, and
lineage and answers "which records may be seen and are still believed". Neither is authoritative alone;
`reconcile` is a first-class operation (the drift-repair pattern `hs` already uses).

### 5.4 Graph: the minimum that is justified

Phrase↔record incidence in SQLite, loaded into `petgraph` for personalized PageRank when the read policy
routes a query as multi-hop. Seeds are query entities matched to nodes by `argmax cos(M(c_i), M(e_j))`, equal
reset probability, node specificity `s_i = |P_i|⁻¹`, all-record seeds at weight 0.05, **1-hop expansion only**
(`07-graph-memory.md` §1, §5 — 1-hop + PPR: R@5 72.5 vs 59.2 for naive neighbors; 2–3 hops degrade).
Damping is `[UNVERIFIED]` in the corpus; default 0.5 per the Forgetting pack and treat it as a tuned parameter.

Synonym edges are the storage risk (~10× extracted edges); cap them by threshold and degree, and measure the
edge count as a first-class metric.

---

## 6. Write path

```
ingest → extract → consolidate → index
```

### 6.1 `ingest` — segmentation

Turn a stream of turns/tool-results into `Episodic` records. Segment on boundary signals (topic shift, task
boundary, elapsed time), not fixed windows; LLMs segment text into events about as humans do
(`10.48550/arxiv.2502.06975` RQ2). Chunking for retrievable text: sentence-level, 512-token units, 20-token
overlap. The supporting table lives in the **arXiv preprint** `arXiv:2407.01219` §3.2.1 Table 3, *not* in the
published `10.18653/v1_2024.emnlp-main.981` version, and it is a small single-document eval (60 pages of a
10-K, zephyr-7b-alpha generator, gpt-3.5-turbo evaluator): 512 tokens is best on **faithfulness** (97.59) but
**256 wins on relevancy** (97.78 vs 97.41). Treat 512/20 as a starting point to tune on our own eval, not a
settled result. Keep multi-turn units intact for episodic material (`03-retrieval.md` §7 row 0,
audit `audit/retrieval-claims.md` claim 4).

Raw episodes are stored **losslessly**; derived records point back at them (Zep's bidirectional episode↔semantic
index, `06-consolidation-forgetting.md` §1).

### 6.2 `extract` — one model call per episode

Emit candidate `Semantic` records (atomic facts with `t_valid`), entity mentions, and optional `Procedural`
records (how a task was accomplished, including failure modes — LME-V2's "gotchas" ability is exactly this).
Structured output against a schema; validated; failures quarantined, never dropped silently.

Extraction correctness is the dominant error source in graph-memory systems (`01-systems.md` finding 3), so
extraction gets its own unit-level eval: precision/recall of facts against hand-labelled episodes.

### 6.3 `consolidate` — the 4-op delta, with gates

For each candidate, retrieve the k nearest existing records in scope, then decide
`ADD | UPDATE | DELETE | NOOP` (Mem0/Memory-R1). Deterministic guards around the model call, standing in for
the RL-learned policies we are not training (`02-learned-policies.md` "What a Rust system can borrow"):

- **Dedup gate** — cosine ≥ `τ_dup` (default 0.95) and same scope → `NOOP`, bump access count.
- **Contradiction gate** — a candidate that contradicts a `Verified` core fact is rejected, not applied
  (SSGM write gate `ΔM ∧ M_core ⊨ ⊥`, C8). Rejection is logged with the conflicting record id.
- **Supersession** — `UPDATE` sets predecessor `t_invalid = candidate.t_valid` and writes a `supersedes` link.
  This is the mechanism that answers knowledge-update questions; without it they are unanswerable.
- **Trust gate** — composite trust score: poison-pattern filter ("ignore previous", "refer X to Y",
  "Knowledge:"), source tier, PII scan, optional external verification of effect. Below threshold →
  `quarantine` (C4, C5). Confidence is not a security filter: the EHR study recorded 54 poisoned entries
  accepted at trust = 1.0.
- **Reflection trigger** — accumulate per-record importance; when the running sum crosses `θ_reflect`
  (Generative Agents used 150, `10.48550/arxiv.2304.03442`), run one abstraction pass producing higher-level
  `Semantic` records with `derived_from` lineage. Deterministic accumulator, one model call per firing.

### 6.4 `index`

Batch-embed dense (bge-m3, 1024-d), let Qdrant compute `lex` from the record text, optionally compute `late`
vectors, upsert with payload, update `incidence`. Adaptive batch sizing follows the EWMA hill-climber already
in `hs-distill/src/adaptive_batch.rs`.

---

## 7. Read path

Two modes over one store. This is the direct consequence of LME-V2: a coding-agent controller over raw files
beats the same authors' RAG memory by **+16.3 points on LME-V2-Small and +13.1 on Medium** while costing
roughly an order of magnitude more latency (Codex ≈182 s/query, 6.9× AgentRunbook-R). Neither point dominates,
so the system exposes both and the caller picks a point on the Pareto frontier.

### 7.1 `recall` — fast mode (target p95 < 100 ms, no LLM in the loop)

```
scope-filter  → payload predicates: tenant, agent, session?, kind?, t_valid ≤ now < t_invalid,
                trust_tier ≠ Quarantined                               (§2 finding 3; control C7)
retrieve      → one query_points: prefetch{dense, limit 50} + prefetch{lex(BM25), limit 50}
fuse          → RRF in Rust, k = 60 (configurable; Qdrant's own is k = 1, §5.2)
rerank        → late-interaction max_sim in Qdrant, or bge-reranker cross-encoder   (§2 finding 2)
route?        → if query classified multi-hop: PPR 1-hop expansion, merge, re-rank  (§2 finding 7)
compose       → top-k, k default 6                                     (§2 findings 4 and 9)
```

`k` is small by evidence and by threat model: HiGMem's precision erosion and MINJA's ASR climbing 6% → 38% as
k goes 3 → 10 point the same direction.

### 7.2 `investigate` — agentic mode (accuracy over latency)

Expose the raw episode store as a searchable corpus and let the calling agent iterate:
`search → read → reflect → search again`, with a sufficiency-and-conflict stop gate (AMA's refresh-on-conflict
gate: 0.897 vs 0.568 on knowledge-update without it, arXiv 2601.20352). Bounded by a step budget and a token
budget; returns the same `EvidenceSet` type as `recall`, so the consumer is unchanged.

This is the mode that competes with AgentRunbook-C, and it is why the MCP surface must expose primitives
(`search`, `read`, `neighbors`) and not only a single `get_context` call.

### 7.3 `compose` — the part everyone gets wrong

`EvidenceSet` is a budgeted, ordered, provenance-carrying structure, not a blob. Its wire form is fixed by
**R1**: it serialises to `list[{type: "text"|"image", value: str}]`, because that is the LongMemEval-V2
`query()` return type (§4.1). Everything below is how we decide *which* items go in that list and in what
order:

- Order: highest-relevance items at the **head and tail**, filler in the middle — the U-curve is real and the
  middle of a long context scores below closed-book (`10.48550/arxiv.2307.03172`).
- Per-kind token budgets; total capped at a fraction of the window (MEMAGENT's working split was
  memory 1,024 / chunk 5,000 / total ≤ 8,192, `10.48550/arxiv.2507.02259`).
- Each item carries `id`, `t_valid`, source pointer, trust tier. This is what makes grounding measurable:
  the harness can check whether a claim in the answer is traceable to an item in the set.
- Dedup by near-duplicate cosine before emitting.
- Freshness gate `w(Δτ) ≥ θ_fresh` and ACL predicate applied **after** semantic top-k, before composing (C7).

---

## 8. Forgetting

Deterministic arithmetic; no model calls.

- **Decay** — Ebbinghaus retention `R = e^{−t/S}`, `S` strengthened on access (MemoryBank,
  `10.48550/arxiv.2305.10250`). Retrieval-induced strengthening dominates time-induced decay because the
  reinforcement weight is the larger one: in MOOM (`10.48550/arxiv.2509.11860`) **β = 0.9 is the
  retrieval-reinforcement weight and α = 0.1 is the temporal-decay weight** — larger means stronger.
  (An earlier draft of this plan had the two swapped; see `audit/security-eval-claims.md` defect 1.)
- **Retrieval score** — `a_recency·recency + a_importance·importance + a_relevance·relevance`, all min-max
  normalized, all weights 1 in the reference implementation (Generative Agents).
- **Eviction** — budget-driven; `Episodic` records are never evicted from the ledger, only from the hot index.
- **Expiry** — TTL per scope; `t_expired` set, record leaves read paths, ledger entry remains.
- **Hard delete / unlearn** — remove from Qdrant + `record` + `incidence`, re-derive or delete descendants,
  record the deletion in the ledger (C11). No corpus paper quantifies memory unlearning, so this ships as a
  tested capability with an explicitly unevidenced effectiveness claim.

---

## 9. Security and governance

Implement C1–C13 from `05-security-governance.md`; C14 (isolate-then-aggregate) is deferred.

The ones that shape the architecture rather than adding a check:

- **Per-tenant isolation is the primary defense** (C12). MINJA's premise is a *shared* memory bank; isolation
  removes the premise. `scope.tenant` is in every payload index and every filter, and the harness has a test
  that asserts no cross-tenant leak on every read path.
- **Bipartite ACL** (C2): persist `G_UA(t)`, `G_AR(t)`, enforce `ℳ(u,a,t)` before any fragment is surfaced;
  revocation is edge removal. Views are projected, never copied.
- **Dual store** (C9): mutable projection + append-only ledger, with periodic `reconcile`; SSGM Theorem 1
  bounds drift at `O(N·ε_step)`, where **`N` is the reconciliation *interval*** (steps between
  reconciliations, not the total horizon) and `ε_step` is the per-consolidation-step error bound. The point
  of the theorem is that drift depends on the cadence we choose, not on how long the system has run — so
  reconciliation cadence is a tunable drift budget.
- **Audit log** (C10): per write — inputs, trust score, each check, decision, reason; per read — requester,
  fragments projected. This is what makes an attack post-mortem possible.
- **Per-tenant IDF** — global IDF statistics leak cross-tenant term distributions. Qdrant documents per-tenant
  IDF scoping but our probe of `params.idf` returned `400 data did not match any variant of untagged enum
  IdfParams`; the exact shape must be read off the gRPC proto before relying on it
  (`00-verified-environment.md` §3.6). Until then, one collection per tenant class, or accept the leak and
  document it.

---

## 10. `myelin-mcp` tool surface

Designed as the "context gathering" interface LME-V2 formalizes: the memory system consumes history and
returns compact evidence. Two write tools, four read tools, three admin tools. Every tool returns provenance.

| tool | args | returns | notes |
|---|---|---|---|
| `remember` | `text, kind?, scope?, t_valid?, source?` | `Delta[]` applied + quarantined count | runs the full write path; idempotent by content hash |
| `observe` | `turns[]` | episode ids | bulk ingest of a conversation/trajectory segment |
| `recall` | `query, scope?, k?, budget_tokens?, kinds?` | `EvidenceSet` | fast mode, §7.1 |
| `investigate` | `question, scope?, max_steps?, budget_tokens?` | `EvidenceSet` + trace | agentic mode, §7.2 |
| `search` | `query, filter?, limit?` | record stubs | primitive for caller-driven iteration |
| `neighbors` | `record_id, relation?, hops?` | linked records | 1-hop graph access |
| `forget` | `selector, mode: soft\|hard` | affected ids | C11; `hard` requires confirmation |
| `review_quarantine` | `limit?` | staged writes + reasons | C4 human/agent-in-the-loop |
| `explain` | `record_id` | lineage tree + audit entries | I4/C10 made usable |

Resources: `myelin://record/{id}`, `myelin://episode/{id}`, `myelin://scope/{tenant}/{agent}` — mirroring the
`catalog:///{stem}` resource-template pattern in `hs-mcp`.

Design rules, from the evidence: tool results are summaries with identifiers, never raw dumps (progressive
disclosure); `recall` never returns more than `k` items; every returned item is individually addressable so the
caller can fetch detail on demand instead of receiving it speculatively.

---

## 11. `myelin-eval` — the part that makes "SOTA" a fact

**The full specification is [`docs/EVALUATION.md`](docs/EVALUATION.md).** It is the longer and more
important document: benchmark inventory with measured shapes, the official LME-V2 harness integration and
adapter ABI, the query-privacy rule, the LAFS metric with its computed reference values and sensitivity
tables, the agentic loop with parameter justifications, the statistical rules, the robustness suite, the
ablation matrix, the run-manifest schema, the CI tiers, and the disclosed threats to validity.

This section keeps only what changes the *design* of the other two crates. Everything else lives there.

### 11.0 What the eval spec forces back onto the design

| discovery | consequence in this plan |
|---|---|
| LME-V2's reference RAG memory is **dense-only, no BM25, no reranking** (`enable_rerank: false`, `Qwen3-Embedding-8B`) | our hybrid BM25+dense+rerank `recall` mode attacks an identified gap in the published frontier — §2 finding 1 is not just literature, it is the competitive thesis |
| LAFS integrates from $T_{\min}=1\,$s over log-latency | a fast operating point is worth ~3× a slow one; **p95 < 100 ms is a product SLO, not a scoring objective** |
| Multi-operating-point submissions share one built memory | R4: mode/`k`/budgets are query-time parameters (§4.1) |
| Backends are metadata-blind | R6: the §7.1 router classifies from query text alone (§4.1) |
| Prebuilt memories load only on exact config match | R3: `export_namespace`/`import_namespace` in `store/` (§4.1) |
| 295 of 451 LME-V2 questions are judge-free; correct abstention is literally `\boxed{unknown}` | `EvidenceSet` must make *insufficiency* decidable, not silently return nearest neighbours (§7.3) |
| Controller protocol is temperature **0.6**, not 0 | results are random variables: 3 seeds, bootstrap CIs, paired tests (`EVALUATION.md` §5) |
| One RTX 3090; distill's 4.9 GB makes a 27B model OOM; claim frees it to 343 MiB | eval runs take a `gpu-tenant claim` and record tenancy; pipeline and eval cannot overlap (`00-verified-environment.md` §7) |
| Local tool-calling verified on `qwen3:8b` alongside a 13.7 GB resident model | the agentic loop runs entirely on the home cloud — no external controller needed |

### 11.1 Backends under test, behind one trait

```rust
#[async_trait] pub trait MemoryBackend: Send + Sync {           // R5
    /// One whole trajectory / session, as the benchmark hands it over (R2).
    async fn insert(&self, unit: &IngestUnit) -> Result<IngestStats>;
    /// Metadata-blind: question text + optional image + opaque invocation id (R6).
    async fn query(&self, q: &BlindQuery) -> Result<EvidenceSet>;
    /// Non-persisted operating-point overrides: mode, k, rrf_k, step/token budgets (R4).
    fn configure_runtime(&self, o: &RuntimeOverrides) -> Result<()>;
    /// Byte-faithful export/import of a built memory plus its pinned config (R3).
    async fn export(&self, dir: &Path) -> Result<MemoryConfigJson>;
    async fn import(dir: &Path, cfg: &MemoryConfigJson) -> Result<Self> where Self: Sized;
}
```

The trait deliberately mirrors the LME-V2 `Memory` base class so the Python adapter
(`memory_modules/myelin.py`, ~100 lines, registered with `@register_memory`) is a pure forwarder over
`myelin-mcp`'s streamable-HTTP transport and contains no logic of its own.

Implementations: `Myelin{recall}`, `Myelin{investigate}`, `FullContext` (whole history in the prompt — the
ceiling and the token-cost baseline), `DenseOnly`, `Bm25Only`, `HybridNoRerank`. The last four exist so every
architectural claim in §2 is testable as an ablation on our own code rather than a citation
(`EVALUATION.md` §8).

### 11.2 Datasets — acquisition verified

| dataset | acquisition | measured shape | role |
|---|---|---|---|
| LoCoMo | `raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json` → 200, 2.8 MB | 10 convs, 19–32 sessions, 369–689 turns, 1,986 QA (1,540 excl. adversarial) | primary gate + abstention |
| LongMemEval_S | HF `xiaowu0162/longmemeval`, ungated | 500 items, mean 50 haystack sessions, 278 MB | primary gate |
| LongMemEval_M | same repo | larger haystack | scaling curve |
| LongMemEval-V2 | HF `xiaowu0162/longmemeval-v2`, **ungated, Apache-2.0** — `questions.jsonl` 0.3 MB, `haystacks/lme_v2_small.json` 0.8 MB, `haystacks/lme_v2_medium.json` 4.1 MB, `trajectories.jsonl` **1.20 GB**, plus `trajectory_screenshots/*.tar.gz` **5.9 GB** (text-only runs can skip these) | 451 Q, 5 abilities, 100-traj/≈25M-token Small and 500-traj/≈115M-token Medium haystacks | the 2026 target; `investigate` mode |
| LongMemEval-V2 **harness + leaderboard** | GitHub `xiaowu0162/LongMemEval-V2`, Apache-2.0, 158 stars, pushed 2026-08-09 — ships `evaluation/qa_eval_metrics.py`, `evaluation/run_eval.py`, `memory_modules/memory.py`, `leaderboard/compute_lafs.py`, `tests/test_query_privacy.py`, and the reference `memory_configs/*.json` | 295/451 questions scored deterministically, 156 by LLM judge; reader pinned `qwen3.5-9b`, judge pinned `gpt-5.2` | the G1 instrument — vendored, never forked |
| MINJA-style attack corpus | constructed from LoCoMo sessions per `10.48550/arXiv.2503.03704` | — | G3 |
| DMR | MSC subset | 500 conversations | saturated; smoke test only |

Text-only LME-V2 is therefore a **1.5 GB** download and fits trivially on `big`'s 1.5 TB free NVMe; the 5.9 GB
of screenshots are only needed if we ever evaluate a multimodal reader, which v1 does not. `checksums.sha256`
ships with the repo, so the harness pins dataset integrity rather than trusting the download.

BEAM is reported by `11-frontier-2026.md` as a real ICLR 2026 benchmark (1M/10M-token scales) rather than a
vendor artifact; acquisition is not yet verified, so it is a stretch target, not a gate.

### 11.3 Judging — the #1 reproducibility risk

Open-weights, pinned, reproducible (`04-benchmarks.md` §Reproducible harness):

1. Fixed judge model + quant + temperature 0, single engine, prompt hash recorded.
2. **Reuse the benchmark authors' published judge prompts verbatim.** Hand-written rubrics make numbers
   incomparable; Zep's high human correlation came from using LongMemEval's own prompts.
3. Position-shuffle each candidate under two orderings and average (MT-Bench position bias,
   `10.48550/arxiv.2306.05685`).
4. Three judges (e.g. `qwen3:8b`, `qwen2.5:7b`, `gpt-oss-20b`), averaged, **and report inter-judge Cohen's κ**.
   EverMemOS's 3-blind-judge protocol reaches **κ = 0.891 on LoCoMo and 0.979 on LongMemEval** against five
   human annotators over 25 Q&A pairs each (`10.48550/arxiv.2601.02163`) — that is the bar for the harness to
   be believed. Note this is a *judge-reliability* figure; EverMemOS's 93.05 in §11.5 is its LoCoMo accuracy,
   a different quantity (`audit/security-eval-claims.md` defect 2).
5. Calibrate on 50 gold answers before any full run; require the local panel to reproduce the reference
   ranking at ρ ≥ 0.9. If it does not, the judge is the finding, not the memory system.
6. Never mix judge families inside one comparison; print the judge alongside every number.

Deterministic scorers (LoCoMo token-F1, BLEU-1, exact match, recall@k against gold evidence sessions) run
alongside the judge, always. A judge-only result is not reportable.

### 11.4 Metrics — five axes, reported together

| axis | metric |
|---|---|
| accuracy | per-category F1 / accuracy / judge score; abstention-F1 on LoCoMo cat 5 |
| token-cost | ingest tokens, per-query prompt tokens, total \$-equivalent at local rates |
| latency | p50/p95/p99 for ingest-per-session and query, split into embed / search / rerank / llm |
| grounding | fraction of answer claims traceable to a returned item's provenance; contradiction rate against `t_invalid` records |
| robustness | MINJA ISR (injection success) and ASR (attack success) at k ∈ {3, 6, 10}, with and without pre-populated legitimate memory |

Every run emits a manifest: dataset SHA, subset, backend config, model ids, judge panel + prompt hashes, seeds,
commit SHA, hardware, GPU tenancy state. `myelin-eval report` renders the Pareto plot and the comparison table.

### 11.5 Acceptance thresholds

Published numbers to beat, with provenance flags. `paper` = from a preprint; `vendor` = vendor blog, not
reproducible from any paper.

| benchmark | metric | best published | system | source | flag |
|---|---|---|---|---|---|
| LoCoMo (n=1540) | avg judge score | **84.93** | MemPro-15 @ gpt-4o-mini | arXiv 2606.00619 | paper |
| LoCoMo | LLM-judge 0–100 | 93.05 | EverMemOS @ gpt-4.1-mini | `10.48550/arxiv.2601.02163` | paper |
| LoCoMo | overall string-F1 | 40.00 | GAM @ gpt-4o-mini | `10.48550/arxiv.2604.12285` | paper |
| LoCoMo | adversarial F1 | 0.78 | HiGMem | `10.48550/arxiv.2604.18349` | paper |
| LoCoMo | vendor claim | 92.5% | Mem0 | mem0.ai blog, 2026 | **vendor** |
| LongMemEval_S | accuracy | **79.00** | MemPro-15 @ gpt-4o-mini | arXiv 2606.00619 | paper |
| LongMemEval_S | accuracy | 74.6 | NEMORI @ gpt-4o | `10.48550/arxiv.2508.03341` | paper |
| LongMemEval_S | accuracy | 71.2% | Zep @ gpt-4o | `10.48550/arxiv.2501.13956` | paper |
| LongMemEval | vendor claim | 94.4% | Mem0 | mem0.ai blog, 2026 | **vendor** |
| LongMemEval-V2 S/M | accuracy | 74.9 / 70.1 | AgentRunbook-C | arXiv 2605.12493 | paper |
| LongMemEval-V2 S/M | accuracy (RAG class) | 58.6 / 57.0 | AgentRunbook-R | arXiv 2605.12493 | paper |
| LongMemEval-V2 | overall | 69.3, ≈182 s/query | vanilla Codex agent | arXiv 2605.12493 | paper |
| LongMemEval-V2 | **LAFS, tier small** | reference frontier = **55.765** | RAG 51.0@0.2s · AR-R 58.6@26.9s · AR-C 74.9@108.3s (Codex dominated) | computed from `leaderboard/compute_lafs.py` | source + probed |
| LongMemEval-V2 | **LAFS, tier medium** | reference frontier = **51.074** | RAG 45.9@0.3s · AR-R 57.0@25.8s · AR-C 70.1@139.9s (Codex dominated) | computed from `leaderboard/compute_lafs.py` | source + probed |
| DMR | accuracy | 98.2% @ gpt-4o-mini / 94.8% @ gpt-4-turbo | Zep | `10.48550/arxiv.2501.13956` | paper — saturated |

Note the internal inconsistency in the earlier evidence table (`04-benchmarks.md` §SOTA marks Zep 71.2% as best
LongMemEval_S while listing NEMORI at 74.6 as "2nd"): NEMORI is higher, and MemPro's 79.00 supersedes both.

Every number in this table was re-checked against its primary source by three independent adversarial
verification passes; the verdict tables, including the defects they found in an earlier draft of this plan,
are in [`docs/research/audit/`](docs/research/audit/). Claims that survived unchanged are marked CONFIRMED
there; three were corrected and one was dropped as unverifiable.

**The comparability problem, and its solution.** Those numbers use gpt-4o-mini-class answer models; we serve
`qwen3.8-27b`. Comparing our local model against their frontier model — in either direction — proves nothing.
MemPro fortunately publishes both backbones, verified from arXiv 2606.00619v1 Table 1:

| backbone | LongMemEval avg | LoCoMo avg |
|---|---|---|
| gpt-4o-mini | 79.00 | 84.93 |
| Qwen3-30B-A3B-Instruct-2507 | **80.80** | **77.85** |

So the **open-weights gate is LongMemEval_S ≥ 80.80 and LoCoMo(n=1540) ≥ 77.85**, which is the honest target
for a locally-served system, and the frontier-model rows stay in the table as context rather than as the bar.
Note that the open-weights backbone scores *higher* on LongMemEval and lower on LoCoMo — further evidence that
cross-model comparison on these benchmarks is not monotonic and must not be hand-waved.

Efficiency and robustness targets, from the same literature:

- Token cost ≤ 1/10 full-context — Mem0 reports >90% token saving and 91% lower p95 (1.44 s vs ≈17 s) **against
  the full-context baseline on LoCoMo with gpt-4o-mini**, and +26% relative LLM-judge against the ChatGPT
  memory baseline; Zep 115k → 1.6k average context tokens on gpt-4o-mini.
- p95 query latency < 100 ms for `recall` search+fuse+rerank (SwiftMem's 10.834 ms search-only is the floor).
- MINJA ASR ≤ 10% with pre-populated memory at k = 6 (literature: 62% → 6.7% when memory is pre-populated).

---

## 12. Milestones

Each milestone ends with a runnable command and a number, not a description.

| # | Milestone | Exit criterion |
|---|---|---|
| M0 | Workspace + probes as tests | `cargo test -p myelin-core --features integration` asserts §5.1–5.2 probe results against live Qdrant, including the k = 1 RRF constant and the BM25 IDF arithmetic |
| M1 | Store + ledger + record model | Property tests for I1–I5; `export`/`import` round-trips a built namespace byte-faithfully with a config-equal check (R3); `reconcile` repairs an artificially drifted store |
| M2 | Home-cloud model prerequisites | `Qwen/Qwen3.5-9B` and `Qwen/Qwen3-Embedding-8B` served on `big` per `skill://serve-gguf-on-big`, both resident simultaneously under a `gpu-tenant claim`, with measured VRAM peak and a tool-calling smoke test returning `finish_reason: "tool_calls"` |
| M3 | Write path end to end | Ingest all 10 LoCoMo conversations **and** the LME-V2-Small haystack via `insert(trajectory)` (R2); report records/unit, tokens, wall time; extraction precision/recall on 50 hand-labelled episodes |
| M4 | `recall` read path | Ablation table (`EVALUATION.md` §8) on a LoCoMo dev split: BM25-only / dense-only / hybrid / hybrid+rerank, plus RRF k=1 vs k=60; must reproduce the BM25 > dense ordering of §2 finding 1 or explain why not |
| M5 | `myelin-eval` skeleton + official harness wired | `memory_modules/myelin.py` registered; the benchmark's own `tests/test_query_privacy.py` passes against our adapter (R6); `no_retrieval` and `rag` reference baselines reproduce their published numbers within CI |
| M6 | **G1 break-even** | A single `recall` operating point on LME-V2-Small clears accuracy > 51.1 at ≤1 s ⇒ positive LAFS gain, with a full manifest and the judge-free 295-item column reported separately |
| M7 | `investigate` mode | Marginal-step-value curve over `max_steps` ∈ {1,2,4,6,8}; accuracy above `recall` at a stated latency; agentic metrics (steps-to-answer, tool-selection error rate, wasted retrieval) reported |
| M8 | **G1 headline** | Two operating points on LME-V2-Small and -Medium, 3 seeds, bootstrap CIs; target `recall` ≥65 @ ≤1 s + `investigate` ≥80 @ ≤40 s ⇒ +13.79 LAFS gain on `small`; leaderboard package built by the official two-step tool and validated |
| M9 | **G2** | LoCoMo(1540/1986/abstention) and LongMemEval_S with the 3-judge panel at κ ≥ 0.89 and calibration ρ ≥ 0.9; paired CIs vs baselines; contamination probe reported per model |
| M10 | `myelin-mcp` | Server passes a real MCP client handshake; the nine tools exercised against a live store; `explain` returns a lineage tree; `recall`/`investigate` selectable per call on one store (R4) |
| M11 | **G3** | E1–E6 of `EVALUATION.md` §7: ASR ≤ 10% at k=6 pre-populated, ≥90% templated-poison quarantine, zero cross-tenant leaks, unlearning invariant holds |

M0 and M5 are not ceremony. M0 pins the four probe findings in §5 — exactly the kind of thing a Qdrant point
release changes underneath us. M5 pins the benchmark's own privacy test against our adapter, which is the
only mechanical defence against accidentally optimising on metadata we are forbidden to see.

---

## 13. Risks

| risk | evidence it is real | mitigation |
|---|---|---|
| Judge disagreement swamps the effect we are measuring | LoCoMo/LongMemEval/BEAM leaderboards disagree; "LoCoMo Refined" exists because of it | 3-judge panel + κ + deterministic scorers + published prompts; report κ with every number |
| Comparing our open-weights answer model against papers' gpt-4o-mini | direct threat to G2's validity | G2 is stated against MemPro's *own* Qwen3-30B-A3B row (80.80 / 77.85); never compare across answer-model classes |
| Benchmark contamination in local models | SWE-Bench+ found 32.67% solution leakage; ABC / LiveCodeBench document the pattern | closed-book probe on 50 items per model before use; keep adversarial/unanswerable items; report the check |
| Qdrant 1.19 quirks (REST multivector, `IdfParams`, k = 1 RRF) | all three measured this session | M0 pins them as tests; client-side fusion removes the k dependency |
| Extraction quality dominates end-to-end quality | HippoRAG's error analysis; `01-systems.md` finding 3 | M3 measures extraction directly, separately from retrieval |
| Synonym-edge explosion in the graph | 1,125,951 synonym vs 140,830 extracted edges on MuSiQue | threshold + degree cap; edge count is a tracked metric |
| GPU contention makes latency numbers invalid, not just slow | measured: distill's 4,893 MiB caused `cudaMalloc failed` for a 27B model; claiming freed the card to 343 MiB | harness takes a `gpu-tenant claim`, records tenancy + VRAM peak, and **fails fast** if another tenant holds it; `memory_query_avg_seconds` is half of LAFS |
| A leaderboard-valid LME-V2 run **requires** a `gpt-5.2` judge — an external, paid dependency | step-1 validator checks the judge model string | only 156 of 451 questions need a judge; the other 295 are deterministic, so the external spend is bounded and the judge-free column is always published; a local judge panel runs the dev loop and its delta against `gpt-5.2` is tracked |
| Silent failure masquerading as success | llama-swap returns HTTP 200 with a zero-byte body when a model fails to load; this hid the stall for ~21 h | R7: the `Llm` trait rejects empty completions as errors; the harness treats them as run failures |
| LAFS reference frontier is hard-coded from the paper | `compute_lafs.py` embeds the four reference points | we add operating points to the released frontier as the tool intends and never re-derive the baselines; disclosed in `EVALUATION.md` §11 |
| Optimising on forbidden metadata by accident | the benchmark ships a test that forbids it | run the benchmark's own `tests/test_query_privacy.py` against our adapter in CI (M5) |
| Scope creep into a graph database | GraphRAG insert ≈14M tokens; QueryLink beats Zep without a graph | graph stays a SQLite table + `petgraph` in memory; revisit only with an eval delta to justify it |

---

## 14. Non-goals for v1

No RL fine-tuning of memory policies (§2 finding 10). No parametric consolidation into model weights. No
distributed/multi-node store. No Postgres. No community-summary graph layer. No cross-encoder training. No
web UI — the MCP surface and the eval report are the interfaces.

---

## 15. Immediate next step

Two things, in this order.

1. **M0** — scaffold the workspace and turn `docs/research/00-verified-environment.md` §3 into
   `crates/myelin-core/tests/qdrant_capability.rs`, so the four probe findings become assertions that fail
   loudly when the environment shifts. The probe code that produced them already exists, is preserved in
   [`docs/probes/qdrant_capability.rs`](docs/probes/qdrant_capability.rs), and ran green against the live
   instance this session.
2. **M2** — serve `Qwen/Qwen3.5-9B` and `Qwen/Qwen3-Embedding-8B` on `big` per
   `skill://serve-gguf-on-big`. Nothing about G1 can be measured until the pinned reader and embedder exist
   locally, and the VRAM envelope (§7 of the environment doc) says they only coexist under a
   `gpu-tenant claim`. Establishing that envelope early de-risks every later milestone.
