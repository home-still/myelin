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

`petgraph` is an unconditional dependency, not a feature. It was gated as `graph` until M12 put PPR
on the default read path (`RetrieveConfig::graph`), at which point a cargo feature was gating the
module rather than the mechanism.

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
graph?        → PPR over phrase↔record incidence, a third ranked list  (§2 finding 7; built in
                M12, default OFF — docs/measurements/m12-graph-route.md)
fuse          → RRF in Rust, k = 60 (configurable; Qdrant's own is k = 1, §5.2)
rerank        → late-interaction max_sim in Qdrant, or bge-reranker cross-encoder   (§2 finding 2)
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
returns compact evidence. Two write tools, five read tools, three admin tools. Every tool returns provenance.

| tool | args | returns | notes |
|---|---|---|---|
| `remember` | `text, kind?, scope?, t_valid?, source?, as_profile?` | `Delta[]` applied + quarantined count | runs the full write path; idempotent by content hash. `as_profile` asserts the text verbatim as a `Profile` record instead of extracting facts, through the same consolidator — so re-asserting a changed preference supersedes the old one |
| `observe` | `turns[]` | episode ids | bulk ingest of a conversation/trajectory segment |
| `recall` | `query, scope?, k?, budget_tokens?, kinds?` | `EvidenceSet` | fast mode, §7.1 |
| `investigate` | `question, scope?, max_steps?, budget_tokens?` | `EvidenceSet` + trace | agentic mode, §7.2 |
| `search` | `query, filter?, limit?` | record stubs | primitive for caller-driven iteration |
| `profile` | `tenant, namespace?, agent?, limit?` | record stubs | the user's durable dispositions, newest first; fetched by scope, not by relevance (M20) |
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

Published numbers to beat live in **[`docs/sota/registry.json`](docs/sota/registry.json)**, one row per claim,
each carrying its provenance flag (`paper` | `leaderboard` | `vendor` | `abstract_only` | `paywalled` |
`project_gate`), the population `n` it is over, the answer-model and judge classes behind it, and a verbatim
quote with line numbers from the locally converted source. There is no table here any more: two copies of the
published numbers is the failure M18 existed to remove.

```
myelin-eval standing            # join the registry against runs/, print a verdict per row
myelin-eval standing --gate     # exit 1 while any gate row is unsupported or unbeaten
```

`standing` classifies every row `comparable` | `caveat-judge` | `caveat-backbone` |
`not-comparable(subset|source)` | `incomplete-artifact` | `missing-artifact`, computes the gap only where a
gap is a quantity, and refuses to let an unciteable number satisfy a gate. The gate rows are the five below;
the verdict as of M18 is in [`docs/measurements/m18-sota-standing.md`](docs/measurements/m18-sota-standing.md).

| gate row | bar |
|---|---|
| `locomo.judge.mempro15.qwen3_30b` | LoCoMo(n=1540) judge score ≥ **77.85** |
| `longmemeval_s.judge.mempro15.qwen3_30b` | LongMemEval_S judge score ≥ **80.80** |
| `lme_v2_small.agentrunbook_c` | LME-V2-Small (n=451) ≥ **74.9** |
| `lme_v2_small.lafs_gain.frontier` | LAFS gain, tier small, **> 0** (strict: a tie is what a dominated point scores) |
| `minja.asr.g3_gate` | MINJA-style ASR at k=6, pre-populated, defended ≤ **10%** |

Note the internal inconsistency in the earlier evidence table (`04-benchmarks.md` §SOTA marks Zep 71.2% as best
LongMemEval_S while listing NEMORI at 74.6 as "2nd"): NEMORI is higher, and MemPro's 79.00 supersedes both —
and EverMemOS's 83.00, which this plan never carried, supersedes all three at a frontier backbone.

Every number was re-checked against its primary source by three independent adversarial verification passes;
the verdict tables, including the defects they found in an earlier draft of this plan, are in
[`docs/research/audit/`](docs/research/audit/). M18's re-verification pass found four more, all now corrected
in the registry's `caveat` fields: GAM's 40.00 is the **Qwen2.5-7B** row and a macro mean of four category F1s
(its gpt-4o-mini row is 43.14), NEMORI's 74.6 is **gpt-4.1-mini** (its gpt-4o-mini LongMemEval average is
64.2), vanilla Codex is 69.9 at 177.2 s on tier small (69.3 ≈182 s is the abstract's tier-unspecific figure),
and the "62% → 6.7% when memory is pre-populated" attributed to MINJA is actually the EHR-poisoning paper's
Table 1 at k=3 — MINJA contains no such pair.

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
| M10 | `myelin-mcp` | Server passes a real MCP client handshake; the ten tools exercised against a live store; `explain` returns a lineage tree; `recall`/`investigate` selectable per call on one store (R4) |
| M11 | **G3** | E1–E6 of `EVALUATION.md` §7: ASR ≤ 10% at k=6 pre-populated, ≥90% templated-poison quarantine, zero cross-tenant leaks, unlearning invariant holds |
| M12 | graph route (`EVALUATION.md` §8 row 7) | PPR fused as a third channel behind one switch, incidence backfilled on both benched corpora with no GPU, paired per-category CIs on both; default set by a rule fixed in advance. **Measured: no gain in any category, −0.7 pts on LongMemEval_S (CI [−1.5, −0.1]) ⇒ default stays off** (`docs/measurements/m12-graph-route.md`) |
| M13 | temporal axis | Evidence order (`ComposeConfig::chronological`) and a LoCoMo `<today>` (`bench --question-date`) as two independent query-time switches, each measured alone against the M9 baselines with paired per-category CIs on both corpora; defaults set by a rule fixed in advance. **Measured: neither moves its target stratum — LoCoMo cat 2 +0.2 (CI [−1.8, +2.2]) for `question_date`, −0.3 (CI [−1.9, +1.3]) for `chronological`, LME_S temporal-reasoning −1.1 (CI [−5.2, +3.1]) ⇒ both defaults stay off.** Off-target and worth keeping: `<today>` is +3.1 pts of LoCoMo abstention accuracy (CI [+0.7, +5.6]) and time order is +8.0 pts on LME_S knowledge-update (`docs/measurements/m13-temporal-axis.md`) |
| M14 | temporal scorer | Date-aware deterministic scoring (`myelin_eval::temporal`) beside token F1, both columns on every row; an offline `myelin-eval rescore` that re-read all eleven runs on disk at zero GPU cost; a reader-only `myelin-eval judge` as arbiter; default set by an agreement rule fixed in advance. **Measured: the date-aware scorer agrees with the judge 96.7% vs token F1's 84.9% on LoCoMo cat 2 (+11.8 pts, CI [+7.7, +15.8], κ 0.60 → 0.91), and −0.02 pts off-stratum ⇒ default flips to `temporal` for LoCoMo; `token-f1` kept for LongMemEval_S, where only 26/470 golds resolve.** Token F1 had inflated LoCoMo cat 2 by 8.03 pts (0.2825 → 0.2022) with partial credit for anchor-instead-of-offset answers; M12's and M13's verdicts all survive the re-read (`docs/measurements/m14-temporal-scorer.md`) |
| M15 | injection adjudication (G3) | A content-level injection adjudicator (`pipeline::adjudicate`) on every episode **before** it is written, independent of the declared `SourceTier`; E1 widened from 5 to 40 attacks (8 surface forms × 5 domains) plus a 10-item ungated adaptive probe; measured off and on at both `Untrusted` and `Asserted` with Wilson intervals, and a false-positive pass over all 550 real LoCoMo episodes; default set by a rule fixed in advance. **Measured: ASR 77.5% [62.5, 87.7] → 15.0% [7.1, 29.1] at k=6 pre-populated — 62.5 points, and identical at `Asserted` (75.0% → 15.0%), with 0/550 LoCoMo and 0/12 benign false positives — but the gate is ≤10%, so `adjudicate` ships off and G3 stays open with a measured bound.** Every surviving attack is one of two surface forms (forged audit provenance, negating redirect) whose only defect is being false; the ungated adaptive probe (no mechanic at all) is admitted 10/10 by design, which is the ceiling of any content classifier (`docs/measurements/m15-injection-adjudication.md`) |
| M16 | evidence sufficiency audit (G1) | A reader-only `myelin-eval evidence-audit` over the vendored LME-V2 harness rows: for every answerable question the harness scored wrong, a constrained-decoding judge over the **evidence the reader was actually shown** splits the loss into retrieval's and the reader's, with Wilson intervals, per-category cells, a deterministic gold-token proxy, both counterfactual ceilings and five verbatim labels per diagnostic cell; run at **two** operating points (646 judgements, 4 runs, 902 questions); branch set by a rule fixed in advance. **Measured: S = P(sufficient \| wrong) = 7.4% [4.4, 12.0] at `recall` k=25 and 12.2% [8.1, 17.9] at `investigate max_steps=2` ⇒ retrieval-limited at both. A perfect reader over today's evidence reaches 38.8% / 44.6% against the 51.0 bar (50.8% for web alone at the better point, 0.2 short); retrieval repair reaches 66–68% at the measured P(correct \| sufficient) ≈ 78–82% ⇒ M17 runs the pre-registered width arm, and no reader- or prompt-side change can reach G1.** *insufficient + wrong* is the largest cell everywhere (52.4% / 56.8% of answerable at k=25). The diagnosis was tested out of sample: the agentic point shrank that cell 88→68 (web) and 88→83 (enterprise) and answerable accuracy rose **+14 against +16.5 predicted** and **+4 against +4.0**. The pre-registered instrument clause fired (*insufficient + correct* 20.2–23.9% > 20%) and the verdict survives it: the judge's label moves P(correct) 28.5% → 81.8% (+53.4 pts) without ever seeing the answer, ~40% of that cell is multiple-choice guessing above chance, and the maximally adversarial worst case (S = 56.1%) selects the same width arm. Side result: **`investigate max_steps=2` at full set is +8.3 pts on web (CI [+2.9, +13.8], p = 0.0026) but −0.5 on enterprise with a significant −8.9-pt abstention loss**, so M7's levels are domain-specific, not a global default (`docs/measurements/m16-evidence-sufficiency.md`) |
| M18 | SOTA standing, verified programmatically | `docs/sota/registry.json` (25 published claims, 10 locally converted sources, verbatim quotes with line numbers, per-row `n`/judge-class/backbone-class/provenance) joined against the run artifacts by a new `myelin-eval standing` that replaces the unimplemented `report`: it classifies every comparison `comparable` / `caveat-judge` / `caveat-backbone` / `not-comparable(subset\|source)` / `incomplete-artifact` / `missing-artifact`, computes the gap only where a gap is a quantity, calls the leaderboard's own `compute_lafs.py` through `adapters/lafs_point.py` rather than reimplementing it, and exits 1 under `--gate`. Our side became artifacts in the same pass: judged columns on both G2 runs (1,477 verdicts) and the first serialised G3 sweep (`runs/attack_live_m18/attack_live.json`, 48 min, six conditions × 8 cohorts). **Measured: all five gates unsupported, every one on quality rather than a missing artifact — LoCoMo judged 62.66 vs the 77.85 bar (−15.19), LongMemEval_S 52.00 vs 80.80 (−28.80), LME-V2-Small 39.91 vs 74.9 (−34.99), LAFS gain exactly 0.0 (both operating points dominated), MINJA ASR defended 12.5% vs the ≤10% bar. Exactly one published claim is beaten (GAM's LoCoMo token F1, +13.15) and 17 of 25 rows carry a judge-class caveat. Four defects found in §11.5's own table (GAM's backbone and metric, NEMORI's backbone, Codex's tier figure, and a "62% → 6.7%" pre-population pair attributed to MINJA that is in the EHR-poisoning paper).** Branch: M19 takes the one apples-to-apples headroom in the table — AgentRunbook-R's 58.60 at the same `Qwen3.5-9B` reader against our 39.91 — as write-time runbook synthesis (`docs/measurements/m18-sota-standing.md`) |
| M19 | **temporal resolution (G2)** | Time made computable rather than textual: the M14 temporal grammar moved into `myelin_core::time` so the read path can call it, plus an anchored `resolve_relative` over exactly the unanchored forms the gold grammar refuses, an `is_interval_question` classifier, a `[timeline]` dated index gated per query, persisted composed evidence on every bench row, a `--categories` stratum filter and a judge-backed `rescore --scorer judge`; three mechanisms measured alone and in combination on their own strata against rules fixed in advance, with paired CIs quoted from the CLI. **Measured: the resolved annotation is +37.6 pts on LoCoMo cat 2 (n=321, CI [+32.2, +43.2]), the reader-side date clause +14.3 ([+10.2, +18.7]), both together +42.8 ([+37.3, +48.5]) — and the marginals say both are needed (+5.2 and +28.4) ⇒ both ship on. The dated index is +6.8 on LongMemEval temporal-reasoning (CI [+3.0, +11.3]) and +6.8 again at k=25, while breadth alone is +3.8 with a CI spanning zero and `investigate` is exactly 0.0 at 6.5× the latency ⇒ the index ships on, `k`/`mode` stay per-call (R4).** Full-set judged: LoCoMo 62.66 → **69.87** (+7.2, CI [+5.5, +9.0]) and LongMemEval_S 52.00 → **56.40** (+4.4, CI [+1.6, +7.2]), halving the LoCoMo G2 gap from −15.19 to **−7.98**, with no off-target category moving on an interval that excludes zero. The diagnosis is per question rather than inferred: 103 of the 321 answers were bare relative expressions against absolute golds, the gold evidence turn was in the composed evidence for **101 of them** (so retrieval had found it), and 94 now carry an absolute date. Found and fixed on the way: **LongMemEval_S's session timestamps had never parsed**, so all 162,181 records carried the build date as `t_valid` and every one of its 500 memories had been showing the reader `[2026-09-15]` since M13 — an in-place repair is correctly refused by I1's trigger, so the corpus was re-ingested, worth 45 → 19 declines on its 61 duration questions on its own (`docs/measurements/m19-temporal-resolution.md`) |
| M20 | preference/persona profile layer | A typed `RecordKind::Profile` minted by a user-turn-only extraction pass (`WritePath::extract_profiles`, 12.6% of LongMemEval_S's tokens, model call skipped entirely below `PROFILE_MIN_CHARS`), lineage-bound by I4, fetched by scope through a kind-filtered `Ledger::visible_of_kind`, and composed as an always-emitted `[profile]` block at the head of the evidence set; plus a reader clause, a tenth MCP tool `profile`, `remember --as_profile` with supersession through the existing `Delta::Update`, and `build --question-types`/`--concurrency`. Two arms measured alone and together on `single-session-preference` (n=30, judged) against a rule fixed in advance, CIs from the CLI. **Measured: arm A (the block) +0.0 (CI [−16.7, +16.7]), arm B (the clause) +3.3 ([−10.0, +16.7]), A+B +6.7 ([−10.0, +23.3]) ⇒ both defaults stay off.** The null has a cause: the store holds a median of **124 dispositions per tenant** and `PROFILE_MAX_RECORDS = 8` selects by recency, so the block carries 6.5% of them at a gold-content recall of **0.042** (9 of 30 questions at zero). The clause alone moves declines 11 → 7. Selection among in-scope material — neither retrieval nor generation — is the open problem (`docs/measurements/m20-preference-profile.md`) |
| M21 | **evidence selection, and the instrument for it** | `myelin-eval coverage`: gold-turn recall of the *composed* evidence computed offline and deterministically from each corpus's own annotation (LongMemEval_S `has_answer`, LoCoMo `dia_id`), written to `<run>/coverage.json`, pinned against the three M19 runs (0.662 / 0.653 / 0.852, `all = 106`) and hard-erroring on the twelve pre-M19 artifacts that record no evidence rather than reporting their 0.000 as a measurement. Two selectors then measured against a rule fixed in advance: `ComposeConfig::mmr_lambda` (maximal marginal relevance, rank-based relevance so no cross-encoder logit is ever mixed with a cosine) and `pipeline::select::Selector` + `Retriever::with_llm` + `RetrieveConfig::select_sufficient` (ask the model which candidates jointly answer, stable-partitioned so nothing is dropped). **Measured: MMR is a regression — recall 0.658 → 0.550 / 0.471, judged −5.3 ([−11.3, +0.0]) and −9.8 ([−15.8, −3.8], p = 0.0018) — because the memories that jointly answer one question resemble *each other* 1.60× more than they resemble the rest of the set, so a redundancy penalty is aimed at co-evidence. The sufficiency selector clears the bar — recall 0.658 → 0.838 / 0.809 against a 0.852 pool ceiling, judged +7.5 ([−0.8, +15.8]) and +6.0 ([+1.5, +10.5], p = 0.0081), +3.8 over all 500 ([+1.0, +6.6], p = 0.0087) with no stratum regressing — and ships off anyway: §7.1 forbids an LLM in `recall`, and in `investigate`, the one path it was allowed to default on, it is exactly +0.0 ([−3.8, +3.8]) at +1.87 s/query because that loop unions its probes into a 60-record pool and re-composes. Does not transfer to LoCoMo multi-hop (−3.2, CI [−7.8, +1.1]; recall 0.478 → 0.367). All three switches off.** Also fixed `standing`, which was publishing the 60.40 arm instead of the 56.60 shipped configuration: selection is now complete → shipped-default → population → value. `docs/measurements/m21-evidence-selection.md` |
| M22 | **G1 at its own operating point** | Pool-level selection (`investigate.rs::select_pool` replacing M21's per-probe view, one question-conditioned judgement over the accumulated pool because its cross-probe `score`s are logits from different queries and not mutually comparable), an undated-corpus gate (`dated`, suppressing `stamp_valid_time`/`resolve_relative`/`timeline`), both reachable as query-time operating points on **both** MCP tools — plus the missing `.with_llm` that had made the selector inert on the MCP path, `select`/`dated` in `standing::PAIR_KEYS`, and a population guard that stops a `--limit` pilot pairing into a 451-question submission. Three arms over the full tier-small pair (web 240 + enterprise 211), CIs from the CLI over the pooled 451, `evidence-audit` beside every accuracy, against a rule fixed in advance. **Measured: both switches are nulls — `dated=false` 39.02 vs a fresh same-code base of 36.59 (+2.4, CI [−1.1, +6.0]) and `select` 37.92, *negative* against `nodate` (−1.1, CI [−4.7, +2.4]) ⇒ both ship off and G1's 51.0 break-even is not claimed.** Three findings outrank the nulls: all **85,589** LME-V2 records carry the build date as `t_valid` (one calendar day, `t_valid == t_ingested` to the nanosecond) so M19's two shipped compose defaults have been annotating against a meaningless anchor — a `[timeline]` block on 240/240 web questions, and suppressing them is worth **−7.4 s/query**; the fresh base is **3.3 points below** the 39.91 `standing` still publishes, which pre-M19 code produced; and M16's S = P(sufficient \| wrong) reproduces at **12.1% [8.1, 17.6]** against its 12.2%, so the reader-fix ceiling (41.5) is still below break-even and retrieval is still the binding constraint (`docs/measurements/m22-g1-selection.md`) |
| M23 | **the progression ratchet** | The instrument that compares us to *ourselves*: `Ours::unrecorded` + `Verdict::StaleConfig` refuse to publish an artifact that does not record the operating point it ran at, drift-size ordering prefers the artifact closest to today's code, `harness_arm` computes `arm` on the LME-V2 path for the first time, and `myelin-eval ratchet` pins a floor per metric in `docs/sota/progression.json` and exits non-zero below it — quotable rows only, direction read from the metric so `minja.asr.*` is not graded upside down, `--update` raising only. Plus the measurement legs M23's mechanism arms were missing: `bench --rerank-pool/--premise/--typed-probes/--untrusted-max`, recorded in `BenchRun`, read back by `rescore`, and all four counted as arms. **Found: `standing` had been publishing 39.91 on `lme_v2_small.overall_full_set.combined` from a pre-M19 artifact; the shipped configuration is 36.59. Three separate defects — no era signal, value-ordering among stale artifacts, and `arm` hardcoded `false` on the LME-V2 path since it was written, which was publishing M22's measured-null `dated=false` arm at 39.02.** No mechanism ships on; the reader-dependent arms were deferred rather than taken by evicting a live household tenant off the GPU (`docs/measurements/m23-progression-ratchet.md`) |

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
| Synonym-edge explosion in the graph | 1,125,951 synonym vs 140,830 extracted edges on MuSiQue | no synonym edges are created at all — exact-phrase incidence only, capped at 32 phrases/record; edge count is a tracked metric and `myelin-eval phrases` reports it (M12: 17,794 on LoCoMo, 2,762,496 on LongMemEval_S) |
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

M0–M23 are done and committed. `docs/measurements/` carries one file per milestone; the standing table
against the published literature is `docs/sota/registry.json` + `runs/standing/`, and the floor under
our own numbers is `docs/sota/progression.json` + `myelin-eval ratchet`.

**M24 — parallel query decomposition: retrieve for the sub-questions, not the question.**

Every retrieval mechanism this project has measured since M12 has been a way of *reordering or
trimming* one candidate pool drawn from one query. Graph fusion (M12), evidence order (M13),
width at k=25 (M19), MMR and sufficiency selection (M21), pool-level selection (M22): six
mechanisms, and the only one that moved a stratum was M19's, which changed **what the records
said** rather than which records were drawn. The pattern is not an accident and the literature
names the alternative.

**AgentRunbook-R is the case to answer.** It scores **58.60** on LME-V2-Small with the same
Qwen3.5-9B reader we serve, against our 36.59 — the only apples-to-apples comparison in the
registry, and now a 22.0-point gap rather than the 18.7 the stale artifact implied. The
vendored implementation is in-repo at
`crates/myelin-eval/vendor/longmemeval-v2/memory_modules/agentrunbook_r.py`, so the mechanism
can be read rather than inferred, and it is not the three pools:

- **One** model call emits a *structured bundle* of sub-queries —
  `{"raw_state_queries": [≤5], "event_query": str, "note_query": str}` (`QUERY_GENERATION_SYSTEM_PROMPT`,
  L126–162). Each sub-query names a distinct surface, and the prompt explicitly forbids splitting
  one surface into attribute-level queries.
- Each sub-query is retrieved **separately and in parallel**, then each block is reranked against
  the **original question** and selected independently (`_query_with_rerank`, L1245–1324).
- The blocks are composed in a fixed order, so every question gets evidence from every pool that
  answered.

Our `investigate` is the sequential dual of this: `step_k = 10`, `max_steps = 2`, one query per
step, each conditioned on what the last step returned. M19 measured two steps at **exactly 0.0**
on LongMemEval temporal and M22 measured pool selection at **−1.1** on LME-V2 — both are
*sequential* mechanisms, and both are null. Parallel decomposition has never been tried here.

Why it is the largest remaining item, by the research brief's own ranking
(`docs/sota/research-brief.md` §2, answers-at-gate ÷ cost-of-one-arm):

- It is RQ2's option (b) literally — "query decomposition into sub-questions with separate
  retrievals" — and RQ2 is **+99 answers at the gate profile** (49 LoCoMo multi-hop + 50
  LongMemEval multi-session), 201 of headroom, the largest pool on the board.
- M16 and M22 both measure us **retrieval-limited**: S = P(sufficient | wrong) = 12.1%
  [8.1, 17.6], so a perfect reader over today's evidence reaches 41.5 and a perfect retrieval
  reaches 76.9. Only a mechanism that changes *which records are drawn* can move that, and
  decomposition draws from N queries where every prior arm drew from one.
- It is a **read-path** arm: minutes per stratum, no re-ingest. D1's typed pools cost a build;
  decomposition does not, and the two are independent — the pools make decomposition *better*,
  they are not a prerequisite for measuring it.
- Latency is affordable. The measured points are 10.95–18.31 s against a 40 s budget, and LAFS
  gain is 0.00 at every point because the frontier dominates on **accuracy**, not time.

Constraints to carry in, all measured:

- **Re-measure the base on the same store, every time.** M23 makes this mechanical: any artifact
  that does not record its operating point is `stale-config` and unpublishable, and `ratchet
  --strict` fails on it. The first thing M24 owes is a fresh base pair at the shipped defaults.
- **Decomposition must not become selection wearing a hat.** M21's verdict is that the memories
  which jointly answer one question resemble *each other* 1.60× more than the rest of the set. A
  decomposition that de-duplicates across sub-queries will re-inflict that. Union, then rank; do
  not diversify.
- **The sub-query generator is an LLM in the loop, so it cannot default on in `recall`** (§7.1).
  `investigate` is where it may default, which is the same constraint the sufficiency selector
  had — and the reason to measure it on `bench --categories` strata first, where `recall` and
  `investigate` are both available and the population is 282 or 133 rather than 451.
- **Rerank each block against the original question, not the sub-query.** This is AgentRunbook-R's
  own choice and it is the same finding M22 reached independently: cross-probe scores are logits
  from different queries and are not mutually comparable. `InvestigateConfig::rerank_pool` (M23 A2)
  already implements the pool-level half and is unmeasured; decomposition and it should be
  measured as separate arms before either is combined.

Pre-registered rule, fixed before any arm runs: decomposition ships on for `investigate` if it
beats the fresh same-code base by **≥ +5.0 judged points** on LoCoMo multi-hop (n=282) *or*
LongMemEval multi-session (n=133) with a paired CI excluding zero, and regresses no other
stratum by more than 2.0. Anything else is a measured null and the switch stays off.

