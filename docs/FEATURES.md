# Features

This document is the complete feature reference for myelin, derived from the
source code. It covers the 10 MCP tools, both read paths, the write path,
retrieval architecture, multi-tenancy, configuration, and the eval harness.

## MCP tool surface

myelin exposes 10 tools through the Model Context Protocol. The server
(`myelin-mcp`) supports both stdio and streamable-HTTP (`--serve HOST:PORT`)
transports. Source of truth: `crates/myelin-mcp/src/server.rs`.

### `recall` — fast hybrid retrieval

Fast hybrid retrieval over one tenant's memory. BM25 + dense in one round
trip, RRF-fused, cross-encoder reranked, budgeted and deduplicated. **No LLM
in the loop** — this is the fast path, pinned at "no LLM" by design (PLAN.md
§7.1).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | — | The question, verbatim |
| `tenant` | string | yes | — | Mandatory; no read path spans tenants (C12) |
| `namespace` | string | no | none | Scope filter |
| `agent` | string | no | none | Scope filter |
| `session` | string | no | none | Scope filter |
| `k` | usize | no | 6 | Evidence-set size |
| `budget_tokens` | usize | no | 2048 | Token ceiling for composed set |
| `kinds` | [string] | no | none | Filter: `episodic`, `semantic`, `procedural`, `working` |
| `tau_abstain` | f32 | no | none | Withhold evidence when top score below this |
| `select` | bool | no | false | LLM selects jointly-answering candidates (operating point, never a `recall` default) |
| `dated` | bool | no | true | Whether records carry real timestamps |
| `decompose` | usize | no | none | Split into N sub-queries, fuse into same RRF call |

**Returns:** `RecallResult` — `items` (R1 wire shape `{type, value}`),
`record_ids`, `tokens`, and `trace` (retrieval diagnostics: hit counts,
latencies, top score, abstained flag).

**Example** (captured from a real run against `locomo/conv-26`):
```json
{
  "items": [
    {"type": "text", "value": "[2023-07-12] Melanie: ..."}
  ],
  "record_ids": ["8e0d200c-8eec-547f-a304-fab868c559b8"],
  "tokens": 1664,
  "trace": {
    "abstained": false, "admitted": 25, "dense_hits": 50,
    "embed_ms": 45, "fused": 54, "lex_hits": 5,
    "rerank_ms": 579, "reranked": 25, "search_ms": 58,
    "top_score": -6.42, "total_ms": 686
  }
}
```

### `investigate` — agentic search→reflect loop

Agentic retrieval: search, reflect, search again, until the evidence is
sufficient and self-consistent or the step budget runs out. Slower and more
accurate than `recall`; returns the same evidence shape with additional
`queries` and `trace` fields.

**`select_sufficient` default is ON for `investigate`** (M32): the LLM
selects which of the accumulated pool's records jointly answer the question,
once, before `compose` truncates to `k`. This is the only read path where it
defaults on — `recall` stays at "no LLM in the loop." Measured on
LongMemEval_S, all 500 questions: 56.2 → 62.0 judged (+5.8, 95% CI [+2.8,
+8.8], p = 0.0001), cost +2.04 s/query.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `question` | string | yes | — | The question |
| `tenant` | string | yes | — | Mandatory |
| `namespace` | string | no | none | Scope filter |
| `k` | usize | no | 6 | Final evidence-set size |
| `budget_tokens` | usize | no | 2048 | Token ceiling |
| `max_steps` | usize | no | 2 | Iteration cap (model calls in the gate) |
| `select` | bool | no | true | Select sufficient records from the pool (default ON) |
| `dated` | bool | no | true | Whether records carry real timestamps |
| `pool_rerank` | bool | no | false | Rerank accumulated pool against original question |
| `premise` | bool | no | false | Emit premise analysis when loop stops unsatisfied |
| `typed_probes` | bool | no | false | Let reflect gate aim probes at record kinds |
| `decompose` | usize | no | none | Split each probe into N sub-queries |

**Returns:** `InvestigateResult` — `items`, `record_ids`, `tokens`, `queries`
(list of probe queries the loop generated), `trace`.

**Example:**
```json
{
  "items": [{"type": "text", "value": "..."}],
  "record_ids": ["..."],
  "tokens": 1820,
  "queries": ["What did the user say about hiking?", "What outdoor activities does the user enjoy?"],
  "trace": {"...": "..."}
}
```

### `remember` — write one statement

Write one statement through the full write path: extract, consolidate against
neighbours, apply a four-op delta, index. Idempotent by content. Set
`as_profile` to assert it verbatim as a durable preference.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | yes | — | The statement to remember |
| `tenant` | string | yes | — | Scope |
| `agent` | string | no | `"myelin"` | Agent identifier |
| `namespace` | string | no | `"default"` | Namespace |
| `speaker` | string | no | `"user"` | Who said it |
| `source` | string | no | `"remember"` | Source label |
| `unit` | string | no | `"remember"` | Unit label |
| `t_valid` | datetime | no | now | When the fact became true |
| `as_profile` | bool | no | false | Assert as durable preference instead of extracting facts |

**Returns:** `WriteResult` — `added`, `updated`, `deleted`, `duplicates`,
`quarantined`, `adjudicated_out`, `profiles`, `wall_ms`, and more.

**Example** (captured from a real run):
```json
{
  "added": 1, "adjudicated_out": 0, "candidates": 1, "deleted": 0,
  "duplicates": 0, "episodes": 1, "noop": 0, "profiles": 1,
  "quarantined": 0, "rejected": 0, "updated": 0, "wall_ms": 237
}
```

### `observe` — bulk-ingest a conversation

Bulk-ingest a conversation or trajectory segment. Segmentation into episodes
is the server's job, not the caller's.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `turns` | [ObserveTurn] | yes | — | Array of turns |
| `tenant` | string | yes | — | Scope |
| `agent` | string | no | `"myelin"` | Agent identifier |
| `namespace` | string | no | `"default"` | Namespace |
| `unit` | string | no | `"observe"` | Unit label |

`ObserveTurn`: `speaker` (string), `text` (string), `at` (datetime, optional),
`source` (string, optional).

**Returns:** `WriteResult` (same shape as `remember`).

### `search` — list record stubs

Record stubs matching a scope filter. The primitive for caller-driven
iteration: identifiers and previews, never full records.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tenant` | string | yes | — | Scope |
| `namespace` | string | no | none | Scope filter |
| `agent` | string | no | none | Scope filter |
| `limit` | usize | no | 20 | Max records (capped at 200) |

**Returns:** `SearchResult` — `{ records: [RecordStub] }` where each stub has
`id`, `kind`, `preview` (180 chars), `t_valid`, `trust_tier`.

### `profile` — user preferences

What this user is known to prefer: their durable dispositions, newest first.
Retrieved by scope, not by relevance — a preference is about the user, not
about the question.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tenant` | string | yes | — | Scope |
| `namespace` | string | no | none | Scope filter |
| `agent` | string | no | none | Scope filter |
| `limit` | usize | no | 20 | Max records (capped at 200) |

**Returns:** `ProfileResult` — `{ records: [RecordStub] }`.

### `neighbors` — graph walk

Records linked to this one by a typed edge. One hop; call again to walk.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `record_id` | string | yes | — | UUID of the record |
| `relation` | string | no | none | Filter: `supersedes` or `derived_from` |

**Returns:** `NeighborsResult` — `{ neighbors: [{ relation, inbound, record }] }`.
`inbound: true` means the edge points *at* the queried record (direction matters:
"what replaced this" vs "what did this replace").

### `forget` — invalidate or erase

Invalidate a record (soft) or erase it and everything derived from it (hard).
Hard deletion requires `confirm: true` and cannot be undone.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `record_id` | string | yes | — | UUID of the record |
| `mode` | string | yes | — | `"soft"` or `"hard"` |
| `confirm` | bool | no | false | Required for `hard` mode |
| `reason` | string | no | `"forget via mcp"` | Audit reason |
| `actor` | string | no | `"mcp"` | Audit actor |

**Returns:** `ForgetResult` — `{ mode, affected: [uuid] }`. For hard delete,
`affected` includes all records in the descendant closure (C11 unlearning).

### `review_quarantine` — staged writes

Staged writes awaiting a decision, with the reason each was held.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | usize | no | 20 | Max staged records (capped at 200) |

**Returns:** `QuarantineResult` — `{ staged: [{ id, reason, at, record }] }`.

### `explain` — lineage and audit

Why this record exists: its lineage back to source episodes, and every audit
event that touched it.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `record_id` | string | yes | — | UUID of the record |

**Returns:** `Explanation` — `{ lineage: { id, kind, preview, ancestors: [...] },
events: [{ seq, at, kind, actor, reason }] }`.

---

## Read paths

### `recall` — fast, no LLM

`recall` is the fast read path. It runs hybrid retrieval (dense + BM25) in one
round trip, fuses with RRF (k=1), reranks with a cross-encoder, composes to k
records within a token budget, and returns. No LLM is involved — this is by
design (PLAN.md §7.1: "no LLM in the loop"). Typical latency: ~700 ms for k=3.

**When to use:** you know what you are looking for, you want it fast, and you
do not need the agent to reflect on whether the evidence is sufficient.

### `investigate` — agentic loop

`investigate` runs a search→reflect→search-again loop. Each step retrieves
evidence (using the same hybrid pipeline as `recall`), then the LLM reflects
on whether the evidence is sufficient and self-consistent. If not, it
generates a new probe query and searches again, up to `max_steps` (default 2).

After the loop, `select_sufficient` (default ON) asks the LLM which of the
accumulated pool's records jointly answer the question, and puts those first
before `compose` truncates to `k`. This is the mechanism that earned +5.8
judged points on LongMemEval_S (56.2 → 62.0).

**When to use:** the question is complex, multi-hop, or you are not sure the
first retrieval will find everything. Cost: ~2 s/query extra over `recall`.

### When to use which

| | `recall` | `investigate` |
|---|---|---|
| LLM in loop | no | yes (reflect + select) |
| Latency | ~700 ms | ~3–5 s |
| Accuracy | good for direct lookups | better for complex questions |
| `select_sufficient` | off (never a default) | on (M32 shipped default) |

---

## Write path

The write path (`remember` / `observe`) runs: **ingest → extract → adjudicate
→ consolidate → index**.

1. **Ingest:** segment the input into episodes. Raw episodes are stored
   losslessly before extraction runs, so if extraction fails the material is
   not lost.

2. **Extract:** the LLM pulls structured facts (candidates) from each episode.
   Each candidate has a kind (`episodic`, `semantic`, `procedural`, `profile`)
   and a text payload.

3. **Adjudicate (injection gate, M15):** judges the *text*, not the declared
   trust tier. Detects three injection mechanics: instruction override, forged
   provenance, and indication redirect. Off by default; when on, refused
   episodes are staged in quarantine, not dropped silently.

4. **Consolidate:** for each candidate, find the nearest stored fact. If cosine
   ≥ `tau_dup` (0.95), it is a duplicate — bump the existing record's salience.
   If cosine < `tau_relevant` (0.60), it is unrelated — ADD. Otherwise, ask the
   LLM to decide: ADD, UPDATE (supersede), or DELETE. The four-op delta
   (ADD/UPDATE/DELETE/NOOP) is auditable.

5. **Index:** embed the record and upsert to Qdrant. The SQLite ledger records
   the provenance, lineage, and audit trail.

### Quarantine

Records that fail the trust gate or the injection adjudicator are staged in
quarantine, not applied. A non-zero `quarantined` count in `WriteResult` is the
caller's cue to run `review_quarantine`. Quarantined records are never
retrievable — they never reach the index.

---

## Retrieval architecture

### Three channels

myelin fuses three retrieval channels:

1. **Dense** — bge-m3 embeddings (1024-d), searched via Qdrant's HNSW index.
   Wins for paraphrase-heavy queries.
2. **Lexical (BM25)** — Qdrant's server-side BM25 with IDF scoring. Wins for
   exact-match domains (code symbols, filenames, config keys). Costs 12.68
   points to remove (MemPro ablation).
3. **Graph** — personalized PageRank over the phrase↔record incidence graph.
   Off by default; enabled with `MYELIN_QDRANT__GRAPH=true` (requires
   `myelin-eval phrases` to build the graph first).

### RRF fusion

Fused with Reciprocal Rank Fusion at **k = 1** (not Cormack's 60). This is a
measurement overruling a prediction: M4 measured k=1 at 0.8815 dev recall@6 vs
k=60 at 0.8252. The mechanism: BM25 alone scores 0.8666 and dense alone 0.7314,
so flattening the head (k=60) averages a strong ranking with a weak one. k=1
keeps BM25's confident head and lets dense contribute only where it is also
confident.

### Cross-encoder rerank

After fusion, the top `rerank_depth` (default 25) candidates are scored by a
cross-encoder (`bge-reranker-v2-m3`). This is the highest-leverage stage: MS
MARCO MRR@10 goes 18.7 → 36.5 with a cross-encoder over BM25.

### Compose

`compose` selects the final evidence set from the reranked pool:
- **k** (default 6): maximum records returned
- **max_tokens** (default 2048): token ceiling — the budget loop continues
  rather than breaks, so a tight budget silently reshapes the emitted set
  toward shorter records
- **Near-dedup** at cosine ≥ 0.93
- **`stamp_valid_time`** (default ON): prefix each item with its `t_valid` date
- **`resolve_relative`** (default ON): annotate relative time references with
  absolute dates — the largest single measured win in the project (+37.6 points
  on LoCoMo temporal)
- **`timeline`** (default ON): append a synthetic `[timeline]` item for
  duration/interval questions
- **`bookend`** ordering: relevance-interleaved, not chronological

---

## Multi-tenancy

Every read and write is scoped to a **tenant**. There is no read path that
spans tenants (C12: scope-before-ranking). The scope filter is:

```
tenant / namespace / agent / session
```

- `tenant` is mandatory on every tool that reads or writes.
- `namespace` segments a tenant's memory (e.g., `work` vs `personal`).
- `agent` identifies which agent wrote the record.
- `session` further scopes within an agent.

**The scope-before-ranking invariant:** metadata predicates mask inadmissible
memories *before* ranking, not after. This is the ShardMemo scope-before-routing
pattern — post-filtering wastes budget on inadmissible memories.

> **Known usability gap:** a wrong `tenant` returns 0 items with no error. See
> [Quickstart](QUICKSTART.md#known-usability-pitfalls) for how to list valid
> tenants.

---

## Configuration

### Layering

1. Serialized defaults (hardcoded in `config.rs`)
2. `~/.myelin/config.yml` (YAML, overrides defaults)
3. `MYELIN_`-prefixed env vars (`__` marks nesting, e.g.
   `MYELIN_QDRANT__URL`)

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MYELIN_QDRANT__URL` | `http://192.168.1.110:6334` | Qdrant gRPC endpoint (port 6334, never 6333) |
| `MYELIN_QDRANT__COLLECTION` | `myelin_memory` | Qdrant collection name |
| `MYELIN_LLM__URL` | `http://127.0.0.1:5810/v1` | Reader LLM endpoint (OpenAI-compatible `/v1`) |
| `MYELIN_LLM__MODEL` | `qwen3.5-9b` | Reader LLM model name |
| `MYELIN_EMBED__URL` | `http://192.168.1.110:11434/v1` | Embedder endpoint (OpenAI-compatible `/v1`) |
| `MYELIN_EMBED__MODEL` | `bge-m3` | Embedding model name |
| `MYELIN_EMBED__DIM` | `1024` | Output dimension after Matryoshka truncation |
| `MYELIN_RERANK__URL` | `http://127.0.0.1:5813` | Reranker endpoint (server root, not `/v1`) |
| `MYELIN_RERANK__MODEL` | `bge-reranker-v2-m3` | Reranker model name |
| `MYELIN_LEDGER` | `data/myelin.ledger` | SQLite ledger path (admissibility authority) |
| `MYELIN_MCP_TOKEN` | (none) | Bearer token for non-loopback `--serve` binds |

> **The Qdrant, embed, and rerank defaults point at the author's LAN
> (`192.168.1.110`).** Override them for your environment. The reader LLM
> defaults to `127.0.0.1:5810` because the author reaches it through an SSH
> tunnel.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--serve HOST:PORT` | (stdio) | Run as streamable-HTTP server |
| `--collection NAME` | config | Qdrant collection to read |
| `--ledger PATH` | config | SQLite ledger path (must match collection) |
| `--prefetch-limit N` | 50 | Candidates per channel before fusion |
| `--rerank-depth N` | 25 | Min candidates the reranker scores |

### How to regenerate this table

The defaults are read from `crates/myelin-core/src/config.rs` (`impl Default`
blocks for `QdrantConfig`, `LlmConfig`, `EmbedConfig`, `RerankConfig`). The CLI
flags are in `crates/myelin-mcp/src/main.rs` (`struct Args`). If a default
changes, update this table from those source files.

---

## `myelin-eval` harness

The evaluation harness is a **research tool**, not a product surface. It scores
myelin against the LoCoMo and LongMemEval-S benchmarks.

### Subcommands

| Command | Description |
|---------|-------------|
| `fetch` | Download and checksum-pin benchmark datasets |
| `build` | Build a memory from a dataset into a backend |
| `phrases` | Populate the phrase↔record incidence graph (pure SQLite) |
| `bench` | Score end-to-end: retrieve, read, grade |
| `attack` | Run the MINJA-style poisoning attack suite |
| `adjudicate-probe` | Test injection adjudicator false-positive rate |
| `ablate` | Retrieval ablation against a built memory |
| `rescore` | Re-score a finished run under a different scorer (pure CPU) |
| `judge` | Grade answers with the local reader |
| `evidence-audit` | Audit whether wrong answers had the evidence |
| `coverage` | Gold-unit recall per category (fully offline) |
| `standing` | Compare against published systems |
| `ratchet` | Regression check against our own pinned floor |
| `package` | Package a leaderboard submission |

### Corpora

| Corpus | Collection | Ledger | Default scorer |
|--------|------------|--------|----------------|
| LoCoMo | `myelin_locomo` | `data/locomo.ledger` | temporal |
| LongMemEval-S | `myelin_longmemeval_s` | `data/longmemeval_s.ledger` | token-f1 |

> `bench`, `attack --live`, `judge`, and `ablate` require GPU services. `rescore`,
> `coverage`, `standing`, `ratchet`, and `phrases` are CPU-only.