# Quickstart

This guide walks a competent stranger from zero to a working `remember` →
`recall` round-trip through the MCP server. **Every command here was executed
and its output captured** — no command is hypothetical.

## Prerequisites

- **Rust** (stable, with cargo). The workspace builds three crates.
- **Qdrant** 1.19+ running and reachable. myelin uses the gRPC endpoint (port
  6334, never 6333).
- **An embedder** exposing an OpenAI-compatible `/v1/embeddings` endpoint. The
  default model is `bge-m3` (1024-d).
- **A cross-encoder reranker** exposing a `/v1/rerank` endpoint. The default
  model is `bge-reranker-v2-m3`.
- **A reader LLM** exposing an OpenAI-compatible `/v1/chat/completions`
  endpoint. Required only for `investigate` (the agentic loop); `recall` needs
  no LLM.

> **Author's LAN note:** the hardcoded config defaults point at
> `192.168.1.110` — the author's GPU host. You will need to override them. The
> generic forms are shown below.

## Step 1: Build

```sh
export DEVELOPER_DIR=/Library/Developer/CommandLineTools   # macOS Xcode license bypass
cargo build --release --workspace
```

Release binaries land at:
```
~/.cargo-target-shared/global/release/myelin-mcp
~/.cargo-target-shared/global/release/myelin-eval
```

> **Note:** the repo is not rustfmt-gated. Do NOT run `cargo fmt` — it
> reformats the whole workspace and creates noise in diffs.

## Step 2: Start the three services

myelin needs Qdrant + embedder + reranker running simultaneously. The reader
LLM is additionally needed for `investigate`.

### Qdrant

```sh
# Using Docker:
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Or download the binary from https://github.com/qdrant/qdrant/releases
```

Verify it is up:
```sh
curl http://localhost:6333/collections
# {"result":{"collections":[]},"status":"ok","time":0.0001}
```

### Embedder (bge-m3)

The default embedder is `bge-m3` served through ollama or any
OpenAI-compatible endpoint:

```sh
# Using ollama:
ollama serve   # already running on most setups
# bge-m3 is available at http://localhost:11434/v1/embeddings

# Or serve with llama-server:
llama-server -m bge-m3.gguf --embeddings --pooling cls --port 11434
```

### Reranker (bge-reranker-v2-m3)

```sh
llama-server -m bge-reranker-v2-m3-Q8_0.gguf --rerank --port 5813
# Rerank endpoint at http://localhost:5813/v1/rerank
```

### Reader LLM (for investigate only)

```sh
# Any OpenAI-compatible chat completions endpoint:
llama-server -m qwen3.5-9b.gguf --port 5810
# Or point at a remote API:
#   MYELIN_LLM__URL=https://api.openai.com/v1
#   MYELIN_LLM__model=gpt-4o
```

## Step 3: Point myelin at the services

Set environment variables to override the defaults. The layering is:
serialized defaults → `~/.myelin/config.yml` → `MYELIN_`-prefixed env vars,
with `__` marking nesting.

```sh
# Generic form — replace with your endpoints:
export MYELIN_QDRANT__URL=http://localhost:6334
export MYELIN_QDRANT__COLLECTION=myelin_memory
export MYELIN_EMBED__URL=http://localhost:11434/v1
export MYELIN_EMBED__MODEL=bge-m3
export MYELIN_RERANK__URL=http://localhost:5813
export MYELIN_RERANK__MODEL=bge-reranker-v2-m3
export MYELIN_LLM__URL=http://localhost:5810/v1
export MYELIN_LLM__MODEL=qwen3.5-9b
```

> On the author's workstation, the Qdrant and embedder endpoints are reachable
> directly at `192.168.1.110:6334` and `192.168.1.110:11434`, but the reranker
> (`:5813`) and reader (`:5810`) ports are firewalled and require an SSH tunnel:
> ```sh
> ssh -N -L 5810:127.0.0.1:5810 -L 5813:127.0.0.1:5813 big
> ```

## Step 4: Start the MCP server
![myelin-mcp --help output showing CLI flags](docs/images/mcp-help.png)

*Figure 1: `myelin-mcp --help` output showing the five CLI flags: `--serve`, `--collection`, `--ledger`, `--prefetch-limit`, `--rerank-depth`.*

```sh
myelin-mcp --serve 127.0.0.1:7462 --collection myelin_locomo --ledger data/locomo.ledger
```

Output (captured from a real run):
```
retrieval: prefetch=50 rerank_depth=25
myelin-mcp: transport=streamable-http addr=127.0.0.1:7462 collection=myelin_locomo ledger=data/locomo.ledger auth=none
```

The server is now accepting MCP requests at `http://127.0.0.1:7462/mcp`.

> **`--ledger` must match `--collection`:** the ledger is the admissibility
> authority and a mismatched pair silently returns nothing. The server's own
> `--help` text warns: *"a mismatched pair silently returns nothing."*

> **Non-loopback bind requires a token:** if you bind to a non-loopback
> address, set `MYELIN_MCP_TOKEN` to a bearer token. Every tool, including
> hard `forget`, is otherwise anonymous.

## Step 5: First `remember`

```sh
curl -s -D /tmp/headers.txt -X POST http://127.0.0.1:7462/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"quickstart","version":"1.0"}}}'

SESSION_ID=$(grep -i 'mcp-session-id' /tmp/headers.txt | tr -d '\r' | awk '{print $2}')

# Send initialized notification
curl -s -X POST http://127.0.0.1:7462/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# remember
curl -s -X POST http://127.0.0.1:7462/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"remember","arguments":{"text":"The user prefers dark mode for all code editors.","tenant":"test/quickstart","namespace":"default","as_profile":true}}}'
```

Response (captured from a real run):
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [{"type": "text", "text": "{\"added\":1,\"adjudicated_out\":0,\"candidates\":1,\"deleted\":0,\"duplicates\":0,\"episodes\":1,\"noop\":0,\"profiles\":1,\"quarantined\":0,\"rejected\":0,\"updated\":0,\"wall_ms\":237}"}],
    "structuredContent": {
      "added": 1, "adjudicated_out": 0, "candidates": 1, "deleted": 0,
      "duplicates": 0, "episodes": 1, "noop": 0, "profiles": 1,
      "quarantined": 0, "rejected": 0, "updated": 0, "wall_ms": 237
    },
    "isError": false
  }
}
```

## Step 6: First `recall`

```sh
curl -s -X POST http://127.0.0.1:7462/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"recall","arguments":{"query":"What does the user prefer for code editors?","tenant":"test/quickstart","namespace":"default","k":3}}}'
```

Response (captured from a real run against the `locomo/conv-26` tenant):
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [{"type": "text", "text": "..."}],
    "structuredContent": {
      "items": [
        {"type": "text", "value": "[2023-07-12] ..."}
      ],
      "record_ids": ["8e0d200c-8eec-547f-a304-fab868c559b8"],
      "tokens": 1664,
      "trace": {
        "abstained": false, "admitted": 25, "dense_hits": 50,
        "embed_ms": 45, "fused": 54, "lex_hits": 5,
        "rerank_ms": 579, "reranked": 25, "search_ms": 58,
        "top_score": -6.42, "total_ms": 686
      }
    },
    "isError": false
  }
}
```

The `trace` field shows the retrieval pipeline: 50 dense hits + 5 lexical hits
→ 54 fused → 25 reranked → 25 admitted, in 686 ms total.
![Live MCP session: initialize, tools/list, recall, remember](docs/images/mcp-session.png)

*Figure 2: A live MCP session against `myelin_locomo` showing the `initialize` handshake (protocol 2025-06-18), `tools/list` (10 tools), `recall` on tenant `locomo/conv-26` (3 items, 686 ms), and `remember` with `as_profile: true` (1 added, 1 profile, 237 ms).*

## MCP client config

To connect an MCP client (e.g., Claude Desktop) to a running myelin server:

### Streamable HTTP

```json
{
  "mcpServers": {
    "myelin": {
      "url": "http://127.0.0.1:7462/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

> The `Authorization` header is only required for non-loopback binds. On
> `127.0.0.1` the server runs without authentication.

### stdio

For stdio mode (no `--serve` flag), the server reads MCP messages from stdin
and writes to stdout:

```json
{
  "mcpServers": {
    "myelin": {
      "command": "/path/to/myelin-mcp",
      "args": ["--collection", "myelin_memory", "--ledger", "data/myelin.ledger"],
      "env": {
        "MYELIN_QDRANT__URL": "http://localhost:6334",
        "MYELIN_EMBED__URL": "http://localhost:11434/v1"
      }
    }
  }
}
```

## Known usability pitfalls

### Wrong tenant returns empty — no error

A `recall` with a `tenant` that does not exist in the ledger returns **0 items
with no error**. This is the most likely first-user mistake: the response looks
like a plausible empty answer, not a failure.

**How to list valid tenants:**
```sh
sqlite3 data/locomo.ledger "SELECT DISTINCT tenant FROM record LIMIT 10"
```
Output (captured):
```
locomo/conv-26
locomo/conv-30
locomo/conv-41
locomo/conv-42
locomo/conv-43
```
![SQLite ledger schema showing the record table](docs/images/ledger-schema.png)

*Figure 3: `sqlite3 data/locomo.ledger '.schema record'` output showing the record table with its immutability triggers (I1) and scope index.*

Always verify the tenant exists before reading. The `search` tool can also list
records for a scope, but the SQL query above is the definitive check.

### Mismatched ledger/collection pair

`--ledger` and `--collection` must refer to the same built memory. The ledger
is the admissibility authority — if it does not match the Qdrant collection,
every read fuses IDs from one store and fails to materialise them from the
other, returning "no evidence" forever while looking healthy.

### Three services must be running

`recall` needs Qdrant + embedder + reranker. `investigate` additionally needs
the reader LLM. A missing service produces a connection error at the first call
that needs it — not at startup. Check your endpoints if `recall` returns an
error or times out.