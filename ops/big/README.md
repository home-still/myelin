# Serving myelin's models on `big`

M2 of [`PLAN.md`](../../PLAN.md): the reader, the embedder and the reranker,
co-resident on one RTX 3090, measured.

## The models

| role | weights | size | why |
|---|---|---|---|
| reader | `~/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf` + `mmproj-F16.gguf` | 5.97 GB + 0.92 GB | extraction (§6.2), consolidation (§6.3), `investigate` (§7.2) |
| embedder | `~/models/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf` | 8.05 GB | dense channel (§5.1) — **opt-in**, see below |
| embedder (default) | `bge-m3` via the resident `ollama` | 1.2 GB | dense channel (§5.1), 1024-d natively |
| reranker | `~/models/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q8_0.gguf` | 0.64 GB | cross-encoder rerank (§7.1) |

`Qwen/Qwen3.5-9B` publishes **safetensors only** and `Qwen/Qwen3.5-9B-GGUF`
returns 401, so the reader GGUF comes from `unsloth/Qwen3.5-9B-GGUF` — the same
publisher as the existing `Qwen3.8-27B-UD-Q4_K_XL` on this box, so the `UD-`
naming and quant recipe match what is already here.

The embedder is **Q8_0, not Q4**. Quantization distorts embedding geometry more
than it distorts generation quality, and the dense channel is one of only two
retrieval channels; a 3 GB saving is not worth a silent recall regression. The
same argument picks Q8_0 for the reranker, where it costs only 0.3 GB.

The reranker is a 568M-parameter model, not a 7B one, and that is the point: it
buys the largest single accuracy gain in the read path (MS MARCO MRR@10
18.7 → 36.5 over BM25, `PLAN.md` §2 finding 2) for 636 MB on a card that also
hosts a live voice assistant.

## Run it

There is no checkout on `big`; pipe the scripts in from this repo. This is the
exact invocation that was verified:

```bash
ssh big gpu-tenant claim coding          # see the caveat below — this is not enough
ssh big bash -s < ops/big/serve-models.sh
# Overrides MUST be set on the REMOTE side; `ssh` does not forward the
# environment, so `MYELIN_READER_CTX=65536 ssh big bash -s < ...` silently
# uses the default and gives each slot 4096 tokens:
ssh big "MYELIN_READER_CTX=65536 bash -s" < ops/big/serve-models.sh
ssh -N -L 5810:127.0.0.1:5810 -L 5813:127.0.0.1:5813 big &   # see "firewall"
# ... work ...
ssh big bash -s < ops/big/stop-models.sh
ssh big gpu-tenant release
```

`serve-models.sh` blocks until every enabled `/health` answers 200 and then
prints the ports and card occupancy, so it is safe to chain. It exits 1 with the
log paths if any server fails to come up. Two switches:

| variable | default | effect |
|---|---|---|
| `MYELIN_EMBEDDER` | `bge` | `qwen` also serves the 8B embedder on :5811 |
| `MYELIN_RERANK` | `1` | `0` skips the cross-encoder on :5813 |
| `MYELIN_READER_THINK_BUDGET` | `-1` | `--reasoning-budget` for the reader: `-1` unrestricted (every run before M44), `N` closes a thinking trace at N tokens. M44 R2 runs at `1024`; `bench --reader-thinking` probes it before the first row |

### The firewall makes a tunnel mandatory

`big` runs an allowlist firewall. Measured from the workstation on 2026-09-14:
**6333, 6334, 8081 and 11434 connect; 5810 and 7434 time out** — dropped, not
refused, while the same request from `big` itself answers 200 in 0.3 ms. So the
servers bind `127.0.0.1` and the driver reaches them over SSH forwarding. That
is also the better posture: an unauthenticated LLM endpoint does not belong on
the LAN just because the firewall would have to be asked nicely.

Qdrant (6334) and ollama (11434) need no tunnel, which is a second reason the
default dense embedder is bge-m3 through ollama.

## `gpu-tenant claim` does not free the card

This is the thing that will bite you. `gpu-tenant` pauses exactly three units:

```
hs-serve-distill.service  llama-swap.service  trellis2-mcp.service
```

It does **not** manage `ollama`, and it does **not** manage `voice-serve` (the
household voice assistant, another agent's process). Both stay resident through
a claim. Measured on 2026-09-14: immediately after `claim coding` the card still
held 11,273 MiB — ollama `qwen3:8b` at 5,644 MiB plus two `voice-serve`
instances — leaving 12,849 MiB against 13,900 MiB of weights. The claim alone
did not fit.

So the real budget is:

```
free = 24576 - voice-serve(~2600) - ollama(varies, 0 if evicted)
```

Evict an idle ollama model with `keep_alive: 0`; it reloads on demand:

```bash
ssh big 'curl -s http://localhost:11434/api/generate -d "{\"model\":\"qwen3:8b\",\"keep_alive\":0}"'
```

**Coordinate before claiming.** `big:/tmp/agent_chat` is the shared append-only
log; other agents run latency-sensitive gates and a live voice assistant that
OOMs if someone sizes a model against the wrong free-VRAM number. Announce the
window, then release and say so.

## Measured, 2026-09-14

Both resident simultaneously under `gpu-tenant claim coding`:

| | VRAM |
|---|---|
| reader (`n_ctx` 16384, kv `q8_0`) | 7,042 MiB |
| embedder (`n_ctx` 8192, pooling `last`) | 9,430 MiB |
| foreign tenant (`voice-serve`) | 2,640 MiB |
| **peak card occupancy under concurrent load** | **20,335 / 24,576 MiB** |

Tool calling, which the agentic read path depends on:

```json
{"finish_reason":"tool_calls",
 "tool_calls":[{"function":{"name":"shell","arguments":"{\"cmd\":\"ls -la /tmp\"}"}}]}
```

Embeddings return **4096 dimensions**.

> **Settled in M3: the default embedder is bge-m3, not this one.** Three
> reasons, in order of weight. (1) `PLAN.md` §5.1 specifies the `dense` channel
> as bge-m3 1024-d, which bge-m3 produces natively — no Matryoshka truncation,
> no geometry question. (2) VRAM: the 8B embedder costs 8,858 MiB, and holding
> it alongside `voice-serve` and a woken ollama left **890 MiB** free against an
> operator request for ~5 GB. bge-m3 is 1.2 GB inside a process that keeps
> waking anyway. (3) ollama's port is open; :5811 is firewalled. Set
> `MYELIN_EMBEDDER=qwen` to serve the 8B for the M4 embedder ablation — that
> comparison is exactly why the code keeps both paths.

## Build

`~/.local/llama.cpp/cuda-ece963` (build 1498, commit `ece963f4`). Verified to
carry the `qwen35` arch:

```bash
ssh big 'strings -a ~/.local/llama.cpp/cuda-ece963/libllama.so.0.1.0 | grep -xE "qwen[0-9a-z_.]*"'
```

The older `~/.local/llama.cpp/cuda` (434b2a1) does not have it and cannot load
the reader.

## Getting the weights onto the box

Download on the workstation and push over the LAN. Measured the same minute on
2026-09-14:

| source → HuggingFace | throughput |
|---|---|
| `big` | 374 KB/s |
| workstation | 17.3 MB/s |

14 GB direct to `big` was on track for ~10 hours; workstation download (4.5 min)
plus the LAN push (21 min) took ~25. Verify byte counts against the HF API
afterwards — `hf download` exits 0 even when a requested filename does not
exist.

Use `scp`, not `rsync`: macOS 25.6 ships openrsync, which segfaults
(`child exited with status 11`) partway through a multi-hundred-MB transfer to
this host. `scp` moved the 636 MB reranker in 55 s (11.6 MB/s).

Killing an `ssh`-wrapped `hf download` from the client does **not** kill it on
`big`; it orphans and competes with its own replacement. Use
`pkill -f "bin/hf download"` on the box.
