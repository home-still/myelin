# Serving myelin's models on `big`

M2 of [`PLAN.md`](../../PLAN.md): the reader and the embedder, co-resident on one
RTX 3090, measured.

## The models

| role | weights | size | why |
|---|---|---|---|
| reader | `~/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf` + `mmproj-F16.gguf` | 5.97 GB + 0.92 GB | extraction (§6.2), consolidation (§6.3), `investigate` (§7.2) |
| embedder | `~/models/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf` | 8.05 GB | dense channel (§5.1) |

`Qwen/Qwen3.5-9B` publishes **safetensors only** and `Qwen/Qwen3.5-9B-GGUF`
returns 401, so the reader GGUF comes from `unsloth/Qwen3.5-9B-GGUF` — the same
publisher as the existing `Qwen3.8-27B-UD-Q4_K_XL` on this box, so the `UD-`
naming and quant recipe match what is already here.

The embedder is **Q8_0, not Q4**. Quantization distorts embedding geometry more
than it distorts generation quality, and the dense channel is one of only two
retrieval channels; a 3 GB saving is not worth a silent recall regression.

## Run it

There is no checkout on `big`; pipe the scripts in from this repo. This is the
exact invocation that was verified:

```bash
ssh big gpu-tenant claim coding          # see the caveat below — this is not enough
ssh big bash -s < ops/big/serve-models.sh
# ... work ...
ssh big bash -s < ops/big/stop-models.sh
ssh big gpu-tenant release
```

`serve-models.sh` blocks until both `/health` endpoints answer 200 and then
prints the port pair and card occupancy, so it is safe to chain. It exits 1
with the log paths if either server fails to come up.

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

> **Open decision for M3.** 4096-d is 4× the 1024-d bge-m3 that `PLAN.md` §5.1
> assumes for the `dense` channel, so the Qdrant collection dimension and the
> storage cost per record both change. Settle it in M3 with a measurement, not
> by defaulting.

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
plus `rsync` over the LAN (21 min) took ~25. Verify byte counts against the HF
API afterwards — `hf download` exits 0 even when a requested filename does not
exist.

Killing an `ssh`-wrapped `hf download` from the client does **not** kill it on
`big`; it orphans and competes with its own replacement. Use
`pkill -f "bin/hf download"` on the box.
