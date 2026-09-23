# `big_mac` — a second extraction host

`big_mac` (Apple M1 Max, 32-core GPU, 400 GB/s memory bandwidth, 32 GB)
serves the myelin reader for **build passes only**. It joined on 2026-09-23
because `big`'s 3090 measured only ~1.3× faster running arms side by side
(`ops/big/README.md`, `MYELIN_READER_KV_UNIFIED`), which left the M50
events extraction (~6 GPU-hours) competing with three arms for one card.

## What runs here, and what never does

| | here | why |
|---|---|---|
| `myelin-eval events-extract` | **yes** | a build pass: its output is a store, and a base and an arm both read that same store, so the extractor's backend never enters a comparison |
| `bench`, the LME-V2 harness | no | the memory side makes greedy model calls; on bmb (Metal) 3 of 5 probe rows composed different evidence than on the 3090, so an arm split across backends measures two systems |
| `judge` | no | every verdict in `runs/` came from the 3090, and verdicts are reused by answer (M42) |

Laya (`:9393`, MPS) shares the GPU. Measured with extraction running: memory
71% free and `/health` answering; its latency rises while a shard runs.

## Setup (done 2026-09-23)

```bash
# llama.cpp: the official macOS arm64 release, unpacked (Homebrew on this
# host failed installing a dependency; the release changes nothing system-wide)
ssh big_mac 'mkdir -p ~/llm/llama-b11126 && cd ~/llm/llama-b11126 &&
  curl -sSLO https://github.com/ggml-org/llama.cpp/releases/download/b11126/llama-b11126-bin-macos-arm64.tar.gz &&
  tar xzf llama-b11126-bin-macos-arm64.tar.gz'
# the reader, straight from big (big_mac holds a key for big; relaying through
# a laptop ran at ~5 MB/s), then compare sha256 on both ends
ssh big_mac 'mkdir -p ~/llm/models/qwen && cd ~/llm/models/qwen &&
  scp big:<big models dir>/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf . && shasum -a 256 *.gguf'
```

The build is b11126, not the b10450 `big` and `bmb` run (that release is no
longer published). For a build pass that does not matter, for the reason in
the table.

## Run

```bash
ssh big_mac bash -s < ops/big_mac/serve-reader.sh      # 4 slots x 16k, thinking off
ssh -fN -L 127.0.0.1:25810:127.0.0.1:5810 big_mac
MYELIN_LLM__URL=http://127.0.0.1:25810/v1 myelin-eval events-extract --corpus longmemeval-s \
  --questions docs/measurements/m50-pilot-questions.txt --shard <i>/8 --concurrency 4 \
  --out data/events/longmemeval_s.s<i>.jsonl
ssh big_mac 'kill $(cat /tmp/myelin-reader.pid)'        # after use
```

Two hosts must never extract the same shard at once: two extractions of one
session differ, and `load_cache` refuses a session cached twice with
different events. Claim a shard before starting it (the 2026-09-23 run used
an atomic `mkdir` per shard).

## Measured, 2026-09-23

M50 pilot extraction (LongMemEval_S sessions, ~2.4k prompt tokens each,
schema-constrained output), 4 slots, `--concurrency 4`:

| host | backend | seconds / session |
|---|---|---|
| `big_mac` (M1 Max, 32 GPU cores) | Metal, llama.cpp b11126 | **8.1** |
| `bmb` (M4 Pro, 20 GPU cores) | Metal, llama.cpp b10450 | 10.9 |
| `big` (3090), sharing the card with three arms | CUDA, b10450 | 16.5 |

Together the two Macs extract ~0.21 sessions/s: the pilot's 4,502 distinct
sessions in ~6 h, without any time on `big`. The M1 Max outpacing the M4 Pro
fits its memory bandwidth (400 vs 273 GB/s) — decoding a 9B is bandwidth-bound.
