#!/usr/bin/env bash
# Bring up the myelin reader + embedder on `big`, co-resident.
#
# Run this FROM `big` (or via `ssh big "bash -lc '...'"`). It assumes the GPU
# is already claimed — see ops/big/README.md for why the claim alone is not
# sufficient to free the card.
#
# Deliberately NOT wired into llama-swap. llama-swap runs strict swap (mutual
# exclusion, one model resident at a time — see the header of
# ~/.home-still/llama-swap.yaml), and myelin needs the reader and the embedder
# at the same time: the write path embeds and extracts in one pass. Two direct
# llama-server processes are the simplest thing that satisfies that, and they
# touch no shared configuration.
#
# Build: cuda-ece963 (build 1498, commit ece963f4). Verified to carry the
# `qwen35` arch; the older 434b2a1 install does not and cannot load the reader.
set -euo pipefail

LC=/home/ladvien/.local/llama.cpp/cuda-ece963
export LD_LIBRARY_PATH="$LC"
R=/home/ladvien/models/Qwen3.5-9B-GGUF
E=/home/ladvien/models/Qwen3-Embedding-8B-GGUF

READER_PORT="${MYELIN_READER_PORT:-5810}"
EMBED_PORT="${MYELIN_EMBED_PORT:-5811}"

# Contexts are small on purpose. Per skill://serve-gguf-on-big the binding
# constraint on this 24 GB card is CUDA graph capture, not weights: a request
# that fits in VRAM can still OOM at cudaGraphInstantiate. Measured peak with
# these values was 20,335 MiB of 24,576 with a foreign 2,640 MiB tenant also
# resident, so there is ~4 GB of headroom to grow into if a milestone needs it.
READER_CTX="${MYELIN_READER_CTX:-16384}"
EMBED_CTX="${MYELIN_EMBED_CTX:-8192}"

nohup "$LC/llama-server" \
  -m "$R/Qwen3.5-9B-UD-Q4_K_XL.gguf" \
  --mmproj "$R/mmproj-F16.gguf" \
  --host 127.0.0.1 --port "$READER_PORT" \
  -c "$READER_CTX" -ngl 999 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -np 1 -cb \
  --jinja \
  > /tmp/myelin-reader.log 2>&1 &
echo $! > /tmp/myelin-reader.pid

# --pooling last is required: Qwen3-Embedding derives the sentence vector from
# the final token, not a mean over the sequence. Mean pooling silently returns
# usable-looking vectors with worse geometry, which would show up as a quiet
# retrieval regression rather than an error.
nohup "$LC/llama-server" \
  -m "$E/Qwen3-Embedding-8B-Q8_0.gguf" \
  --host 127.0.0.1 --port "$EMBED_PORT" \
  -c "$EMBED_CTX" -ngl 999 \
  --embedding --pooling last \
  > /tmp/myelin-embed.log 2>&1 &
echo $! > /tmp/myelin-embed.pid

for _ in $(seq 1 60); do
  r=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$READER_PORT/health" || true)
  e=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$EMBED_PORT/health" || true)
  if [ "$r" = 200 ] && [ "$e" = 200 ]; then
    echo "ready reader=:$READER_PORT embed=:$EMBED_PORT vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    exit 0
  fi
  sleep 5
done

echo "FAILED to become ready; see /tmp/myelin-reader.log and /tmp/myelin-embed.log" >&2
exit 1
