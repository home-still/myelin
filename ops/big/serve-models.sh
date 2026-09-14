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
K=/home/ladvien/models/bge-reranker-v2-m3-GGUF

# Bind localhost, like the llama-swap children do.
#
# Binding the LAN address does not help anyway: `big` runs an allowlist
# firewall that drops inbound on these ports. Measured from the workstation on
# 2026-09-14 -- 6333, 6334, 8081 and 11434 connect; 5810 and 7434 time out
# (dropped, not refused) while the same request from `big` itself answers 200
# in 0.3 ms. Changing that needs sudo, so the driver reaches these over an SSH
# tunnel:
#
#   ssh -N -L 5810:127.0.0.1:5810 -L 5811:127.0.0.1:5811 big
#
# which is also the better posture: an unauthenticated LLM endpoint does not
# belong on the LAN just because the firewall would have to be asked nicely.
BIND_HOST="${MYELIN_BIND_HOST:-127.0.0.1}"
READER_PORT="${MYELIN_READER_PORT:-5810}"
EMBED_PORT="${MYELIN_EMBED_PORT:-5811}"
RERANK_PORT="${MYELIN_RERANK_PORT:-5813}"

# Contexts are small on purpose, for two reasons.
#
# 1. Per skill://serve-gguf-on-big the binding constraint on this 24 GB card is
#    CUDA graph capture, not weights: a request that fits in VRAM can still OOM
#    at cudaGraphInstantiate.
# 2. This card is shared with a live household voice assistant whose operator
#    asked for ~5 GB to be left free, and with ollama, which reloads models on
#    demand without warning. At 16384/8192 the card sat at 486 MiB free once
#    ollama woke up -- one allocation spike from killing someone else's work.
#
# The write path does not need the headroom anyway: episodes are capped at 512
# tokens by the segmenter, so extraction prompts land around 1-2k and the
# embedder never sees more than one episode at a time.
# -np 4 with -c 16384 gives each slot 4096 tokens, which is ample: episodes
# are capped at 512 tokens by the segmenter and the largest prompt is a
# consolidation judgement over 6 neighbours. Slots are the throughput lever --
# consolidation was 68% of a measured 768 s conversation, all of it queued
# behind a single slot.
READER_SLOTS="${MYELIN_READER_SLOTS:-4}"
READER_CTX="${MYELIN_READER_CTX:-16384}"
EMBED_CTX="${MYELIN_EMBED_CTX:-4096}"

# mmproj is ON by default. It costs ~920 MiB, and 29 of LongMemEval-V2's 451
# questions carry a `question_screenshots/*.png`; without it the harness dies
# on the first one with `image input is not supported`. Dropping those 29
# would bias G1 rather than save memory. MYELIN_MMPROJ=0 for a text-only run.
MMPROJ_ARGS=()
if [ "${MYELIN_MMPROJ:-1}" = "1" ]; then
  MMPROJ_ARGS=(--mmproj "$R/mmproj-F16.gguf")
fi

# Thinking off SERVER-WIDE, not per request.
#
# `llm/openai.rs` already sends `enable_thinking: false` on every call, but
# third-party clients do not. The vendored LongMemEval-V2 harness disables
# thinking for the READER only, and only when `--model` is the literal string
# `Qwen/Qwen3.5-9B`; its EVALUATOR path never sends it at all. The judge then
# spent its whole budget on `reasoning_content`, returned empty content, and
# the run died with `Empty judgement response from evaluator model.`
# A server-side default fixes every client at once.
TEMPLATE_KWARGS='{"enable_thinking":false}'

# Wait for VRAM before loading.
#
# Killing a llama-server does not free its VRAM synchronously. A 3-second
# sleep between kill and reload produced `cudaMalloc failed: out of memory`
# on a card that reported 22.7 GB free moments later.
for _ in $(seq 1 30); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free_mib" -ge 9000 ] && break
  sleep 2
done

nohup "$LC/llama-server" \
  -m "$R/Qwen3.5-9B-UD-Q4_K_XL.gguf" \
  "${MMPROJ_ARGS[@]}" \
  --host "$BIND_HOST" --port "$READER_PORT" \
  -c "$READER_CTX" -ngl 999 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -np "$READER_SLOTS" -cb \
  --jinja --chat-template-kwargs "$TEMPLATE_KWARGS" \
  > /tmp/myelin-reader.log 2>&1 &
echo $! > /tmp/myelin-reader.pid

# --pooling last is required: Qwen3-Embedding derives the sentence vector from
# the final token, not a mean over the sequence. Mean pooling silently returns
# usable-looking vectors with worse geometry, which would show up as a quiet
# retrieval regression rather than an error.
# The 8B embedder is opt-in. Default dense is bge-m3 via ollama (1024-d
# native, 1.2 GB, already resident) -- see EmbedConfig::default for why.
# MYELIN_EMBEDDER=qwen serves this one for the M4 ablation.
if [ "${MYELIN_EMBEDDER:-bge}" = "qwen" ]; then
nohup "$LC/llama-server" \
  -m "$E/Qwen3-Embedding-8B-Q8_0.gguf" \
  --host "$BIND_HOST" --port "$EMBED_PORT" \
  -c "$EMBED_CTX" -ngl 999 \
  --embedding --pooling last \
  > /tmp/myelin-embed.log 2>&1 &
echo $! > /tmp/myelin-embed.pid
fi

# The cross-encoder. 606 MiB of weights, and the single largest accuracy lever
# in the read path: MS MARCO MRR@10 18.7 -> 36.5 for a cross-encoder over BM25
# (PLAN.md 2, finding 2), where fusion is worth 1-2 points.
#
# --reranking implies --embedding internally and REQUIRES --pooling rank; the
# server refuses to start otherwise.
#
# -b/-ub 8192 are NOT decoration. llama.cpp's physical batch defaults to 512
# and a cross-encoder must process a (query, document) pair in ONE physical
# batch, so any document over ~500 tokens fails the whole call with
# `input (575 tokens) is too large to process`. LME-V2 accessibility-tree
# chunks are ~450-600 tokens, so the default silently made the reranker
# unusable on that corpus -- found by the first end-to-end query, not by any
# unit test. 8192 matches the context so no admissible document can exceed
# it.
#
# Opt out with MYELIN_RERANK=0 when only the write path is needed; a loaded
# reranker costs VRAM continuously and compute only when queried.
if [ "${MYELIN_RERANK:-1}" = "1" ]; then
nohup "$LC/llama-server" \
  -m "$K/bge-reranker-v2-m3-Q8_0.gguf" \
  --host "$BIND_HOST" --port "$RERANK_PORT" \
  -c 8192 -b 8192 -ub 8192 -ngl 999 \
  --reranking --pooling rank \
  > /tmp/myelin-rerank.log 2>&1 &
echo $! > /tmp/myelin-rerank.pid
fi

for _ in $(seq 1 60); do
  r=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$BIND_HOST:$READER_PORT/health" || true)
  if [ "${MYELIN_EMBEDDER:-bge}" = "qwen" ]; then
    e=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$BIND_HOST:$EMBED_PORT/health" || true)
  else
    e=200
  fi
  if [ "${MYELIN_RERANK:-1}" = "1" ]; then
    k=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$BIND_HOST:$RERANK_PORT/health" || true)
  else
    k=200
  fi
  if [ "$r" = 200 ] && [ "$e" = 200 ] && [ "$k" = 200 ]; then
    echo "ready host=$BIND_HOST reader=:$READER_PORT embed=:$EMBED_PORT rerank=:$RERANK_PORT vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    exit 0
  fi
  sleep 5
done

echo "FAILED to become ready; see /tmp/myelin-{reader,embed,rerank}.log" >&2
exit 1
