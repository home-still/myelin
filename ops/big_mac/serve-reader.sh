#!/usr/bin/env bash
# Serve the myelin reader on `big_mac` (Apple M1 Max, 32-core GPU, 32 GB,
# Metal) for BUILD passes only — today, `myelin-eval events-extract` (M50).
#
# Run FROM big_mac: `ssh big_mac bash -s < ops/big_mac/serve-reader.sh`.
# Reader only: no embedder, no reranker. A build pass needs neither; `bench`
# and the judge never run here (see ops/big_mac/README.md for why).
#
# llama.cpp comes from the official macOS arm64 release unpacked under
# ~/llm/ (no Homebrew: big_mac's Homebrew failed installing a dependency on
# 2026-09-23, and a self-contained release changes nothing system-wide).
set -euo pipefail

LC="${MYELIN_LLAMA_DIR:-$HOME/llm/llama-b11126/llama-b11126}"
MODEL="$HOME/llm/models/qwen/Qwen3.5-9B-UD-Q4_K_XL.gguf"
PORT="${MYELIN_READER_PORT:-5810}"
SLOTS="${MYELIN_READER_SLOTS:-4}"
CTX="${MYELIN_READER_CTX:-65536}"
# Thinking off server-wide, as on big and bmb: the extractor sends
# enable_thinking=false per request anyway, and anything else that reaches
# this port should get a plain completion.
TEMPLATE_KWARGS='{"enable_thinking":false}'

[ -s "$MODEL" ] || { echo "serve-reader: missing $MODEL — copy it from bmb (ops/big_mac/README.md)" >&2; exit 1; }
[ -x "$LC/llama-server" ] || { echo "serve-reader: missing $LC/llama-server" >&2; exit 1; }

pids=$(lsof -ti "tcp:$PORT" 2>/dev/null || true)
[ -n "$pids" ] && { echo "retiring what listens on :$PORT"; kill $pids 2>/dev/null || true; sleep 2; }

nohup "$LC/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" -np "$SLOTS" -ngl 999 --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --chat-template-kwargs "$TEMPLATE_KWARGS" \
  > /tmp/myelin-reader.log 2>&1 &
echo $! > /tmp/myelin-reader.pid

for _ in $(seq 1 90); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health" || true)" = 200 ]; then
    echo "ready :$PORT slots=$SLOTS ctx=$CTX"; exit 0
  fi
  sleep 2
done
echo "FAILED; see /tmp/myelin-reader.log on big_mac" >&2
exit 1
