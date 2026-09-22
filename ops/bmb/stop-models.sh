#!/usr/bin/env bash
# Stop the myelin reader, embedder and reranker on `bmb`. By port, not by
# name: bmb's llama-swap runs its own llama-server children, and a pattern
# kill would take them too (the same rule as ops/big/stop-models.sh).
set -uo pipefail

READER_PORT="${MYELIN_READER_PORT:-5810}"
EMBED_PORT="${MYELIN_EMBED_PORT:-5811}"
RERANK_PORT="${MYELIN_RERANK_PORT:-5813}"

for port in "$READER_PORT" "$EMBED_PORT" "$RERANK_PORT"; do
  pids=$(lsof -ti "tcp:$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null && echo "stopped :$port ($pids)"
  else
    echo ":$port not running"
  fi
done
rm -f /tmp/myelin-reader.pid /tmp/myelin-embed.pid /tmp/myelin-rerank.pid
