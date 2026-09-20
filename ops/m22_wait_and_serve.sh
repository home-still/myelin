#!/usr/bin/env bash
# Bring myelin's reader+reranker up on `big` as soon as the card has room.
#
# `big` is shared and two other tenants are outside `gpu-tenant`: llama-swap
# (TTL-evicted, 600 s) and ollama (rolling keep_alive, refreshed by the
# household voice assistant). Evicting either mid-use is not ours to do, so
# poll for headroom instead. serve-models.sh needs ~12 GB at
# SLOTS=2 / CTX=65536.
set -u
NEED_MIB="${NEED_MIB:-12500}"
cd "$(dirname "$0")/.." || exit 1

while :; do
  free=$(ssh -o ConnectTimeout=10 big \
    "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits" 2>/dev/null \
    | awk -F', *' '{print $1 - $2}')
  if [ -z "${free:-}" ]; then
    echo "$(date -u +%H:%M:%S) big unreachable"
    sleep 30
    continue
  fi
  echo "$(date -u +%H:%M:%S) free=${free} MiB (need ${NEED_MIB})"
  if [ "$free" -ge "$NEED_MIB" ]; then
    echo "starting models"
    if ssh big "MYELIN_READER_SLOTS=2 MYELIN_READER_CTX=65536 bash -s" < ops/big/serve-models.sh; then
      echo "MODELS READY $(date -u +%H:%M:%S)"
      exit 0
    fi
    echo "serve-models.sh failed; retrying"
  fi
  sleep 30
done
