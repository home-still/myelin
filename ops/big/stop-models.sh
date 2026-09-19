#!/usr/bin/env bash
# Stop the myelin reader, embedder and reranker on `big`. Safe to run when
# they are not up.
#
# This does NOT release the GPU claim — that is deliberate, so a caller can
# stop and restart models inside one window. Release separately with
# `gpu-tenant release`, and say so in /tmp/agent_chat: other agents on this box
# size their work against free VRAM.
set -uo pipefail

# Ports, so an orphan cannot survive a stop. Kept parallel to the pidfile
# names below.
declare -A PORT=( [reader]="${MYELIN_READER_PORT:-5810}"
                  [embed]="${MYELIN_EMBED_PORT:-5811}"
                  [rerank]="${MYELIN_RERANK_PORT:-5813}" )

for name in reader embed rerank; do
  pidfile="/tmp/myelin-$name.pid"
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    if kill "$pid" 2>/dev/null; then
      echo "stopped $name (pid $pid)"
    else
      echo "$name (pid $pid) was not running"
    fi
    rm -f "$pidfile"
  else
    echo "$name: no pidfile"
  fi
  # A pidfile is a claim; the listening socket is the fact.
  #
  # `serve-models.sh` writes the pidfile unconditionally, so two runs in one
  # session leave the first process with no pidfile pointing at it. In M19
  # that orphaned a reranker on 994 MiB of a SHARED card for two and a half
  # hours *after* this script had reported it "not running" and the session
  # had announced itself finished. Reconciling against the port closes that.
  #
  # `fuser -k` and not `pkill -f llama-server`: another tenant on this box
  # runs its own llama-server processes and a pattern kill would take theirs.
  if fuser -k -n tcp "${PORT[$name]}" 2>/dev/null; then
    echo "  also retired an orphan still listening on :${PORT[$name]}"
  fi
done

sleep 3
echo "vram=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
