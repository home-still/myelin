#!/usr/bin/env bash
# Stop the myelin reader + embedder on `big`. Safe to run when they are not up.
#
# This does NOT release the GPU claim — that is deliberate, so a caller can
# stop and restart models inside one window. Release separately with
# `gpu-tenant release`, and say so in /tmp/agent_chat: other agents on this box
# size their work against free VRAM.
set -uo pipefail

for name in reader embed; do
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
done

sleep 3
echo "vram=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
