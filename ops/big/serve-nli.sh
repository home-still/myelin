#!/usr/bin/env bash
# Serve M82's NLI cross-encoder on big (127.0.0.1:5820 by default), from the
# LME-V2 venv (torch + transformers). Run under the GPU lease; it takes about 1 GB of VRAM.
# `ops/big/stop-models.sh` does not stop it: stop it with `pkill -f nli_server.py`.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${MYELIN_NLI_PORT:-5820}"
PY="${MYELIN_NLI_PYTHON:-$HOME/myelin-lmev2/.venv/bin/python}"
pkill -f nli_server.py 2>/dev/null || true
MYELIN_NLI_PORT=$PORT nohup "$PY" nli_server.py > "$HOME/myelin-r4/logs/nli_server.log" 2>&1 &
for _ in $(seq 1 120); do
  curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q '"status": "ok"' && { echo "nli serving on :$PORT"; exit 0; }
  sleep 5
done
echo "nli server did not come up; see logs/nli_server.log" >&2
exit 1
