#!/usr/bin/env bash
# Bring up the myelin reader + reranker + embedder on `bmb` (Apple M4 Pro,
# 48 GB unified memory, Metal), as standalone llama-servers on the same ports
# `ops/big/serve-models.sh` uses on the 3090 — so the driver on the Mac needs
# only a different tunnel target, not a different configuration.
#
# Run this FROM `bmb` (`ssh bmb bash -s < ops/bmb/serve-models.sh`). It is the
# fallback for the evenings `big`'s card is full of ungoverned use (measured
# 2026-09-22: ollama + the distill embedder left ~3 GB). Same llama.cpp commit
# on both hosts (b10450 / ece963f41 — Homebrew here, the cuda-ece963 build on
# big), same GGUFs byte for byte (copied from big, sha256-verified), so the
# only thing that differs is the backend, Metal against CUDA.
#
# **A Metal arm pairs against a Metal base.** Greedy decoding on a different
# backend can change an answer on some rows, and every paired CI in this
# project rests on byte-identical answers on untouched rows. Before any arm
# runs here, re-run the shipped operating point here and pair against THAT
# (`runs/bmb_base`), not against a CUDA base. The delta between the two bases
# is itself a reproducibility measurement worth recording.
#
# Deliberately NOT wired into bmb's llama-swap (0.0.0.0:9292, API-key gated,
# one model resident at a time): a reranker call followed by a reader call on
# every query would swap models twice per row.
set -euo pipefail

LS=/opt/homebrew/bin/llama-server
M="$HOME/llm/models"
BIND_HOST="${MYELIN_BIND_HOST:-127.0.0.1}"
READER_PORT="${MYELIN_READER_PORT:-5810}"
EMBED_PORT="${MYELIN_EMBED_PORT:-5811}"
RERANK_PORT="${MYELIN_RERANK_PORT:-5813}"
READER_SLOTS="${MYELIN_READER_SLOTS:-2}"
READER_CTX="${MYELIN_READER_CTX:-32768}"
# `--reasoning-budget`: -1 unrestricted (every run before M44); M44 R2 runs
# at 1024, and `bench --reader-thinking` probes it before the first row.
READER_THINK_BUDGET="${MYELIN_READER_THINK_BUDGET:-1024}"
# Thinking off server-wide, for the reason big's script gives: third-party
# clients (the vendored LME-V2 evaluator) never send the flag themselves.
TEMPLATE_KWARGS='{"enable_thinking":false}'

for f in qwen/Qwen3.5-9B-UD-Q4_K_XL.gguf bge/bge-reranker-v2-m3-Q8_0.gguf bge/bge-m3.gguf; do
  if [ ! -s "$M/$f" ]; then
    echo "serve-models: missing $M/$f — copy it from big first (see ops/bmb/README.md)" >&2
    exit 1
  fi
done

# Retire whatever already listens on our ports (the process list is the
# fact, the pidfile is a claim — see the M19 note in ops/big/serve-models.sh).
# `lsof`, because macOS has no `fuser -k`; by port and not by name, because
# bmb's llama-swap runs its own llama-server children.
for port in "$READER_PORT" "$EMBED_PORT" "$RERANK_PORT"; do
  pids=$(lsof -ti "tcp:$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "retiring an orphan already listening on :$port"
    kill $pids 2>/dev/null || true
  fi
done
sleep 2

# The reader. Flags identical to big's apart from the absent vision
# projector (text-only benchmarks; it was never on the text path).
nohup "$LS" \
  -m "$M/qwen/Qwen3.5-9B-UD-Q4_K_XL.gguf" \
  --host "$BIND_HOST" --port "$READER_PORT" \
  -c "$READER_CTX" -ngl 999 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -np "$READER_SLOTS" -cb \
  --jinja --chat-template-kwargs "$TEMPLATE_KWARGS" \
  --reasoning-budget "$READER_THINK_BUDGET" \
  > /tmp/myelin-reader.log 2>&1 &
echo $! > /tmp/myelin-reader.pid

# The dense embedder: the same bge-m3 file ollama serves on big (its blob,
# copied), CLS pooling as ollama uses for bge-m3, served over llama.cpp's
# OpenAI-compatible /v1/embeddings. Point the driver at it with
# MYELIN_EMBED__URL=http://127.0.0.1:<tunnelled port>/v1 when big's ollama is
# not reachable; otherwise the default (big's ollama) still works over the LAN.
nohup "$LS" \
  -m "$M/bge/bge-m3.gguf" \
  --host "$BIND_HOST" --port "$EMBED_PORT" \
  -c 8192 -b 8192 -ub 8192 -ngl 999 \
  --embeddings --pooling cls --no-webui \
  > /tmp/myelin-embed.log 2>&1 &
echo $! > /tmp/myelin-embed.pid

# The cross-encoder, flags as on big: --reranking requires --pooling rank, and
# -b/-ub 8192 so a (query, document) pair fits one physical batch.
nohup "$LS" \
  -m "$M/bge/bge-reranker-v2-m3-Q8_0.gguf" \
  --host "$BIND_HOST" --port "$RERANK_PORT" \
  -c 8192 -b 8192 -ub 8192 -ngl 999 \
  --reranking --pooling rank \
  > /tmp/myelin-rerank.log 2>&1 &
echo $! > /tmp/myelin-rerank.pid

for _ in $(seq 1 90); do
  r=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$BIND_HOST:$READER_PORT/health" || true)
  e=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$BIND_HOST:$EMBED_PORT/health" || true)
  k=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$BIND_HOST:$RERANK_PORT/health" || true)
  if [ "$r" = 200 ] && [ "$e" = 200 ] && [ "$k" = 200 ]; then
    echo "ready host=$BIND_HOST reader=:$READER_PORT embed=:$EMBED_PORT rerank=:$RERANK_PORT slots=$READER_SLOTS ctx=$READER_CTX think_budget=$READER_THINK_BUDGET"
    exit 0
  fi
  sleep 2
done
echo "FAILED to become ready; see /tmp/myelin-{reader,embed,rerank}.log on bmb" >&2
exit 1
