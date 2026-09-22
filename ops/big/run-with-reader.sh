#!/usr/bin/env bash
# Run one `myelin-eval` command, bringing the reader back if it dies mid-run.
#
# Why this exists: `big` is a shared box with 31 GB of RAM and no useful swap,
# and a household voice assistant holds ~12.4 GB RSS for its ollama model on a
# rolling keep_alive. Under that pressure the reader `llama-server` has been
# killed out from under a run three times in one session — the harness sees
# `error sending request for url (http://127.0.0.1:5810/...)` and a 35-minute
# full-set run is lost, because `bench` writes its artifact only at the end.
#
# A retry alone is not enough: the reader stays dead, so all five attempts fail
# the same way. This checks the endpoint between attempts and re-serves it.
#
# Deliberately NOT a retry inside `bench`: the run is the measurement, and a
# harness that silently papers over a dead dependency hides exactly the kind of
# degradation `/tmp/chat.md` on `big` documents (STT/LLM quality falls before
# the process dies). Here the recovery is visible in the log.
#
#   ops/big/run-with-reader.sh bench --corpus locomo --out runs/x
set -uo pipefail

BIN="${MYELIN_EVAL_BIN:-$HOME/.cargo-target-shared/global/release/myelin-eval}"
READER_URL="${MYELIN_READER_HEALTH:-http://127.0.0.1:5810/health}"
ATTEMPTS="${MYELIN_ATTEMPTS:-5}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

reader_up() {
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$READER_URL" || true)" = 200 ]
}

serve() {
  echo "run-with-reader: reader is down, re-serving on big" >&2
  ssh big "MYELIN_GPU_WHO=${MYELIN_GPU_WHO:-myelin@mac_air} \
           MYELIN_READER_SLOTS=${MYELIN_READER_SLOTS:-1} \
           MYELIN_READER_CTX=${MYELIN_READER_CTX:-16384} \
           MYELIN_READER_THINK_BUDGET=${MYELIN_READER_THINK_BUDGET:--1} \
           MYELIN_MMPROJ=${MYELIN_MMPROJ:-1} bash -s" < "$HERE/serve-models.sh" >&2
}

for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "run-with-reader: attempt $attempt/$ATTEMPTS: $*"
  if ! reader_up; then
    serve
    sleep 5
  fi
  # A retried `bench` resumes the rows the failed attempt finished, instead
  # of scoring them again: on 2026-09-22 a reader kill 249 rows into a
  # 500-row arm cost the whole 249 because the retry started over.
  extra=()
  if [ "$attempt" -gt 1 ] && [ "${1:-}" = bench ]; then
    extra=(--resume)
  fi
  if "$BIN" "$@" "${extra[@]}"; then
    exit 0
  fi
  echo "run-with-reader: attempt $attempt failed" >&2
  sleep 20
done
echo "run-with-reader: giving up after $ATTEMPTS attempts" >&2
exit 1
