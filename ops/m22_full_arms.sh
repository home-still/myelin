#!/usr/bin/env bash
# M22 full arm sweep: 3 operating points x 2 domains over LME-V2 tier small.
# Sequential by necessity — one GPU serves the reranker, the reflect gate, the
# selector and the reader.
set -u
cd "$(dirname "$0")/../crates/myelin-eval" || exit 1
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
export OPENAI_API_KEY=dummy

# `big` is a shared host. Another tenant starting vLLM/olmOCR drove it to load
# 151 twice in one session; sshd stopped answering, the 5810/5813 tunnel went
# with it, and a 35-minute arm died on APIConnectionError. Block until every
# dependency answers rather than burning an arm into a dead host.
wait_for_services() {
  local waited=0
  while :; do
    if curl -s -m 8 http://192.168.1.110:6333/collections -o /dev/null \
      && curl -s -m 8 http://192.168.1.110:11434/api/tags -o /dev/null \
      && curl -s -m 8 http://127.0.0.1:5810/health -o /dev/null \
      && curl -s -m 8 http://127.0.0.1:5813/health -o /dev/null; then
      [ "$waited" -gt 0 ] && echo "SERVICES BACK after ${waited}s $(date -u +%H:%M:%S)"
      return 0
    fi
    if [ "$waited" -eq 0 ]; then
      echo "WAITING for qdrant/ollama/reader/rerank $(date -u +%H:%M:%S)"
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

run_arm() {
  local out="$1" domain="$2"
  shift 2
  if [ -f "../../runs/$out/aggregated_metrics.json" ]; then
    echo "SKIP $out (already complete)"
    return 0
  fi
  # Up to three attempts per arm. The harness keeps generations in memory and
  # writes per_question.jsonl only during scoring, so a mid-run host outage
  # destroys the whole arm and the only recovery is to run it again.
  local try
  for try in 1 2 3; do
    wait_for_services
    echo "START $out (try $try) $(date -u +%H:%M:%S)"
    rm -rf "../../runs/$out"
    PYTHONPATH=vendor/longmemeval-v2:adapters ../../.venv/bin/python adapters/run_myelin.py \
      --data-root ../../data/lmev2 --domain "$domain" --tier small \
      --mcp-url http://127.0.0.1:7447/mcp --mode investigate --k 25 --budget-tokens 10000 \
      --evaluator-model Qwen/Qwen3.5-9B --evaluator-base-url http://127.0.0.1:5810/v1 \
      --openai-max-retries 60 \
      "$@" --output-dir "../../runs/$out" 2>&1 \
      | grep -Ev "^(Building prompts|Generating|Scoring|Building memory)" | tail -4
    if [ -f "../../runs/$out/aggregated_metrics.json" ]; then
      echo "DONE  $out $(date -u +%H:%M:%S)"
      return 0
    fi
    echo "FAILED $out (try $try) $(date -u +%H:%M:%S)"
  done
  echo "GIVING UP on $out"
  return 1
}

run_arm m22_base_web   web                    
run_arm m22_base_ent   enterprise             
run_arm m22_nodate_web web        --undated
run_arm m22_nodate_ent enterprise --undated
run_arm m22_sel_web    web        --undated --select
run_arm m22_sel_ent    enterprise --undated --select
echo "SWEEP COMPLETE $(date -u +%H:%M:%S)"
