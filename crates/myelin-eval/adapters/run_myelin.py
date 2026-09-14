#!/usr/bin/env python3
"""Run the official LongMemEval-V2 harness against `myelin`.

A sibling of the vendored `evaluation/run_eval.py`, not a replacement for it.
`run_eval.py` does three things — materialise the runtime question and
haystack files, write a `memory_config.json`, then set `sys.argv` and call
`evaluation.harness.main()` — and it is unusable for us only because its
`--method` is a closed `choices=` set. Everything downstream takes
`--memory-config-path` and is fully general, so this file does the same three
things with `"memory_type": "myelin"` and calls the same `main()`. The
harness, its scorers and its leaderboard builders run unmodified
(`PLAN.md` §3.3).

Two things this can do that `run_eval.py` cannot, both because it talks to
`evaluation.harness` directly:

* **`--evaluator-base-url`.** `harness.py` accepts one; `run_eval.py` never
  passes it, hardwiring the 156 LLM-judged questions to OpenAI. We point the
  judge wherever the run's manifest says, including a local server.
* **Importing the adapter first.** `memory_modules/memory.py` populates its
  registry from imports at the bottom of its own file, so `myelin` only
  exists as a `memory_type` once `adapters/myelin.py` has been imported.
  That import is the registration.

The memory itself is **not** built here. `MyelinMemory.insert()` asserts
rather than indexes, because building means running the Rust write path over
the haystack:

    myelin-eval build --corpus lme-v2-small
    myelin-mcp --serve 127.0.0.1:7446 \\
        --collection myelin_lme_v2_small --ledger data/lme_v2_small.ledger

Usage:

    PYTHONPATH=vendor/longmemeval-v2:adapters ../../.venv/bin/python \\
        adapters/run_myelin.py \\
        --data-root /tmp/lmev2 --domain web --tier small \\
        --output-dir runs/myelin_fast_web_small
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "longmemeval-v2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.public_data import (  # noqa: E402
    materialize_runtime_haystack,
    materialize_runtime_questions,
    write_json,
)

# Importing the adapter is what runs `@register_memory`. Without this line
# the harness raises `Unknown memory_type: myelin`, which is the failure mode
# a lazy `# noqa` would hide — hence the explicit note rather than a bare
# import.
import myelin  # noqa: E402,F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official LongMemEval-V2 harness against myelin."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--domain", choices=["web", "enterprise"], required=True)
    parser.add_argument("--tier", choices=["small", "medium"], default="small")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-ids", nargs="*", default=None)

    # Which memory to read. The tenant defaults to `<tier>/<domain>`, which is
    # exactly what `myelin-eval build --corpus lme-v2-*` writes, so the two
    # halves cannot drift apart silently.
    parser.add_argument("--mcp-url", default=os.getenv("MYELIN_MCP_URL", "http://127.0.0.1:7446/mcp"))
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--budget-tokens", type=int, default=2048)
    parser.add_argument("--mode", choices=["recall", "investigate"], default="recall")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument(
        "--tau-abstain",
        type=float,
        default=None,
        help="Withhold the evidence set when the best cross-encoder score is below "
        "this, so the reader abstains instead of answering from a bad pool.",
    )

    # Reader. Defaults are this project's tunnelled llama-server, not the
    # harness's `localhost:8023`, because a default that points at nothing is
    # a 10-minute debugging session every time.
    # `Qwen/Qwen3.5-9B` exactly, not the local llama-server alias.
    # harness.py:857 gates `chat_template_kwargs={"enable_thinking": False}`
    # on `args.model == "Qwen/Qwen3.5-9B"` — a string compare — so any other
    # spelling silently leaves thinking ON, and this reader then spends the
    # whole completion budget on reasoning_content and returns content "",
    # which the harness rejects as `Model returned empty text`. Measured:
    # that is exactly how the first end-to-end run failed. llama-server
    # ignores the field and serves the loaded GGUF regardless, so the
    # upstream spelling is both correct for the manifest and required here.
    parser.add_argument("--reader-model", default=os.getenv("READER_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--reader-base-url", default=os.getenv("READER_BASE_URL", "http://127.0.0.1:5810/v1"))
    parser.add_argument("--reader-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reader-temperature", type=float, default=0.6)
    parser.add_argument("--reader-top-p", type=float, default=0.95)
    parser.add_argument("--reader-top-k", type=int, default=20)
    parser.add_argument("--reader-max-concurrent-requests", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=20000)
    parser.add_argument("--memory-context-max-tokens", type=int, default=200000)
    # Thinking off by default: measured on this reader, Qwen3.5-9B spends the
    # whole completion budget on `reasoning_content` and returns
    # `content: ""`. See `llm/openai.rs`.
    parser.add_argument(
        "--reader-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prompt-build-max-workers", type=int, default=4)

    # Judge. `run_eval.py` hardwires OpenAI; `harness.py` does not, so the
    # base URL is exposed here and recorded in the manifest.
    parser.add_argument("--evaluator-model", default=os.getenv("EVALUATOR_MODEL", "gpt-5.2"))
    parser.add_argument("--evaluator-base-url", default=os.getenv("EVALUATOR_BASE_URL"))
    parser.add_argument("--evaluator-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--evaluator-reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--evaluator-max-completion-tokens", type=int, default=4096)

    parser.add_argument("--shuffle-questions-seed", type=int, default=None)
    return parser.parse_args()


def parse_question_ids(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    ids: list[str] = []
    for chunk in raw:
        ids.extend(part for part in chunk.replace(",", " ").split() if part)
    return ids or None


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runtime_dir = output_dir / "runtime_inputs"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    selected_questions = materialize_runtime_questions(
        data_root=data_root,
        domain=args.domain,
        question_ids=parse_question_ids(args.question_ids),
        limit=args.limit,
        output_path=runtime_dir / "questions.json",
    )
    materialize_runtime_haystack(
        data_root=data_root,
        tier=args.tier,
        selected_questions=selected_questions,
        output_path=runtime_dir / "haystack.json",
    )

    tier_slug = f"lme_v2_{args.tier}"
    memory_config = {
        "memory_type": "myelin",
        "memory_params": {
            "url": args.mcp_url,
            "tenant": args.tenant or f"{tier_slug}/{args.domain}",
            "namespace": args.namespace or tier_slug,
            "k": args.k,
            "budget_tokens": args.budget_tokens,
            "tau_abstain": args.tau_abstain,
            "mode": args.mode,
            "max_steps": args.max_steps,
        },
    }
    memory_config_path = runtime_dir / "memory_config.json"
    write_json(memory_config_path, memory_config)

    harness_argv = [
        "evaluation.harness",
        "--domain", args.domain,
        "--questions-path", str(runtime_dir / "questions.json"),
        "--haystack-path", str(runtime_dir / "haystack.json"),
        "--trajectories-path", str(data_root / "trajectories.jsonl"),
        "--memory-config-path", str(memory_config_path),
        "--output-dir", str(output_dir),
        "--model", args.reader_model,
        "--base-url", args.reader_base_url,
        "--api-key-env", args.reader_api_key_env,
        "--temperature", str(args.reader_temperature),
        "--top-p", str(args.reader_top_p),
        "--top-k", str(args.reader_top_k),
        "--max-completion-tokens", str(args.max_completion_tokens),
        "--memory-context-max-tokens", str(args.memory_context_max_tokens),
        "--reader-max-concurrent-requests", str(args.reader_max_concurrent_requests),
        "--prompt-build-max-workers", str(args.prompt_build_max_workers),
        "--evaluator-model", args.evaluator_model,
        "--evaluator-api-key-env", args.evaluator_api_key_env,
        "--evaluator-reasoning-effort", args.evaluator_reasoning_effort,
        "--evaluator-max-completion-tokens", str(args.evaluator_max_completion_tokens),
    ]
    if args.evaluator_base_url:
        harness_argv.extend(["--evaluator-base-url", args.evaluator_base_url])
    if not args.reader_enable_thinking:
        harness_argv.append("--reader-disable-thinking")
    if args.shuffle_questions_seed is not None:
        harness_argv.extend(["--shuffle-questions-seed", str(args.shuffle_questions_seed)])

    from evaluation.harness import main as harness_main  # noqa: E402

    old_argv = sys.argv
    try:
        sys.argv = harness_argv
        harness_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
