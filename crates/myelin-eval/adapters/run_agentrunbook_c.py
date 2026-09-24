#!/usr/bin/env python3
"""Run the LME-V2 harness's own AgentRunbook-C with a LOCAL controller (M54).

AgentRunbook-C (`10.48550/arXiv.2605.12493` §4.2) stores trajectories as
files and lets a coding agent search, inspect and select evidence at query
time; with Codex + GPT-5.4-mini it reaches 72.5 on LME-V2 against the best RAG
system's 48.5, with the same Qwen3.5-9B reader. This driver runs the vendored
implementation unmodified (`memory_modules/agentrunbook_c.py`), with the
Codex CLI pointed at a local model through an isolated `CODEX_HOME` — the
project runs on local models only.

Two phases, because the controller (Ternary Bonsai 2 27B, ~10 GB) and the
reader (Qwen3.5-9B) do not fit on big's card beside its other tenants:

  1. `--prompts-only`: the controller builds every question's memory context;
     the run stops cleanly once `prompt_rows.jsonl` is written, before the
     reader is needed.
  2. `--reuse-prompts-from <phase-1 dir>`: the reader answers from those exact
     prompts and the harness scores them (`run_myelin.reuse_prompts_from`).

The controller path, all local:

    codex exec -> responses_shim.py :5821 -> ssh tunnel :18081 -> big's llama-swap :8081

`codex_bonsai.config.toml` is the `CODEX_HOME/config.toml` (copy it into an
empty directory; set `<repo root>`). `responses_shim.py` makes two rewrites
llama.cpp's /v1/responses needs: extra system messages folded into
`instructions`, and an image in a tool output (Codex's `view_image`) replaced
by a text note, since the controller is text-only and llama.cpp refuses any
tool output that is not text (HTTP 400; 5 of the first 9 pilot questions died
on it before the rewrite).

Usage (phase 1):

    CODEX_HOME=<dir with config.toml> PYTHONPATH=vendor/longmemeval-v2:adapters \\
      ../../.venv/bin/python adapters/run_agentrunbook_c.py --data-root ../../data/lmev2 \\
      --domain web --questions <ids file> --codex-model qwen3.8-27b --prompts-only \\
      --output-dir ../../runs/m54_arc_web_prompts < /dev/null

`< /dev/null` matters: `codex exec` reads stdin until EOF when it is not a TTY
and otherwise waits forever (measured 2026-09-23).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import run_myelin as rm  # noqa: E402  (the myelin driver's tested helpers, and the harness path)
from data.public_data import (  # noqa: E402
    materialize_runtime_haystack,
    materialize_runtime_questions,
    write_json,
)


# The controller model every M54 host serves: one PTQ1_0 file, sha256-checked on
# each host (amendment 2). Which host, slots and Codex budget built each
# question's memory is in `controller_hosts` (`merge_arc_chunks.py`).
CONTROLLER_MODEL = "Ternary Bonsai 2 27B"


class PromptsBuilt(Exception):
    """Raised in place of the reader call when `--prompts-only` is set."""


def read_ids(path: str) -> list[str]:
    ids = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not ids:
        raise SystemExit(f"{path} lists no question ids")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path} repeats a question id")
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--domain", choices=["web", "enterprise"], required=True)
    p.add_argument("--tier", choices=["small", "medium"], default="small")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--questions", default=None, help="Question-id file (one per line, # comments).")
    # Controller: the Codex CLI with a local provider from $CODEX_HOME/config.toml.
    p.add_argument("--codex-binary", default="/opt/homebrew/bin/codex")
    p.add_argument("--codex-model", required=True)
    p.add_argument("--codex-reasoning-effort", default="medium")
    p.add_argument("--codex-timeout-seconds", type=float, default=1800.0)
    p.add_argument("--codex-max-retries", type=int, default=1)
    # Text only: the controller is served without a vision projector.
    p.add_argument("--evidence-mode", choices=["axtree", "image", "both"], default="axtree")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--reuse-prompts-from", default=None)
    # Reader and judge: the same as run_myelin's LME-V2 runs.
    p.add_argument("--reader-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--reader-base-url", default="http://127.0.0.1:5810/v1")
    p.add_argument("--reader-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--reader-temperature", type=float, default=0.6)
    p.add_argument("--reader-top-p", type=float, default=0.95)
    p.add_argument("--reader-top-k", type=int, default=20)
    p.add_argument("--reader-max-concurrent-requests", type=int, default=2)
    p.add_argument("--max-completion-tokens", type=int, default=20000)
    p.add_argument("--memory-context-max-tokens", type=int, default=200000)
    p.add_argument("--prompt-build-max-workers", type=int, default=1)
    p.add_argument("--evaluator-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--evaluator-base-url", default="http://127.0.0.1:5810/v1")
    p.add_argument("--evaluator-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--evaluator-reasoning-effort", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--evaluator-max-completion-tokens", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompts_only and args.reuse_prompts_from:
        raise SystemExit("--prompts-only and --reuse-prompts-from are the two phases; pass one")
    if not args.reuse_prompts_from:
        home = os.environ.get("CODEX_HOME")
        if not home or not (Path(home) / "config.toml").is_file():
            raise SystemExit(
                "CODEX_HOME must name a directory holding config.toml with the local provider; "
                "the controller runs on local models only"
            )
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runtime_dir = output_dir / "runtime_inputs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    selected = materialize_runtime_questions(
        data_root=data_root,
        domain=args.domain,
        question_ids=read_ids(args.questions) if args.questions else None,
        limit=None,
        output_path=runtime_dir / "questions.json",
    )
    materialize_runtime_haystack(
        data_root=data_root,
        tier=args.tier,
        selected_questions=selected,
        output_path=runtime_dir / "haystack.json",
    )
    # Provenance, written unconditionally (`standing` pairs on these keys).
    # Phase one reads nothing and records no reader. Phase two replays the
    # merged chunks' prompts and carries their controller mixture from
    # `controller_hosts.json`, which the merge writes from each chunk's
    # `controller.json` (M54 amendment, 2026-09-24).
    controller_hosts = None
    if args.reuse_prompts_from:
        manifest = Path(args.reuse_prompts_from).expanduser().resolve() / "controller_hosts.json"
        if not manifest.is_file():
            raise SystemExit(f"{manifest} does not exist: the merged prompts must name their controllers. Nothing was built.")
        controller_hosts = json.loads(manifest.read_text(encoding="utf-8"))
    memory_config = {
        "memory_type": "agentrunbook_c",
        "memory_params": {
            "controller_model": CONTROLLER_MODEL,
            "controller_hosts": controller_hosts,
            "reader_served_model": None if args.prompts_only else rm.served_model(args.reader_base_url),
            "evidence_mode": args.evidence_mode,
            "trajectory_pool_root": None,
            "query_codex_params": {
                "binary": args.codex_binary,
                "model": args.codex_model,
                "reasoning_effort": args.codex_reasoning_effort,
                "timeout_seconds": args.codex_timeout_seconds,
                "max_retries": args.codex_max_retries,
                "extra_config": [],
                "extra_args": [],
            },
        },
    }
    write_json(runtime_dir / "memory_config.json", memory_config)
    harness_argv = [
        "evaluation.harness",
        "--domain", args.domain,
        "--questions-path", str(runtime_dir / "questions.json"),
        "--haystack-path", str(runtime_dir / "haystack.json"),
        "--trajectories-path", str(data_root / "trajectories.jsonl"),
        "--memory-config-path", str(runtime_dir / "memory_config.json"),
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
        "--evaluator-base-url", args.evaluator_base_url,
        "--evaluator-api-key-env", args.evaluator_api_key_env,
        "--evaluator-reasoning-effort", args.evaluator_reasoning_effort,
        "--evaluator-max-completion-tokens", str(args.evaluator_max_completion_tokens),
        "--reader-disable-thinking",
    ]
    from evaluation import harness as harness_module  # noqa: E402

    if args.prompts_only:
        async def stop_before_reader(_args, _rows):
            raise PromptsBuilt()

        harness_module.generate_all_reader_outputs = stop_before_reader
    else:
        if args.reuse_prompts_from:
            rm.reuse_prompts_from(
                Path(args.reuse_prompts_from).expanduser().resolve(), selected, runtime_dir, harness_module
            )
        rm.preflight_reader_images(args, selected, harness_module)
    old_argv = sys.argv
    try:
        sys.argv = harness_argv
        harness_module.main()
    except PromptsBuilt:
        print(f"prompts built and saved to {output_dir}; stopped before the reader (--prompts-only)", flush=True)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
