#!/usr/bin/env python3
"""Merge M54's phase-one chunks into one domain's prompt set for the reader.

The M54 full run built AgentRunbook-C's memory prompts in chunks, on more than
one controller (amendments 1 and 2 in `docs/measurements/m54-local-file-controller.md`).
Each chunk directory holds the harness's `prompt_rows.jsonl` plus a
`controller.json` naming the host, slots and Codex context budget it ran on.
This writes one directory that `run_agentrunbook_c.py --reuse-prompts-from`
replays. It contains:

- `prompt_rows.jsonl`: every source's rows, in source order;
- `controller_hosts.json`: the controller mixture (`{"<host>/<slots>x<budget>": n}`)
  and the configuration each question ran on, so the pair can be reported by
  host and configuration as the amendment pre-registers.

Refuses a source without `controller.json` or `prompt_rows.jsonl`, a question
that appears twice, a source whose controller ran a different model file, and
an output directory that already exists. The reader phase itself refuses a
merge that does not cover exactly the domain's questions.

Usage (from the repo root):

    python3 crates/myelin-eval/adapters/merge_arc_chunks.py --out runs/m54_full_web_prompts \\
        runs/m54_arc_web_prompts runs/m54_full_web_c00a_prompts ...
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

# The one controller file every M54 chunk ran (amendment 2: sha256-checked on each host).
CONTROLLER_MODEL_FILE = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
CONTROLLER_SHA256_PREFIX = "53107f530aa52eb0"


def configuration(controller: dict) -> str:
    """`big/1x96000`: host, parallel slots and Codex context budget."""
    return f"{controller['host']}/{controller['slots']}x{controller['codex_context_window']}"


def read_source(src: Path) -> tuple[list[dict], dict]:
    rows_path, ctrl_path = src / "prompt_rows.jsonl", src / "controller.json"
    for path in (rows_path, ctrl_path):
        if not path.is_file():
            raise SystemExit(f"{path} does not exist: every source must be a finished chunk with its controller named")
    controller = json.loads(ctrl_path.read_text(encoding="utf-8"))
    if (controller.get("model_file"), controller.get("model_sha256_prefix")) != (
        CONTROLLER_MODEL_FILE,
        CONTROLLER_SHA256_PREFIX,
    ):
        raise SystemExit(f"{ctrl_path} names another controller model: {controller}")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if not rows:
        raise SystemExit(f"{rows_path} holds no rows")
    return rows, controller


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("sources", nargs="+")
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; a merge is written once")

    merged: list[dict] = []
    per_question: dict[str, str] = {}
    sources = []
    for src in map(Path, args.sources):
        rows, controller = read_source(src)
        config = configuration(controller)
        for row in rows:
            qid = row["question_id"]
            if qid in per_question:
                raise SystemExit(f"{qid} appears in {src} and in an earlier source")
            per_question[qid] = config
            merged.append(row)
        sources.append({"dir": str(src), "configuration": config, "questions": len(rows)})

    out.mkdir(parents=True)
    with (out / "prompt_rows.jsonl").open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row) + "\n")
    manifest = {
        "model_file": CONTROLLER_MODEL_FILE,
        "model_sha256_prefix": CONTROLLER_SHA256_PREFIX,
        "mixture": dict(sorted(Counter(per_question.values()).items())),
        "per_question": per_question,
        "sources": sources,
    }
    (out / "controller_hosts.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"{out}: {len(merged)} questions from {len(sources)} sources; mixture {manifest['mixture']}")


if __name__ == "__main__":
    sys.exit(main())
