#!/usr/bin/env python3
"""M54's pre-registered breakdown: the paired result by controller configuration.

Amendment 2 in `docs/measurements/m54-local-file-controller.md` scores the full
AgentRunbook-C pair once, and also reports accuracy by the host and
configuration each question's memory was built on, each with a 95% CI. The
configuration of every question comes from the arm's own
`runtime_inputs/memory_config.json` (`memory_params.controller_hosts.per_question`,
written by `merge_arc_chunks.py`), so this reads nothing the run did not record.

Refuses an arm half whose memory config carries no per-question controller.

Usage (from the repo root):

    python3 crates/myelin-eval/adapters/arc_by_controller.py \\
        runs/m47_base_web+runs/m47_base_ent runs/m54_full_web+runs/m54_full_ent
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_ci import load_scores, paired_bootstrap  # noqa: E402

BOOTSTRAP_ITERATIONS = 20000


def controller_of(arm_spec: str) -> dict[str, str]:
    configs: dict[str, str] = {}
    for d in arm_spec.split("+"):
        path = Path(d) / "runtime_inputs" / "memory_config.json"
        hosts = json.loads(path.read_text(encoding="utf-8"))["memory_params"].get("controller_hosts")
        if not hosts or "per_question" not in hosts:
            raise SystemExit(f"{path} records no per-question controller; was the reader fed a merged prompt set?")
        configs.update(hosts["per_question"])
    return configs


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    base, arm = load_scores(sys.argv[1]), load_scores(sys.argv[2])
    configs = controller_of(sys.argv[2])
    shared = sorted(set(base) & set(arm))
    missing = [q for q in shared if q not in configs]
    if missing:
        raise SystemExit(f"{len(missing)} paired questions have no controller recorded (first: {missing[0]})")
    groups: dict[str, list[str]] = defaultdict(list)
    for q in shared:
        groups[configs[q]].append(q)
    print(f"paired {len(shared)}; by controller configuration (host/slots x Codex budget)")
    for name, ids in [("all", shared)] + sorted(groups.items()):
        a, b = [arm[q] for q in ids], [base[q] for q in ids]
        d, lo, hi, _ = paired_bootstrap(a, b, iterations=BOOTSTRAP_ITERATIONS, seed=0)
        print(
            f"{name:16s} n={len(ids):4d} base={100 * sum(b) / len(ids):6.2f} "
            f"arm={100 * sum(a) / len(ids):6.2f} d={100 * d:+6.2f} [{100 * lo:+6.2f}, {100 * hi:+6.2f}]"
        )


if __name__ == "__main__":
    sys.exit(main())
