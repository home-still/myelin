"""Agreement between the strict judge and a second grader, as Cohen's kappa.

Why this exists: the strict 9B judge decides every arm, and a benchmark's own
grader (LongMemEval's official grader, LoCoMo's LightMem grader) decides the
SOTA comparison. Their agreement used to be quoted as a percentage. That
overstates it whenever one label dominates: across 21 judges and ~541,000
judgments, raw agreement exceeds Cohen's kappa by 33-41 points (Norman, Rivera
& Hughes 2026, "Reliability without Validity", arXiv 2606.19544).

- kappa is `scorer_agreement.cohens_kappa`, so the repository keeps one
  implementation.
- Its 95% interval uses the large-sample standard error for two raters and
  two labels (Fleiss, Cohen & Everitt 1969, "Large sample standard errors of
  kappa and weighted kappa", `10.1037/h0028106`).
- The discordant cells are printed beside it, so the direction of each
  disagreement is visible, not only its size.

Inputs:
- RUN_SPEC is a run directory, or ','-joined seed replicates, as for
  `paired_ci.py`.
- The strict judge is each row's `score` in `<run>_judged/per_question.jsonl`
  (the rescored run `paired_ci.py` reads).
- The second grader is the `--verdicts` file inside `<run>`. Pairs are its
  questions; strict rows it does not grade are counted and printed.
- A strict score that is not 0 or 1 is refused, not rounded.

Replicates share their questions, so the pooled line has no interval; the
per-replicate lines do.

    python3 crates/myelin-eval/adapters/judge_agreement.py \\
        runs/r5_bundle_s1_grounded,runs/r5_bundle_s2_grounded,runs/r5_bundle_s3_grounded
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scorer_agreement import cohens_kappa, confusion  # noqa: E402

# Two-sided 95% normal quantile.
Z_95 = 1.959963984540054
DEFAULT_VERDICTS = "judge_verdicts_lme_official.json"


def kappa_se(pairs: list[tuple[int, int]]) -> float:
    """Large-sample standard error of Cohen's kappa, two raters, two labels.

    Fleiss, Cohen & Everitt 1969 (`10.1037/h0028106`), the non-null variance:

        var = ( sum_i p_ii [1 - (p_i. + p_.i)(1 - k)]^2
              + (1 - k)^2 sum_{i != j} p_ij (p_.i + p_j.)^2
              - [k - p_e (1 - k)]^2 ) / (n (1 - p_e)^2)
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("no paired items")
    labels = (0, 1)
    p = {(i, j): sum(1 for a, b in pairs if a == i and b == j) / n for i in labels for j in labels}
    row = {i: p[(i, 0)] + p[(i, 1)] for i in labels}
    col = {j: p[(0, j)] + p[(1, j)] for j in labels}
    p_e = sum(row[i] * col[i] for i in labels)
    if p_e == 1.0:
        raise ValueError("both raters give one label to every item; kappa has no standard error")
    k = cohens_kappa(pairs)
    diagonal = sum(p[(i, i)] * (1 - (row[i] + col[i]) * (1 - k)) ** 2 for i in labels)
    off = (1 - k) ** 2 * sum(p[(i, j)] * (col[i] + row[j]) ** 2 for i in labels for j in labels if i != j)
    var = (diagonal + off - (k - p_e * (1 - k)) ** 2) / (n * (1 - p_e) ** 2)
    return math.sqrt(max(var, 0.0))


def strict_scores(run_dir: str) -> dict[str, int]:
    judged = Path(f"{run_dir.rstrip('/')}_judged") / "per_question.jsonl"
    out: dict[str, int] = {}
    with open(judged, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            score = float(row["score"])
            if score not in (0.0, 1.0):
                raise SystemExit(f"{judged}: {row['question_id']} has strict score {score}, not 0 or 1")
            out[str(row["question_id"])] = int(score)
    if not out:
        raise SystemExit(f"{judged} is empty")
    return out


def second_verdicts(run_dir: str, verdicts: str) -> dict[str, int]:
    path = Path(run_dir) / verdicts
    table = json.loads(path.read_text(encoding="utf-8"))["verdicts"]
    return {str(k): int(v) for k, v in table.items()}


def pairs_for(run_dir: str, verdicts: str) -> tuple[list[tuple[int, int]], int]:
    """The (strict, second) pairs on the second grader's questions, and how
    many strict rows it does not grade (LoCoMo's LightMem grader skips the
    adversarial category). A question graded only by the second grader is an
    error."""
    a, b = strict_scores(run_dir), second_verdicts(run_dir, verdicts)
    unknown = set(b) - set(a)
    if unknown:
        raise SystemExit(f"{run_dir}: {len(unknown)} questions in {verdicts} have no strict row")
    return [(a[q], b[q]) for q in sorted(b)], len(set(a) - set(b))


def line(label: str, pairs: list[tuple[int, int]], interval: bool) -> str:
    both, strict_only, second_only, neither = confusion(pairs)
    n = len(pairs)
    k = cohens_kappa(pairs)
    agree = (both + neither) / n
    ci = ""
    if interval:
        se = kappa_se(pairs)
        ci = f" [{k - Z_95 * se:.3f}, {k + Z_95 * se:.3f}]"
    return (
        f"{label}: n={n} agreement {100 * agree:.1f}% kappa {k:.3f}{ci} | "
        f"both right {both}, strict only {strict_only}, second only {second_only}, both wrong {neither}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_spec", help="a run directory, or ','-joined seed replicates")
    parser.add_argument("--verdicts", default=DEFAULT_VERDICTS, help="the second grader's verdict file inside each run")
    args = parser.parse_args()
    runs = [r for r in args.run_spec.split(",") if r]
    print(f"strict = <run>_judged score; second = {args.verdicts}")
    pooled: list[tuple[int, int]] = []
    for run in runs:
        pairs, ungraded = pairs_for(run, args.verdicts)
        pooled += pairs
        print(line(run, pairs, interval=True) + (f" | {ungraded} strict rows not graded by the second" if ungraded else ""))
    if len(runs) > 1:
        print(line("pooled (replicates share questions: no interval)", pooled, interval=False))


if __name__ == "__main__":
    main()
