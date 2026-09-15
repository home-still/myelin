"""Paired bootstrap confidence intervals over two LongMemEval-V2 runs.

Two runs on the same question set are *paired*: question `q` is either hard
for both systems or easy for both, and that shared difficulty is nuisance
variance. Comparing two independent CIs throws the pairing away and is
needlessly conservative — overlapping marginal intervals routinely hide a
difference that is consistent question-by-question. Resampling the *per-
question difference* keeps it.

Why this exists: `docs/measurements/m7-step-value-curve.md` reports a curve
whose abstention column moves in 5.9-point increments over 17 questions, and
records the levels as provisional. This is how that caveat gets settled rather
than repeated.

The statistic is the paired mean difference

    d_bar = (1/n) * sum_q [ score_A(q) - score_B(q) ]

resampled with replacement over questions (BCa is not used: `score` here is a
0/1 indicator, the bootstrap distribution is discrete and near-symmetric, and
the acceleration term adds complexity for no meaningful correction at these
sample sizes).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_scores(run_dir: str) -> dict[str, float]:
    """Map question_id -> score for one completed run."""
    scores: dict[str, float] = {}
    with open(Path(run_dir) / "per_question.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scores[str(row["question_id"])] = float(row["score"])
    return scores


def load_flags(run_dir: str) -> dict[str, bool]:
    """Map question_id -> is_abstention_problem, for stratified reporting."""
    flags: dict[str, bool] = {}
    with open(Path(run_dir) / "per_question.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            flags[str(row["question_id"])] = bool(row["is_abstention_problem"])
    return flags


def paired_bootstrap(
    a: list[float],
    b: list[float],
    iterations: int = 20000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Return (mean_diff, lo, hi, p_two_sided) for paired samples a - b.

    `p_two_sided` is the bootstrap achieved significance level: the share of
    resamples whose mean difference falls on the opposite side of zero from
    the observed one, doubled. With no resamples crossing zero it is reported
    as `< 2/iterations` by the caller rather than as a bare 0.
    """
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("no paired questions")
    diffs = [x - y for x, y in zip(a, b)]
    observed = sum(diffs) / n

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    lo = means[int((alpha / 2) * iterations)]
    hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]

    # Achieved significance level against H0: d_bar == 0.
    if observed >= 0:
        tail = sum(1 for m in means if m <= 0.0)
    else:
        tail = sum(1 for m in means if m >= 0.0)
    p = min(1.0, 2.0 * tail / iterations)
    return observed, lo, hi, p


def compare(run_a: str, run_b: str, iterations: int, seed: int) -> None:
    sa, sb = load_scores(run_a), load_scores(run_b)
    flags = load_flags(run_a)
    shared = sorted(set(sa) & set(sb))
    if not shared:
        raise SystemExit(f"no shared question ids between {run_a} and {run_b}")
    dropped = (len(sa) - len(shared)) + (len(sb) - len(shared))

    print(f"A = {run_a}")
    print(f"B = {run_b}")
    print(f"paired on {len(shared)} questions" + (f" ({dropped} unpaired dropped)" if dropped else ""))
    print()

    strata = [
        ("overall", shared),
        ("non-abstention", [q for q in shared if not flags.get(q, False)]),
        ("abstention", [q for q in shared if flags.get(q, False)]),
    ]
    header = f"{'stratum':<16}{'n':>5}{'A':>9}{'B':>9}{'A-B':>9}{'95% CI':>20}{'p':>10}"
    print(header)
    print("-" * len(header))
    for name, ids in strata:
        if not ids:
            continue
        a = [sa[q] for q in ids]
        b = [sb[q] for q in ids]
        d, lo, hi, p = paired_bootstrap(a, b, iterations=iterations, seed=seed)
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        ci = f"[{lo * 100:+.1f}, {hi * 100:+.1f}]"
        p_str = f"<{2 / iterations:.4f}" if p == 0.0 else f"{p:.4f}"
        sig = "  *" if lo > 0 or hi < 0 else ""
        print(
            f"{name:<16}{len(ids):>5}{mean_a * 100:>8.1f}%{mean_b * 100:>8.1f}%"
            f"{d * 100:>+8.1f}{ci:>20}{p_str:>10}{sig}"
        )
    print()
    print("* = 95% CI excludes zero.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    compare(args.run_a, args.run_b, args.iterations, args.seed)


if __name__ == "__main__":
    main()
