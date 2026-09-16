"""Agreement between two deterministic scorers and an LLM judge.

Why this exists: `docs/measurements/m13-temporal-axis.md` closed by observing
that the temporal stratum was being measured as string overlap against golds
like `The sunday before 25 May 2023`, and M14 replaces that with a date-aware
interval scorer (`crates/myelin-eval/src/temporal.rs`). "The new scorer scores
higher" cannot be the gate — the whole finding is that token F1 was scoring
*too high*, by handing partial credit to date-shaped near-misses. So the gate
is agreement with something that is not either scorer: a judge.

The judge is not a metric and never enters a reported accuracy number
(`crates/myelin-eval/src/judge.rs` says why). It is an arbiter, and the
disagreement dump at the bottom of this report is the audit trail that lets a
reader check the judge rather than trust it.

Both scorers are binarised at `--threshold` (default 0.5, fixed in advance)
because the judge's verdict is binary. The paired bootstrap comes from
`paired_ci.paired_bootstrap` — one bootstrap implementation in this repo, not
two — over the per-question agreement indicators

    a_q = 1[temporal_bin(q) == verdict(q)]
    b_q = 1[token_f1_bin(q) == verdict(q)]

so the interval is on the *difference in agreement*, not on either scorer's
score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paired_ci import paired_bootstrap  # noqa: E402


def load_rows(run_dir: str) -> dict[str, dict]:
    """Map question_id -> row for one rescored run."""
    rows: dict[str, dict] = {}
    with open(Path(run_dir) / "per_question.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["question_id"])] = row
    if not rows:
        raise SystemExit(f"{run_dir}/per_question.jsonl is empty")
    missing = [k for k in ("score_token_f1", "score_temporal", "temporal_kind") if k not in next(iter(rows.values()))]
    if missing:
        raise SystemExit(
            f"{run_dir}/per_question.jsonl predates M14: it has no {missing}. "
            "Re-score it with `myelin-eval rescore`."
        )
    return rows


def load_verdicts(path: str) -> tuple[str, dict[str, int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(payload["model"]), {str(k): int(v) for k, v in payload["verdicts"].items()}


def cohens_kappa(pairs: list[tuple[int, int]]) -> float:
    """Cohen's kappa for two binary raters over the same items.

    Reported alongside raw agreement because agreement alone is inflated when
    one label dominates, and on this docket the judge marks most answers
    wrong.
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("no paired items")
    observed = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def confusion(pairs: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """(scorer 1 & judge 1, scorer 1 & judge 0, scorer 0 & judge 1, scorer 0 & judge 0)."""
    tp = sum(1 for s, j in pairs if s == 1 and j == 1)
    fp = sum(1 for s, j in pairs if s == 1 and j == 0)
    fn = sum(1 for s, j in pairs if s == 0 and j == 1)
    tn = sum(1 for s, j in pairs if s == 0 and j == 0)
    return tp, fp, fn, tn


def report(
    run_dir: str,
    judge_path: str,
    category: int | None,
    threshold: float,
    iterations: int,
    seed: int,
) -> None:
    rows = load_rows(run_dir)
    model, verdicts = load_verdicts(judge_path)

    orphans = sorted(q for q in verdicts if q not in rows)
    if orphans:
        raise SystemExit(
            f"{len(orphans)} judged questions have no row in {run_dir} "
            f"(e.g. {orphans[0]!r}); the verdicts file and the run must be the "
            "same question set"
        )

    # Sorted so the bootstrap is reproducible: `paired_bootstrap` draws indices
    # in list order, and a differently-ordered id list shifts the bounds.
    ids = sorted(
        q
        for q in verdicts
        if category is None or int(rows[q]["category"]) == category
    )
    if not ids:
        raise SystemExit(
            f"no judged questions left after --category {category}"
        )

    binarise = lambda x: int(float(x) >= threshold)  # noqa: E731
    temporal = [(binarise(rows[q]["score_temporal"]), verdicts[q]) for q in ids]
    token_f1 = [(binarise(rows[q]["score_token_f1"]), verdicts[q]) for q in ids]

    print(f"run    = {run_dir}")
    print(f"judge  = {judge_path} ({model})")
    print(
        f"n      = {len(ids)}"
        + (f", category {category}" if category is not None else "")
        + f", threshold {threshold}"
    )
    print(f"judge marks {sum(verdicts[q] for q in ids)} of {len(ids)} correct")
    print()

    header = f"{'scorer':<12}{'n':>5}{'agree':>9}{'kappa':>9}{'S1J1':>7}{'S1J0':>7}{'S0J1':>7}{'S0J0':>7}"
    print(header)
    print("-" * len(header))
    for name, pairs in (("token-f1", token_f1), ("temporal", temporal)):
        agree = sum(1 for s, j in pairs if s == j) / len(pairs)
        tp, fp, fn, tn = confusion(pairs)
        print(
            f"{name:<12}{len(pairs):>5}{agree * 100:>8.1f}%{cohens_kappa(pairs):>9.4f}"
            f"{tp:>7}{fp:>7}{fn:>7}{tn:>7}"
        )
    print()
    print("S1J1 = scorer says correct and judge agrees; S1J0 = scorer credits an")
    print("answer the judge rejects; S0J1 = scorer denies one the judge accepts.")
    print()

    a = [int(s == j) for s, j in temporal]
    b = [int(s == j) for s, j in token_f1]
    d, lo, hi, p = paired_bootstrap(a, b, iterations=iterations, seed=seed)
    p_str = f"<{2 / iterations:.4f}" if p == 0.0 else f"{p:.4f}"
    sig = "  *  (95% CI excludes zero)" if lo > 0 or hi < 0 else ""
    print(
        f"temporal - token-f1 agreement: {d * 100:+.1f} points, "
        f"95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}], p {p_str}{sig}"
    )
    print()

    differ = [q for q in ids if binarise(rows[q]["score_temporal"]) != binarise(rows[q]["score_token_f1"])]
    print(f"--- {len(differ)} items where the two scorers' binary verdicts differ ---")
    for q in differ:
        row = rows[q]
        print()
        print(f"[{q}]  kind={row['temporal_kind']}  judge={verdicts[q]}")
        print(f"  question  {row['question_text']}")
        print(f"  gold      {row['answer_gold']!r}")
        print(f"  answer    {row['response_raw']!r}")
        print(
            f"  token_f1  {float(row['score_token_f1']):.4f}"
            f"   temporal  {float(row['score_temporal']):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="a run directory rescored by `myelin-eval rescore`")
    parser.add_argument(
        "--judge", required=True, help="path to a judge_verdicts.json written by `myelin-eval judge`"
    )
    parser.add_argument("--category", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="score at or above which a scorer is taken to say `correct` (fixed in advance)",
    )
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report(
        args.run,
        args.judge,
        args.category,
        args.threshold,
        args.iterations,
        args.seed,
    )


if __name__ == "__main__":
    main()
