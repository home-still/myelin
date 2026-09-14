#!/usr/bin/env python3
"""Reproducible LAFS analysis for docs/EVALUATION.md §3.

LAFS is LongMemEval-V2's official accuracy-latency composite. This file reimplements the released
formula exactly as published in `leaderboard/compute_lafs.py` of
https://github.com/xiaowu0162/LongMemEval-V2 (Apache-2.0), *only* so that the reference frontier value
and the sensitivity tables in the evaluation spec are re-derivable without a checkout.

For any actual submission we call the authors' code, never this file.

Verified output (2026-09-14):

    small: reference LAFS = 55.765   frontier 51@0.2s, 58.6@26.9s, 74.9@108.3s   (Codex dominated)
    medium: reference LAFS = 51.074  frontier 45.9@0.3s, 57@25.8s, 70.1@139.9s   (Codex dominated)

    fast point (recall), tier small, LAFS gain:
      latency 0.3/0.5/1.0s identical -> acc 50: +0.00  55: +2.49  60: +5.96  65: +10.38  70: +14.80
      latency 2.0s                   -> acc 55: +1.96  60: +4.78  65: +8.55   70: +12.32
      latency 5.0s                   -> acc 55: +1.27  60: +3.23  65: +6.13   70: +9.03

    slow point (investigate), tier small, LAFS gain:
      10s  -> 60: +2.05  70: +6.55  76: +9.37  80: +11.63  85: +14.46
      20s  -> 60: +0.87  70: +4.06  76: +6.10  80: +7.84   85: +10.01
      30s  -> 60: +0.34  70: +2.76  76: +4.34  80: +5.78   85: +7.57
      60s  -> 60: +0.16  70: +1.27  76: +2.07  80: +2.98   85: +4.11
      110s -> 60: +0.00  70: +0.00  76: +0.12  80: +0.58   85: +1.14

    combined, tier small:
      recall 60@0.5                            +5.96  (abs 61.73)
      investigate 76@60                        +2.07  (abs 57.83)
      recall 60@0.5 + investigate 76@60        +7.87  (abs 63.64)
      recall 65@0.5 + investigate 80@40       +13.79  (abs 69.56)
      recall 55@0.5 + mid 70@20 + inv 78@90    +6.96  (abs 62.72)

    break-even for a sub-1s fast point:
      small: accuracy > 51.1   (reference floor 51.0)
      medium: accuracy > 46.0  (reference floor 45.9)
"""

import math
from dataclasses import dataclass
from typing import List

T_MIN, T_MAX, FLOOR = 1.0, 200.0, 0.0


@dataclass(frozen=True)
class Point:
    name: str
    acc: float      # accuracy in percentage points
    latency: float  # memory_query_avg_seconds


# Released reference frontier, hard-coded in the official tool.
REF = {
    "small": [
        Point("RAG slice+notes", 51.0, 0.2),
        Point("Codex", 69.9, 177.2),
        Point("AgentRunbook-R", 58.6, 26.9),
        Point("AgentRunbook-C", 74.9, 108.3),
    ],
    "medium": [
        Point("RAG slice+notes", 45.9, 0.3),
        Point("Codex", 68.7, 185.8),
        Point("AgentRunbook-R", 57.0, 25.8),
        Point("AgentRunbook-C", 70.1, 139.9),
    ],
}


def pareto(points: List[Point]) -> List[Point]:
    out: List[Point] = []
    for p in sorted(points, key=lambda p: (p.latency, -p.acc)):
        if not out or p.acc > out[-1].acc:
            out.append(p)
    return out


def best_acc(points: List[Point], budget: float, floor: float = FLOOR) -> float:
    valid = [p.acc for p in points if p.latency <= budget]
    return max(valid) if valid else floor


def lafs(points: List[Point], t_min: float = T_MIN, t_max: float = T_MAX,
         floor: float = FLOOR) -> float:
    """Mean best-accuracy-under-budget over a log-uniform latency budget."""
    frontier = pareto(points)
    bps = sorted({t_min, t_max} | {p.latency for p in frontier if t_min < p.latency < t_max})
    denom = math.log(t_max / t_min)
    area = sum(best_acc(frontier, l, floor) * math.log(r / l) for l, r in zip(bps[:-1], bps[1:]))
    return area / denom


def gain(tier: str, pts: List[Point]) -> float:
    return lafs(REF[tier] + pts) - lafs(REF[tier])


def main() -> None:
    for tier in ("small", "medium"):
        fr = pareto(REF[tier])
        print(f"{tier}: reference LAFS = {lafs(REF[tier]):.3f}")
        print("  frontier:", ", ".join(f"{p.acc:g}@{p.latency:g}s ({p.name})" for p in fr))
        print("  dominated:", ", ".join(p.name for p in REF[tier] if p not in fr) or "none")

    print("\nfast point (recall), tier=small — LAFS gain")
    print(f"{'lat(s)':>7} " + " ".join(f"{'acc='+str(a):>9}" for a in (50, 55, 60, 65, 70)))
    for lat in (0.3, 0.5, 1.0, 2.0, 5.0):
        print(f"{lat:>7} " + " ".join(
            f"{gain('small', [Point('r', a, lat)]):>+9.2f}" for a in (50, 55, 60, 65, 70)))

    print("\nslow point (investigate), tier=small — LAFS gain")
    print(f"{'lat(s)':>7} " + " ".join(f"{'acc='+str(a):>9}" for a in (60, 70, 76, 80, 85)))
    for lat in (10, 20, 30, 60, 110):
        print(f"{lat:>7} " + " ".join(
            f"{gain('small', [Point('i', a, lat)]):>+9.2f}" for a in (60, 70, 76, 80, 85)))

    print("\ncombined submissions, tier=small")
    for label, pts in [
        ("recall 60@0.5 only", [Point("r", 60, 0.5)]),
        ("investigate 76@60 only", [Point("i", 76, 60)]),
        ("recall 60@0.5 + inv 76@60", [Point("r", 60, 0.5), Point("i", 76, 60)]),
        ("recall 65@0.5 + inv 80@40", [Point("r", 65, 0.5), Point("i", 80, 40)]),
        ("recall 55@0.5 + mid 70@20 + inv 78@90",
         [Point("r", 55, 0.5), Point("m", 70, 20), Point("i", 78, 90)]),
    ]:
        print(f"  {label:38s} gain {gain('small', pts):+.2f}  abs {lafs(REF['small'] + pts):.2f}")

    print("\nbreak-even accuracy for a sub-1s fast point")
    for tier in ("small", "medium"):
        floor = pareto(REF[tier])[0].acc
        be = next(a / 10 for a in range(300, 900) if gain(tier, [Point("r", a / 10, 0.5)]) > 1e-9)
        print(f"  {tier}: > {be:.1f} (reference floor {floor:g})")


if __name__ == "__main__":
    main()
