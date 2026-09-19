"""Multi-judge agreement over the LLM-scored slice of a LongMemEval-V2 run.

`PLAN.md` M9 asks for a judge panel with inter-rater agreement, because a
single LLM judge's verdict is one model's opinion and an accuracy number built
on it inherits that model's quirks. Agreement is how you find out whether the
verdicts are a property of the answers or of the grader.

# Only 35% of the benchmark needs a judge

Measured across our 451-question run:

    200  norm_phrase_set_match          deterministic
    128  llm_abstention_checker         LLM
     68  mc_choice_match                deterministic
     28  llm_gotchas_checker            LLM
     26  norm_phrase_set_match_ordered  deterministic
      1  mc_choice_set_match            deterministic

So 156 of 451 are judged and 295 are string or choice matching. That bounds
how much a judge disagreement can move a headline number — and it means every
abstention verdict is LLM-decided, which is exactly the slice where our
numbers are weakest.

# The judges are the vendored functions, unmodified

This calls `qa_eval_metrics.llm_abstention_checker` and `llm_gotchas_checker`
directly, varying only `evaluator_model` and `evaluator_base_url`. The judge
prompts, parsing and retry behaviour are the harness's own. Reimplementing
them here would measure agreement between my prompt and theirs.

Gemini is reached through its OpenAI-compatible endpoint, which is why no
adapter code is needed.

# Statistic

Fleiss' kappa over binary verdicts, which is the right choice for three fixed
raters scoring the same items — Cohen's handles two. Reported alongside raw
pairwise agreement, because kappa is unstable when one category dominates: if
every judge marks 95% of answers correct, agreement is high and kappa can
still look poor. Both numbers together are readable; either alone is not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "longmemeval-v2"
sys.path.insert(0, str(VENDOR))

from evaluation import qa_eval_metrics  # noqa: E402

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Free-tier Gemini allows 20 generate_content requests per day PER MODEL
# (quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier). A panel over
# the 156 judged questions therefore cannot complete in one sitting on a free
# key, so verdicts are cached on disk and each run judges only what is
# missing. Re-running on successive days accumulates a complete panel; a paid
# key finishes it in one pass. Without the cache every 429 would throw away
# the work already paid for.
#
# The cache lives outside the repo by default (`~/.cache/myelin/`) so a
# checkout is never polluted with verdicts; `MYELIN_JUDGE_CACHE` overrides it.
# Keys are `<model>|<run>|<question_id>`: two runs routinely share question
# ids — a rescore of the same corpus shares all of them — and a key without
# run identity silently blends their verdicts into one kappa.
CACHE_PATH = Path(
    os.environ.get("MYELIN_JUDGE_CACHE", Path.home() / ".cache/myelin/judge_cache.json")
)

LLM_EVAL_FUNCTIONS = {"llm_abstention_checker", "llm_gotchas_checker"}


def load_judged_rows(run_dir: str) -> list[dict[str, Any]]:
    """Rows whose scoring actually went through an LLM judge."""
    rows: list[dict[str, Any]] = []
    with open(Path(run_dir) / "per_question.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["eval_function"].split("|")[0] in LLM_EVAL_FUNCTIONS:
                rows.append(row)
    return rows


def load_cache() -> dict[str, int]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict[str, int]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")


class QuotaExhausted(Exception):
    """The daily per-model quota is gone. Stop cleanly, keep what we have."""


def judge_one(row: dict[str, Any], model: str, base_url: str, api_key: str) -> int:
    """One judge's binary verdict on one answer. Errors surface, never score 0.

    A judge error silently scored as "wrong" would look like disagreement and
    quietly deflate both the accuracy and the kappa.
    """
    name = row["eval_function"].split("|")[0]
    fn = getattr(qa_eval_metrics, name)
    verdict = fn(
        row.get("response_parsed_boxed") or row.get("response_raw"),
        row["answer_gold"],
        question_item={"question": row["question_text"]},
        parsed_prediction=row.get("response_parsed_boxed"),
        model_response=row.get("response_raw"),
        evaluator_model=model,
        evaluator_base_url=base_url,
        evaluator_api_key=api_key,
        evaluator_max_completion_tokens=2048,
    )
    return int(bool(verdict))


def fleiss_kappa(table: list[list[int]]) -> float:
    """Fleiss' kappa. `table[i][c]` = number of raters assigning item i to category c."""
    n_items = len(table)
    if n_items == 0:
        raise ValueError("no items")
    n_raters = sum(table[0])
    if n_raters < 2:
        raise ValueError("need at least two raters")

    # Per-item agreement.
    p_i = [
        (sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in table
    ]
    p_bar = sum(p_i) / n_items

    # Chance agreement from marginal category frequencies.
    n_cats = len(table[0])
    p_j = [
        sum(table[i][j] for i in range(n_items)) / (n_items * n_raters)
        for j in range(n_cats)
    ]
    p_e = sum(p * p for p in p_j)

    if abs(1.0 - p_e) < 1e-12:
        # Every rater chose one category for everything: agreement is total
        # and chance agreement is also total, so kappa is undefined rather
        # than perfect. Saying 1.0 here would be a lie.
        return float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def pairwise_agreement(verdicts: dict[str, list[int]]) -> list[tuple[str, str, float]]:
    names = sorted(verdicts)
    out: list[tuple[str, str, float]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            va, vb = verdicts[a], verdicts[b]
            same = sum(1 for x, y in zip(va, vb) if x == y)
            out.append((a, b, same / len(va)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument(
        "--gemini-models",
        nargs="*",
        default=["gemini-3.1-pro-preview", "gemini-2.5-pro"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="seconds between judge calls; free-tier also caps requests per minute",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set")

    rows = load_judged_rows(args.run_dir)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} LLM-judged questions in {args.run_dir}")

    # Judge 1 is the local Qwen3.5-9B that scored the run. Its verdicts are
    # already on disk; re-asking it would cost a GPU window to reproduce a
    # number we already have.
    verdicts: dict[str, list[int]] = {
        "qwen3.5-9b (local, original)": [int(r["score"] >= 0.5) for r in rows]
    }

    cache = load_cache()
    # The run directory's own name is the run identity. Two runs over the same
    # corpus carry identical question ids, so without it a second run reads the
    # first one's verdicts back as its own.
    run_key = Path(args.run_dir).resolve().name
    for model in args.gemini_models:
        todo = [r for r in rows if f"{model}|{run_key}|{r['question_id']}" not in cache]
        have = len(rows) - len(todo)
        print(f"  {model}: {have} cached, {len(todo)} to judge", flush=True)

        exhausted = False
        for row in todo:
            if exhausted:
                break
            try:
                cache[f"{model}|{run_key}|{row['question_id']}"] = judge_one(
                    row, model, GEMINI_BASE_URL, key
                )
            except Exception as exc:  # noqa: BLE001 - want the message, not the type
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    print(
                        f"    quota exhausted after {len(rows) - len(todo) + todo.index(row)}"
                        f" of {len(rows)}; keeping partial results",
                        flush=True,
                    )
                    exhausted = True
                    break
                raise
            save_cache(cache)
            time.sleep(args.sleep)
        save_cache(cache)

        got = [cache.get(f"{model}|{run_key}|{r['question_id']}") for r in rows]
        verdicts[model] = got

    # Only items every judge has scored can enter the panel.
    complete = [
        i
        for i in range(len(rows))
        if all(verdicts[n][i] is not None for n in verdicts)
    ]
    if not complete:
        raise SystemExit(
            "no question has a verdict from every judge yet; re-run tomorrow "
            "to accumulate more free-tier quota, or use a paid key"
        )
    if len(complete) < len(rows):
        print(
            f"\n  panel restricted to {len(complete)} of {len(rows)} questions "
            f"that every judge has scored"
        )
    rows = [rows[i] for i in complete]
    verdicts = {n: [v[i] for i in complete] for n, v in verdicts.items()}

    print()
    names = list(verdicts)
    print(f"{'judge':<32}{'marked correct':>16}")
    print("-" * 48)
    for n in names:
        v = verdicts[n]
        print(f"{n:<32}{sum(v) / len(v) * 100:>15.1f}%")

    print()
    print(f"{'pair':<58}{'raw agreement':>15}")
    print("-" * 73)
    for a, b, agree in pairwise_agreement(verdicts):
        print(f"{a + '  vs  ' + b:<58}{agree * 100:>14.1f}%")

    table = [
        [sum(1 for n in names if verdicts[n][i] == 0), sum(1 for n in names if verdicts[n][i] == 1)]
        for i in range(len(rows))
    ]
    kappa = fleiss_kappa(table)
    print()
    print(f"Fleiss' kappa over {len(names)} judges, {len(rows)} items: {kappa:.4f}")
    print("PLAN.md M9 gate: kappa >= 0.89 -> " + ("PASS" if kappa >= 0.89 else "FAIL"))

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "run_dir": args.run_dir,
                    "items": len(rows),
                    "judges": names,
                    "marked_correct": {n: sum(verdicts[n]) / len(rows) for n in names},
                    "pairwise_agreement": [
                        {"a": a, "b": b, "agreement": g}
                        for a, b, g in pairwise_agreement(verdicts)
                    ],
                    "fleiss_kappa": kappa,
                    "verdicts": verdicts,
                    "question_ids": [r["question_id"] for r in rows],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
