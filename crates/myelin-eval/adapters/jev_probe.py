#!/usr/bin/env python3
"""M56 Step 0: replay existing answers through Jev and simulate the verify policy.

An external checker decides whether an answer is supported by the memories
the reader saw, and whether the question rests on a premise the memories do
not state. Every gate this project tried before was decided by the reader or
the cross-encoder and failed (M6-M47; M45: "the discriminator has to be
external"). Prior art for an external evaluator:

* CRAG (Yan et al., 2024, arXiv:2401.15884): a separate lightweight retrieval
  evaluator gates what generation may use.
* Self-RAG (Asai et al., 2023, arXiv:2310.11511): `IsSup`, whether the output
  is supported by the retrieved passages.
* "Don't Hallucinate, Abstain" (Feng et al., ACL 2024,
  10.18653/v1/2024.acl-long.786): abstention decided by other models beats
  self-evaluation.

The checker is TypeSafe's Jev (a "System One" decision model) through
OpenRouter's Decisions API. The pre-registration, policy and gate are in
`docs/measurements/m56-verified-answers.md`; this file implements exactly
that and nothing tuned after seeing a verdict.

No reader or judge is called. Scores are rebuilt from each run's existing
verdicts: a kept or adopted answer keeps its verdict; a decline scores 0 on an
answerable row and 1 on an abstention / adversarial row (the rule
`bench::is_abstention` applies, mirrored in `is_decline`).

Usage (from crates/myelin-eval):

    ../../.venv/bin/python adapters/jev_probe.py --out ../../runs/m56_jev_probe
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import re
import string
import sys
import threading
import time

import httpx

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"
KEY_PATH = Path("~/.config/myelin/openrouter.key").expanduser()
# The one accepted key-file format (decided 2026-09-24): a single line.
KEY_LINE = re.compile(r"^OPENROUTER_API_KEY=(\S+)$")
IN_FLIGHT = 8
REQUEST_TIMEOUT_S = 60.0
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0
# Retry only what a later attempt can fix: rate limits and server errors.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
PREMISE_THRESHOLD = 0.5
DECLINE = "I don't know."
LOCOMO_ANSWERABLE = (1, 2, 3, 4)
LOCOMO_ADVERSARIAL = 5

SUPPORT_QUESTION = {
    "type": "choice",
    "instructions": "Compare the answer with the memories.",
    "criteria": {
        "supported": "the answer addresses the question and every fact in it appears in the memories",
        "unsupported": "the answer states something the memories do not contain or contradict, "
        "or answers a different question",
        "declined": "the answer says the memories do not cover the question and adds no facts of its own",
    },
}
# Declarative on purpose: Laya's noul head returns 0 for every interrogative
# instruction (big_mac bakeoff, 2026-09-21), and Jev is the same family of API.
PREMISE_QUESTION = {
    "type": "noul",
    "instructions": "The question takes for granted something the memories do not say.",
}

ARTICLES = {"a", "an", "the"}
PUNCT = set(string.punctuation)  # == Rust's `char::is_ascii_punctuation`
DECLINE_EXACT = {"i dont know", "i don t know", "unknown", "no information"}
DECLINE_PREFIXES = (
    "i dont know",
    "i don t know",
    "i cannot determine",
    "i can t determine",
    "there is no information",
    "no information",
)


def normalize(text: str) -> str:
    """`bench::normalize`, joined: lowercase, ASCII punctuation to spaces, articles dropped."""
    spaced = "".join(" " if c in PUNCT else c for c in text.lower())
    return " ".join(w for w in spaced.split() if w not in ARTICLES)


def is_decline(text: str) -> bool:
    """`bench::is_abstention`, verbatim in behaviour."""
    n = normalize(text)
    return not n or n in DECLINE_EXACT or n.startswith(DECLINE_PREFIXES)


def read_key() -> str:
    if not KEY_PATH.exists():
        raise SystemExit(f"{KEY_PATH} does not exist; it must hold one line OPENROUTER_API_KEY=<key>")
    lines = [line for line in KEY_PATH.read_text(encoding="utf-8").split("\n") if line.strip()]
    if len(lines) != 1 or not KEY_LINE.match(lines[0].strip()):
        raise SystemExit(f"{KEY_PATH} must hold exactly one line OPENROUTER_API_KEY=<key>; refusing to guess")
    return KEY_LINE.match(lines[0].strip()).group(1)


class Jev:
    """One Decisions request per (memories, question, answer), cached on disk by content."""

    def __init__(self, cache_path: Path):
        self.key = read_key()
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT_S)
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for row in jsonl(cache_path):
                self.cache[row["key"]] = row["verdict"]
        self.lock = threading.Lock()
        self.cost = 0.0
        self.calls = 0

    @staticmethod
    def cache_key(state: dict) -> str:
        blob = json.dumps({"model": JEV_MODEL, "state": state, "q": [SUPPORT_QUESTION, PREMISE_QUESTION]}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def verdict(self, memories: list[str], question: str, answer: str) -> dict:
        state = {"memories": memories, "question": question, "answer": answer}
        key = self.cache_key(state)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        body = {"model": JEV_MODEL, "state": state,
                "questions": {"support": SUPPORT_QUESTION, "false_premise": PREMISE_QUESTION}}
        last = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                r = self.client.post(DECISIONS_URL, json=body, headers={"Authorization": f"Bearer {self.key}"})
            except httpx.HTTPError as exc:
                last = f"transport: {exc}"
            else:
                if r.status_code == 200:
                    payload = r.json()
                    answers = payload["answers"]
                    verdict = {
                        "model": payload["model"],
                        "support": answers["support"]["choice"],
                        "support_probabilities": answers["support"]["probabilities"],
                        "false_premise": answers["false_premise"]["noul"],
                        "cost": payload.get("usage", {}).get("cost", 0.0),
                    }
                    with self.lock:
                        self.cache[key] = verdict
                        self.cost += verdict["cost"]
                        self.calls += 1
                        with self.cache_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"key": key, "verdict": verdict}) + "\n")
                    return verdict
                if r.status_code not in RETRYABLE_STATUS:
                    raise RuntimeError(f"Jev refused (HTTP {r.status_code}): {r.text[:300]}")
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(BACKOFF_BASE_S * (2**attempt))
        raise RuntimeError(f"Jev failed after {MAX_ATTEMPTS} attempts: {last}")


def policy(verdict: dict) -> str:
    """The pre-registered steps 1-3: 'decline' or 'keep'."""
    if verdict["false_premise"] > PREMISE_THRESHOLD:
        return "decline"
    if verdict["support"] == "unsupported":
        return "decline"
    return "keep"


def adopt(verdict: dict) -> bool:
    """Step 4: a stand-in answer for a declined row is adopted only if clean and supported."""
    return verdict["support"] == "supported" and verdict["false_premise"] <= PREMISE_THRESHOLD


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney U / (n_pos * n_neg), ties counted half."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def jsonl(path: Path) -> list[dict]:
    """One JSON object per "\n"-terminated line. Not `splitlines()`: it also
    splits on U+2028/U+2029/U+0085, which occur inside the stored evidence."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def load_rows(run: Path) -> dict[str, dict]:
    return {r["question_id"]: r for r in jsonl(run / "per_question.jsonl")}


def run_parallel(jev: Jev, jobs: list[tuple]) -> dict:
    """jobs: (key, memories, question, answer) -> {key: verdict}."""
    out = {}
    with cf.ThreadPoolExecutor(IN_FLIGHT) as pool:
        futures = {pool.submit(jev.verdict, m, q, a): k for k, m, q, a in jobs}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            out[futures[fut]] = fut.result()
            if i % 200 == 0:
                print(f"    {i}/{len(jobs)} verdicts, ${jev.cost:.4f} so far", flush=True)
    return out


def longmemeval(jev: Jev, judged_run: Path) -> dict:
    rows = load_rows(judged_run)
    base = 100 * sum(r["score"] for r in rows.values()) / len(rows)
    jobs = [(q, r["evidence"], r["question_text"], r["response_raw"])
            for q, r in rows.items() if not is_decline(r["response_raw"])]
    verdicts = run_parallel(jev, jobs)
    score, declines, flips_lost, flips_gained = 0.0, 0, 0, 0
    abst_ok = abst_n = 0
    probs, labels = [], []
    for q, r in rows.items():
        base_score = r["score"]
        if q in verdicts:
            v = verdicts[q]
            if not r["is_abstention_problem"]:
                probs.append(v["support_probabilities"].get("supported", 0.0))
                labels.append(int(base_score >= 1.0))
            action = policy(v)
        else:
            action = "decline"  # the reader already declined
        s = (1.0 if r["is_abstention_problem"] else 0.0) if action == "decline" else base_score
        declines += action == "decline" and not r["is_abstention_problem"]
        flips_lost += s < base_score
        flips_gained += s > base_score
        if r["is_abstention_problem"]:
            abst_n += 1
            abst_ok += s >= 1.0
        score += s
    return {
        "run": str(judged_run), "rows": len(rows), "base": round(base, 2),
        "simulated": round(100 * score / len(rows), 2),
        "abstention": f"{abst_ok}/{abst_n}",
        "base_abstention": f"{sum(1 for r in rows.values() if r['is_abstention_problem'] and r['score'] >= 1.0)}/{abst_n}",
        "answerable_declines": declines,
        "rows_lost": flips_lost, "rows_gained": flips_gained,
        "jev_calls": len(jobs),
        "auroc_supported_vs_judged_correct": round(auroc(probs, labels) or float("nan"), 3),
    }


def locomo(jev: Jev, arm_run: Path, proxy_run: Path) -> dict:
    arm, proxy = load_rows(arm_run), load_rows(proxy_run)
    arm_v = json.loads((arm_run / "judge_verdicts.json").read_text())["verdicts"]
    proxy_v = json.loads((proxy_run / "judge_verdicts.json").read_text())["verdicts"]

    def judged(verdicts, q):
        return float(verdicts.get(q) == 1)

    jobs, proxy_jobs = [], []
    for q, r in arm.items():
        if not is_decline(r["response_raw"]):
            jobs.append((q, r["evidence"], r["question_text"], r["response_raw"]))
        elif q in proxy and not is_decline(proxy[q]["response_raw"]):
            proxy_jobs.append((q, r["evidence"], r["question_text"], proxy[q]["response_raw"]))
    verdicts = run_parallel(jev, jobs)
    proxy_verdicts = run_parallel(jev, proxy_jobs)

    ans_correct = adv_ok = 0
    n_ans = n_adv = declines = adopted = 0
    base_correct = base_adv = 0
    probs, labels = [], []
    for q, r in arm.items():
        cat = r["category"]
        answerable = cat in LOCOMO_ANSWERABLE
        if answerable:
            n_ans += 1
            base_correct += judged(arm_v, q) if not is_decline(r["response_raw"]) else 0.0
        elif cat == LOCOMO_ADVERSARIAL:
            n_adv += 1
            base_adv += r["score"]
        if q in verdicts:
            v = verdicts[q]
            if answerable:
                probs.append(v["support_probabilities"].get("supported", 0.0))
                labels.append(int(judged(arm_v, q)))
            if policy(v) == "keep":
                ans_correct += judged(arm_v, q) if answerable else 0.0
                continue  # adversarial answered -> wrong
            declines += answerable
            adv_ok += cat == LOCOMO_ADVERSARIAL
            continue
        if q in proxy_verdicts and adopt(proxy_verdicts[q]):
            adopted += 1
            ans_correct += judged(proxy_v, q) if answerable else 0.0
            continue  # adversarial answered by the stand-in -> wrong
        declines += answerable
        adv_ok += cat == LOCOMO_ADVERSARIAL
    return {
        "arm_run": str(arm_run), "proxy_run": str(proxy_run),
        "base_judge_1_4": round(100 * base_correct / n_ans, 2),
        "simulated_judge_1_4": round(100 * ans_correct / n_ans, 2),
        "base_adversarial": round(100 * base_adv / n_adv, 2),
        "simulated_adversarial": round(100 * adv_ok / n_adv, 2),
        "answerable_declines": declines, "adopted_stand_ins": adopted,
        "jev_calls": len(jobs) + len(proxy_jobs),
        "auroc_supported_vs_judged_correct": round(auroc(probs, labels) or float("nan"), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--runs-root", default="../../runs")
    p.add_argument("--smoke", type=int, default=0, help="Verify N rows of the Bonsai LongMemEval_S run and stop.")
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.runs_root).resolve()
    jev = Jev(out / "jev_verdicts.jsonl")
    if args.smoke:
        rows = list(load_rows(root / "m55_bonsai_s1_judged").values())[: args.smoke]
        for r in rows:
            v = jev.verdict(r["evidence"], r["question_text"], r["response_raw"])
            print(json.dumps({"q": r["question_id"], **{k: v[k] for k in ("model", "support", "false_premise", "cost")}}))
        print(f"smoke: {jev.calls} new calls, ${jev.cost:.6f}")
        return
    summary = {"jev_model": JEV_MODEL, "policy": "docs/measurements/m56-verified-answers.md Step 0"}
    for name, run in (("longmemeval_s_bonsai", "m55_bonsai_s1_judged"), ("longmemeval_s_9b", "m44_r2_s1_judged")):
        print(f"== {name}", flush=True)
        summary[name] = longmemeval(jev, root / run)
        print(json.dumps(summary[name], indent=1), flush=True)
    print("== locomo_bonsai", flush=True)
    summary["locomo_bonsai"] = locomo(jev, root / "m55b_locomo_bonsai", root / "m19_locomo_full")
    print(json.dumps(summary["locomo_bonsai"], indent=1), flush=True)
    summary["new_calls"] = jev.calls
    summary["new_cost_usd"] = round(jev.cost, 4)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {out / 'summary.json'}; {jev.calls} new calls, ${jev.cost:.4f}")


if __name__ == "__main__":
    sys.exit(main())
