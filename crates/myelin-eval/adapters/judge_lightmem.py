#!/usr/bin/env python3
"""M68: grade a LoCoMo run under LightMem's judge protocol, for a matched comparison.

MemPro's LoCoMo numbers, including the same-size Qwen3-30B-A3B row we chase
(77.85), were graded by gpt-4o-mini with LightMem's LoCoMo judge prompt
(Liu et al. 2026, "MemPro", arXiv:2606.00619, L122 and Fig. 10). Our headline
uses a strict local 9B rubric (`src/judge.rs`), so every LoCoMo comparison
with MemPro carried a caveat-judge verdict. This adapter reproduces the other
side's grader and nothing else, so the two numbers can be compared directly:

* the prompt is LightMem's `ACCURACY_PROMPT`, copied byte for byte from
  zjunlp/LightMem at commit LIGHTMEM_COMMIT, `experiments/locomo/llm_judge.py`
  (Fig. 10 of MemPro prints it without the "First, provide a short (one
  sentence) explanation" line; the code is what ran, so the code is used);
* one user message, `temperature=0.0`, `response_format={"type": "json_object"}`,
  model gpt-4o-mini, label parsed with LightMem's `extract_json`, and
  verdict 1 iff the label is exactly "CORRECT" (their `evaluate_llm_judge`);
* every category 1-4 row is judged, declines included, and category 5 is
  skipped, as their `main` does.

The strict verdicts in `judge_verdicts.json` are never read or written. The
output is `<run>/judge_verdicts_lightmem.json`, the same shape as
`judge::JudgeFile` plus the protocol that produced it. A cache or seed
written under a different protocol is refused, never mixed. The OpenRouter
key comes from ~/.config/myelin/openrouter.key and is never printed.

Usage (from crates/myelin-eval):

    ../../.venv/bin/python adapters/judge_lightmem.py --run ../../runs/m63_locomo_base
    ../../.venv/bin/python adapters/judge_lightmem.py --run ../../runs/m19_locomo_full \
        --seed ../../runs/m63_locomo_base
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

import httpx

OUT_NAME = "judge_verdicts_lightmem.json"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "openai/gpt-4o-mini"
# MemPro and LightMem called OpenAI's own API; pin that provider on OpenRouter
# rather than let it route the judge to another host of the same weights.
PROVIDER = {"order": ["OpenAI"], "allow_fallbacks": False}
TEMPERATURE = 0.0
RESPONSE_FORMAT = {"type": "json_object"}
LIGHTMEM_REPO = "zjunlp/LightMem"
LIGHTMEM_COMMIT = "8449d574df6bae1bdf3314a1564da65e2f37e046"
LIGHTMEM_FILE = "experiments/locomo/llm_judge.py"
KEY_PATH = Path("~/.config/myelin/openrouter.key").expanduser()
# The one accepted key-file format (decided 2026-09-24): a single line.
KEY_LINE = re.compile(r"^OPENROUTER_API_KEY=(\S+)$")
IN_FLIGHT = 8
REQUEST_TIMEOUT_S = 60.0
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0
# Retry only what a later attempt can fix: rate limits, server errors, and a
# reply the protocol's own parser cannot read (their script would crash).
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
JUDGED_CATEGORIES = (1, 2, 3, 4)
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}

# Verbatim from LIGHTMEM_REPO@LIGHTMEM_COMMIT:LIGHTMEM_FILE, including its
# typographic quotes and trailing spaces. Do not edit; the sha is recorded.
ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def protocol() -> dict:
    return {
        "name": "lightmem-locomo",
        "source": f"{LIGHTMEM_REPO}@{LIGHTMEM_COMMIT}:{LIGHTMEM_FILE}",
        "prompt_sha256": hashlib.sha256(ACCURACY_PROMPT.encode("utf-8")).hexdigest(),
        "model": JUDGE_MODEL,
        "provider": PROVIDER,
        "temperature": TEMPERATURE,
        "response_format": RESPONSE_FORMAT,
        "categories": list(JUDGED_CATEGORIES),
    }


def extract_json(text: str) -> str:
    """LightMem's `extract_json`, verbatim in behaviour."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def read_key() -> str:
    if not KEY_PATH.exists():
        raise SystemExit(f"{KEY_PATH} does not exist; it must hold one line OPENROUTER_API_KEY=<key>")
    lines = [line for line in KEY_PATH.read_text(encoding="utf-8").split("\n") if line.strip()]
    if len(lines) != 1 or not KEY_LINE.match(lines[0].strip()):
        raise SystemExit(f"{KEY_PATH} must hold exactly one line OPENROUTER_API_KEY=<key>; refusing to guess")
    return KEY_LINE.match(lines[0].strip()).group(1)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_verdict_file(path: Path, what: str) -> dict | None:
    """A verdict file written under exactly this protocol, or a refusal."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol") != protocol():
        raise SystemExit(
            f"{path} ({what}) was written under protocol {data.get('protocol')!r}, not this one; "
            "mixing two graders' verdicts is refused"
        )
    return data


class Judge:
    def __init__(self) -> None:
        self.key = read_key()
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT_S)
        self.lock = threading.Lock()
        self.served_models: set[str] = set()
        self.providers: set[str] = set()
        self.cost = 0.0

    def grade(self, question: str, gold: str, answer: str) -> int:
        body = {
            "model": JUDGE_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": ACCURACY_PROMPT.format(
                        question=question, gold_answer=gold, generated_answer=answer
                    ),
                }
            ],
            "response_format": RESPONSE_FORMAT,
            "temperature": TEMPERATURE,
            "provider": PROVIDER,
            "usage": {"include": True},
        }
        last = ""
        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                time.sleep(BACKOFF_BASE_S * 2 ** (attempt - 1))
            try:
                r = self.client.post(CHAT_URL, json=body, headers={"Authorization": f"Bearer {self.key}"})
            except httpx.HTTPError as e:
                last = f"transport: {type(e).__name__}"
                continue
            if r.status_code in RETRYABLE_STATUS:
                last = f"HTTP {r.status_code}"
                continue
            if r.status_code != 200:
                raise SystemExit(f"judge request refused: HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            try:
                content = data["choices"][0]["message"]["content"]
                label = json.loads(extract_json(content))["label"]
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
                last = f"unparseable reply ({type(e).__name__})"
                continue
            with self.lock:
                self.served_models.add(str(data.get("model", "")))
                self.providers.add(str(data.get("provider", "")))
                self.cost += float((data.get("usage") or {}).get("cost") or 0.0)
            return 1 if label == "CORRECT" else 0
        raise SystemExit(f"judge failed after {MAX_ATTEMPTS} attempts: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="a LoCoMo bench run directory")
    ap.add_argument(
        "--seed",
        type=Path,
        action="append",
        default=[],
        help="another run whose verdicts are reused for byte-identical (question, answer) pairs",
    )
    args = ap.parse_args()

    agg = json.loads((args.run / "aggregated_metrics.json").read_text(encoding="utf-8"))
    if agg.get("corpus") != "locomo":
        raise SystemExit(f"{args.run} is corpus {agg.get('corpus')!r}; this protocol is LoCoMo's")
    rows = [r for r in jsonl(args.run / "per_question.jsonl") if int(r["category"]) in JUDGED_CATEGORIES]

    out_path = args.run / OUT_NAME
    existing = load_verdict_file(out_path, "cache")
    verdicts: dict[str, int] = {}
    answers: dict[str, str] = {}
    known: dict[tuple[str, str], int] = {}
    if existing:
        for qid, v in existing["verdicts"].items():
            known[(qid, existing["answers"][qid])] = v
    for seed in args.seed:
        seeded = load_verdict_file(seed / OUT_NAME, f"seed {seed}")
        if seeded is None:
            raise SystemExit(f"seed {seed} has no {OUT_NAME}")
        for qid, v in seeded["verdicts"].items():
            known.setdefault((qid, seeded["answers"][qid]), v)

    todo = []
    for r in rows:
        key = (r["question_id"], r["response_raw"])
        if key in known:
            verdicts[r["question_id"]] = known[key]
            answers[r["question_id"]] = r["response_raw"]
        else:
            todo.append(r)
    reused = len(verdicts)

    judge = Judge()
    done = 0

    def one(r: dict) -> tuple[str, str, int]:
        return r["question_id"], r["response_raw"], judge.grade(
            r["question_text"], str(r["answer_gold"]), r["response_raw"]
        )

    with cf.ThreadPoolExecutor(max_workers=IN_FLIGHT) as pool:
        for qid, answer, v in pool.map(one, todo):
            verdicts[qid] = v
            answers[qid] = answer
            done += 1
            if done % 200 == 0:
                print(f"  judged {done}/{len(todo)}", file=sys.stderr)

    file = {
        "model": JUDGE_MODEL,
        "protocol": protocol(),
        "served": {"models": sorted(judge.served_models), "providers": sorted(judge.providers)},
        "verdicts": dict(sorted(verdicts.items())),
        "answers": dict(sorted(answers.items())),
    }
    out_path.write_text(json.dumps(file, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    by_cat: dict[int, list[int]] = {c: [] for c in JUDGED_CATEGORIES}
    for r in rows:
        by_cat[int(r["category"])].append(verdicts[r["question_id"]])
    total = [v for vs in by_cat.values() for v in vs]
    print(f"{args.run.name}: LightMem protocol, {JUDGE_MODEL}: {sum(total)}/{len(total)} = "
          f"{100 * sum(total) / len(total):.2f} (judged {len(todo)}, reused {reused}, cost ${judge.cost:.4f})")
    for c, vs in by_cat.items():
        print(f"  cat {c} {CATEGORY_NAMES[c]:<12} n={len(vs):<4} {100 * sum(vs) / len(vs):6.2f}")
    print(f"  served models={sorted(judge.served_models)} providers={sorted(judge.providers)}")


if __name__ == "__main__":
    main()
