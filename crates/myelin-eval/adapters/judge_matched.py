#!/usr/bin/env python3
"""Matched judges: grade a run the way the row it is compared with was graded (M68, M68b, M70).

Our headline numbers are graded by a strict local 9B rubric (`src/judge.rs`).
The same-size rows we compare against were graded by gpt-4o-mini under other
prompts, so every such comparison carried a caveat-judge verdict. This adapter
reproduces the other side's grader, protocol by protocol, and nothing else:

* `lightmem-locomo` (M68). LightMem's `ACCURACY_PROMPT`, byte for byte from
  `zjunlp/LightMem@8449d574df6bae1bdf3314a1564da65e2f37e046:experiments/locomo/llm_judge.py`,
  the prompt MemPro's paper cites and prints (Liu et al. 2026, arXiv:2606.00619,
  L122 and Fig. 10). One user message, `response_format=json_object`, the
  label parsed with LightMem's `extract_json`; correct iff it is "CORRECT".
* `mempro-locomo` (M68b). MemPro's own repo grades LoCoMo with a paraphrase of
  that prompt: `wanghai673/MemPro@834b1cef87c80f3a8dbb2bcd398a6c7d6104d380:eval/locomo_test.py`
  (`JUDGE_PROMPT_TEMPLATE`, `JUDGE_SCHEMA`, `call_llm_judge`), with strict
  json_schema output. Correct iff the label, stripped and upper-cased, is
  "CORRECT" (its `llm_judge_score`).
* `longmemeval-official` (M70). LongMemEval's own grader, byte for byte from
  `xiaowu0162/LongMemEval@9e0b455f4ef0e2ab8f2e582289761153549043fc:src/evaluation/evaluate_qa.py`
  (`get_anscheck_prompt`, Wu et al. 2024, arXiv:2410.10813): five templates, one
  per task type, and one for abstention routed by `_abs` in the question id.
  `max_tokens=10`, correct iff "yes" is in the lower-cased reply. The model is
  its `gpt-4o-mini` entry, the `gpt-4o-mini-2024-07-18` snapshot. MemPro says
  its LongMemEval judge "follows GAM", which never published one; its Fig. 11
  is a hand-merged paraphrase with no abstention path, and its repo grades
  research summaries rather than answers. Neither is a runnable reading of how
  answers were graded, so the benchmark's own code is the one used.

Every protocol uses `temperature=0.0` and is pinned to OpenAI's own provider on
OpenRouter (no fallback host). LoCoMo protocols judge every category 1-4 row,
declines included, and skip category 5, as both sources do. The LongMemEval
protocol judges all 500 rows, declines included.

A failed or unparseable call is retried and then refused: MemPro's code scores
a judge failure as WRONG, which would silently mix a transport error into the
number, so this adapter stops instead.

The strict verdicts in `judge_verdicts.json` are never read or written. Each
protocol writes its own `<run>/judge_verdicts_<slug>.json`, shaped like
`judge::JudgeFile` plus the protocol that produced it. A cache or seed written
under a different protocol is refused, never mixed. The OpenRouter key comes
from ~/.config/myelin/openrouter.key and is never printed.

Usage (from crates/myelin-eval):

    ../../.venv/bin/python adapters/judge_matched.py --protocol lightmem-locomo --run ../../runs/m63_locomo_base
    ../../.venv/bin/python adapters/judge_matched.py --protocol mempro-locomo --run ../../runs/m63_locomo_base
    ../../.venv/bin/python adapters/judge_matched.py --protocol longmemeval-official \\
        --run ../../runs/m57_bonsai_premise_s1 --dataset ../../data/longmemeval_s.json
    ../../.venv/bin/python adapters/judge_matched.py --protocol lightmem-locomo --verify-upstream <llm_judge.py>
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

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
# MemPro, LightMem and LongMemEval called OpenAI's own API; pin that provider
# on OpenRouter rather than let it route the judge to another host.
PROVIDER = {"order": ["OpenAI"], "allow_fallbacks": False}
TEMPERATURE = 0.0
KEY_PATH = Path("~/.config/myelin/openrouter.key").expanduser()
# The one accepted key-file format (decided 2026-09-24): a single line.
KEY_LINE = re.compile(r"^OPENROUTER_API_KEY=(\S+)$")
IN_FLIGHT = 8
REQUEST_TIMEOUT_S = 60.0
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0
# Retry only what a later attempt can fix: rate limits, server errors, and a
# reply the protocol's own parser cannot read.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
LOCOMO_JUDGED_CATEGORIES = (1, 2, 3, 4)
LME_MAX_TOKENS = 10

# --- LightMem (verbatim; see the module doc). Do not edit; the sha is recorded.
LIGHTMEM_PROMPT = """
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

# --- MemPro's repo (verbatim; see the module doc). Do not edit.
MEMPRO_PROMPT = """Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG. The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic. The generated answer might be much longer, but you should be generous with your grading. As long as it touches on the same topic as the gold answer, it should be counted as CORRECT. For time related questions, the gold answer will often be a specific date, month, year, or time period. The generated answer might be longer or use relative references like "last week" or "the week before", but you should be generous if it refers to the same date or time period as the gold answer. Return the label CORRECT or WRONG in JSON format with the key "label". Do not include both CORRECT and WRONG in your response. Question: {question} Gold answer: {gold_answer} Generated answer: {generated_answer}"""
MEMPRO_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["CORRECT", "WRONG"]
        }
    },
    "required": ["label"],
    "additionalProperties": False
}

# --- LongMemEval's get_anscheck_prompt templates (verbatim; see the module doc).
LME_GENERAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
LME_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
LME_KNOWLEDGE_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
LME_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
LME_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
LME_TEMPLATE_BY_TYPE = {
    "single-session-user": LME_GENERAL,
    "single-session-assistant": LME_GENERAL,
    "multi-session": LME_GENERAL,
    "temporal-reasoning": LME_TEMPORAL,
    "knowledge-update": LME_KNOWLEDGE_UPDATE,
    "single-session-preference": LME_PREFERENCE,
}


def lightmem_extract_json(text: str) -> str:
    """LightMem's `extract_json`, verbatim in behaviour."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def parse_lightmem(content: str) -> int:
    return 1 if json.loads(lightmem_extract_json(content))["label"] == "CORRECT" else 0


def parse_mempro(content: str) -> int:
    # `call_llm_judge` returns `.get("label", "WRONG")`; `llm_judge_score` strips and upper-cases.
    return 1 if str(json.loads(content).get("label", "WRONG")).strip().upper() == "CORRECT" else 0


def parse_lme(content: str) -> int:
    # `evaluate_qa.py`: `label = 'yes' in eval_response.lower()`.
    return 1 if "yes" in content.lower() else 0


PROTOCOLS = {
    "lightmem-locomo": {
        "slug": "lightmem",
        "corpus": "locomo",
        "source": "zjunlp/LightMem@8449d574df6bae1bdf3314a1564da65e2f37e046:experiments/locomo/llm_judge.py",
        "model": "openai/gpt-4o-mini",
        "prompts": (LIGHTMEM_PROMPT,),
        "response_format": {"type": "json_object"},
        "max_tokens": None,
        "parse": parse_lightmem,
    },
    "mempro-locomo": {
        "slug": "mempro",
        "corpus": "locomo",
        "source": "wanghai673/MemPro@834b1cef87c80f3a8dbb2bcd398a6c7d6104d380:eval/locomo_test.py",
        "model": "openai/gpt-4o-mini",
        "prompts": (MEMPRO_PROMPT, json.dumps(MEMPRO_SCHEMA, sort_keys=True)),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judge_response", "schema": MEMPRO_SCHEMA, "strict": True},
        },
        "max_tokens": None,
        "parse": parse_mempro,
    },
    "longmemeval-official": {
        "slug": "lme_official",
        "corpus": "longmemeval_s",
        "source": "xiaowu0162/LongMemEval@9e0b455f4ef0e2ab8f2e582289761153549043fc:src/evaluation/evaluate_qa.py",
        "model": "openai/gpt-4o-mini-2024-07-18",
        "prompts": (LME_GENERAL, LME_TEMPORAL, LME_KNOWLEDGE_UPDATE, LME_PREFERENCE, LME_ABSTENTION),
        "response_format": None,
        "max_tokens": LME_MAX_TOKENS,
        "parse": parse_lme,
    },
}


def protocol_record(name: str) -> dict:
    """What a verdict file records about the grader that wrote it."""
    p = PROTOCOLS[name]
    record = {
        "name": name,
        "source": p["source"],
        "prompt_sha256": hashlib.sha256("\x00".join(p["prompts"]).encode("utf-8")).hexdigest(),
        "model": p["model"],
        "provider": PROVIDER,
        "temperature": TEMPERATURE,
        "response_format": p["response_format"],
    }
    if p["corpus"] == "locomo":
        record["categories"] = list(LOCOMO_JUDGED_CATEGORIES)
    else:
        record["max_tokens"] = p["max_tokens"]
        record["rows"] = "all"
    return record


def read_key() -> str:
    if not KEY_PATH.exists():
        raise SystemExit(f"{KEY_PATH} does not exist; it must hold one line OPENROUTER_API_KEY=<key>")
    lines = [line for line in KEY_PATH.read_text(encoding="utf-8").split("\n") if line.strip()]
    if len(lines) != 1 or not KEY_LINE.match(lines[0].strip()):
        raise SystemExit(f"{KEY_PATH} must hold exactly one line OPENROUTER_API_KEY=<key>; refusing to guess")
    return KEY_LINE.match(lines[0].strip()).group(1)


def jsonl(path: Path) -> list[dict]:
    # "\n" only: `str.splitlines` also breaks on U+2028 and friends, which
    # occur inside JSON strings in LongMemEval answers.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def load_verdict_file(path: Path, record: dict, what: str) -> dict | None:
    """A verdict file written under exactly this protocol, or a refusal."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol") != record:
        raise SystemExit(
            f"{path} ({what}) was written under protocol {data.get('protocol')!r}, not this one; "
            "mixing two graders' verdicts is refused"
        )
    return data


def locomo_items(run: Path) -> list[dict]:
    rows = [r for r in jsonl(run / "per_question.jsonl") if int(r["category"]) in LOCOMO_JUDGED_CATEGORIES]
    return [
        {
            "id": r["question_id"],
            "group": f"cat {r['category']}",
            "answer": r["response_raw"],
            "fill": {"question": r["question_text"], "gold_answer": str(r["answer_gold"]), "generated_answer": r["response_raw"]},
        }
        for r in rows
    ]


def lme_items(run: Path, dataset: Path) -> list[dict]:
    """All 500 rows, with each question's type, question and answer from the dataset (the
    official script's reference file), and abstention routed by `_abs` in the id."""
    refs = {d["question_id"]: d for d in json.loads(dataset.read_text(encoding="utf-8"))}
    items = []
    for r in jsonl(run / "per_question.jsonl"):
        ref = refs.get(r["question_id"])
        if ref is None:
            raise SystemExit(f"{r['question_id']} is not in {dataset}")
        abstention = "_abs" in r["question_id"]
        if abstention:
            template = LME_ABSTENTION
        elif ref["question_type"] in LME_TEMPLATE_BY_TYPE:
            template = LME_TEMPLATE_BY_TYPE[ref["question_type"]]
        else:
            raise SystemExit(f"{r['question_id']}: no official template for type {ref['question_type']!r}")
        items.append(
            {
                "id": r["question_id"],
                "group": ref["question_type"] + (" (abstention)" if abstention else ""),
                "answer": r["response_raw"],
                "prompt": template.format(ref["question"], ref["answer"], r["response_raw"]),
            }
        )
    return items


class Judge:
    def __init__(self, name: str) -> None:
        self.p = PROTOCOLS[name]
        self.key = read_key()
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT_S)
        self.lock = threading.Lock()
        self.served_models: set[str] = set()
        self.providers: set[str] = set()
        self.cost = 0.0

    def grade(self, prompt: str) -> int:
        body = {
            "model": self.p["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "provider": PROVIDER,
            "usage": {"include": True},
        }
        if self.p["response_format"] is not None:
            body["response_format"] = self.p["response_format"]
        if self.p["max_tokens"] is not None:
            body["max_tokens"] = self.p["max_tokens"]
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
                verdict = self.p["parse"](data["choices"][0]["message"]["content"] or "")
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
                last = f"unparseable reply ({type(e).__name__})"
                continue
            with self.lock:
                self.served_models.add(str(data.get("model", "")))
                self.providers.add(str(data.get("provider", "")))
                self.cost += float((data.get("usage") or {}).get("cost") or 0.0)
            return verdict
        raise SystemExit(f"judge failed after {MAX_ATTEMPTS} attempts: {last}")


def verify_upstream(name: str, upstream: Path) -> None:
    """Check this adapter's copy of a protocol's prompt against the pinned upstream file."""
    import ast

    tree = ast.parse(upstream.read_text(encoding="utf-8"))
    assigned: dict[str, list] = {}
    # `ast.walk` is breadth-first; sort by line so repeated names (the five
    # `template = ...` branches) come back in source order.
    nodes = sorted(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)),
        key=lambda n: n.lineno,
    )
    for node in nodes:
        try:
            assigned.setdefault(node.targets[0].id, []).append(ast.literal_eval(node.value))
        except ValueError:
            continue
    if name == "lightmem-locomo":
        ok = assigned.get("ACCURACY_PROMPT") == [LIGHTMEM_PROMPT]
    elif name == "mempro-locomo":
        ok = assigned.get("JUDGE_PROMPT_TEMPLATE") == [MEMPRO_PROMPT] and assigned.get("JUDGE_SCHEMA") == [MEMPRO_SCHEMA]
    else:
        ok = assigned.get("template") == [LME_GENERAL, LME_TEMPORAL, LME_KNOWLEDGE_UPDATE, LME_PREFERENCE, LME_ABSTENTION]
    print(f"{name}: {'byte-identical to' if ok else 'DIFFERS from'} {upstream}")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    ap.add_argument("--run", type=Path, help="a bench run directory of the protocol's corpus")
    ap.add_argument("--dataset", type=Path, help="longmemeval-official: the LongMemEval_S reference file")
    ap.add_argument("--seed", type=Path, action="append", default=[],
                    help="another run whose verdicts are reused for byte-identical (question, answer) pairs")
    ap.add_argument("--verify-upstream", type=Path, help="check the prompt copy against the pinned upstream file and exit")
    args = ap.parse_args()
    if args.verify_upstream:
        verify_upstream(args.protocol, args.verify_upstream)
        return
    if args.run is None:
        ap.error("--run is required")
    p = PROTOCOLS[args.protocol]
    record = protocol_record(args.protocol)

    agg = json.loads((args.run / "aggregated_metrics.json").read_text(encoding="utf-8"))
    if agg.get("corpus") != p["corpus"]:
        raise SystemExit(f"{args.run} is corpus {agg.get('corpus')!r}; {args.protocol} grades {p['corpus']}")
    if p["corpus"] == "locomo":
        items = locomo_items(args.run)
        prompt_of = lambda it: p["prompts"][0].format(**it["fill"])  # noqa: E731
    else:
        if args.dataset is None:
            ap.error("--dataset is required for longmemeval-official")
        items = lme_items(args.run, args.dataset)
        prompt_of = lambda it: it["prompt"]  # noqa: E731

    out_name = f"judge_verdicts_{p['slug']}.json"
    out_path = args.run / out_name
    known: dict[tuple[str, str], int] = {}
    existing = load_verdict_file(out_path, record, "cache")
    served_before: dict[str, set[str]] = {"models": set(), "providers": set()}
    if existing:
        for qid, v in existing["verdicts"].items():
            known[(qid, existing["answers"][qid])] = v
        # Reused verdicts keep the record of what served them.
        for k in served_before:
            served_before[k].update(existing.get("served", {}).get(k, []))
    for seed in args.seed:
        seeded = load_verdict_file(seed / out_name, record, f"seed {seed}")
        if seeded is None:
            raise SystemExit(f"seed {seed} has no {out_name}")
        for qid, v in seeded["verdicts"].items():
            known.setdefault((qid, seeded["answers"][qid]), v)

    verdicts: dict[str, int] = {}
    answers: dict[str, str] = {}
    todo = []
    for it in items:
        if (it["id"], it["answer"]) in known:
            verdicts[it["id"]] = known[(it["id"], it["answer"])]
            answers[it["id"]] = it["answer"]
        else:
            todo.append(it)
    reused = len(verdicts)

    judge = Judge(args.protocol)
    done = 0

    def one(it: dict) -> tuple[str, str, int]:
        return it["id"], it["answer"], judge.grade(prompt_of(it))

    with cf.ThreadPoolExecutor(max_workers=IN_FLIGHT) as pool:
        for qid, answer, v in pool.map(one, todo):
            verdicts[qid] = v
            answers[qid] = answer
            done += 1
            if done % 200 == 0:
                print(f"  graded {done}/{len(todo)}", file=sys.stderr)

    file = {
        "model": p["model"],
        "protocol": record,
        "served": {
            "models": sorted(judge.served_models | served_before["models"]),
            "providers": sorted(judge.providers | served_before["providers"]),
        },
        "verdicts": dict(sorted(verdicts.items())),
        "answers": dict(sorted(answers.items())),
    }
    out_path.write_text(json.dumps(file, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    groups: dict[str, list[int]] = {}
    for it in items:
        groups.setdefault(it["group"], []).append(verdicts[it["id"]])
    total = [v for vs in groups.values() for v in vs]
    print(f"{args.run.name}: {args.protocol}, {p['model']}: {sum(total)}/{len(total)} = "
          f"{100 * sum(total) / len(total):.2f} (graded {len(todo)}, reused {reused}, cost ${judge.cost:.4f})")
    for g in sorted(groups):
        vs = groups[g]
        print(f"  {g:<40} n={len(vs):<4} {100 * sum(vs) / len(vs):6.2f}")
    print(f"  served models={sorted(judge.served_models)} providers={sorted(judge.providers)}")


if __name__ == "__main__":
    main()
