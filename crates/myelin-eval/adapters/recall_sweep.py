"""Retrieval-only recall sweep: does the gold answer string reach the evidence?

M37's diagnostic. Reads the same store the LME-V2 arms read, calls the memory
server's `recall` tool at a range of `k`, and asks one question per row: is the
gold answer string present in the composed evidence at all?

No reader, no judge, no scoring — so a full 451-question sweep costs embedding
and search only. The point is to separate "the reader got it wrong" from "the
reader was never shown it", which judged accuracy alone cannot do.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from myelin import _McpSession  # type: ignore[attr-defined]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def gold_phrases(row: dict) -> list[str]:
    """The strings that must appear for the evidence to contain the answer.

    Phrase-set answers are scored as a set by the harness, so every member has
    to be present; a single-phrase answer is one member.
    """
    answer = row.get("answer")
    if isinstance(answer, bool):
        return []
    if isinstance(answer, list):
        parts = [str(x) for x in answer]
    else:
        text = str(answer)
        eval_fn = row.get("eval_function") or ""
        parts = re.split(r"[,;]", text) if "separators=" in eval_fn else [text]
    return [p for p in (norm(x) for x in parts) if p]


def checkable(row: dict) -> bool:
    """Rows whose gold is a string that could appear verbatim in evidence.

    Multiple-choice and boolean golds ("true", "B") match by accident, so they
    carry no information about retrieval and are excluded rather than counted.
    """
    eval_fn = (row.get("eval_function") or "").split("|")[0]
    if eval_fn in {"mc_choice_match"}:
        return False
    return bool(gold_phrases(row))


def evidence_text(result: dict) -> str:
    items = result.get("evidence") or result.get("items") or []
    out = []
    for item in items:
        if isinstance(item, dict):
            out.append(str(item.get("value") or item.get("text") or ""))
        else:
            out.append(str(item))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7446/mcp")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--k", type=int, nargs="+", default=[25])
    ap.add_argument("--mode", default="recall", choices=["recall", "investigate"])
    ap.add_argument("--tenant-prefix", default="lme_v2_small")
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--budget-tokens", type=int, default=10_000)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line]
    if args.domain:
        rows = [r for r in rows if r.get("domain") == args.domain]
    rows = [r for r in rows if checkable(r)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} string-checkable rows", file=sys.stderr)

    session = _McpSession(args.url, timeout=600.0)
    session.initialize()

    def probe(row: dict, k: int) -> dict:
        arguments: dict[str, object] = {
            "tenant": f"{args.tenant_prefix}/{row['domain']}",
            "k": k,
            "budget_tokens": args.budget_tokens,
            "select": args.select,
        }
        if args.mode == "investigate":
            arguments["question"] = row["question"]
            arguments["max_steps"] = args.max_steps
        else:
            arguments["query"] = row["question"]
        result = session.call_tool(args.mode, arguments)
        text = norm(evidence_text(result))
        phrases = gold_phrases(row)
        meta = result.get("trace") or {}
        return {
            "id": row["id"],
            "domain": row.get("domain"),
            "k": k,
            "present": all(p in text for p in phrases),
            "n_items": len(result.get("evidence") or result.get("items") or []),
            "pool": meta.get("pool"),
            "chars": len(text),
        }

    out_rows = []
    for k in args.k:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            got = list(pool.map(lambda r: probe(r, k), rows))
        out_rows.extend(got)
        hit = sum(1 for g in got if g["present"])
        items = sum(g["n_items"] for g in got) / max(len(got), 1)
        print(
            f"k={k:4}  recall={hit / len(got):6.1%}  ({hit}/{len(got)})  "
            f"mean_items={items:5.1f}",
            file=sys.stderr,
        )

    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")


if __name__ == "__main__":
    main()
