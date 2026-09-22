"""Retrieval-only recall: does the evidence contain what answers the question?

M37's diagnostic, M38's generalisation. Reads the same store an arm reads,
calls the memory server at a range of `k`, and scores retrieval alone — no
reader, no judge — so a full sweep costs embedding and search only. The point
is to separate "the reader got it wrong" from "the reader was never shown it",
which judged accuracy cannot do.

Two scoring modes, because two corpora annotate differently:

`--mode-score string` (default)
    Is every gold phrase present verbatim in the composed evidence? Used for
    LME-V2, whose answers are exact UI labels and which ships no evidence
    annotation. A proxy: it undercounts paraphrase, so it is a lower bound.

`--mode-score session`
    Did retrieval return a record from a gold session? Exact, not a proxy.
    LongMemEval ships `answer_session_ids`, and the session a record came from
    is recoverable from `prov_source.doc` in the ledger, so this needs no
    re-ingest. This is the measurement to trust where it is available.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from myelin import _McpSession  # type: ignore[attr-defined]


def sessions_by_record(ledger: str, tenant_prefix: str) -> dict[str, str]:
    """Record id -> the session it was written from.

    Recovered from `prov_source.doc`, which the corpus builds set to
    `<session_id>#<turn>`. `record.scope.session` is the field that *should*
    carry this and every corpus build so far has left it empty; until that is
    fixed this mapping is how a gold-session score is possible at all.
    """
    con = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    out: dict[str, str] = {}
    for rid, prov in con.execute(
        "SELECT id, prov_source FROM record WHERE tenant LIKE ? AND t_invalid IS NULL",
        (f"{tenant_prefix}%",),
    ):
        doc = (json.loads(prov or "{}").get("doc") or "")
        if doc:
            out[str(rid)] = doc.split("#", 1)[0]
    con.close()
    return out


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
    ap.add_argument(
        "--select-coverage",
        action="store_true",
        help="ask the selector for every needed memory rather than the fewest "
        "(M38). Inert without --select.",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--score",
        default="string",
        choices=["string", "session"],
        help="string: gold phrases present verbatim (proxy). session: retrieval "
        "returned a record from a gold session (exact, needs --ledger).",
    )
    ap.add_argument("--ledger", default=None, help="required for --score session")
    ap.add_argument(
        "--tenant-field",
        default="domain",
        help="question field appended to --tenant-prefix to form the tenant. "
        "LME-V2 shares one tenant per domain; LongMemEval has one per question.",
    )
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line]
    if args.domain:
        rows = [r for r in rows if r.get("domain") == args.domain]
    if args.score == "string":
        rows = [r for r in rows if checkable(r)]
    else:
        rows = [r for r in rows if r.get("answer_session_ids")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} scoreable rows ({args.score})", file=sys.stderr)

    record_session: dict[str, str] = {}
    if args.score == "session":
        if not args.ledger:
            raise SystemExit("--score session requires --ledger")
        record_session = sessions_by_record(args.ledger, args.tenant_prefix)
        print(f"{len(record_session):,} records mapped to sessions", file=sys.stderr)

    session = _McpSession(args.url, timeout=600.0)
    session.initialize()

    def probe(row: dict, k: int) -> dict:
        """One retrieval, scored.

        Never raises: a sweep is a long serial measurement and losing the
        whole run to one bad row wastes the wall-clock it took to reach it.
        A failed row is recorded with `error` set and excluded from the
        denominator, so a partial sweep reports what it actually measured
        instead of silently scoring a failure as a miss.
        """
        try:
            return _probe(row, k)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"  row {row.get('id')} failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return {"id": row.get("id"), "k": k, "error": repr(exc),
                    "present": False, "complete": False,
                    "gold_total": 0, "gold_hit": 0, "n_items": 0}

    def _probe(row: dict, k: int) -> dict:
        arguments: dict[str, object] = {
            "tenant": f"{args.tenant_prefix}/{row[args.tenant_field]}",
            "k": k,
            "budget_tokens": args.budget_tokens,
            "select": args.select,
        }
        if args.mode == "investigate":
            arguments["question"] = row["question"]
            arguments["select_coverage"] = args.select_coverage
            arguments["max_steps"] = args.max_steps
        else:
            arguments["query"] = row["question"]
        result = session.call_tool(args.mode, arguments)
        meta = result.get("trace") or {}
        ids = [str(i) for i in (result.get("record_ids") or [])]
        gold_total = gold_hit = 0
        if args.score == "session":
            gold = set(row.get("answer_session_ids") or [])
            got = {record_session.get(i) for i in ids}
            gold_total = len(gold)
            gold_hit = len(gold & got)
            # `present` stays "any gold session reached the reader", because
            # that is the number every LongMemEval baseline reports. It is
            # lenient by construction: multi-session questions need 2.59 gold
            # sessions on average and temporal-reasoning 2.20, so "any" scores
            # a question as retrieved when the evidence cannot answer it.
            # `complete` is the honest one for multi-hop.
            present = gold_hit > 0
        else:
            text = norm(evidence_text(result))
            present = all(p in text for p in gold_phrases(row))
        return {
            "id": row.get("id") or row.get("question_id"),
            "domain": row.get("domain"),
            "question_type": row.get("question_type"),
            "k": k,
            "present": present,
            "gold_total": gold_total,
            "gold_hit": gold_hit,
            "complete": gold_total > 0 and gold_hit == gold_total,
            "n_items": len(result.get("evidence") or result.get("items") or []),
            "pool": meta.get("pool"),
            "rerank_depth": meta.get("rerank_depth"),
        }

    out_rows = []
    for k in args.k:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            got = list(pool.map(lambda r: probe(r, k), rows))
        out_rows.extend(got)
        failed = [g for g in got if g.get("error")]
        ok = [g for g in got if not g.get("error")]
        if not ok:
            print(f"k={k:4}  every row failed", file=sys.stderr)
            continue
        hit = sum(1 for g in ok if g["present"])
        comp = sum(1 for g in ok if g["complete"])
        items = sum(g["n_items"] for g in ok) / len(ok)
        cov = [g["gold_hit"] / g["gold_total"] for g in ok if g["gold_total"]]
        extra = ""
        if cov:
            extra = (f"  complete={comp / len(ok):6.1%}  "
                     f"coverage={sum(cov) / len(cov):6.1%}")
        note = f"  [{len(failed)} failed]" if failed else ""
        print(
            f"k={k:4}  any={hit / len(ok):6.1%}  ({hit}/{len(ok)}){extra}  "
            f"mean_items={items:5.1f}{note}",
            file=sys.stderr,
        )

    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")


if __name__ == "__main__":
    main()
