#!/usr/bin/env python3
"""Run the official LongMemEval-V2 harness against `myelin`.

A sibling of the vendored `evaluation/run_eval.py`, not a replacement for it.
`run_eval.py` does three things — materialise the runtime question and
haystack files, write a `memory_config.json`, then set `sys.argv` and call
`evaluation.harness.main()` — and it is unusable for us only because its
`--method` is a closed `choices=` set. Everything downstream takes
`--memory-config-path` and is fully general, so this file does the same three
things with `"memory_type": "myelin"` and calls the same `main()`. The
harness, its scorers and its leaderboard builders run unmodified
(`PLAN.md` §3.3).

Two things this can do that `run_eval.py` cannot, both because it talks to
`evaluation.harness` directly:

* **`--evaluator-base-url`.** `harness.py` accepts one; `run_eval.py` never
  passes it, hardwiring the 156 LLM-judged questions to OpenAI. We point the
  judge wherever the run's manifest says, including a local server.
* **Importing the adapter first.** `memory_modules/memory.py` populates its
  registry from imports at the bottom of its own file, so `myelin` only
  exists as a `memory_type` once `adapters/myelin.py` has been imported.
  That import is the registration.

The memory itself is **not** built here. `MyelinMemory.insert()` asserts
rather than indexes, because building means running the Rust write path over
the haystack:

    myelin-eval build --corpus lme-v2-small
    myelin-mcp --serve 127.0.0.1:7446 \\
        --collection myelin_lme_v2_small --ledger data/lme_v2_small.ledger

Usage:

    PYTHONPATH=vendor/longmemeval-v2:adapters ../../.venv/bin/python \\
        adapters/run_myelin.py \\
        --data-root /tmp/lmev2 --domain web --tier small \\
        --output-dir runs/myelin_fast_web_small
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "longmemeval-v2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.public_data import (  # noqa: E402
    materialize_runtime_haystack,
    materialize_runtime_questions,
    write_json,
)

# Importing the adapter is what runs `@register_memory`. Without this line
# the harness raises `Unknown memory_type: myelin`, which is the failure mode
# a lazy `# noqa` would hide — hence the explicit note rather than a bare
# import.
import myelin  # noqa: E402,F401

# One reader request, shaped like the harness's, before any prompt is built.
# `max_tokens` 1: the probe asks whether the endpoint ACCEPTS the content
# type, not what the model says about it.
READER_PREFLIGHT_MAX_TOKENS = 1
READER_PREFLIGHT_TEXT = "Reply with the single word: ok."
READER_PREFLIGHT_TIMEOUT_S = 300.0

# `GET /v1/models` answers at once on a live llama-server; a server that has
# not answered in this long is not serving.
SERVED_MODEL_TIMEOUT_S = 30.0
# `investigate`'s step cap: the measured peak (docs/measurements/m7-step-value-curve.md).
INVESTIGATE_MAX_STEPS = 2
# M62's trajectory controller: `trajectory_agent::DEFAULT_MAX_STEPS` in
# myelin-core, tool calls before the answer is forced.
TRAJECTORY_MAX_STEPS = 16
# Seconds one MCP tool call may take. A search call answers in seconds; a
# trajectory-controller call runs up to TRAJECTORY_MAX_STEPS model calls, so
# it gets M54's controller budget (`run_agentrunbook_c.py --codex-timeout-seconds`).
SEARCH_CALL_TIMEOUT_S = 120.0
TRAJECTORY_CALL_TIMEOUT_S = 1800.0


def served_model(base_url: str) -> str:
    """The model file an OpenAI-compatible server reports serving (M55).

    The same measurement as `bench`'s `served_model`: llama-server answers
    `GET /v1/models` with the GGUF path, and the file name is kept. Asked of
    the server itself, not read from a flag, so an artifact names the model
    that actually served it. An unreachable server refuses the run.
    """
    import urllib.request  # noqa: E402

    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=SERVED_MODEL_TIMEOUT_S) as resp:
            body = json.load(resp)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"GET {url}: {exc}. Nothing was built.") from exc
    try:
        model_id = body["data"][0]["id"]
    except (KeyError, IndexError, TypeError):
        raise SystemExit(f"{url} reported no model id: {body}. Nothing was built.") from None
    return Path(model_id).name


def source_memory_model(source_dir: Path) -> str | None:
    """The memory-side model a `--reuse-prompts-from` source recorded.

    `None` when the source predates the field; `standing` reads that as the
    model every such run was served (the 9B), so the absence is carried, not
    guessed here.
    """
    path = source_dir / "runtime_inputs" / "memory_config.json"
    if not path.exists():
        raise SystemExit(f"--reuse-prompts-from: {path} does not exist. Nothing was built.")
    params = json.loads(path.read_text(encoding="utf-8")).get("memory_params") or {}
    return params.get("memory_llm_served_model")


class PromptsBuilt(Exception):
    """Raised in place of the reader call when `--prompts-only` is set."""


def first_question_image(selected_questions: list[dict]) -> str | None:
    """The screenshot path of the first selected question that carries one.

    `question` is a string on most LME-V2 rows and `{text, image}` on the
    rows with a screenshot (web small: 15 of 240; enterprise: none).
    """
    for question in selected_questions:
        field = question.get("question")
        if isinstance(field, dict) and field.get("image"):
            return field["image"]
    return None


def preflight_reader_images(args: argparse.Namespace, selected_questions: list[dict], harness_module) -> None:
    """Refuse to start when the reader cannot take the screenshots this run sends.

    `harness.build_messages` attaches a question's screenshot as `image_url`
    content. A reader served without its vision projector answers HTTP 500
    ("image input is not supported") to that request — and on 2026-09-23 it
    did so 45 minutes in, after every prompt had been built, which the harness
    cannot resume (BACKLOG.md, operational debt). So when any selected question
    carries an image, the first thing this driver does is send that image to
    the reader through the harness's own `to_data_url`, the same bytes it will
    send later; a refusal ends the run at the door with the fix named, and a
    run with no screenshots sends nothing.
    """
    image_path = first_question_image(selected_questions)
    if image_path is None:
        return
    import openai  # noqa: E402  (the harness's client library; imported here so a text-only run never needs it early)

    client = openai.OpenAI(
        base_url=args.reader_base_url,
        api_key=os.getenv(args.reader_api_key_env) or "EMPTY",
        max_retries=0,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": READER_PREFLIGHT_TEXT},
                {"type": "image_url", "image_url": {"url": harness_module.to_data_url(image_path)}},
            ],
        }
    ]
    try:
        client.chat.completions.create(
            model=args.reader_model,
            messages=messages,
            max_tokens=READER_PREFLIGHT_MAX_TOKENS,
            timeout=READER_PREFLIGHT_TIMEOUT_S,
        )
    except openai.APIError as exc:
        raise SystemExit(
            f"reader preflight: {args.reader_base_url} refused an image_url request: {exc}. "
            "This run's questions carry screenshots (question.image); serve the reader with its "
            "vision projector (ops/big/serve-models.sh with MYELIN_MMPROJ=1) and relaunch. "
            "Nothing was built."
        ) from exc
    print(f"reader preflight: {args.reader_base_url} accepted an image_url request ({image_path})", flush=True)


# M52. A thinking probe: long enough for the budget to matter, short enough
# to cost seconds. The completion ceiling leaves room for a full 1,024-token
# trace (the serve default, `MYELIN_READER_THINK_BUDGET`) plus an answer.
READER_THINKING_PROBE = "How many days are there between 2023-03-28 and 2023-05-02? Answer with the number."
READER_THINKING_PROBE_MAX_TOKENS = 2048


def enable_harness_thinking(harness_module) -> None:
    """Make `--reader-enable-thinking` actually enable thinking.

    `harness.build_extra_body` sends `chat_template_kwargs: {enable_thinking:
    False}` when thinking is off and sends *nothing* when it is on, because
    the vLLM server it was written for thinks by default. Ours does not:
    `ops/big/serve-models.sh` starts llama.cpp with thinking off server-wide
    so the evaluator, which shares the endpoint and never sends the flag,
    stays a plain judge. So "thinking on" reached our reader as no flag and
    ran with thinking OFF — silently.

    The fix wraps the harness's own function rather than editing it
    (`PLAN.md` §3.3 keeps the vendored harness unmodified): every field it
    computes is kept, and the one it leaves to the server's default is
    stated. The evaluator builds its requests elsewhere and is untouched.
    """
    original = harness_module.build_extra_body

    def build_extra_body(args):
        extra = original(args) or {}
        extra["chat_template_kwargs"] = {"enable_thinking": True}
        return extra

    harness_module.build_extra_body = build_extra_body


class ReplayedMemory:
    """A memory that answers one question from a finished run's prompt row.

    Implements exactly the four calls `harness.build_prompt_row` makes, so
    the harness's own truncation and message building run unchanged over the
    same memory context the source run's memory produced.
    """

    def __init__(self, source_row: dict):
        self.row = source_row

    def set_query_context(self, **_kwargs) -> None:
        return None

    def clear_query_context(self) -> None:
        return None

    def query(self, _text, query_image=None):
        return self.row["memory_context"]

    def post_query_hook(self, **_kwargs):
        return self.row["memory_post_query_metadata"]


def reuse_prompts_from(source_dir: Path, selected_questions: list[dict], runtime_dir: Path, harness_module) -> None:
    """Answer every memory query from `source_dir`'s prompt rows (M52).

    An arm that changes only the *harness reader* — M52's thinking — does not
    need its memory side re-run: the base already produced it, and re-running
    it costs ~1 GPU-hour per domain and, co-scheduled, is not even
    byte-identical (llama.cpp batching). Replaying the base's memory context
    makes the memory side of base and arm identical by construction, so the
    reader is the only thing that differs.

    Wraps the harness's own `build_prompt_row` (vendored file unmodified,
    `PLAN.md` §3.3): the replayed memory answers `query`, the harness builds
    the messages, and the source's measured memory latencies are carried
    over rather than the replay's ~0. Refuses when the source does not cover
    exactly this run's questions, or a rebuilt context differs from the
    source's.
    """
    rows_path = source_dir / "prompt_rows.jsonl"
    if not rows_path.exists():
        raise SystemExit(f"--reuse-prompts-from: {rows_path} does not exist (the source run has not finished building prompts)")
    rows = {}
    with rows_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                rows[row["question_id"]] = row
    wanted = {q["id"] for q in selected_questions}
    if set(rows) != wanted:
        raise SystemExit(
            f"--reuse-prompts-from: {source_dir} holds {len(rows)} prompt rows, this run selects "
            f"{len(wanted)} questions, {len(set(rows) ^ wanted)} differ. Nothing was built."
        )
    original = harness_module.build_prompt_row

    def build_prompt_row(item, *, haystack_ids, memory, system_prompt, memory_context_max_tokens):
        source = rows[item["question_id"]]
        row = original(
            item,
            haystack_ids=haystack_ids,
            memory=ReplayedMemory(source),
            system_prompt=system_prompt,
            memory_context_max_tokens=memory_context_max_tokens,
        )
        if row["memory_context"] != source["memory_context"] or row["messages"] != source["messages"]:
            raise RuntimeError(f"replayed prompt for {item['question_id']} differs from {source_dir}")
        row["memory_query_duration_seconds"] = source["memory_query_duration_seconds"]
        row["memory_post_query_duration_seconds"] = source["memory_post_query_duration_seconds"]
        return row

    harness_module.build_prompt_row = build_prompt_row
    write_json(
        runtime_dir / "reused_prompts.json",
        {"source": str(source_dir), "prompt_rows": len(rows), "rule": "memory context replayed byte-identical; harness reader re-run"},
    )
    print(f"reusing {len(rows)} prompt rows from {source_dir}", flush=True)


def reuse_responses_from(source_dir: Path, below_tokens: int, runtime_dir: Path, harness_module) -> None:
    """Reuse a failed run's scored answers; generate only the rest (M52).

    The harness keeps every answer in memory until scoring and cannot resume,
    so a run that dies while scoring loses answers it paid for. This reads
    the source run's `per_question.jsonl` and reuses an answer only when it
    is non-empty (the harness writes `""` for a rejected reader request) and
    its prompt plus completion stayed under `below_tokens` — i.e. the serving
    slot did not cut it off. Every other question is generated again.

    Wraps the harness's own `generate_all_reader_outputs` (vendored file
    unmodified). Scoring re-runs on every row, reused or not.
    """
    path = source_dir / "per_question.jsonl"
    if not path.exists():
        raise SystemExit(f"--reuse-responses-from: {path} does not exist")
    cache = {}
    skipped = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            usage = row.get("usage") or {}
            total = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
            if row.get("response_raw") and total < below_tokens:
                cache[row["question_id"]] = {
                    "response_raw": row["response_raw"],
                    "response_parsed_boxed": row["response_parsed_boxed"],
                    "is_unknown": row["is_unknown"],
                    "usage": usage,
                }
            else:
                skipped.append(row["question_id"])
    original = harness_module.generate_all_reader_outputs

    async def generate_all_reader_outputs(args, prompt_rows):
        todo = [r for r in prompt_rows if r["question_id"] not in cache]
        outputs = await original(args, todo) if todo else {}
        for r in prompt_rows:
            if r["question_id"] in cache:
                outputs[r["question_id"]] = dict(cache[r["question_id"]])
        return outputs

    harness_module.generate_all_reader_outputs = generate_all_reader_outputs
    write_json(
        runtime_dir / "reused_responses.json",
        {
            "source": str(source_dir),
            "reused": len(cache),
            "not_reused_from_source": skipped,
            "rule": f"non-empty and prompt+completion < {below_tokens} tokens; everything else regenerated",
        },
    )
    print(f"reusing {len(cache)} answers from {source_dir}; {len(skipped)} of its rows regenerated", flush=True)


def preflight_reader_thinking(args: argparse.Namespace) -> None:
    """Refuse to start a thinking run whose reader does not think.

    One request with `enable_thinking: true`: the reply must carry a non-empty
    `reasoning_content` (llama.cpp's field for the trace) and a non-empty
    answer. The first catches a server that ignores the flag; the second, the
    failure that made thinking default-off here — a trace that eats the whole
    completion and leaves `content: ""` — which the server's
    `--reasoning-budget` exists to prevent.
    """
    import openai  # noqa: E402

    client = openai.OpenAI(
        base_url=args.reader_base_url,
        api_key=os.getenv(args.reader_api_key_env) or "EMPTY",
        max_retries=0,
    )
    try:
        reply = client.chat.completions.create(
            model=args.reader_model,
            messages=[{"role": "user", "content": READER_THINKING_PROBE}],
            max_tokens=READER_THINKING_PROBE_MAX_TOKENS,
            timeout=READER_PREFLIGHT_TIMEOUT_S,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
    except openai.APIError as exc:
        raise SystemExit(f"reader thinking preflight: request failed: {exc}. Nothing was built.") from exc
    message = reply.choices[0].message
    trace = getattr(message, "reasoning_content", None) or (getattr(message, "model_extra", None) or {}).get("reasoning_content")
    answer = (message.content or "").strip()
    if not trace:
        raise SystemExit(
            "reader thinking preflight: the reply carried no reasoning_content, so the server "
            "ignored enable_thinking. Nothing was built."
        )
    if not answer:
        raise SystemExit(
            "reader thinking preflight: the trace consumed the completion and the answer is empty; "
            "serve the reader with a --reasoning-budget (MYELIN_READER_THINK_BUDGET). Nothing was built."
        )
    print(
        f"reader thinking preflight: trace {len(trace)} chars, answer {answer[:40]!r}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official LongMemEval-V2 harness against myelin."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--domain", choices=["web", "enterprise"], required=True)
    parser.add_argument("--tier", choices=["small", "medium"], default="small")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-ids", nargs="*", default=None)

    # Which memory to read. The tenant defaults to `<tier>/<domain>`, which is
    # exactly what `myelin-eval build --corpus lme-v2-*` writes, so the two
    # halves cannot drift apart silently.
    parser.add_argument("--mcp-url", default=os.getenv("MYELIN_MCP_URL", "http://127.0.0.1:7446/mcp"))
    parser.add_argument(
        "--ledger",
        default=None,
        help="SQLite ledger the served collection is backed by. Read-only, and "
        "only to census it: the census goes into `memory_config.json` as "
        "`store_fingerprint` so two runs that read DIFFERENT STORE CONTENT "
        "through the same collection name cannot be paired as one submission. "
        "M34 created exactly that situation -- it minted the events/notes "
        "pools into `myelin_lme_v2_small`, after which `standing` paired an "
        "M34 web run with an M33 enterprise run and published 39.47, a "
        "combined number no configuration ever produced.",
    )
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--budget-tokens", type=int, default=2048)
    parser.add_argument("--mode", choices=["recall", "investigate", "trajectories"], default="recall")
    # Default per mode, resolved after parsing: investigate's measured peak,
    # or the trajectory controller's own budget. One number for both would
    # quietly cut M62's controller off after two tool calls.
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--mcp-timeout-seconds", type=float, default=None,
                        help="Per MCP call; default per mode (search 120, trajectories 1800).")
    # Retrieval width, recorded only. The server takes these from its own CLI
    # (`myelin-mcp --prefetch-limit --rerank-depth`); they are written to
    # `runtime_inputs/memory_config.json` so an operating point is
    # reproducible from the run directory, which is all `evidence-audit` and
    # any later comparison read. They are deliberately NOT sent in the
    # `recall` call: `RecallParams` accepts nothing beyond query, scope, `k`
    # and `budget_tokens`, and widening that surface widens what the
    # benchmark's own `tests/test_query_privacy.py` has to guard.
    parser.add_argument("--prefetch-limit", type=int, default=None)
    parser.add_argument("--rerank-depth", type=int, default=None)
    parser.add_argument(
        "--tau-abstain",
        type=float,
        default=None,
        help="Withhold the evidence set when the best cross-encoder score is below "
        "this, so the reader abstains instead of answering from a bad pool.",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Ask the model which retrieved candidates jointly answer the question and "
        "put those first. One extra model call per query, so it is an operating point "
        "(R4) and never a `recall` default.",
    )
    parser.add_argument(
        "--pool-rerank",
        action="store_true",
        help="Rerank the investigate loop's whole accumulated pool against the original "
        "question before composing (M23 A2). One reranker call per query; investigate-only.",
    )
    parser.add_argument(
        "--premise",
        action="store_true",
        help="When the investigate loop stops unsatisfied, replace the bare insufficiency "
        "statement with an explicit premise analysis in the evidence channel (M23 A3). "
        "Implies the insufficiency gate. investigate-only.",
    )
    # M43 shipped the dated digest ON for `investigate`. The adapter records
    # both switches unconditionally, as it does `select`: `standing` reads an
    # artifact that lacks the keys as a pre-M43 run — which ran without the
    # digest and is therefore an arm of today's defaults — so a run at the
    # shipped configuration must say so or it cannot be quoted.
    parser.add_argument(
        "--no-item-digest",
        dest="item_digest",
        action="store_false",
        help="Turn the per-memory digest off (M40/M43; shipped ON). investigate-only.",
    )
    parser.add_argument(
        "--no-digest-dates",
        dest="digest_dates",
        action="store_false",
        help="Leave the digest's lines undated (M41/M43; shipped ON). Inert on an undated "
        "corpus, where no line carries a stamp to copy. investigate-only.",
    )
    parser.set_defaults(item_digest=True, digest_dates=True)
    parser.add_argument(
        "--premise-check",
        action="store_true",
        help="Verify what the question assumes against the composed memories and append "
        "a [premise] line only when a memory contradicts it (M47). Silence appends "
        "nothing, which is the one rule that separates it from --premise. investigate-only.",
    )
    parser.add_argument(
        "--reuse-prompts-from",
        default=None,
        help="Answer every memory query from this finished run's prompt_rows.jsonl instead of "
        "querying the memory: for an arm that changes only the harness reader (M52). The memory "
        "side is then byte-identical to the source run by construction.",
    )
    parser.add_argument(
        "--reuse-responses-from",
        default=None,
        help="Reuse the answers a failed run already scored (per_question.jsonl) and generate "
        "only the rest. Requires --reuse-responses-below-tokens.",
    )
    parser.add_argument(
        "--reuse-responses-below-tokens",
        type=int,
        default=None,
        help="Reuse an answer only when its prompt+completion stayed below this many tokens "
        "(the source run's slot size, less a margin): a longer one may have been cut off.",
    )
    parser.add_argument(
        "--typed-probes",
        action="store_true",
        help="Let the reflect gate aim each investigate probe at a pool: "
        '"event" for the state-transition pool, "note" for the procedure/hint '
        "pool; untagged probes search the whole store. Off by default, and "
        "untagged means every measured arm is byte-identical (M23 D2).",
    )
    parser.add_argument(
        "--answerability-gate",
        action="store_true",
        help="Judge whether the composed evidence answers the question and act on a "
        "graded verdict (supported / ambiguous / unsupported). Replaces the loop's "
        "stop reason as the abstention trigger, which M36 measured at 1.16x lift. "
        "investigate-only.",
    )
    parser.add_argument(
        "--kind-quota",
        action="store_true",
        help="Allocate k's slots per record kind -- AgentRunbook-R's top-6 events, "
        "top-3 notes, raw states taking the rest -- instead of handing every slot "
        "to whichever kind wins one fused ranking. M34 measured our emitted mix at "
        "10.6%% events against their 31.6%%. investigate and recall both.",
    )
    parser.add_argument(
        "--decompose",
        type=int,
        default=None,
        help="Split each question into at most N sub-queries and retrieve for each, "
        "fusing them into the same RRF call as the original (M24). One model call "
        "per query; on investigate, one per probe.",
    )
    parser.add_argument(
        "--dated",
        action="store_true",
        help="Turn the date mechanisms (stamp_valid_time, resolve_relative, timeline) back ON. "
        "Off is LME-V2's shipped point (decided 2026-09-23): all 85,589 of its records "
        "carry the ingest timestamp as t_valid because its trajectories are agent task "
        "logs with no dates, and M22 measured undated at +2.4 over dated (a null). Every "
        "LME-V2 run since M33 passed the old --undated.",
    )

    # M55: the memory side and the harness reader can be different models.
    # The LME-V2 protocol fixes the READER to Qwen3.5-9B for every system
    # (`10.48550/arXiv.2605.12493`), so a stronger system model can only be
    # compared there by building memory with it and reading with the 9B:
    # phase one `--prompts-only` with the system model served, phase two
    # `--reuse-prompts-from` with the 9B served.
    parser.add_argument(
        "--prompts-only",
        action="store_true",
        help="Build and save every prompt row, then stop before the reader (phase one of a "
        "two-model run). Pair with --reuse-prompts-from <this output dir> in phase two.",
    )
    parser.add_argument(
        "--memory-llm-url",
        default=os.getenv("MEMORY_LLM_URL", "http://127.0.0.1:5810/v1"),
        help="The endpoint the myelin MCP server's own LLM calls go to (its config's llm.url). "
        "Asked once for the model it serves, recorded as memory_llm_served_model.",
    )

    # Reader. Defaults are this project's tunnelled llama-server, not the
    # harness's `localhost:8023`, because a default that points at nothing is
    # a 10-minute debugging session every time.
    # `Qwen/Qwen3.5-9B` exactly, not the local llama-server alias.
    # harness.py:857 gates `chat_template_kwargs={"enable_thinking": False}`
    # on `args.model == "Qwen/Qwen3.5-9B"` — a string compare — so any other
    # spelling silently leaves thinking ON, and this reader then spends the
    # whole completion budget on reasoning_content and returns content "",
    # which the harness rejects as `Model returned empty text`. Measured:
    # that is exactly how the first end-to-end run failed. llama-server
    # ignores the field and serves the loaded GGUF regardless, so the
    # upstream spelling is both correct for the manifest and required here.
    parser.add_argument("--reader-model", default=os.getenv("READER_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--reader-base-url", default=os.getenv("READER_BASE_URL", "http://127.0.0.1:5810/v1"))
    parser.add_argument("--reader-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reader-temperature", type=float, default=0.6)
    parser.add_argument("--reader-top-p", type=float, default=0.95)
    parser.add_argument("--reader-top-k", type=int, default=20)
    parser.add_argument("--reader-max-concurrent-requests", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=20000)
    parser.add_argument("--memory-context-max-tokens", type=int, default=200000)
    # Thinking off by default: measured on this reader, Qwen3.5-9B spends the
    # whole completion budget on `reasoning_content` and returns
    # `content: ""`. See `llm/openai.rs`. Since M44 R2 the server caps the
    # trace (`--reasoning-budget`), and `--reader-enable-thinking` is M52's
    # arm: it is preflighted and made explicit (`enable_harness_thinking`),
    # because the harness leaves "on" to a server default that is "off" here.
    parser.add_argument(
        "--reader-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prompt-build-max-workers", type=int, default=4)

    # Judge. `run_eval.py` hardwires OpenAI; `harness.py` does not, so the
    # base URL is exposed here and recorded in the manifest.
    parser.add_argument("--evaluator-model", default=os.getenv("EVALUATOR_MODEL", "gpt-5.2"))
    parser.add_argument("--evaluator-base-url", default=os.getenv("EVALUATOR_BASE_URL"))
    parser.add_argument("--evaluator-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--evaluator-reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument(
        "--openai-max-retries",
        type=int,
        default=10,
        help="Retry budget for every reader and judge call. 10 is the vendored default "
        "and the openai SDK caps its backoff at 8 s, so 10 covers only ~55 s of outage. "
        "`big` is a shared host: another tenant's vLLM start drove it to load 151, sshd "
        "stopped answering, the 5810/5813 tunnel dropped for ~5 minutes, and a 35-minute "
        "arm died with APIConnectionError. Raise this on a contended host — the harness "
        "keeps generations in memory and writes per_question.jsonl only during scoring, "
        "so one exhausted retry budget destroys the whole run.",
    )
    parser.add_argument("--evaluator-max-completion-tokens", type=int, default=4096)

    parser.add_argument("--shuffle-questions-seed", type=int, default=None)
    args = parser.parse_args()
    if args.max_steps is None:
        args.max_steps = TRAJECTORY_MAX_STEPS if args.mode == "trajectories" else INVESTIGATE_MAX_STEPS
    if args.mcp_timeout_seconds is None:
        args.mcp_timeout_seconds = (
            TRAJECTORY_CALL_TIMEOUT_S if args.mode == "trajectories" else SEARCH_CALL_TIMEOUT_S
        )
    return args


def parse_question_ids(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    ids: list[str] = []
    for chunk in raw:
        ids.extend(part for part in chunk.replace(",", " ").split() if part)
    return ids or None


def store_fingerprint(ledger: str | None) -> str | None:
    """Census the ledger by record kind, as an order-independent string.

    The collection NAME is not the store's identity: M34 minted two new
    record kinds into `myelin_lme_v2_small` without renaming it, so runs
    before and after read materially different stores through one name. The
    counts are what actually differ, and they are cheap to read.

    `None` when no ledger was given, which is how every pre-M34 artifact
    reads — and `standing` refuses to pair an artifact that names its store
    with one that does not, because "unknown" is not a match.
    """
    if not ledger:
        return None
    import sqlite3

    con = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT kind, COUNT(*) FROM record WHERE t_invalid IS NULL GROUP BY kind"
        ).fetchall()
    finally:
        con.close()
    return " ".join(f"{k}={n}" for k, n in sorted(rows))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runtime_dir = output_dir / "runtime_inputs"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    selected_questions = materialize_runtime_questions(
        data_root=data_root,
        domain=args.domain,
        question_ids=parse_question_ids(args.question_ids),
        limit=args.limit,
        output_path=runtime_dir / "questions.json",
    )
    materialize_runtime_haystack(
        data_root=data_root,
        tier=args.tier,
        selected_questions=selected_questions,
        output_path=runtime_dir / "haystack.json",
    )

    if args.prompts_only and (args.reuse_prompts_from or args.reuse_responses_from):
        raise SystemExit("--prompts-only builds prompts and --reuse-* consume them; pass one. Nothing was built.")
    if args.reuse_prompts_from:
        memory_model = source_memory_model(Path(args.reuse_prompts_from).expanduser().resolve())
    else:
        memory_model = served_model(args.memory_llm_url)
    print(f"memory side served by {memory_model or 'the 9B (source predates the record)'}", flush=True)

    tier_slug = f"lme_v2_{args.tier}"
    memory_config = {
        "memory_type": "myelin",
        "memory_params": {
            "url": args.mcp_url,
            "tenant": args.tenant or f"{tier_slug}/{args.domain}",
            "namespace": args.namespace or tier_slug,
            "k": args.k,
            "budget_tokens": args.budget_tokens,
            "prefetch_limit": args.prefetch_limit,
            "rerank_depth": args.rerank_depth,
            "tau_abstain": args.tau_abstain,
            "mode": args.mode,
            "max_steps": args.max_steps,
            "timeout": args.mcp_timeout_seconds,
            # Written unconditionally as booleans, never omitted: these are
            # in `standing.rs::PAIR_KEYS`, and a key that appears only when
            # the flag is set would give one arm a `null` and the other a
            # `false` for the same configuration, splitting a pair over a
            # schema difference rather than an operating-point difference.
            "select": args.select,
            "dated": args.dated,
            # M23 Phase A. Same rule: unconditional booleans, query-time
            # switches the server applies per call.
            "pool_rerank": args.pool_rerank,
            "premise": args.premise,
            "typed_probes": args.typed_probes,
            # M24. An integer or null, written unconditionally for the same
            # reason: it is in `PAIR_KEYS`, and an artifact that omits it is
            # `stale-config` and unpublishable.
            "decompose": args.decompose,
            # Not an operating-point SWITCH but part of the operating point:
            # which store was read. See `store_fingerprint`.
            "store_fingerprint": store_fingerprint(args.ledger),
            # M35, unconditional for the reason `select` is: after a default
            # moves, an omitted key stops meaning "off" and the artifact
            # would misdescribe its own run.
            "kind_quota": args.kind_quota,
            "answerability_gate": args.answerability_gate,
            # M47, unconditional for the same reason.
            "premise_check": args.premise_check,
            # M43's shipped digest, unconditional: absence reads as pre-M43.
            "item_digest": args.item_digest,
            "digest_dates": args.digest_dates,
            # M52, unconditional for the reason `select` is. Not a memory
            # switch, but part of the operating point `standing` pairs on.
            "reader_thinking": args.reader_enable_thinking,
            # M55, unconditional. Which model built the memory (select,
            # digest) and which one read it, each asked of its server. A
            # replayed memory keeps the model that built it; a prompts-only
            # run reads nothing, so it records no reader.
            "memory_llm_served_model": memory_model,
            "reader_served_model": None if args.prompts_only else served_model(args.reader_base_url),
        },
    }
    memory_config_path = runtime_dir / "memory_config.json"
    write_json(memory_config_path, memory_config)

    harness_argv = [
        "evaluation.harness",
        "--domain", args.domain,
        "--questions-path", str(runtime_dir / "questions.json"),
        "--haystack-path", str(runtime_dir / "haystack.json"),
        "--trajectories-path", str(data_root / "trajectories.jsonl"),
        "--memory-config-path", str(memory_config_path),
        "--output-dir", str(output_dir),
        "--model", args.reader_model,
        "--base-url", args.reader_base_url,
        "--api-key-env", args.reader_api_key_env,
        "--temperature", str(args.reader_temperature),
        "--top-p", str(args.reader_top_p),
        "--top-k", str(args.reader_top_k),
        "--max-completion-tokens", str(args.max_completion_tokens),
        "--memory-context-max-tokens", str(args.memory_context_max_tokens),
        "--reader-max-concurrent-requests", str(args.reader_max_concurrent_requests),
        "--prompt-build-max-workers", str(args.prompt_build_max_workers),
        "--evaluator-model", args.evaluator_model,
        "--evaluator-api-key-env", args.evaluator_api_key_env,
        "--evaluator-reasoning-effort", args.evaluator_reasoning_effort,
        "--evaluator-max-completion-tokens", str(args.evaluator_max_completion_tokens),
    ]
    if args.evaluator_base_url:
        harness_argv.extend(["--evaluator-base-url", args.evaluator_base_url])
    if not args.reader_enable_thinking:
        harness_argv.append("--reader-disable-thinking")
    if args.shuffle_questions_seed is not None:
        harness_argv.extend(["--shuffle-questions-seed", str(args.shuffle_questions_seed)])

    from evaluation import harness as harness_module  # noqa: E402
    from evaluation import qa_eval_metrics  # noqa: E402

    # Set on the modules rather than in the vendored files: `PLAN.md` §3.3
    # requires the harness, its scorers and its leaderboard builders to run
    # unmodified, and a retry budget edited into vendor code is a diff that
    # silently travels with every future run. Both modules build their own
    # client, so both have to be told.
    harness_module.OPENAI_MAX_RETRIES = args.openai_max_retries
    qa_eval_metrics.OPENAI_MAX_RETRIES = args.openai_max_retries

    if args.prompts_only:
        async def stop_before_reader(_args, _rows):
            raise PromptsBuilt()

        harness_module.generate_all_reader_outputs = stop_before_reader
    elif args.reader_enable_thinking:
        preflight_reader_thinking(args)
        enable_harness_thinking(harness_module)
    if args.reuse_prompts_from:
        reuse_prompts_from(
            Path(args.reuse_prompts_from).expanduser().resolve(), selected_questions, runtime_dir, harness_module
        )
    if args.reuse_responses_from:
        if args.reuse_responses_below_tokens is None:
            raise SystemExit("--reuse-responses-from needs --reuse-responses-below-tokens")
        reuse_responses_from(
            Path(args.reuse_responses_from).expanduser().resolve(),
            args.reuse_responses_below_tokens,
            runtime_dir,
            harness_module,
        )
    if not args.prompts_only:
        preflight_reader_images(args, selected_questions, harness_module)

    harness_main = harness_module.main

    old_argv = sys.argv
    try:
        sys.argv = harness_argv
        harness_main()
    except PromptsBuilt:
        print(f"prompts built and saved to {output_dir}; stopped before the reader (--prompts-only)", flush=True)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
