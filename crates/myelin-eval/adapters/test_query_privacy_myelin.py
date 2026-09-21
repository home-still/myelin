"""R6 — query privacy, for `myelin` (`PLAN.md` §3.3, R6).

The harness ships `tests/test_query_privacy.py`, and it passes in our
vendored tree (7/7). But it iterates *their* six backends; it cannot see
ours. This file applies the same machinery — their `build_prompt_row`, their
prompt-row shape, their opaque-context contract — to `MyelinMemory`, because
"their test passes" is only evidence about our adapter if our adapter is the
thing under test.

It also asserts something their test cannot: that no benchmark metadata
leaves the process. Their spy checks what the *memory* was told. Ours checks
what the memory then *sent over the wire*, which for a forwarding adapter is
where a leak would actually happen.

Run:

    PYTHONPATH=vendor/longmemeval-v2:adapters \\
        ../../.venv/bin/python -m unittest \\
        test_query_privacy_myelin -v

A live `myelin-mcp` is not required: the transport is replaced with a
recorder, because what is under test is the adapter's behaviour, not the
server's.
"""

from __future__ import annotations

import itertools
import threading
import time
import unittest
from typing import Any

from evaluation.harness import build_prompt_row

import myelin


SECRETS = {
    "question_id": "secret-question-id",
    "answer": "secret-answer",
    "eval_function": "secret-evaluator",
    "goal": "secret-original-goal",
    "category": "procedure-abs",
    "invocation": "opaque-invocation-id",
}

QUESTION = "How do I complete the task?"


def prompt_item() -> dict[str, object]:
    """The harness's own fixture, verbatim from `tests/test_query_privacy.py`.

    Copied rather than imported because importing their test module would
    also run their `unittest` discovery against their backends. The shape is
    what matters: every field here except `question_text` is something a
    memory backend must never see.
    """
    return {
        "index": 0,
        "question_item": {
            "id": SECRETS["question_id"],
            "question_type": "procedure-abs",
            "question": QUESTION,
            "answer": SECRETS["answer"],
            "eval_function": SECRETS["eval_function"],
            "metadata": {"original_goal": [SECRETS["goal"]]},
        },
        "question_id": SECRETS["question_id"],
        "query_invocation_id": SECRETS["invocation"],
        "question_type": "procedure-abs",
        "category": SECRETS["category"],
        "eval_function": SECRETS["eval_function"],
        "eval_name": "llm_abstention_checker",
        "question_text": QUESTION,
        "question_image": None,
        "answer_gold": SECRETS["answer"],
    }


class RecordingSession:
    """Stands in for `_McpSession`, capturing every outbound tool call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def initialize(self) -> None:
        pass

    def list_tools(self) -> list[str]:
        return ["recall"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {
            "items": [{"type": "text", "value": "Dana adopted a rescue dog."}],
            "record_ids": ["00000000-0000-0000-0000-000000000001"],
            "tokens": 7,
            # The trace `post_query_hook` reports. A real server always sends
            # one; `model_declined` is the benign fallback cause M32 split out
            # of `CallFailed`, and it is the value a healthy run mostly shows.
            "trace": {
                "selected": 3,
                "select_ms": 412,
                "select_degraded": "model_declined",
                "pool": 41,
                "steps": 2,
                "stopped_because": "sufficient",
                "abstained": False,
            },
        }


def build_memory() -> tuple[myelin.MyelinMemory, RecordingSession]:
    """Construct through the REAL `__init__`, with the transport swapped.

    Hand-setting attributes is what this used to do, and it drifted: M33 made
    `query()` send `select` unconditionally and these three tests started
    raising `AttributeError` for a field the constructor sets and the fixture
    did not. They are `unittest`, so `cargo test` never runs them and nothing
    in the landing checklist caught it.

    Going through `__init__` means a new operating-point field can never make
    the fixture stale again — it either works for the real thing or it works
    for neither.
    """
    recorder = RecordingSession()
    original = myelin._McpSession
    myelin._McpSession = lambda url, timeout: recorder  # noqa: ARG005
    try:
        memory = myelin.MyelinMemory({"tenant": "t/privacy", "url": "recorder://"})
    finally:
        myelin._McpSession = original
    return memory, recorder


class MyelinQueryPrivacyTest(unittest.TestCase):
    def test_adapter_receives_only_the_opaque_invocation_id(self) -> None:
        memory, _ = build_memory()
        seen: list[dict[str, str]] = []
        original = memory.query

        def spy(query: str, query_image: str | None = None):
            seen.append(memory.get_query_context())
            return original(query, query_image)

        memory.query = spy  # type: ignore[method-assign]
        build_prompt_row(
            prompt_item(),
            haystack_ids=[],
            memory=memory,
            system_prompt="Answer using memory.",
            memory_context_max_tokens=1,
        )

        self.assertEqual(seen, [{"query_invocation_id": SECRETS["invocation"]}])
        self.assertEqual(memory.get_query_context(), {})

    def test_no_benchmark_metadata_reaches_the_wire(self) -> None:
        """The claim their test cannot make: nothing secret is *sent*."""
        memory, recorder = build_memory()
        build_prompt_row(
            prompt_item(),
            haystack_ids=[],
            memory=memory,
            system_prompt="Answer using memory.",
            memory_context_max_tokens=1,
        )

        self.assertEqual(len(recorder.calls), 1, "expected exactly one recall call")
        name, arguments = recorder.calls[0]
        self.assertEqual(name, "recall")
        self.assertEqual(arguments["query"], QUESTION)

        payload = repr(arguments)
        for label, secret in SECRETS.items():
            self.assertNotIn(
                secret,
                payload,
                f"adapter leaked {label} to the server: {payload}",
            )

        self.assertEqual(
            set(arguments),
            # `select` joined this set deliberately in M33. M32 made
            # `select_sufficient` the `investigate` default, so an omitted key
            # stopped meaning "off" — the server would turn the selector on
            # while `memory_config.json` recorded `false`, and the artifact
            # would describe a run that did not happen. It carries a boolean
            # the caller already chose, so it adds no benchmark metadata.
            {"query", "tenant", "k", "budget_tokens", "select"},
            "the outbound argument set is the privacy surface; a new key here "
            "is a new opportunity to leak benchmark metadata",
        )

    def test_query_returns_the_r1_wire_shape_and_nothing_else(self) -> None:
        memory, _ = build_memory()
        items = memory.query(QUESTION)
        self.assertEqual(
            [set(item) for item in items],
            [{"type", "value"}],
            "R1: LongMemEval-V2 requires list[{type, value}] exactly; record "
            "ids and scores must not ride along",
        )


class MyelinTraceReportingTest(unittest.TestCase):
    """`post_query_hook` must report THIS question's trace, on this thread."""

    def test_the_trace_reaches_the_harness_keyed_to_its_question(self) -> None:
        memory, _ = build_memory()
        memory.query(QUESTION)
        meta = memory.post_query_hook(
            query=QUESTION, query_image=None, memory_context=[]
        )
        self.assertEqual(meta["selected"], 3)
        self.assertEqual(meta["select_degraded"], "model_declined")
        self.assertEqual(meta["pool"], 41)

    def test_a_cleared_context_does_not_report_a_stale_trace(self) -> None:
        """`harness.py` clears in a `finally`, so a query that raised must not
        leave its trace for the next question on this thread to claim."""
        memory, _ = build_memory()
        memory.query(QUESTION)
        memory.clear_query_context()
        self.assertIsNone(
            memory.post_query_hook(
                query=QUESTION, query_image=None, memory_context=[]
            )
        )

    def test_concurrent_workers_never_report_each_others_traces(self) -> None:
        """**The defect this design exists to prevent.**

        `harness.py` builds prompts across four worker threads against ONE
        shared `Memory`. A plain `self._last_trace` would let one question's
        trace be written to another question's row — a silent
        mis-attribution, which is worse than no instrument, because the join
        it enables would look valid and be wrong.

        Each thread here gets a distinct `selected` from the transport, so a
        shared attribute shows up as a thread reading a value it never
        produced.
        """
        memory, recorder = build_memory()
        n = 8
        # One distinct trace per thread, handed out in call order.
        counter = itertools.count()
        lock = threading.Lock()

        def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            with lock:
                i = next(counter)
            # Widen the window a real scheduler would give us.
            time.sleep(0.01)
            return {
                "items": [{"type": "text", "value": f"answer {i}"}],
                "record_ids": [],
                "tokens": 1,
                "trace": {"selected": i, "pool": 100 + i},
            }

        recorder.call_tool = call_tool  # type: ignore[method-assign]
        seen: list[tuple[int, int]] = []
        seen_lock = threading.Lock()

        def one_question() -> None:
            try:
                memory.query(QUESTION)
                # `harness.py` does real work between these two calls; the
                # sleep stands in for it. Without a window here a shared
                # attribute would usually survive by luck, and this test
                # would assert nothing. Verified: with `self._shared_trace`
                # in place of the thread-local, this fails.
                time.sleep(0.02)
                meta = memory.post_query_hook(
                    query=QUESTION, query_image=None, memory_context=[]
                )
                with seen_lock:
                    seen.append((meta["selected"], meta["pool"]))
            finally:
                memory.clear_query_context()

        threads = [threading.Thread(target=one_question) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(seen), n)
        for selected, pool in seen:
            self.assertEqual(
                pool,
                100 + selected,
                "a thread reported a trace it did not produce: the per-query "
                "state is shared, not thread-local",
            )
        self.assertEqual(
            sorted(s for s, _ in seen),
            list(range(n)),
            "every thread must report its own distinct trace exactly once",
        )


if __name__ == "__main__":
    unittest.main()
