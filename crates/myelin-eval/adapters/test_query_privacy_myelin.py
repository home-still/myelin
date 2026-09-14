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
        }


def build_memory() -> tuple[myelin.MyelinMemory, RecordingSession]:
    recorder = RecordingSession()
    memory = object.__new__(myelin.MyelinMemory)
    # `Memory.__init__` sets up the thread-local query context the harness
    # uses; skipping it would make the test vacuous.
    myelin.Memory.__init__(memory, {"tenant": "t/privacy"})
    memory.tenant = "t/privacy"
    memory.namespace = None
    memory.k = 6
    memory.budget_tokens = 2048
    memory.tau_abstain = None
    memory.mode = "recall"
    memory.max_steps = 4
    memory.url = "recorder://"
    memory._session = recorder
    memory._inserted = set()
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
            {"query", "tenant", "k", "budget_tokens"},
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


if __name__ == "__main__":
    unittest.main()
