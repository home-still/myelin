"""MCP protocol conformance for `myelin-mcp` (`PLAN.md` M10).

**Uses the official `mcp` Python SDK on purpose.** A hand-rolled client
exercises the messages *we* thought to send; the SDK validates every
response against the published schema. That distinction is not theoretical:
the first run of this file rejected three of our ten tools with

    ListToolsResult: tools.3.outputSchema.type
    Input should be 'object' [input_value='array']

because `search`, `neighbors` and `review_quarantine` returned bare JSON
arrays. MCP requires every tool's `outputSchema.type` to be `"object"`. The
hand-rolled client in `myelin-eval/adapters/myelin.py` had been talking to
that server happily for an hour.

Run against a live server:

    myelin-mcp --serve 127.0.0.1:7446 --collection myelin_locomo \\
        --ledger data/locomo.ledger &
    MYELIN_MCP_URL=http://127.0.0.1:7446/mcp \\
        .venv/bin/python -m unittest \\
        crates.myelin-mcp.conformance.handshake -v

`MYELIN_MCP_TENANT` selects a tenant that exists in the served collection;
the read assertions need real data behind them, because a server that
returns nothing passes every shape check.
"""

from __future__ import annotations

import asyncio
import os
import unittest

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = os.getenv("MYELIN_MCP_URL", "http://127.0.0.1:7446/mcp")
TENANT = os.getenv("MYELIN_MCP_TENANT", "locomo/conv-30")
NAMESPACE = os.getenv("MYELIN_MCP_NAMESPACE", "locomo")

# `PLAN.md` §8: two write tools, five read tools, three admin tools.
EXPECTED_TOOLS = {
    "remember",
    "observe",
    "recall",
    "investigate",
    "search",
    "profile",
    "neighbors",
    "forget",
    "review_quarantine",
    "explain",
}


def run(coro):
    return asyncio.run(coro)


class McpConformanceTest(unittest.TestCase):
    def test_handshake_and_tool_surface(self) -> None:
        async def go():
            async with streamable_http_client(URL) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    init = await session.initialize()
                    tools = await session.list_tools()
                    return init, tools

        init, tools = run(go())
        self.assertEqual(init.server_info.name, "myelin-mcp")
        self.assertIsNotNone(
            init.capabilities.tools,
            "a server that advertises no tools capability cannot be driven by an agent",
        )
        self.assertEqual({t.name for t in tools.tools}, EXPECTED_TOOLS)

    def test_every_tool_declares_an_object_output_schema(self) -> None:
        """The check the SDK enforces, asserted here so a regression names itself.

        Without this the failure surfaces as a pydantic `ValidationError`
        inside an anyio `ExceptionGroup` during `list_tools`, which says
        nothing about which tool is wrong.
        """

        async def go():
            async with streamable_http_client(URL) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    return await session.list_tools()

        tools = run(go())
        for tool in tools.tools:
            with self.subTest(tool=tool.name):
                self.assertEqual(
                    (tool.output_schema or {}).get("type"),
                    "object",
                    f"{tool.name} must return a JSON object; MCP rejects a bare array",
                )
                self.assertEqual(tool.input_schema.get("type"), "object")

    def test_recall_returns_the_r1_wire_shape_over_the_protocol(self) -> None:
        async def go():
            async with streamable_http_client(URL) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    return await session.call_tool(
                        "recall",
                        {
                            "query": "What does this person do for work?",
                            "tenant": TENANT,
                            "namespace": NAMESPACE,
                            "k": 3,
                        },
                    )

        res = run(go())
        self.assertFalse(res.is_error, res.content)
        items = (res.structured_content or {}).get("items")
        self.assertTrue(items, f"no evidence for tenant {TENANT!r}; is the memory built?")
        self.assertLessEqual(len(items), 3, "recall must never exceed k")
        for item in items:
            self.assertEqual(set(item), {"type", "value"}, "R1 wire shape")

    def test_explain_returns_a_lineage_tree(self) -> None:
        """M10's exit criterion, and I4 made usable.

        A semantic record that cannot name the episode it came from is
        exactly the ungrounded memory `PLAN.md` §7.3 is built to avoid.
        """

        async def go():
            async with streamable_http_client(URL) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    found = await session.call_tool(
                        "search", {"tenant": TENANT, "namespace": NAMESPACE, "limit": 25}
                    )
                    records = (found.structured_content or {}).get("records", [])
                    semantic = next(r for r in records if r["kind"] == "semantic")
                    explained = await session.call_tool(
                        "explain", {"record_id": semantic["id"]}
                    )
                    return semantic, explained

        semantic, explained = run(go())
        payload = explained.structured_content or {}
        lineage = payload["lineage"]
        self.assertEqual(lineage["id"], semantic["id"])
        self.assertTrue(
            lineage["ancestors"],
            "a semantic record with no ancestors violates I4",
        )
        self.assertEqual(lineage["ancestors"][0]["kind"], "episodic")
        self.assertTrue(payload["events"], "every record must carry audit history (C10)")

    def test_hard_forget_refuses_without_confirmation(self) -> None:
        """C11 erases descendants too, so the caller rarely knows the blast radius."""

        async def go():
            async with streamable_http_client(URL) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    found = await session.call_tool(
                        "search", {"tenant": TENANT, "namespace": NAMESPACE, "limit": 1}
                    )
                    record = (found.structured_content or {}).get("records", [])[0]
                    try:
                        return await session.call_tool(
                            "forget", {"record_id": record["id"], "mode": "hard"}
                        )
                    except Exception as exc:  # noqa: BLE001 - the SDK raises on -32602
                        return exc

        outcome = run(go())
        text = str(getattr(outcome, "content", outcome))
        self.assertIn("confirm", text.lower(), f"unconfirmed hard delete was not refused: {text}")


if __name__ == "__main__":
    unittest.main()
