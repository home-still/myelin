"""`myelin` as a LongMemEval-V2 memory backend (`PLAN.md` §3.3, R1, R6).

A pure forwarder. Every retrieval decision — fusion, reranking, budgeting,
ordering — lives in `myelin-core` and is exercised identically by
`myelin-eval ablate`; this file only moves bytes. That is deliberate: a
benchmark adapter that contains logic is a second implementation, and the
number it produces then belongs to the adapter rather than to the system.

Two contracts it must not break:

* **R1 — wire shape.** `query()` returns `list[{"type": ..., "value": ...}]`
  and nothing else. The server already emits exactly that (`EvidenceSet::
  to_wire`), so this file passes it through untouched rather than rebuilding
  it. Record ids, scores and provenance travel in sibling fields and are
  dropped here.
* **R6 — query privacy.** `query()` receives only the question text and an
  optional image path. The harness's `get_query_context()` carries a single
  opaque `query_invocation_id` and this adapter never reads it, never logs
  it, and never forwards it. A memory backend that can see the benchmark's
  question id, category or gold answer is disqualified, and the harness ships
  `tests/test_query_privacy.py` to check precisely that.

Ingest is *not* done through this class. `insert()` is called by the harness
once per trajectory, but building a myelin memory means running the Rust
write path (segmentation, embedding, Qdrant upserts, ledger deltas), which is
`myelin-eval build --corpus lme-v2-small`. So `insert()` verifies that the
trajectory it is handed is already present in the memory being served and
raises if it is not — silently accepting it would produce a run whose memory
does not contain the haystack it claims to.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from memory_modules.memory import (
    Memory,
    MemoryContextItem,
    register_memory,
    require,
)

_JSON = "application/json"


def _decode(raw: bytes, content_type: str) -> dict[str, Any] | None:
    """Parse a streamable-HTTP body, which may be JSON or a single SSE event.

    The transport picks per response, not per server: `myelin-mcp` runs with
    `json_response`, yet rmcp still answers `initialize` as `text/event-stream`.
    Rather than depend on a server flag, decode what the spec allows — a JSON
    object, or `data:` lines carrying one.
    """
    text = raw.decode("utf-8", "replace")
    if "text/event-stream" not in content_type:
        return _loads(text)
    payload = "".join(
        line[len("data:") :].strip()
        for line in text.splitlines()
        if line.startswith("data:")
    )
    return _loads(payload) if payload else None


def _loads(text: str) -> dict[str, Any]:
    """`json.loads` that names the body it could not parse.

    A bare `JSONDecodeError` says "Expecting value: line 1 column 1" and
    nothing else, which is indistinguishable between an HTML error page, a
    proxy timeout and a truncated stream. The first 200 characters of the
    body identify all three on sight.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise McpError(
            f"myelin-mcp returned a body that is not JSON ({exc}): {text[:200]!r}"
        ) from exc


class McpError(RuntimeError):
    """A transport or tool-level failure talking to `myelin-mcp`."""


class _McpSession:
    """Minimal streamable-HTTP MCP client.

    Deliberately not the `mcp` Python SDK: the SDK is asyncio-first, and the
    harness calls `query()` from a synchronous `ThreadPoolExecutor`. Bridging
    an event loop per call costs more than the three JSON-RPC messages this
    needs, and an `asyncio.run` inside a worker thread is a known source of
    "attached to a different loop" failures under the harness's own
    concurrency (R5).

    One session is initialized per instance and reused; the session id is
    returned by the server on `initialize` and echoed on every later call.
    """

    PROTOCOL_VERSION = "2025-11-25"

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._next_id = 0
        # Guards session re-establishment. `_epoch` increments once per
        # successful `initialize`, so four harness workers that all notice the
        # same dead session re-create it once between them instead of four
        # times.
        self._init_lock = threading.Lock()
        self._epoch = 0

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, str]]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": _JSON,
            # Streamable HTTP requires the client to advertise both, even
            # when the server is configured for plain JSON responses.
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(self.url, body, headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                received = {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as exc:  # pragma: no cover - server bug
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise McpError(f"{self.url}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise McpError(f"{self.url}: {exc.reason}") from exc
        if not raw.strip():
            return None, received
        return _decode(raw, received.get("content-type", "")), received

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
        message, _ = self._post(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        if message is None:
            raise McpError(f"{method}: empty response body")
        if "error" in message:
            raise McpError(f"{method}: {message['error']}")
        return message.get("result", {})

    def initialize(self) -> None:
        with self._init_lock:
            self._initialize_locked()

    def _initialize_locked(self) -> None:
        self._session_id = None
        message, headers = self._post(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "myelin-lmev2-adapter", "version": "0.1.0"},
                },
            }
        )
        if message is None or "result" not in message:
            raise McpError(f"initialize failed: {message}")
        self._session_id = headers.get("mcp-session-id")
        # `notifications/initialized` has no id and no response body; the
        # server refuses tool calls until it has been sent.
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        self._epoch += 1

    # The server may evict a session out from under us, and retrying a dead
    # one can only fail. Measured: the harness spends >5 minutes loading its
    # 1.1 GB `trajectories.jsonl` between this adapter's construction-time
    # liveness probe and the first prompt-building query, and `rmcp`'s
    # `LocalSessionManager` drops the idle session in that window — every
    # later call then returns `HTTP 404: Session not found` and a
    # 240-question arm dies four minutes in. A client that cannot re-establish
    # its own session is not resilient, whatever the eviction policy is.
    SESSION_LOST = ("session not found", "session error", "http 404")

    def _recover_session(self, seen_epoch: int) -> bool:
        """Re-establish the session once per loss, across all worker threads."""
        with self._init_lock:
            if self._epoch != seen_epoch:
                return True  # another worker already rebuilt it
            try:
                self._initialize_locked()
            except McpError as exc:
                print(
                    f"myelin adapter: session re-initialize failed ({exc})",
                    file=sys.stderr,
                    flush=True,
                )
                return False
        print(
            f"myelin adapter: server dropped the session; re-initialized as "
            f"{self._session_id}",
            file=sys.stderr,
            flush=True,
        )
        return True

    # Reads are idempotent, so a failed `tools/call` is retried a bounded
    # number of times. This is harness plumbing, not a change to the system
    # under test: `recall`/`investigate` are pure functions of the store and
    # the query, and nothing here is retried that succeeded.
    #
    # It exists because the reader and reranker on `big` are reachable only
    # through an SSH tunnel (the host firewalls those ports), and one
    # `Connection reset by peer` on that tunnel killed a 240-question run at
    # question 97d5309a after 14 minutes — `harness.py` raises
    # `Prompt building failed for question ...` on the first exception. The
    # retry is logged to stderr so a run's log shows how often it fired; a
    # retried question's `memory_query_duration_seconds` includes the wait.
    # Six attempts with capped exponential backoff — about three minutes.
    # Sized against the observed outage, not against a round number: another
    # tenant starting vLLM drove `big` to load 151, sshd stopped answering,
    # and the 5810/5813 tunnel was down for five minutes. Three attempts over
    # six seconds cannot cross that; a two-hour arm dying at minute 35 costs
    # far more than a few wasted retries.
    CALL_ATTEMPTS = 6
    CALL_BACKOFF_SECONDS = 4.0
    CALL_BACKOFF_CAP_SECONDS = 60.0

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        last: McpError | None = None
        for attempt in range(1, self.CALL_ATTEMPTS + 1):
            epoch = self._epoch
            try:
                return self._call_tool_once(name, arguments)
            except McpError as exc:
                last = exc
                if attempt == self.CALL_ATTEMPTS:
                    break
                # A dropped session is not a transient network fault: waiting
                # cannot fix it, and re-establishing it costs one round trip.
                text = str(exc).lower()
                if any(marker in text for marker in self.SESSION_LOST):
                    if self._recover_session(epoch):
                        continue
                delay = min(
                    self.CALL_BACKOFF_SECONDS * 2 ** (attempt - 1),
                    self.CALL_BACKOFF_CAP_SECONDS,
                )
                print(
                    f"myelin adapter: {name} attempt {attempt}/{self.CALL_ATTEMPTS} "
                    f"failed ({exc}); retrying in {delay:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        raise McpError(f"{name}: {self.CALL_ATTEMPTS} attempts failed; last: {last}")

    def _call_tool_once(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise McpError(f"{name}: {result.get('content')}")
        structured = result.get("structuredContent")
        if structured is None:
            raise McpError(f"{name}: no structuredContent in {result}")
        return structured

    def list_tools(self) -> list[str]:
        return [t["name"] for t in self._rpc("tools/list", {}).get("tools", [])]


@register_memory
class MyelinMemory(Memory):
    """Forwards `query()` to the `recall` tool of a running `myelin-mcp`.

    `memory_params`:

    | key | default | meaning |
    |---|---|---|
    | `url` | `$MYELIN_MCP_URL` or `http://127.0.0.1:7446/mcp` | server endpoint |
    | `tenant` | required | which memory to read; C12 forbids spanning tenants |
    | `namespace` | `None` | extra scope predicate |
    | `k` | `6` | evidence-set size |
    | `budget_tokens` | `2048` | compose budget |
    | `mode` | `recall` | `recall` (fast) or `investigate` (agentic) |
    | `max_steps` | `2` | `investigate` only: iteration cap |
    | `tau_abstain` | `None` | withhold evidence below this rerank score |
    | `timeout` | `120.0` | per-call seconds |
    | `prefetch_limit` | `None` | recorded only; set on the server's CLI |
    | `rerank_depth` | `None` | recorded only; set on the server's CLI |

    `tenant` is required and has no default on purpose. The whole point of a
    per-domain build is that a `web` question must not be answered from the
    `enterprise` memory, and a defaulted tenant is how that silently happens.

    `prefetch_limit` and `rerank_depth` are read by nothing here. They are
    carried in the manifest so a run directory names the candidate-pool width
    the server was started with, since `recall` takes no such argument.
    """

    memory_type = "myelin"

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)
        params = self.memory_params
        tenant = params.get("tenant")
        require(
            isinstance(tenant, str) and tenant.strip(),
            "myelin memory_params requires a non-empty 'tenant'",
        )
        self.tenant: str = str(tenant)
        self.namespace = params.get("namespace")
        self.k = int(params.get("k", 6))
        self.budget_tokens = int(params.get("budget_tokens", 2048))
        # Abstention gate. None keeps the server's default (off).
        tau = params.get("tau_abstain")
        self.tau_abstain = None if tau is None else float(tau)
        # Operating-point switches (M22). `select` defaults off; `dated` is
        # kept as None when absent so a config written before M22 still means
        # "whatever the server is configured for" rather than asserting a
        # corpus property the run never decided.
        self.select = bool(params.get("select", False))
        self.dated = params.get("dated")
        # M23 Phase A. Same contract as `dated`: None when absent means
        # "whatever the server is configured for", not an asserted value.
        # investigate-only switches; `recall` takes neither.
        self.pool_rerank = params.get("pool_rerank")
        self.premise = params.get("premise")
        # R4: the mode is a query-time parameter against one identical
        # store, which is the whole reason a leaderboard submission can
        # present two operating points from one built memory.
        # M23 D2. `None` means probes are untagged (whole store); a bool on
        # `investigate` lets the reflect gate aim each probe at the event
        # (`RecordKind::Semantic`) or note (`RecordKind::Procedural`) pool.
        self.typed_probes = params.get("typed_probes")
        # M24. An integer cap or None; passed on both tools, because unlike
        # the M23 switches this one changes the candidate pool rather than
        # the loop, and `recall` has a pool too.
        self.decompose = params.get("decompose")
        # M35: allocate `k`'s slots per record kind (AgentRunbook-R's top-6
        # events / top-3 notes) instead of one fused ranking. `None` means
        # "do not override the server's default", which is the contract every
        # switch here has and the one M33's `select` defect violated.
        self.kind_quota = params.get("kind_quota")
        # The last query's retrieval trace, per worker thread.
        #
        # THREAD-LOCAL, not an attribute, for the reason the base class's own
        # `_query_context_local` is: `harness.py` builds prompts across four
        # worker threads against one shared `Memory`, so a plain
        # `self._last_trace` would let one question's trace be reported
        # against another's — a silent mis-attribution, which is worse than
        # no instrument at all.
        self._trace_local = threading.local()
        self.mode = str(params.get("mode", "recall"))
        require(
            self.mode in {"recall", "investigate"},
            f"mode must be 'recall' or 'investigate', got {self.mode!r}",
        )
        # 2 is the measured peak; see docs/measurements/m7-step-value-curve.md.
        self.max_steps = int(params.get("max_steps", 2))
        url = params.get("url") or os.getenv("MYELIN_MCP_URL") or "http://127.0.0.1:7446/mcp"
        self.url = str(url)
        self._session = _McpSession(self.url, float(params.get("timeout", 120.0)))
        self._session.initialize()
        tools = self._session.list_tools()
        require(
            self.mode in tools,
            f"{self.url} exposes no {self.mode!r} tool (got {tools})",
        )
        self._inserted: set[str] = set()

    def insert(self, trajectory: dict[str, object]) -> None:
        """Verify the trajectory's memory is already built; never index here.

        The harness calls this once per haystack trajectory. Indexing through
        it would mean running the Rust write path over ~43M tokens inside the
        evaluation loop; the memory is built beforehand by `myelin-eval
        build`. But accepting the call and doing nothing is worse than
        either, because an unbuilt or mis-tenanted memory would then score as
        though it were complete — every question answered from an empty
        evidence set, reported as a legitimate accuracy number.

        So the first call probes the served tenant with the trajectory's own
        goal text and requires a non-empty result. That is exactly the
        failure mode worth catching: forgot to build, wrong collection, wrong
        tenant, server pointed at a different ledger. The goal comes from the
        haystack, never from the question, so this costs nothing under R6.
        """
        trajectory_id = trajectory.get("id")
        require(isinstance(trajectory_id, str), "trajectory has no string id")
        if not self._inserted:
            goal = trajectory.get("goal")
            probe = goal if isinstance(goal, str) and goal.strip() else "memory"
            # Always `recall` here regardless of mode: this is a liveness
            # check on the store, and paying for an agentic loop to answer
            # it would cost seconds per run for no extra information.
            hits = self._session.call_tool(
                "recall",
                {"query": probe, "tenant": self.tenant, "k": 1, "budget_tokens": 256},
            )
            require(
                bool(hits.get("items")),
                f"tenant {self.tenant!r} at {self.url} returned nothing for the first "
                "haystack trajectory: the memory is not built, or the server is "
                "serving a different collection/ledger. Run `myelin-eval build` first.",
            )
        self._inserted.add(str(trajectory_id))

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        # `investigate` names its question field differently: the tool takes
        # a question to reason about, not a search string to match.
        arguments: dict[str, Any] = {
            "tenant": self.tenant,
            "k": self.k,
            "budget_tokens": self.budget_tokens,
        }
        if self.mode == "investigate":
            arguments["question"] = query
            arguments["max_steps"] = self.max_steps
        else:
            arguments["query"] = query
        if self.namespace:
            arguments["namespace"] = self.namespace
        if self.tau_abstain is not None:
            arguments["tau_abstain"] = self.tau_abstain
        # Sent UNCONDITIONALLY, unlike the switches below. M32 made
        # `select_sufficient` the `investigate` default, so an omitted key no
        # longer means "off" — the server would turn the selector ON while
        # `memory_config.json` recorded `select: false`, and the artifact
        # would misdescribe the run it came from. `standing` reads that key
        # to decide whether a run is an arm, so the lie would decide which
        # number gets published.
        arguments["select"] = bool(self.select)
        if self.dated is not None:
            arguments["dated"] = bool(self.dated)
        if self.kind_quota is not None:
            arguments["kind_quota"] = bool(self.kind_quota)
        # M24 is declared on both schemas, so it needs no mode branch.
        if self.decompose is not None:
            arguments["decompose"] = int(self.decompose)
        # `pool_rerank`, `premise`, and `typed_probes` exist on `investigate`
        # alone; forwarding a parameter the `recall` schema does not declare is
        # an invalid-params error, not a silent no-op.
        if self.mode == "investigate":
            if self.pool_rerank is not None:
                arguments["pool_rerank"] = bool(self.pool_rerank)
            if self.premise is not None:
                arguments["premise"] = bool(self.premise)
            if self.typed_probes is not None:
                arguments["typed_probes"] = bool(self.typed_probes)
        # `query_image` is accepted and ignored for now: the dense channel is
        # text-only (bge-m3), so forwarding a path the server cannot embed
        # would be a lie in the trace. 29 of 451 questions carry one; they are
        # answered from text evidence like any other.
        result = self._session.call_tool(self.mode, arguments)
        self._trace_local.trace = result.get("trace") or {}
        items = result.get("items", [])
        require(isinstance(items, list), f"recall returned non-list items: {items!r}")
        return items

    def post_query_hook(
        self,
        *,
        query: str,
        query_image: str | None,
        memory_context: list[MemoryContextItem],
    ) -> dict[str, object] | None:
        """Report the retrieval trace for the question just queried.

        `harness.py` calls this immediately after `query()` on the same
        thread and writes the result to `per_question.jsonl` as
        `memory_post_query_metadata` — **keyed to the question id by the
        harness itself**. That is the whole reason this replaced M33's
        sidecar file: a sidecar has no question identifier, and prompts are
        built across four worker threads, so its line order is not the
        question order and the rows cannot be joined to outcomes at all.
        M33 shipped one and could only report aggregate counts.

        Why the trace is worth reporting: the fallback in
        `Selector::select` is `0..k`, so a selecting arm whose calls fail
        emits the *unselected* arm's evidence set, scores what it scores,
        and reads as a clean null for a mechanism that never ran. M22's
        LME-V2 selector arm ran against a reader serving 4,096 tokens per
        slot, where a 60-candidate selector prompt does not fit, and nothing
        in its artifact can say whether the mechanism ran.

        Returns `None` when the server sent no trace, which is the base
        class's own "nothing to report" and keeps the key `null` rather than
        an empty object that would read as a measurement.
        """
        trace = getattr(self._trace_local, "trace", None)
        if not trace:
            return None
        return {
            "selected": trace.get("selected"),
            "select_ms": trace.get("select_ms"),
            "select_degraded": trace.get("select_degraded"),
            "pool": trace.get("pool"),
            "steps": trace.get("steps"),
            "stopped_because": trace.get("stopped_because"),
            "abstained": trace.get("abstained"),
        }

    def clear_query_context(self) -> None:
        """Drop this thread's trace along with the base class's context.

        `harness.py` calls this in a `finally`, so a query that raised must
        not leave its trace behind for the next question on this thread to
        report as its own.
        """
        super().clear_query_context()
        if hasattr(self._trace_local, "trace"):
            del self._trace_local.trace
