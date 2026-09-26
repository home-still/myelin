"""Local shim between the Codex CLI and llama.cpp's /v1/responses (M54).

Codex sends `instructions` plus extra `developer`/`system` messages inside
`input`; llama.cpp renders each as a system message, and Qwen3.5's chat
template raises "System message must be at the beginning" on any system
message after the first. This folds every such message into `instructions`
(order preserved) and forwards the request unchanged otherwise, streaming the
response back byte for byte.

Second rewrite (M54, measured 2026-09-23): Codex's `view_image` tool returns
its image INSIDE the tool output (`function_call_output.output` as a list with
an `input_image` part), and llama.cpp's /v1/responses refuses any tool output
that is not text ("Output of tool call should be 'Input text'", HTTP 400) —
5 of the first 9 pilot questions died on it. The M54 controller is text-only
by pre-registration (evidence `axtree`), so an image in a tool output becomes
a text note saying it was not shown; text parts pass through verbatim.
Nothing else is rewritten.

Cloud upstream (M54 amendment 4, 2026-09-25): when the controller is served
by OpenRouter rather than our own llama-server, the settings llama-server
carried as its defaults have to travel with each request instead. Four
optional environment variables, all or none:
  SHIM_UPSTREAM_KEY_FILE  a file holding one line OPENROUTER_API_KEY=<key>;
                          the key replaces the client's Authorization header
                          and is never printed or logged
  SHIM_MODEL              the upstream model id, replacing Codex's alias
  SHIM_PROVIDER           JSON provider routing, e.g. one pinned provider
                          with no fallbacks
  SHIM_SAMPLING           JSON fields merged into every /responses body: the
                          pre-registered temperature, top_p and top_k
usage: responses_shim.py <listen_port> <upstream_base e.g. http://127.0.0.1:5810>"""
import json, os, re, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT, UP = int(sys.argv[1]), sys.argv[2].rstrip("/")
CHUNK = 4096
CLOUD_VARS = ("SHIM_UPSTREAM_KEY_FILE", "SHIM_MODEL", "SHIM_PROVIDER", "SHIM_SAMPLING")
_set = [v for v in CLOUD_VARS if os.environ.get(v)]
if _set and len(_set) != len(CLOUD_VARS):
    sys.exit(f"responses_shim: a cloud upstream needs all of {CLOUD_VARS}; got only {_set}")
CLOUD = bool(_set)
if CLOUD:
    _lines = [l.strip() for l in open(os.path.expanduser(os.environ["SHIM_UPSTREAM_KEY_FILE"]), encoding="utf-8") if l.strip()]
    _m = re.match(r"^OPENROUTER_API_KEY=(\S+)$", _lines[0]) if len(_lines) == 1 else None
    if not _m:
        sys.exit("responses_shim: the key file must hold exactly one line OPENROUTER_API_KEY=<key>")
    UP_KEY = _m.group(1)
    UP_MODEL = os.environ["SHIM_MODEL"]
    UP_PROVIDER = json.loads(os.environ["SHIM_PROVIDER"])
    UP_SAMPLING = json.loads(os.environ["SHIM_SAMPLING"])
def text_of(item):
    c = item.get("content")
    if isinstance(c, str): return c
    return "\n".join(p.get("text", "") for p in (c or []) if isinstance(p, dict))
IMAGE_NOTE = "[image not shown: this controller reads text only]"
def text_output(out):
    if isinstance(out, str): return out
    parts = []
    for p in out or []:
        if not isinstance(p, dict): continue
        parts.append(p.get("text", "") if p.get("type") in ("input_text", "output_text", "text") else IMAGE_NOTE)
    return "\n".join(parts)
def fold(body):
    inp = body.get("input")
    if not isinstance(inp, list): return body
    for it in inp:
        if isinstance(it, dict) and it.get("type") in ("function_call_output", "custom_tool_call_output"):
            it["output"] = text_output(it.get("output"))
    extra, keep = [], []
    for it in inp:
        if isinstance(it, dict) and it.get("role") in ("developer", "system"):
            extra.append(text_of(it))
        else:
            keep.append(it)
    if extra:
        body["instructions"] = "\n\n".join([body.get("instructions") or ""] + extra).strip()
        body["input"] = keep
    return body
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _forward(self, data):
        # Optional timing trace (SHIM_TRACE_LOG): when, how long to the first
        # byte and to the end, the status and the byte count of each request.
        # Never the content, never the key.
        import time as _t
        trace, t0, first, nbytes = os.environ.get("SHIM_TRACE_LOG"), _t.time(), None, 0
        req = urllib.request.Request(UP + self.path, data=data, method=self.command)
        for k in ("Content-Type", "Authorization", "Accept"):
            if self.headers.get(k): req.add_header(k, self.headers[k])
        if CLOUD:
            req.add_header("Authorization", f"Bearer {UP_KEY}")
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            resp = e
            # Keep every refused request for diagnosis (M54: llama.cpp 400s).
            d = os.environ.get("SHIM_REJECT_DIR")
            if d and data:
                os.makedirs(d, exist_ok=True)
                open(f"{d}/{_t.time():.3f}-{e.code}.json", "wb").write(data)
        self.send_response(resp.status if hasattr(resp, "status") else resp.code)
        for k, v in resp.headers.items():
            if k.lower() in ("content-length", "transfer-encoding", "connection"): continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        while True:
            b = resp.read1(CHUNK) if hasattr(resp, "read1") else resp.read(CHUNK)
            if not b: break
            if first is None: first = _t.time() - t0
            nbytes += len(b)
            self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n"); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        if trace:
            status = resp.status if hasattr(resp, "status") else resp.code
            with open(trace, "a") as fh:
                fh.write(json.dumps({"t0": round(t0, 3), "path": self.path, "status": status,
                                     "first_byte_s": None if first is None else round(first, 2),
                                     "total_s": round(_t.time() - t0, 2), "bytes": nbytes,
                                     "dropped_tools": getattr(self, "_dropped_tools", [])}) + "\n")
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/responses"):
            body = fold(json.loads(raw))
            if CLOUD:
                body["model"] = UP_MODEL
                body["provider"] = UP_PROVIDER
                body.update(UP_SAMPLING)
                # Only function tools, as our llama-server offers the model. Codex also sends
                # hosted tools (`web_search`), which llama.cpp never exposed but OpenRouter
                # executes server-side as `openrouter:web_search`: on a memory benchmark that
                # is outside knowledge. Found 2026-09-25 (M54 amendment 4b); every cloud chunk
                # run before this was discarded.
                tools = body.get("tools") or []
                body["tools"] = [t for t in tools if isinstance(t, dict) and t.get("type") == "function"]
                self._dropped_tools = sorted({str(t.get("type")) for t in tools
                                              if not (isinstance(t, dict) and t.get("type") == "function")})
            raw = json.dumps(body).encode()
        self._forward(raw)
    def do_GET(self): self._forward(None)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
