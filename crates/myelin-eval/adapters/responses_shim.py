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
usage: responses_shim.py <listen_port> <upstream_base e.g. http://127.0.0.1:5810>"""
import json, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT, UP = int(sys.argv[1]), sys.argv[2].rstrip("/")
CHUNK = 4096
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
        req = urllib.request.Request(UP + self.path, data=data, method=self.command)
        for k in ("Content-Type", "Authorization", "Accept"):
            if self.headers.get(k): req.add_header(k, self.headers[k])
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            resp = e
            # Keep every refused request for diagnosis (M54: llama.cpp 400s).
            import os, time
            d = os.environ.get("SHIM_REJECT_DIR")
            if d and data:
                os.makedirs(d, exist_ok=True)
                open(f"{d}/{time.time():.3f}-{e.code}.json", "wb").write(data)
        self.send_response(resp.status if hasattr(resp, "status") else resp.code)
        for k, v in resp.headers.items():
            if k.lower() in ("content-length", "transfer-encoding", "connection"): continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        while True:
            b = resp.read1(CHUNK) if hasattr(resp, "read1") else resp.read(CHUNK)
            if not b: break
            self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n"); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/responses"):
            raw = json.dumps(fold(json.loads(raw))).encode()
        self._forward(raw)
    def do_GET(self): self._forward(None)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
