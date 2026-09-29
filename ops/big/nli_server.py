#!/usr/bin/env python3
"""NLI cross-encoder over HTTP, for M82's premise check (docs/measurements/m82-nli-premise.md).

POST /nli   {"premise": str, "hypothesis": str}
         -> {"entailment": p, "neutral": p, "contradiction": p, "model": id}
GET  /health -> {"status": "ok", "model": id, "device": str}

The model is Laurer et al.'s DeBERTa-v3-large, trained on MNLI, FEVER-NLI,
ANLI, LingNLI and WANLI ("Less Annotating, More Classifying", Political
Analysis 2023, 10.1017/pan.2023.20). The premise is truncated, never the
hypothesis, so the statement being checked is always read whole. A malformed
request is refused with HTTP 400; nothing is guessed.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
PORT = int(os.environ.get("MYELIN_NLI_PORT", "5820"))
HOST = os.environ.get("MYELIN_NLI_HOST", "127.0.0.1")
MAX_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE).eval()
LABELS = [MODEL.config.id2label[i].lower() for i in range(MODEL.config.num_labels)]
if sorted(LABELS) != ["contradiction", "entailment", "neutral"]:
    raise SystemExit(f"{MODEL_ID} labels are {LABELS}, not the three NLI classes")
LOCK = threading.Lock()


def classify(premise: str, hypothesis: str) -> dict:
    enc = TOKENIZER(premise, hypothesis, truncation="only_first", max_length=MAX_TOKENS,
                    return_tensors="pt").to(DEVICE)
    with LOCK, torch.no_grad():
        probs = torch.softmax(MODEL(**enc).logits[0], -1).tolist()
    return {**dict(zip(LABELS, probs)), "model": MODEL_ID}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": MODEL_ID, "device": DEVICE})
        else:
            self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/nli":
            self._send(404, {"error": f"no route {self.path}"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            premise, hypothesis = body["premise"], body["hypothesis"]
            if not isinstance(premise, str) or not isinstance(hypothesis, str) or not premise or not hypothesis:
                raise ValueError("premise and hypothesis must be non-empty strings")
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"malformed request: {e}"})
            return
        self._send(200, classify(premise, hypothesis))

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
