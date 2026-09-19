"""Bridge to the official LAFS tool: {"tier": "small", "points": [{"name","acc","latency"}]} on
stdin, `lafs_summary_for_submission` JSON on stdout. Exists so `myelin-eval standing` never
reimplements the leaderboard's integral."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "longmemeval-v2" / "leaderboard"))
import compute_lafs as L
req = json.load(sys.stdin)
points = [L.Point(p["name"], acc=float(p["acc"]), latency=float(p["latency"])) for p in req["points"]]
json.dump(L.lafs_summary_for_submission(req["tier"], points), sys.stdout, default=str)
