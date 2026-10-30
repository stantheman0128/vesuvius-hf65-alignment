"""Fetch <seg>.zarr/.zattrs of every training segment canvas; keep all keys except multiscales.
-> results/canvas_zattrs.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from common import fetch, hf
R = json.load(open(os.path.join(AUDIT, "results", "phase1.json")))
out = {}
for r in R:
    if r["unused"] or "canvas_shape" not in r:
        continue
    d = fetch(hf(f"{r['path']}/{r['segment']}.zarr/.zattrs"))
    z = json.loads(d) if d else {}
    out[r["path"]] = {k: v for k, v in z.items() if k != "multiscales"}
    ms = z.get("multiscales", [{}])[0]
    out[r["path"]]["_ms_metadata"] = ms.get("metadata")
json.dump(out, open(os.path.join(AUDIT, "results", "canvas_zattrs.json"), "w"), indent=1)
for k, v in out.items():
    print(k, json.dumps(v)[:700])
