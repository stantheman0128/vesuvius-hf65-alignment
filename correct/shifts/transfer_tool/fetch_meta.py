"""Fetch meta.json for the 26 offset segments through this repo's own fetch() and cache
(../../../audit/code/common.py).

usage: python -I fetch_meta.py      (needs httpx; downloads about 13 kB)
Writes meta/<scroll>__<segment>.json and meta/fetch_check.json next to this script, replacing the
copies here. The download cache and ledger go to $HF65_DATA if it is set, otherwise to hf65_data/
next to this script. Bytecode is off.
"""
import csv
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF65_DATA", os.path.join(HERE, "hf65_data"))
os.makedirs(os.path.join(os.environ["HF65_DATA"], "cache"), exist_ok=True)
PUB = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

spec = importlib.util.spec_from_file_location("hf65_common", os.path.join(PUB, "audit", "code", "common.py"))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

rows = list(csv.DictReader(open(os.path.join(PUB, "correct", "shifts", "shifts.csv"), newline="")))
phase1 = {(e["scroll"], e["segment"]): e for e in json.load(open(os.path.join(PUB, "audit", "results", "phase1.json")))}
out = []
for r in rows:
    e = phase1[(r["scroll"], r["segment"])]
    raw = common.fetch(common.hf(f"{e['path']}/meta.json"))
    meta = json.loads(raw)
    with open(os.path.join(HERE, "meta", f"{r['scroll']}__{r['segment']}.json"), "wb") as f:
        f.write(raw)
    out.append({
        "key": r["key"],
        "path": e["path"],
        "meta_scale_live": meta.get("scale"),
        "meta_scale_phase1": e["meta_scale"],
        "equal": meta.get("scale") == e["meta_scale"],
        "bytes": len(raw),
    })
    print(r["key"], out[-1]["equal"], meta.get("scale"))
with open(os.path.join(HERE, "meta", "fetch_check.json"), "w", newline="\n") as f:
    json.dump({"rows": out, "all_equal": all(o["equal"] for o in out),
               "ledger_bytes": common.ledger_total()}, f, indent=1)
print("all_equal", all(o["equal"] for o in out), "ledger_bytes", common.ledger_total())
