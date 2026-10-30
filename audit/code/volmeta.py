"""Fetch metadata.json (and transform.json presence) of S3 CT volumes; print any raw-data / source names.
-> results/volmeta.json"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from common import fetch, s3
p = os.path.join(AUDIT, "results", "volmeta.json")
out = json.load(open(p)) if os.path.exists(p) else {}
for v in sys.argv[1:]:
    d = fetch(s3(f"{v}/metadata.json"))
    t = fetch(s3(f"{v}/transform.json"))
    m = json.loads(d) if d else None
    out[v] = {"metadata": m, "transform": json.loads(t) if t else None}
    s = json.dumps(m)
    names = sorted(set(re.findall(r"[A-Za-z0-9_./-]*(?:esrf|SCROLLS|HEL|TA_?\d)[A-Za-z0-9_./-]*", s)))
    print("==", v, "| keys:", list(m.keys()) if isinstance(m, dict) else type(m).__name__, "| transform:", t is not None)
    for n in names[:10]:
        print("    ", n)
json.dump(out, open(p, "w"), indent=1)
