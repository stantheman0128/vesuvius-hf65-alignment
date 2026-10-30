"""Print .zattrs (abridged) and 0/ and 2/ .zarray of candidate S3 CT volumes -> results/volinfo.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from common import fetch, s3
out = {}
for v in sys.argv[1:]:
    rec = {}
    for rel in (".zattrs", "0/.zarray", "2/.zarray", "zarr.json", "meta.json"):
        d = fetch(s3(f"{v}/{rel}"))
        if d is not None:
            try:
                rec[rel] = json.loads(d)
            except Exception:
                rec[rel] = d[:500].decode("utf-8", "replace")
    out[v] = rec
    za = rec.get("2/.zarray", {})
    zt = json.dumps(rec.get(".zattrs", rec.get("meta.json", {})))
    print("==", v)
    print("   2/.zarray:", {k: za.get(k) for k in ("shape", "chunks", "dtype", "compressor", "dimension_separator")})
    print("   0 shape:", rec.get("0/.zarray", {}).get("shape"))
    import re
    for m in sorted(set(re.findall(r"[A-Za-z0-9_./-]*(?:esrf|SCROLLS|HEL_)[A-Za-z0-9_./-]*", zt)))[:12]:
        print("   name:", m)
p = os.path.join(AUDIT, "results", "volinfo.json")
old = json.load(open(p)) if os.path.exists(p) else {}
old.update(out)
json.dump(old, open(p, "w"), indent=1)
