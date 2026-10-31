"""List the six 1667 training folders (non-recursive), then list inside every label/mask zarr
of w029 and the loader-picked supervision masks of the others, and fetch their small metadata."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hfio

OUT = os.environ["W029_OUT"]
SEGS = [
    "w013_20240304141531_2um", "w018_20240304144031_2um", "w023_20240304161941_2um",
    "w028_20251208130119156_2um", "w029_20251212185248662_2um", "w031_2025122323_2um",
]
res = {"scroll_dir": None, "folders": {}, "zarrs": {}}
top = hfio.listing("ink/1667")
res["scroll_dir"] = [(e["type"], e["path"]) for e in top]
for s in SEGS:
    ent = hfio.listing(f"ink/1667/{s}")
    res["folders"][s] = [{k: e.get(k) for k in ("type", "path", "size", "mtime", "uploadedAt")} for e in ent]

def walk_zarr(zp, depth=0):
    ent = hfio.listing(zp)
    rows = [{k: e.get(k) for k in ("type", "path", "size")} for e in ent]
    meta = {}
    for e in ent:
        name = e["path"].rsplit("/", 1)[-1]
        if e["type"] == "file" and name in (".zattrs", ".zgroup", "zarr.json", ".zarray", ".zmetadata"):
            meta[name] = json.loads(hfio.get(e["path"]).decode())
    # pyramid level dirs: fetch only their array metadata, never list their chunks
    levels = {}
    for e in ent:
        if e["type"] == "directory":
            lv = e["path"].rsplit("/", 1)[-1]
            for mname in (".zarray", "zarr.json"):
                d = hfio.get(e["path"] + "/" + mname)
                if d is not None:
                    levels[lv] = {mname: json.loads(d.decode())}
                    break
    return {"entries": rows, "meta": meta, "levels": levels}

for s in SEGS:
    for e in res["folders"][s]:
        p = e["path"]
        name = p.rsplit("/", 1)[-1]
        if e["type"] != "directory" or not name.endswith(".zarr"):
            continue
        if not any(k in name for k in ("supervision_mask", "inklabels", "validation_mask")):
            continue
        if s != "w029_20251212185248662_2um" and "supervision_mask" not in name:
            continue
        res["zarrs"][p] = walk_zarr(p)

with open(os.path.join(OUT, "listing.json"), "w") as f:
    json.dump(res, f, indent=1)
print("spent bytes", hfio.spent())
