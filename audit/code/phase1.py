"""Phase 1: metadata-only consistency check of every segment under hf://buckets/scrollprize/datasets/ink/.

Input: results/listing.jsonl (list_bucket.py). For each segment directory it reads, by HTTP (cached,
byte-counted in data/ledger.tsv through common.fetch):
  meta.json, x.tif header (first IFD only), <seg>.zarr/0/.zarray, 0/.zarray of every other .zarr in the
  segment dir, and the first IFD of every TIFF the villa validator would check.
Output: results/phase1.json (one record per segment, everything measured) and printed summary.

Rules reproduced from villa origin/main 37d1897fb:
  loader  = vesuvius/src/vesuvius/ink_detection/data/segment.py (discover_segment_labels, gather_segments)
  validator = vesuvius/src/vesuvius/ink_detection/preprocessing/validate_segments.py (metadata part only)
  full-res tifxyz size = max(1, lround(stored * (1/float32(scale)))) (vesuvius tifxyz, PR #1699)
"""
import os, sys, json, re, struct, math
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import fetch, hf, ledger_total

AUDIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTING = os.path.join(AUDIT, "results", "listing.jsonl")

# ---------- naming rules ----------
LOADER_SUFFIXES = (("_inklabels", "inklabels"), ("_supervision_mask", "supervision_mask"), ("_validation_mask", "validation_mask"))
VALIDATOR_LABEL_RE = re.compile(r"^(?P<prefix>.*)_(?P<kind>inklabels|supervision_mask|validation_mask)(?:_v(?P<v>\d+))?(?P<ext>\.tiff?)$", re.I)


def parse_label_asset(name):
    """villa segment.parse_label_asset_path on a basename -> (prefix, kind, version, ext) or None."""
    base, ext = os.path.splitext(name)
    ext = ext.lower()
    if ext not in (".tif", ".tiff", ".zarr"):
        return None
    stem, ver = base, 1
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].startswith("v") and parts[1][1:].isdigit():
        ver, stem = int(parts[1][1:]), parts[0]
    for suf, kind in LOADER_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)], kind, ver, ext
    return None


def validator_target(name):
    """validate_segments._is_target_tiff (label_version=None) on a basename."""
    if os.path.splitext(name)[1].lower() not in (".tif", ".tiff"):
        return False
    if VALIDATOR_LABEL_RE.match(name):
        return True
    n = name.lower()
    return "max" in n or "composite" in n


# ---------- readers ----------
TT = {1: ("B", 1), 2: ("c", 1), 3: ("H", 2), 4: ("I", 4), 16: ("Q", 8), 6: ("b", 1), 8: ("h", 2), 9: ("i", 4), 11: ("f", 4), 12: ("d", 8)}


def tiff_header(url):
    """First IFD of a TIFF -> dict(H, W, spp, bits, compression, tiled, extra, more_pages)."""
    head = fetch(url, (0, 15))
    if head is None:
        return {"error": "404"}
    bo = "<" if head[:2] == b"II" else ">" if head[:2] == b"MM" else None
    if bo is None:
        return {"error": "not a TIFF"}
    big = struct.unpack(bo + "H", head[2:4])[0] == 43
    off = struct.unpack(bo + ("Q" if big else "I"), head[8:16] if big else head[4:8])[0]
    ifd = fetch(url, (off, off + 4095), allow_short=True)
    if big:
        n, p, esz, vo, ps = struct.unpack(bo + "Q", ifd[:8])[0], 8, 20, 12, 8
    else:
        n, p, esz, vo, ps = struct.unpack(bo + "H", ifd[:2])[0], 2, 12, 8, 4
    if p + n * esz + ps > len(ifd):
        ifd = fetch(url, (off, off + p + n * esz + ps - 1))
    t = {}
    for i in range(n):
        e = ifd[p + i * esz: p + (i + 1) * esz]
        tag, typ = struct.unpack(bo + "HH", e[:4])
        cnt = struct.unpack(bo + ("Q" if big else "I"), e[4:vo])[0]
        if typ not in TT:
            continue
        fmt, sz = TT[typ]
        if cnt * sz <= ps:
            t[tag] = struct.unpack(bo + fmt * cnt, e[vo:vo + cnt * sz])
        else:
            t[tag] = ("ool", cnt)  # value out of line, not needed
    nxt = struct.unpack(bo + ("Q" if big else "I"), ifd[p + n * esz: p + n * esz + ps])[0]
    g = lambda k, d=None: (t[k][0] if k in t and t[k] and t[k][0] != "ool" else d)
    return {"H": g(257), "W": g(256), "spp": g(277, 1), "bits": g(258), "compression": g(259, 1),
            "tiled": 322 in t, "extra_samples": 338 in t, "more_pages": nxt != 0, "bigtiff": big}


def zarr_meta(zpath):
    """Shape/dtype/chunks of array '0' of an OME-Zarr group (v2 .zarray, else v3 zarr.json); also a bare array."""
    for rel in ("0/.zarray", ".zarray"):
        d = fetch(hf(f"{zpath}/{rel}"))
        if d is not None:
            za = json.loads(d)
            return {"shape": za["shape"], "chunks": za.get("chunks"), "dtype": za.get("dtype"), "fmt": 2, "array": rel.split("/")[0] if "/" in rel else "."}
    for rel in ("0/zarr.json", "zarr.json"):
        d = fetch(hf(f"{zpath}/{rel}"))
        if d is not None:
            za = json.loads(d)
            if za.get("node_type") == "array":
                return {"shape": za["shape"], "chunks": za.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape"),
                        "dtype": za.get("data_type"), "fmt": 3, "array": rel.split("/")[0] if "/" in rel else "."}
    return {"error": "no .zarray / zarr.json at 0/ or root"}


def spatial(shape):
    """validate_segments._normalize_zarr_shape: last two non-singleton dims."""
    dims = [int(v) for v in shape if int(v) != 1]
    return (dims[-2], dims[-1]) if len(dims) >= 2 else None


def tiff_spatial(h):
    """validate_segments._normalize_tiff_shape on the first page (series[0] of a 1-page TIFF)."""
    if h.get("spp", 1) > 1:
        dims = [x for x in (h["H"], h["W"], h["spp"]) if x != 1]
    else:
        dims = [x for x in (h["H"], h["W"]) if x != 1]
    if len(dims) == 2:
        return tuple(dims)
    if len(dims) == 3:
        if dims[-1] <= 4 and dims[0] > 4 and dims[1] > 4:
            return dims[0], dims[1]
        return dims[-2], dims[-1]
    return None


def full_res(stored, scale):
    s = float(np.float32(scale))
    v = stored * (1.0 / s)
    return max(1, int(math.floor(v + 0.5)))  # lround for positive values


# ---------- segment discovery from the listing ----------
def load_listing():
    rows = [json.loads(l) for l in open(LISTING)]
    children = {}
    for r in rows:
        par = r["path"].rsplit("/", 1)[0]
        children.setdefault(par, []).append(r)
    return rows, children


def segment_dirs(children):
    """(scroll, segment, path, unused) for every directory that holds an x.tif or a <name>.zarr; plus
    validator-style segment dirs (children of a scroll dir) even when empty."""
    out = []
    scrolls = [r["path"] for r in children["ink"] if r["type"] == "directory"]
    for sd in scrolls:
        if sd == "ink/unused":
            for r in children.get(sd, []):
                if r["type"] != "directory":
                    continue
                for s in children.get(r["path"], []):
                    if s["type"] == "directory" and not s["path"].lower().endswith(".zarr"):
                        out.append((r["path"].split("/")[-1], s["path"].split("/")[-1], s["path"], True))
            continue
        for r in children.get(sd, []):
            if r["type"] == "directory" and not r["path"].lower().endswith(".zarr"):
                out.append((sd.split("/")[-1], r["path"].split("/")[-1], r["path"], False))
    return out


def check_segment(arg, children):
    scroll, seg, path, unused = arg
    ent = children.get(path, [])
    names = {e["path"].split("/")[-1]: e for e in ent}
    rec = {"scroll": scroll, "segment": seg, "path": path, "unused": unused, "errors": []}
    rec["subdirs"] = sorted(n for n, e in names.items() if e["type"] == "directory" and not n.lower().endswith(".zarr"))
    # meta + tifxyz
    if "meta.json" in names:
        try:
            m = json.loads(fetch(hf(f"{path}/meta.json")))
            rec["meta_scale"] = m.get("scale")
            rec["meta_volume"] = m.get("volume")
            rec["meta_scroll_source"] = m.get("scroll_source")
        except Exception as ex:
            rec["errors"].append(f"meta.json unreadable: {ex}")
    else:
        rec["errors"].append("meta.json missing")
    if "x.tif" in names:
        try:
            h = tiff_header(hf(f"{path}/x.tif"))
            rec["xtif"] = h
            if "error" in h:
                rec["errors"].append(f"x.tif: {h['error']}")
        except Exception as ex:
            rec["errors"].append(f"x.tif unreadable: {ex}")
    else:
        rec["errors"].append("x.tif missing at segment top level")
    for a in ("y.tif", "z.tif"):
        if a not in names:
            rec["errors"].append(f"{a} missing")
    if rec.get("xtif", {}).get("H") and rec.get("meta_scale"):
        sx, sy = rec["meta_scale"][0], rec["meta_scale"][1]
        rec["stored_hw"] = [rec["xtif"]["H"], rec["xtif"]["W"]]
        rec["full_hw"] = [full_res(rec["xtif"]["H"], sy), full_res(rec["xtif"]["W"], sx)]
    # zarrs
    zarrs = {}
    for n, e in names.items():
        if e["type"] == "directory" and n.lower().endswith(".zarr"):
            try:
                zarrs[n] = zarr_meta(f"{path}/{n}")
            except Exception as ex:
                zarrs[n] = {"error": str(ex)}
    rec["zarrs"] = zarrs
    cz = f"{seg}.zarr"
    if cz in zarrs and "shape" in zarrs[cz]:
        rec["canvas_shape"] = zarrs[cz]["shape"]
        rec["canvas_hw"] = list(spatial(zarrs[cz]["shape"]))
    elif cz in zarrs:
        rec["errors"].append(f"{cz} unreadable: {zarrs[cz].get('error')}")
    else:
        rec["errors"].append(f"{cz} missing")
    # loader selection (.zarr, prefix == segment name, highest version per kind)
    cand = {"inklabels": {}, "supervision_mask": {}, "validation_mask": {}}
    labels = []
    for n in sorted(names):
        p = parse_label_asset(n)
        if p is None:
            continue
        prefix, kind, ver, ext = p
        labels.append({"name": n, "prefix": prefix, "kind": kind, "version": ver, "ext": ext})
        if ext == ".zarr" and prefix == seg:
            cand[kind][ver] = n
    rec["loader_pick"] = {k: (v[max(v)] if v else None) for k, v in cand.items()}
    rec["label_assets"] = labels
    rec["other_label_like_zarrs"] = sorted(n for n in zarrs if "inklabel" in n.lower() and parse_label_asset(n) is None)
    # validator reproduction (metadata part)
    vt = sorted(n for n in names if names[n]["type"] == "file" and validator_target(n))
    tiffs = {}
    for n in vt:
        try:
            tiffs[n] = tiff_header(hf(f"{path}/{n}"))
        except Exception as ex:
            tiffs[n] = {"error": str(ex)}
    rec["validator_tiffs"] = tiffs
    issues = []
    if cz not in zarrs:
        issues.append(f"Expected segment zarr directory is missing: {cz}")
    exp = tuple(rec["canvas_hw"]) if "canvas_hw" in rec else None
    kinds = {VALIDATOR_LABEL_RE.match(n).group("kind").lower() for n in vt if VALIDATOR_LABEL_RE.match(n)}
    for k in ("inklabels", "supervision_mask"):
        if k not in kinds:
            issues.append(f"missing required label TIFF containing '{k}'")
    for n, h in tiffs.items():
        if "error" in h:
            issues.append(f"{n}: Failed to read TIFF shape: {h['error']}")
            continue
        sp = tiff_spatial(h)
        h["spatial"] = sp
        if exp is not None and sp != exp:
            issues.append(f"{n}: shape {sp[0]}x{sp[1]} does not match zarr {exp[0]}x{exp[1]}")
        if VALIDATOR_LABEL_RE.match(n) and h.get("spp") in (2, 4):
            issues.append(f"{n}: label image has an alpha channel")
    rec["validator_issues"] = issues
    return rec


def main():
    rows, children = load_listing()
    segs = segment_dirs(children)
    b0 = ledger_total()
    with ThreadPoolExecutor(4) as ex:
        recs = list(ex.map(lambda a: check_segment(a, children), segs))
    with open(os.path.join(AUDIT, "results", "phase1.json"), "w") as f:
        json.dump(recs, f, indent=1)
    print("segments", len(recs), "bytes this run", ledger_total() - b0, "ledger total", ledger_total())


if __name__ == "__main__":
    main()
