"""Flags d/e/f: label assets the villa loader ignores or rejects, using villa's own functions.

The functions are not reimplemented: parse_label_asset_path and discover_segment_labels (plus the module
constants they read) are taken verbatim from villa 37d1897fb
vesuvius/src/vesuvius/ink_detection/data/segment.py, extracted with ast and executed with stub
Segment/config objects, because the rest of the module imports the vesuvius package. villa source is not
copied here; pass it as the only argument, either a villa checkout (the file is read with
`git -C <checkout> show 37d1897fb:<path>`) or that file already extracted. Its sha256 is checked.
Input: results/listing.jsonl and results/phase1.json. Output: results/labels_check.json.

  d  label-like asset whose parsed prefix != folder name (discover_segment_labels skips it)
  e  name contains inklabels / supervision_mask / validation_mask but parse_label_asset_path returns None
  f  an ignored (d) asset of the same kind and extension the loader reads (.zarr) has a higher version than
     the one discover_segment_labels selects, or is the only validation_mask of the folder
Also runs discover_segment_labels with label_version "v1" and "v2" (required=True) to show which folders
would raise if a config set label_version (no shipped config at 37d1897fb sets it)."""
import ast, hashlib, json, os, subprocess, sys
from dataclasses import dataclass, replace, field
from pathlib import PurePosixPath, Path

AUDIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VILLA_COMMIT = "37d1897fb"
VILLA_PATH = "vesuvius/src/vesuvius/ink_detection/data/segment.py"
VILLA_SHA256 = "7d8466baaa6057907ad1fd20d4620079e30130a5885aa7c2fa6e336fcff1d849"
if len(sys.argv) != 2:
    raise SystemExit("usage: labels_check.py <villa checkout | segment.py extracted at 37d1897fb>")
if os.path.isdir(sys.argv[1]):
    raw = subprocess.run(["git", "-C", sys.argv[1], "show", f"{VILLA_COMMIT}:{VILLA_PATH}"],
                         check=True, capture_output=True).stdout
else:
    raw = open(sys.argv[1], "rb").read()
if hashlib.sha256(raw).hexdigest() != VILLA_SHA256:
    raise SystemExit(f"{sys.argv[1]}: not {VILLA_PATH} at villa {VILLA_COMMIT} (sha256 differs)")
SRC = f"villa {VILLA_COMMIT}:{VILLA_PATH}"
tree = ast.parse(raw.decode("utf-8"))
keep = [n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in ("parse_label_asset_path", "discover_segment_labels"))
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("_LABEL_SUFFIXES", "_LABEL_EXTENSIONS") for t in n.targets))]
assert len(keep) == 4, [getattr(n, "name", None) for n in keep]
mod = ast.Module(body=keep, type_ignores=[])
NS = {"Path": PurePosixPath, "replace": replace, "Segment": object, "InkDataConfig": object}
exec(compile(mod, SRC, "exec"), NS)
parse_label_asset_path = NS["parse_label_asset_path"]
discover_segment_labels = NS["discover_segment_labels"]


@dataclass
class Cfg:
    label_version: str | None = None


class Dir:
    """Stands in for segment_dir: iterdir() yields the bucket entries of the folder as paths."""

    def __init__(self, path, names):
        self.path, self.names = path, names

    def iterdir(self):
        return (PurePosixPath(self.path) / n for n in self.names)

    def __str__(self):
        return self.path


@dataclass
class Seg:
    segment_dir: object
    segment_name: str
    data_config: Cfg
    inklabels: object = None
    supervision_mask: object = None
    validation_mask: object = None


def main():
    rows = [json.loads(l) for l in open(os.path.join(AUDIT, "results", "listing.jsonl"))]
    kids = {}
    for r in rows:
        kids.setdefault(r["path"].rsplit("/", 1)[0], []).append(r["path"].rsplit("/", 1)[1])
    P1 = json.load(open(os.path.join(AUDIT, "results", "phase1.json")))
    out = []
    for s in P1:
        names = sorted(kids.get(s["path"], []))
        seg = Seg(Dir(s["path"], names), s["segment"], Cfg(None))
        try:
            sel = discover_segment_labels(seg, extension=".zarr", required=False)
            used = {k: (getattr(sel, k).name if getattr(sel, k) is not None else None) for k in ("inklabels", "supervision_mask", "validation_mask")}
        except Exception as ex:
            used = {"error": str(ex)}
        raises = {}
        for v in ("v1", "v2"):
            try:
                discover_segment_labels(Seg(Dir(s["path"], names), s["segment"], Cfg(v)), extension=".zarr", required=True)
                raises[v] = None
            except ValueError as ex:
                raises[v] = str(ex).replace(s["path"], "<seg>")
        assets, d, e = [], [], []
        for n in names:
            p = parse_label_asset_path(PurePosixPath(s["path"]) / n)
            labelish = any(k in n.lower() for k in ("inklabels", "supervision_mask", "validation_mask"))
            if p is None:
                if labelish:
                    e.append(n)
                continue
            a = {"name": n, "prefix": p["prefix"], "kind": p["label_kind"], "version": p["version_num"], "ext": p["extension"]}
            assets.append(a)
            if p["prefix"] != s["segment"]:
                d.append(a)
        used_ver = {}
        for k in ("inklabels", "supervision_mask", "validation_mask"):
            n = used.get(k)
            if n:
                used_ver[k] = parse_label_asset_path(PurePosixPath(n))["version_num"]
        f = []
        for a in d:
            if a["ext"] != ".zarr":
                continue
            k = a["kind"]
            if k in used_ver and a["version"] > used_ver[k]:
                f.append(f"{a['name']} (v{a['version']} > used v{used_ver[k]})")
            elif k == "validation_mask" and k not in used_ver and not s["unused"]:
                f.append(f"{a['name']} (only validation_mask; loader uses none)")
            elif k in ("inklabels", "supervision_mask") and k not in used_ver and not s["unused"]:
                f.append(f"{a['name']} (loader finds no {k})")
        out.append({"path": s["path"], "scroll": s["scroll"], "segment": s["segment"], "unused": s["unused"],
                    "used": used, "assets": assets, "d_ignored_prefix": [x["name"] for x in d],
                    "e_rejected": e, "f_newer_or_only": f, "raises_if_label_version": raises})
    json.dump(out, open(os.path.join(AUDIT, "results", "labels_check.json"), "w"), indent=1)
    tr = [o for o in out if not o["unused"]]
    print("training folders", len(tr), "| d:", sum(bool(o["d_ignored_prefix"]) for o in tr),
          "e:", sum(bool(o["e_rejected"]) for o in tr), "f:", sum(bool(o["f_newer_or_only"]) for o in tr),
          "| raise if label_version=v2:", sum(o["raises_if_label_version"]["v2"] is not None for o in tr),
          "v1:", sum(o["raises_if_label_version"]["v1"] is not None for o in tr))
    for o in sorted(tr, key=lambda o: (o["segment"] != "w018_20240304144031_2um", o["path"])):
        if o["d_ignored_prefix"] or o["e_rejected"] or o["f_newer_or_only"]:
            print(o["path"], "| used", o["used"], "| d", o["d_ignored_prefix"], "| e", o["e_rejected"], "| f", o["f_newer_or_only"])
    un = [o for o in out if o["unused"]]
    print("unused folders", len(un), "| d:", sum(bool(o["d_ignored_prefix"]) for o in un), "e:", sum(bool(o["e_rejected"]) for o in un))
    for o in un:
        if o["d_ignored_prefix"] or o["e_rejected"]:
            print("  unused", o["path"], "| d", o["d_ignored_prefix"], "| e", o["e_rejected"])


main()
