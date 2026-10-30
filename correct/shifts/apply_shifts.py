"""Apply a row of correct/shifts/shifts.json: canvas/label pixel <-> tifxyz cell, and label tiles into the tifxyz frame.

The maths is correct/apply_fit.py's (Model.fwd, Model.inv, resample); this file only looks up the
per-segment parameters in shifts.json. Conventions are in shifts.json "convention":
canvas content at level-0 pixel (y, x) lies at tifxyz full-res cell (y + dy, x + dx).

A segment can be named by its key (c013), its folder (w013_20240304141531_2um), scroll/folder
(1667/w013_20240304141531_2um) or bucket path (ink/1667/w013_20240304141531_2um).
Segments listed under "aligned_no_correction" get the identity map. Any other name is an error.

Library use:
    import apply_shifts as A
    m = A.model("w013_20240304141531_2um")
    ty, tx = m.fwd(y, x); y, x = m.inv(ty, tx)
    out, ok, (ty0, tx0) = A.resample_tile("w013_20240304141531_2um", label_tile, origin=(Y0, X0), level=2, order=0)

CLI (python -I correct/shifts/apply_shifts.py ...):
  list                                  the rows
  fwd SEG y x                           canvas -> tifxyz
  inv SEG ty tx                         tifxyz -> canvas
  resample SEG IN OUT --origin Y0 X0 [--level K] [--pyr-shift PY PX] [--order 0|1]
           [--out-origin TY0 TX0] [--shape H W] [--fill V]
      same arguments and tile conventions as apply_fit.py resample: IN (.npy, .png, .tif) is a 2D tile at
      pyramid level K whose pixel (0, 0) is level-K canvas pixel (Y0, X0); OUT pixel (r, c) is tifxyz
      level-K cell (TY0 + r, TX0 + c). --order 0 (nearest) for labels and masks, 1 for images.
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import apply_fit

SHIFTS = os.environ.get("HF65_SHIFTS", os.path.join(HERE, "shifts.json"))
_doc = None


def doc():
    global _doc
    if _doc is None:
        _doc = json.load(open(SHIFTS))
    return _doc


def _names(bucket_path, key):
    seg = bucket_path.split("/", 1)[1] if bucket_path.startswith("ink/") else bucket_path
    return {key, bucket_path, seg, seg.split("/", 1)[1]}


def lookup(name):
    """-> ("offset", row) | ("aligned", entry). Raises KeyError for anything else."""
    name = name.strip().rstrip("/")
    hits = [("offset", r) for r in doc()["segments"] if name in _names(r["bucket_path"], r["key"])]
    hits += [("aligned", a) for a in doc()["aligned_no_correction"] if name in _names(a["bucket_path"], a["key"])]
    if len(hits) > 1:
        raise KeyError(f"{name!r} is ambiguous ({[h[1]['bucket_path'] for h in hits]}); use scroll/folder")
    if not hits:
        nm = [n["bucket_path"] for n in doc().get("not_measured", []) if name in _names(n["bucket_path"], "")]
        why = f" (listed as not measured: {nm[0]})" if nm else ""
        raise KeyError(f"{name!r} is not in {SHIFTS}{why}")
    return hits[0]


def model(name):
    kind, r = lookup(name)
    if kind == "aligned":
        return apply_fit.Model((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), f"{r['key']}/identity")
    return apply_fit.Model((r["dy_a"], r["dy_per_y"], r["dy_per_x"]), (r["dx_a"], r["dx_per_y"], r["dx_per_x"]),
                           f"{r['key']}/shifts")


def resample_tile(name, tile, origin, level=0, pyr_shift=(0.0, 0.0), order=0, out_origin=None, shape=None,
                  fill=0.0, tile_valid=None):
    """Level-K canvas tile with pixel (0, 0) = level-K canvas pixel origin -> the same content on the tifxyz
    level-K lattice. Returns (out, ok, (TY0, TX0)); out[r, c] is tifxyz level-K cell (TY0 + r, TX0 + c),
    ok marks pixels whose source fell inside the tile. Same conventions as apply_fit.py resample."""
    M = model(name)
    tile = np.asarray(tile)
    s = 2 ** level
    h0 = (s - 1) / 2.0
    tile_origin = (s * origin[0] + h0 - pyr_shift[0], s * origin[1] + h0 - pyr_shift[1])
    shape = tuple(shape) if shape else tile.shape
    if out_origin is None:
        cy = tile_origin[0] + s * (tile.shape[0] - 1) / 2
        cx = tile_origin[1] + s * (tile.shape[1] - 1) / 2
        dy, dx = M.offset(cy, cx)
        out_origin = (origin[0] + int(np.rint(dy / s)) + (tile.shape[0] - shape[0]) // 2,
                      origin[1] + int(np.rint(dx / s)) + (tile.shape[1] - shape[1]) // 2)
    oo = (s * out_origin[0] + h0, s * out_origin[1] + h0)
    out, ok = apply_fit.resample(tile, M, tile_origin, s, oo, s, shape, order=order, fill=fill, tile_valid=tile_valid)
    if order == 0:
        out = out.astype(tile.dtype)
    return out, ok, tuple(int(v) for v in out_origin)


def cmd_list(_):
    print(f"| key | segment | group | model | dy | dx | fit regions |")
    print("|---|---|---|---|---|---|---|")
    for r in doc()["segments"]:
        print(f"| {r['key']} | {r['scroll']}/{r['segment']} | {r['group']} | {r['model']} |"
              f" {r['dy_a']:+.3f} {r['dy_per_y']:+.7f}*y {r['dy_per_x']:+.7f}*x |"
              f" {r['dx_a']:+.3f} {r['dx_per_y']:+.7f}*y {r['dx_per_x']:+.7f}*x | {r['n_fit_regions']} |")
    for a in doc()["aligned_no_correction"]:
        print(f"| {a['key']} | {a['bucket_path']} | aligned | identity | 0 | 0 | measured {a['measured']} |")


def cmd_fwd(a):
    ty, tx = model(a.seg).fwd(a.y, a.x)
    print(f"{ty:.3f} {tx:.3f}")


def cmd_inv(a):
    y, x = model(a.seg).inv(a.ty, a.tx)
    print(f"{y:.3f} {x:.3f}")


def cmd_resample(a):
    tile = apply_fit.read_2d(a.inp)
    out, ok, (oy, ox) = resample_tile(a.seg, tile, a.origin, a.level, a.pyr_shift, a.order, a.out_origin, a.shape, a.fill)
    if a.order == 1 and np.issubdtype(tile.dtype, np.integer) and os.path.splitext(a.out)[1].lower() != ".npy":
        out = np.clip(np.rint(out), np.iinfo(tile.dtype).min, np.iinfo(tile.dtype).max).astype(tile.dtype)
    apply_fit.write_2d(a.out, out)
    M = model(a.seg)
    print(f"{a.inp} ({tile.shape[0]}x{tile.shape[1]}, level {a.level}, canvas origin {a.origin[0]},{a.origin[1]}) -> {a.out}"
          f" ({out.shape[0]}x{out.shape[1]}, tifxyz level-{a.level} origin {oy},{ox}); model {M.name};"
          f" {int((~ok).sum())} px filled with {a.fill} (source outside tile)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list").set_defaults(fn=cmd_list)
    f = sp.add_parser("fwd"); f.add_argument("seg"); f.add_argument("y", type=float); f.add_argument("x", type=float); f.set_defaults(fn=cmd_fwd)
    i = sp.add_parser("inv"); i.add_argument("seg"); i.add_argument("ty", type=float); i.add_argument("tx", type=float); i.set_defaults(fn=cmd_inv)
    r = sp.add_parser("resample")
    r.add_argument("seg"); r.add_argument("inp"); r.add_argument("out")
    r.add_argument("--origin", type=int, nargs=2, required=True, metavar=("Y0", "X0"))
    r.add_argument("--level", type=int, default=0)
    r.add_argument("--pyr-shift", type=float, nargs=2, default=(0.0, 0.0), metavar=("PY", "PX"))
    r.add_argument("--order", type=int, choices=(0, 1), default=0)
    r.add_argument("--out-origin", type=int, nargs=2, metavar=("TY0", "TX0"))
    r.add_argument("--shape", type=int, nargs=2, metavar=("H", "W"))
    r.add_argument("--fill", type=float, default=0.0)
    r.set_defaults(fn=cmd_resample)
    a = p.parse_args(argv)
    try:
        a.fn(a)
    except KeyError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
