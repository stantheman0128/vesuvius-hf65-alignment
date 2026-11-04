"""Check the built AS and C0 arms of w01 against their definitions over the whole segment.

usage: python -I check_arms_w01.py LABEL_ROOT RAW_ROOT REGIONS_JSON

Written apart from build_labels.py. It recomputes what each arm must hold from the shipped label
TIFFs and compares it with the built zarrs, pixel for pixel, on every 4096 px tile of the tifxyz
grid at level 0:
  AS canonical   cell (ty, tx) carries canvas (ty, tx)
  C0 canonical   cell (ty, tx) carries canvas (rint(y), rint(x)), (y, x) = closed-form inverse of
                 the HF65 fit at (ty, tx) (apply_fit.Model.inv_exact, a 2 x 2 solve; the builder
                 uses the fixed-point inverse and separable index maps instead)
  HELD, GUARD    the held-out canvas boxes mapped by fwd (corners), padded by margin_px; GUARD is
                 HELD padded by guard_px. Recomputed here and compared with the manifest's boxes.
  arm X          inklabels = X canonical outside HELD and C0 canonical inside HELD;
                 supervision_mask = X canonical supervision with GUARD set to 0;
                 validation_mask = C0 canonical supervision inside HELD, 0 elsewhere.
It also checks that every level l equals level 0 [::2^l, ::2^l], reports where AS and C0 differ in
the training area, and the offset (dy, dx) of the fit over the supervised canvas.
Prints one line per check and ALL_OK or SOME_FAILED at the end.
"""
import json
import os
import sys
import time

import numpy as np
import zarr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "hf65"))
from labellib import read_label_tif  # noqa: E402
import apply_fit  # noqa: E402

TILE, LEVELS, SLICE = 4096, 6, 32
KINDS = ("inklabels", "supervision_mask", "validation_mask")
root, raw_root, regions = sys.argv[1:4]
R = json.load(open(regions))
man = json.load(open(os.path.join(root, "manifest.json")))
ok = True


def report(good, text):
    global ok
    ok &= bool(good)
    print(("ok   " if good else "FAIL ") + text, flush=True)


for key, S in R["segments"].items():
    t0 = time.time()
    M = man["segments"][key]
    H, W = M["tifxyz_full_shape"]
    model = apply_fit.model(S["fit_key"])
    raw = os.path.join(raw_root, S.get("bucket_scroll", S["scroll"]), S["segment"])
    ink = read_label_tif(os.path.join(raw, S["ink_tif"]))
    sup = read_label_tif(os.path.join(raw, S["sup_tif"]))
    Hc, Wc = ink.shape
    # boxes, recomputed
    held = []
    for b in S["heldout"].values():
        ty, tx = model.fwd(np.array([b[0], b[0], b[1], b[1]], float), np.array([b[2], b[3], b[2], b[3]], float))
        m = R["margin_px"]
        held.append([max(0, int(np.floor(ty.min())) - m), min(H, int(np.ceil(ty.max())) + m),
                     max(0, int(np.floor(tx.min())) - m), min(W, int(np.ceil(tx.max())) + m)])
    g = R["guard_px"]
    guard = [[max(0, y0 - g), min(H, y1 + g), max(0, x0 - g), min(W, x1 + g)] for y0, y1, x0, x1 in held]
    report(held == [list(v) for v in M["heldout_tifxyz_boxes"]], f"{key} held-out boxes recomputed == manifest {held}")
    report(guard == [list(v) for v in M["guard_boxes"]], f"{key} guard boxes recomputed == manifest")
    # offset over the supervised canvas (a 1/16 grid)
    ys, xs = np.nonzero(sup[::16, ::16])
    dy, dx = model.offset(ys * 16.0, xs * 16.0)
    print(f"{key}: fit offset over the supervised canvas: dy {dy.min():+.1f} to {dy.max():+.1f} px, "
          f"dx {dx.min():+.1f} to {dx.max():+.1f} px; |d| up to {np.hypot(dy, dx).max():.1f} px", flush=True)
    z = {arm: {k: zarr.open_group(os.path.join(root, arm, S["scroll"], S["segment"], f"{S['segment']}_{k}.zarr"),
                                  mode="r", zarr_format=2) for k in KINDS} for arm in ("AS", "C0")}
    bad = {(arm, k): 0 for arm in ("AS", "C0") for k in KINDS}
    lvl_bad = {(arm, k): 0 for arm in ("AS", "C0") for k in KINDS}
    cnt = {(arm, k): [0] * LEVELS for arm in ("AS", "C0") for k in KINDS}
    diff_train = {"cells_supervised_in_either": 0, "sup_differs": 0, "ink_differs_where_both_supervised": 0}
    for ty0 in range(0, H, TILE):
        for tx0 in range(0, W, TILE):
            h, w = min(TILE, H - ty0), min(TILE, W - tx0)
            TY, TX = np.meshgrid(np.arange(ty0, ty0 + h, dtype=float), np.arange(tx0, tx0 + w, dtype=float), indexing="ij")
            cy, cx = model.inv_exact(TY, TX)
            iy, ix = np.rint(cy).astype(np.int64), np.rint(cx).astype(np.int64)
            valid = (iy >= 0) & (iy < Hc) & (ix >= 0) & (ix < Wc)
            c0 = {}
            for name, src in (("ink", ink), ("sup", sup)):
                a = np.zeros((h, w), np.uint8)
                a[valid] = src[iy[valid], ix[valid]]
                c0[name] = a
            del TY, TX, cy, cx, iy, ix, valid
            asc = {}
            for name, src in (("ink", ink), ("sup", sup)):
                a = np.zeros((h, w), np.uint8)
                hh, ww = max(0, min(Hc - ty0, h)), max(0, min(Wc - tx0, w))
                a[:hh, :ww] = src[ty0:ty0 + hh, tx0:tx0 + ww]
                asc[name] = a
            hm = np.zeros((h, w), bool)
            gm = np.zeros((h, w), bool)
            for boxes, mask in ((held, hm), (guard, gm)):
                for y0, y1, x0, x1 in boxes:
                    mask[max(y0 - ty0, 0):max(min(y1 - ty0, h), 0), max(x0 - tx0, 0):max(min(x1 - tx0, w), 0)] = True
            val = np.where(hm, c0["sup"], 0).astype(np.uint8)
            exp = {}
            for arm, can in (("AS", asc), ("C0", c0)):
                e_ink = can["ink"].copy()
                e_ink[hm] = c0["ink"][hm]
                e_sup = can["sup"].copy()
                e_sup[gm] = 0
                exp[arm] = {"inklabels": e_ink, "supervision_mask": e_sup, "validation_mask": val}
            got = {}
            for arm in ("AS", "C0"):
                got[arm] = {}
                for k in KINDS:
                    a0 = np.asarray(z[arm][k]["0"][SLICE, ty0:ty0 + h, tx0:tx0 + w])
                    got[arm][k] = a0
                    bad[(arm, k)] += int(np.count_nonzero(a0 != exp[arm][k]))
                    cnt[(arm, k)][0] += int(np.count_nonzero(a0))
                    for lv in range(1, LEVELS):
                        f = 2 ** lv
                        al = np.asarray(z[arm][k][str(lv)][SLICE, ty0 // f:ty0 // f + -(-h // f), tx0 // f:tx0 // f + -(-w // f)])
                        lvl_bad[(arm, k)] += int(np.count_nonzero(al != a0[::f, ::f][:al.shape[0], :al.shape[1]]))
                        cnt[(arm, k)][lv] += int(np.count_nonzero(al))
            sa, sc = got["AS"]["supervision_mask"] > 0, got["C0"]["supervision_mask"] > 0
            diff_train["cells_supervised_in_either"] += int(np.count_nonzero(sa | sc))
            diff_train["sup_differs"] += int(np.count_nonzero(sa != sc))
            both = sa & sc
            diff_train["ink_differs_where_both_supervised"] += int(np.count_nonzero(
                (got["AS"]["inklabels"][both] > 0) != (got["C0"]["inklabels"][both] > 0)))
    for (arm, k), n in bad.items():
        report(n == 0, f"{key} {arm} {k}: level 0 differs from the definition at {n} px")
    for (arm, k), n in lvl_bad.items():
        report(n == 0, f"{key} {arm} {k}: levels 1 to 5 differ from level 0 [::2^l] at {n} px")
    for (arm, k), c in cnt.items():
        report(c == M["nonzero_per_level"][arm][k], f"{key} {arm} {k}: nonzero per level {c} == manifest")
    report(all(c > 0 for (arm, k), cs in cnt.items() for c in cs), f"{key}: every arm, array and level is nonzero")
    e = diff_train
    print(f"{key}: training area (outside GUARD), level 0: {e['cells_supervised_in_either']} cells supervised in AS or C0; "
          f"supervision differs at {e['sup_differs']} ({e['sup_differs'] / max(1, e['cells_supervised_in_either']):.1%}); "
          f"where both are supervised the ink label differs at {e['ink_differs_where_both_supervised']} cells", flush=True)
    print(f"{key}: {time.time() - t0:.0f} s", flush=True)
    del ink, sup
print("ALL_OK" if ok else "SOME_FAILED")
