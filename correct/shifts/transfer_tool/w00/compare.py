"""Pixel agreement of transfer.py outputs on w00 with the apply_fit.py correction (shifts.csv w00 row).

Grids (all 32281 x 51345, rows x cols):
  loader grid: pixel (r, c) = tifxyz full-res cell (r, c) = stored (r * sy, c * sx); this is what
      apply_fit.py's resample writes (out_origin 0, step 1) and what the full_3d loader reads.
  tool grid: pixel (i, j) = stored ((i + 0.5) * S_H / O_H, (j + 0.5) * S_W / O_W) (vc_reader pixelCenter).
References:
  AF       apply_fit readme (= shifts.csv w00) resample on the loader grid (closed form, checked against
           apply_fit.resample itself on sample blocks first).
  AF_tool  the same model sampled at the tool grid's pixel centres.
  SHIP     the shipped label read one-to-one, what the loader reads with no correction.
Region: supervision mask resampled with AF, > 0. A pixel is compared when the tool marked it
measured (validity 255) and the AF sample fell inside the label.
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

import compare_lib as C  # noqa: E402

BLOCK = 1024
t0 = time.time()
print(f"model {C.MODEL.name}: dy {C.MODEL.cy.tolist()}, dx {C.MODEL.cx.tolist()}", flush=True)
label = C.load_label()
mask = tifffile.imread(C.MASK_TIF)
print(f"label {label.shape} {label.dtype} values {np.unique(label[::97, ::89]).tolist()}; "
      f"mask {mask.shape} values {np.unique(mask[::97, ::89]).tolist()}; read {time.time() - t0:.0f}s", flush=True)

for r0 in (0, 16000, C.O_H - 256):
    ref, ref_ok = C.apply_fit.resample(label, C.MODEL, (0.0, 0.0), 1.0, (float(r0), 0.0), 1.0, (256, C.O_W), order=0)
    mine, mine_ok = C.nearest(label, np.arange(r0, r0 + 256, dtype=np.float64), np.arange(C.O_W, dtype=np.float64))
    print(f"closed form vs apply_fit.resample, rows {r0}..{r0 + 255}: differing labels "
          f"{int(np.count_nonzero(ref.astype(label.dtype) != mine))}, differing in-bounds flags "
          f"{int(np.count_nonzero(ref_ok != mine_ok))}", flush=True)

runs = {"offset": C.open_tiff(C.OFFSET_TIF), "control": C.open_tiff(C.CONTROL_TIF), "loader": C.open_tiff(C.LOADER_TIF)}
valids = {
    "offset": C.open_tiff(C.OFFSET_TIF.replace(".tif", ".valid.tif")),
    "control": C.open_tiff(C.CONTROL_TIF.replace(".tif", ".valid.tif")),
    "loader": C.open_tiff(C.LOADER_TIF.replace(".tif", ".valid.tif")),
}
for name, arr in runs.items():
    print(f"{name}: shape {arr.shape} dtype {arr.dtype}", flush=True)

pairs = ("offset~AF", "control~AF", "offset~AF_tool", "loader~AF", "loader~AF_tool", "SHIP~AF")
stats = {k: dict(compared=0, equal=0, ink_inter=0, ink_union=0, ink_a=0, ink_b=0) for k in pairs}
valid_hist = {k: {} for k in runs}
region_pixels = 0
coarse = np.zeros(((C.O_H + 2047) // 2048, (C.O_W + 2047) // 2048), np.int64)
cols_loader = np.arange(C.O_W, dtype=np.float64)
cols_tool = (np.arange(C.O_W) + 0.5) * C.S_W / C.O_W / C.SX
for r0 in range(0, C.O_H, BLOCK):
    r1 = min(C.O_H, r0 + BLOCK)
    af, af_ok = C.nearest(label, np.arange(r0, r1, dtype=np.float64), cols_loader)
    sm, sm_ok = C.nearest(mask, np.arange(r0, r1, dtype=np.float64), cols_loader)
    region = sm_ok & (sm > 0)
    del sm, sm_ok
    region_pixels += int(region.sum())
    af_tool, _ = C.nearest(label, (np.arange(r0, r1) + 0.5) * C.S_H / C.O_H / C.SY, cols_tool)
    ship = np.zeros((r1 - r0, C.O_W), label.dtype)
    ship_ok = np.zeros((r1 - r0, C.O_W), bool)
    if r0 < label.shape[0]:
        rr = min(r1, label.shape[0])
        ship[: rr - r0] = label[r0:rr, : C.O_W]
        ship_ok[: rr - r0] = True
    tool, tool_ok = {}, {}
    for name in runs:
        tool[name] = np.asarray(runs[name][r0:r1])
        v = np.asarray(valids[name][r0:r1])
        tool_ok[name] = v == 255
        values, counts = np.unique(v[region], return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            valid_hist[name][int(value)] = valid_hist[name].get(int(value), 0) + int(count)
    combos = {
        "offset~AF": (tool["offset"], tool_ok["offset"], af),
        "control~AF": (tool["control"], tool_ok["control"], af),
        "offset~AF_tool": (tool["offset"], tool_ok["offset"], af_tool),
        "loader~AF": (tool["loader"], tool_ok["loader"], af),
        "loader~AF_tool": (tool["loader"], tool_ok["loader"], af_tool),
        "SHIP~AF": (ship, ship_ok, af),
    }
    for key, (a, a_ok, b) in combos.items():
        sel = region & a_ok & af_ok
        av, bv = a[sel], b[sel]
        ia, ib = av > 0, bv > 0
        s = stats[key]
        s["compared"] += int(sel.sum())
        s["equal"] += int(np.count_nonzero(av == bv))
        s["ink_inter"] += int(np.count_nonzero(ia & ib))
        s["ink_union"] += int(np.count_nonzero(ia | ib))
        s["ink_a"] += int(ia.sum())
        s["ink_b"] += int(ib.sum())
    ink = (af > 0) & region  # BLOCK divides 2048, so a block lies in one coarse row
    coarse[r0 // 2048] += np.add.reduceat(ink.sum(axis=0), np.arange(0, C.O_W, 2048))
    if (r0 // BLOCK) % 8 == 0:
        print(f"  rows {r0}..{r1 - 1} done, {time.time() - t0:.0f}s", flush=True)

result = {
    "model": {"name": C.MODEL.name, "dy": C.MODEL.cy.tolist(), "dx": C.MODEL.cx.tolist()},
    "region_pixels": region_pixels,
    "pairs": {},
    "validity_in_region": valid_hist,
}
print(f"supervised region: {region_pixels} pixels", flush=True)
for key, s in stats.items():
    result["pairs"][key] = dict(
        s, fraction_equal=s["equal"] / max(1, s["compared"]), ink_iou=s["ink_inter"] / max(1, s["ink_union"])
    )
    print(f"{key}: compared {s['compared']}, fraction equal {s['equal'] / max(1, s['compared']):.6f}, "
          f"ink IoU {s['ink_inter'] / max(1, s['ink_union']):.4f} (ink in first {s['ink_a']}, in second {s['ink_b']})",
          flush=True)
print("validity counts inside the supervised region:", json.dumps(valid_hist), flush=True)
np.save(f"{C.OUT}/coarse_ink_2048.npy", coarse)
with open(f"{C.OUT}/compare_agreement.json", "w") as handle:
    json.dump(result, handle, indent=2)
print(f"total {time.time() - t0:.0f}s", flush=True)
