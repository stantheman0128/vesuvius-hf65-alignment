"""Second shift estimate per region: ink IoU of A(u) against AF(u + s) over integer s,
with a parabolic peak per axis. Same windows as cc.py (read from compare_shifts.json)."""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import compare_lib as C  # noqa: E402

regions = json.load(open(f"{C.OUT}/compare_shifts.json"))
label = C.load_label()
runs = {"offset": C.open_tiff(C.OFFSET_TIF), "control": C.open_tiff(C.CONTROL_TIF), "loader": C.open_tiff(C.LOADER_TIF)}
M = 320  # margin so AF covers every tested shift
out = {}
for name, entry in regions.items():
    r0, r1 = entry["window_rows"]
    c0, c1 = entry["window_cols"]
    big_r = np.arange(r0 - M, r1 + M, dtype=np.float64)
    big_c = np.arange(c0 - M, c1 + M, dtype=np.float64)
    af_big, _ = C.nearest(label, big_r, big_c)
    af_big = af_big > 0
    ship = np.zeros((r1 - r0, c1 - c0), bool)
    rr, cc = min(r1, label.shape[0]), min(c1, label.shape[1])
    ship[: rr - r0, : cc - c0] = label[r0:rr, c0:cc] > 0
    af_tool, _ = C.nearest(label, (np.arange(r0, r1) + 0.5) * C.S_H / C.O_H / C.SY,
                           (np.arange(c0, c1) + 0.5) * C.S_W / C.O_W / C.SX)
    crops = {
        "offset": np.asarray(runs["offset"][r0:r1, c0:c1]) > 0,
        "control": np.asarray(runs["control"][r0:r1, c0:c1]) > 0,
        "loader": np.asarray(runs["loader"][r0:r1, c0:c1]) > 0,
        "AF_tool": af_tool > 0,
        "SHIP": ship,
    }
    print(f"\n{name}", flush=True)
    out[name] = {}
    for key, a in crops.items():
        centre = np.rint(entry[key]["measured"]).astype(int)
        span = range(-4, 5)
        iou = np.zeros((len(span), len(span)))
        for p, dy in enumerate(span):
            for q, dx in enumerate(span):
                sy, sx = centre[0] + dy, centre[1] + dx
                b = af_big[M + sy: M + sy + a.shape[0], M + sx: M + sx + a.shape[1]]
                iou[p, q] = np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))
        p, q = np.unravel_index(int(np.argmax(iou)), iou.shape)
        est = []
        for axis, k in ((0, p), (1, q)):
            if 0 < k < len(span) - 1:
                v = iou[k - 1, q] if axis == 0 else iou[p, k - 1], iou[p, q], iou[k + 1, q] if axis == 0 else iou[p, k + 1]
                d = v[0] - 2 * v[1] + v[2]
                sub = 0.5 * (v[0] - v[2]) / d if abs(d) > 1e-12 else 0.0
            else:
                sub = float("nan")
            est.append(float(centre[axis] + span[k] + sub))
        zero = None
        if abs(centre[0]) <= 4 and abs(centre[1]) <= 4:
            zero = float(iou[4 - centre[0], 4 - centre[1]])
        out[name][key] = {"iou_peak_shift": est, "iou_at_peak": float(iou[p, q]), "iou_at_zero_shift": zero}
        print(f"  {key:8s} IoU-peak shift ({est[0]:+8.2f}, {est[1]:+8.2f}), IoU at peak {iou[p, q]:.4f}"
              + (f", IoU at zero shift {zero:.4f}" if zero is not None else ""), flush=True)
with open(f"{C.OUT}/compare_shifts_iou.json", "w") as handle:
    json.dump(out, handle, indent=2)
