"""Residual shift between transfer outputs and the apply_fit correction, per region.

shift s is defined by  A(u) ~= AF(u + s)  on the loader grid (rows, cols), measured by
linear FFT cross-correlation of Gaussian-blurred label crops with a parabolic sub-pixel peak.
Predictions use the same coordinate maps independently of the images.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

import compare_lib as C  # noqa: E402

WIN = 2048
coarse = np.load(f"{C.OUT}/coarse_ink_2048.npy")
label = C.load_label()
print("coarse ink grid", coarse.shape, "max block ink", int(coarse.max()), flush=True)

cand = np.argwhere(coarse >= max(20000, 0.2 * coarse.max()))
centre = np.array([C.O_H / 2048 / 2, C.O_W / 2048 / 2])
picks = {
    "far corner (max row+col)": cand[np.argmax(cand.sum(1))],
    "top-left (min row+col)": cand[np.argmin(cand.sum(1))],
    "top-right (max col-row)": cand[np.argmax(cand[:, 1] - cand[:, 0])],
    "bottom-left (max row-col)": cand[np.argmax(cand[:, 0] - cand[:, 1])],
    "centre": cand[np.argmin(np.linalg.norm(cand + 0.5 - centre, axis=1))],
}


shift = C.shift
runs = {"offset": C.open_tiff(C.OFFSET_TIF), "control": C.open_tiff(C.CONTROL_TIF), "loader": C.open_tiff(C.LOADER_TIF)}
report = {}
for name, (gi, gj) in picks.items():
    r0, c0 = int(gi) * WIN, int(gj) * WIN
    r1, c1 = min(C.O_H, r0 + WIN), min(C.O_W, c0 + WIN)
    rows = np.arange(r0, r1, dtype=np.float64)
    cols = np.arange(c0, c1, dtype=np.float64)
    af, _ = C.nearest(label, rows, cols)
    af_tool, _ = C.nearest(label, (np.arange(r0, r1) + 0.5) * C.S_H / C.O_H / C.SY,
                           (np.arange(c0, c1) + 0.5) * C.S_W / C.O_W / C.SX)
    ship = np.zeros_like(af)
    rr, cc = min(r1, label.shape[0]), min(c1, label.shape[1])
    if rr > r0 and cc > c0:
        ship[: rr - r0, : cc - c0] = label[r0:rr, c0:cc]
    crops = {
        "offset": np.asarray(runs["offset"][r0:r1, c0:c1]),
        "control": np.asarray(runs["control"][r0:r1, c0:c1]),
        "loader": np.asarray(runs["loader"][r0:r1, c0:c1]),
        "AF_tool": af_tool,
        "SHIP": ship,
    }
    ic, jc = (r0 + r1 - 1) / 2.0, (c0 + c1 - 1) / 2.0
    pred = {
        "offset": C.predict_tool(ic, jc, C.OFFSET_FULL),
        "control": C.predict_tool(ic, jc, (0.0, 0.0)),
        "loader": C.predict_tool(ic, jc, C.LOADER_FULL),
        "AF_tool": ((ic + 0.5) * C.S_H / C.O_H / C.SY - ic, (jc + 0.5) * C.S_W / C.O_W / C.SX - jc),
        "SHIP": tuple(float(v) for v in C.MODEL.offset(ic, jc)),
    }
    entry = {"window_rows": [r0, r1], "window_cols": [c0, c1], "ink_pixels_af": int((af > 0).sum())}
    print(f"\n{name}: rows {r0}..{r1 - 1}, cols {c0}..{c1 - 1}, ink px {entry['ink_pixels_af']}", flush=True)
    for key, crop in crops.items():
        s = shift(crop, af)
        entry[key] = {"measured": s, "predicted": [float(pred[key][0]), float(pred[key][1])]}
        print(f"  {key:8s} measured ({s[0]:+8.2f}, {s[1]:+8.2f})  predicted ({pred[key][0]:+8.2f}, {pred[key][1]:+8.2f})", flush=True)
    report[name] = entry
with open(f"{C.OUT}/compare_shifts.json", "w") as handle:
    json.dump(report, handle, indent=2)
