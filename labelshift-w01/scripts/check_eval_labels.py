"""Check the eval label folders of the eval offset scan (SCAN-2026-11-01.md) before any GPU time.

usage: python -I check_eval_labels.py EVAL_ROOT MANIFEST_FULL_JSON RAW_ROOT REGIONS_JSON

1. E0's three zarrs are byte-identical to build_labels.py's C0 (sha256 in MANIFEST_FULL_JSON).
2. EkY / EkX: at level 0 (slice 32), in a band of 256 px around each held-out box and its moved copy,
   validation_mask and inklabels equal E0's moved by k cells inside the moved box. Outside it the validation
   mask is zero and the ink labels are E0's. The level-0 validation count equals E0's.
3. EAS: inside each placed box, validation_mask and inklabels equal the shipped canvas supervision and ink at
   the same (y, x), and validation_mask is zero in the band outside the placed boxes.
Prints one line per check, then ALL_OK or SOME_FAILED.
"""
import json
import os
import sys

import numpy as np
import zarr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from labellib import read_label_tif  # noqa: E402

root, manifest_full, raw_root, regions = sys.argv[1:5]
E = json.load(open(os.path.join(root, "manifest.json")))
F = json.load(open(manifest_full))
R = json.load(open(regions))
unit = int(R["shift_cell_level0_px"])
ok = True


def report(good, text):
    global ok
    ok &= bool(good)
    print(("ok   " if good else "FAIL ") + text, flush=True)


def level0(arm, S, kind, y0, y1, x0, x1):
    p = os.path.join(root, arm, S["scroll"], S["segment"], f"{S['segment']}_{kind}.zarr")
    return np.asarray(zarr.open_group(p, mode="r", zarr_format=2)["0"][32, y0:y1, x0:x1])


for key, S in R["segments"].items():
    m = E["segments"][key]
    H, W = m["tifxyz_full_shape"]
    for kind in ("inklabels", "supervision_mask", "validation_mask"):
        report(m["sha256"]["E0"][kind] == F["segments"][key]["sha256"]["C0"][kind], f"{key} E0 {kind} sha256 == C0")
    cnt = m["nonzero_per_level"]
    raw = None
    for arm in m["placed_boxes"]:
        if arm == "E0":
            continue
        if arm != "EAS":
            k = int(arm[1:-1]) * unit
            dy, dx = (k, 0) if arm[-1] == "y" else (0, k)
            report(cnt[arm]["validation_mask"][0] == cnt["E0"]["validation_mask"][0],
                   f"{key} {arm} level-0 validation count {cnt[arm]['validation_mask'][0]} == E0")
            for (y0, y1, x0, x1), (py0, py1, px0, px1) in zip(m["heldout_tifxyz_boxes"], m["placed_boxes"][arm]):
                b0, b1, c0, c1 = max(0, y0 - 256), min(H, y1 + dy + 256), max(0, x0 - 256), min(W, x1 + dx + 256)
                for kind in ("validation_mask", "inklabels"):
                    a, e = level0(arm, S, kind, b0, b1, c0, c1), level0("E0", S, kind, b0, b1, c0, c1)
                    inside = np.zeros(a.shape, bool)
                    inside[py0 - b0:py1 - b0, px0 - c0:px1 - c0] = True
                    moved = np.zeros_like(e)  # E0 moved by (dy, dx) within the band
                    moved[dy:, dx:] = e[:e.shape[0] - dy, :e.shape[1] - dx]
                    same_in = np.array_equal(a[inside], moved[inside])
                    # outside the moved box: no validation at all, and C0's ink labels as in E0
                    same_out = (not a[~inside].any()) if kind == "validation_mask" else np.array_equal(a[~inside], e[~inside])
                    report(same_in and same_out, f"{key} {arm} {kind} box {y0},{x0}: moved inside {same_in}, "
                           f"outside as required {same_out}, nonzero inside {int(np.count_nonzero(a[inside]))}")
        else:
            if raw is None:
                d = os.path.join(raw_root, S.get("bucket_scroll", S["scroll"]), S["segment"])
                raw = {"inklabels": read_label_tif(os.path.join(d, S["ink_tif"])),
                       "validation_mask": read_label_tif(os.path.join(d, S["sup_tif"]))}
            for (py0, py1, px0, px1) in m["placed_boxes"][arm]:
                b0, b1, c0, c1 = max(0, py0 - 256), min(H, py1 + 256), max(0, px0 - 256), min(W, px1 + 256)
                for kind in ("validation_mask", "inklabels"):
                    a = level0(arm, S, kind, b0, b1, c0, c1)
                    src = raw[kind]
                    want = np.zeros_like(a)
                    hh, ww = min(b1, src.shape[0]) - b0, min(c1, src.shape[1]) - c0
                    want[:hh, :ww] = src[b0:b0 + hh, c0:c0 + ww]
                    inside = np.zeros(a.shape, bool)
                    inside[py0 - b0:py1 - b0, px0 - c0:px1 - c0] = True
                    same_in = np.array_equal(a[inside], want[inside])
                    if kind == "validation_mask":
                        rest = not a[~inside].any()
                        report(same_in and rest, f"{key} EAS {kind} box {py0},{px0}: as shipped inside {same_in}, "
                               f"zero outside {rest}, nonzero inside {int(np.count_nonzero(a[inside]))}")
                    else:
                        report(same_in, f"{key} EAS {kind} box {py0},{px0}: as shipped inside {same_in}")
    del raw
print("ALL_OK" if ok else "SOME_FAILED")
