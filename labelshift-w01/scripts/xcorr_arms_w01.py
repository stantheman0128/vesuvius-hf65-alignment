"""Second, independent check that the built C0 arm is the AS arm moved by the HF65 w01 offset.

usage: python -I xcorr_arms_w01.py LABEL_ROOT COMPONENTS_LOG REGIONS_JSON APPLY_FIT_DIR
For every training component (not held out), take the same tifxyz window from the AS and C0 inklabels
(level 0, slice 32), the component's canvas box padded by 320 px, and find the integer shift that best
aligns them by FFT cross-correlation. Prediction: C0 = AS moved by (dy, dx) of the fit at the box centre.
"""
import json, os, sys
import numpy as np
import zarr
root, comp_log, regions, afdir = sys.argv[1:5]
sys.path.insert(0, afdir)
import apply_fit
R = json.load(open(regions)); S = R["segments"]["w01"]
model = apply_fit.model("w01")
held = set(S["heldout"])
seg = S["segment"]
z = {arm: zarr.open_array(os.path.join(root, arm, S["scroll"], seg, f"{seg}_inklabels.zarr", "0"), mode="r") for arm in ("AS", "C0")}
H, W = z["AS"].shape[1:]
P = 320
worst = 0.0
rows = 0
for line in open(comp_log):
    if not line.startswith("{"):
        continue
    c = json.loads(line)
    if str(c["id"]) in held:
        continue
    y0, y1 = max(0, c["y0"] - P), min(H, c["y1"] + P)
    x0, x1 = max(0, c["x0"] - P), min(W, c["x1"] + P)
    A = (np.asarray(z["AS"][32, y0:y1, x0:x1]) > 0).astype(np.float32)
    B = (np.asarray(z["C0"][32, y0:y1, x0:x1]) > 0).astype(np.float32)
    cc = np.fft.irfft2(np.fft.rfft2(B) * np.conj(np.fft.rfft2(A)), s=A.shape)
    sy, sx = np.unravel_index(np.argmax(cc), cc.shape)
    sy = sy - A.shape[0] if sy > A.shape[0] // 2 else sy
    sx = sx - A.shape[1] if sx > A.shape[1] // 2 else sx
    dy, dx = model.offset((c["y0"] + c["y1"]) / 2, (c["x0"] + c["x1"]) / 2)
    err = max(abs(sy - dy), abs(sx - dx))
    worst = max(worst, err)
    rows += 1
    print(f"component {c['id']:2d}: measured shift ({sy:+d}, {sx:+d}) px, fit offset at centre ({float(dy):+.1f}, {float(dx):+.1f}), "
          f"difference {sy - float(dy):+.1f}, {sx - float(dx):+.1f}; overlap at best shift {cc.max() / max(A.sum(), 1):.3f} of AS ink", flush=True)
print(f"{rows} training components; largest |measured - predicted| {worst:.1f} px")
