"""Pick a fully valid 1024 px block far from a given point, from cached tifxyz rows (every 16th stored row).
usage: vfar.py <key> <y_ref> <x_ref>"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))  # audit/independent/scripts
AUDIT = os.path.dirname(os.path.dirname(HERE))  # audit/
sys.path.insert(0, HERE)
import numpy as np
from vcore import Tifxyz, hf, mine, BS
T = json.load(open(os.path.join(AUDIT, "code", "targets.json")))
key, yr, xr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
seg = T[key]["seg"]
S = Tifxyz(lambda nm: hf(f"{seg}/{nm}"))
rows = np.arange(0, S.H, 16)
X = np.stack([S.t["x"].block(r, r + 1, 0, S.W)[0] for r in rows])
V = np.isfinite(X) & (X != -1)
print(key, "stored", S.H, S.W, "scale", S.sy, S.sx, "valid", V.mean(), "dl so far", mine())
cand = []
B = 1024
for y0 in range(0, int(S.H / S.sy) - B, 256):
    for x0 in range(0, int(S.W / S.sx) - B, 256):
        r0, r1 = (y0 - 160) * S.sy, (y0 + B + 160) * S.sy
        c0, c1 = int((x0 - 160) * S.sx), int((x0 + B + 160) * S.sx)
        if r0 < 0 or c0 < 0 or r1 >= S.H - 1 or c1 >= S.W - 1:
            continue
        ri = (rows >= r0) & (rows <= r1)
        if ri.sum() < 2:
            continue
        v = V[ri][:, c0:c1 + 1].mean()
        if v >= 0.999:
            cand.append((float(np.hypot(y0 - yr, x0 - xr)), y0, x0))
cand.sort(reverse=True)
print("n cand", len(cand), "farthest", cand[:5])
