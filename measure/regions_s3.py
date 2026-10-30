"""Regions for the S3 pipeline controls (S3 mesh + S3 surface volume rendered from it).
argv: key [n]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import s3, ledger_total
from lib import S3Image, surf_from_url
from sources import S3SEG, HERE

key = sys.argv[1]
NREG = int(sys.argv[2]) if len(sys.argv) > 2 else 4
seg, mesh, sv, vol = S3SEG[key]
S = surf_from_url(s3(f"{seg}/{mesh}"))
rows = np.arange(0, S.H, 16)
X = np.stack([S.t["x"].rows(r, r + 1, 0, S.W)[0] for r in rows])
V = (X != -1) & np.isfinite(X)
im5 = S3Image(s3(f"{seg}/{sv}"), 5, [54])
I5 = im5.region(0, im5.full_shape[1], 0, im5.full_shape[2])[0] > 0
H0, W0 = S3Image(s3(f"{seg}/{sv}"), 0, [54]).full_shape[1:]
print(f"{key}: mesh stored {S.H}x{S.W} scale ({S.sy}, {S.sx}) full {S.full}; image level-0 {H0}x{W0}; level-5 nonzero {I5.mean():.3f}")
BLOCK, STEP, MARGIN = 1024, 512, 160
cand = []
for y0 in range(0, H0 - BLOCK + 1, STEP):
    for x0 in range(0, W0 - BLOCK + 1, STEP):
        ry0, ry1 = (y0 - MARGIN) * S.sy, (y0 + BLOCK + MARGIN) * S.sy
        cx0, cx1 = (x0 - MARGIN) * S.sx, (x0 + BLOCK + MARGIN) * S.sx
        if ry0 < 0 or cx0 < 0 or ry1 >= S.H - 1 or cx1 >= S.W - 1:
            continue
        ri = (rows >= ry0) & (rows <= ry1)
        v = V[np.ix_(ri, np.arange(int(cx0), int(cx1) + 1))].mean() if ri.any() else 0
        nz = I5[y0 // 32:(y0 + BLOCK) // 32, x0 // 32:(x0 + BLOCK) // 32].mean()
        if v >= 0.99 and nz >= 0.99:
            cand.append((y0, x0, float(v), float(nz)))
print("candidates", len(cand))
P = np.array([(c[0], c[1]) for c in cand], float)
chosen = [int(np.argmin(((P - P.mean(0)) ** 2).sum(1)))]
while len(chosen) < min(NREG, len(cand)):
    dmin = np.min(((P[:, None, :] - P[chosen][None, :, :]) ** 2).sum(-1), axis=1)
    chosen.append(int(np.argmax(dmin)))
out = [{"y0": cand[i][0], "x0": cand[i][1], "size": BLOCK, "tifxyz_valid": cand[i][2], "img_nonzero_l5": cand[i][3]} for i in chosen]
for r in out:
    print("  ", r)
json.dump(out, open(os.path.join(HERE, f"regions_{key}.json"), "w"), indent=1)
print("ledger total bytes", ledger_total())
