"""Integer pyramid-shift fit for an S3 (uncompressed) surface volume, middle slice only.
argv: key level y0 x0   -> writes pyrshift_<key>.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import s3, ledger_total
from lib import S3Image
from sources import S3SEG, HERE

key, k, y0, x0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
seg, mesh, sv, vol = S3SEG[key]
f, n = 2 ** k, 256
I0, Ik = S3Image(s3(f"{seg}/{sv}"), 0, [54]), S3Image(s3(f"{seg}/{sv}"), k, [54])
A0 = I0.region(y0 - f, y0 + n + f, x0 - f, x0 + n + f)[0].astype(float)
Ak = Ik.region(y0 // f, (y0 + n) // f, x0 // f, (x0 + n) // f)[0].astype(float)
print(f"{key}: level 0 {I0.full_shape}, level {k} {Ik.full_shape}, region y{y0} x{x0}, nonzero {np.mean(A0 > 0):.2f}")
res = []
for sy in range(0, f):
    for sx in range(0, f):
        S = A0[f - sy: f - sy + n, f - sx: f - sx + n]
        P = S.reshape(n // f, f, n // f, f).mean(axis=(1, 3))
        a, b = Ak[1:-1, 1:-1].ravel(), P[1:-1, 1:-1].ravel()
        res.append((np.corrcoef(a, b)[0, 1], np.abs(a - b).mean(), sy, sx))
res.sort(reverse=True)
for r, d, sy, sx in res[:3]:
    print(f"  shift (sy, sx)=({sy}, {sx}): r={r:.5f} mean|diff|={d:.2f}")
json.dump({"shift": [int(res[0][2]), int(res[0][3])], "r": float(res[0][0]), "level": k},
          open(os.path.join(HERE, f"pyrshift_{key}.json"), "w"))
print("ledger total bytes", ledger_total())
