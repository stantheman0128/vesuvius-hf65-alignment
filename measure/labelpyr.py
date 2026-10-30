"""How does a label zarr's level k relate to its level 0?  Integer shift fit for mean / max /
stride pooling. argv: zarr_path level y0 x0 n"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import ledger_total
from lib import Image

z, k, y0, x0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
f = 2 ** k
I0, Ik = Image(z, 0), Image(z, k)
sl = 32 if I0.shape[0] > 32 else 0
A0 = I0.region(y0 - f, y0 + n + f, x0 - f, x0 + n + f)[sl].astype(float)
Ak = Ik.region(y0 // f, (y0 + n) // f, x0 // f, (x0 + n) // f)[sl if Ik.shape[0] > sl else 0].astype(float)
print(f"{z}: level 0 {I0.shape} level {k} {Ik.shape}; values level0 {np.unique(A0)[:5]} level{k} {np.unique(Ak)[:8]};"
      f" level-0 frac>0 {np.mean(A0 > 0):.3f}")
res = []
for sy in range(0, f):
    for sx in range(0, f):
        S = A0[f - sy: f - sy + n, f - sx: f - sx + n].reshape(n // f, f, n // f, f)
        for nm, P in (("mean", S.mean(axis=(1, 3))), ("max", S.max(axis=(1, 3))), ("stride", S[:, 0, :, 0])):
            a, b = Ak[1:-1, 1:-1].ravel(), P[1:-1, 1:-1].ravel()
            r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
            res.append((r, np.abs(a - b).mean(), nm, sy, sx))
res.sort(key=lambda t: -np.nan_to_num(t[0], nan=-9))
for r, d, nm, sy, sx in res[:4]:
    print(f"  {nm} shift (sy, sx)=({sy}, {sx}): r={r:.5f} mean|diff|={d:.3f}")
print("ledger total bytes", ledger_total())
