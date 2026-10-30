"""Integer-shift fit of an HF65 pyramid level against level 0 (slice 32 unless the level is
depth-truncated, in which case the same slice index is used):
level-k pixel (i, j) == mean of level-0 pixels [f*i - sy, f*i - sy + f) x [f*j - sx, f*j - sx + f) ?
argv: zarr_path level y0 x0 [n]   (y0, x0 in level-0 px, multiples of 128)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import ledger_total
from lib import Image

z, k, y0, x0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
n = int(sys.argv[5]) if len(sys.argv) > 5 else 256
f = 2 ** k
I0, Ik = Image(z, 0), Image(z, k)
sl = 32
A0 = Ik_slice = None
A0 = I0.region(y0 - f, y0 + n + f, x0 - f, x0 + n + f)[sl].astype(float)
Ak = Ik.region(y0 // f, (y0 + n) // f, x0 // f, (x0 + n) // f)[sl if Ik.shape[0] > sl else 0].astype(float)
print(f"{z}: level 0 {I0.shape}, level {k} {Ik.shape}, region y{y0} x{x0} n{n}, level-0 nonzero {np.mean(A0 > 0):.2f}")
res = []
for sy in range(0, f):
    for sx in range(0, f):
        S = A0[f - sy: f - sy + n, f - sx: f - sx + n]
        P = S.reshape(n // f, f, n // f, f).mean(axis=(1, 3))
        a, b = Ak[1:-1, 1:-1].ravel(), P[1:-1, 1:-1].ravel()
        r = np.corrcoef(a, b)[0, 1]
        res.append((r, np.abs(a - b).mean(), sy, sx))
res.sort(reverse=True)
for r, d, sy, sx in res[:3]:
    print(f"  shift (sy, sx)=({sy}, {sx}) level-0 px: r={r:.5f} mean|diff|={d:.2f}")
r, d, sy, sx = res[0]
print(f"  => level-{k} pixel i covers level-0 [{f}i-{sy}, {f}i-{sy}+{f}) rows, [{f}j-{sx}, {f}j-{sx}+{f}) cols;"
      f" centre at level-0 ({f}i+{(f - 1) / 2 - sy:.1f}, {f}j+{(f - 1) / 2 - sx:.1f})")
print("ledger total bytes", ledger_total())
if len(sys.argv) > 6:
    import json
    json.dump({"shift": [int(sy), int(sx)], "r": float(r), "level": k, "at": [y0, x0]},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"pyrshift_{sys.argv[6]}.json"), "w"))
    print("wrote", f"pyrshift_{sys.argv[6]}.json")
