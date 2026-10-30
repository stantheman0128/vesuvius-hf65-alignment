"""Pyramid check: level-k pixel i vs mean of level-0 rows [f*i+s, f*i+s+f), s in -8..8, both axes;
and (for z-downsampled pyramids) which level-0 slices a level-k slice corresponds to."""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from canvas import Canvas
key, k, Y0, X0, N = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
f = 2 ** k
c0, ck = Canvas(key, 0), Canvas(key, k)
print(key, "L0", c0.shape, f"L{k}", ck.shape)
zsk = list(range(ck.shape[0]))
Ak = ck.read(zsk, Y0 // f, (Y0 + N) // f, X0 // f, (X0 + N) // f).astype(float)
z0s = list(range(c0.shape[0])) if ck.shape[0] != c0.shape[0] else [16, 32, 48]
A0 = c0.read(z0s, Y0 - 16, Y0 + N + 16, X0 - 16, X0 + N + 16).astype(float)
n = N // f
best = []
for zi, z0 in enumerate(z0s):
    for sy in range(-8, 9):
        for sx in range(-8, 9):
            S = A0[zi, 16 + sy:16 + sy + N, 16 + sx:16 + sx + N].reshape(n, f, n, f).mean(axis=(1, 3))
            for zk in (zsk if ck.shape[0] != c0.shape[0] else [zi_k for zi_k in [z0]]):
                a = Ak[zk, 1:-1, 1:-1]; b = S[1:-1, 1:-1]
                best.append((np.abs(a - b).mean(), z0, zk, sy, sx))
best.sort()
for r in best[:6]:
    print("  mean|diff| %.3f  L0 slice %d  L%d slice %d  shift rows %+d cols %+d" % (r[0], r[1], k, r[2], r[3], r[4]))
