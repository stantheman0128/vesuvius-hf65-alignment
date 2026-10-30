"""S3 surface-volume pyramid: level-2 slice k pixel i vs mean of level-0 slice k rows [4i+s, 4i+s+4)."""
import sys, json
sys.path.insert(0, sys.argv[1])
import numpy as np
import store
from geom import S3
base = S3 + sys.argv[2]
Y0, X0, N, k = int(sys.argv[3]), int(sys.argv[4]), 256, 54
def sl(level, i0, i1, j0, j1):
    za = json.loads(store.get(f"{base}/{level}/.zarray")); C = za["chunks"][1]
    out = np.zeros((i1 - i0, j1 - j0), np.uint8)
    for cy in range(i0 // C, (i1 - 1) // C + 1):
        for cx in range(j0 // C, (j1 - 1) // C + 1):
            c = np.frombuffer(store.get_range(f"{base}/{level}/0/{cy}/{cx}", k * C * C, (k + 1) * C * C - 1), np.uint8).reshape(C, C)
            a0, a1 = max(i0, cy * C), min(i1, cy * C + C); b0, b1 = max(j0, cx * C), min(j1, cx * C + C)
            out[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = c[a0 - cy * C:a1 - cy * C, b0 - cx * C:b1 - cx * C]
    return out.astype(float)
A0 = sl(0, Y0 - 16, Y0 + N + 16, X0 - 16, X0 + N + 16)
A2 = sl(2, Y0 // 4, (Y0 + N) // 4, X0 // 4, (X0 + N) // 4)
res = []
for sy in range(-8, 9):
    for sx in range(-8, 9):
        P = A0[16 + sy:16 + sy + N, 16 + sx:16 + sx + N].reshape(N // 4, 4, N // 4, 4).mean(axis=(1, 3))
        res.append((np.abs(P[1:-1, 1:-1] - A2[1:-1, 1:-1]).mean(), sy, sx))
res.sort()
print(sys.argv[2].split("/")[0], res[:3], "new bytes", store.new_bytes())
