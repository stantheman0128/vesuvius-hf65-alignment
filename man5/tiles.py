"""Does the best MAN5 -> segment-grid shift vary across the slice? Level 1, slice 32, tiles of
T x T level-1 px over the MAN5 ink box; per tile with enough MAN5 ink, the integer level-1 shift
(-6..30 on both axes) maximising IoU with front_inklabels inside the tile. A pure crop gives the
same shift everywhere, up to how sharp each tile's IoU peak is (reported as the IoU drop 4 level-1
px away from the peak).  argv: [T]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "measure"))
import numpy as np
from place import Lab, grid_shape

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
GH, GW = grid_shape(1)
F = Lab("front", 1).slice2d(32, 0, GH, 0, GW) > 0
Lm = Lab("inklabels", 1)
M = Lm.slice2d(32, 0, Lm.shape[1], 0, Lm.shape[2]) > 0
lo, hi = -6, 30
ys, xs = np.nonzero(M)
rows = []
for ty in range(ys.min(), ys.max(), T):
    for tx in range(xs.min(), xs.max(), T):
        Mt = M[ty:ty + T, tx:tx + T]
        if Mt.sum() < 0.05 * Mt.size:
            continue
        best = None
        S = {}
        for sy in range(lo, hi + 1):
            for sx in range(lo, hi + 1):
                a, b = ty + sy, tx + sx
                if a < 0 or b < 0 or a + Mt.shape[0] > GH or b + Mt.shape[1] > GW:
                    continue
                Ft = F[a:a + Mt.shape[0], b:b + Mt.shape[1]]
                i = np.count_nonzero(Mt & Ft)
                u = np.count_nonzero(Mt | Ft)
                S[(sy, sx)] = i / u
        (by, bx), bv = max(S.items(), key=lambda t: t[1])
        drop = bv - np.mean([S.get((by + dy, bx + dx), np.nan) for dy, dx in ((4, 0), (-4, 0), (0, 4), (0, -4))])
        edge = by in (lo, hi) or bx in (lo, hi)
        rows.append((ty, tx, by, bx, bv, drop, edge, Mt.mean()))
        print(f"tile y {ty}..{ty + T} x {tx}..{tx + T} (level-1 label px), MAN5 ink {Mt.mean():.2f}: best level-1 shift ({by}, {bx})"
              f" = level-0 ({2 * by}, {2 * bx}), IoU {bv:.3f}, IoU drop at +-4 px {drop:.4f}, on search edge {edge}", flush=True)
R = np.array([(r[2], r[3]) for r in rows if not r[6]], float)
print(f"\n{len(rows)} tiles, {len(R)} not on the search edge; level-1 best shift dy median {np.median(R[:, 0]):.1f} range {R[:, 0].min():.0f}..{R[:, 0].max():.0f},"
      f" dx median {np.median(R[:, 1]):.1f} range {R[:, 1].min():.0f}..{R[:, 1].max():.0f}")
