"""Independent check of Surf.cells (bilinear, cell r -> stored r*scale) against villa's own
vesuvius.tifxyz.upsampling.interpolate_at_points (Catmull-Rom), which Tifxyz.__getitem__ calls with
source_grid = full-res index * scale (types.py:153-154 at 7154682e1). argv: key ..."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from vesuvius.tifxyz.upsampling import interpolate_at_points
import sources

for key in sys.argv[1:]:
    src = sources.get(key)
    S = src["surf"]
    reg = json.load(open(os.path.join(sources.HERE, f"regions_{key}.json")))[0]
    rows = reg["y0"] + np.arange(0, reg["size"], 20) + 0.5
    cols = reg["x0"] + np.arange(0, reg["size"], 20) + 0.5
    mine = S.cells(rows, cols)
    sr, sc = rows * S.sy, cols * S.sx
    r0, r1 = int(sr.min()) - 4, int(sr.max()) + 5
    c0, c1 = int(sc.min()) - 4, int(sc.max()) + 5
    g = S.stored_rows(r0, r1, c0, c1)
    valid = (g["x"] != -1) & (g["y"] != -1) & (g["z"] != -1)
    qy, qx = np.meshgrid(sr - r0, sc - c0, indexing="ij")
    vx, vy, vz, vv = interpolate_at_points(g["x"], g["y"], g["z"], valid, qy.astype(np.float32), qx.astype(np.float32),
                                           scale=(1.0, 1.0), method="catmull_rom")
    villa = np.stack([vz, vy, vx], -1).astype(float)
    ok = vv & np.all(np.isfinite(mine), -1)
    d = np.linalg.norm(villa[ok] - mine[ok], axis=-1)
    half = S.cells(rows + 0.5, cols + 0.5)
    dh = np.linalg.norm(half[ok] - mine[ok], axis=-1)
    print(f"{key}: {ok.sum()} points; |villa catmull-rom - mine bilinear| median {np.median(d):.3f} max {d.max():.3f} voxels;"
          f" for scale: moving the cell by +0.5 moves the point by median {np.median(dh):.3f} voxels")
