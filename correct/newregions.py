"""Pick new regions the way measure/regions.py does, then continue its farthest-point selection.

Same candidates as regions.py (1024 px level-0 canvas blocks on a 512 px grid, tifxyz valid >= 0.98
over the block + 160 px margin on a 16-row coarse grid, supervision mask level 5 >= 0.9). The first
NREG=6 picks are recomputed and must equal measure/regions_<key>.json; picks 7, 8, ... are new.
One added filter: the tifxyz must also be valid (>= 0.98) around the block moved by the README
offset at its centre, since that is where the corrected sampling lands (w01 moves up to 254 px).
Also prints the same two numbers for an explicit block (the held-out w01 region).
argv: key [n_new] [--at Y0,X0]   -> writes correct/regions_new_<key>.json (unless --at)"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "measure"))
sys.path.insert(0, HERE)
import numpy as np
import common
import rangecache
from lib import Image, Surf
import regions as R0
import apply_fit


def main(key, n_new, at=None):
    scroll, seg, sup, img = R0.SEGS[key]
    d = f"ink/{scroll}/{seg}"
    S = Surf(d, None)
    rs = 16
    rows = np.arange(0, S.H, rs)
    X = np.stack([S.t["x"].rows(r, r + 1, 0, S.W)[0] for r in rows])
    V = (X != -1) & np.isfinite(X)
    M = Image(f"{d}/{sup}", 5)
    sm = M.region(0, M.shape[1], 0, M.shape[2]).max(axis=0) > 0
    im = Image(f"{d}/{img}", 0)
    H0, W0 = im.shape[1], im.shape[2]
    mdl = apply_fit.model(key)
    B, MG = R0.BLOCK, R0.MARGIN

    def tif_valid(y0, x0, dy=0.0, dx=0.0):
        ry0, ry1 = (y0 + dy - MG) * S.sy, (y0 + dy + B + MG) * S.sy
        cx0, cx1 = (x0 + dx - MG) * S.sx, (x0 + dx + B + MG) * S.sx
        if ry0 < 0 or cx0 < 0 or ry1 >= S.H - 1 or cx1 >= S.W - 1:
            return 0.0
        ri = (rows >= ry0) & (rows <= ry1)
        return float(V[np.ix_(ri, np.arange(int(cx0), int(cx1) + 1))].mean()) if ri.any() else 0.0

    def sup_frac(y0, x0):
        return float(sm[y0 // 32:(y0 + B) // 32, x0 // 32:(x0 + B) // 32].mean())

    if at is not None:
        y0, x0 = at
        dy, dx = mdl.offset(y0 + B / 2, x0 + B / 2)
        print(f"{key} block y0={y0} x0={x0} (centre {y0 + B // 2},{x0 + B // 2}): tifxyz valid {tif_valid(y0, x0):.3f},"
              f" at README-shifted position ({dy:+.1f},{dx:+.1f}) {tif_valid(y0, x0, dy, dx):.3f}; supervision mask level-5 frac {sup_frac(y0, x0):.3f}")
        return
    cand = []
    for y0 in range(0, H0 - B + 1, R0.STEP):
        for x0 in range(0, W0 - B + 1, R0.STEP):
            v = tif_valid(y0, x0)
            s5 = sup_frac(y0, x0)
            if v >= 0.98 and s5 >= 0.9:
                cand.append((y0, x0, v, s5))
    print(f"{key}: candidates {len(cand)} (regions.py criteria)")
    P = np.array([(c[0] + B / 2, c[1] + B / 2) for c in cand], float)
    ctr = P.mean(0)
    chosen = [int(np.argmin(((P - ctr) ** 2).sum(1)))]
    while len(chosen) < R0.NREG:
        dmin = np.min(((P[:, None, :] - P[chosen][None, :, :]) ** 2).sum(-1), axis=1)
        chosen.append(int(np.argmax(dmin)))
    old = json.load(open(os.path.join(HERE, "..", "measure", f"regions_{key}.json")))
    same = [(cand[i][0], cand[i][1]) for i in chosen] == [(r["y0"], r["x0"]) for r in old]
    print(f"   first {R0.NREG} picks equal measure/regions_{key}.json: {same}")
    if not same:
        print("   recomputed:", [(cand[i][0], cand[i][1]) for i in chosen])
        print("   published :", [(r["y0"], r["x0"]) for r in old])
        chosen = [next(i for i, c in enumerate(cand) if (c[0], c[1]) == (r["y0"], r["x0"])) for r in old]
    new = []
    excluded = set()
    while len(new) < n_new:
        dmin = np.min(((P[:, None, :] - P[chosen][None, :, :]) ** 2).sum(-1), axis=1)
        dmin[list(excluded)] = -1
        i = int(np.argmax(dmin))
        if dmin[i] < 0:
            break
        y0, x0, v, s5 = cand[i]
        dy, dx = mdl.offset(y0 + B / 2, x0 + B / 2)
        vs = tif_valid(y0, x0, dy, dx)
        overlap = any(abs(y0 - r["y0"]) < B and abs(x0 - r["x0"]) < B for r in old)
        print(f"   pick y0={y0} x0={x0} dist to nearest chosen {np.sqrt(dmin[i]):.0f} px; tifxyz valid {v:.3f},"
              f" shifted ({dy:+.1f},{dx:+.1f}) {vs:.3f}; supmask {s5:.3f}; overlaps a published region: {overlap}")
        if vs < 0.98 or overlap:
            excluded.add(i)
            continue
        chosen.append(i)
        new.append({"y0": y0, "x0": x0, "size": B, "tifxyz_valid": v, "tifxyz_valid_shifted": vs, "supmask": s5,
                    "pick": len(chosen), "min_dist_to_published_px": float(np.sqrt(dmin[i]))})
    with open(os.path.join(HERE, f"regions_new_{key}.json"), "w") as f:
        json.dump(new, f, indent=1)
    print(f"   wrote regions_new_{key}.json: {[(r['y0'], r['x0']) for r in new]}")


if __name__ == "__main__":
    rangecache.install()
    a = sys.argv[1:]
    at = None
    if "--at" in a:
        i = a.index("--at")
        at = tuple(int(v) for v in a[i + 1].split(","))
        a = a[:i] + a[i + 2:]
    main(a[0], int(a[1]) if len(a) > 1 else 2, at)
    print("process ledger bytes", common.ledger_total())
