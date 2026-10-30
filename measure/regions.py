"""Pick 1024x1024 level-0 canvas regions (aligned to 512 px) spread over a segment, where the
tifxyz grid is valid around the region (plus search margin) and the supervision mask is set.
argv: key  -> writes regions_<key>.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import fetch, hf, ledger_total, MIRROR
from lib import Image, Surf

SEGS = {
    "w00": ("phercparis4", "w00_20231016151002", "w00_20231016151002_supervision_mask.zarr", "w00_20231016151002.zarr"),
    "w01": ("phercparis4", "w01_20230702185753_r15", "w01_20230702185753_r15_supervision_mask.zarr", "w01_20230702185753_r15.zarr"),
    "w05": ("phercparis4", "w05_4424", "4424_supervision_mask.zarr", "w05_4424.zarr"),
    "c018": ("1667", "w018_20240304144031_2um", "w018_20240304144031_2um_supervision_mask.zarr", "w018_20240304144031_2um.zarr"),
}
BLOCK = 1024      # level-0 canvas px per region side
STEP = 512        # candidate alignment (level-2 image chunk = 512 level-0 px)
MARGIN = 160      # level-0 px of tifxyz grid needed around the region for the shift search
NREG = int(os.environ.get("NREG", 6))


def main(key):
    scroll, seg, sup, img = SEGS[key]
    d = f"ink/{scroll}/{seg}"
    local = os.path.join(MIRROR, "r4", scroll, seg)
    S = Surf(d, local if os.path.isdir(local) else None)
    # tifxyz valid mask on a coarse grid of stored rows (every `rs`-th row)
    rs = 1 if os.path.isdir(local) else 16
    rows = np.arange(0, S.H, rs)
    if rs == 1:
        X = S.t["x"].rows(0, S.H, 0, S.W)
    else:
        X = np.stack([S.t["x"].rows(r, r + 1, 0, S.W)[0] for r in rows])
    V = (X != -1) & np.isfinite(X)
    print(f"{key}: stored {S.H}x{S.W} scale (y {S.sy:.6f}, x {S.sx:.6f}) full {S.full}; valid frac {V.mean():.3f}")
    # supervision mask at level 5 (any slice)
    M = Image(f"{d}/{sup}", 5)
    sm = M.region(0, M.shape[1], 0, M.shape[2]).max(axis=0) > 0
    im = Image(f"{d}/{img}", 0)
    H0, W0 = im.shape[1], im.shape[2]
    print(f"   image level-0 {H0}x{W0}; supervision mask level 5 {sm.shape} set frac {sm.mean():.3f}")
    cand = []
    for y0 in range(0, H0 - BLOCK + 1, STEP):
        for x0 in range(0, W0 - BLOCK + 1, STEP):
            # tifxyz validity over the region + margin (full-res cell = canvas px)
            ry0, ry1 = (y0 - MARGIN) * S.sy, (y0 + BLOCK + MARGIN) * S.sy
            cx0, cx1 = (x0 - MARGIN) * S.sx, (x0 + BLOCK + MARGIN) * S.sx
            if ry0 < 0 or cx0 < 0 or ry1 >= S.H - 1 or cx1 >= S.W - 1:
                continue
            ri = (rows >= ry0) & (rows <= ry1)
            v = V[np.ix_(ri, np.arange(int(cx0), int(cx1) + 1))].mean() if ri.any() else 0
            s5 = sm[y0 // 32:(y0 + BLOCK) // 32, x0 // 32:(x0 + BLOCK) // 32].mean()
            if v >= 0.98 and s5 >= 0.9:
                cand.append((y0, x0, float(v), float(s5)))
    print(f"   candidates: {len(cand)}")
    if not cand:
        return
    P = np.array([(c[0] + BLOCK / 2, c[1] + BLOCK / 2) for c in cand], float)
    ctr = P.mean(0)
    chosen = [int(np.argmin(((P - ctr) ** 2).sum(1)))]
    while len(chosen) < min(NREG, len(cand)):
        dmin = np.min(((P[:, None, :] - P[chosen][None, :, :]) ** 2).sum(-1), axis=1)
        chosen.append(int(np.argmax(dmin)))
    out = [{"y0": cand[i][0], "x0": cand[i][1], "size": BLOCK, "tifxyz_valid": cand[i][2], "supmask": cand[i][3]} for i in chosen]
    for r in out:
        print("   ", r)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"regions_{key}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("ledger total bytes", ledger_total())


if __name__ == "__main__":
    main(sys.argv[1])
