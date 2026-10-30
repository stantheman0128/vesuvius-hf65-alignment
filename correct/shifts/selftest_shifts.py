"""Self-test of apply_shifts.py on synthetic tiles (no data needed).

1. Every row of shifts.json resolves by key, folder, scroll/folder and bucket path; the five aligned
   segments resolve to the identity; unknown and not-measured names raise.
2. For the four segments that apply_fit.py already corrects, apply_shifts.model equals apply_fit.model("readme").
3. fwd(inv(t)) = t over each segment's tifxyz extent (max error printed).
4. resample_tile on one segment per group: single labelled pixels (order 0) land at round(fwd(y, x)) / 2^K,
   at levels 0 and 2; a canvas -> tifxyz -> canvas round trip with nearest sampling.
5. Aligned segment: resample_tile returns the tile unchanged.

python -I correct/shifts/selftest_shifts.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import apply_fit
import apply_shifts as A

ok_all = True


def check(cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg)


doc = A.doc()
check(len(doc["segments"]) == 26, f"{len(doc['segments'])} offset rows")
for r in doc["segments"]:
    names = [f"{r['scroll']}/{r['segment']}", r["bucket_path"], r["segment"]]
    if r["key"] != "w00":   # key "w00" collides with the folder 841/w00 and is rejected as ambiguous (tested below)
        names.append(r["key"])
    got = {A.lookup(n)[1]["bucket_path"] for n in names}
    if got != {r["bucket_path"]}:
        check(False, f"lookup {names} -> {got}")
check(True, "every row resolves by key, folder, scroll/folder, bucket path to itself")
for a in doc["aligned_no_correction"]:
    m = A.model(a["bucket_path"])
    check(not (m.cy.any() or m.cx.any()), f"aligned {a['bucket_path']} -> identity")
for bad in ("no_such_segment", "814_46527_2um_try2", "w00"):
    try:
        A.lookup(bad)
        check(False, f"{bad!r} should raise")
    except KeyError as e:
        check(True, f"{bad!r} raises: {e}")

for k in ("w00", "w01", "w05", "c018"):
    m1 = A.model(f"{apply_fit.FITS[k]['segment']}")
    m2 = apply_fit.model(k, "readme")
    check(np.array_equal(m1.cy, m2.cy) and np.array_equal(m1.cx, m2.cx), f"{k}: shifts row equals apply_fit FITS")

worst = 0.0
for r in doc["segments"]:
    m = A.model(r["bucket_path"])
    Ht, Wt = r["tifxyz_full_hw"]
    ty, tx = np.meshgrid(np.linspace(-300, Ht + 300, 129), np.linspace(-300, Wt + 300, 129), indexing="ij")
    y, x = m.inv(ty, tx)
    fy, fx = m.fwd(y, x)
    worst = max(worst, float(np.abs(fy - ty).max()), float(np.abs(fx - tx).max()))
check(worst < 1e-6, f"fwd(inv(t)) - t max {worst:.2e} px over all 26 tifxyz extents")

rng = np.random.default_rng(0)
for seg in ("1667/w013_20240304141531_2um", "0139/w040_2025083102", "phercparis4/w07_1619"):
    m = A.model(seg)
    for K in (0, 2):
        s = 2 ** K
        n = 512
        Y0, X0 = 4000 // s, 9000 // s
        tile = np.zeros((n, n), np.uint8)
        pts = rng.integers(40, n - 40, size=(30, 2))
        for i, (a, b) in enumerate(pts):
            tile[a, b] = i + 1
        out, okm, (oy, ox) = A.resample_tile(seg, tile, (Y0, X0), level=K, order=0)
        bad = 0
        for i, (a, b) in enumerate(pts):
            cy, cx = s * (Y0 + a) + (s - 1) / 2, s * (X0 + b) + (s - 1) / 2   # level-0 centre of the labelled pixel
            ty, tx = m.fwd(cy, cx)
            r_, c_ = np.argwhere(out == i + 1).T
            # expected tifxyz level-K cell: centre s*T + (s-1)/2 nearest to fwd
            ey, ex = np.rint((ty - (s - 1) / 2) / s) - oy, np.rint((tx - (s - 1) / 2) / s) - ox
            if not (len(r_) >= 1 and np.all(np.abs(r_ - ey) <= 1) and np.all(np.abs(c_ - ex) <= 1)
                    and any((rr == ey and cc == ex) for rr, cc in zip(r_, c_))):
                bad += 1
        check(bad == 0, f"{seg} level {K}: 30 labelled pixels land at round(fwd) ({bad} off)")
    # round trip at level 2: canvas tile -> tifxyz -> canvas, nearest
    tile = rng.integers(0, 255, size=(400, 400)).astype(np.uint8)
    out, okm, (oy, ox) = A.resample_tile(seg, tile, (1000, 2000), level=2, order=0)
    # back: treat out as a tifxyz tile and sample it at fwd(canvas lattice)
    s = 4
    gy, gx = np.meshgrid(s * (1000 + np.arange(400)) + 1.5, s * (2000 + np.arange(400)) + 1.5, indexing="ij")
    ty, tx = m.fwd(gy, gx)
    u = np.rint((ty - 1.5) / s - oy).astype(int)
    v = np.rint((tx - 1.5) / s - ox).astype(int)
    inside = (u >= 0) & (u < 400) & (v >= 0) & (v < 400)
    inside[inside] &= okm[u[inside], v[inside]]
    back = np.zeros_like(tile)
    back[inside] = out[u[inside], v[inside]]
    agree = float((back[inside] == tile[inside]).mean())
    check(agree > 0.99, f"{seg} level-2 round trip agrees on {agree:.4%} of {int(inside.sum())} pixels")

tile = rng.integers(0, 255, size=(256, 256)).astype(np.uint8)
out, okm, org = A.resample_tile("0500p2/-1", tile, (100, 200), level=1, order=0)
check(np.array_equal(out, tile) and okm.all() and org == (100, 200), "aligned 0500p2/-1: tile unchanged, origin unchanged")
print("ALL PASS" if ok_all else "SOME FAILED")
