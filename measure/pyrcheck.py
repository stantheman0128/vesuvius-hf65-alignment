"""Pyramid conventions: how level k relates to level 0, for the CT volumes and the HF65 images.
Prints correlation / mean |diff| of level k against level-0 pooled with each start offset."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import s3, hf, ledger_total, MIRROR
from lib import Volume, Image, Surf

P4 = s3("PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr")
C1667 = s3("PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr")


def cmp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m].astype(float), b[m].astype(float)
    return float(np.corrcoef(a, b)[0, 1]), float(np.abs(a - b).mean())


def vol_check(url, p0_zyx, k=2, n=12):
    """Compare level k block (n^3) with level-0 pooled with origin offset o in 0..f-1, and stride."""
    f = 2 ** k
    vk = Volume(url, k)
    v0 = Volume(url, 0)
    c = (np.asarray(p0_zyx) // f).astype(int)
    z0, y0, x0 = c - n // 2
    Bk, _ = vk.block(z0, z0 + n, y0, y0 + n, x0, x0 + n)
    Z0, Y0, X0 = z0 * f - f, y0 * f - f, x0 * f - f
    B0, info = v0.block(Z0, Z0 + (n + 2) * f, Y0, Y0 + (n + 2) * f, X0, X0 + (n + 2) * f)
    print(f"  level {k} block {Bk.shape} mean {np.nanmean(Bk):.1f}; level-0 block {B0.shape} mean {np.nanmean(B0):.1f}")
    for o in range(f):
        s = B0[f - o: f - o + n * f, f - o: f - o + n * f, f - o: f - o + n * f]
        pooled = s.reshape(n, f, n, f, n, f).mean(axis=(1, 3, 5))
        r, d = cmp(Bk, pooled)
        print(f"    mean-pool, level-0 start shifted by -{o}: r={r:.4f} mean|diff|={d:.2f}")
    s = B0[f: f + n * f: f, f: f + n * f: f, f: f + n * f: f]
    r, d = cmp(Bk, s)
    print(f"    stride (level-0 voxel f*i): r={r:.4f} mean|diff|={d:.2f}")


def img_check(zpath, k, cy0, cx0, slices):
    """HF65 image level k vs level 0 pooled 2^k in XY, for the given slice pairs (k-slice, [0-slices])."""
    f = 2 ** k
    I0 = Image(zpath, 0)
    Ik = Image(zpath, k)
    y0, x0 = cy0 * 128, cx0 * 128
    A0 = I0.region(y0, y0 + 128, x0, x0 + 128).astype(float)
    Ak = Ik.region(y0 // f, y0 // f + 128 // f, x0 // f, x0 // f + 128 // f).astype(float)
    print(f"  {zpath} level {k} shape {Ik.shape} vs level 0 {I0.shape}; region level-0 y{y0} x{x0}; level-0 nonzero {np.mean(A0 > 0):.2f}")
    for ks, zs in slices:
        L = A0[zs].mean(axis=0)
        for o in range(f):
            # level-k pixel i pools level-0 pixels [f*i - o, f*i - o + f)
            Lp = np.pad(L, ((o, f), (o, f)), mode="edge")[: 128, : 128]
            pooled = Lp.reshape(128 // f, f, 128 // f, f).mean(axis=(1, 3))
            r, d = cmp(Ak[ks][1:-1, 1:-1], pooled[1:-1, 1:-1])
            print(f"    level-{k} slice {ks} vs level-0 slices {zs} pooled, start -{o}: r={r:.4f} mean|diff|={d:.2f}")
        st = L[:: f, :: f]
        r, d = cmp(Ak[ks][1:-1, 1:-1], st[1:-1, 1:-1])
        print(f"    level-{k} slice {ks} vs level-0 slices {zs} stride: r={r:.4f} mean|diff|={d:.2f}")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "vol_p4":
        s = Surf("ink/phercparis4/w00_20231016151002", os.path.join(MIRROR, "r4/phercparis4/w00_20231016151002"))
        p = s.cells([16000.0], [25000.0])[0, 0]
        print("w00 full-res cell (16000, 25000) -> zyx", p)
        vol_check(P4, p, k=2)
    elif what == "vol_1667":
        s = Surf("ink/1667/w018_20240304144031_2um")
        p = s.cells([21000.0], [49000.0])[0, 0]
        print("w018 full-res cell (21000, 49000) -> zyx", p)
        vol_check(C1667, p, k=2)
    elif what == "img_p4":
        img_check("ink/phercparis4/w00_20231016151002/w00_20231016151002.zarr", 2, 125, 200,
                  [(32, [32]), (31, [31])])
    elif what == "img_1667":
        img_check("ink/1667/w018_20240304144031_2um/w018_20240304144031_2um.zarr", 1, 166, 384,
                  [(32, [32]), (32, [64, 65]), (16, [16]), (16, [32, 33])])
    print("ledger total bytes", ledger_total())


def img_fit(zpath, k, cy0, cx0, kslice, zs, kref=0):
    """Fit level-k slice = pool_f(blur_sigma(shift_(sy,sx)(level-kref slices zs))) over shift and blur."""
    from scipy.ndimage import shift as ndshift, gaussian_filter
    f = 2 ** (k - kref)
    Ir = Image(zpath, kref)
    Ik = Image(zpath, k)
    n = 128 * 2  # level-kref region size
    y0, x0 = cy0 * 128, cx0 * 128
    A0 = Ir.region(y0, y0 + n, x0, x0 + n).astype(float)[zs].mean(axis=0)
    Ak = Ik.region(y0 // f, (y0 + n) // f, x0 // f, (x0 + n) // f).astype(float)[kslice]
    best = None
    rows = []
    for sig in (0.0, 0.25, 0.5, 0.75, 1.0):
        B = gaussian_filter(A0, sig * f) if sig > 0 else A0
        for sy in np.arange(-2.0, 2.01, 0.25):
            for sx in np.arange(-2.0, 2.01, 0.25):
                S = ndshift(B, (sy, sx), order=3, mode="nearest")
                P = S.reshape(n // f, f, n // f, f).mean(axis=(1, 3))
                r, d = cmp(Ak[2:-2, 2:-2], P[2:-2, 2:-2])
                if best is None or r > best[0]:
                    best = (r, d, sig, sy, sx)
        rows.append(best)
    print(f"  {zpath}: level {k} slice {kslice} vs level {kref} slices {zs} (pool {f})")
    r, d, sig, sy, sx = best
    print(f"    best: r={r:.5f} mean|diff|={d:.2f} blur sigma={sig}*f shift (y, x)=({sy}, {sx}) level-{kref} px"
          f"  -> level-{k} pixel i centre at level-{kref} coordinate f*i + {(f - 1) / 2:.2f} - shift")
    for sig in (0.0, 0.5, 1.0):
        B = gaussian_filter(A0, sig * f) if sig > 0 else A0
        S = ndshift(B, (best[3], best[4]), order=3, mode="nearest")
        P = S.reshape(n // f, f, n // f, f).mean(axis=(1, 3))
        print(f"    at that shift, blur {sig}*f: r={cmp(Ak[2:-2, 2:-2], P[2:-2, 2:-2])[0]:.5f}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "fit":
    pass
