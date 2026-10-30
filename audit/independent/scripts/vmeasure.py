"""Independent canvas vs CT-at-tifxyz offset measurement.
usage: vmeasure.py <tag> <key> <y0> <x0> <n> <L> [--vol S3VOLPATH] [--mesh S3MESHDIR] [--strided-ct]
  key from code/targets.json (read-only); region [y0, y0+n) x [x0, x0+n) in level-0 canvas px;
  L = canvas level to read (lattice spacing 2**L level-0 px).
Writes independent/out/<tag>.npz and appends independent/out/results.jsonl ($HF65_RESULTS instead when set)."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))  # audit/independent/scripts
AUDIT = os.path.dirname(os.path.dirname(HERE))  # audit/
sys.path.insert(0, HERE)
import numpy as np
from scipy import ndimage as ndi
from vcore import Tifxyz, CT, prep, phasecorr, refine, hf, s3, mine, BS
sys.path.insert(0, os.path.join(AUDIT, "code"))
from lib import Image

OUT = os.environ.get("HF65_RESULTS", os.path.join(AUDIT, "independent", "out"))
os.makedirs(OUT, exist_ok=True)
T = json.load(open(os.path.join(AUDIT, "code", "targets.json")))


def canvas_model(cz, L, at):
    """Own pyramid check at a level-0 block: which level-L slice equals level-0 slice 32, and the
    level-0 position of level-L pixel centres (cL), by phase correlation against a block mean."""
    f = 2 ** L
    I0 = Image(cz, 0)
    y0, x0 = at
    n = 256
    A0 = I0.region(y0, y0 + n, x0, x0 + n).astype(float)
    IL = Image(cz, L)
    AL = IL.region(y0 // f, (y0 + n) // f, x0 // f, (x0 + n) // f).astype(float)
    D = A0[32].reshape(n // f, f, n // f, f).mean(axis=(1, 3))
    v = np.ones(D.shape, bool)
    best = None
    for k in range(AL.shape[0]):
        a = AL[k]
        if a.std() == 0:
            continue
        r = float(np.corrcoef(a[2:-2, 2:-2].ravel(), D[2:-2, 2:-2].ravel())[0, 1])
        if best is None or r > best[1]:
            best = (k, r)
    k = best[0]
    pa, pd = prep(AL[k], v), prep(D, v)
    _, dy, dx, pk, sec = phasecorr(pa, pd)
    ty, tx, _ = refine(pa, pd, round(dy), round(dx), up=40, half=1.0)
    cL = (f * ty + (f - 1) / 2.0, f * tx + (f - 1) / 2.0)
    return {"slice": k, "r_slice": best[1], "t": [ty, tx], "cL": list(cL), "pc_peak": pk}


def main():
    a = sys.argv[1:]
    tag, key, y0, x0, n, L = a[0], a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5])
    opt = dict(zip(a[6::2], a[7::2])) if len(a) > 6 else {}
    strided_ct = "--strided-ct" in a
    t = T[key]
    seg = t["seg"]
    name = seg.rsplit("/", 1)[1]
    cz = f"{seg}/{name}.zarr"
    vol = opt.get("--vol", t["vol"])
    mesh = opt.get("--mesh")
    m0 = mine()
    t0 = time.time()
    pyr = json.load(open(os.path.join(AUDIT, "results", "phase2", f"pyr_{key}.json")))
    cm = canvas_model(cz, L, pyr["at"])
    f = 2 ** L
    h = n // f
    img = Image(cz, L)
    i0, j0 = y0 // f, x0 // f
    A = img.region(i0, i0 + h, j0, j0 + h)[cm["slice"]].astype(np.float64)
    rows = f * np.arange(i0, i0 + h) + cm["cL"][0]
    cols = f * np.arange(j0, j0 + h) + cm["cL"][1]
    S = Tifxyz((lambda nm: s3(f"{mesh}/{nm}")) if mesh else (lambda nm: hf(f"{seg}/{nm}")))
    P = S.zyx_at_full(rows, cols)
    ct = CT(s3(vol), 2, centre_mean=not strided_ct)
    C = ct.sample(P)
    va = A > 0
    vc = np.isfinite(C) & (C > 0)
    pa, pc = prep(A, va), prep(np.where(vc, C, 0), vc)
    Pm, dy, dx, pk, sec = phasecorr(pa, pc)
    ry, rx, rpk = refine(pa, pc, round(dy), round(dx))
    # second, independent sub-pixel read-out: raw-intensity NCC after shifting C by the estimate
    def ncc_at(sy, sx):
        Cs = ndi.shift(np.where(vc, C, np.nan), (-sy, -sx), order=1, mode="constant", cval=np.nan)
        m = va & np.isfinite(Cs)
        m = ndi.binary_erosion(m, iterations=2)
        x, y = A[m] - A[m].mean(), Cs[m] - Cs[m].mean()
        return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
    res = {"tag": tag, "key": key, "seg": seg, "vol": vol, "mesh": mesh, "region": [y0, x0, n], "L": L,
           "canvas_model": cm, "strided_ct": strided_ct,
           "valid_canvas": float(va.mean()), "valid_ct": float(vc.mean()),
           "pc_int": [dy, dx], "pc_peak": pk, "pc_second": sec, "pc_ratio": pk / sec if sec > 0 else None,
           "level0_dy": f * ry, "level0_dx": f * rx, "refined_peak": rpk,
           "ncc_raw_at_shift": ncc_at(ry, rx), "ncc_raw_at_zero": ncc_at(0.0, 0.0),
           "my_bytes_this_run": mine() - m0, "sec": round(time.time() - t0, 1)}
    # quadrant consistency (local drift inside the region)
    q = []
    hh = h // 2
    for qy in (0, 1):
        for qx in (0, 1):
            sl = (slice(qy * hh, qy * hh + hh), slice(qx * hh, qx * hh + hh))
            qa, qc = prep(A[sl], va[sl]), prep(np.where(vc, C, 0)[sl], vc[sl])
            _, qdy, qdx, qpk, qsec = phasecorr(qa, qc)
            qry, qrx, _ = refine(qa, qc, round(qdy), round(qdx), up=10)
            q.append([f * qry, f * qrx, round(qpk, 3), round(qsec, 3)])
    res["quadrants"] = q
    np.savez_compressed(os.path.join(OUT, f"{tag}.npz"), A=A.astype(np.float32), C=C.astype(np.float32), va=va, vc=vc,
                        Pm=Pm.astype(np.float32), f=f, i0=i0, j0=j0, d=np.array([ry, rx]))
    with open(os.path.join(OUT, "results.jsonl"), "a") as fo:
        fo.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("seg",)}))
    print("mine total bytes", mine())


if __name__ == "__main__":
    main()
