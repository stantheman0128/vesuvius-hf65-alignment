"""Phase 2: canvas vs tifxyz registration for any ink-bucket segment.

Generalises the published measure/reg.py + measure/regions.py + measure/pyrshift.py
(stantheman0128/vesuvius-hf65-alignment) to segments listed in code/targets.json:
  key -> {"seg": "ink/<scroll>/<segment>", "sup": "<supervision mask zarr>", "vol": "<S3 CT volume path>"}
The canvas is always "<segment>.zarr" next to the tifxyz. Everything is read at level 2 (4 level-0 px).

What differs from reg.py, and why:
  * level-2 canvases here have 17 slices on most segments (z downsampled too), 65 on Paris4; the slice
    compared is the one holding level-0 slice 32 (index 32 * Z2 // 65 -> 8 or 32).
  * the pyramid model is measured, not assumed: level-2 pixel i = mean of level-0 [4i - s, 4i - s + 4)
    ("mean", centre 4i + 1.5 - s) or = level-0 pixel 4i + o ("strided", centre 4i + o); integer s/o per
    axis fitted on one 256 px block (pyr()).
  * search radius R=32 level-2 px (+-128 level-0 px) as in reg.py; with pred=1 the window is centred on
    the size-ratio prediction (reg.py PRED=1).
Commands:
  reg2.py pyr <key>                 -> results/phase2/pyr_<key>.json
  reg2.py regions <key> [nreg]      -> results/phase2/regions_<key>.json
  reg2.py reg <key> <idx...> [--pred] -> appends results/phase2/reg_<key>.jsonl
Outputs go to $HF65_RESULTS instead when it is set; downloads are cached under $HF65_DATA (common.py).
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import fetch, hf, s3, ledger_total, DATA
from lib import Image, Surf, Volume, highpass, ncc_map, peaks

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
# HF65_RESULTS redirects all phase-2 outputs (and the pyr/regions inputs of "reg"), so a rerun
# does not append to the committed results/phase2/reg_<key>.jsonl
OUT = os.environ.get("HF65_RESULTS", os.path.join(AUDIT, "results", "phase2"))
os.makedirs(OUT, exist_ok=True)
TARGETS = json.load(open(os.path.join(HERE, "targets.json")))
F = 4
R = int(os.environ.get("R", 32))
SIG = float(os.environ.get("SIG", 4))
BLOCK, STEP, MARGIN = 1024, 512, 160


def tgt(key):
    t = TARGETS[key]
    seg = t["seg"]
    name = seg.rsplit("/", 1)[1]
    return seg, f"{seg}/{name}.zarr", f"{seg}/{t['sup']}", t["vol"]


def slice_of(Z):
    return 32 * Z // 65 if Z != 65 else 32


# ---------------- pyramid model ----------------
def _fit_level(A0, AL, f, n, y0, x0):
    """For every level-L slice k, test which level-0 source it is: slice k (depth-truncated), slice k*f
    (strided z) or mean of [k*f, (k+1)*f) (mean z); in-plane mean-with-shift s or strided-with-offset o."""
    out = []
    for k in range(AL.shape[0]):
        a = AL[k][1:-1, 1:-1].ravel().astype(float)
        if a.std() == 0:
            continue
        cands = [("z%d" % k, A0[k])] if k < A0.shape[0] else []
        if k * f < A0.shape[0]:
            cands.append(("z%d" % (k * f), A0[k * f]))
            cands.append(("mean%dz%d" % (f, k * f), A0[k * f:k * f + f].mean(axis=0)))
        best = None
        for zname, Z0 in cands:
            for sy in range(f):
                for sx in range(f):
                    S = Z0[f - sy: f - sy + n, f - sx: f - sx + n]
                    P = S.reshape(n // f, f, n // f, f).mean(axis=(1, 3))
                    b = P[1:-1, 1:-1].ravel()
                    r = (float(np.corrcoef(a, b)[0, 1]), "mean", sy, sx, zname)
                    best = r if best is None or r[0] > best[0] else best
                    S = Z0[f + sy: f + sy + n: f, f + sx: f + sx + n: f]
                    b = S[1:-1, 1:-1].ravel()
                    r = (float(np.corrcoef(a, b)[0, 1]), "strided", sy, sx, zname)
                    best = r if best is None or r[0] > best[0] else best
        out.append((k,) + best)
    return out


def pyr(key):
    """Pick the coarsest level (2, then 1) that has a slice equal to level-0 slice 32 (or the mean block
    starting at 32), and its in-plane model. -> results/phase2/pyr_<key>.json"""
    seg, cz, _, _ = tgt(key)
    I0 = Image(cz, 0)
    H0, W0 = I0.shape[1:]
    n = 256
    best = None
    for (fy, fx) in [(0.5, 0.5), (0.35, 0.35), (0.65, 0.65), (0.35, 0.65), (0.65, 0.35)]:
        y0 = int(H0 * fy) // 128 * 128
        x0 = int(W0 * fx) // 128 * 128
        A0 = I0.region(y0 - F, y0 + n + F, x0 - F, x0 + n + F).astype(float)
        if (A0[32] > 0).mean() > 0.9:
            best = (y0, x0, A0)
            break
    if best is None:
        res = {"key": key, "error": "no block with >90% non-zero canvas found at the 5 probe positions"}
        json.dump(res, open(os.path.join(OUT, f"pyr_{key}.json"), "w"))
        print(res)
        return res
    y0, x0, A0 = best
    res = {"key": key, "at": [y0, x0], "levels": {}}
    chosen = None
    for L in (2, 1):
        f = 2 ** L
        IL = Image(cz, L)
        AL = IL.region(y0 // f, (y0 + n) // f, x0 // f, (x0 + n) // f).astype(float)
        A0c = A0[:, F - f: F - f + n + 2 * f, F - f: F - f + n + 2 * f]
        fits = _fit_level(A0c, AL, f, n, y0, x0)
        res["levels"][L] = {"Z": IL.shape[0], "fits": fits}
        for (k, r, model, sy, sx, zname) in fits:
            if zname in ("z32", "mean%dz32" % f) and r > 0.99:
                c = (f - 1) / 2.0 - sy if model == "mean" else float(sy)
                cx = (f - 1) / 2.0 - sx if model == "mean" else float(sx)
                chosen = {"level": L, "slice": k, "model": model, "s": [sy, sx], "z_source": zname, "r": r,
                          "centre_offset_L": [c, cx]}
                break
        if chosen:
            break
    res["chosen"] = chosen
    if chosen is None:
        res["error"] = "no level-2 or level-1 slice equals level-0 slice 32 (r > 0.99)"
    else:
        g = 4 // (2 ** chosen["level"])
        f = 2 ** chosen["level"]
        # level-0 canvas coordinate of the level-2 lattice point I: 4 I + f (g - 1) / 2 + c_L
        res["centre_offset"] = [f * (g - 1) / 2.0 + chosen["centre_offset_L"][0], f * (g - 1) / 2.0 + chosen["centre_offset_L"][1]]
    json.dump(res, open(os.path.join(OUT, f"pyr_{key}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "levels"}))
    for L, v in res["levels"].items():
        print("  level", L, "Z", v["Z"], "fits (k, r, model, sy, sx, z):", [(f[0], round(f[1], 4), f[2], f[3], f[4], f[5]) for f in v["fits"] if f[1] > 0.95][:6])
    return res


# ---------------- regions ----------------
def regions(key, nreg=4):
    seg, cz, sup, _ = tgt(key)
    S = Surf(seg)
    rs = 16 if S.H > 400 else 4
    rows = np.arange(0, S.H, rs)
    X = np.stack([S.t["x"].rows(r, r + 1, 0, S.W)[0] for r in rows])
    V = (X != -1) & np.isfinite(X)
    M = Image(sup, 5)
    sm = M.region(0, M.shape[1], 0, M.shape[2]).max(axis=0) > 0
    im = Image(cz, 0)
    H0, W0 = im.shape[1], im.shape[2]
    print(f"{key}: stored {S.H}x{S.W} full {S.full}; canvas {H0}x{W0}; tifxyz valid {V.mean():.3f}; supmask L5 {sm.shape} set {sm.mean():.3f}")
    cand = []
    for y0 in range(0, H0 - BLOCK + 1, STEP):
        for x0 in range(0, W0 - BLOCK + 1, STEP):
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
    out = []
    if cand:
        P = np.array([(c[0] + BLOCK / 2, c[1] + BLOCK / 2) for c in cand], float)
        ctr = P.mean(0)
        chosen = [int(np.argmin(((P - ctr) ** 2).sum(1)))]
        while len(chosen) < min(nreg, len(cand)):
            dmin = np.min(((P[:, None, :] - P[chosen][None, :, :]) ** 2).sum(-1), axis=1)
            chosen.append(int(np.argmax(dmin)))
        out = [{"y0": cand[i][0], "x0": cand[i][1], "size": BLOCK, "tifxyz_valid": cand[i][2], "supmask": cand[i][3]} for i in chosen]
    for r in out:
        print("   ", r)
    json.dump({"key": key, "n_candidates": len(cand), "canvas": [H0, W0], "full": list(S.full), "regions": out},
              open(os.path.join(OUT, f"regions_{key}.json"), "w"), indent=1)
    return out


# ---------------- registration ----------------
def reg(key, idxs, pred=False):
    seg, cz, _, vol = tgt(key)
    pm = json.load(open(os.path.join(OUT, f"pyr_{key}.json")))
    if "error" in pm:
        raise SystemExit(f"no pyramid model for {key}: {pm['error']}")
    oy0, ox0 = pm["centre_offset"]
    surf = Surf(seg)
    L = pm["chosen"]["level"]
    mid = pm["chosen"]["slice"]
    g = 4 // (2 ** L)
    img = Image(cz, L)
    V = Volume(s3(vol), 2)
    H0, W0 = Image(cz, 0).shape[1:]
    regs = json.load(open(os.path.join(OUT, f"regions_{key}.json")))["regions"]
    for ri in idxs:
        reg_ = regs[ri]
        t0, b0 = time.time(), ledger_total()
        y0, x0, n = reg_["y0"], reg_["x0"], reg_["size"]
        h = n // F
        gi0, gj0 = y0 // F, x0 // F
        AL = img.region(g * gi0, g * gi0 + g * h, g * gj0, g * gj0 + g * h).astype(np.float32)
        A = AL.reshape(AL.shape[0], h, g, h, g).mean(axis=(2, 4)) if g > 1 else AL
        I = A[mid]
        Im = I > 0
        oy = ox = 0
        if pred:
            oy = int(round((surf.full[0] / H0 - 1) * (y0 + n / 2) / F))
            ox = int(round((surf.full[1] / W0 - 1) * (x0 + n / 2) / F))
        ii = np.arange(gi0 + oy - R, gi0 + oy + h + R)
        jj = np.arange(gj0 + ox - R, gj0 + ox + h + R)
        P = surf.cells(F * ii + oy0, F * jj + ox0)
        vals, info = V.sample(P.reshape(-1, 3))
        S = vals.reshape(len(ii), len(jj))
        Sm = np.isfinite(S) & (S > 0)
        S = np.where(Sm, S, 0).astype(np.float32)
        Ih, Sh = highpass(I, Im, SIG), highpass(S, Sm, SIG)
        Mh = ncc_map(Ih, Im, Sh, Sm, R)
        Mr = ncc_map(I, Im, S, Sm, R)
        ph, pr = peaks(Mh), peaks(Mr)
        id_h = float(Mh[R - oy, R - ox]) if abs(oy) <= R and abs(ox) <= R else None
        dy, dx = ph["dy"], ph["dx"]
        Sw = Sh[R + dy:R + dy + h, R + dx:R + dx + h]
        Smw = Sm[R + dy:R + dy + h, R + dx:R + dx + h]
        prof = []
        for k in range(A.shape[0]):
            Ik = A[k]
            m = Smw & (Ik > 0)
            if m.sum() < 0.5 * max(Im.sum(), 1):
                prof.append(float("nan"))
                continue
            Ikh = highpass(Ik, Ik > 0, SIG)
            a, b = Ikh[m].astype(float), Sw[m].astype(float)
            a -= a.mean()
            b -= b.mean()
            den = math.sqrt((a * a).sum() * (b * b).sum())
            prof.append(float((a * b).sum() / den) if den > 0 else float("nan"))
        prof = np.array(prof)
        kbest = int(np.nanargmax(prof)) if np.isfinite(prof).any() else -1
        row = {"key": key, "seg": seg, "region": ri, "y0": y0, "x0": x0, "size": n, "pred_window": pred,
               "img_nonzero": float(Im.mean()), "surf_valid": float(Sm.mean()),
               "hp": ph, "raw": pr, "ncc_at_zero_shift_hp": id_h, "window_centre_l2": [oy, ox],
               "level0_dy": F * (oy + ph["dy_sub"]), "level0_dx": F * (ox + ph["dx_sub"]),
               "size_ratio_pred_at_centre": [(surf.full[0] / H0 - 1) * (y0 + n / 2), (surf.full[1] / W0 - 1) * (x0 + n / 2)],
               "slice_compared": mid, "best_slice": kbest, "best_slice_ncc": float(prof[kbest]) if kbest >= 0 else None,
               "compared_slice_ncc": float(prof[mid]) if np.isfinite(prof[mid]) else None,
               "canvas_level": L, "pyr_model": pm["chosen"]["model"], "pyr_centre_offset": [oy0, ox0],
               "vol_bytes_req": info["bytes_req"], "bytes_downloaded": ledger_total() - b0, "sec": round(time.time() - t0, 1)}
        with open(os.path.join(OUT, f"reg_{key}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        np.savez_compressed(os.path.join(DATA, f"reg_{key}_{ri}.npz"), I=I, S=S, Im=Im, Sm=Sm, Mh=Mh, prof=prof)
        print(f"{key} r{ri} ({y0},{x0}) win {oy},{ox}: hp peak ({ph['dy']},{ph['dx']}) ncc={ph['ncc']:.3f} 2nd={ph['second']}"
              f" edge={ph['on_edge']}; level-0 (dy,dx)=({row['level0_dy']:.1f},{row['level0_dx']:.1f});"
              f" NCC at 0 {id_h}; slice best {kbest} ({row['best_slice_ncc']}) vs compared {mid};"
              f" img nz {row['img_nonzero']:.2f} surf valid {row['surf_valid']:.2f}; dl {row['bytes_downloaded']/1e6:.1f} MB {row['sec']} s", flush=True)
    print("ledger total bytes", ledger_total())


if __name__ == "__main__":
    cmd, key = sys.argv[1], sys.argv[2]
    if cmd == "pyr":
        pyr(key)
    elif cmd == "regions":
        regions(key, int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    elif cmd == "reg":
        args = [a for a in sys.argv[3:] if not a.startswith("--")]
        reg(key, [int(a) for a in args], pred="--pred" in sys.argv)
