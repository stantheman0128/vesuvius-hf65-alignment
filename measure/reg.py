"""Core test: sample the CT volume at the tifxyz surface on the HF65 image's pixel lattice and
find the shift (dy, dx) that maximises NCC with HF65 slice 32.

Lattice (level-2 spacing, 4 level-0 px): image pixel (i, j) has its centre at level-0 canvas
coordinate (4i + 1.5 - py, 4j + 1.5 - px), where (py, px) is the measured pyramid shift of the
image level used (pyrshift.py). S(i, j) is the volume (level 2, trilinear) at tifxyz full-res
cell = that same canvas coordinate (villa convention: cell c -> stored c*scale, bilinear).
NCC is computed between I(i, j) and S(i + dy, j + dx); d = (dy, dx) != 0 means the content of
canvas px p is found at tifxyz cell p + 4d.  argv: key [region indices]"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import s3, ledger_total, MIRROR, DATA
from lib import highpass, ncc_map, peaks

HERE = os.path.dirname(os.path.abspath(__file__))
import sources
R = int(os.environ.get("R", 32))       # search radius, level-2 px
SIG = float(os.environ.get("SIG", 4))  # high-pass sigma, level-2 px
VLEVEL = 2
F = 4


def run(key, idxs):
    src = sources.get(key)
    surf, img, klev, vol, mid = src["surf"], src["img"], src["klev"], src["vol"], src["mid"]
    regions = json.load(open(os.path.join(HERE, f"regions_{key}.json")))
    py, px = src["shift"]
    res = []
    for ri in idxs:
        reg = regions[ri]
        t0 = time.time()
        b0 = ledger_total()
        y0, x0, n = reg["y0"], reg["x0"], reg["size"]
        h = n // F
        gi0, gj0 = y0 // F, x0 // F          # level-2 lattice index of the region corner
        # ---- image at level-2 spacing, all slices
        if klev == 2:
            A = img.region(gi0, gi0 + h, gj0, gj0 + h).astype(np.float32)
        elif klev == 1:
            A1 = img.region(2 * gi0, 2 * gi0 + 2 * h, 2 * gj0, 2 * gj0 + 2 * h).astype(np.float32)
            A = A1.reshape(A1.shape[0], h, 2, h, 2).mean(axis=(2, 4))
        else:
            raise ValueError(klev)
        I = A[mid]
        Im = I > 0
        # ---- surface samples on the extended lattice
        # optional search-window centre (level-2 px): the size-ratio prediction
        # (tifxyz full-res extent / image extent - 1) * region centre; only places the window
        oy = ox = 0
        if os.environ.get("PRED") == "1":
            H0, W0 = src["img_l0"]
            oy = int(round((surf.full[0] / H0 - 1) * (y0 + n / 2) / F))
            ox = int(round((surf.full[1] / W0 - 1) * (x0 + n / 2) / F))
        ii = np.arange(gi0 + oy - R, gi0 + oy + h + R)
        jj = np.arange(gj0 + ox - R, gj0 + ox + h + R)
        cy = F * ii + 1.5 - py
        cx = F * jj + 1.5 - px
        P = surf.cells(cy, cx)
        vals, info = vol.sample(P.reshape(-1, 3))
        S = vals.reshape(len(ii), len(jj))
        Sm = np.isfinite(S) & (S > 0)
        S = np.where(Sm, S, 0).astype(np.float32)
        # ---- NCC, high-pass (primary) and raw
        Ih, Sh = highpass(I, Im, SIG), highpass(S, Sm, SIG)
        Mh = ncc_map(Ih, Im, Sh, Sm, R)
        Mr = ncc_map(I, Im, S, Sm, R)
        ph, pr = peaks(Mh), peaks(Mr)
        # identity (0, 0) values
        id_h, id_r = float(Mh[R, R]), float(Mr[R, R])
        # ---- depth profile: which HF65 slice matches the tifxyz surface best, at the hp peak
        dy, dx = ph["dy"], ph["dx"]
        Sw = Sh[R + dy:R + dy + h, R + dx:R + dx + h]
        Smw = Sm[R + dy:R + dy + h, R + dx:R + dx + h]
        prof = []
        for k in range(A.shape[0]):
            Ik = A[k]
            m = Smw & (Ik > 0)
            if m.sum() < 0.5 * Im.sum():
                prof.append(np.nan)
                continue
            Ikh = highpass(Ik, Ik > 0, SIG)
            a, b = Ikh[m].astype(float), Sw[m].astype(float)
            a -= a.mean()
            b -= b.mean()
            prof.append(float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())))
        prof = np.array(prof)
        kbest = int(np.nanargmax(prof)) if np.isfinite(prof).any() else -1
        row = {
            "key": key, "region": ri, "y0": y0, "x0": x0, "size": n,
            "img_nonzero": float(Im.mean()), "surf_valid": float(Sm.mean()),
            "hp": ph, "raw": pr, "ncc_identity_hp": id_h, "ncc_identity_raw": id_r,
            "window_centre_l2": [oy, ox],
            "level0_dy": F * (oy + ph["dy_sub"]), "level0_dx": F * (ox + ph["dx_sub"]),
            "best_slice": kbest, "best_slice_ncc": float(prof[kbest]) if kbest >= 0 else None,
            "mid_slice": mid, "mid_slice_ncc": float(prof[mid]),
            "vol_bytes_req": info["bytes_req"], "vol_requests": info["requests"],
            "bytes_downloaded": ledger_total() - b0, "sec": round(time.time() - t0, 1),
            "centre_zyx": [float(v) for v in np.nanmean(P.reshape(-1, 3), axis=0)],
        }
        res.append(row)
        np.savez_compressed(os.path.join(DATA, f"reg_{key}{os.environ.get('RESSUF', '')}_{ri}.npz"), I=I, S=S, Im=Im, Sm=Sm, Mh=Mh, Mr=Mr,
                            prof=prof, A=A[::8])
        print(f"{key} r{ri} y0={y0} x0={x0} window centre {oy},{ox}: hp peak (dy,dx)=({ph['dy']},{ph['dx']}) sub=({ph['dy_sub']:.2f},{ph['dx_sub']:.2f})"
              f" ncc={ph['ncc']:.3f} 2nd={ph['second']} at ({ph['second_dy']},{ph['second_dx']}) edge={ph['on_edge']};"
              f" raw peak ({pr['dy']},{pr['dx']}) ncc={pr['ncc']:.3f} 2nd={pr['second']};"
              f" identity hp {id_h:.3f}; level-0 (dy,dx)=({row['level0_dy']:.1f},{row['level0_dx']:.1f});"
              f" best slice {kbest} ({row['best_slice_ncc']}) vs mid {row['mid_slice_ncc']:.3f};"
              f" img nz {row['img_nonzero']:.2f} surf valid {row['surf_valid']:.2f};"
              f" dl {row['bytes_downloaded'] / 1e6:.1f} MB, {row['sec']} s", flush=True)
    out = os.path.join(os.environ.get("HF65_RESULTS", os.path.join(HERE, "..", "results")), f"results_{key}{os.environ.get('RESSUF', '')}.jsonl")
    with open(out, "a") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print("ledger total bytes", ledger_total())


if __name__ == "__main__":
    key = sys.argv[1]
    idxs = [int(v) for v in sys.argv[2:]] or list(range(len(json.load(open(os.path.join(HERE, f"regions_{key}.json"))))))
    run(key, idxs)
