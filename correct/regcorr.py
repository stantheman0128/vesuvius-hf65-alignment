"""Before/after registration of HF65 slice 32 against the CT sampled on the tifxyz.

This is measure/reg.py's run() with one change, marked AFTER below: in mode "after" the CT is
sampled at tifxyz cell fwd(canvas lattice point) given by apply_fit.py's README model, instead of
at tifxyz cell = canvas lattice point, and the search window is centred on zero. Same lattice,
image levels, pyramid shift (sources.pyr_shift: w00 -> (3, 0)), volume level 2, trilinear,
high-pass sigma 4, NCC search radius R = 32 level-2 px, parabolic sub-pixel peak (lib.peaks).

Mode "before" is reg.py unchanged (PRED=1 places the window as in the published w01/w05 runs), plus
a second, download-free check of the correction ("method B"): the HF65 slice is resampled into the
tifxyz frame with apply_fit.resample (inverse map, bilinear) onto the lattice of the CT samples S
that this same run already holds, then the same high-pass + NCC + peak code is run. A correct
model gives a peak at (0, 0).

argv: key mode region...
  key    w00 | w01 | w05 | c018
  mode   before | after
  region an index into measure/regions_<key>.json (in-sample), "new<i>" for entry i of
         correct/regions_new_<key>.json, or "Y0,X0" for a 1024 px region with that level-0 corner.
env: PRED (before only; default per key as in the published runs), HF65_DATA, HF65_LEDGER_TAG."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "measure"))
sys.path.insert(0, HERE)
import numpy as np
import common
import rangecache
from lib import highpass, ncc_map, peaks
import sources
import apply_fit

R = int(os.environ.get("R", 32))
SIG = float(os.environ.get("SIG", 4))
F = 4
PRED_DEFAULT = {"w00": "0", "w01": "1", "w05": "1", "c018": "0"}   # as in published results_<key>.jsonl
OUT = os.environ.get("HF65CORR_OUT", os.path.join(HERE, ".."))
MEASURE = os.path.join(HERE, "..", "measure")


def region_list(key, specs):
    old = json.load(open(os.path.join(MEASURE, f"regions_{key}.json")))
    newp = os.path.join(HERE, f"regions_new_{key}.json")
    new = json.load(open(newp)) if os.path.exists(newp) else []
    out = []
    for s in specs:
        if s.startswith("new"):
            r = new[int(s[3:])]
            out.append((s, r["y0"], r["x0"], r["size"], "out-of-sample (new region)"))
        elif "," in s:
            y0, x0 = (int(v) for v in s.split(","))
            out.append((f"at{y0}_{x0}", y0, x0, 1024, "out-of-sample (explicit corner)"))
        else:
            r = old[int(s)]
            out.append((f"r{int(s)}", r["y0"], r["x0"], r["size"], "in-sample (used by the fit)"))
    return out


def method_b(I, Im, S, Sm, mdl, gi0, gj0, oy, ox, h, py, px):
    """Resample the canvas slice I (level-2 tile; pixel (m, n) centre at canvas 4*(gi0+m)+1.5-py,
    4*(gj0+n)+1.5-px) into the tifxyz frame on S's inner lattice (tifxyz cell 4*(gi0+oy+i)+1.5-py,
    ...), then the same high-pass/NCC/peak as reg.py. dy, dx = 0 means the model is right."""
    tile_origin = (F * gi0 + 1.5 - py, F * gj0 + 1.5 - px)
    out_origin = (F * (gi0 + oy) + 1.5 - py, F * (gj0 + ox) + 1.5 - px)
    It, ok = apply_fit.resample(I, mdl, tile_origin, F, out_origin, F, (h, h), order=1, fill=0.0, tile_valid=Im)
    It = It.astype(np.float32)
    Itm = ok & (It > 0)
    Ith = highpass(It, Itm, SIG)
    Sh = highpass(S, Sm, SIG)
    M = ncc_map(Ith, Itm, Sh, Sm, R)
    p = peaks(M)
    return {"model": mdl.name, "dy_sub": p["dy_sub"], "dx_sub": p["dx_sub"], "level0_dy": F * p["dy_sub"],
            "level0_dx": F * p["dx_sub"], "ncc": p["ncc"], "ncc_zero": float(M[R, R]), "second": p["second"],
            "on_edge": p["on_edge"], "valid_frac": float(Itm.mean())}, It, Itm


def run(key, mode, specs):
    src = sources.get(key)
    surf, img, klev, vol, mid = src["surf"], src["img"], src["klev"], src["vol"], src["mid"]
    py, px = src["shift"]
    readme = apply_fit.model(key, "readme")
    affine3 = apply_fit.model(key, "affine3", os.path.join(HERE, "..", "results"))
    pred = os.environ.get("PRED", PRED_DEFAULT[key]) == "1" and mode == "before"
    print(f"{key} mode={mode} pyr_shift=({py},{px}) klev={klev} PRED={int(pred)} R={R} SIG={SIG}"
          f" model={readme.name} dy={readme.cy.tolist()} dx={readme.cx.tolist()}", flush=True)
    res = []
    for label, y0, x0, n, sample in region_list(key, specs):
        t0 = time.time()
        b0 = common.ledger_total()
        h = n // F
        gi0, gj0 = y0 // F, x0 // F
        assert gi0 * F == y0 and gj0 * F == x0
        if klev == 2:
            A = img.region(gi0, gi0 + h, gj0, gj0 + h).astype(np.float32)
        elif klev == 1:
            A1 = img.region(2 * gi0, 2 * gi0 + 2 * h, 2 * gj0, 2 * gj0 + 2 * h).astype(np.float32)
            A = A1.reshape(A1.shape[0], h, 2, h, 2).mean(axis=(2, 4))
        else:
            raise ValueError(klev)
        I = A[mid]
        Im = I > 0
        oy = ox = 0
        if pred:
            H0, W0 = src["img_l0"]
            oy = int(round((surf.full[0] / H0 - 1) * (y0 + n / 2) / F))
            ox = int(round((surf.full[1] / W0 - 1) * (x0 + n / 2) / F))
        ii = np.arange(gi0 + oy - R, gi0 + oy + h + R)
        jj = np.arange(gj0 + ox - R, gj0 + ox + h + R)
        cy = F * ii + 1.5 - py
        cx = F * jj + 1.5 - px
        if mode == "after":
            # AFTER: sample the tifxyz at the corrected cell. The README model is separable
            # (dy depends on y only, dx on x only), so rows and columns map independently.
            assert readme.separable
            ty, _ = readme.fwd(cy, np.zeros_like(cy))
            _, tx = readme.fwd(np.zeros_like(cx), cx)
            P = surf.cells(ty, tx)
        else:
            P = surf.cells(cy, cx)
        vals, info = vol.sample(P.reshape(-1, 3))
        S = vals.reshape(len(ii), len(jj))
        Sm = np.isfinite(S) & (S > 0)
        S = np.where(Sm, S, 0).astype(np.float32)
        Ih, Sh = highpass(I, Im, SIG), highpass(S, Sm, SIG)
        Mh = ncc_map(Ih, Im, Sh, Sm, R)
        Mr = ncc_map(I, Im, S, Sm, R)
        ph, pr = peaks(Mh), peaks(Mr)
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
        # NCC at zero canvas->tifxyz shift, when the window contains it
        zy, zx = R - oy, R - ox
        ncc_zero = float(Mh[zy, zx]) if 0 <= zy < Mh.shape[0] and 0 <= zx < Mh.shape[1] else None
        row = {
            "key": key, "mode": mode, "region": label, "sample": sample, "y0": y0, "x0": x0, "size": n,
            "centre": [y0 + n / 2, x0 + n / 2], "pyr_shift": [py, px],
            "model": readme.name if mode == "after" else None,
            "img_nonzero": float(Im.mean()), "surf_valid": float(Sm.mean()),
            "hp": ph, "raw": pr, "ncc_window_centre_hp": float(Mh[R, R]), "ncc_zero_shift_hp": ncc_zero,
            "window_centre_l2": [oy, ox],
            "level0_dy": F * (oy + ph["dy_sub"]), "level0_dx": F * (ox + ph["dx_sub"]),
            "best_slice": kbest, "best_slice_ncc": float(prof[kbest]) if kbest >= 0 else None,
            "mid_slice": mid, "mid_slice_ncc": float(prof[mid]),
            "vol_bytes_req": info["bytes_req"], "vol_requests": info["requests"],
            "centre_zyx": [float(v) for v in np.nanmean(P.reshape(-1, 3), axis=0)],
        }
        extra = {}
        if mode == "before":
            pr_ = readme.offset(y0 + n / 2, x0 + n / 2)
            row["readme_pred_at_centre"] = [float(pr_[0]), float(pr_[1])]
            row["methodB"] = {}
            for mdl in (readme, affine3):
                mb, It, Itm = method_b(I, Im, S, Sm, mdl, gi0, gj0, oy, ox, h, py, px)
                row["methodB"][mdl.name.split("/")[1]] = mb
                if mdl is readme:
                    extra = {"It": It.astype(np.float32), "Itm": Itm}
        row["bytes_downloaded"] = common.ledger_total() - b0
        row["sec"] = round(time.time() - t0, 1)
        res.append(row)
        np.savez_compressed(os.path.join(OUT, "npz", f"{mode}_{key}_{label}.npz"), I=I, S=S, Im=Im, Sm=Sm, Mh=Mh,
                            prof=prof, oy=oy, ox=ox, **extra)
        msg = (f"{key} {mode} {label} ({sample}) y0={y0} x0={x0} win {oy},{ox}: level-0 (dy,dx)=({row['level0_dy']:+.2f},{row['level0_dx']:+.2f})"
               f" hp ncc={ph['ncc']:.3f} 2nd={ph['second']:.3f} edge={ph['on_edge']} zero-shift ncc={ncc_zero}"
               f" best slice {kbest}; surf valid {row['surf_valid']:.3f} img nz {row['img_nonzero']:.3f}")
        if mode == "before":
            msg += (f"; README prediction ({row['readme_pred_at_centre'][0]:+.2f},{row['readme_pred_at_centre'][1]:+.2f});"
                    + "".join(f" methodB[{k}] ({v['level0_dy']:+.2f},{v['level0_dx']:+.2f}) ncc {v['ncc']:.3f} zero {v['ncc_zero']:.3f} valid {v['valid_frac']:.2f};"
                              for k, v in row["methodB"].items()))
        msg += f" dl {row['bytes_downloaded'] / 1e6:.2f} MB, {row['sec']} s"
        print(msg, flush=True)
        with open(os.path.join(OUT, "results", f"{mode}_{key}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    print("rangecache stats", rangecache.STATS, "process ledger bytes", common.ledger_total(), flush=True)


if __name__ == "__main__":
    if os.environ.get("HF65_COVER", "1") == "1":
        n = rangecache.install()
    else:
        n = 0
        if os.environ.get("HF65_LEDGER_TAG"):
            common.LEDGER = os.path.join(common.DATA, f"ledger_{os.environ['HF65_LEDGER_TAG']}.tsv")
    print(f"rangecache: {n} cached ranges indexed; ledger {common.LEDGER}", flush=True)
    run(sys.argv[1], sys.argv[2], sys.argv[3:])
