"""Validate shifts.json: re-register regions after applying each segment's correction.

Measurement code is the audit's (audit/code/lib.py: Surf.cells, Volume.sample, highpass, ncc_map, peaks)
and the "before" run is audit/code/reg2.py reg() step for step: the same canvas level and slice and
pyramid lattice (audit/results/phase2/pyr_<key>.json), CT level 2 trilinear, high-pass sigma 4, NCC radius
R = 32 level-2 px, parabolic sub-pixel peak. The Paris4 window is centred on the size-ratio prediction
(reg2 --pred), every other window on zero, as in the audit. Each region then gets two checks:

  after (method A)  the CT is sampled at tifxyz cell fwd(canvas lattice point) from apply_shifts.model(seg)
                    instead of at the canvas lattice point, window centred on zero; the peak is the residual.
  method B          the canvas slice is moved into the tifxyz frame with apply_fit.resample (inverse map,
                    bilinear) through the same model, onto the lattice of the before-run's CT samples; the
                    peak of the same NCC is the residual. This is the path a user takes for label tiles.
  prediction error  before-run offset minus the model offset at the region centre.

For the aligned controls the model is the identity; the run also records whether the after-run CT
samples are bit-identical to the before-run's and whether method B returns the canvas tile unchanged.

Usage (WSL venv, see README):
  HF65_DATA=<fresh folder for new downloads> AUDIT_DATA=<audit folder>/data \
  python -I correct/shifts/validate.py plan            # the held-out region chosen for each row
  python -I correct/shifts/validate.py run SPEC...     # SPEC = key:idx (audit regions_<key>.json),
                                                       #        key:newN (correct/regions_new_<key>.json), key:Y0,X0
  python -I correct/shifts/validate.py table           # validation.csv and the README table
Results append to correct/shifts/validation/results.jsonl. LIMIT_MB (default 560) stops new regions once
$HF65_DATA/ledger.tsv holds that many downloaded MB.
"""
import os, sys, json, time, math, csv
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "audit", "code"))
sys.path.insert(0, os.path.join(REPO, "correct"))
sys.path.insert(0, HERE)
import numpy as np
import common
import cachelayer
from lib import Image, Surf, Volume, highpass, ncc_map, peaks
import apply_fit
import apply_shifts

F, R, SIG = 4, 32, 4.0
PHASE2 = os.path.join(REPO, "audit", "results", "phase2")
OUTD = os.path.join(HERE, "validation")
RESULTS = os.path.join(OUTD, "results.jsonl")
TARGETS = json.load(open(os.path.join(REPO, "audit", "code", "targets.json")))
P4VOL = "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
# shifts.json keys whose data/pyramid model live under another audit key, or outside targets.json
DATA_KEY = {"w00": "p4w00", "c018": "c018"}
EXTRA = {  # published measure/pyrshift_<key>.json: level 2, mean, shift (0, 0) -> lattice centre 1.5
    "w01": {"seg": "ink/phercparis4/w01_20230702185753_r15", "vol": P4VOL,
            "pyr": {"chosen": {"level": 2, "slice": 32, "model": "mean"}, "centre_offset": [1.5, 1.5],
                    "source": "measure/pyrshift_w01.json shift (0, 0)"}},
    "w05": {"seg": "ink/phercparis4/w05_4424", "vol": P4VOL,
            "pyr": {"chosen": {"level": 2, "slice": 32, "model": "mean"}, "centre_offset": [1.5, 1.5],
                    "source": "measure/pyrshift_w05.json shift (0, 0)"}},
}
CONTROLS = ["p500", "n841", "n841a", "n841b", "man5"]
FAR_MIN = 2048   # a held-out pick closer than this to a fit region is replaced by a farregion() pick


def last_by_region(path):
    rows = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            rows[r["region"]] = r
    return rows


def source(key):
    """-> seg (bucket path), canvas zarr, CT volume path, pyramid model dict, group"""
    if key in EXTRA:
        e = EXTRA[key]
        seg, vol, pm = e["seg"], e["vol"], e["pyr"]
    else:
        dk = DATA_KEY.get(key, key)
        seg, vol = TARGETS[dk]["seg"], TARGETS[dk]["vol"]
        pm = json.load(open(os.path.join(PHASE2, f"pyr_{dk}.json")))
    name = seg.rsplit("/", 1)[1]
    kind, row = apply_shifts.lookup(seg)
    group = row["group"] if kind == "offset" else "aligned_control"
    return seg, f"{seg}/{name}.zarr", vol, pm, group


def region(key, spec):
    """-> (label, y0, x0, size, sample, audit_row or None)"""
    if spec.startswith("new"):
        r = json.load(open(os.path.join(REPO, "correct", f"regions_new_{key}.json")))[int(spec[3:])]
        return spec, r["y0"], r["x0"], r["size"], "held-out", None
    if spec.startswith("far"):
        r = json.load(open(os.path.join(OUTD, f"regions_far_{key}.json")))["regions"][int(spec[3:])]
        return spec, r["y0"], r["x0"], r["size"], "held-out", None
    if "," in spec:
        y0, x0 = (int(v) for v in spec.split(","))
        return f"at{y0}_{x0}", y0, x0, 1024, "held-out", None
    dk = DATA_KEY.get(key, key)
    regs = json.load(open(os.path.join(PHASE2, f"regions_{dk}.json")))["regions"]
    i = int(spec)
    r = regs[i]
    done = last_by_region(os.path.join(PHASE2, f"reg_{dk}.jsonl"))
    fit_keys = {(f["y0"], f["x0"]) for f in fit_rows(key)}
    if key in CONTROLS:
        sample = "control"
    else:
        sample = "in-sample" if (r["y0"], r["x0"]) in fit_keys else "held-out"
    return f"r{i}", r["y0"], r["x0"], r["size"], sample, done.get(i)


def fit_rows(key):
    if key in CONTROLS:
        return []
    row = next(r for r in apply_shifts.doc()["segments"] if r["key"] == key)
    out = []
    for s in row["fit_regions"]:
        y0 = int(s.split("y0 ")[1].split(",")[0])
        x0 = int(s.split("x0 ")[1].rstrip(")"))
        out.append({"y0": y0, "x0": x0})
    return out


def plan():
    out = []
    for r in apply_shifts.doc()["segments"]:
        key = r["key"]
        if key in ("w00", "w01", "w05", "c018"):
            regs = json.load(open(os.path.join(REPO, "correct", f"regions_new_{key}.json")))
            cand = [("new1", regs[1])]
        else:
            regs = json.load(open(os.path.join(PHASE2, f"regions_{key}.json")))["regions"]
            fit = {(f["y0"], f["x0"]) for f in fit_rows(key)}
            cand = [(str(i), g) for i, g in enumerate(regs) if (g["y0"], g["x0"]) not in fit]
        if not cand:
            out.append((key, None, None))
            continue
        dist = [min(math.hypot(g["y0"] - f["y0"], g["x0"] - f["x0"]) for f in fit_rows(key)) for _, g in cand]
        i = int(np.argmax(dist))   # the unused pick farthest from every fit region
        spec, d = cand[i][0], dist[i]
        if d < FAR_MIN:
            fp = os.path.join(OUTD, f"regions_far_{key}.json")
            if os.path.exists(fp):
                g = json.load(open(fp))["regions"][0]
                spec, d = "far0", min(math.hypot(g["y0"] - f["y0"], g["x0"] - f["x0"]) for f in fit_rows(key))
            else:
                spec = f"NEEDS farregion (best unused pick {spec} is {d:.0f} px from a fit region)"
        out.append((key, f"{key}:{spec}", d))
    for key, spec, d in out:
        print(f"{key:6s} {spec}  nearest fit region {d:.0f} px" if spec else f"{key:6s} NO UNUSED REGION")
    return out


def farregion(key, supmask_min):
    """reg2.regions() candidate scan (1024 px blocks on a 512 px grid, tifxyz valid >= 0.98 over the block plus
    a 160 px margin) with the supervision-mask threshold lowered to supmask_min, then the candidate farthest
    from every fit region. For segments whose audit picks all sit next to the fit regions.
    -> validation/regions_far_<key>.json"""
    BLOCK, STEP, MARGIN = 1024, 512, 160
    dk = DATA_KEY.get(key, key)
    t = TARGETS[dk]
    seg = t["seg"]
    S = Surf(seg)
    rs = 16 if S.H > 400 else 4
    rows = np.arange(0, S.H, rs)
    X = np.stack([S.t["x"].rows(r, r + 1, 0, S.W)[0] for r in rows])
    V = (X != -1) & np.isfinite(X)
    Mk = Image(f"{seg}/{t['sup']}", 5)
    sm = Mk.region(0, Mk.shape[1], 0, Mk.shape[2]).max(axis=0) > 0
    H0, W0 = Image(f"{seg}/{seg.rsplit('/', 1)[1]}.zarr", 0).shape[1:]
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
            if v >= 0.98 and s5 >= supmask_min:
                cand.append((y0, x0, float(v), float(s5)))
    fit = fit_rows(key)
    best = max(cand, key=lambda c: min(math.hypot(c[0] - f["y0"], c[1] - f["x0"]) for f in fit))
    d = min(math.hypot(best[0] - f["y0"], best[1] - f["x0"]) for f in fit)
    res = {"key": key, "supmask_min": supmask_min, "n_candidates": len(cand), "rule": "reg2.regions scan with lowered supervision-mask threshold, farthest from the fit regions",
           "regions": [{"y0": best[0], "x0": best[1], "size": BLOCK, "tifxyz_valid": best[2], "supmask": best[3],
                        "min_dist_to_fit_px": d}]}
    os.makedirs(OUTD, exist_ok=True)
    json.dump(res, open(os.path.join(OUTD, f"regions_far_{key}.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


def ncc_peak(I, Im, S, Sm):
    Mh = ncc_map(highpass(I, Im, SIG), Im, highpass(S, Sm, SIG), Sm, R)
    return Mh, peaks(Mh)


def run_one(key, spec):
    seg, cz, vol, pm, group = source(key)
    label, y0, x0, n, sample, audit_row = region(key, spec)
    M = apply_shifts.model(seg)
    t0, b0 = time.time(), common.ledger_total()
    oy0, ox0 = pm["centre_offset"]
    L, mid = pm["chosen"]["level"], pm["chosen"]["slice"]
    g = 4 // (2 ** L)
    surf = Surf(seg)
    img = Image(cz, L)
    V = Volume(common.s3(vol), 2)
    H0, W0 = Image(cz, 0).shape[1:]
    h = n // F
    gi0, gj0 = y0 // F, x0 // F
    assert gi0 * F == y0 and gj0 * F == x0
    AL = img.region(g * gi0, g * gi0 + g * h, g * gj0, g * gj0 + g * h).astype(np.float32)
    A = AL.reshape(AL.shape[0], h, g, h, g).mean(axis=(2, 4)) if g > 1 else AL
    I = A[mid]
    Im = I > 0
    yc, xc = y0 + n / 2, x0 + n / 2
    pred = [float(v) for v in M.offset(yc, xc)]
    # ---- before: reg2.reg() unchanged (window centre as in the audit)
    oy = ox = 0
    if group == "paris4_size_ratio":
        oy = int(round((surf.full[0] / H0 - 1) * yc / F))
        ox = int(round((surf.full[1] / W0 - 1) * xc / F))
    ii = np.arange(gi0 + oy - R, gi0 + oy + h + R)
    jj = np.arange(gj0 + ox - R, gj0 + ox + h + R)
    P = surf.cells(F * ii + oy0, F * jj + ox0)
    vals, _ = V.sample(P.reshape(-1, 3))
    S = vals.reshape(len(ii), len(jj))
    Sm = np.isfinite(S) & (S > 0)
    S = np.where(Sm, S, 0).astype(np.float32)
    Mb, pb = ncc_peak(I, Im, S, Sm)
    d_before = [F * (oy + pb["dy_sub"]), F * (ox + pb["dx_sub"])]
    # ---- after, method A: CT at tifxyz cell fwd(canvas lattice point), window centred on zero
    assert M.separable
    ia = np.arange(gi0 - R, gi0 + h + R)
    ja = np.arange(gj0 - R, gj0 + h + R)
    ty, _ = M.fwd(F * ia + oy0, np.zeros(len(ia)))
    _, tx = M.fwd(np.zeros(len(ja)), F * ja + ox0)
    Pa = surf.cells(ty, tx)
    va, _ = V.sample(Pa.reshape(-1, 3))
    Sa = va.reshape(len(ia), len(ja))
    Sam = np.isfinite(Sa) & (Sa > 0)
    Sa = np.where(Sam, Sa, 0).astype(np.float32)
    Ma, pa = ncc_peak(I, Im, Sa, Sam)
    # ---- method B: canvas slice into the tifxyz frame on the before-run's lattice
    tile_origin = (F * gi0 + oy0, F * gj0 + ox0)
    out_origin = (F * (gi0 + oy) + oy0, F * (gj0 + ox) + ox0)
    It, ok = apply_fit.resample(I, M, tile_origin, F, out_origin, F, (h, h), order=1, fill=0.0, tile_valid=Im)
    It = It.astype(np.float32)
    Itm = ok & (It > 0)
    Mbb, pbb = ncc_peak(It, Itm, S, Sm)
    row = {
        "key": key, "segment": seg, "group": group, "region": label, "sample": sample, "y0": y0, "x0": x0, "size": n,
        "centre": [yc, xc], "model": M.name, "model_dy": M.cy.tolist(), "model_dx": M.cx.tolist(),
        "model_offset_at_centre": pred, "canvas_level": L, "slice": mid, "lattice_centre_offset": [oy0, ox0],
        "before": {"level0_dy": d_before[0], "level0_dx": d_before[1], "ncc": pb["ncc"], "second": pb["second"],
                   "on_edge": pb["on_edge"], "window_centre_l2": [oy, ox]},
        "prediction_error": [d_before[0] - pred[0], d_before[1] - pred[1]],
        "after": {"level0_dy": F * pa["dy_sub"], "level0_dx": F * pa["dx_sub"], "ncc": pa["ncc"],
                  "ncc_zero": float(Ma[R, R]), "second": pa["second"], "on_edge": pa["on_edge"],
                  "int_peak": [pa["dy"], pa["dx"]]},
        "methodB": {"level0_dy": F * pbb["dy_sub"], "level0_dx": F * pbb["dx_sub"], "ncc": pbb["ncc"],
                    "ncc_zero": float(Mbb[R, R]), "second": pbb["second"], "on_edge": pbb["on_edge"],
                    "valid_frac": float(Itm.mean())},
        "img_nonzero": float(Im.mean()), "surf_valid": float(Sm.mean()), "surf_valid_after": float(Sam.mean()),
    }
    if audit_row is not None:
        row["audit_level0"] = [audit_row["level0_dy"], audit_row["level0_dx"]]
        row["before_minus_audit"] = [d_before[0] - audit_row["level0_dy"], d_before[1] - audit_row["level0_dx"]]
    if not (M.cy.any() or M.cx.any()):  # identity: the correction must change nothing
        row["noop"] = {"after_ct_samples_identical": bool(np.array_equal(S, Sa) and np.array_equal(Sm, Sam)),
                       "after_ncc_map_identical": bool(np.array_equal(Mb, Ma, equal_nan=True)),
                       "methodB_tile_max_abs_diff": float(np.abs(It[ok] - I[ok]).max()),
                       "methodB_tile_pixels_compared": int(ok.sum()),
                       "methodB_tile_pixels_filled": int((~ok).sum())}
    row["bytes_downloaded"] = common.ledger_total() - b0
    row["sec"] = round(time.time() - t0, 1)
    os.makedirs(OUTD, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    npz = os.path.join(common.DATA, "npz")
    os.makedirs(npz, exist_ok=True)
    np.savez_compressed(os.path.join(npz, f"{key}_{label}.npz"), I=I, S=S, Sm=Sm, Sa=Sa, Sam=Sam, It=It, Itm=Itm,
                        Mb=Mb, Ma=Ma, Mbb=Mbb, oy=oy, ox=ox)
    msg = (f"{key} {label} ({sample}, {group}) y0={y0} x0={x0}: before ({d_before[0]:+.2f},{d_before[1]:+.2f}) ncc {pb['ncc']:.3f};"
           f" model ({pred[0]:+.2f},{pred[1]:+.2f}); pred err ({row['prediction_error'][0]:+.2f},{row['prediction_error'][1]:+.2f});"
           f" after A ({row['after']['level0_dy']:+.2f},{row['after']['level0_dx']:+.2f}) ncc {pa['ncc']:.3f} edge {pa['on_edge']};"
           f" B ({row['methodB']['level0_dy']:+.2f},{row['methodB']['level0_dx']:+.2f}) ncc {pbb['ncc']:.3f} valid {row['methodB']['valid_frac']:.2f}")
    if "before_minus_audit" in row:
        msg += f"; before - audit ({row['before_minus_audit'][0]:+.4f},{row['before_minus_audit'][1]:+.4f})"
    if "noop" in row:
        msg += f"; noop {row['noop']}"
    msg += f"; dl {row['bytes_downloaded'] / 1e6:.2f} MB, {row['sec']} s"
    print(msg, flush=True)
    return row


def done_specs():
    out = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            r = json.loads(line)
            out.add((r["key"], r["region"]))
    return out


def run(specs):
    limit = float(os.environ.get("LIMIT_MB", 560)) * 1e6
    done = done_specs()
    for s in specs:
        key, sp = s.split(":", 1)
        lab = (sp if sp.startswith("new") else (f"r{sp}" if "," not in sp else "at" + sp.replace(",", "_")))
        if (key, lab) in done:
            print(f"skip {s} (in results.jsonl)", flush=True)
            continue
        used = common.ledger_total()
        if used >= limit:
            print(f"STOP before {s}: new downloads {used / 1e6:.1f} MB >= LIMIT_MB", flush=True)
            break
        try:
            run_one(key, sp)
        except Exception as e:  # keep going; the failure is logged and the row stays missing
            print(f"FAILED {s}: {type(e).__name__}: {e}", flush=True)
    print(f"cache stats {cachelayer.STATS}; new downloads total {common.ledger_total()} bytes", flush=True)


def table():
    rows = {}
    for line in open(RESULTS):
        r = json.loads(line)
        rows[(r["key"], r["region"])] = r
    order = [r["key"] for r in apply_shifts.doc()["segments"]] + CONTROLS
    cols = ["key", "segment", "group", "region", "sample", "centre_y", "centre_x", "before_dy", "before_dx", "model_dy", "model_dx",
            "pred_err_dy", "pred_err_dx", "after_dy", "after_dx", "after_ncc", "after_edge", "methodB_dy", "methodB_dx",
            "methodB_ncc", "before_ncc", "noop", "bytes_downloaded"]
    with open(os.path.join(OUTD, "validation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in order:
            for (kk, lab), r in sorted(rows.items()):
                if kk != k:
                    continue
                w.writerow([k, r["segment"], r["group"], lab, r["sample"], r["centre"][0], r["centre"][1],
                            round(r["before"]["level0_dy"], 3), round(r["before"]["level0_dx"], 3),
                            round(r["model_offset_at_centre"][0], 3), round(r["model_offset_at_centre"][1], 3),
                            round(r["prediction_error"][0], 3), round(r["prediction_error"][1], 3),
                            round(r["after"]["level0_dy"], 3), round(r["after"]["level0_dx"], 3), round(r["after"]["ncc"], 3),
                            r["after"]["on_edge"], round(r["methodB"]["level0_dy"], 3), round(r["methodB"]["level0_dx"], 3),
                            round(r["methodB"]["ncc"], 3), round(r["before"]["ncc"], 3),
                            json.dumps(r.get("noop")) if "noop" in r else "", r["bytes_downloaded"]])
    print("| segment | group | region | sample | before dy, dx | model at centre | after residual dy, dx | after NCC | method B dy, dx |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in order:
        for (kk, lab), r in sorted(rows.items()):
            if kk != k:
                continue
            b, a, mb, p = r["before"], r["after"], r["methodB"], r["model_offset_at_centre"]
            print(f"| {r['segment'].split('/', 1)[1]} | {r['group']} | {lab} ({int(r['centre'][0])}, {int(r['centre'][1])}) | {r['sample']} |"
                  f" {b['level0_dy']:+.2f}, {b['level0_dx']:+.2f} | {p[0]:+.2f}, {p[1]:+.2f} | {a['level0_dy']:+.2f}, {a['level0_dx']:+.2f} |"
                  f" {a['ncc']:.3f} | {mb['level0_dy']:+.2f}, {mb['level0_dx']:+.2f} |")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "table":
        table()
    elif cmd == "farregion":
        cachelayer.install(os.environ["AUDIT_DATA"])
        farregion(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
        print(f"new downloads total {common.ledger_total()} bytes", flush=True)
    elif cmd == "run":
        old = os.environ.get("AUDIT_DATA")
        if not old:
            raise SystemExit("set AUDIT_DATA to the audit's data folder (its cache is read first)")
        n_old, n_new = cachelayer.install(old)
        print(f"cachelayer: {n_old} audit ranges, {n_new} own ranges indexed; new downloads -> {common.LEDGER}", flush=True)
        run(sys.argv[2:])
