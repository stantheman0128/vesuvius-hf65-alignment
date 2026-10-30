"""Build correct/shifts/shifts.json and shifts.csv: one canvas -> tifxyz correction per offset training segment.

Inputs, all in this repo:
  audit/results/segments.csv            canvas and tifxyz full-res sizes (phase 1)
  audit/results/phase2/reg_<key>.jsonl  per-region offsets measured by audit/code/reg2.py
  results/results_<key>.jsonl           per-region offsets of the original HF65 measurement (w00, w01, w05, c018)
  correct/apply_fit.py FITS             the published, tested fits for those four segments

Rules:
  one_cell, second_shift  constant d = mean over the segment's own measured regions.
  paris4_size_ratio       d = a + s * position per axis; s = tifxyz full / canvas - 1 from the metadata,
                          a = mean over the segment's own measured regions of (d - s * region centre).
  the four segments that apply_fit.py already corrects keep its FITS unchanged.
Fit residual = largest |measured - model| over the fit regions, per axis (empty when one region).

python -I correct/shifts/build_shifts.py
"""
import os, sys, csv, json
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "correct"))
import numpy as np
import apply_fit

PHASE2 = os.path.join(REPO, "audit", "results", "phase2")
TARGETS = json.load(open(os.path.join(REPO, "audit", "code", "targets.json")))

# key -> (group, source); source "audit" = reg_<key>.jsonl, "published" = apply_fit FITS + results_<key>.jsonl
ROWS = [
    ("c013", "one_cell", "audit"), ("c018", "one_cell", "published"), ("c023", "one_cell", "audit"),
    ("c028", "one_cell", "audit"), ("c029v", "one_cell", "audit"), ("c031", "one_cell", "audit"),
    ("b0009", "one_cell", "audit"),
    ("t016", "one_cell", "audit"), ("t017", "one_cell", "audit"), ("t028", "one_cell", "audit"),
    ("t029", "one_cell", "audit"), ("t030", "one_cell", "audit"),
    ("t035", "second_shift", "audit"), ("t039", "second_shift", "audit"), ("t040", "second_shift", "audit"),
    ("t041", "second_shift", "audit"), ("t043", "second_shift", "audit"), ("t044", "second_shift", "audit"),
    ("w00", "paris4_size_ratio", "published"), ("w01", "paris4_size_ratio", "published"),
    ("p4w02", "paris4_size_ratio", "audit"), ("p4w03", "paris4_size_ratio", "audit"),
    ("w05", "paris4_size_ratio", "published"), ("p4w06", "paris4_size_ratio", "audit"),
    ("p4w07", "paris4_size_ratio", "audit"), ("p4w09", "paris4_size_ratio", "audit"),
]
# aligned in phase 2 (|d - 0.35| <= 1.5 px): no correction; listed so the apply step can tell "aligned" from "unknown"
ALIGNED = ["p500", "n841", "n841a", "n841b", "man5"]
# training segment that phase 2 could not register (canvas CT not public)
UNMEASURED = {"814/814_46527_2um_try2": "canvas came from a 20250717 2.403um TA2 scan of 814 that is not on public S3"}
PUBLISHED_SEG = {k: f["segment"] for k, f in apply_fit.FITS.items()}
P4VOL = "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
C1667VOL = "PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr"
PUBLISHED_VOL = {"w00": P4VOL, "w01": P4VOL, "w05": P4VOL, "c018": C1667VOL}

CONVENTION = {
    "units": "level-0 canvas pixels; slopes are pixels per pixel",
    "canvas": "(y, x) = level-0 pixel of <segment>.zarr in hf://buckets/scrollprize/datasets/ink/<scroll>/<segment>/; "
              "the ink labels and supervision masks there share this grid; pixel centres at integers",
    "tifxyz": "(ty, tx) = full-resolution cell of the tifxyz (x.tif, y.tif, z.tif, meta.json) in the same folder, villa convention: "
              "full-res cell (r, c) is stored cell (r * scale_y, c * scale_x), bilinear (vesuvius tifxyz/types.py:153-154)",
    "model": "canvas content at (y, x) lies at tifxyz cell (y + dy, x + dx), with dy = dy_a + dy_per_y * y + dy_per_x * x and "
             "dx = dx_a + dx_per_y * y + dx_per_x * x, evaluated at canvas coordinates (same sign and form as correct/apply_fit.py)",
    "full_3d": "villa full_3d pairs label pixel (y, x) with tifxyz cell (y, x); the label value that belongs to tifxyz cell (ty, tx) "
               "is the canvas/label value at inv(ty, tx)",
    "measurement_offset": "the same measurement reads +0.4 to +0.5 px on both axes on the five aligned segments; the parameters "
                          "are the measured values and do not remove it (as in apply_fit.py)",
}


def last_by_region(path, region_key="region"):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r[region_key]] = r
    return [rows[k] for k in sorted(rows)]


def seg_info():
    out = {}
    for r in csv.DictReader(open(os.path.join(REPO, "audit", "results", "segments.csv"))):
        if r["unused_dir"] != "no":
            continue
        out[f"{r['scroll']}/{r['segment']}"] = {
            "canvas_hw": [int(v) for v in r["canvas_hw"].split("x")],
            "tifxyz_full_hw": [int(v) for v in r["tifxyz_full_hw"].split("x")],
            "scale_xy": [float(v) for v in r["tifxyz_scale_xy"].split()],
        }
    return out


def build():
    info = seg_info()
    rows = []
    for key, group, source in ROWS:
        if source == "audit":
            t = TARGETS[key]
            seg = t["seg"].split("/", 1)[1]
            vol = t["vol"]
            regs = last_by_region(os.path.join(PHASE2, f"reg_{key}.jsonl"))
            fit_regions = [f"{key} r{r['region']} (y0 {r['y0']}, x0 {r['x0']})" for r in regs]
            fit_from = f"audit/results/phase2/reg_{key}.jsonl"
        else:
            seg = PUBLISHED_SEG[key]
            vol = PUBLISHED_VOL[key]
            regs = last_by_region(os.path.join(REPO, "results", f"results_{key}.jsonl"))
            fit_regions = [f"{key} r{r['region']} (y0 {r['y0']}, x0 {r['x0']})" for r in regs]
            fit_from = f"correct/apply_fit.py FITS['{key}'] (fit to results/results_{key}.jsonl)"
        si = info[seg]
        H, W = si["canvas_hw"]
        Ht, Wt = si["tifxyz_full_hw"]
        Y = np.array([r["y0"] + r["size"] / 2 for r in regs], float)
        X = np.array([r["x0"] + r["size"] / 2 for r in regs], float)
        DY = np.array([r["level0_dy"] for r in regs], float)
        DX = np.array([r["level0_dx"] for r in regs], float)
        NCC = np.array([r["hp"]["ncc"] for r in regs], float)
        if source == "published":
            f = apply_fit.FITS[key]
            cy, cx = list(f["dy"]), list(f["dx"])
            slope_src = "constant" if group == "one_cell" else "fitted slopes of the published fit"
        elif group == "paris4_size_ratio":
            sy, sx = Ht / H - 1, Wt / W - 1
            cy = [float(np.mean(DY - sy * Y)), sy, 0.0]
            cx = [float(np.mean(DX - sx * X)), 0.0, sx]
            slope_src = "metadata: tifxyz full / canvas - 1"
        else:
            cy, cx = [float(DY.mean()), 0.0, 0.0], [float(DX.mean()), 0.0, 0.0]
            slope_src = "constant"
        mdl = apply_fit.Model(cy, cx)
        py_, px_ = mdl.offset(Y, X)
        ry, rx = DY - py_, DX - px_
        model_type = "constant" if (cy[1] == cy[2] == cx[1] == cx[2] == 0) else "linear"
        scroll, folder = seg.split("/", 1)
        rows.append({
            "key": key, "scroll": scroll, "segment": folder, "bucket_path": f"ink/{seg}", "group": group,
            "model": model_type,
            "dy_a": round(cy[0], 4), "dy_per_y": round(cy[1], 7), "dy_per_x": round(cy[2], 7),
            "dx_a": round(cx[0], 4), "dx_per_y": round(cx[1], 7), "dx_per_x": round(cx[2], 7),
            "slope_source": slope_src, "params_from": fit_from,
            "fit_regions": fit_regions, "n_fit_regions": len(regs),
            "fit_resid_max_dy": round(float(np.abs(ry).max()), 3) if len(regs) > 1 else None,
            "fit_resid_max_dx": round(float(np.abs(rx).max()), 3) if len(regs) > 1 else None,
            "ncc_min": round(float(NCC.min()), 3), "ncc_max": round(float(NCC.max()), 3),
            "canvas_hw": si["canvas_hw"], "tifxyz_full_hw": si["tifxyz_full_hw"], "tifxyz_scale_xy": si["scale_xy"],
            "ct_volume": vol,
            "offset_at_canvas_corners": [[round(float(v), 2) for v in mdl.offset(yy, xx)]
                                         for yy, xx in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1))],
        })
    aligned = []
    for key in ALIGNED:
        t = TARGETS[key]
        regs = last_by_region(os.path.join(PHASE2, f"reg_{key}.jsonl"))
        aligned.append({"key": key, "bucket_path": t["seg"], "measured": [[round(r["level0_dy"], 2), round(r["level0_dx"], 2)] for r in regs],
                        "regions": [f"{key} r{r['region']} (y0 {r['y0']}, x0 {r['x0']})" for r in regs]})
    doc = {"description": "Per-segment canvas -> tifxyz offsets for the ink training segments of "
                          "hf://buckets/scrollprize/datasets/ink/ (villa#2370 and the 2026-10-30 ink bucket audit)",
           "convention": CONVENTION, "segments": rows, "aligned_no_correction": aligned,
           "not_measured": [{"bucket_path": f"ink/{k}", "reason": v} for k, v in UNMEASURED.items()]}
    with open(os.path.join(HERE, "shifts.json"), "w") as f:
        json.dump(doc, f, indent=1)
    cols = ["key", "scroll", "segment", "group", "model", "dy_a", "dy_per_y", "dy_per_x", "dx_a", "dx_per_y", "dx_per_x",
            "slope_source", "n_fit_regions", "fit_regions", "fit_resid_max_dy", "fit_resid_max_dx", "ncc_min", "ncc_max",
            "canvas_hw", "tifxyz_full_hw", "ct_volume", "params_from"]
    with open(os.path.join(HERE, "shifts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["; ".join(r[c]) if c == "fit_regions" else ("x".join(map(str, r[c])) if c.endswith("_hw") else
                        ("" if r[c] is None else r[c])) for c in cols])
    for r in rows:
        print(f"{r['key']:6s} {r['scroll']:12s} {r['segment']:42s} {r['group']:18s} {r['model']:8s} "
              f"dy {r['dy_a']:+8.3f} {r['dy_per_y']:+.6f}*y | dx {r['dx_a']:+8.3f} {r['dx_per_x']:+.6f}*x "
              f"n={r['n_fit_regions']} resid {r['fit_resid_max_dy']},{r['fit_resid_max_dx']} ncc {r['ncc_min']}-{r['ncc_max']}")
    print(len(rows), "rows;", len(aligned), "aligned;", len(UNMEASURED), "not measured")


if __name__ == "__main__":
    build()
