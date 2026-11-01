"""Ceiling check for the label-shift study (2026-11-01): how much held-out AUC would an idealized model lose if it
reproduced the moved labels exactly? CkY moves C0 down by k*4 level-0 px in the tifxyz frame. Here the move is made on
the label canvas, before the HF65 remap, which is smooth and separable, so this is an approximation.

usage: python -I ceiling.py RAW_ROOT REGIONS_JSON OUT_JSON
RAW_ROOT holds <bucket scroll>/<segment>/ with the label TIFFs, as pod_run.sh prepare downloads them to data/raw.
Scored at level 2 (canvas[::4, ::4], the nearest pyramid the trainer reads at volume_scale 2) on HELD & supervision,
where HELD is each held-out box padded by margin_px, pooled over a segment's boxes. Predictors: the moved ink itself
(binary, the sharpest possible model) and the moved ink blurred by a Gaussian of sigma 0.5 to 4 level-2 px for
softer models. The first run had sigma 2 and 4 only; 0.5 and 1 were added after it. AUC by Mann-Whitney ranks, ties
counted half. Decision rule, written here before the first run: if the binary model's 64 px cost, as a share of its
headroom, is at most 0.166 (the upper end the w00 data allow for C16y), a stronger model cannot move the C16y
conclusion past what is already ruled out.
"""
import json
import os
import sys

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(1 << 36))
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

SHIFTS = (0, 32, 64, 128)
SIGMAS = (0.5, 1.0, 2.0, 4.0)
CTX = 64  # extra level-0 px around each crop so the blur has real context at the box edge


def read_tif(path):
    try:
        import tifffile
        return np.ascontiguousarray(tifffile.imread(path))
    except ValueError as exc:
        if "imagecodecs" not in str(exc):
            raise
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cv2 could not read {path}")
    return np.ascontiguousarray(img[..., 0] if img.ndim == 3 else img)


def auc(score, truth):
    pos = truth.sum()
    neg = truth.size - pos
    r = rankdata(score)
    return float((r[truth].sum() - pos * (pos + 1) / 2) / (pos * neg))


def crop(a, y0, y1, x0, x1):
    """a[y0:y1, x0:x1] with zeros outside the array."""
    out = np.zeros((y1 - y0, x1 - x0), a.dtype)
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, a.shape[0]), max(x0, 0), min(x1, a.shape[1])
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out


raw_root, regions, out_json = sys.argv[1:4]
R = json.load(open(regions, encoding="utf-8"))
margin = int(R["margin_px"])
res = {"shifts_level0_px": SHIFTS, "sigmas_level2_px": SIGMAS, "segments": {}}
for key, S in R["segments"].items():
    d = os.path.join(raw_root, S.get("bucket_scroll", S["scroll"]), S["segment"])
    ink_raw = read_tif(os.path.join(d, S["ink_tif"]))
    vals, cnts = np.unique(ink_raw[::16, ::16], return_counts=True)
    ink = ink_raw > 0
    del ink_raw
    sup = read_tif(os.path.join(d, S["sup_tif"])) > 0
    boxes = []
    for b in S["heldout"].values():
        y0, y1, x0, x1 = b
        # padded box, start snapped down to the level-2 grid (multiples of 4 on the canvas)
        Y0, X0 = (y0 - margin) // 4 * 4, (x0 - margin) // 4 * 4
        boxes.append((Y0, y1 + margin, X0, x1 + margin))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2], ("padded boxes overlap", a, b)
    scores = {s: {"bin": [], **{f"g{g:g}": [] for g in SIGMAS}} for s in SHIFTS}
    truth, overlap = [], {s: [0, 0] for s in SHIFTS}
    for Y0, Y1, X0, X1 in boxes:
        ev = crop(sup, Y0, Y1, X0, X1)[::4, ::4]
        tr = crop(ink, Y0, Y1, X0, X1)[::4, ::4]
        truth.append(tr[ev])
        for s in SHIFTS:
            # moved[y] = ink[y - s]; read with CTX of context on every side, then cut back to the box
            big = crop(ink, Y0 - s - CTX, Y1 - s + CTX, X0 - CTX, X1 + CTX)[::4, ::4].astype(np.float32)
            c = CTX // 4
            mv = big[c:-c, c:-c]
            assert mv.shape == tr.shape
            scores[s]["bin"].append(mv[ev])
            overlap[s][0] += int((mv[ev] > 0)[tr[ev]].sum())
            overlap[s][1] += int(tr[ev].sum())
            for g in SIGMAS:
                scores[s][f"g{g:g}"].append(gaussian_filter(big, g)[c:-c, c:-c][ev])
    t = np.concatenate(truth)
    seg = {"segment": S["segment"], "label_values_sampled": dict(zip(map(int, vals), map(int, cnts))),
           "eval_px_level2": int(t.size), "ink_share": float(t.mean()), "auc": {}, "ink_still_covered": {}}
    for s in SHIFTS:
        seg["auc"][s] = {k: auc(np.concatenate(v), t) for k, v in scores[s].items()}
        seg["ink_still_covered"][s] = overlap[s][0] / overlap[s][1]
    seg["cost_share_of_headroom"] = {
        s: {k: (seg["auc"][0][k] - seg["auc"][s][k]) / (seg["auc"][0][k] - 0.5) for k in seg["auc"][s]}
        for s in SHIFTS if s}
    res["segments"][key] = seg
    del ink, sup
    print(key, S["segment"], "eval px", t.size, "ink share", round(float(t.mean()), 4), "label values", seg["label_values_sampled"])
    for s in SHIFTS:
        a = seg["auc"][s]
        print(f"  shift {s:3d}: covered {seg['ink_still_covered'][s]:.3f}  AUC " + "  ".join(f"{k} {v:.4f}" for k, v in a.items())
              + ("" if not s else "  share " + "  ".join(f"{k} {v:.3f}" for k, v in seg["cost_share_of_headroom"][s].items())))
json.dump(res, open(out_json, "w", encoding="utf-8", newline="\n"), indent=1)
print("wrote", out_json)
