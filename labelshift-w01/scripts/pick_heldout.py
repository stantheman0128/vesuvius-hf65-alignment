"""Pick the three held-out components of w01 by a fixed rule (PREREG.md section 2).

usage: python -I pick_heldout.py COMPONENTS_LOG SUP_ZARR_DIR FIT_KEY OUT_JSON

COMPONENTS_LOG is the output of list_components.py for the shipped w01 labels: one JSON line per
supervised component (level-4 supervision after a 48 px dilation), with its level-0 canvas box
[y0, y1) x [x0, x1) and its share of the segment's ink. SUP_ZARR_DIR is the shipped
supervision_mask.zarr (level 4 is read, the level list_components.py uses).

October's rule for w00 was "three labelled components holding roughly 10 to 20% of the ink, mapped
into the tifxyz frame and padded by 64 px; supervision within 512 px of a box is removed from
training". It does not say which three. This script fixes that choice, written before any w01
component list was looked at:

  eligible component: its box mapped by fwd and padded by margin_px lies fully inside the tifxyz
      grid; the box padded by margin_px in the canvas frame (where the shipped labels and the AS arm
      put it) lies inside the guard box (mapped box padded by margin_px + guard_px), so neither the
      AS copy of a held-out component nor the as-shipped eval boxes reach training supervision; and
      it holds at least 2% of the segment's ink.
  candidate triple: three eligible components whose mapped, padded boxes do not overlap and whose
      ink shares sum to between 10% and 20%.
  choice: the candidate triple that removes the least training supervision, counted as supervised
      level-4 pixels inside the union of the three guard boxes minus those inside the union of the
      three held-out boxes, both in the canvas frame (the correction is a translation plus a scale
      of under 0.5%, so the canvas frame stands in for the tifxyz frame here). Ties: ink share
      closest to 15%, then the smallest ids.

Writes OUT_JSON with the chosen ids and boxes, every eligible component and the ten best triples.
"""
import itertools
import json
import os
import sys

import numpy as np
import zarr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hf65"))
import apply_fit  # noqa: E402

MARGIN, GUARD, F = 64, 512, 16  # level-0 px; level 4 = 16 level-0 px
MIN_SHARE, LO, HI, TARGET = 0.02, 0.10, 0.20, 0.15

comp_log, sup_zarr, fit_key, out_json = sys.argv[1:5]
model = apply_fit.model(fit_key)
H, W = apply_fit.FITS[apply_fit.ALIAS.get(fit_key, fit_key)]["tifxyz"]
comps = [json.loads(line) for line in open(comp_log) if line.startswith("{")]

a = zarr.open_array(os.path.join(sup_zarr, "4"), mode="r")
sup4 = np.asarray(a[a.shape[0] // 2]) > 0
ii = np.zeros((sup4.shape[0] + 1, sup4.shape[1] + 1), np.int64)
ii[1:, 1:] = sup4.cumsum(0).cumsum(1)


def rect_sum(r):
    """supervised level-4 px in a level-0 canvas box [y0, y1, x0, x1), clipped to the array."""
    y0, y1 = max(0, r[0] // F), min(sup4.shape[0], -(-r[1] // F))
    x0, x1 = max(0, r[2] // F), min(sup4.shape[1], -(-r[3] // F))
    if y0 >= y1 or x0 >= x1:
        return 0
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def inter(rs):
    return (max(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), min(r[3] for r in rs))


def union_sum(rs):
    """inclusion-exclusion over the boxes; an empty intersection sums to 0."""
    tot = 0
    for k in range(1, len(rs) + 1):
        for sub in itertools.combinations(rs, k):
            tot += (-1) ** (k + 1) * rect_sum(inter(sub))
    return tot


def pad(r, p):
    return (r[0] - p, r[1] + p, r[2] - p, r[3] + p)


def mapped(c, p):
    ty, tx = model.fwd(np.array([c["y0"], c["y0"], c["y1"], c["y1"]], float),
                       np.array([c["x0"], c["x1"], c["x0"], c["x1"]], float))
    return (int(np.floor(ty.min())) - p, int(np.ceil(ty.max())) + p, int(np.floor(tx.min())) - p, int(np.ceil(tx.max())) + p)


def inside(r, outer):
    return r[0] >= outer[0] and r[1] <= outer[1] and r[2] >= outer[2] and r[3] <= outer[3]


rows = []
for c in comps:
    held_t = mapped(c, MARGIN)
    guard_t = pad(held_t, GUARD)
    canvas_held = pad((c["y0"], c["y1"], c["x0"], c["x1"]), MARGIN)
    dy, dx = model.offset((c["y0"] + c["y1"]) / 2, (c["x0"] + c["x1"]) / 2)
    e = {"id": c["id"], "box": [c["y0"], c["y1"], c["x0"], c["x1"]], "ink_share": c["ink_share"],
         "offset_at_centre": [round(float(dy), 1), round(float(dx), 1)],
         "in_grid": held_t[0] >= 0 and held_t[1] <= H and held_t[2] >= 0 and held_t[3] <= W,
         "as_box_inside_guard": inside(canvas_held, guard_t),
         "share_ok": c["ink_share"] >= MIN_SHARE}
    e["eligible"] = e["in_grid"] and e["as_box_inside_guard"] and e["share_ok"]
    rows.append(e)
    print(json.dumps(e), flush=True)

elig = [e for e in rows if e["eligible"]]
by_id = {c["id"]: c for c in comps}
cands = []
for t in itertools.combinations(elig, 3):
    share = sum(e["ink_share"] for e in t)
    if not LO <= share <= HI:
        continue
    held_t = [mapped(by_id[e["id"]], MARGIN) for e in t]
    if any(inter([p, q])[0] < inter([p, q])[1] and inter([p, q])[2] < inter([p, q])[3]
           for p, q in itertools.combinations(held_t, 2)):
        continue
    boxes = [tuple(e["box"]) for e in t]
    lost = union_sum([pad(b, MARGIN + GUARD) for b in boxes]) - union_sum([pad(b, MARGIN) for b in boxes])
    cands.append({"ids": [e["id"] for e in t], "ink_share": round(share, 4), "lost_level4_px": lost,
                  "lost_level0_px": lost * F * F})
cands.sort(key=lambda c: (c["lost_level4_px"], abs(c["ink_share"] - TARGET), c["ids"]))
total_sup = int(sup4.sum())
print(f"{len(comps)} components, {len(elig)} eligible, {len(cands)} candidate triples; "
      f"supervised level-4 px in the segment {total_sup}", flush=True)
for c in cands[:10]:
    print(json.dumps({**c, "lost_share_of_supervision": round(c["lost_level4_px"] / total_sup, 4)}), flush=True)
out = {"rule": __doc__, "components": rows, "n_candidates": len(cands), "best10": cands[:10],
       "supervised_level4_px": total_sup, "chosen": None}
if cands:
    best = cands[0]
    out["chosen"] = {"ids": best["ids"], "ink_share": best["ink_share"],
                     "boxes": {str(i): [by_id[i]["y0"], by_id[i]["y1"], by_id[i]["x0"], by_id[i]["x1"]]
                               for i in best["ids"]},
                     "lost_share_of_supervision": round(best["lost_level4_px"] / total_sup, 4)}
    print("CHOSEN", json.dumps(out["chosen"]), flush=True)
else:
    print("NO CANDIDATE TRIPLE: the rule cannot be applied as written", flush=True)
json.dump(out, open(out_json, "w"), indent=1)
