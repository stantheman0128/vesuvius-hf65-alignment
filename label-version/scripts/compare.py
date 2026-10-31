"""Pixel comparisons between the w029 label arrays at one level (middle slice z=32).
usage: compare.py <arr dir> <level>"""
import json, os, sys
import numpy as np

D, L = sys.argv[1], sys.argv[2]
names = ["2um_supervision_mask", "supervision_mask", "supervision_mask_v2", "validation_mask_v2",
         "2um_inklabels", "inklabels", "inklabels_v2"]
a = {n: np.load(os.path.join(D, f"w029_{n}_L{L}.npz"))["mid"] > 0 for n in names}
H, W = a[names[0]].shape
out = {"level": L, "shape_HW": [H, W], "px": H * W}
for n in names:
    m = a[n]
    ys, xs = np.nonzero(m)
    out[n] = {"set": int(m.sum()), "frac": float(m.mean()),
              "bbox_yx": None if not len(ys) else [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]}
sup_c, sup_1, sup_2, val_2 = a["2um_supervision_mask"], a["supervision_mask"], a["supervision_mask_v2"], a["validation_mask_v2"]
ink_c, ink_1, ink_2 = a["2um_inklabels"], a["inklabels"], a["inklabels_v2"]
pair = lambda x, y: {"x": int(x.sum()), "y": int(y.sum()), "and": int((x & y).sum()),
                     "x_not_y": int((x & ~y).sum()), "y_not_x": int((y & ~x).sum()), "equal": bool((x == y).all())}
out["chosen_sup_vs_v1_sup"] = pair(sup_c, sup_1)
out["chosen_ink_vs_v1_ink"] = pair(ink_c, ink_1)
out["chosen_sup_vs_v2_sup"] = pair(sup_c, sup_2)
out["chosen_ink_vs_v2_ink"] = pair(ink_c, ink_2)
out["v2_sup_vs_v2_val"] = pair(sup_2, val_2)
out["label_coverage"] = {
    "chosen_ink_in_chosen_sup": int((ink_c & sup_c).sum()),
    "chosen_ink_total": int(ink_c.sum()),
    "v2_ink_total": int(ink_2.sum()),
    "v2_ink_in_chosen_sup": int((ink_2 & sup_c).sum()),
    "v2_ink_in_v2_sup": int((ink_2 & sup_2).sum()),
    "v2_ink_in_v2_sup_not_val": int((ink_2 & sup_2 & ~val_2).sum()),
    "v2_ink_in_v2_val": int((ink_2 & val_2).sum()),
    "v2_ink_in_v2_sup_or_val": int((ink_2 & (sup_2 | val_2)).sum()),
    "chosen_ink_in_v2_sup": int((ink_c & sup_2).sum()),
    "v2_sup_not_val": int((sup_2 & ~val_2).sum()),
}
print(json.dumps(out, indent=1))
