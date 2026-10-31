"""Level-5 middle-slice comparison across the six 1667 training folders.
chosen = <folder>_supervision_mask / _inklabels (what villa picks); v2 = <name>_*_v2 (ignored)."""
import json, os, sys
import numpy as np

D = sys.argv[1]
segs = ["w013", "w018", "w023", "w028", "w029", "w031"]


def ld(seg, name):
    p = os.path.join(D, f"{seg}_{name}_L5.npz")
    return (np.load(p)["mid"] > 0) if os.path.exists(p) else None


rows = []
for s in segs:
    cs, ci = ld(s, "2um_supervision_mask"), ld(s, "2um_inklabels")
    v1s = ld(s, "supervision_mask")
    vs, vi, vv = ld(s, "supervision_mask_v2"), ld(s, "inklabels_v2"), ld(s, "validation_mask_v2")
    r = {"seg": s, "HW": list(cs.shape), "chosen_sup": int(cs.sum()), "chosen_sup_frac": float(cs.mean()),
         "chosen_ink": int(ci.sum()), "chosen_ink_frac": float(ci.mean()),
         "chosen_sup_equals_v1_sup": None if v1s is None else bool((cs == v1s).all())}
    if vs is not None:
        r.update({"v2_sup": int(vs.sum()), "v2_sup_frac": float(vs.mean()),
                  "v2_ink": int(vi.sum()), "chosen_sup_over_v2_sup": float(cs.sum() / vs.sum()),
                  "chosen_ink_over_v2_ink": float(ci.sum() / vi.sum()),
                  "chosen_sup_not_in_v2_sup": int((cs & ~vs).sum()),
                  "v2_val": int(vv.sum()), "chosen_sup_in_v2_val": int((cs & vv).sum()),
                  "v2_ink_in_chosen_sup": int((vi & cs).sum()),
                  "v2_ink_train_part": int((vi & vs & ~vv).sum())})
    rows.append(r)
    print(json.dumps(r))
json.dump(rows, open(os.path.join(os.path.dirname(D.rstrip("/")), "crossseg_L5.json"), "w"), indent=1)
