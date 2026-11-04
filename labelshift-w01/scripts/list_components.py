"""List the supervised areas of a segment as connected components (level 4, canvas frame).

usage: python -I list_components.py RAW_SEG_DIR SEG_NAME [VAL_ZARR_NAME]
Components of supervision > 0 at level 4 after a 3-px (48 level-0 px) dilation, so the letters of one
boxed word join. Prints id, level-0 canvas bbox, supervised and ink level-0 px (from level 4), and
the fraction of each component covered by the validation mask if one is given.
"""
import json
import os
import sys

import numpy as np
import zarr
from scipy import ndimage

raw, seg = sys.argv[1], sys.argv[2]
valname = sys.argv[3] if len(sys.argv) > 3 else None
L, f = 4, 16


def mid(name):
    a = zarr.open_array(os.path.join(raw, name, str(L)), mode="r")
    return np.asarray(a[a.shape[0] // 2]) > 0


sup = mid(f"{seg}_supervision_mask.zarr")
ink = mid(f"{seg}_inklabels.zarr")
val = mid(valname) if valname else None
lab, n = ndimage.label(ndimage.binary_dilation(sup, iterations=3), structure=np.ones((3, 3)))
rows = []
for i, sl in enumerate(ndimage.find_objects(lab), 1):
    m = lab[sl] == i
    s = int((sup[sl] & m).sum()) * f * f
    k = int((ink[sl] & m).sum()) * f * f
    v = None if val is None else round(float((val[sl] & m & sup[sl]).sum() * f * f) / max(s, 1), 3)
    rows.append(dict(id=i, y0=sl[0].start * f, y1=sl[0].stop * f, x0=sl[1].start * f, x1=sl[1].stop * f,
                     sup_px=s, ink_px=k, val_frac=v))
tot_ink = sum(r["ink_px"] for r in rows)
print(f"{seg}: {n} components, ink {tot_ink} level-0 px (from level 4)")
for r in rows:
    print(json.dumps({**r, "ink_share": round(r["ink_px"] / tot_ink, 3)}))
