"""Check that a label TIFF equals the label zarr the trainer reads, level by level.

usage: python -I check_tif_vs_zarr.py TIF ZARR_DIR LISTING_JSON
For every chunk file of the zarr that is present on disk (and listed in the bucket listing), compare
its slice-32 plane with TIF[::2**k, ::2**k] over the same window (villa create_label_zarrs writes
labels with "nearest" downsampling, i.e. a [::2, ::2] stride per level, into slice 32 of 65).
Also checks that the other 64 slices are zero, and that every listed chunk was compared or is absent.
"""
import json
import os
import sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labellib import read_label_tif  # noqa: E402
import zarr

tif_path, zdir, listing = sys.argv[1:4]
T = read_label_tif(tif_path)
print(f"tif {os.path.basename(tif_path)} {T.shape} {T.dtype} values {np.unique(T[::64, ::64])}")
listed = {}
for e in json.load(open(listing)):
    if e["type"] == "file":
        inner = e["path"].split(".zarr/", 1)[1]
        if not os.path.basename(inner).startswith("."):
            listed.setdefault(inner.split("/")[0], []).append(inner)
for lvl in sorted(listed, key=int):
    a = zarr.open_array(os.path.join(zdir, lvl), mode="r")
    k = 2 ** int(lvl)
    S = T[::k, ::k]
    if S.shape != a.shape[1:]:
        print(f"level {lvl}: SHAPE MISMATCH zarr {a.shape} vs tif stride {S.shape}")
        continue
    n = bad = other = 0
    present = 0
    for inner in listed[lvl]:
        _, cz, cy, cx = inner.split("/")
        if not os.path.exists(os.path.join(zdir, inner)):
            continue
        present += 1
        y0, x0 = int(cy) * 128, int(cx) * 128
        blk = np.asarray(a[:, y0:y0 + 128, x0:x0 + 128])
        ref = S[y0:y0 + 128, x0:x0 + 128]
        bad += int((blk[32] != ref).sum())
        other += int(np.count_nonzero(blk[np.arange(65) != 32]))
        n += ref.size
    print(f"level {lvl}: {present}/{len(listed[lvl])} listed chunks present, {n} px compared, "
          f"{bad} differ from tif[::{k}, ::{k}], {other} nonzero px outside slice 32")
