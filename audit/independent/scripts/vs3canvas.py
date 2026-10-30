"""Compare the bucket canvas (level 0, slice 32) of w040 with the S3 surface volume rendered on
20260102150214 (109 slices), and the bucket x.tif with the S3 on-20260102150214 mesh x.tif, at the
audit's cached pyramid block. usage: vs3canvas.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))  # audit/independent/scripts
AUDIT = os.path.dirname(os.path.dirname(HERE))  # audit/
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AUDIT, "code"))
import numpy as np
from vcore import fetch_counted, hf, s3, mine, BS, prep, phasecorr, refine, TiffRows
from lib import Image

SEG = "PHerc0139/segments/20250831000000-w040_2025083102"
SV = f"{SEG}/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr"
za = json.loads(fetch_counted(s3(f"{SV}/0/.zarray")))
print("S3 surface volume L0", {k: za[k] for k in ("shape", "chunks", "compressor", "dtype")}, za.get("dimension_separator"))
y0, x0 = 12416, 15488
n = 256
B = Image("ink/0139/w040_2025083102/w040_2025083102.zarr", 0).region(y0, y0 + n, x0, x0 + n).astype(float)
sep = za.get("dimension_separator", ".")
Z, ch = za["shape"][0], za["chunks"]
assert za["compressor"] is None and ch[0] == Z
res = {}
for k in range(46, 63):
    out = np.zeros((n, n))
    for cy in range(y0 // 128, (y0 + n) // 128):
        for cx in range(x0 // 128, (x0 + n) // 128):
            u = s3(f"{SV}/0/0{sep}{cy}{sep}{cx}")
            d = BS.read(u, k * 16384, (k + 1) * 16384 - 1)
            out[cy * 128 - y0:cy * 128 - y0 + 128, cx * 128 - x0:cx * 128 - x0 + 128] = np.frombuffer(d, np.uint8).reshape(128, 128)
    res[k] = out
best = None
for k, o in res.items():
    for kb in (31, 32, 33):
        r = np.corrcoef(o.ravel(), B[kb].ravel())[0, 1]
        eq = float((o == B[kb]).mean())
        if best is None or r > best[0]:
            best = (r, k, kb, eq)
print("best S3 slice vs bucket slice: r=%.5f S3 k=%d bucket k=%d frac identical=%.4f" % best)
v = np.ones((n, n), bool)
pa, pb = prep(B[best[2]], v), prep(res[best[1]], v)
_, dy, dx, pk, sec = phasecorr(pa, pb)
print("phase corr bucket->S3 render:", refine(pa, pb, round(dy), round(dx)), "peak", pk, "2nd", sec)
# tifxyz bytes: bucket vs S3 mesh
bt = TiffRows(hf("ink/0139/w040_2025083102/x.tif"))
st = TiffRows(s3(f"{SEG}/mesh/20250831000000-on-20260102150214-2.399um.tifxyz/x.tif"))
print("x.tif dims bucket", bt.H, bt.W, "S3", st.H, st.W)
r = int(y0 * 0.05)
a = bt.block(r, r + 1, 0, bt.W)[0]
b = st.block(r, r + 1, 0, st.W)[0]
m = np.isfinite(a) & np.isfinite(b) & (a != -1) & (b != -1)
print("row", r, "valid", m.sum(), "max abs diff x", float(np.abs(a[m] - b[m]).max()) if m.any() else None)
print("mine", mine())
