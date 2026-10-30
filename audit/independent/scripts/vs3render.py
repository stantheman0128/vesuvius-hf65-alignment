"""S3 official render of a 0139 segment on 20260102150214 vs (a) bucket canvas and (b) my CT-at-tifxyz
samples, on a region already measured by vmeasure.py.
usage: vs3render.py <tag> <S3 surface-volume zarr path> [slices a,b]"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))  # audit/independent/scripts
AUDIT = os.path.dirname(os.path.dirname(HERE))  # audit/
sys.path.insert(0, HERE)
import numpy as np
from vcore import fetch_counted, s3, mine, BS, prep, phasecorr, refine

tag, SV = sys.argv[1], sys.argv[2]
sl = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(50, 59))
OUT = os.environ.get("HF65_RESULTS", os.path.join(AUDIT, "independent", "out"))
z = np.load(os.path.join(OUT, f"{tag}.npz"))
row = [json.loads(l) for l in open(os.path.join(OUT, "results.jsonl")) if json.loads(l)["tag"] == tag][-1]
y0, x0, n = row["region"]
f = int(z["f"])
za = json.loads(fetch_counted(s3(f"{SV}/0/.zarray")))
sep = za.get("dimension_separator", ".")
assert za["compressor"] is None and za["chunks"][0] == za["shape"][0]
print("S3 render shape", za["shape"])


def s3slice(k):
    out = np.zeros((n, n))
    for cy in range(y0 // 128, (y0 + n) // 128):
        for cx in range(x0 // 128, (x0 + n) // 128):
            d = BS.read(s3(f"{SV}/0/0{sep}{cy}{sep}{cx}"), k * 16384, (k + 1) * 16384 - 1)
            out[cy * 128 - y0:cy * 128 - y0 + 128, cx * 128 - x0:cx * 128 - x0 + 128] = np.frombuffer(d, np.uint8).reshape(128, 128)
    return out.reshape(n // f, f, n // f, f).mean(axis=(1, 3))  # centre f*i + (f-1)/2


A, C, va, vc = z["A"], z["C"], z["va"], z["vc"]
v = np.ones(A.shape, bool)
for k in sl:
    R = s3slice(k)
    vr = R > 0
    if vr.mean() < 0.5:
        print("slice", k, "mostly empty", vr.mean())
        continue
    pr = prep(R, vr)
    out = []
    for nm, X, vx in (("canvas", A, va), ("ct@tifxyz", np.where(vc, C, 0), vc)):
        px = prep(X, vx)
        _, dy, dx, pk, sec = phasecorr(px, pr)
        ry, rx, _ = refine(px, pr, round(dy), round(dx))
        out.append("%s->S3render d=(%.1f, %.1f) peak %.3f 2nd %.3f" % (nm, f * ry, f * rx, pk, sec))
    print("S3 slice", k, " | ".join(out))
print("mine", mine())
