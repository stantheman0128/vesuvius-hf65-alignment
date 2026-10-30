"""Overlay PNG for a vmeasure.py result.
Panels: top-left canvas; top-right CT sampled at tifxyz cells (as full_3d sees it), same lattice;
bottom-left red=canvas / cyan=CT unshifted; bottom-right red=canvas / cyan=CT shifted by the measured d.
Yellow: ink-label contour (label zarr on the canvas grid), if the inklabels zarr can be read.
usage: vpng.py <tag> <png name> [crop lattice px]"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))  # audit/independent/scripts
AUDIT = os.path.dirname(os.path.dirname(HERE))  # audit/
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AUDIT, "code"))
import numpy as np
from scipy import ndimage as ndi
from PIL import Image as PImage, ImageDraw
from vcore import mine

OUT = os.environ.get("HF65_RESULTS", os.path.join(AUDIT, "independent", "out"))
tag, png = sys.argv[1], sys.argv[2]
crop = int(sys.argv[3]) if len(sys.argv) > 3 else 160
z = np.load(os.path.join(OUT, f"{tag}.npz"))
row = [json.loads(l) for l in open(os.path.join(OUT, "results.jsonl")) if json.loads(l)["tag"] == tag][-1]
A, C, va, vc = z["A"], z["C"], z["va"], z["vc"]
f = int(z["f"])
dy, dx = [float(v) for v in z["d"]]
T = json.load(open(os.path.join(AUDIT, "code", "targets.json")))
key = row["key"]
seg = T[key]["seg"]
lab = None
try:
    from lib import Image
    lname = T[key]["sup"].replace("supervision_mask", "inklabels")
    y0, x0, n = row["region"]
    li = Image(f"{seg}/{lname}", row["L"])
    ch = row["canvas_model"]["slice"]
    Lr = li.region(y0 // f, y0 // f + A.shape[0], x0 // f, x0 // f + A.shape[1])
    lab = Lr.max(axis=0) > 0
    labname = lname
except Exception as e:
    labname = f"none ({type(e).__name__}: {e})"


def norm(a, m):
    lo, hi = np.percentile(a[m], [1, 99]) if m.any() else (0, 1)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


Cs = ndi.shift(np.where(vc, C, np.nan), (-dy, -dx), order=1, mode="constant", cval=np.nan)
cy, cx = A.shape[0] // 2, A.shape[1] // 2
s = (slice(cy - crop // 2, cy + crop // 2), slice(cx - crop // 2, cx + crop // 2))
a = norm(A, va)[s]
c0 = norm(np.where(vc, C, 0), vc)[s]
c1 = norm(np.nan_to_num(Cs), np.isfinite(Cs))[s]
edge = None
if lab is not None:
    l = lab[s]
    edge = l ^ ndi.binary_erosion(l, iterations=1)


def rgb_gray(x):
    return np.stack([x, x, x], -1)


def rgb_pair(r, c):
    return np.stack([r, c, c], -1)


panels = [rgb_gray(a), rgb_gray(c0), rgb_pair(a, c0), rgb_pair(a, c1)]
up = 3
ims = []
for p in panels:
    p = p.copy()
    if edge is not None:
        p[edge] = [1.0, 1.0, 0.0]
    im = PImage.fromarray((p * 255).astype(np.uint8)).resize((crop * up, crop * up), PImage.NEAREST)
    ims.append(im)
W = crop * up
canvas = PImage.new("RGB", (2 * W + 10, 2 * W + 10 + 70), (255, 255, 255))
for k, im in enumerate(ims):
    canvas.paste(im, ((k % 2) * (W + 10), 70 + (k // 2) * (W + 10)))
d = ImageDraw.Draw(canvas)
d.text((4, 2), f"{tag}: {seg}  region y0,x0,n={row['region']}  canvas level {row['L']} (1 lattice px = {f} level-0 px), crop {crop} lattice px", fill=(0, 0, 0))
d.text((4, 16), f"measured d (level-0 px) = ({f * dy:+.1f}, {f * dx:+.1f}); phase-corr peak {row['pc_peak']:.3f}, 2nd {row['pc_second']:.3f}; raw NCC at 0 = {row['ncc_raw_at_zero']:.3f}, at d = {row['ncc_raw_at_shift']:.3f}", fill=(0, 0, 0))
d.text((4, 30), "TL canvas | TR CT sampled at tifxyz cells (full_3d) | BL red=canvas cyan=CT unshifted | BR red=canvas cyan=CT shifted by d", fill=(0, 0, 0))
d.text((4, 44), f"yellow = ink label contour: {labname}", fill=(0, 0, 0))
canvas.save(os.path.join(os.environ.get("HF65_RESULTS", os.path.join(AUDIT, "independent")), png))
print("wrote", png, "label", labname, "mine", mine())
