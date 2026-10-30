"""Before/after overlay of MAN5_outer_3_inklabels on front_inklabels, level 1, slice 32.
Blue: MAN5_outer_3_inklabels only; orange: front_inklabels only; black: both.
Before: offset (0, 0). After: the offset given (level-0 px), placed with place.py (a fractional
level-1 shift is linear-interpolated and thresholded at 128 for display).
argv: dy dx out.png [crop]  (crop: level-1 px per side, default 560; IoU numbers read from
results/man5_eval.json, level 1)"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "measure"))
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import place

dy, dx, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
C = int(sys.argv[4]) if len(sys.argv) > 4 else 560
GAP = 12
GH, GW = place.grid_shape(1)
F = place.Lab("front", 1).slice2d(32, 0, GH, 0, GW) > 0
P0, _ = place.place("inklabels", 1, 0, 0)
P1, _ = place.place("inklabels", 1, dy, dx)
B0, B1 = P0 >= 128, P1 >= 128
# crop: the C x C level-1 block with the most overlap after the shift
S = np.cumsum(np.cumsum(np.pad((B1 & F).astype(np.int64), ((1, 0), (1, 0))), 0), 1)
d = S[C:, C:] - S[:-C, C:] - S[C:, :-C] + S[:-C, :-C]
i, j = np.unravel_index(np.argmax(d[::16, ::16]), d[::16, ::16].shape)
y0, x0 = 16 * i, 16 * j


def rgb(B):
    f, b = F[y0:y0 + C, x0:x0 + C], B[y0:y0 + C, x0:x0 + C]
    im = np.full((C, C, 3), 255, np.uint8)
    im[f & ~b] = (230, 159, 0)
    im[b & ~f] = (0, 114, 178)
    im[b & f] = (0, 0, 0)
    return im


ev = json.load(open(os.path.join(HERE, "..", "results", "man5_eval.json")))["levels"]["1"]["rows"]
def row(a, b):
    return next(r for r in ev if r["dy"] == a and r["dx"] == b)
r0, r1 = row(0, 0), row(dy, dx)
try:
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=13)
except TypeError:
    font = small = ImageFont.load_default()
top, bot = 46, 58
img = PILImage.new("RGB", (2 * C + 3 * GAP, top + C + bot), (255, 255, 255))
img.paste(PILImage.fromarray(rgb(B0)), (GAP, top))
img.paste(PILImage.fromarray(rgb(B1)), (2 * GAP + C, top))
dr = ImageDraw.Draw(img)
dr.text((GAP, 6), f"MAN5_outer_3, level 1, slice 32, segment grid y {y0}..{y0 + C}, x {x0}..{x0 + C} (level-1 px);"
                  f" blue MAN5_outer_3_inklabels only, orange front_inklabels only, black both", fill=(0, 0, 0), font=small)
dr.text((GAP, 24), "BEFORE: label pixel (y, x) at grid (y, x)", fill=(0, 0, 0), font=font)
dr.text((2 * GAP + C, 24), f"AFTER: label pixel (y, x) at grid (y + {dy}, x + {dx}), level-0 px", fill=(0, 0, 0), font=font)
for k, (r, x) in enumerate(((r0, GAP), (r1, 2 * GAP + C))):
    dr.text((x, top + C + 6), f"whole slice: IoU {r['iou']:.3f}", fill=(0, 0, 0), font=font)
    dr.text((x, top + C + 26), f"MAN5 ink on front ink {r['share_man5_on_front']:.3f}", fill=(0, 0, 0), font=font)
img.save(out, optimize=True)
print(f"{out}: {img.size}, {os.path.getsize(out) / 1024:.0f} KB; crop y0={y0} x0={x0} C={C}")
