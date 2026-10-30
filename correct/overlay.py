"""Before/after checkerboard: high-passed HF65 slice 32 and the CT sampled on the tifxyz, in
alternating 32 x 32 level-2 px squares (128 level-0 px). Where the two line up, structures run
straight across the square edges; where they do not, every edge shows a break.
  before: CT at tifxyz cell = canvas px (no correction)
  after : CT at tifxyz cell = fwd(canvas px), README fit (regcorr.py mode after, zero shift)
Needs npz/before_<key>_<label>.npz and npz/after_<key>_<label>.npz from regcorr.py.
argv: key label out.png [scale]"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "measure"))
import numpy as np
from lib import highpass
from PIL import Image as PILImage, ImageDraw, ImageFont

R, SQ, GAP = 32, 32, 12
key, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
scale = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ROOT = os.path.join(HERE, "..")


def row(mode):
    for line in open(os.path.join(ROOT, "results", f"{mode}_{key}.jsonl")):
        r = json.loads(line)
        if r["region"] == label:
            last = r
    return last


def norm(a, m):
    b = highpass(a, m, 4)
    lo, hi = np.percentile(b[m], [1, 99])
    return (np.clip((b - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


zb = np.load(os.path.join(ROOT, "npz", f"before_{key}_{label}.npz"))
za = np.load(os.path.join(ROOT, "npz", f"after_{key}_{label}.npz"))
I, Im = zb["I"], zb["Im"]
h = I.shape[0]
oy, ox = int(zb["oy"]), int(zb["ox"])
y, x = R - oy, R - ox
assert 0 <= y and y + h <= zb["S"].shape[0] and 0 <= x and x + h <= zb["S"].shape[1], "identity window not in the before samples"
assert np.array_equal(za["I"], I)
Sb, Smb = zb["S"][y:y + h, x:x + h], zb["Sm"][y:y + h, x:x + h]
Sa, Sma = za["S"][R:R + h, R:R + h], za["Sm"][R:R + h, R:R + h]
A = norm(I, Im)
yy, xx = np.mgrid[:h, :h]
chk = ((yy // SQ) + (xx // SQ)) % 2 == 0
pb = np.where(chk, A, norm(Sb, Smb))
pa = np.where(chk, A, norm(Sa, Sma))
pb = np.kron(pb, np.ones((scale, scale), np.uint8))
pa = np.kron(pa, np.ones((scale, scale), np.uint8))
rb, ra = row("before"), row("after")
H = pb.shape[0]
try:
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=13)
except TypeError:
    font = small = ImageFont.load_default()
top, bot = 46, 58
W = 2 * H + 3 * GAP
img = PILImage.new("L", (W, top + H + bot), 255)
img.paste(PILImage.fromarray(pb), (GAP, top))
img.paste(PILImage.fromarray(pa), (2 * GAP + H, top))
d = ImageDraw.Draw(img)
seg = {"w00": "PHercParis4 w00", "w01": "PHercParis4 w01_r15", "w05": "PHercParis4 w05_4424", "c018": "PHerc1667 w018"}[key]
d.text((GAP, 6), f"{seg}, canvas level-0 y {rb['y0']}..{rb['y0'] + rb['size']}, x {rb['x0']}..{rb['x0'] + rb['size']}, {'in-sample, used by the fit' if rb['sample'].startswith('in') else 'out-of-sample, not used by the fit'}; "
                 f"squares alternate HF65 slice 32 and CT on the tifxyz, 128 px each", fill=0, font=small)
d.text((GAP, 24), "BEFORE: CT at tifxyz cell (y, x)", fill=0, font=font)
d.text((2 * GAP + H, 24), "AFTER: CT at tifxyz cell fwd(y, x), README fit", fill=0, font=font)
zb_ncc = rb["ncc_zero_shift_hp"]
d.text((GAP, top + H + 6), f"measured offset ({rb['level0_dy']:+.1f}, {rb['level0_dx']:+.1f}) px", fill=0, font=font)
d.text((GAP, top + H + 26), f"NCC at zero shift {zb_ncc:.2f}, peak {rb['hp']['ncc']:.2f}", fill=0, font=font)
d.text((2 * GAP + H, top + H + 6), f"residual ({ra['level0_dy']:+.2f}, {ra['level0_dx']:+.2f}) px", fill=0, font=font)
d.text((2 * GAP + H, top + H + 26), f"NCC at zero shift {ra['ncc_zero_shift_hp']:.2f}, peak {ra['hp']['ncc']:.2f}", fill=0, font=font)
img.save(out, optimize=True)
print(f"{out}: {img.size[0]}x{img.size[1]}, {os.path.getsize(out) / 1024:.0f} KB; before offset ({rb['level0_dy']:+.2f},{rb['level0_dx']:+.2f})"
      f" zero-shift ncc {zb_ncc:.3f}; after residual ({ra['level0_dy']:+.2f},{ra['level0_dx']:+.2f}) zero-shift ncc {ra['ncc_zero_shift_hp']:.3f}")
