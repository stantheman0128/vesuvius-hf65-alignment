"""Local checks of figlib on synthetic arrays (no torch, no villa, no data).

usage: python -B test_figlib.py PUBLISH_A_DIR
PUBLISH_A_DIR is the labelshift-w01 folder; its scripts/build_labels.py and scripts/hf65/apply_fit.py are
imported read-only (-B keeps Python from writing __pycache__ there) for the label-geometry checks.
Writes test_out/ next to this file.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figlib  # noqa: E402

A_DIR = sys.argv[1]
sys.path.insert(0, os.path.join(A_DIR, "scripts"))
sys.path.insert(0, os.path.join(A_DIR, "scripts", "hf65"))
import apply_fit  # noqa: E402
from build_labels import box_to_tifxyz, gather, index_map  # noqa: E402

rng = np.random.default_rng(0)
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(("ok   " if cond else "FAIL ") + name)


# 1. readout at step 0 picks exactly the voxel int(position) - crop_start
D = 32
vol = np.arange(D ** 3, dtype=np.float64).reshape(D, D, D)
start = np.array([100, 200, 300])
pos = start + rng.uniform(-4, D + 4, size=(20, 24, 3))
valid = rng.random((20, 24)) > 0.1
out = figlib.project_to_flat(vol, start, pos, valid)
loc = np.trunc(pos).astype(int) - start
inside = valid & (loc >= 0).all(-1) & (loc < D).all(-1)
expect = np.full((20, 24), np.nan)
expect[inside] = vol[loc[inside][:, 0], loc[inside][:, 1], loc[inside][:, 2]]
check("step-0 readout equals vol[int(pos) - start], NaN outside crop or invalid",
      np.array_equal(np.isnan(out), np.isnan(expect)) and np.allclose(out[inside], expect[inside]))
check("normal_steps(0.75) == [0] (A: 3 level-0 voxels / factor 4)", figlib.normal_steps(0.75) == [0])
check("normal_steps(1.5) == [-1, 0, 1]", figlib.normal_steps(1.5) == [-1, 0, 1])

# 2. with thickness: mean / max over +-1 along a unit normal (normals given unnormalized)
pos2 = np.full((1, 1, 3), [110.2, 210.7, 310.9])
nrm = np.full((1, 1, 3), [2.0, 0.0, 0.0])  # +z, magnitude 2
m = figlib.project_to_flat(vol, start, pos2, np.ones((1, 1), bool), nrm, 1.5, reduce="mean")[0, 0]
mx = figlib.project_to_flat(vol, start, pos2, np.ones((1, 1), bool), nrm, 1.5, reduce="max")[0, 0]
vals = [vol[10 + s, 10, 10] for s in (-1, 0, 1)]
check("thickness 1.5: mean and max over z-1, z, z+1", np.isclose(m, np.mean(vals)) and np.isclose(mx, max(vals)))

# 3. roundtrip with villa's flat -> 3D scatter rule (geometry.project_flat_patch, copied here):
#    a flat mask scattered into a crop and read back returns the mask wherever the point is in the crop
#    and no other point lands on the same voxel with a different value
flat = (rng.random((40, 40)) > 0.6).astype(np.uint8)
yy, xx = np.mgrid[0:40, 0:40]
pos3 = np.stack([np.full((40, 40), 50.5), 1000 + yy * 1.0 + 0.3, 2000 + xx * 1.0 + 0.3], -1)  # one voxel per pixel
crop = (40, 990, 1990, 72, 1022, 2022)
scatter = np.zeros((32, 32, 32), np.uint8)
mapped = pos3[flat > 0].astype(np.int64) - np.array(crop[:3])
inb = (mapped >= 0).all(1) & (mapped < 32).all(1)
np.maximum.at(scatter.reshape(-1), np.ravel_multi_index(mapped[inb].T, scatter.shape), 1)
back = figlib.project_to_flat(scatter, crop[:3], pos3, np.ones((40, 40), bool), reduce="max")
fin = np.isfinite(back)
check("scatter (villa rule) -> readout roundtrip recovers the flat mask inside the crop",
      fin.sum() == 22 * 22 and np.array_equal(back[fin] > 0.5, flat[fin] > 0))

# 4. blending: Gaussian weights, NaN ignored, coverage
wgt = figlib.gaussian_weight_2d(8, 8)
check("gaussian weight: max 1 at centre, symmetric, positive",
      np.isclose(wgt.max(), 1.0) and np.allclose(wgt, wgt[::-1, ::-1]) and wgt.min() > 0)
acc = figlib.FlatAccumulator((10, 20, 10, 22), ["p"])
t1 = np.full((8, 8), 0.2)
t2 = np.full((8, 8), 0.8)
t2[0, 0] = np.nan
acc.add("p", 8, 8, t1, wgt)      # partly outside the window (top-left)
acc.add("p", 12, 12, t2, wgt)
r = acc.result("p")  # window pixel (i, j) is global (10 + i, 10 + j)
check("blend: single-tile pixel, weighted mean on overlap, NaN pixel takes the other tile only",
      np.isclose(r[1, 1], 0.2)
      and np.isclose(r[3, 3], (0.2 * wgt[5, 5] + 0.8 * wgt[1, 1]) / (wgt[5, 5] + wgt[1, 1]))
      and np.isclose(r[2, 2], 0.2))
check("blend: uncovered pixels are NaN", np.isnan(r[-1, -1]) and not acc.coverage("p")[-1, -1])

# 5. lattice: villa's corner rule (patch_finding_default.find_segment_patches, copied) on a synthetic
#    scan-scale support mask with a non-integer scale; every corner whose window meets [lo, hi) is found
scale = 3.9987
size = 128
stride_scan = max(1, int(round(int(size * 0.5) / scale)))
support = np.zeros((400, 400), bool)
support[150:230, 90:260] = True
ys, xs = np.nonzero(support)
corners = np.unique(np.stack([ys // stride_scan * stride_scan, xs // stride_scan * stride_scan], 1), axis=0)
villa_y0 = sorted({int(round(c * scale)) for c in corners[:, 0]})
lo, hi = 600, 900
want = [y for y in villa_y0 if y < hi and y + size > lo]
got = figlib.lattice_starts(lo, hi, size, stride_scan, scale)
check(f"lattice_starts covers villa's corners meeting the window ({len(want)} corners)", set(want) <= set(got))
check("lattice_starts returns only windows that meet [lo, hi)", all(y < hi and y + size > lo for y in got))

# 6. A's label geometry offline: held-out boxes through box_to_tifxyz match the build manifest
R = json.load(open(os.path.join(A_DIR, "regions.json")))
M = json.load(open(os.path.join(A_DIR, "labels", "manifest_full.json")))["segments"]["w01"]
S = R["segments"]["w01"]
H, W = M["tifxyz_full_shape"]
fit = apply_fit.model("w01")
held = [list(box_to_tifxyz(fit, b, R["margin_px"], (H, W))) for b in S["heldout"].values()]
check("box_to_tifxyz(regions) == manifest heldout_tifxyz_boxes (A's build)", held == M["heldout_tifxyz_boxes"])
f = 4
for (bid, cb), hb in zip(S["heldout"].items(), held):
    mg = R["margin_px"]
    shipped = [cb[0] - mg, cb[1] + mg, cb[2] - mg, cb[3] + mg]
    win = figlib.flat_window([hb, shipped], f, 32, (-(-H // f), -(-W // f)))
    rs = figlib.box_in_window(hb, win, f)
    rp = figlib.box_in_window(shipped, win, f)
    ok = all(0 <= r[0] < r[1] <= win[1] - win[0] and 0 <= r[2] < r[3] <= win[3] - win[2] for r in (rs, rp))
    check(f"box {bid}: flat window {win} ({win[1] - win[0]}x{win[3] - win[2]}) holds both placements", ok)
check("EAS boxes in manifest_eval equal canvas box +- margin",
      json.load(open(os.path.join(A_DIR, "labels", "manifest_eval.json")))["segments"]["w01"]["placed_boxes"]["EAS"]
      == [[b[0] - R["margin_px"], b[1] + R["margin_px"], b[2] - R["margin_px"], b[3] + R["margin_px"]] for b in S["heldout"].values()])

# 7. label window gather == build_labels.gather on the whole canvas (synthetic canvas, real index maps
#    restricted to a box): C0 via the HF65 inverse, AS identity
Hc, Wc = M["canvas_shape"]
inv_y = fit.inv(np.arange(H, dtype=float), np.zeros(H))[0]
inv_x = fit.inv(np.zeros(W), np.arange(W, dtype=float))[1]
c0_rows, c0_cols = index_map(inv_y, H, Hc, 0), index_map(inv_x, W, Wc, 0)
win = figlib.flat_window([held[0]], f, 32, (-(-H // f), -(-W // f)))
ty, tx = np.arange(win[0], win[1]) * f, np.arange(win[2], win[3]) * f
r0, r1 = figlib.span(c0_rows[ty], ty)
k0, k1 = figlib.span(c0_cols[tx], tx)
# a synthetic canvas only over the rows/cols needed, embedded at its true offset in a sparse "full" view
block = rng.integers(0, 2, size=(r1 - r0, k1 - k0)).astype(np.uint8) * 255


def full_gather(rows, cols):
    rv, cv = rows >= 0, cols >= 0
    o = np.zeros((len(rows), len(cols)), np.uint8)
    o[np.ix_(rv, cv)] = block[np.ix_(rows[rv] - r0, cols[cv] - k0)]
    return o


g_c0, known = figlib.gather_window(block, r0, k0, c0_rows[ty], c0_cols[tx])
g_as, _ = figlib.gather_window(block, r0, k0, ty, tx)
check("gather_window(C0) equals build_labels-style gather over the canvas", np.array_equal(g_c0, full_gather(c0_rows[ty], c0_cols[tx])))
check("gather_window(AS identity) equals canvas[ty, tx]", np.array_equal(g_as, block[np.ix_(ty - r0, tx - k0)]))
d_rows = (c0_rows[ty] - ty)
check(f"C0 vs AS row offset in box 5 window is {d_rows.min()}..{d_rows.max()} canvas px (nonzero: the two label panels differ)",
      np.abs(d_rows).max() > 0)
src = np.zeros((10, 10), np.uint8)
src[3, 4] = 7
check("build_labels.gather semantics (-1 -> 0) match gather_window",
      np.array_equal(gather(src, np.array([3, -1]), np.array([4, 4])),
                     figlib.gather_window(src, 0, 0, np.array([3, -1]), np.array([4, 4]))[0]))

# 8. rank AUC vs brute force
s = rng.normal(size=300)
s[:40] = 0.5  # ties
lab = rng.random(300) > 0.6
pairs = [(1.0 if a > b else 0.5 if a == b else 0.0) for a in s[lab] for b in s[~lab]]
check("rank_auc equals pairwise AUC with ties", np.isclose(figlib.rank_auc(s, lab), np.mean(pairs)))
check("rank_auc is None for one class", figlib.rank_auc(s, np.ones(300, bool)) is None)

# 9. figure composition on synthetic panels, as main() builds them
out_dir = os.path.join(HERE, "test_out")
os.makedirs(out_dir, exist_ok=True)
rows, heads = [], []
for bid, (h, w) in {"5": (158, 120), "17": (190, 248)}.items():
    yy, xx = np.mgrid[0:h, 0:w]
    ink = ((yy // 12 + xx // 12) % 5 == 0).astype(np.uint8)
    sup = np.ones_like(ink)
    sup[:, : w // 5] = 0
    known = np.ones_like(ink, bool)
    known[:4] = False
    pred = np.clip(ink * 0.8 + rng.normal(0, 0.1, ink.shape), 0, 1)
    pred[-6:] = np.nan
    ct = rng.normal(size=ink.shape)
    panels = [figlib.render_gray(ct), figlib.render_labels(ink, sup, known),
              figlib.render_labels(np.roll(ink, 18, 0), sup, known),
              figlib.render_gray(np.roll(pred, 18, 0), 0, 1), figlib.render_gray(pred, 0, 1)]
    for p in panels:
        figlib.draw_rect(p, (10, h - 10, 8, w - 8), figlib.SHIPPED_RGB)
        figlib.draw_rect(p, (16, h - 4, 8, w - 8), figlib.SCORED_RGB)
    row = figlib.compose_row(panels, ["CT", "C0", "AS", "pred AS", "pred C0"])
    figlib.save_png(os.path.join(out_dir, f"box{bid}_synthetic.png"), figlib.stack_rows([row], [f"box {bid}"], footer=["legend"]))
    rows.append(figlib.compose_row([figlib.resize_to_height(p, 120) for p in panels], ["CT", "C0", "AS", "pred AS", "pred C0"]))
    heads.append(f"box {bid}")
    check(f"box {bid}: label render colours (ink white, unsupervised grey, unknown blue)",
          tuple(panels[1][5, w - 1]) in (figlib.INK_RGB, figlib.BG_RGB) and tuple(panels[1][0, 0]) == figlib.UNKNOWN_RGB
          and tuple(panels[1][h // 2, 1]) in (figlib.UNSUP_RGB, figlib.INK_RGB))
    check(f"box {bid}: NaN prediction rows rendered dark blue, scored rect drawn",
          tuple(panels[4][-1, w // 2]) == figlib.NAN_RGB and tuple(panels[4][16, w // 2]) == figlib.SCORED_RGB)
combined = figlib.stack_rows(rows, heads, footer=["line 1", "line 2"])
figlib.save_png(os.path.join(out_dir, "combined_synthetic.png"), combined)
check(f"combined figure built: {combined.shape}", combined.ndim == 3 and combined.shape[2] == 3 and combined.shape[0] > 240)
np.savez_compressed(os.path.join(out_dir, "synthetic.npz"), ink=ink, pred=pred)
check("npz round trip", np.array_equal(np.load(os.path.join(out_dir, "synthetic.npz"))["ink"], ink))

n_fail = sum(not ok for _, ok in results)
print(f"{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
