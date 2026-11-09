"""Numpy-only helpers for render_beforeafter.py: flat-grid geometry, 3D -> 2D readout, blending,
label windows, a rank AUC and figure composition. Nothing here imports torch or villa, so these
parts can be tested on synthetic arrays (test_figlib.py).

Frames. "level 0" is the tifxyz full-resolution cell grid (ty, tx). "flat" is villa's flat grid
at the training volume_scale: flat pixel (Y, X) is level-0 cell (Y * f, X * f) with
f = native_tifxyz_pyramid_params(scale)[0] (4 for A's volume_scale 2). Patch windows, label
zarr levels and every 2D array written by the renderer are on the flat grid.
"""
from __future__ import annotations

import math

import numpy as np


# --- geometry -------------------------------------------------------------------------------------

def normal_steps(half_thickness: float) -> list[int]:
    """Integer normal offsets villa's label projection marks (geometry._project_mask_along_normals):
    every step in [-ceil(h), ceil(h)] with |step| <= h + 1e-6. For A (3 level-0 voxels / 4) only 0."""
    radius = int(math.ceil(half_thickness))
    return [s for s in range(-radius, radius + 1) if abs(s) <= half_thickness + 1e-6]


def project_to_flat(volume, crop_start, positions, valid, normals=None, half_thickness=0.0,
                    reduce="mean"):
    """Read a ZYX crop back onto the flat grid it was cut around.

    volume      (D, H, W) array whose voxel (0, 0, 0) is native voxel crop_start (z0, y0, x0)
    positions   (h, w, 3) native ZYX position of each flat pixel (geometry.read_tifxyz_on_flat_grid)
    valid       (h, w) bool
    normals     (h, w, 3) ZYX normals, needed only when normal_steps(half_thickness) has a nonzero step
    Each flat pixel reads the voxels the label projection would mark for it at step s: position
    + s * unit normal, truncated to int like geometry._mark and geometry.project_flat_patch. The
    values over the steps that land inside the crop are averaged ("mean") or maxed ("max").
    Returns float32 (h, w) with NaN where no step lands inside the crop.
    """
    volume = np.asarray(volume)
    positions = np.asarray(positions, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    h, w = valid.shape
    if positions.shape != (h, w, 3):
        raise ValueError(f"positions {positions.shape} vs valid {valid.shape}")
    steps = normal_steps(half_thickness)
    ok = valid & np.isfinite(positions).all(axis=-1)
    unit = None
    if any(s != 0 for s in steps):
        if normals is None:
            raise ValueError("normals are required for a nonzero half thickness")
        normals = np.asarray(normals, dtype=np.float32)
        if normals.shape != (h, w, 3):
            raise ValueError(f"normals {normals.shape} vs valid {valid.shape}")
        mag = np.sqrt((normals.astype(np.float64) ** 2).sum(axis=-1))
        ok &= np.isfinite(mag) & (mag > 1e-6)
        unit = normals / np.where(mag > 1e-6, mag, 1.0)[..., None]
    start = np.asarray(crop_start, dtype=np.int64)
    shape = np.asarray(volume.shape, dtype=np.int64)
    acc = np.zeros((h, w), np.float64)
    cnt = np.zeros((h, w), np.int64)
    best = np.full((h, w), -np.inf)
    for s in steps:
        p = positions if s == 0 else positions + float(s) * unit
        p = np.where(ok[..., None], p, 0.0)
        local = np.trunc(p).astype(np.int64) - start
        inb = ok & (local >= 0).all(axis=-1) & (local < shape).all(axis=-1)
        if not inb.any():
            continue
        li = local[inb]
        vals = volume[li[:, 0], li[:, 1], li[:, 2]].astype(np.float64)
        acc[inb] += vals
        cnt[inb] += 1
        best[inb] = np.maximum(best[inb], vals)
    out = np.full((h, w), np.nan, np.float32)
    hit = cnt > 0
    if reduce == "mean":
        out[hit] = (acc[hit] / cnt[hit]).astype(np.float32)
    elif reduce == "max":
        out[hit] = best[hit].astype(np.float32)
    else:
        raise ValueError(f"reduce must be mean or max, got {reduce!r}")
    return out


def gaussian_weight_2d(h, w, sigma_scale=0.125):
    """2D slice of villa's create_importance_map(mode='gaussian') (infer_full3d_tifxyz.py):
    separable Gaussian, sigma = size * sigma_scale, max 1, clipped at float32 eps."""
    axes = []
    for size in (h, w):
        sigma = max(float(size) * sigma_scale, 1e-6)
        c = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
        axes.append(np.exp(-0.5 * (c / sigma) ** 2).astype(np.float32))
    wgt = axes[0][:, None] * axes[1][None, :]
    wgt /= max(float(wgt.max()), np.finfo(np.float32).eps)
    return np.clip(wgt, np.finfo(np.float32).eps, None).astype(np.float32)


class FlatAccumulator:
    """Weighted average of overlapping flat tiles inside one window [y0, y1) x [x0, x1).
    NaN pixels of a tile contribute nothing."""

    def __init__(self, window, names):
        self.y0, self.y1, self.x0, self.x1 = (int(v) for v in window)
        shape = (self.y1 - self.y0, self.x1 - self.x0)
        self.sum = {n: np.zeros(shape, np.float64) for n in names}
        self.wsum = {n: np.zeros(shape, np.float64) for n in names}

    def overlaps(self, ty0, tx0, h, w):
        return ty0 < self.y1 and ty0 + h > self.y0 and tx0 < self.x1 and tx0 + w > self.x0

    def add(self, name, ty0, tx0, tile, weight):
        tile = np.asarray(tile, np.float64)
        weight = np.asarray(weight, np.float64)[: tile.shape[0], : tile.shape[1]]
        h, w = tile.shape
        a, b = max(ty0, self.y0), min(ty0 + h, self.y1)
        c, d = max(tx0, self.x0), min(tx0 + w, self.x1)
        if a >= b or c >= d:
            return
        t = tile[a - ty0:b - ty0, c - tx0:d - tx0]
        wt = weight[a - ty0:b - ty0, c - tx0:d - tx0]
        good = np.isfinite(t)
        dst = (slice(a - self.y0, b - self.y0), slice(c - self.x0, d - self.x0))
        self.sum[name][dst] += np.where(good, t * wt, 0.0)
        self.wsum[name][dst] += np.where(good, wt, 0.0)

    def result(self, name):
        s, wsum = self.sum[name], self.wsum[name]
        out = np.full(s.shape, np.nan, np.float32)
        hit = wsum > 0
        out[hit] = (s[hit] / wsum[hit]).astype(np.float32)
        return out

    def coverage(self, name):
        return (self.wsum[name] > 0)


def flat_window(boxes_level0, factor, pad, flat_hw):
    """Smallest flat window holding every level-0 box [y0, y1, x0, x1], padded by `pad` flat px and
    clipped to the flat grid (flat_hw = ceil(level-0 shape / factor))."""
    b = np.asarray(boxes_level0, dtype=np.int64)
    y0 = int(b[:, 0].min()) // factor - pad
    y1 = -(-int(b[:, 1].max()) // factor) + pad
    x0 = int(b[:, 2].min()) // factor - pad
    x1 = -(-int(b[:, 3].max()) // factor) + pad
    return max(0, y0), min(flat_hw[0], y1), max(0, x0), min(flat_hw[1], x1)


def lattice_starts(lo, hi, size, stride_scan, scale):
    """Flat starts y0 = round(k * stride_scan * scale), k >= 0, whose window [y0, y0 + size) meets
    [lo, hi). This is the corner lattice of patch_finding_default.find_segment_patches (corners at
    scan-grid multiples of the stride, mapped to the flat grid by round(y_scan * scale_y))."""
    step = stride_scan * scale
    k0 = max(0, int(math.floor((lo - size) / step)) - 1)
    k1 = int(math.ceil(hi / step)) + 1
    out = []
    for k in range(k0, k1 + 1):
        y0 = int(round(k * stride_scan * scale))
        if y0 < hi and y0 + size > lo:
            out.append(y0)
    return sorted(set(out))


def box_in_window(box_level0, window, factor):
    """Level-0 box [y0, y1, x0, x1] -> [r0, r1, c0, c1] in pixels of a flat window array."""
    wy0, _, wx0, _ = window
    y0, y1, x0, x1 = box_level0
    return (y0 // factor - wy0, -(-y1 // factor) - wy0, x0 // factor - wx0, -(-x1 // factor) - wx0)


# --- labels ---------------------------------------------------------------------------------------

def gather_window(canvas_win, r0, c0, rows, cols):
    """canvas[rows][:, cols] read from canvas_win, the canvas block starting at (r0, c0).
    rows/cols are canvas indices (-1 = outside the canvas, as build_labels.index_map gives).
    Returns (values, known) where known is False where the row or column index is -1."""
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    rv, cv = rows >= 0, cols >= 0
    out = np.zeros((len(rows), len(cols)), canvas_win.dtype)
    if rv.any() and cv.any():
        rr, cc = rows[rv] - r0, cols[cv] - c0
        if rr.min() < 0 or cc.min() < 0 or rr.max() >= canvas_win.shape[0] or cc.max() >= canvas_win.shape[1]:
            raise IndexError("canvas window does not cover the requested rows/cols")
        out[np.ix_(rv, cv)] = canvas_win[np.ix_(rr, cc)]
    known = rv[:, None] & cv[None, :]
    return out, known


def span(*index_arrays):
    """(lo, hi) over the non-negative entries of several index arrays (hi exclusive)."""
    vals = np.concatenate([np.asarray(a, np.int64)[np.asarray(a) >= 0] for a in index_arrays])
    if vals.size == 0:
        raise ValueError("no index inside the canvas")
    return int(vals.min()), int(vals.max()) + 1


# --- metrics ----------------------------------------------------------------------------------------

def rank_auc(scores, labels):
    """ROC AUC by the Mann-Whitney rank sum with average ranks for ties; None if a class is empty."""
    scores = np.asarray(scores, np.float64).ravel()
    labels = np.asarray(labels, bool).ravel()
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    _, first, counts = np.unique(s, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0  # 1-based average rank of each tie group
    ranks = np.empty(s.size)
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --- rendering --------------------------------------------------------------------------------------

NAN_RGB = (40, 40, 90)
INK_RGB = (255, 255, 255)
BG_RGB = (0, 0, 0)
UNSUP_RGB = (110, 110, 110)
UNKNOWN_RGB = NAN_RGB
SCORED_RGB = (0, 200, 255)    # held-out box where A scored against C0 (C0 placement)
SHIPPED_RGB = (255, 150, 0)   # the same component where the shipped labels put it


def render_labels(ink, sup, known=None):
    """ink white, supervised background black, unsupervised grey, outside the canvas dark blue."""
    ink = np.asarray(ink) > 0
    sup = np.asarray(sup) > 0
    rgb = np.empty(ink.shape + (3,), np.uint8)
    rgb[:] = UNSUP_RGB
    rgb[sup] = BG_RGB
    rgb[ink] = INK_RGB
    if known is not None:
        rgb[~np.asarray(known, bool)] = UNKNOWN_RGB
    return rgb


def render_gray(values, vmin=None, vmax=None):
    """Linear grey ramp; NaN dark blue. Without vmin/vmax, the 1st and 99th percentiles."""
    v = np.asarray(values, np.float64)
    fin = np.isfinite(v)
    if vmin is None or vmax is None:
        if fin.any():
            lo, hi = np.percentile(v[fin], [1, 99])
        else:
            lo, hi = 0.0, 1.0
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    scale = (vmax - vmin) if vmax > vmin else 1.0
    g = np.clip((np.where(fin, v, vmin) - vmin) / scale, 0, 1)
    rgb = np.repeat((g * 255 + 0.5).astype(np.uint8)[..., None], 3, axis=-1)
    rgb[~fin] = NAN_RGB
    return rgb


def draw_rect(rgb, rect, color, width=2):
    """Outline [r0, r1, c0, c1] (clipped) on an (H, W, 3) array in place."""
    H, W = rgb.shape[:2]
    r0, r1, c0, c1 = (int(v) for v in rect)
    r0c, r1c, c0c, c1c = max(r0, 0), min(r1, H), max(c0, 0), min(c1, W)
    if r0c >= r1c or c0c >= c1c:
        return rgb
    for k in range(width):
        if 0 <= r0 + k < H:
            rgb[r0 + k, c0c:c1c] = color
        if 0 <= r1 - 1 - k < H:
            rgb[r1 - 1 - k, c0c:c1c] = color
        if 0 <= c0 + k < W:
            rgb[r0c:r1c, c0 + k] = color
        if 0 <= c1 - 1 - k < W:
            rgb[r0c:r1c, c1 - 1 - k] = color
    return rgb


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _text_block(width, height, text, size, bg=(255, 255, 255), fg=(0, 0, 0)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (max(1, width), max(1, height)), bg)
    ImageDraw.Draw(img).text((4, 2), text, fill=fg, font=_font(size))
    return np.asarray(img)


def resize_to_height(rgb, height):
    from PIL import Image
    h, w = rgb.shape[:2]
    if h == height:
        return rgb
    new_w = max(1, int(round(w * height / h)))
    img = Image.fromarray(rgb).resize((new_w, height), Image.Resampling.BOX if height < h else Image.Resampling.NEAREST)
    return np.asarray(img)


def compose_row(panels, titles, gap=8, title_h=24, font_size=16, bg=255):
    """Panels side by side (all resized to the first panel's height) with a title above each."""
    height = panels[0].shape[0]
    cols = []
    for p, t in zip(panels, titles, strict=True):
        p = resize_to_height(np.asarray(p, np.uint8), height)
        head = _text_block(p.shape[1], title_h, t, font_size)
        cols.append(np.concatenate([head, p], axis=0))
    spacer = np.full((height + title_h, gap, 3), bg, np.uint8)
    parts = []
    for i, c in enumerate(cols):
        if i:
            parts.append(spacer)
        parts.append(c)
    return np.concatenate(parts, axis=1)


def stack_rows(rows, headers, gap=12, header_h=26, font_size=18, footer=None, bg=255):
    """Rows (from compose_row) stacked vertically, each with a header line; widths padded to the widest."""
    width = max(r.shape[1] for r in rows)
    parts = []
    for r, hdr in zip(rows, headers, strict=True):
        parts.append(_text_block(width, header_h, hdr, font_size))
        pad = np.full((r.shape[0], width - r.shape[1], 3), bg, np.uint8)
        parts.append(np.concatenate([r, pad], axis=1))
        parts.append(np.full((gap, width, 3), bg, np.uint8))
    if footer:
        for line in footer:
            parts.append(_text_block(width, 20, line, 14))
    return np.concatenate(parts, axis=0)


def save_png(path, rgb):
    from PIL import Image
    Image.fromarray(np.asarray(rgb, np.uint8)).save(path)
