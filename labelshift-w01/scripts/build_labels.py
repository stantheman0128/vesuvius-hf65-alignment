"""Build the label-shift arms as trainer-ready segment folders.

usage: python -I build_labels.py RAW_ROOT REGIONS_JSON VILLA_SRC OUT_ROOT [--arms AS,C0,C2y,...] [--smoke]

RAW_ROOT/<bucket scroll>/<segment>/ holds the downloaded meta.json, x/y/z.tif and the label TIFFs
(scripts/check_tif_vs_zarr.py shows they equal the label zarrs the trainer reads). VILLA_SRC is the
vesuvius/src of the villa tree the trainer runs from; its label_zarr writer and tifxyz reader are
used so the output has the shipped layout (65 x H x W, label in slice 32, 6 levels, [::2] nearest
pyramid) on the grid the trainer reads (tifxyz full-resolution shape).

Frames. canvas (y, x) = pixel of the shipped label TIFF. tifxyz cell (ty, tx) = full-res cell read by
full_3d training. The HF65 correction (scripts/hf65/apply_fit.py, "readme" model, the tested one)
says canvas content at (y, x) belongs at cell fwd(y, x); its inverse gives, for every cell, the
canvas pixel whose label it should carry (nearest, as apply_fit.resample --order 0). For these
models the map is separable (dy depends on y only, dx on x only), so each arm is a row index map
and a column index map into the canvas arrays:
  AS   cell (ty, tx) <- canvas (ty, tx)                      what the trainer reads today
  C0   cell (ty, tx) <- canvas (inv_y(ty), inv_x(tx))        corrected
  CkY  cell (ty, tx) <- C0 cell (ty - k*u, tx)               C0 moved down by k cells of u px
  CkX  cell (ty, tx) <- C0 cell (ty, tx - k*u)               C0 moved right
Labels and supervision move together (a registration error moves the whole annotation).
Every arm then gets the same held-out treatment, all in the tifxyz frame:
  HELD  = union of the held-out component boxes mapped by fwd and padded by margin_px
  inklabels         = arm labels, but C0 labels inside HELD (validation is always scored on C0)
  supervision_mask  = arm supervision minus GUARD (HELD padded by guard_px), so no arm trains
                      on or right next to a held-out tile
  validation_mask   = C0 supervision inside HELD (identical in every arm)
--smoke keeps only the smoke_train / smoke_heldout boxes (for the CPU smoke test).

Writes OUT_ROOT/<ARM>/<scroll>/<segment>/{<segment>_inklabels.zarr, _supervision_mask.zarr,
_validation_mask.zarr, meta.json, x.tif, y.tif, z.tif (relative symlinks)} and OUT_ROOT/manifest.json
with per-arm, per-level nonzero counts and a sha256 over every chunk file.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "hf65"))
from labellib import read_label_tif  # noqa: E402
import apply_fit  # noqa: E402

LEVELS = 6
TILE = 4096  # level-0 px; a multiple of 128 * 2**5, so every tile is whole chunks at every level
SLICE = 32


def parse_arms(text):
    arms = []
    for name in text.split(","):
        name = name.strip()
        if name in ("AS", "C0"):
            arms.append((name, None, 0))
        else:
            assert name[0] == "C" and name[-1] in "yx", name
            arms.append((name, name[-1], int(name[1:-1])))
    return arms


def box_to_tifxyz(model, box, pad, shape):
    """canvas box [y0, y1, x0, x1] -> tifxyz box via fwd of its corners, padded, clipped."""
    y0, y1, x0, x1 = box
    ty, tx = model.fwd(np.array([y0, y0, y1, y1], float), np.array([x0, x1, x0, x1], float))
    return (max(0, int(np.floor(ty.min())) - pad), min(shape[0], int(np.ceil(ty.max())) + pad),
            max(0, int(np.floor(tx.min())) - pad), min(shape[1], int(np.ceil(tx.max())) + pad))


def box_mask(boxes, ty0, tx0, h, w):
    m = np.zeros((h, w), bool)
    for y0, y1, x0, x1 in boxes:
        a, b = max(y0 - ty0, 0), min(y1 - ty0, h)
        c, d = max(x0 - tx0, 0), min(x1 - tx0, w)
        if a < b and c < d:
            m[a:b, c:d] = True
    return m


def gather(src, rows, cols):
    """src[rows][:, cols] with -1 (out of range) -> 0."""
    rv, cv = rows >= 0, cols >= 0
    out = np.zeros((len(rows), len(cols)), src.dtype)
    if rv.any() and cv.any():
        out[np.ix_(rv, cv)] = src[np.ix_(rows[rv], cols[cv])]
    return out


def index_map(inv, n_out, n_src, shift):
    """Canvas index for each output cell after moving content by +shift cells; -1 outside."""
    t = np.arange(n_out) - shift
    idx = np.full(n_out, -1, np.int64)
    ok = (t >= 0) & (t < n_out)
    s = np.rint(inv[t[ok]]).astype(np.int64)
    s[(s < 0) | (s >= n_src)] = -1
    idx[ok] = s
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_root")
    p.add_argument("regions")
    p.add_argument("villa_src")
    p.add_argument("out_root")
    p.add_argument("--arms", default="AS,C0,C2y,C4y,C8y,C2x,C4x,C8x")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--only", default=None, help="comma list of segment keys")
    a = p.parse_args()
    sys.path.insert(0, a.villa_src)
    from pathlib import Path
    import vesuvius.tifxyz as tifxyz
    from vesuvius.label_zarr import create_label_array, create_label_group, pyramid_shapes

    R = json.load(open(a.regions))
    unit = int(R["shift_cell_level0_px"])
    arms = parse_arms(a.arms)
    manifest = {"regions": R, "arms": a.arms, "smoke": a.smoke, "villa_src": a.villa_src, "segments": {}}
    for key, S in R["segments"].items():
        if a.only and key not in a.only.split(","):
            continue
        t0 = time.time()
        bucket_scroll = S.get("bucket_scroll", S["scroll"])
        raw = os.path.join(a.raw_root, bucket_scroll, S["segment"])
        surf = tifxyz.read_tifxyz(Path(raw))
        H, W = (int(v) for v in surf.full_resolution_shape)
        fit = apply_fit.FITS[apply_fit.ALIAS.get(S["fit_key"], S["fit_key"])]
        assert (H, W) == tuple(fit["tifxyz"]), f"{key}: villa full-res shape {(H, W)} != fit table {fit['tifxyz']}"
        model = apply_fit.model(S["fit_key"])
        assert model.separable, "builder assumes dy(y), dx(x)"
        ink = read_label_tif(os.path.join(raw, S["ink_tif"]))
        sup = read_label_tif(os.path.join(raw, S["sup_tif"]))
        assert ink.shape == sup.shape == tuple(fit["canvas"]), (ink.shape, sup.shape, fit["canvas"])
        Hc, Wc = ink.shape
        # separable inverse: canvas row for every tifxyz row, canvas col for every tifxyz col
        inv_y = model.inv(np.arange(H, dtype=float), np.zeros(H))[0]
        inv_x = model.inv(np.zeros(W), np.arange(W, dtype=float))[1]
        held = [box_to_tifxyz(model, b, R["margin_px"], (H, W)) for b in S["heldout"].values()]
        guard = [(max(0, y0 - R["guard_px"]), min(H, y1 + R["guard_px"]),
                  max(0, x0 - R["guard_px"]), min(W, x1 + R["guard_px"])) for y0, y1, x0, x1 in held]
        smoke_train = [box_to_tifxyz(model, b, R["margin_px"], (H, W)) for b in S["smoke_train"].values()]
        smoke_held = [box_to_tifxyz(model, b, R["margin_px"], (H, W)) for b in S["smoke_heldout"].values()]
        c0_rows, c0_cols = index_map(inv_y, H, Hc, 0), index_map(inv_x, W, Wc, 0)
        maps = {}
        for name, axis, k in arms:
            if name == "AS":
                r = np.arange(H, dtype=np.int64); r[r >= Hc] = -1
                c = np.arange(W, dtype=np.int64); c[c >= Wc] = -1
            else:
                r = index_map(inv_y, H, Hc, k * unit if axis == "y" else 0)
                c = index_map(inv_x, W, Wc, k * unit if axis == "x" else 0)
            maps[name] = (r, c)
        # output zarrs
        writers = {}
        for name, _, _ in arms:
            seg_dir = os.path.join(a.out_root, name, S["scroll"], S["segment"])
            os.makedirs(seg_dir, exist_ok=True)
            for f in ("meta.json", "x.tif", "y.tif", "z.tif"):
                link = os.path.join(seg_dir, f)
                if not os.path.lexists(link):
                    os.symlink(os.path.relpath(os.path.join(raw, f), seg_dir), link)
            writers[name] = {}
            for kind in ("inklabels", "supervision_mask", "validation_mask"):
                zp = Path(seg_dir) / f"{S['segment']}_{kind}.zarr"
                g = create_label_group(zp, levels=LEVELS)
                writers[name][kind] = [create_label_array(g, str(l), shape=s, dtype=np.uint8)
                                       for l, s in enumerate(pyramid_shapes((H, W), LEVELS))]
        # tiles that can hold anything: canvas supervision bbox, mapped and padded
        ys, xs = np.nonzero(sup[::16, ::16])
        cb = (ys.min() * 16, (ys.max() + 1) * 16, xs.min() * 16, (xs.max() + 1) * 16)
        pad = R["guard_px"] + 8 * unit + 64
        tb = box_to_tifxyz(model, cb, pad, (H, W))
        counts = {name: {kind: [0] * LEVELS for kind in writers[name]} for name in writers}
        ntiles = 0
        for ty0 in range(tb[0] // TILE * TILE, tb[1], TILE):
            for tx0 in range(tb[2] // TILE * TILE, tb[3], TILE):
                h, w = min(TILE, H - ty0), min(TILE, W - tx0)
                held_m = box_mask(held, ty0, tx0, h, w)
                guard_m = box_mask(guard, ty0, tx0, h, w)
                c0r, c0c = c0_rows[ty0:ty0 + h], c0_cols[tx0:tx0 + w]
                c0_sup = gather(sup, c0r, c0c)
                val = np.where(held_m, c0_sup, 0).astype(np.uint8)
                if a.smoke:
                    val[~box_mask(smoke_held, ty0, tx0, h, w)] = 0
                    keep = box_mask(smoke_train, ty0, tx0, h, w)
                any_written = False
                for name, (r, c) in maps.items():
                    rr, cc = r[ty0:ty0 + h], c[tx0:tx0 + w]
                    a_sup = gather(sup, rr, cc)
                    a_sup[guard_m] = 0
                    if a.smoke:
                        a_sup[~keep] = 0
                    a_ink = gather(ink, rr, cc)
                    if held_m.any():
                        a_ink[held_m] = gather(ink, c0r, c0c)[held_m]
                    for kind, arr in (("inklabels", a_ink), ("supervision_mask", a_sup), ("validation_mask", val)):
                        if not arr.any():
                            continue
                        any_written = True
                        for l in range(LEVELS):
                            f = 2 ** l
                            sub = arr[::f, ::f]
                            z = writers[name][kind][l]
                            y0l, x0l = ty0 // f, tx0 // f
                            # write only the 128 x 128 chunks that hold a nonzero pixel
                            hb, wb = -(-sub.shape[0] // 128), -(-sub.shape[1] // 128)
                            occ = np.zeros((hb * 128, wb * 128), bool)
                            occ[:sub.shape[0], :sub.shape[1]] = sub > 0
                            occ = occ.reshape(hb, 128, wb, 128).any(axis=(1, 3))
                            for by, bx in zip(*np.nonzero(occ)):
                                blk = sub[by * 128:(by + 1) * 128, bx * 128:(bx + 1) * 128]
                                yy, xx = y0l + by * 128, x0l + bx * 128
                                z[SLICE, yy:yy + blk.shape[0], xx:xx + blk.shape[1]] = blk
                            counts[name][kind][l] += int(np.count_nonzero(sub))
                ntiles += any_written
        # sha256 over chunk files (path + bytes) per arm and kind
        digests = {}
        for name in writers:
            digests[name] = {}
            for kind in writers[name]:
                zp = os.path.join(a.out_root, name, S["scroll"], S["segment"], f"{S['segment']}_{kind}.zarr")
                h_ = hashlib.sha256()
                for root, _, files in sorted(os.walk(zp)):
                    for fn in sorted(files):
                        fp = os.path.join(root, fn)
                        h_.update(os.path.relpath(fp, zp).encode())
                        h_.update(open(fp, "rb").read())
                digests[name][kind] = h_.hexdigest()
        manifest["segments"][key] = {
            "canvas_shape": [Hc, Wc], "tifxyz_full_shape": [H, W], "model": model.name,
            "model_dy": model.cy.tolist(), "model_dx": model.cx.tolist(),
            "heldout_tifxyz_boxes": held, "guard_boxes": guard,
            "smoke_train_boxes": smoke_train if a.smoke else None,
            "smoke_heldout_boxes": smoke_held if a.smoke else None,
            "nonzero_per_level": counts, "sha256": digests, "tiles_written": ntiles,
            "seconds": round(time.time() - t0, 1)}
        print(json.dumps({"segment": key, "tiles": ntiles, "seconds": round(time.time() - t0, 1),
                          "level0_nonzero": {n: {k: v[0] for k, v in counts[n].items()} for n in counts}}), flush=True)
        del ink, sup
    os.makedirs(a.out_root, exist_ok=True)
    with open(os.path.join(a.out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
