"""Build evaluation-only label folders for the eval offset scan (SCAN-2026-11-01.md).

usage: python -I build_eval_labels.py RAW_ROOT REGIONS_JSON VILLA_SRC OUT_ROOT [--arms E0,E4y,E8y,E16y,E32y,EAS]

Every training arm of build_labels.py carries C0 labels and the same validation mask inside the held-out boxes, so
scoring a checkpoint with another arm's config scores it against C0 again. These folders move the held-out
annotation instead. Each one is build_labels.py's C0 arm with one change: where the held-out annotation (C0 ink
labels plus the validation mask, which is C0 supervision inside HELD) sits.
  E0    unchanged. It must come out byte-identical to build_labels.py's C0 (sha256 in labels/manifest_full.json).
  EkY   the held-out annotation moved down by k cells (k * shift_cell_level0_px level-0 px): inside HELD moved
        down by k cells, cell (ty, tx) carries C0's cell (ty - k*u, tx), the move CkY makes in the training area.
  EkX   the same, moved right.
  EAS   the held-out annotation as shipped: inside the canvas boxes padded by margin_px, cell (ty, tx) carries
        canvas pixel (ty, tx), the way the AS arm reads its training labels.
supervision_mask is C0's (training area minus GUARD) and inklabels outside the moved held-out area are C0's. The
largest move here is 32 cells = 128 px < guard_px, so the moved annotation stays inside GUARD, where no arm has
training supervision. Output layout and manifest as in build_labels.py.
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
from build_labels import LEVELS, SLICE, TILE, apply_fit, box_mask, box_to_tifxyz, gather, index_map  # noqa: E402
from labellib import read_label_tif  # noqa: E402


def parse_arms(text):
    arms = []
    for name in text.split(","):
        name = name.strip()
        if name in ("E0", "EAS"):
            arms.append((name, None, 0))
        else:
            assert name[0] == "E" and name[-1] in "yx", name
            arms.append((name, name[-1], int(name[1:-1])))
    return arms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_root")
    p.add_argument("regions")
    p.add_argument("villa_src")
    p.add_argument("out_root")
    p.add_argument("--arms", default="E0,E4y,E8y,E16y,E32y,EAS")
    a = p.parse_args()
    sys.path.insert(0, a.villa_src)
    from pathlib import Path
    import vesuvius.tifxyz as tifxyz
    from vesuvius.label_zarr import create_label_array, create_label_group, pyramid_shapes

    R = json.load(open(a.regions))
    unit = int(R["shift_cell_level0_px"])
    arms = parse_arms(a.arms)
    assert max(k for _, _, k in arms) * unit < R["guard_px"], "a move must stay inside GUARD"
    manifest = {"regions": R, "arms": a.arms, "villa_src": a.villa_src, "segments": {}}
    for key, S in R["segments"].items():
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
        inv_y = model.inv(np.arange(H, dtype=float), np.zeros(H))[0]
        inv_x = model.inv(np.zeros(W), np.arange(W, dtype=float))[1]
        held = [box_to_tifxyz(model, b, R["margin_px"], (H, W)) for b in S["heldout"].values()]
        guard = [(max(0, y0 - R["guard_px"]), min(H, y1 + R["guard_px"]),
                  max(0, x0 - R["guard_px"]), min(W, x1 + R["guard_px"])) for y0, y1, x0, x1 in held]
        c0_rows, c0_cols = index_map(inv_y, H, Hc, 0), index_map(inv_x, W, Wc, 0)
        m = R["margin_px"]
        # per arm: where the held-out annotation goes (boxes) and which canvas pixel each cell there carries
        place = {}
        for name, axis, k in arms:
            if name == "EAS":
                r = np.arange(H, dtype=np.int64); r[r >= Hc] = -1
                c = np.arange(W, dtype=np.int64); c[c >= Wc] = -1
                boxes = [(max(0, y0 - m), min(H, y1 + m), max(0, x0 - m), min(W, x1 + m))
                         for y0, y1, x0, x1 in S["heldout"].values()]
            else:
                dy, dx = (k * unit, 0) if axis == "y" else (0, k * unit)
                r, c = index_map(inv_y, H, Hc, dy), index_map(inv_x, W, Wc, dx)
                boxes = [(min(H, y0 + dy), min(H, y1 + dy), min(W, x0 + dx), min(W, x1 + dx))
                         for y0, y1, x0, x1 in held]
            place[name] = (r, c, boxes)
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
        # same tile range as build_labels.py; its padding (guard_px + 8 cells + 64 px) holds a 128 px move
        ys, xs = np.nonzero(sup[::16, ::16])
        cb = (ys.min() * 16, (ys.max() + 1) * 16, xs.min() * 16, (xs.max() + 1) * 16)
        pad = R["guard_px"] + 8 * unit + 64
        tb = box_to_tifxyz(model, cb, pad, (H, W))
        counts = {name: {kind: [0] * LEVELS for kind in writers[name]} for name in writers}
        ntiles = 0
        for ty0 in range(tb[0] // TILE * TILE, tb[1], TILE):
            for tx0 in range(tb[2] // TILE * TILE, tb[3], TILE):
                h, w = min(TILE, H - ty0), min(TILE, W - tx0)
                guard_m = box_mask(guard, ty0, tx0, h, w)
                c0r, c0c = c0_rows[ty0:ty0 + h], c0_cols[tx0:tx0 + w]
                a_sup = gather(sup, c0r, c0c)
                a_sup[guard_m] = 0
                c0_ink = gather(ink, c0r, c0c)
                any_written = False
                for name, (r, c, boxes) in place.items():
                    rr, cc = r[ty0:ty0 + h], c[tx0:tx0 + w]
                    pm = box_mask(boxes, ty0, tx0, h, w)
                    val = np.where(pm, gather(sup, rr, cc), 0).astype(np.uint8)
                    a_ink = c0_ink.copy()
                    if pm.any():
                        a_ink[pm] = gather(ink, rr, cc)[pm]
                    for kind, arr in (("inklabels", a_ink), ("supervision_mask", a_sup), ("validation_mask", val)):
                        if not arr.any():
                            continue
                        any_written = True
                        for l in range(LEVELS):
                            f = 2 ** l
                            sub = arr[::f, ::f]
                            z = writers[name][kind][l]
                            y0l, x0l = ty0 // f, tx0 // f
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
            "heldout_tifxyz_boxes": held, "guard_boxes": guard,
            "placed_boxes": {name: place[name][2] for name in place},
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
