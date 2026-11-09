"""Before/after figure for experiment A (labelshift-w01): held-out ink on Paris4 w01, AS vs C0 checkpoints.

usage (from the A work tree, in A's venv):
  python -I render_beforeafter.py VILLA_SRC OUT_DIR [--seed 29] [--work .] [--boxes 5,17,19]
         [--box ID:y0,y1,x0,x1 ...] [--as-ckpt PATH] [--c0-ckpt PATH] [--ckpt-name ckpt_012000.pth]
         [--eval-config PATH] [--pad 32] [--batch 2] [--workers 8] [--reduce mean|max]
         [--panel-height 360] [--plan-only] [--allow-other-villa]

VILLA_SRC  villa's vesuvius/src at the commit A trained with (37d1897fb; checked unless --allow-other-villa)
OUT_DIR    created if missing; gets box<ID>_s<seed>.png per box, beforeafter_s<seed>.png (all boxes),
           beforeafter_s<seed>.npz (every 2D array) and provenance_s<seed>.json
--box      a level-0 canvas box [y0, y1, x0, x1] (the regions.json convention) under any ID; repeatable,
           added to or replacing --boxes ids taken from regions.json
--plan-only  CPU only, no checkpoint: labels, label cross-checks, patch plan, one patch per box through
           the data path (CT panel partly filled), label-only figure and provenance. Use it first.

What it shows, per box (columns): CT at the surface (the normalized model input read back onto the
surface), labels corrected for the HF65 offset (C0), labels as shipped (AS), ink probability of the AS
checkpoint, ink probability of the C0 checkpoint (same seed). All panels are on villa's flat grid at the
training volume_scale (flat pixel = 4 x 4 tifxyz cells for A). Cyan box: the held-out box A scored (C0
placement, padded by margin_px). Orange box: where the labels as shipped put the same component.

How it gets there (nothing about model, normalization or geometry is reimplemented):
  - config, dataset and forward exactly as scripts/eval_heldout.py: stage_training_request on the C0 eval
    config, training_dataset_config, villa's InkDataset(do_augmentations=False), make_model from the
    checkpoint's config, weights from select_inference_weights (EMA when present), run_model_forward,
    prepare_validation_loss_inputs, sigmoid, autocast as the config says.
  - patches: the corner lattice of villa's patch_finding_default.find_segment_patches (its own
    _scan_geometry and surface_patch_bbox) over a window holding both placements of each box, so the scored
    validation patches are a subset (checked and reported) and the area around the shipped placement is
    covered too. Each patch goes through InkDataset._native_sample directly (not __getitem__, which would
    silently swap an inadmissible patch for a random other one).
  - 3D -> 2D: each flat pixel reads the probability at the voxel(s) villa's label projection marks for it
    (figlib.project_to_flat; for A that is the single voxel int(position), since the label half thickness
    is 3 / 4 voxel). Overlapping patches are blended with villa's Gaussian importance weights (2D slice).
    The 3D labels of each sample are read back the same way and compared with the flat C0 labels
    (label_roundtrip in the provenance): a misaligned readout shows up there.
  - labels: the shipped label TIFF (data/raw/...) mapped with A's own build_labels.index_map and HF65
    model (scripts/hf65/apply_fit.py), identity for AS. Checked against the label zarrs the trainer and the
    two evaluations read (C0 everywhere; validation mask; EAS inside its boxes).
The 2D AUCs in the provenance are a sanity check of this readout, not A's scored metric (that one is over
3D voxels of the scored patches only; RESULTS.md).
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import figlib  # noqa: E402

EXPECTED_VILLA = "37d1897fb"
ARMS = ("AS", "C0")
CHANNELS = ("ct", "pred_AS", "pred_C0", "label3d")


def collate_list(items):
    return items


class FlatPatchSource:
    """Map-style dataset: one villa native sample per patch plus the flat geometry needed to read the
    prediction back onto the surface. Picklable for spawn workers (InkDataset already is)."""

    def __init__(self, inner, patches, stride, coord_scale, half_thickness):
        self.inner = inner
        self.patches = list(patches)
        self.stride = int(stride)
        self.coord_scale = float(coord_scale)
        self.half_thickness = float(half_thickness)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        from vesuvius.ink_detection.data.geometry import (
            compute_native_crop_bbox, read_tifxyz_on_flat_grid)
        p = self.patches[int(index)]
        _, y0, x0, _, y1, x1 = p.bbox
        out = {"i": int(index), "y0": int(y0), "x0": int(x0), "ok": False, "inadmissible": False, "error": None}
        try:
            sample = self.inner._native_sample(p)
            if sample is None:
                out["inadmissible"] = True  # villa's __getitem__ would resample another patch here
                return out
            surf = self.inner._tifxyz(p.segment)
            positions, valid = read_tifxyz_on_flat_grid(
                surf, y0=y0, y1=y1, x0=x0, x1=x1,
                flat_grid_stride=self.stride, native_coordinate_scale=self.coord_scale)
            # the same crop _native_sample cut (no augmentation, so no translation)
            crop = compute_native_crop_bbox(positions, valid, self.inner.patch_size)
            normals = None
            if any(s != 0 for s in figlib.normal_steps(self.half_thickness)):
                s = self.stride
                nx, ny, nz = surf.get_normals(y0 * s, y1 * s, x0 * s, x1 * s)
                normals = np.stack([nz, ny, nx], axis=-1)[::s, ::s].astype(np.float32, copy=False)
        except Exception as exc:  # reported per patch; the run fails if too many patches fail
            out["error"] = f"{type(exc).__name__}: {exc}"
            return out
        out.update(ok=True, sample=sample, positions=positions, valid=valid, normals=normals, crop=crop)
        return out


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("villa_src")
    p.add_argument("out_dir")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--work", default=".", help="A work tree (holds runs/, labels/, data/, scripts/, regions.json)")
    p.add_argument("--boxes", default=None, help="comma list of held-out ids from regions.json (default: all)")
    p.add_argument("--box", action="append", default=[], metavar="ID:y0,y1,x0,x1",
                   help="explicit level-0 canvas box; repeatable")
    p.add_argument("--regions", default=None, help="default WORK/regions.json")
    p.add_argument("--segment-key", default="w01")
    p.add_argument("--as-ckpt", default=None, help="default WORK/runs/AS_s<seed>/out/<ckpt-name>")
    p.add_argument("--c0-ckpt", default=None, help="default WORK/runs/C0_s<seed>/out/<ckpt-name>")
    p.add_argument("--ckpt-name", default="ckpt_012000.pth")
    p.add_argument("--eval-config", default=None, help="default WORK/runs/eval_C0_s<seed>/config.json")
    p.add_argument("--manifest", default=None, help="default WORK/labels/full/manifest.json")
    p.add_argument("--raw-root", default=None, help="default WORK/data/raw")
    p.add_argument("--scripts", default=None, help="A's scripts folder, default WORK/scripts")
    p.add_argument("--pad", type=int, default=32, help="flat px around the union of both placements")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=8, help="DataLoader workers (A's evals used 8)")
    p.add_argument("--reduce", choices=("mean", "max"), default="mean")
    p.add_argument("--panel-height", type=int, default=360, help="panel height in the combined PNG")
    p.add_argument("--max-failed-fraction", type=float, default=0.10)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--allow-other-villa", action="store_true")
    a = p.parse_args(argv)
    w = os.path.abspath(a.work)
    a.work = w
    a.regions = a.regions or os.path.join(w, "regions.json")
    a.as_ckpt = a.as_ckpt or os.path.join(w, "runs", f"AS_s{a.seed}", "out", a.ckpt_name)
    a.c0_ckpt = a.c0_ckpt or os.path.join(w, "runs", f"C0_s{a.seed}", "out", a.ckpt_name)
    a.eval_config = a.eval_config or os.path.join(w, "runs", f"eval_C0_s{a.seed}", "config.json")
    a.manifest = a.manifest or os.path.join(w, "labels", "full", "manifest.json")
    a.raw_root = a.raw_root or os.path.join(w, "data", "raw")
    a.scripts = a.scripts or os.path.join(w, "scripts")
    return a


def parse_box(text):
    bid, _, coords = text.partition(":")
    vals = [int(v) for v in coords.split(",")]
    if not bid or len(vals) != 4 or vals[0] >= vals[1] or vals[2] >= vals[3]:
        raise SystemExit(f"--box wants ID:y0,y1,x0,x1 with y0 < y1, x0 < x1; got {text!r}")
    return bid, vals


def villa_commit(villa_src):
    root = os.path.dirname(os.path.dirname(os.path.abspath(villa_src)))
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def read_tif_windows(path, windows, read_label_tif):
    """{key: canvas[r0:r1, c0:c1]} for windows {key: (r0, r1, c0, c1)}. Tries a tiled read through
    tifffile's zarr view; falls back to A's labellib.read_label_tif (whole canvas, a few GB)."""
    try:
        import tifffile
        import zarr
        store = tifffile.imread(path, aszarr=True)
        try:
            z = zarr.open(store, mode="r")
            if not hasattr(z, "shape"):
                z = z["0"]
            out = {k: np.ascontiguousarray(np.asarray(z[r0:r1, c0:c1])) for k, (r0, r1, c0, c1) in windows.items()}
        finally:
            store.close()
        method = "tifffile+zarr window"
    except Exception as exc:
        log(f"windowed TIFF read failed ({type(exc).__name__}: {exc}); reading the whole canvas")
        full = read_label_tif(path)
        out = {k: full[r0:r1, c0:c1].copy() for k, (r0, r1, c0, c1) in windows.items()}
        del full
        gc.collect()
        method = "labellib.read_label_tif (full)"
    for k, v in out.items():
        if v.ndim == 3:
            out[k] = v[..., 0]
    return out, method


def main(argv=None):
    a = parse_args(argv)
    t_start = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    out_dir = os.path.abspath(a.out_dir)
    sys.path.insert(0, os.path.abspath(a.villa_src))
    sys.path.insert(0, a.scripts)
    sys.path.insert(0, os.path.join(a.scripts, "hf65"))

    commit = villa_commit(a.villa_src)
    log(f"villa {commit}")
    if not (commit or "").startswith(EXPECTED_VILLA) and not a.allow_other_villa:
        raise SystemExit(f"villa at {a.villa_src} is {commit}, not {EXPECTED_VILLA} (A's trainer); "
                         "pass --allow-other-villa to render anyway")

    # --- A's label geometry ----------------------------------------------------------------------
    import apply_fit
    from build_labels import box_to_tifxyz, index_map
    from labellib import read_label_tif

    R = json.load(open(a.regions))
    S = R["segments"][a.segment_key]
    manifest = json.load(open(a.manifest))
    M = manifest["segments"][a.segment_key]
    H, W = (int(v) for v in M["tifxyz_full_shape"])
    Hc, Wc = (int(v) for v in M["canvas_shape"])
    margin = int(R["margin_px"])
    fit = apply_fit.model(S["fit_key"])
    assert fit.separable, "A's builder assumes dy(y), dx(x)"
    boxes = {}
    ids = list(S["heldout"]) if a.boxes is None else [b.strip() for b in a.boxes.split(",") if b.strip()]
    for bid in ids:
        if bid not in S["heldout"]:
            raise SystemExit(f"box {bid} is not a held-out id in {a.regions} ({list(S['heldout'])})")
        boxes[bid] = [int(v) for v in S["heldout"][bid]]
    for text in a.box:
        bid, vals = parse_box(text)
        boxes[bid] = vals
    if not boxes:
        raise SystemExit("no boxes")
    manifest_held = {bid: M["heldout_tifxyz_boxes"][i] for i, bid in enumerate(S["heldout"])}
    geo = {}
    for bid, cb in boxes.items():
        held = [int(v) for v in box_to_tifxyz(fit, cb, margin, (H, W))]
        if bid in S["heldout"] and cb == [int(v) for v in S["heldout"][bid]]:
            assert held == [int(v) for v in manifest_held[bid]], (bid, held, manifest_held[bid])
        y0, y1, x0, x1 = cb
        shipped = [max(0, y0 - margin), min(H, y1 + margin), max(0, x0 - margin), min(W, x1 + margin)]
        geo[bid] = {"canvas_box": cb, "scored_box_tifxyz": held, "shipped_box_tifxyz": shipped}
    log(f"boxes: {json.dumps(geo)}")

    # --- villa: config and dataset exactly as eval_heldout.py, patch cache kept out of A's run tree -----
    from vesuvius.ink_detection.data.dataset import InkDataset, _read_flat_surface
    from vesuvius.ink_detection.data.geometry import native_tifxyz_pyramid_params
    from vesuvius.ink_detection.data.patch_finding_default import _scan_geometry, surface_patch_bbox
    from vesuvius.ink_detection.training.train import stage_training_request, training_dataset_config
    from vesuvius.ink_detection.types import Patch

    cfg_json = json.load(open(a.eval_config))
    orig_out = cfg_json["out_dir"]
    ds_out = os.path.join(out_dir, "dataset_out")
    os.makedirs(ds_out, exist_ok=True)
    copied = []
    for f in glob.glob(os.path.join(orig_out, "flat_ink_patches_*.json")):
        shutil.copy2(f, ds_out)
        copied.append(os.path.basename(f))
    cfg_json["out_dir"] = ds_out
    cfg_copy = os.path.join(out_dir, f"eval_config_C0_s{a.seed}.json")
    with open(cfg_copy, "w") as f:
        json.dump(cfg_json, f, indent=2)
    config = stage_training_request(cfg_copy).config
    data_config = training_dataset_config(config)
    t0 = time.time()
    shared = InkDataset(data_config, do_augmentations=False)
    val = list(shared.validation_patches)
    log(f"InkDataset: {len(shared.patches)} patches, {len(val)} validation, {time.time() - t0:.1f}s "
        f"(patch cache copied: {copied or 'none'})")
    if not val:
        raise SystemExit("no validation patches: is this the C0 eval config?")
    seg = val[0].segment
    stride, coord_scale, _ = native_tifxyz_pyramid_params(seg.scale)
    f = int(stride)
    flat_hw = (-(-H // f), -(-W // f))
    ps = tuple(int(v) for v in shared.patch_size)
    label_ht = float(shared._normal_thicknesses(seg.scale)[0])
    scan_scale, _, _, surface, scale_y, scale_x = _scan_geometry(seg, shared._open)
    overlap = data_config.patch_finding.overlap
    stride_scan = max(1, int(round(int(ps[1] * overlap) / scale_y)))
    log(f"flat factor {f}, patch {ps}, label half thickness {label_ht} voxel -> steps {figlib.normal_steps(label_ht)}, "
        f"scan scale {scan_scale} (x{scale_y:.4f}, x{scale_x:.4f}), corner stride {stride_scan} scan px, surface z {surface}")

    # --- per box: window, labels, cross-checks, patches ------------------------------------------
    eas_inklabels = str(seg.inklabels).replace("/labels/full/C0/", "/labels/full_eval/EAS/")
    c0_rows_all = index_map(fit.inv(np.arange(H, dtype=float), np.zeros(H))[0], H, Hc, 0)
    c0_cols_all = index_map(fit.inv(np.zeros(W), np.arange(W, dtype=float))[1], W, Wc, 0)
    win_idx = {}
    for bid, g in geo.items():
        win = figlib.flat_window([g["scored_box_tifxyz"], g["shipped_box_tifxyz"]], f, a.pad, flat_hw)
        g["flat_window"] = list(win)
        ty = np.arange(win[0], win[1]) * f
        tx = np.arange(win[2], win[3]) * f
        as_r = np.where(ty < Hc, ty, -1)
        as_c = np.where(tx < Wc, tx, -1)
        win_idx[bid] = (c0_rows_all[ty], c0_cols_all[tx], as_r, as_c)
    windows = {}
    for bid, (c0r, c0c, asr, asc) in win_idx.items():
        r0, r1 = figlib.span(c0r, asr)
        c0_, c1 = figlib.span(c0c, asc)
        windows[bid] = (r0, r1, c0_, c1)
    raw = os.path.join(a.raw_root, S.get("bucket_scroll", S["scroll"]), S["segment"])
    ink_win, ink_method = read_tif_windows(os.path.join(raw, S["ink_tif"]), windows, read_label_tif)
    sup_win, sup_method = read_tif_windows(os.path.join(raw, S["sup_tif"]), windows, read_label_tif)
    arrays = {}
    checks = {}
    for bid, g in geo.items():
        c0r, c0c, asr, asc = win_idx[bid]
        r0, _, c0_, _ = windows[bid]
        lab_c0, known_c0 = figlib.gather_window(ink_win[bid], r0, c0_, c0r, c0c)
        sup_c0, _ = figlib.gather_window(sup_win[bid], r0, c0_, c0r, c0c)
        lab_as, known_as = figlib.gather_window(ink_win[bid], r0, c0_, asr, asc)
        sup_as, _ = figlib.gather_window(sup_win[bid], r0, c0_, asr, asc)
        win = g["flat_window"]
        scored = figlib.box_in_window(g["scored_box_tifxyz"], win, f)
        shipped = figlib.box_in_window(g["shipped_box_tifxyz"], win, f)
        held_m = np.zeros(lab_c0.shape, bool)
        hy0, hy1, hx0, hx1 = g["scored_box_tifxyz"]
        Ys = np.arange(win[0], win[1]) * f
        Xs = np.arange(win[2], win[3]) * f
        held_m[np.ix_((Ys >= hy0) & (Ys < hy1), (Xs >= hx0) & (Xs < hx1))] = True
        # what the trainer and the evaluations read at this level
        zc0 = _read_flat_surface(shared._open(seg.inklabels, seg.scale), y0=win[0], y1=win[1], x0=win[2], x1=win[3])
        zval = _read_flat_surface(shared._open(seg.validation_mask, seg.scale), y0=win[0], y1=win[1], x0=win[2], x1=win[3])
        ck = {"c0_labels_vs_C0_zarr_agree": float(np.mean((np.asarray(zc0) > 0) == (lab_c0 > 0))),
              "validation_mask_vs_C0_sup_in_scored_box_agree": float(np.mean((np.asarray(zval) > 0) == ((sup_c0 > 0) & held_m))),
              "validation_px": int(np.count_nonzero(zval))}
        if os.path.exists(eas_inklabels):
            zeas = np.asarray(_read_flat_surface(shared._open(eas_inklabels, seg.scale), y0=win[0], y1=win[1], x0=win[2], x1=win[3]))
            sr = tuple(slice(max(0, lo), max(0, hi)) for lo, hi in ((shipped[0], shipped[1]), (shipped[2], shipped[3])))
            ck["as_labels_vs_EAS_zarr_in_shipped_box_agree"] = float(np.mean((zeas[sr] > 0) == (lab_as[sr] > 0)))
        else:
            ck["as_labels_vs_EAS_zarr_in_shipped_box_agree"] = None
        checks[bid] = ck
        log(f"box {bid}: window {win} ({win[1] - win[0]} x {win[3] - win[2]} flat px), checks {ck}")
        for name, v in ck.items():
            if isinstance(v, float) and v < 0.999:
                log(f"WARNING box {bid}: {name} = {v:.4f} (expected 1.0): label orientation or mapping differs from A's zarrs")
        arrays[bid] = {"label_c0": lab_c0, "sup_c0": sup_c0, "known_c0": known_c0,
                       "label_as": lab_as, "sup_as": sup_as, "known_as": known_as,
                       "validation_mask": (np.asarray(zval) > 0)}
        g["scored_rect_flat"] = list(scored)
        g["shipped_rect_flat"] = list(shipped)
    del ink_win, sup_win

    # patch lattice over every window; the scored validation patches must lie on it
    by_bbox = {}
    for bid, g in geo.items():
        y0w, y1w, x0w, x1w = g["flat_window"]
        for y0 in figlib.lattice_starts(y0w, y1w, ps[1], stride_scan, scale_y):
            for x0 in figlib.lattice_starts(x0w, x1w, ps[2], stride_scan, scale_x):
                bbox = surface_patch_bbox(surface, y0, x0, ps)
                by_bbox.setdefault(bbox, Patch(segment=seg, bbox=bbox, is_validation=True,
                                               supervision_mask_override=seg.validation_mask))
    in_view = [p for p in val if any(
        p.bbox[1] < g["flat_window"][1] and p.bbox[4] > g["flat_window"][0] and
        p.bbox[2] < g["flat_window"][3] and p.bbox[5] > g["flat_window"][2] for g in geo.values())]
    off_lattice = [p.bbox for p in in_view if p.bbox not in by_bbox]
    for p in in_view:
        by_bbox[p.bbox] = p
    patches = [by_bbox[k] for k in sorted(by_bbox)]
    plan = {"patches": len(patches), "validation_patches_total": len(val),
            "validation_patches_in_windows": len(in_view), "validation_patches_off_lattice": len(off_lattice),
            "corner_stride_scan_px": stride_scan, "scan_scale": scan_scale,
            "scale_y": scale_y, "scale_x": scale_x, "surface_z": int(surface)}
    log(f"plan: {plan}")
    if off_lattice:
        log(f"WARNING {len(off_lattice)} scored patches are off the reconstructed lattice (added anyway), e.g. {off_lattice[:3]}")

    source = FlatPatchSource(shared, patches, stride, coord_scale, label_ht)
    accs = {bid: figlib.FlatAccumulator(g["flat_window"], CHANNELS) for bid, g in geo.items()}
    weight = figlib.gaussian_weight_2d(ps[1], ps[2])
    roundtrip = {bid: [0, 0] for bid in geo}
    failures = []
    inadmissible = []

    def consume(item, vols):
        """Read the 3D volumes of one sample back onto the flat grid and add them to every window."""
        crop0 = item["crop"][:3]
        tiles = {}
        for name, vol in vols.items():
            red = "max" if name == "label3d" else a.reduce
            tiles[name] = figlib.project_to_flat(vol, crop0, item["positions"], item["valid"],
                                                 item["normals"], label_ht, reduce=red)
        y0, x0 = item["y0"], item["x0"]
        h, w = tiles["label3d"].shape
        for bid, acc in accs.items():
            if not acc.overlaps(y0, x0, h, w):
                continue
            for name, t in tiles.items():
                acc.add(name, y0, x0, t, weight)
            a0, b0 = max(y0, acc.y0), min(y0 + h, acc.y1)
            c0_, d0 = max(x0, acc.x0), min(x0 + w, acc.x1)
            lab = arrays[bid]["label_c0"][a0 - acc.y0:b0 - acc.y0, c0_ - acc.x0:d0 - acc.x0] > 0
            t = tiles["label3d"][a0 - y0:b0 - y0, c0_ - x0:d0 - x0]
            sel = lab & np.isfinite(t)
            roundtrip[bid][0] += int(sel.sum())
            roundtrip[bid][1] += int((t[sel] >= 0.5).sum())

    models = {}
    weights_key = {}
    ckpt_info = {}
    device_name = "none"
    if a.plan_only:
        # Per box, try patches from the window centre outwards (up to 12) until one is admissible, so the
        # read path and the label roundtrip run on a real patch; corner patches are often inadmissible.
        for bid, g in geo.items():
            wy0, wy1, wx0, wx1 = g["flat_window"]
            cy, cx = (wy0 + wy1) / 2, (wx0 + wx1) / 2
            cand = [i for i, p in enumerate(patches)
                    if p.bbox[1] < wy1 and p.bbox[4] > wy0 and p.bbox[2] < wx1 and p.bbox[5] > wx0]
            cand.sort(key=lambda i: ((patches[i].bbox[1] + patches[i].bbox[4]) / 2 - cy) ** 2
                      + ((patches[i].bbox[2] + patches[i].bbox[5]) / 2 - cx) ** 2)
            for tried, i in enumerate(cand[:12], start=1):
                t1 = time.time()
                item = source[i]
                if not item["ok"]:
                    log(f"plan-only box {bid}: patch {patches[i].bbox} not usable: {item['error'] or 'inadmissible'}")
                    if item["inadmissible"]:
                        inadmissible.append(patches[i].bbox)
                    else:
                        failures.append((patches[i].bbox, item["error"]))
                    continue
                s = item["sample"]
                consume(item, {"ct": s["image"][0].numpy(), "label3d": s["inklabels"][0].numpy()})
                log(f"plan-only box {bid}: patch {patches[i].bbox} crop {item['crop']} read in {time.time() - t1:.1f}s; "
                    f"image {tuple(s['image'].shape)}, valid flat px {int(item['valid'].sum())}; "
                    f"try {tried} of {len(cand)} patches overlapping the window")
                break
    else:
        import torch
        from torch.utils.data import DataLoader
        from vesuvius.ink_detection.models.checkpoint import (
            config_from_checkpoint, load_checkpoint, load_model_state, select_inference_weights)
        from vesuvius.ink_detection.models.model import make_model
        from vesuvius.ink_detection.training.stitching import run_model_forward
        from vesuvius.ink_detection.training.train import prepare_model_input, prepare_validation_loss_inputs

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        if device.type != "cuda":
            log("WARNING no CUDA device: running on CPU (slow)")
        for arm, path in (("AS", a.as_ckpt), ("C0", a.c0_ckpt)):
            t1 = time.time()
            ckpt = load_checkpoint(path)
            model = make_model(config_from_checkpoint(ckpt, source=path))
            name, state = select_inference_weights(ckpt, source=path)
            load_model_state(model, state)
            model.to(device).eval()
            models[arm] = model
            weights_key[arm] = name
            st = os.stat(path)
            ckpt_info[arm] = {"path": os.path.abspath(path), "bytes": st.st_size,
                              "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st.st_mtime)),
                              "weights": name, "top_level_keys": sorted(str(k) for k in ckpt),
                              "step": next((int(ckpt[k]) for k in ("step", "iteration", "global_step")
                                            if isinstance(ckpt.get(k), (int, np.integer))), None)}
            del ckpt, state
            gc.collect()
            log(f"{arm}: {path} weights={name} loaded in {time.time() - t1:.1f}s")
        if config.ink.model.input_pad_depth_to is not None:
            log(f"WARNING input_pad_depth_to={config.ink.model.input_pad_depth_to}: depth-padded predictions are "
                "resized to the crop by prepare_validation_loss_inputs, as in eval_heldout.py")
        amp = config.mixed_precision in ("bf16", "fp16") and device.type == "cuda"
        amp_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        mode = config.ink.data.mode
        loader = DataLoader(source, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                            collate_fn=collate_list,
                            **({"multiprocessing_context": "spawn", "persistent_workers": False} if a.workers else {}))
        t1 = time.time()
        done = 0
        with torch.no_grad():
            for items in loader:
                good = [it for it in items if it["ok"]]
                failures.extend((patches[it["i"]].bbox, it["error"]) for it in items if it["error"])
                inadmissible.extend(patches[it["i"]].bbox for it in items if it["inadmissible"])
                done += len(items)
                if good:
                    keys = [k for k, v in good[0]["sample"].items() if torch.is_tensor(v)]
                    batch = {k: torch.stack([it["sample"][k] for it in good]).to(device) for k in keys}
                    probs = {}
                    for arm, model in models.items():
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                            pred = run_model_forward(model, prepare_model_input(batch, config), config.model_crop_size,
                                                     stitched=config.use_stitched_forward,
                                                     use_gradient_checkpointing=config.stitched_gradient_checkpointing)
                        pred, _, _ = prepare_validation_loss_inputs(pred, batch, mode=mode)
                        if isinstance(pred, (list, tuple)):
                            pred = pred[0]
                        probs[arm] = torch.sigmoid(pred.float())[:, 0].cpu().numpy()
                    images = batch["image"][:, 0].float().cpu().numpy()
                    labels3d = batch["inklabels"][:, 0].float().cpu().numpy()
                    for j, it in enumerate(good):
                        consume(it, {"ct": images[j], "pred_AS": probs["AS"][j], "pred_C0": probs["C0"][j],
                                     "label3d": labels3d[j]})
                if done % 50 < a.batch or done == len(source):
                    el = time.time() - t1
                    log(f"{done}/{len(source)} patches, {len(failures)} failed, {len(inadmissible)} inadmissible, {el:.0f}s "
                        f"(~{el / max(done, 1) * (len(source) - done):.0f}s left)")
        if device.type == "cuda":
            log(f"peak CUDA memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    # --- outputs -----------------------------------------------------------------------------------
    for bid, acc in accs.items():
        for name in CHANNELS:
            arrays[bid][name] = acc.result(name)
        arrays[bid]["weight"] = acc.wsum["label3d"].astype(np.float32)
    sanity = {}
    for bid, A in arrays.items():
        m = A["validation_mask"]
        s = {"label_roundtrip_ink_px": roundtrip[bid][0],
             "label_roundtrip_hit_fraction": (roundtrip[bid][1] / roundtrip[bid][0]) if roundtrip[bid][0] else None}
        for arm in ARMS:
            pr = A[f"pred_{arm}"]
            sel = m & np.isfinite(pr)
            s[f"auc2d_{arm}_vs_C0_labels_in_validation_mask"] = figlib.rank_auc(pr[sel], A["label_c0"][sel] > 0)
            s[f"validation_px_with_prediction_{arm}"] = int(sel.sum())
        sanity[bid] = s
        log(f"box {bid}: {s}")
        if s["label_roundtrip_hit_fraction"] is not None and s["label_roundtrip_hit_fraction"] < 0.95:
            log(f"WARNING box {bid}: only {s['label_roundtrip_hit_fraction']:.3f} of flat C0 ink pixels read back as ink "
                "from villa's 3D labels; the 3D -> 2D readout may be misaligned")

    tag = f"s{a.seed}"
    titles = ["CT at surface (model input)", "labels corrected (C0)", "labels as shipped (AS)",
              f"AS model s{a.seed}: ink prob.", f"C0 model s{a.seed}: ink prob."]
    rows, headers = [], []
    for bid, A in arrays.items():
        g = geo[bid]
        panels = [figlib.render_gray(A["ct"]),
                  figlib.render_labels(A["label_c0"], A["sup_c0"], A["known_c0"]),
                  figlib.render_labels(A["label_as"], A["sup_as"], A["known_as"]),
                  figlib.render_gray(A["pred_AS"], 0.0, 1.0),
                  figlib.render_gray(A["pred_C0"], 0.0, 1.0)]
        for pnl in panels:
            figlib.draw_rect(pnl, g["shipped_rect_flat"], figlib.SHIPPED_RGB)
            figlib.draw_rect(pnl, g["scored_rect_flat"], figlib.SCORED_RGB)
        s = sanity[bid]
        auc_txt = ", ".join(f"{arm} {s[f'auc2d_{arm}_vs_C0_labels_in_validation_mask']:.3f}"
                            for arm in ARMS if s[f"auc2d_{arm}_vs_C0_labels_in_validation_mask"] is not None)
        header = (f"held-out box {bid}: canvas {g['canvas_box']}, flat window {g['flat_window']} (1 px = {f} tifxyz cells)"
                  + (f"; 2D AUC in scored area: {auc_txt}" if auc_txt else ""))
        row = figlib.compose_row(panels, titles)
        figlib.save_png(os.path.join(out_dir, f"box{bid}_{tag}.png"),
                        figlib.stack_rows([row], [header], footer=legend_lines(a)))
        rows.append(figlib.compose_row([figlib.resize_to_height(p, a.panel_height) for p in panels], titles))
        headers.append(header)
    combined = os.path.join(out_dir, f"beforeafter_{tag}.png")
    figlib.save_png(combined, figlib.stack_rows(rows, headers, footer=legend_lines(a)))
    npz = {"flat_factor": np.int64(f)}
    for bid, A in arrays.items():
        for k, v in A.items():
            npz[f"b{bid}_{k}"] = np.asarray(v)
        npz[f"b{bid}_flat_window"] = np.asarray(geo[bid]["flat_window"], np.int64)
        npz[f"b{bid}_scored_rect_flat"] = np.asarray(geo[bid]["scored_rect_flat"], np.int64)
        npz[f"b{bid}_shipped_rect_flat"] = np.asarray(geo[bid]["shipped_rect_flat"], np.int64)
    np.savez_compressed(os.path.join(out_dir, f"beforeafter_{tag}.npz"), **npz)
    n_failed = len(failures)
    prov = {
        "script": os.path.abspath(__file__), "argv": sys.argv, "plan_only": a.plan_only,
        "seed": a.seed, "villa_src": os.path.abspath(a.villa_src), "villa_commit": commit,
        "checkpoints": ckpt_info if ckpt_info else {"AS": os.path.abspath(a.as_ckpt), "C0": os.path.abspath(a.c0_ckpt)},
        "weights_key": weights_key, "device": device_name,
        "eval_config": os.path.abspath(a.eval_config), "eval_config_copy": cfg_copy, "patch_cache_copied": copied,
        "mixed_precision": config.mixed_precision, "mode": config.ink.data.mode,
        "regions": os.path.abspath(a.regions), "manifest": os.path.abspath(a.manifest), "margin_px": margin,
        "hf65_model": fit.name, "tifxyz_full_shape": [H, W], "canvas_shape": [Hc, Wc],
        "label_tif_read": {"inklabels": ink_method, "supervision": sup_method},
        "flat_factor": f, "patch_size": list(ps), "label_half_thickness_voxels": label_ht,
        "normal_steps": figlib.normal_steps(label_ht), "reduce": a.reduce, "pad_flat_px": a.pad,
        "boxes": geo, "label_checks": checks, "plan": plan, "sanity_2d": sanity,
        "failed_patches": n_failed, "failed_examples": [[list(b), e] for b, e in failures[:20]],
        "inadmissible_patches": len(inadmissible), "inadmissible_examples": [list(b) for b in inadmissible[:20]],
        "seconds": round(time.time() - t_start, 1),
        "note": "auc2d_* is a readout sanity check on the flat grid, not A's scored 3D-voxel metric (RESULTS.md)",
    }
    with open(os.path.join(out_dir, f"provenance_{tag}.json"), "w") as fh:
        json.dump(prov, fh, indent=1, default=str)
    log(f"wrote {combined} and box/npz/provenance files in {out_dir} ({time.time() - t_start:.0f}s)")
    if not a.plan_only and patches and n_failed > a.max_failed_fraction * len(patches):
        log(f"FAILED: {n_failed} of {len(patches)} patches failed (> {a.max_failed_fraction:.0%}); see provenance")
        return 3
    return 0


def legend_lines(a):
    return [f"cyan: held-out box A scored (C0 placement, margin included); orange: same component where the shipped labels put it",
            f"labels: white ink, black supervised background, grey unsupervised, dark blue outside the canvas or no prediction",
            f"probabilities 0 (black) to 1 (white), EMA weights at step 12000; 3D->2D readout: {a.reduce} over villa's label-projection voxels"]


if __name__ == "__main__":
    sys.exit(main())
