"""--label-canvas-offset values for `transfer single` (source = target = the segment's tifxyz)
for the 26 rows of ../shifts.csv.

Inputs: ../shifts.csv (the one-to-one-frame model), ../../../audit/results/phase1.json (stored grid
size from the x.tif header, zarr shapes), meta/*.json (meta.json fetched live by fetch_meta.py). Sizes
the tool derives (full_extent, output shape) come from the tool's own code: vc_reader.tifxyz.full_extent
and tifxyz_label_transfer.core.infer_output_shape, loaded from a villa source tree unchanged.

Per axis, with L the label size, S the stored size, s the meta.json scale for that axis and d the
model offset at label pixel (y, x):

  tool frame:   o(y, x) = (s * L / S) * (y + d(y, x)) - y - 0.5      [label px]
  flag value  = o * full_extent(S, s) / L                              [full-res px]

The field is affine in (y, x), so over the canvas its extremes sit at the four corners; the
constant with the smallest worst-case error is the midpoint of the corner range, and the worst
gap is half the range. A dense grid evaluation and a forward pass through the kernel equations
check that.

usage: python -I table.py VILLA_SRC [--vc-reader VC_READER] [--vc-reader-build DIR] [--out OUT_DIR]

VILLA_SRC is villa's vesuvius/src. VC_READER is villa's vc_reader folder with its scikit-build build
(see villa_boot.py); without it an installed vc_reader is used. Writes offsets_transfer_frame.csv
and table_checks.json to OUT_DIR, which defaults to this folder.
"""
import argparse
import csv
import json
import math
import os
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import villa_boot  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("villa_src")
parser.add_argument("--vc-reader")
parser.add_argument("--vc-reader-build")
parser.add_argument("--out", default=HERE)
args = parser.parse_args()
villa_boot.boot(args.villa_src, args.vc_reader, args.vc_reader_build)
import numpy as np  # noqa: E402
from vc_reader.tifxyz import full_extent  # noqa: E402
from vesuvius.tifxyz_label_transfer.core import infer_output_shape  # noqa: E402

ROWS = list(csv.DictReader(open(os.path.join(REPO, "correct", "shifts", "shifts.csv"), newline="")))
PHASE1 = {(e["scroll"], e["segment"]): e
          for e in json.load(open(os.path.join(REPO, "audit", "results", "phase1.json")))}


def hw(text):
    a, b = text.split("x")
    return int(a), int(b)


def model_axes(r):
    cy = (float(r["dy_a"]), float(r["dy_per_y"]), float(r["dy_per_x"]))
    cx = (float(r["dx_a"]), float(r["dx_per_y"]), float(r["dx_per_x"]))
    return cy, cx


def d_at(c, y, x):
    return c[0] + c[1] * y + c[2] * x


def tool_field(c, axis, L, S, s, y, x):
    """o(y, x) in label px for one axis; axis 0 uses y as the own coordinate."""
    own = y if axis == 0 else x
    return (s * L[axis] / S[axis]) * (own + d_at(c, y, x)) - own - 0.5


def loader_field(c, axis, L, O_out, y, x):
    """Offset that puts content belonging at loader cell i into tool output pixel i."""
    own = y if axis == 0 else x
    return (own + d_at(c, y, x) + 0.5) * L[axis] / O_out[axis] - own - 0.5


def corners(L):
    ys = np.array([0.0, 0.0, L[0] - 1.0, L[0] - 1.0])
    xs = np.array([0.0, L[1] - 1.0, 0.0, L[1] - 1.0])
    return ys, xs


def minimax(field_fn, L):
    ys, xs = corners(L)
    v = field_fn(ys, xs)
    lo, hi = float(v.min()), float(v.max())
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def dense_gap(field_fn, L, const):
    """Independent check of the corner argument: max |field - const| on a dense grid with edges."""
    ys = np.unique(np.r_[np.arange(0, L[0], 97, dtype=np.float64), L[0] - 1.0])
    xs = np.unique(np.r_[np.arange(0, L[1], 97, dtype=np.float64), L[1] - 1.0])
    v = field_fn(ys[:, None], xs[None, :])
    return float(np.abs(v - const).max())


def kernel_gap(c, axis, L, S, s, O_out, o_label, other_mid):
    """Forward pass through the kernel equations over every output pixel along one axis.

    Output pixel i sits at stored p = (i + 0.5) * S / O_out (vc_reader pixelCenter), the tool reads
    label coordinate u = p * L / S - o_label - 0.5 (pixel centres at integers), and the model says
    the label pixel that belongs at stored p is y* with (y* + d(y*)) * s = p. Returns the largest
    |u - y*| over pixels whose y* lies on the label canvas, with the other coordinate fixed at
    other_mid (a separable model makes this exact; for a cross term the corners already bound it).
    """
    i = np.arange(O_out[axis], dtype=np.float64)
    p = (i + 0.5) * S[axis] / O_out[axis]
    u = p * L[axis] / S[axis] - o_label - 0.5
    t = p / s
    own_slope = c[1] if axis == 0 else c[2]
    other_term = (c[2] * other_mid) if axis == 0 else (c[1] * other_mid)
    ystar = (t - c[0] - other_term) / (1.0 + own_slope)
    on = (ystar >= 0) & (ystar <= L[axis] - 1)
    return float(np.abs(u - ystar)[on].max()), int(on.sum())


PIXEL_NOTES_PATH = os.path.join(HERE, "pixel_notes.json")
PIXEL_NOTES = json.load(open(PIXEL_NOTES_PATH)) if os.path.exists(PIXEL_NOTES_PATH) else {}

out_rows = []
checks = []
for r in ROWS:
    key = r["key"]
    e = PHASE1[(r["scroll"], r["segment"])]
    meta = json.load(open(os.path.join(HERE, "meta", f"{r['scroll']}__{r['segment']}.json")))
    sx, sy = (float(v) for v in meta["scale"])          # meta.json order is x, y
    s = (sy, sx)
    S = tuple(int(v) for v in e["stored_hw"])
    L = tuple(int(v) for v in e["canvas_hw"])
    loader_label = e["loader_pick"]["inklabels"]
    label_shape = tuple(e["zarrs"][loader_label]["shape"][-2:])
    O_full = (full_extent(S[0], sy), full_extent(S[1], sx))
    O_out = infer_output_shape(SimpleNamespace(shape=S, scale_yx=s), L, SimpleNamespace(shape=S, scale_yx=s))
    cy, cx = model_axes(r)
    coeffs = (cy, cx)
    consistency = {
        "label_zarr_hw_equals_canvas": label_shape == L,
        "shifts_csv_canvas_equals_phase1": hw(r["canvas_hw"]) == L,
        "shifts_csv_full_equals_full_extent": hw(r["tifxyz_full_hw"]) == O_full,
        "phase1_full_equals_full_extent": tuple(e["full_hw"]) == O_full,
    }
    rec = {"key": key, "consistency": consistency}
    flag, gap_full, loader_flag, loader_gap_full = [], [], [], []
    for axis in (0, 1):
        c = coeffs[axis]
        f_tool = lambda y, x, c=c, axis=axis: tool_field(c, axis, L, S, s[axis], y, x)  # noqa: E731
        const, gap = minimax(f_tool, L)
        dg = dense_gap(f_tool, L, const)
        other_mid = (L[1] - 1) / 2 if axis == 0 else (L[0] - 1) / 2
        kg, n_on = kernel_gap(c, axis, L, S, s[axis], O_out, const, other_mid)
        centre = float(f_tool(np.array((L[0] - 1) / 2), np.array((L[1] - 1) / 2)))
        f_load = lambda y, x, c=c, axis=axis: loader_field(c, axis, L, O_out, y, x)  # noqa: E731
        lconst, lgap = minimax(f_load, L)
        to_full = O_full[axis] / L[axis]
        flag.append(const * to_full)
        gap_full.append(gap * to_full)
        loader_flag.append(lconst * to_full)
        loader_gap_full.append(lgap * to_full)
        rec[("dy", "dx")[axis]] = {
            "label_px_const": const, "label_px_gap": gap, "dense_gap": dg,
            "kernel_gap": kg, "kernel_pixels_on_canvas": n_on, "centre_value": centre,
            "corner_minus_dense": gap - dg,
        }
    checks.append(rec)
    max_gap = max(gap_full)
    out_rows.append({
        "key": key,
        "scroll": r["scroll"],
        "segment": r["segment"],
        "group": r["group"],
        "model": r["model"] if r["slope_source"] == r["model"] else f"{r['model']}; {r['slope_source']}",
        "label_hw": f"{L[0]}x{L[1]}",
        "stored_hw": f"{S[0]}x{S[1]}",
        "scale_y": repr(sy),
        "scale_x": repr(sx),
        "full_extent_hw": f"{O_full[0]}x{O_full[1]}",
        "tool_output_hw": f"{O_out[0]}x{O_out[1]}",
        "DY": f"{flag[0]:.4f}",
        "DX": f"{flag[1]:.4f}",
        "chosen_as": "midpoint of the tool-frame field over the canvas corners (same as its value at the canvas centre)",
        "gap_dy_px": f"{gap_full[0]:.3f}",
        "gap_dx_px": f"{gap_full[1]:.3f}",
        "gap_over_1px": "yes" if max_gap > 1.0 else "no",
        "loader_DY": f"{loader_flag[0]:.4f}",
        "loader_DX": f"{loader_flag[1]:.4f}",
        "loader_gap_dy_px": f"{loader_gap_full[0]:.3f}",
        "loader_gap_dx_px": f"{loader_gap_full[1]:.3f}",
        "checked_on_pixels": PIXEL_NOTES.get(key, "no: metadata and fit only"),
        "inputs_consistent": "yes" if all(consistency.values()) else "NO: " + ",".join(k for k, v in consistency.items() if not v),
    })

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "offsets_transfer_frame.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(out_rows)
with open(os.path.join(args.out, "table_checks.json"), "w", newline="\n") as f:
    json.dump(checks, f, indent=1)

worst_dense = max(abs(c[a]["corner_minus_dense"]) for c in checks for a in ("dy", "dx"))
worst_kernel = max(c[a]["kernel_gap"] - c[a]["label_px_gap"] for c in checks for a in ("dy", "dx"))
worst_centre = max(abs(c[a]["centre_value"] - c[a]["label_px_const"]) for c in checks for a in ("dy", "dx"))
print("corner gap minus dense-grid gap, worst |diff|:", worst_dense)
print("kernel forward-pass gap minus corner gap, worst:", worst_kernel)
print("centre value minus midpoint, worst |diff|:", worst_centre)
for row in out_rows:
    print(row["key"], row["label_hw"], row["full_extent_hw"], row["tool_output_hw"], row["DY"], row["DX"],
          row["gap_dy_px"], row["gap_dx_px"], row["gap_over_1px"], row["loader_DY"], row["loader_DX"],
          row["loader_gap_dy_px"], row["loader_gap_dx_px"], row["inputs_consistent"])
