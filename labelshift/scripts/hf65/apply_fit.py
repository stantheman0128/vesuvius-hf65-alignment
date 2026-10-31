"""Correction for the HF65 canvas vs tifxyz offset reported in ScrollPrize/villa#2370.

Coordinates
  canvas (y, x)   level-0 pixel index of the HF65 canvas in scrollprize/datasets ink/<scroll>/<segment>/
                  (the ink labels and the supervision mask share this grid); pixel centres at integers.
  tifxyz (ty, tx) full-resolution cell of the tifxyz in the same folder, villa convention
                  (full-res cell (r, c) = stored cell (r*sy, c*sx), bilinear; vesuvius tifxyz/types.py:153-154).

Model (sign as in the README): canvas content at (y, x) lies at tifxyz cell (y + dy, x + dx), with
  dy = a_y + b_y*y + c_y*x,  dx = a_x + b_x*y + c_x*x.
The README table ("readme") prints per axis only the dominant slope, so b_x = c_y = 0 there; the
intercepts are the ones of the full 3-term region-centre fit made by measure/summarize.py.
`refit()` recomputes the full 3-term fit ("affine3") from a results directory.

Usage (python -I correct/apply_fit.py ...):
  table                                   print the fit table
  fwd SEG y x                             canvas -> tifxyz
  inv SEG ty tx                           tifxyz -> canvas (fixed-point iteration)
  check [RESULTS_DIR]                     cross-check the table against results_<key>.jsonl,
                                          and the inversion error over each segment's extent
  resample SEG IN OUT --origin Y0 X0 [--level K] [--pyr-shift PY PX] [--order 0|1]
           [--out-origin TY0 TX0] [--shape H W] [--fill V] [--model readme|affine3 --results DIR]
      IN (.npy, .png, .tif) is a 2D canvas tile at pyramid level K whose pixel (0, 0) is level-K
      canvas pixel (Y0, X0); level-K pixel i has its centre at level-0 2^K*i + (2^K - 1)/2 - PY.
      OUT (.npy, .png, .tif) gets the same content in the tifxyz frame: OUT pixel (r, c) is tifxyz
      level-K cell (TY0 + r, TX0 + c), centre at full-res cell 2^K*(TY0 + r) + (2^K - 1)/2.
      Default out origin: the tile's own origin moved by the model offset at the tile centre.
      --order 0 (nearest) for labels and masks, 1 (bilinear) for images. Pixels whose source falls
      outside the tile get --fill (default 0); their count is printed.
      PY: the w00 HF65 image pyramid has levels 1, 2, 3 shifted by 1, 3, 7 rows (README); labels
      were not checked, so PY defaults to 0 for everything.
"""
import os, sys, json, argparse
import numpy as np

# key: segment folder, canvas level-0 (H, W), tifxyz full-res (H, W) (README table 2),
#      dy = (a, b*y, c*x), dx = (a, b*y, c*x)  (README table 1)
FITS = {
    "w00": dict(segment="phercparis4/w00_20231016151002", canvas=(32249, 51380), tifxyz=(32281, 51345),
                dy=(-20.70, 0.000993, 0.0), dx=(+11.09, 0.0, -0.000663)),
    "w01": dict(segment="phercparis4/w01_20230702185753_r15", canvas=(65419, 56889), tifxyz=(65698, 56669),
                dy=(+11.51, 0.004263, 0.0), dx=(+11.82, 0.0, -0.003876)),
    "w05": dict(segment="phercparis4/w05_4424", canvas=(51400, 76155), tifxyz=(51440, 76120),
                dy=(-42.39, 0.000789, 0.0), dx=(+22.93, 0.0, -0.000454)),
    # README: "constant (sd 0.03)", range -19.7 to -19.6; the value is the mean of
    # results_c018.jsonl (measure/summarize.py constant model, -19.64 on both axes).
    "c018": dict(segment="1667/w018_20240304144031_2um", canvas=(42632, 98362), tifxyz=(42632, 98362),
                 dy=(-19.64, 0.0, 0.0), dx=(-19.64, 0.0, 0.0)),
}
ALIAS = {"w018": "c018"}
INV_ITERS = 6


class Model:
    def __init__(self, dy, dx, name=""):
        self.cy = np.asarray(dy, np.float64)
        self.cx = np.asarray(dx, np.float64)
        self.name = name

    @property
    def separable(self):
        return self.cy[2] == 0 and self.cx[1] == 0

    def offset(self, y, x):
        y = np.asarray(y, np.float64)
        x = np.asarray(x, np.float64)
        return (self.cy[0] + self.cy[1] * y + self.cy[2] * x,
                self.cx[0] + self.cx[1] * y + self.cx[2] * x)

    def fwd(self, y, x):
        """canvas (y, x) -> tifxyz full-res cell (ty, tx)."""
        dy, dx = self.offset(y, x)
        return np.asarray(y, np.float64) + dy, np.asarray(x, np.float64) + dx

    def inv(self, ty, tx, iters=INV_ITERS):
        """tifxyz cell (ty, tx) -> canvas (y, x) by fixed-point iteration y <- ty - dy(y, x).
        The map contracts by the largest slope (<= 0.0043 here), so 6 iterations are far below
        float64 rounding of the coordinates; see `check`."""
        ty = np.asarray(ty, np.float64)
        tx = np.asarray(tx, np.float64)
        y, x = ty.copy(), tx.copy()
        for _ in range(iters):
            dy, dx = self.offset(y, x)
            y, x = ty - dy, tx - dx
        return y, x

    def inv_exact(self, ty, tx):
        """Closed-form inverse of the affine map (reference for `check`)."""
        a = np.array([[1 + self.cy[1], self.cy[2]], [self.cx[1], 1 + self.cx[2]]])
        ai = np.linalg.inv(a)
        u = np.asarray(ty, np.float64) - self.cy[0]
        v = np.asarray(tx, np.float64) - self.cx[0]
        return ai[0, 0] * u + ai[0, 1] * v, ai[1, 0] * u + ai[1, 1] * v


def model(key, kind="readme", results_dir=None):
    key = ALIAS.get(key, key)
    if kind == "readme":
        f = FITS[key]
        return Model(f["dy"], f["dx"], f"{key}/readme")
    if kind == "affine3":
        return refit(key, results_dir)
    raise ValueError(kind)


def load_results(key, results_dir):
    rows = {}
    for line in open(os.path.join(results_dir, f"results_{key}.jsonl")):
        r = json.loads(line)
        rows[r["region"]] = r  # last run of a region wins, as in measure/summarize.py
    return [rows[k] for k in sorted(rows)]


def refit(key, results_dir):
    """Full 3-term fit d = a + b*y + c*x at region centres (y0 + size/2), as measure/summarize.py.
    For c018 the constant model (mean) is returned, as in the README."""
    rs = load_results(key, results_dir)
    Y = np.array([r["y0"] + r["size"] / 2 for r in rs])
    X = np.array([r["x0"] + r["size"] / 2 for r in rs])
    DY = np.array([r["level0_dy"] for r in rs])
    DX = np.array([r["level0_dx"] for r in rs])
    if key == "c018":
        return Model((DY.mean(), 0, 0), (DX.mean(), 0, 0), f"{key}/mean")
    A = np.stack([np.ones_like(Y), Y, X], 1)
    cy = np.linalg.lstsq(A, DY, rcond=None)[0]
    cx = np.linalg.lstsq(A, DX, rcond=None)[0]
    return Model(cy, cx, f"{key}/affine3")


# ---------------------------------------------------------------- resampling
def resample(tile, mdl, tile_origin, tile_step, out_origin, out_step, out_shape, order=1, fill=0.0,
             tile_valid=None, block_rows=256):
    """out[r, c] = tile sampled at canvas coordinate inv(out_origin + out_step*(r, c)).
    tile pixel (i, j) has its centre at canvas level-0 coordinate tile_origin + tile_step*(i, j);
    out pixel (r, c) has its centre at tifxyz full-res cell out_origin + out_step*(r, c).
    order 0 = nearest, 1 = bilinear (all four neighbours must be inside and valid).
    Returns (out, ok)."""
    tile = np.asarray(tile)
    H, W = out_shape
    th, tw = tile.shape
    if tile_valid is None:
        tile_valid = np.ones(tile.shape, bool)
    out = np.full((H, W), fill, np.float64 if order == 1 else tile.dtype)
    ok = np.zeros((H, W), bool)
    tx = out_origin[1] + out_step * np.arange(W, dtype=np.float64)
    for r0 in range(0, H, block_rows):
        r1 = min(H, r0 + block_rows)
        ty = out_origin[0] + out_step * np.arange(r0, r1, dtype=np.float64)
        TY, TX = np.meshgrid(ty, tx, indexing="ij")
        Y, X = mdl.inv(TY, TX)
        u = (Y - tile_origin[0]) / tile_step
        v = (X - tile_origin[1]) / tile_step
        if order == 0:
            iu = np.rint(u).astype(np.int64)
            iv = np.rint(v).astype(np.int64)
            m = (iu >= 0) & (iu < th) & (iv >= 0) & (iv < tw)
            m[m] = tile_valid[iu[m], iv[m]]
            blk = out[r0:r1]
            blk[m] = tile[iu[m], iv[m]]
        else:
            iu = np.floor(u).astype(np.int64)
            iv = np.floor(v).astype(np.int64)
            fu, fv = u - iu, v - iv
            m = (iu >= 0) & (iu + 1 < th) & (iv >= 0) & (iv + 1 < tw)
            a, b, fu_, fv_ = iu[m], iv[m], fu[m], fv[m]
            m2 = tile_valid[a, b] & tile_valid[a + 1, b] & tile_valid[a, b + 1] & tile_valid[a + 1, b + 1]
            T = tile.astype(np.float64)
            val = (T[a, b] * (1 - fu_) * (1 - fv_) + T[a + 1, b] * fu_ * (1 - fv_)
                   + T[a, b + 1] * (1 - fu_) * fv_ + T[a + 1, b + 1] * fu_ * fv_)
            blk = out[r0:r1]
            sub = blk[m]
            sub[m2] = val[m2]
            blk[m] = sub
            mm = np.zeros_like(m)
            mm[m] = m2
            m = mm
        ok[r0:r1] = m
    return out, ok


# ---------------------------------------------------------------- I/O for the CLI
def read_2d(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        a = np.load(path, allow_pickle=False)
    elif ext == ".png":
        from PIL import Image as PILImage
        a = np.asarray(PILImage.open(path))
    elif ext in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(path)
    else:
        raise SystemExit(f"unsupported input {path}")
    if a.ndim != 2:
        raise SystemExit(f"{path}: expected a 2D array, got shape {a.shape}")
    return a


def write_2d(path, a):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, a)
    elif ext == ".png":
        from PIL import Image as PILImage
        PILImage.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8) if a.dtype != np.uint8 else a).save(path)
    elif ext in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(path, a)
    else:
        raise SystemExit(f"unsupported output {path}")


# ---------------------------------------------------------------- CLI
def cmd_table(_):
    print("| key | segment | dy = a + b*y + c*x | dx = a + b*y + c*x | canvas HxW | tifxyz HxW |")
    print("|---|---|---|---|---|---|")
    for k, f in FITS.items():
        dy, dx = f["dy"], f["dx"]
        print(f"| {k} | {f['segment']} | {dy[0]:+.2f} {dy[1]:+.6f}*y {dy[2]:+.6f}*x | {dx[0]:+.2f} {dx[1]:+.6f}*y {dx[2]:+.6f}*x"
              f" | {f['canvas'][0]}x{f['canvas'][1]} | {f['tifxyz'][0]}x{f['tifxyz'][1]} |")


def cmd_check(a):
    rd = a.results
    print(f"results dir: {rd}")
    for k, f in FITS.items():
        R = model(k)
        A = refit(k, rd)
        rs = load_results(k, rd)
        print(f"\n## {k}  ({len(rs)} regions)")
        print(f"  README : dy {R.cy[0]:+.4f} {R.cy[1]:+.7f}*y {R.cy[2]:+.7f}*x | dx {R.cx[0]:+.4f} {R.cx[1]:+.7f}*y {R.cx[2]:+.7f}*x")
        print(f"  refit  : dy {A.cy[0]:+.4f} {A.cy[1]:+.7f}*y {A.cy[2]:+.7f}*x | dx {A.cx[0]:+.4f} {A.cx[1]:+.7f}*y {A.cx[2]:+.7f}*x  ({A.name})")
        # README digits vs refit rounded the same way
        same = (round(A.cy[0], 2) == R.cy[0] and round(A.cx[0], 2) == R.cx[0]
                and round(A.cy[1], 6) == R.cy[1] and round(A.cx[2], 6) == R.cx[2])
        print(f"  README intercepts/main slopes equal the refit rounded to the printed digits: {same}")
        # per-region: measured vs README model vs refit, at region centre
        print("  region | centre (y, x) | measured dy, dx | README model | measured - README | measured - refit")
        for r in rs:
            cy, cx = r["y0"] + r["size"] / 2, r["x0"] + r["size"] / 2
            my, mx = r["level0_dy"], r["level0_dx"]
            py_, px_ = R.offset(cy, cx)
            ay, ax = A.offset(cy, cx)
            print(f"  {r['region']} | {cy:.0f}, {cx:.0f} | {my:+.2f}, {mx:+.2f} | {py_:+.2f}, {px_:+.2f} |"
                  f" {my - py_:+.2f}, {mx - px_:+.2f} | {my - ay:+.2f}, {mx - ax:+.2f}")
        # README vs refit over the whole canvas (corners and a grid)
        H, W = f["canvas"]
        gy, gx = np.meshgrid(np.linspace(0, H - 1, 65), np.linspace(0, W - 1, 65), indexing="ij")
        ry, rx = R.offset(gy, gx)
        ay, ax = A.offset(gy, gx)
        print(f"  |README - refit| over the canvas: dy max {np.abs(ry - ay).max():.2f} px, dx max {np.abs(rx - ax).max():.2f} px")
        # slopes vs metadata size ratio
        Ht, Wt = f["tifxyz"]
        print(f"  metadata size ratio - 1: rows {Ht / H - 1:+.6f} (README b_y {R.cy[1]:+.6f}), cols {Wt / W - 1:+.6f} (README c_x {R.cx[2]:+.6f})")
        # inversion error over the tifxyz extent (+-300 px margin)
        for M in (R, A):
            ty, tx = np.meshgrid(np.linspace(-300, f["tifxyz"][0] + 300, 513), np.linspace(-300, f["tifxyz"][1] + 300, 513), indexing="ij")
            errs = []
            for it in (1, 2, 3, INV_ITERS):
                y, x = M.inv(ty, tx, iters=it)
                ey, ex = M.inv_exact(ty, tx)
                fy, fx = M.fwd(y, x)
                errs.append((it, max(np.abs(y - ey).max(), np.abs(x - ex).max()), max(np.abs(fy - ty).max(), np.abs(fx - tx).max())))
            print(f"  inversion {M.name}: " + "; ".join(f"{it} iter: |inv - exact| max {e1:.2e} px, |fwd(inv(t)) - t| max {e2:.2e} px" for it, e1, e2 in errs))


def cmd_fwd(a):
    M = model(a.seg)
    ty, tx = M.fwd(a.y, a.x)
    print(f"{ty:.3f} {tx:.3f}")


def cmd_inv(a):
    M = model(a.seg)
    y, x = M.inv(a.ty, a.tx)
    print(f"{y:.3f} {x:.3f}")


def cmd_resample(a):
    M = model(a.seg, a.model, a.results)
    tile = read_2d(a.inp)
    K = a.level
    s = 2 ** K
    py, px = a.pyr_shift
    h0 = (s - 1) / 2.0
    tile_origin = (s * a.origin[0] + h0 - py, s * a.origin[1] + h0 - px)
    shape = tuple(a.shape) if a.shape else tile.shape
    if a.out_origin:
        oy, ox = a.out_origin
    else:
        cy = tile_origin[0] + s * (tile.shape[0] - 1) / 2
        cx = tile_origin[1] + s * (tile.shape[1] - 1) / 2
        dy, dx = M.offset(cy, cx)
        oy = a.origin[0] + int(np.rint(dy / s)) + (tile.shape[0] - shape[0]) // 2
        ox = a.origin[1] + int(np.rint(dx / s)) + (tile.shape[1] - shape[1]) // 2
    out_origin = (s * oy + h0, s * ox + h0)
    out, ok = resample(tile, M, tile_origin, s, out_origin, s, shape, order=a.order, fill=a.fill)
    if a.order == 0:
        out = out.astype(tile.dtype)
    elif np.issubdtype(tile.dtype, np.integer) and os.path.splitext(a.out)[1].lower() != ".npy":
        out = np.clip(np.rint(out), np.iinfo(tile.dtype).min, np.iinfo(tile.dtype).max).astype(tile.dtype)
    write_2d(a.out, out)
    cy = out_origin[0] + s * (shape[0] - 1) / 2
    cx = out_origin[1] + s * (shape[1] - 1) / 2
    dy, dx = M.offset(*M.inv(cy, cx))
    print(f"{a.inp} ({tile.shape[0]}x{tile.shape[1]}, level {K}, canvas origin {a.origin[0]},{a.origin[1]}) -> {a.out}"
          f" ({shape[0]}x{shape[1]}, tifxyz level-{K} origin {oy},{ox}); model {M.name}; offset at out centre"
          f" ({dy:+.2f}, {dx:+.2f}) level-0 px; {int((~ok).sum())} px filled with {a.fill} (source outside tile)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("table").set_defaults(fn=cmd_table)
    c = sp.add_parser("check")
    c.add_argument("results", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    c.set_defaults(fn=cmd_check)
    f = sp.add_parser("fwd"); f.add_argument("seg"); f.add_argument("y", type=float); f.add_argument("x", type=float); f.set_defaults(fn=cmd_fwd)
    i = sp.add_parser("inv"); i.add_argument("seg"); i.add_argument("ty", type=float); i.add_argument("tx", type=float); i.set_defaults(fn=cmd_inv)
    r = sp.add_parser("resample")
    r.add_argument("seg"); r.add_argument("inp"); r.add_argument("out")
    r.add_argument("--origin", type=int, nargs=2, required=True, metavar=("Y0", "X0"))
    r.add_argument("--level", type=int, default=0)
    r.add_argument("--pyr-shift", type=float, nargs=2, default=(0.0, 0.0), metavar=("PY", "PX"))
    r.add_argument("--order", type=int, choices=(0, 1), default=0)
    r.add_argument("--out-origin", type=int, nargs=2, metavar=("TY0", "TX0"))
    r.add_argument("--shape", type=int, nargs=2, metavar=("H", "W"))
    r.add_argument("--fill", type=float, default=0.0)
    r.add_argument("--model", choices=("readme", "affine3"), default="readme")
    r.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    r.set_defaults(fn=cmd_resample)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
