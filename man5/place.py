"""Place the MAN5_outer_3_* label zarrs on the MAN5_outer_3 segment grid (villa#2369).

The three label zarrs MAN5_outer_3_{inklabels,supervision_mask,validation_mask}.zarr are
24061 x 29841 at level 0; the segment zarr, its tifxyz grid and front_inklabels.zarr are 24120 x 29900.
Offset convention: label pixel (y, x) goes to segment-grid pixel (y + dy, x + dx), level-0 px.
Output: the label on the segment grid, padded with 0 and cropped to the segment-grid shape of the
level (L0 24120 x 29900, L1 12060 x 14950, L2 6030 x 7475), one z slice (default 32, the middle).

  level 0: a window of the segment grid, --window Y0 Y1 X0 X1 (level-0 px)
  level 1, 2: the whole slice
At level k the shift is (dy, dx) / 2^k. When that is not an integer the label is shifted by linear
(area) interpolation, so edge pixels get values between 0 and 255; --binarize thresholds at 128.

Data: local copy first (MAN5_LOCAL, default ./man5_data/MAN5_outer_3), otherwise HTTP from the
scrollprize/datasets bucket through measure/common.fetch (cached and logged in HF65_DATA/ledger*.tsv).

argv: name level dy dx out.(npy|npz|png) [--window Y0 Y1 X0 X1] [--z Z] [--binarize]
  name: inklabels | supervision_mask | validation_mask | front (front_inklabels, placed with the
        offset as given; use 0 0 for the reference)"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "measure"))
import numpy as np
import numcodecs

LOCAL = os.environ.get("MAN5_LOCAL", os.path.abspath("man5_data/MAN5_outer_3"))
HFDIR = "ink/man5/MAN5_outer_3"
ZNAME = {"inklabels": "MAN5_outer_3_inklabels.zarr", "supervision_mask": "MAN5_outer_3_supervision_mask.zarr",
         "validation_mask": "MAN5_outer_3_validation_mask.zarr", "front": "front_inklabels.zarr",
         "segment": "MAN5_outer_3.zarr"}
STATS = {"local": 0, "remote": 0, "missing": 0}


def _fetch():
    import common  # imported lazily so a purely local run does not need HF65_DATA
    return common


class Lab:
    def __init__(self, name, level):
        self.name, self.level = name, level
        self.z = ZNAME[name]
        lp = os.path.join(LOCAL, self.z, str(level), ".zarray")
        if os.path.exists(lp):
            za = json.load(open(lp))
        else:
            c = _fetch()
            za = json.loads(c.fetch(c.hf(f"{HFDIR}/{self.z}/{level}/.zarray")))
        self.shape = tuple(za["shape"])
        self.ch = tuple(za["chunks"])
        self.sep = za.get("dimension_separator", ".")
        self.codec = numcodecs.get_codec(za["compressor"]) if za["compressor"] else None
        self.fill = za.get("fill_value", 0) or 0

    def chunk(self, cz, cy, cx):
        key = self.sep.join(str(v) for v in (cz, cy, cx))
        lp = os.path.join(LOCAL, self.z, str(self.level), *key.split("/"))
        raw = None
        if os.path.exists(lp):
            raw = open(lp, "rb").read()
            STATS["local"] += 1
        elif os.path.isdir(os.path.join(LOCAL, self.z, str(self.level), "0")) and os.environ.get("MAN5_LOCAL_COMPLETE", "1") == "1":
            # a level with chunk data in the local tree holds every stored chunk of that level
            # (levels 1, 2, 5 of inklabels and front_inklabels); absent = not stored (fill).
            # MAN5_LOCAL_COMPLETE=0 sends absent chunks to the network instead (spot check).
            STATS["missing"] += 1
        else:
            c = _fetch()
            raw = c.fetch(c.hf(f"{HFDIR}/{self.z}/{self.level}/{key}"))
            STATS["remote" if raw is not None else "missing"] += 1
        if raw is None:
            return np.full(self.ch, self.fill, np.uint8)
        d = self.codec.decode(raw) if self.codec else raw
        return np.frombuffer(d, np.uint8).reshape(self.ch)

    def slice2d(self, z, y0, y1, x0, x1, threads=8):
        """[y0:y1, x0:x1] of slice z, zero outside the array."""
        out = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cz = z // self.ch[0]
        lz = z - cz * self.ch[0]
        ys = range(max(0, y0) // self.ch[1], (min(y1, self.shape[1]) - 1) // self.ch[1] + 1)
        xs = range(max(0, x0) // self.ch[2], (min(x1, self.shape[2]) - 1) // self.ch[2] + 1)
        jobs = [(a, b) for a in ys for b in xs]
        if not jobs or min(y1, self.shape[1]) <= max(0, y0) or min(x1, self.shape[2]) <= max(0, x0):
            return out
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(lambda j: self.chunk(cz, *j)[lz], jobs))
        for (cy, cx), c in zip(jobs, res):
            gy, gx = cy * self.ch[1], cx * self.ch[2]
            sy0, sy1 = max(y0, gy, 0), min(y1, gy + self.ch[1], self.shape[1])
            sx0, sx1 = max(x0, gx, 0), min(x1, gx + self.ch[2], self.shape[2])
            if sy1 > sy0 and sx1 > sx0:
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = c[sy0 - gy:sy1 - gy, sx0 - gx:sx1 - gx]
        return out


def grid_shape(level):
    return Lab("segment", level).shape[1:]


def place(name, level, dy, dx, window=None, z=32, binarize=False):
    """-> (array on the segment grid (window or whole level), info).
    out[Y, X] = label[Y - sy, X - sx] with (sy, sx) = (dy, dx) / 2^level, linear for fractions."""
    f = 2 ** level
    L = Lab(name, level)
    GH, GW = grid_shape(level)
    if window is None:
        Y0, Y1, X0, X1 = 0, GH, 0, GW
    else:
        Y0, Y1, X0, X1 = window
        if level:
            raise SystemExit("--window is for level 0; levels 1 and 2 are whole-slice")
    Y0c, Y1c, X0c, X1c = max(Y0, 0), min(Y1, GH), max(X0, 0), min(X1, GW)   # crop to the grid
    sy, sx = dy / f, dx / f
    qy, qx = int(np.floor(sy)), int(np.floor(sx))
    fy, fx = sy - qy, sx - qx
    # source rows needed: Y - qy - {0, 1}
    src = L.slice2d(z, Y0c - qy - 1, Y1c - qy, X0c - qx - 1, X1c - qx).astype(np.float32)
    h, w = Y1c - Y0c, X1c - X0c
    a00 = src[1:1 + h, 1:1 + w]      # label[Y - qy, X - qx]
    a10 = src[0:h, 1:1 + w]          # label[Y - qy - 1, X - qx]
    a01 = src[1:1 + h, 0:w]
    a11 = src[0:h, 0:w]
    v = (1 - fy) * (1 - fx) * a00 + fy * (1 - fx) * a10 + (1 - fy) * fx * a01 + fy * fx * a11
    out = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    vv = np.rint(v).astype(np.uint8)
    if binarize:
        vv = np.where(v >= 128, 255, 0).astype(np.uint8)
    out[Y0c - Y0:Y1c - Y0, X0c - X0:X1c - X0] = vv
    info = {"name": name, "level": level, "label_shape": L.shape, "grid_shape": (GH, GW), "shift_levelk": (sy, sx),
            "integer_shift": fy == 0 and fx == 0, "window": (Y0, Y1, X0, X1), "z": z}
    return out, info


def save(path, a):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, a)
    elif ext == ".npz":
        np.savez_compressed(path, label=a)
    elif ext == ".png":
        from PIL import Image as PILImage
        PILImage.fromarray(a).save(path, optimize=True)
    else:
        raise SystemExit(f"unsupported output {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=sorted(ZNAME))
    p.add_argument("level", type=int, choices=(0, 1, 2))
    p.add_argument("dy", type=int)
    p.add_argument("dx", type=int)
    p.add_argument("out")
    p.add_argument("--window", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    p.add_argument("--z", type=int, default=32)
    p.add_argument("--binarize", action="store_true")
    a = p.parse_args()
    if a.level == 0 and not a.window:
        raise SystemExit("level 0 needs --window")
    if os.environ.get("HF65_LEDGER_TAG"):
        import common
        common.LEDGER = os.path.join(common.DATA, f"ledger_{os.environ['HF65_LEDGER_TAG']}.tsv")
    out, info = place(a.name, a.level, a.dy, a.dx, a.window, a.z, a.binarize)
    save(a.out, out)
    print(f"{a.out}: {out.shape} uint8, {int((out > 0).sum())} px > 0; {info}; chunks {STATS}")
