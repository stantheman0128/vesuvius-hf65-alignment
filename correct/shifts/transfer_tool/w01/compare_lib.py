"""Shared constants and coordinate maps for compare.py, cc.py and cc2.py.

Environment: W01_DIR is the segment folder (its tifxyz, w01_20230702185753_r15_inklabels.tif and
w01_20230702185753_r15_supervision_mask.tif). RUN_DIR holds the transfer outputs (offset/, control/,
as run_w01.sh writes them) and receives the compare outputs.
"""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import zarr  # noqa: E402

OUT = os.environ["RUN_DIR"]
W01 = os.environ["W01_DIR"]
LABEL_TIF = f"{W01}/w01_20230702185753_r15_inklabels.tif"
MASK_TIF = f"{W01}/w01_20230702185753_r15_supervision_mask.tif"
OFFSET_TIF = f"{OUT}/offset/w01_inklabels_offset.tif"
CONTROL_TIF = f"{OUT}/control/w01_inklabels_nooffset.tif"
CORRECT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PUB = os.path.dirname(CORRECT)
spec = importlib.util.spec_from_file_location("apply_fit", f"{CORRECT}/apply_fit.py")
apply_fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply_fit)

S_H, S_W = 6579, 5694                                  # w01 stored grid
SY, SX = 0.10014063119888306, 0.10047820955514908      # rows, cols
L_H, L_W = 65419, 56889                                # shipped label
O_H, O_W = 65698, 56669                                # transfer output = full_extent
OFFSET_FULL = (11.3181, 11.1463)

MODEL = apply_fit.model("w01", "affine3", f"{PUB}/results")
_A = np.array([[1 + MODEL.cy[1], MODEL.cy[2]], [MODEL.cx[1], 1 + MODEL.cx[2]]])
_AI = np.linalg.inv(_A)


def inv_separable(ty, tx):
    """apply_fit Model.inv_exact on a rows x cols grid, as a row part plus a column part."""
    u = np.asarray(ty, np.float64) - MODEL.cy[0]
    v = np.asarray(tx, np.float64) - MODEL.cx[0]
    return ((_AI[0, 0] * u)[:, None] + (_AI[0, 1] * v)[None, :],
            (_AI[1, 0] * u)[:, None] + (_AI[1, 1] * v)[None, :])


def nearest(tile, ty, tx):
    """apply_fit.resample order 0 (rint; outside gives 0 and ok False) at cells ty x tx."""
    y, x = inv_separable(ty, tx)
    iu = np.rint(y).astype(np.int64)
    iv = np.rint(x).astype(np.int64)
    del y, x
    ok = (iu >= 0) & (iu < tile.shape[0]) & (iv >= 0) & (iv < tile.shape[1])
    out = np.zeros(ok.shape, tile.dtype)
    out[ok] = tile[iu[ok], iv[ok]]
    return out, ok


def open_tiff(path):
    return zarr.open(tifffile.imread(path, aszarr=True), mode="r")


def load_label():
    return tifffile.imread(LABEL_TIF)


def shift(a, b, sigma=2.0):
    """s (rows, cols) with a(u) ~= b(u + s): peak of sum_u a(u) b(u + s), parabolic sub-pixel."""
    from scipy import ndimage

    a = ndimage.gaussian_filter(a.astype(np.float64), sigma)
    b = ndimage.gaussian_filter(b.astype(np.float64), sigma)
    a -= a.mean()
    b -= b.mean()
    shape = (2 * a.shape[0], 2 * a.shape[1])
    corr = np.fft.irfft2(np.conj(np.fft.rfft2(a, shape)) * np.fft.rfft2(b, shape), shape)
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    out = []
    for axis in (0, 1):
        idx = list(peak)
        values = []
        for step in (-1, 0, 1):
            idx[axis] = (peak[axis] + step) % shape[axis]
            values.append(corr[tuple(idx)])
        denom = values[0] - 2 * values[1] + values[2]
        sub = 0.5 * (values[0] - values[2]) / denom if abs(denom) > 1e-12 else 0.0
        signed = peak[axis] if peak[axis] <= shape[axis] // 2 else peak[axis] - shape[axis]
        out.append(float(signed + sub))
    return out


def predict_tool(i, j, offset_full):
    """Shift s (rows, cols) with tool(i, j) = AF(i + s) from the coordinate maps alone.

    The kernel picks label index floor((i + 0.5) * L / O - o) with o = offset * L / full_extent,
    i.e. label centre coordinate z = (i + 0.5) * L / O - o - 0.5; AF shows that label pixel at
    cell fwd(z).
    """
    oy = offset_full[0] * L_H / O_H
    ox = offset_full[1] * L_W / O_W
    zy = (i + 0.5) * L_H / O_H - oy - 0.5
    zx = (j + 0.5) * L_W / O_W - ox - 0.5
    ty, tx = MODEL.fwd(zy, zx)
    return float(ty - i), float(tx - j)
