"""Shared constants and coordinate maps for the w00 compare.py, cc.py and cc2.py.

Same code as ../w01/compare_lib.py with w00 constants, the shifts.csv w00 row as the model
(apply_fit "readme" w00, copied unchanged into shifts.csv), and a third run at the loader-frame offset.

Environment: W00_DIR is the segment folder (its tifxyz, w00_20231016151002_inklabels.tif and
w00_20231016151002_supervision_mask.tif). RUN_DIR holds the transfer outputs (offset/, control/,
loader/, as run_w00.sh writes them) and receives the compare outputs.
"""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import zarr  # noqa: E402

OUT = os.environ["RUN_DIR"]
W00 = os.environ["W00_DIR"]
LABEL_TIF = f"{W00}/w00_20231016151002_inklabels.tif"
MASK_TIF = f"{W00}/w00_20231016151002_supervision_mask.tif"
OFFSET_TIF = f"{OUT}/offset/w00_inklabels_offset.tif"
CONTROL_TIF = f"{OUT}/control/w00_inklabels_nooffset.tif"
LOADER_TIF = f"{OUT}/loader/w00_inklabels_loaderframe.tif"
CORRECT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
spec = importlib.util.spec_from_file_location("apply_fit", f"{CORRECT}/apply_fit.py")
apply_fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply_fit)

S_H, S_W = 3241, 5143                                  # w00 stored grid
SY, SX = 0.10039844363927841, 0.10016534477472305      # rows, cols (meta.json scale is x, y)
L_H, L_W = 32249, 51380                                # shipped label = canvas
O_H, O_W = 32281, 51345                                # transfer output = full_extent
OFFSET_FULL = (-21.3771, 11.0061)                      # table DY DX (tool frame)
LOADER_FULL = (-20.6889, 11.5579)                      # table loader_DY loader_DX

MODEL = apply_fit.model("w00", "readme")
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
