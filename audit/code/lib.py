"""Volume sampling (uncompressed OME-Zarr v2 via HTTP Range), HF65 image reads, masked NCC."""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numcodecs
from common import fetch, hf, s3, Tiff, MIRROR


class Volume:
    """Uncompressed zarr v2 u8 volume, chunks 128^3."""

    def __init__(self, base_url, level):
        self.base = base_url.rstrip("/")
        self.level = level
        za = json.loads(fetch(f"{self.base}/{level}/.zarray"))
        assert za["compressor"] is None and za["dtype"] == "|u1", za
        self.shape = tuple(za["shape"])
        self.ch = tuple(za["chunks"])
        assert self.ch == (128, 128, 128)
        self.sep = za.get("dimension_separator", ".")
        self.f = 2 ** level

    def to_level(self, p0, centre=True):
        """level-0 voxel coordinate (voxel centres at integers) -> this level's continuous index.
        centre=True assumes level-k voxel i is the mean of level-0 voxels [f*i, f*i+f-1]."""
        return (p0 - ((self.f - 1) / 2.0 if centre else 0.0)) / self.f

    def sample(self, zyx0, threads=4, gap=8192, centre=True):
        """Trilinear sample at level-0 ZYX points (N,3), NaN rows -> NaN.
        Fetches only the (chunk, z-slice, y-row-range) pieces the 8 trilinear corners need."""
        zyx0 = np.asarray(zyx0, np.float64)
        ok = np.all(np.isfinite(zyx0), axis=1)
        c = self.to_level(zyx0[ok], centre)
        base = np.floor(c).astype(np.int64)
        fr = (c - base).astype(np.float32)
        shp = np.array(self.shape)
        inside = np.all((base >= 0) & (base + 1 < shp), axis=1)
        out = np.full(len(zyx0), np.nan, np.float32)
        if not inside.any():
            return out, {"bytes_req": 0, "requests": 0}
        idx_ok = np.nonzero(ok)[0][inside]
        base, fr = base[inside], fr[inside]
        corners = np.array([[a, b, d] for a in (0, 1) for b in (0, 1) for d in (0, 1)])
        allv = (base[:, None, :] + corners[None, :, :]).reshape(-1, 3)
        lo = allv.min(0)
        hi = allv.max(0)
        box = np.zeros(tuple(hi - lo + 1), np.uint8)
        cz, cy, cx = (allv // 128).T
        lz, ly = allv[:, 0] % 128, allv[:, 1] % 128
        nc = (shp + 127) // 128
        key = ((cz * nc[1] + cy) * nc[2] + cx) * 128 + lz
        order = np.argsort(key, kind="stable")
        ks, lys = key[order], ly[order]
        uk, start = np.unique(ks, return_index=True)
        ymin = np.minimum.reduceat(lys, start)
        ymax = np.maximum.reduceat(lys, start)
        chunk_of = uk // 128
        lz_of = uk % 128
        reqs = []
        cur = None
        for ck, z, y0, y1 in zip(chunk_of, lz_of, ymin, ymax):
            a = int(z) * 16384 + int(y0) * 128
            b = int(z) * 16384 + (int(y1) + 1) * 128 - 1
            if cur is not None and cur[0] == ck and a - cur[2] <= gap:
                cur[2] = max(cur[2], b)
            else:
                if cur is not None:
                    reqs.append(tuple(cur))
                cur = [ck, a, b]
        reqs.append(tuple(cur))

        def chunk_url(ck):
            k = int(ck)
            x = k % nc[2]
            k //= nc[2]
            y = k % nc[1]
            z = k // nc[1]
            s = self.sep
            return f"{self.base}/{self.level}/{z}{s}{y}{s}{x}", (z, y, x)

        def get(r):
            u, zyx = chunk_url(r[0])
            return zyx, r[1], fetch(u, (r[1], r[2]))

        nbytes = sum(r[2] - r[1] + 1 for r in reqs)
        with ThreadPoolExecutor(threads) as ex:
            for (zc, yc, xc), a, d in ex.map(get, reqs):
                if d is None:
                    continue
                flat = np.arange(a, a + len(d))
                vz, vy, vx = flat // 16384, (flat // 128) % 128, flat % 128
                gz, gy, gx = zc * 128 + vz - lo[0], yc * 128 + vy - lo[1], xc * 128 + vx - lo[2]
                m = (gz >= 0) & (gz < box.shape[0]) & (gy >= 0) & (gy < box.shape[1]) & (gx >= 0) & (gx < box.shape[2])
                box[gz[m], gy[m], gx[m]] = np.frombuffer(d, np.uint8)[m]
        b = base - lo
        val = np.zeros(len(b), np.float32)
        for (a_, b_, d_) in corners:
            w = (fr[:, 0] if a_ else 1 - fr[:, 0]) * (fr[:, 1] if b_ else 1 - fr[:, 1]) * (fr[:, 2] if d_ else 1 - fr[:, 2])
            val += w * box[b[:, 0] + a_, b[:, 1] + b_, b[:, 2] + d_]
        out[idx_ok] = val
        return out, {"bytes_req": nbytes, "requests": len(reqs), "box": tuple(int(v) for v in box.shape)}

    def block(self, z0, z1, y0, y1, x0, x1, threads=4):
        """Dense block at this level via the same Range machinery (for pyramid checks)."""
        zz, yy, xx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        p0 = np.stack([zz, yy, xx], -1).reshape(-1, 3) * self.f + (self.f - 1) / 2.0
        v, info = self.sample(p0, threads=threads)
        return v.reshape(zz.shape), info


class Image:
    """HF65 surface-volume zarr (blosc chunks)."""

    def __init__(self, hf_zarr_path, level):
        self.path = hf_zarr_path.rstrip("/")
        self.level = level
        za = json.loads(fetch(hf(f"{self.path}/{level}/.zarray")))
        self.shape = tuple(za["shape"])
        self.ch = tuple(za["chunks"])
        self.sep = za.get("dimension_separator", ".")
        self.codec = numcodecs.get_codec(za["compressor"]) if za["compressor"] else None
        self.fill = za.get("fill_value", 0) or 0
        mz = os.path.join(MIRROR, "r4", *self.path.split("/")[1:], str(level))
        self.mirror = None  # audit copy: never read a local mirror

    def chunk(self, cy, cx):
        # audit copy: read every z chunk (some canvases have 64-slice chunks over 65 slices)
        nz = -(-self.shape[0] // self.ch[0])
        parts = []
        for cz in range(nz):
            name = f"{cz}{self.sep}{cy}{self.sep}{cx}"
            raw = fetch(hf(f"{self.path}/{self.level}/{name}"))
            if raw is None:
                parts.append(np.full(self.ch, self.fill, np.uint8))
                continue
            d = self.codec.decode(raw) if self.codec else raw
            parts.append(np.frombuffer(d, np.uint8).reshape(self.ch))
        return np.concatenate(parts, axis=0) if nz > 1 else parts[0]

    def region(self, y0, y1, x0, x1, threads=4):
        """All slices of [y0:y1, x0:x1] at this level -> (Z, h, w) uint8 (0 outside)."""
        Z = self.shape[0]
        out = np.zeros((Z, y1 - y0, x1 - x0), np.uint8)
        cys = range(max(0, y0 // self.ch[1]), min((self.shape[1] - 1) // self.ch[1], (y1 - 1) // self.ch[1]) + 1)
        cxs = range(max(0, x0 // self.ch[2]), min((self.shape[2] - 1) // self.ch[2], (x1 - 1) // self.ch[2]) + 1)
        jobs = [(a, b) for a in cys for b in cxs]
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(lambda j: self.chunk(*j), jobs))
        for (cy, cx), c in zip(jobs, res):
            gy0, gx0 = cy * self.ch[1], cx * self.ch[2]
            sy0, sy1 = max(y0, gy0), min(y1, gy0 + self.ch[1], self.shape[1])
            sx0, sx1 = max(x0, gx0), min(x1, gx0 + self.ch[2], self.shape[2])
            out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = c[:Z, sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0]
        return out


class Surf:
    """tifxyz stored grid read on demand. Full-res cell (r, c) -> stored (r*sy, c*sx), the villa
    Tifxyz convention (half=0); half=0.5 gives the vc_render_tifxyz pixel-centre convention.
    Bilinear interpolation of x, y, z; any invalid (-1) neighbour -> NaN."""

    def __init__(self, seg_dir_hf, local_dir=None):
        self.seg = seg_dir_hf
        meta = json.loads(fetch(hf(f"{seg_dir_hf}/meta.json")))
        self.sx = float(np.float32(meta["scale"][0]))
        self.sy = float(np.float32(meta["scale"][1]))
        self.meta = meta
        self.t = {}
        for a in "xyz":
            lp = os.path.join(local_dir, f"{a}.tif") if local_dir else None
            self.t[a] = Tiff(lp if lp and os.path.exists(lp) else hf(f"{seg_dir_hf}/{a}.tif"))
        self.H, self.W = self.t["x"].H, self.t["x"].W
        self.full = (max(1, math.floor(self.H * (1.0 / self.sy) + 0.5)),
                     max(1, math.floor(self.W * (1.0 / self.sx) + 0.5)))

    def stored_rows(self, r0, r1, c0, c1):
        return {a: self.t[a].rows(r0, r1, c0, c1) for a in "xyz"}

    def cells(self, rows, cols, half=0.0):
        """ZYX level-0 points at full-res cells rows x cols -> (len(rows), len(cols), 3)."""
        sr = (np.asarray(rows, np.float64) + half) * self.sy
        sc = (np.asarray(cols, np.float64) + half) * self.sx
        r0, r1 = int(np.floor(sr.min())) - 1, int(np.floor(sr.max())) + 3
        c0, c1 = int(np.floor(sc.min())) - 1, int(np.floor(sc.max())) + 3
        g = self.stored_rows(r0, r1, c0, c1)
        valid = (g["x"] != -1) & (g["y"] != -1) & (g["z"] != -1) & np.isfinite(g["x"]) & (g["z"] > 0)
        ir = np.floor(sr).astype(int) - r0
        ic = np.floor(sc).astype(int) - c0
        fr = (sr - np.floor(sr))[:, None]
        fc = (sc - np.floor(sc))[None, :]
        out = np.empty((len(sr), len(sc), 3))
        v = (valid[np.ix_(ir, ic)] & valid[np.ix_(ir + 1, ic)] & valid[np.ix_(ir, ic + 1)]
             & valid[np.ix_(ir + 1, ic + 1)])
        for k, a in enumerate("zyx"):
            G = g[a].astype(np.float64)
            out[..., k] = (G[np.ix_(ir, ic)] * (1 - fr) * (1 - fc) + G[np.ix_(ir + 1, ic)] * fr * (1 - fc)
                           + G[np.ix_(ir, ic + 1)] * (1 - fr) * fc + G[np.ix_(ir + 1, ic + 1)] * fr * fc)
        out[~v] = np.nan
        return out


def gauss(img, s):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, s, mode="nearest")


def highpass(img, mask, s):
    """Masked high-pass: subtract the local masked mean (Gaussian sigma s), divide by local std."""
    m = mask.astype(np.float32)
    a = np.where(mask, img, 0).astype(np.float32)
    den = np.maximum(gauss(m, s), 1e-3)
    mu = gauss(a, s) / den
    h = np.where(mask, a - mu, 0)
    sd = np.sqrt(gauss(h * h, s) / den)
    return np.where(mask, h / np.maximum(sd, 1e-3), 0).astype(np.float32)


def ncc_map(I, Im, S, Sm, R, min_frac=0.5):
    """NCC(dy, dx) between I[y, x] and S[R + y + dy, R + x + dx] over the joint mask."""
    h, w = I.shape
    out = np.full((2 * R + 1, 2 * R + 1), np.nan, np.float32)
    nI = int(Im.sum())
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            m = Im & Sm[R + dy:R + dy + h, R + dx:R + dx + w]
            n = int(m.sum())
            if n < min_frac * nI or n < 100:
                continue
            a = I[m].astype(np.float64)
            b = S[R + dy:R + dy + h, R + dx:R + dx + w][m].astype(np.float64)
            a -= a.mean()
            b -= b.mean()
            d = math.sqrt((a * a).sum() * (b * b).sum())
            out[dy + R, dx + R] = (a * b).sum() / d if d > 0 else np.nan
    return out


def peaks(M, excl=4):
    """Main peak with parabolic sub-pixel refinement, and the best other 3x3 local maximum
    farther than excl pixels from it."""
    from scipy.ndimage import maximum_filter
    R = M.shape[0] // 2
    A = np.where(np.isfinite(M), M, -np.inf)
    i, j = np.unravel_index(np.argmax(A), A.shape)
    v = A[i, j]

    def sub(a, b, c):
        den = a - 2 * b + c
        if not (np.isfinite(a) and np.isfinite(c)) or den >= 0:
            return float("nan")
        return 0.5 * (a - c) / den

    sy = sub(A[i - 1, j], v, A[i + 1, j]) if 0 < i < A.shape[0] - 1 else float("nan")
    sx = sub(A[i, j - 1], v, A[i, j + 1]) if 0 < j < A.shape[1] - 1 else float("nan")
    lm = (A == maximum_filter(A, size=3, mode="constant", cval=-np.inf)) & np.isfinite(A)
    yy, xx = np.mgrid[:A.shape[0], :A.shape[1]]
    far = (yy - i) ** 2 + (xx - j) ** 2 > excl ** 2
    cand = np.where(lm & far, A, -np.inf)
    i2, j2 = np.unravel_index(np.argmax(cand), cand.shape)
    v2 = cand[i2, j2]
    on_edge = bool(i in (0, A.shape[0] - 1) or j in (0, A.shape[1] - 1))
    return {"dy": int(i - R), "dx": int(j - R), "ncc": float(v),
            "dy_sub": float(i - R + sy), "dx_sub": float(j - R + sx),
            "second": float(v2) if np.isfinite(v2) else None,
            "second_dy": int(i2 - R), "second_dx": int(j2 - R), "on_edge": on_edge}


class S3Image:
    """Uncompressed surface-volume zarr on S3 (chunks (Z,128,128), '/'): fetch only chosen slices
    of each chunk with HTTP Range."""

    def __init__(self, base_url, level, slices):
        self.base = base_url.rstrip("/")
        self.level = level
        za = json.loads(fetch(f"{self.base}/{level}/.zarray"))
        assert za["compressor"] is None and za["dtype"] == "|u1"
        self.full_shape = tuple(za["shape"])
        self.ch = tuple(za["chunks"])
        self.sep = za.get("dimension_separator", ".")
        self.slices = list(slices)
        self.shape = (len(self.slices),) + self.full_shape[1:]

    def chunk(self, cy, cx):
        s = self.sep
        u = f"{self.base}/{self.level}/0{s}{cy}{s}{cx}"
        n = self.ch[1] * self.ch[2]
        out = np.zeros((len(self.slices), self.ch[1], self.ch[2]), np.uint8)
        for i, k in enumerate(self.slices):
            d = fetch(u, (k * n, (k + 1) * n - 1))
            if d is not None:
                out[i] = np.frombuffer(d, np.uint8).reshape(self.ch[1], self.ch[2])
        return out

    region = Image.region


def surf_from_url(base_url):
    """Surf for a tifxyz directory given by URL (S3)."""
    s = Surf.__new__(Surf)
    s.seg = base_url
    meta = json.loads(fetch(f"{base_url}/meta.json"))
    s.sx = float(np.float32(meta["scale"][0]))
    s.sy = float(np.float32(meta["scale"][1]))
    s.meta = meta
    s.t = {a: Tiff(f"{base_url}/{a}.tif") for a in "xyz"}
    s.H, s.W = s.t["x"].H, s.t["x"].W
    s.full = (max(1, math.floor(s.H * (1.0 / s.sy) + 0.5)), max(1, math.floor(s.W * (1.0 / s.sx) + 0.5)))
    return s
