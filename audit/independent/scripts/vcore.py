"""Verifier core: byte store over the audit cache, own TIFF row reader, own tifxyz->CT sampler,
own phase-correlation registration. Reused from the audit code: common.fetch (through vio),
common.hf / common.s3 URL builders, lib.Image (canvas chunk decode only)."""
import os, sys, json, hashlib, struct, re, bisect
HERE = os.path.dirname(os.path.abspath(__file__))  # audit/independent/scripts
AUDIT = os.path.dirname(os.path.dirname(HERE))  # audit/
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AUDIT, "code"))
import numpy as np
from scipy import ndimage as ndi
from vio import fetch_counted, hf, s3, mine
from common import CACHE, LEDGER

# ---------------- byte store over cached ranges ----------------
_IDX = None


def _index():
    global _IDX
    if _IDX is None:
        d = {}
        with open(LEDGER) if os.path.exists(LEDGER) else open(os.devnull) as f:
            for l in f:
                p = l.rstrip("\n").split("\t")
                if len(p) != 3 or p[2] == "None":
                    continue
                m = re.match(r"\((\d+), (\d+)\)", p[2])
                if m:
                    d.setdefault(p[1], []).append((int(m.group(1)), int(m.group(2)), p[2]))
        _IDX = {}
        for k, v in d.items():
            v.sort()
            _IDX[k] = (np.array([s[0] for s in v]), np.array([s[1] for s in v]), v)
    return _IDX


def _cache_file(url, rngstr):
    key = hashlib.sha1(f"{url}|{rngstr}".encode()).hexdigest()
    p = os.path.join(CACHE, key[:2], key)
    return p if os.path.exists(p) else None


class ByteStore:
    """read(url, A, B): assemble bytes [A, B] from any cached ranges of url; fetch only the gaps."""

    def __init__(self):
        self.miss_bytes = 0
        self._mem = {}

    def _load(self, url, rs):
        k = (url, rs)
        if k not in self._mem:
            if len(self._mem) > 4000:
                self._mem.clear()
            p = _cache_file(url, rs)
            self._mem[k] = open(p, "rb").read() if p else None
        return self._mem[k]

    def read(self, url, A, B):
        out = bytearray(B - A + 1)
        have = np.zeros(B - A + 1, bool)
        ent = _index().get(url)
        if ent is not None:
            starts, ends, segs = ent
            hi = bisect.bisect_right(starts, B)
            for t in range(hi):
                a, b, rs = segs[t]
                if b < A:
                    continue
                d = self._load(url, rs)
                if d is None or len(d) != b - a + 1:
                    continue
                s0, s1 = max(a, A), min(b, B)
                out[s0 - A:s1 - A + 1] = d[s0 - a:s1 - a + 1]
                have[s0 - A:s1 - A + 1] = True
        if not have.all():
            miss = np.nonzero(~have)[0]
            br = np.nonzero(np.diff(miss) > 1)[0]
            st = np.r_[miss[0], miss[br + 1]]
            en = np.r_[miss[br], miss[-1]]
            for s0, s1 in zip(st, en):
                d = fetch_counted(url, (int(A + s0), int(A + s1)))
                out[s0:s1 + 1] = d
                self.miss_bytes += len(d)
        return bytes(out)


BS = ByteStore()


# ---------------- TIFF (own parser, float32 uncompressed strips) ----------------
class TiffRows:
    def __init__(self, url):
        self.url = url
        h = fetch_counted(url, (0, 15))
        assert h[:2] == b"II", "big-endian tiff not handled"
        assert struct.unpack("<H", h[2:4])[0] == 42, "bigtiff not handled"
        off = struct.unpack("<I", h[4:8])[0]
        n = struct.unpack("<H", BS.read(url, off, off + 1))[0]
        ifd = BS.read(url, off + 2, off + 2 + 12 * n - 1)
        tags = {}
        for i in range(n):
            e = ifd[12 * i: 12 * i + 12]
            tag, typ, cnt = struct.unpack("<HHI", e[:8])
            tags[tag] = (typ, cnt, e[8:12])

        def val(tag):
            typ, cnt, raw = tags[tag]
            fmt, sz = {3: ("H", 2), 4: ("I", 4)}[typ]
            if cnt * sz <= 4:
                return list(struct.unpack("<" + fmt * cnt, raw[:cnt * sz]))
            p = struct.unpack("<I", raw)[0]
            return list(struct.unpack("<" + fmt * cnt, BS.read(url, p, p + cnt * sz - 1)))

        self.W, self.H = val(256)[0], val(257)[0]
        assert val(258)[0] == 32 and val(339)[0] == 3 and val(259)[0] == 1
        assert 322 not in tags
        self.rps = val(278)[0] if 278 in tags else self.H
        self.so = val(273)

    def block(self, r0, r1, c0, c1):
        out = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
        ca, cb = max(c0, 0), min(c1, self.W)
        for r in range(max(r0, 0), min(r1, self.H)):
            s = r // self.rps
            a = self.so[s] + ((r - s * self.rps) * self.W + ca) * 4
            d = BS.read(self.url, a, a + (cb - ca) * 4 - 1)
            out[r - r0, ca - c0:cb - c0] = np.frombuffer(d, "<f4")
        return out


class Tifxyz:
    """mk(name) -> URL of a file in the tifxyz directory."""

    def __init__(self, mk):
        self.meta = json.loads(fetch_counted(mk("meta.json")))
        self.t = {a: TiffRows(mk(f"{a}.tif")) for a in "xyz"}
        self.sx, self.sy = float(self.meta["scale"][0]), float(self.meta["scale"][1])
        self.H, self.W = self.t["x"].H, self.t["x"].W

    def zyx_at_full(self, rows, cols, half=0.0):
        """Own bilinear interpolation of the stored grid at full-res cells rows x cols;
        stored coordinate = (cell + half) * scale (villa Tifxyz: half = 0).
        Any invalid one of the 4 neighbours -> NaN."""
        sr = (np.asarray(rows, float) + half) * self.sy
        sc = (np.asarray(cols, float) + half) * self.sx
        r0, r1 = int(np.floor(sr.min())) - 2, int(np.floor(sr.max())) + 3
        c0, c1 = int(np.floor(sc.min())) - 2, int(np.floor(sc.max())) + 3
        G = {a: self.t[a].block(r0, r1, c0, c1).astype(np.float64) for a in "xyz"}
        bad = np.zeros_like(G["x"], bool)
        for a in "xyz":
            bad |= ~np.isfinite(G[a]) | (G[a] == -1)
        bad |= ~(G["z"] > 0)
        fr, fc = sr - r0, sc - c0
        ir, ic = np.floor(fr).astype(int), np.floor(fc).astype(int)
        wr, wc = (fr - ir)[:, None], (fc - ic)[None, :]
        out = np.empty((len(sr), len(sc), 3))
        I, J = np.ix_(ir, ic), np.ix_(ir + 1, ic + 1)
        for k, a in enumerate("zyx"):
            g = G[a]
            out[..., k] = (g[ir][:, ic] * (1 - wr) * (1 - wc) + g[ir + 1][:, ic] * wr * (1 - wc)
                           + g[ir][:, ic + 1] * (1 - wr) * wc + g[ir + 1][:, ic + 1] * wr * wc)
        b = bad[ir][:, ic] | bad[ir + 1][:, ic] | bad[ir][:, ic + 1] | bad[ir + 1][:, ic + 1]
        out[b] = np.nan
        return out


# ---------------- CT volume (uncompressed 128^3 u8, own trilinear) ----------------
class CT:
    def __init__(self, vol_url, level, centre_mean=True):
        self.base = vol_url.rstrip("/")
        self.L = level
        za = json.loads(fetch_counted(f"{self.base}/{level}/.zarray"))
        assert za["compressor"] is None and za["dtype"] == "|u1" and tuple(za["chunks"]) == (128, 128, 128)
        self.shape = np.array(za["shape"])
        self.sep = za.get("dimension_separator", ".")
        self.f = 2 ** level
        self.centre_mean = centre_mean

    def lvl(self, p0):
        return (p0 - ((self.f - 1) / 2.0 if self.centre_mean else 0.0)) / self.f

    def rows(self, cz, cy, cx, lz, ya, yb):
        """level-local voxel rows [ya, yb] (inclusive) of z-slice lz in chunk (cz, cy, cx) -> (n, 128)."""
        url = f"{self.base}/{self.L}/{cz}{self.sep}{cy}{self.sep}{cx}"
        d = BS.read(url, lz * 16384 + ya * 128, lz * 16384 + (yb + 1) * 128 - 1)
        return np.frombuffer(d, np.uint8).reshape(-1, 128)

    def sample_level(self, c):
        """own trilinear at continuous level coords c (N, 3); fetch only needed (chunk, z, row-span)."""
        out = np.full(len(c), np.nan, np.float32)
        ok = np.all(np.isfinite(c), 1)
        if not ok.any():
            return out
        cc = c[ok]
        base = np.floor(cc).astype(np.int64)
        lo = base.min(0)
        hi = base.max(0) + 2
        need = np.zeros(tuple(hi - lo), bool)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    need[base[:, 0] - lo[0] + dz, base[:, 1] - lo[1] + dy, base[:, 2] - lo[2] + dx] = True
        box = np.zeros(need.shape, np.float32)
        zs, ys, xs = np.nonzero(need)
        gz, gy, gx = zs + lo[0], ys + lo[1], xs + lo[2]
        inside = (gz >= 0) & (gz < self.shape[0]) & (gy >= 0) & (gy < self.shape[1]) & (gx >= 0) & (gx < self.shape[2])
        gz, gy, gx = gz[inside], gy[inside], gx[inside]
        groups = {}
        for z_, y_, x_ in zip(gz.tolist(), gy.tolist(), gx.tolist()):
            k = (z_ // 128, y_ // 128, x_ // 128, z_ % 128)
            g = groups.get(k)
            if g is None:
                groups[k] = [y_ % 128, y_ % 128]
            else:
                if y_ % 128 < g[0]:
                    g[0] = y_ % 128
                if y_ % 128 > g[1]:
                    g[1] = y_ % 128
        for (czc, cyc, cxc, lz), (ya, yb) in groups.items():
            d = self.rows(czc, cyc, cxc, lz, ya, yb)
            Z = czc * 128 + lz - lo[0]
            Y0 = cyc * 128 + ya - lo[1]
            X0 = cxc * 128 - lo[2]
            xa, xb = max(0, -X0), min(128, box.shape[2] - X0)
            ya2, yb2 = max(0, -Y0), min(d.shape[0], box.shape[1] - Y0)
            if xa < xb and ya2 < yb2:
                box[Z, Y0 + ya2:Y0 + yb2, X0 + xa:X0 + xb] = d[ya2:yb2, xa:xb]
        out[ok] = ndi.map_coordinates(box, (cc - lo).T, order=1, mode="constant", cval=np.nan)
        return out

    def sample(self, zyx0):
        P = np.asarray(zyx0, float)
        sh = P.shape[:-1]
        return self.sample_level(self.lvl(P.reshape(-1, 3))).reshape(sh)


# ---------------- registration: phase correlation on gradient magnitude ----------------
def gradmag(a, sigma=1.0):
    a = a.astype(np.float64)
    return np.hypot(ndi.gaussian_filter(a, sigma, order=(0, 1)), ndi.gaussian_filter(a, sigma, order=(1, 0)))


def prep(a, valid):
    fill = float(np.nanmean(a[valid])) if valid.any() else 0.0
    g = gradmag(np.where(valid, a, fill))
    v = ndi.binary_erosion(valid, iterations=3)
    g = np.where(v, g - g[v].mean(), 0)
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    return g * w


def phasecorr(A, B):
    """Phase-only correlation. Convention: A(y) ~ B(y + d), i.e. the content of A at pixel y is
    found at y + d in B. Returns (surface, dy, dx, peak, best other peak > 3 px away)."""
    FA, FB = np.fft.fft2(A), np.fft.fft2(B)
    Rr = np.conj(FA) * FB
    Rr /= np.maximum(np.abs(Rr), 1e-12)
    P = np.fft.fftshift(np.real(np.fft.ifft2(Rr)))
    cy, cx = np.array(P.shape) // 2
    i, j = np.unravel_index(np.argmax(P), P.shape)
    pk = P[i, j]

    def g3(a, b, c):
        a, b, c = [max(v, 1e-9) for v in (a, b, c)]
        den = 2 * (np.log(a) - 2 * np.log(b) + np.log(c))
        return 0.0 if den == 0 else (np.log(a) - np.log(c)) / den

    sy = g3(P[i - 1, j], pk, P[i + 1, j])
    sx = g3(P[i, j - 1], pk, P[i, j + 1])
    yy, xx = np.mgrid[:P.shape[0], :P.shape[1]]
    far = (yy - i) ** 2 + (xx - j) ** 2 > 9
    sec = P[far].max()
    return P, float(i - cy + sy), float(j - cx + sx), float(pk), float(sec)


def selftest():
    rng = np.random.default_rng(0)
    base = ndi.gaussian_filter(rng.random((400, 400)), 2)
    A = base[100:356, 100:356]
    # B(y, x) = base(y - 7.3, x + 4.6) -> content of A at y sits at y + 7.3 in B: expect (+7.3, -4.6)
    B = ndi.shift(base, (7.3, -4.6), order=3)[100:356, 100:356]
    v = np.ones_like(A, bool)
    _, dy, dx, pk, sec = phasecorr(prep(A, v), prep(B, v))
    return dy, dx, pk, sec


def refine(A, B, dy0, dx0, up=20, half=1.5):
    """Sub-pixel refinement: evaluate the phase-only correlation surface by a direct (matrix) DFT on a
    fine grid of +-half px around (dy0, dx0), step 1/up. Same convention as phasecorr."""
    FA, FB = np.fft.fft2(A), np.fft.fft2(B)
    Rr = np.conj(FA) * FB
    Rr /= np.maximum(np.abs(Rr), 1e-12)
    H, W = A.shape
    ky = np.fft.fftfreq(H)
    kx = np.fft.fftfreq(W)
    ty = dy0 + np.arange(-half, half + 1e-9, 1.0 / up)
    tx = dx0 + np.arange(-half, half + 1e-9, 1.0 / up)
    Ey = np.exp(2j * np.pi * np.outer(ty, ky))  # (nt, H)
    Ex = np.exp(2j * np.pi * np.outer(kx, tx))  # (W, nt)
    S = np.real(Ey @ Rr @ Ex) / (H * W)
    i, j = np.unravel_index(np.argmax(S), S.shape)
    return float(ty[i]), float(tx[j]), float(S[i, j])


def selftest2():
    rng = np.random.default_rng(1)
    base = ndi.gaussian_filter(rng.random((400, 400)), 2)
    A = base[100:356, 100:356]
    out = []
    for s in [(7.3, -4.6), (0.35, 0.35), (-19.6, -19.6), (0.0, 0.0)]:
        B = ndi.shift(base, s, order=3)[100:356, 100:356]
        v = np.ones_like(A, bool)
        pa, pb = prep(A, v), prep(B, v)
        _, dy, dx, pk, sec = phasecorr(pa, pb)
        out.append((s, refine(pa, pb, round(dy), round(dx))))
    return out
