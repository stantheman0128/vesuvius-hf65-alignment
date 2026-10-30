"""My own tifxyz reader (uncompressed float32 strip TIFF) and CT level-L trilinear sampler."""
import json, struct, math
import numpy as np
import store
from canvas import hf

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
CT = {"p1667": S3 + "PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr",
      "p4": S3 + "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"}


class TiffF32:
    def __init__(self, url):
        self.url = url
        h = store.get_range(url, 0, 15)
        bo = "<" if h[:2] == b"II" else ">"
        big = struct.unpack(bo + "H", h[2:4])[0] == 43
        off = struct.unpack(bo + ("Q" if big else "I"), h[8:16] if big else h[4:8])[0]
        ifd = store.get_range(url, off, off + 4095, short_ok=True)
        n = struct.unpack(bo + ("Q" if big else "H"), ifd[:8] if big else ifd[:2])[0]
        p, esz, vo = (8, 20, 12) if big else (2, 12, 8)
        tags = {}
        for i in range(n):
            e = ifd[p + i * esz:p + (i + 1) * esz]
            tag, typ = struct.unpack(bo + "HH", e[:4])
            cnt = struct.unpack(bo + ("Q" if big else "I"), e[4:vo])[0]
            fmt = {3: "H", 4: "I", 16: "Q"}.get(typ)
            if fmt is None:
                continue
            sz = struct.calcsize(fmt)
            if cnt * sz <= (8 if big else 4):
                vals = struct.unpack(bo + fmt * cnt, e[vo:vo + cnt * sz])
            else:
                vp = struct.unpack(bo + ("Q" if big else "I"), e[vo:vo + (8 if big else 4)])[0]
                vals = struct.unpack(bo + fmt * cnt, store.get_range(url, vp, vp + cnt * sz - 1))
            tags[tag] = vals
        self.W, self.H = tags[256][0], tags[257][0]
        assert tags.get(259, (1,))[0] == 1 and tags[258][0] == 32 and 322 not in tags
        self.rps = tags.get(278, (self.H,))[0]
        self.so = tags[273]
        self.dt = np.dtype(bo + "f4")

    def block(self, r0, r1, c0, c1):
        out = np.full((r1 - r0, c1 - c0), -1.0, np.float32)
        a0, a1, b0, b1 = max(r0, 0), min(r1, self.H), max(c0, 0), min(c1, self.W)
        for r in range(a0, a1):
            s = r // self.rps
            a = self.so[s] + ((r - s * self.rps) * self.W + b0) * 4
            out[r - r0, b0 - c0:b1 - c0] = np.frombuffer(store.get_range(self.url, a, a + (b1 - b0) * 4 - 1), self.dt)
        return out


class Mesh:
    def __init__(self, url_of):
        self.meta = json.loads(store.get(url_of("meta.json")))
        self.sx, self.sy = (float(np.float32(v)) for v in self.meta["scale"])
        self.t = {a: TiffF32(url_of(f"{a}.tif")) for a in "xyz"}

    def points(self, rows, cols):
        """full-res cells (rows x cols, float) -> (len(rows), len(cols), 3) as (z, y, x); villa
        convention stored = full * scale; bilinear; any invalid corner -> NaN."""
        u = np.asarray(rows, float) * self.sy
        v = np.asarray(cols, float) * self.sx
        r0, r1 = int(math.floor(u.min())) - 1, int(math.floor(u.max())) + 3
        c0, c1 = int(math.floor(v.min())) - 1, int(math.floor(v.max())) + 3
        G = {a: self.t[a].block(r0, r1, c0, c1).astype(float) for a in "xyz"}
        ok = (G["z"] > 0) & (G["x"] != -1) & (G["y"] != -1) & np.isfinite(G["x"])
        iu, iv = np.floor(u).astype(int) - r0, np.floor(v).astype(int) - c0
        fu, fv = (u - np.floor(u))[:, None], (v - np.floor(v))[None, :]
        out = np.empty((len(u), len(v), 3))
        good = ok[np.ix_(iu, iv)] & ok[np.ix_(iu + 1, iv)] & ok[np.ix_(iu, iv + 1)] & ok[np.ix_(iu + 1, iv + 1)]
        for k, a in enumerate("zyx"):
            g = G[a]
            out[..., k] = (g[np.ix_(iu, iv)] * (1 - fu) * (1 - fv) + g[np.ix_(iu + 1, iv)] * fu * (1 - fv) +
                           g[np.ix_(iu, iv + 1)] * (1 - fu) * fv + g[np.ix_(iu + 1, iv + 1)] * fu * fv)
        out[~good] = np.nan
        return out


class CTVol:
    def __init__(self, base, level, centre=True):
        self.base, self.level = base, level
        za = json.loads(store.get(f"{base}/{level}/.zarray"))
        assert za["compressor"] is None and za["dtype"] == "|u1" and za["chunks"] == [128, 128, 128] and za["order"] == "C", za
        self.shape = np.array(za["shape"])
        self.sep = za.get("dimension_separator", ".")
        self.f = 2 ** level
        self.c = (self.f - 1) / 2.0 if centre else 0.0

    def sample(self, P):
        """P (..., 3) level-0 (z,y,x) -> trilinear value at this level, NaN outside/invalid."""
        sh = P.shape[:-1]
        P = P.reshape(-1, 3)
        out = np.full(len(P), np.nan, np.float32)
        ok = np.all(np.isfinite(P), 1)
        q = (P[ok] - self.c) / self.f
        b = np.floor(q).astype(np.int64)
        fr = q - b
        ins = np.all((b >= 0) & (b + 1 < self.shape), 1)
        b, fr = b[ins], fr[ins]
        idx = np.nonzero(ok)[0][ins]
        corn = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        V = np.stack([b + np.array(c) for c in corn], 1).reshape(-1, 3)
        ch = V // 128
        loc = V % 128
        # group by chunk and z: needed y range
        gk = ((ch[:, 0] * 100000 + ch[:, 1]) * 100000 + ch[:, 2]) * 128 + loc[:, 0]
        order = np.argsort(gk, kind="stable")
        gks, ys = gk[order], loc[order, 1]
        u, st = np.unique(gks, return_index=True)
        ymn, ymx = np.minimum.reduceat(ys, st), np.maximum.reduceat(ys, st)
        vals = np.zeros(len(V), np.float32)
        lut = {}
        for key, y0, y1 in zip(u, ymn, ymx):
            z = int(key % 128); c = int(key // 128)
            cx = c % 100000; cy = (c // 100000) % 100000; cz = c // 10000000000
            s = self.sep
            url = f"{self.base}/{self.level}/{cz}{s}{cy}{s}{cx}"
            a = z * 16384 + int(y0) * 128
            d = store.get_range(url, a, z * 16384 + (int(y1) + 1) * 128 - 1)
            lut[int(key)] = (int(y0), np.frombuffer(d, np.uint8).reshape(-1, 128))
        ends = list(st[1:]) + [len(order)]
        for key, s0, s1 in zip(u, st, ends):
            y0, arr = lut[int(key)]
            m = order[s0:s1]
            vals[m] = arr[loc[m, 1] - y0, loc[m, 2]]
        vals = vals.reshape(len(b), 8)
        res = np.zeros(len(b))
        for n, (i, j, k) in enumerate(corn):
            w = (fr[:, 0] if i else 1 - fr[:, 0]) * (fr[:, 1] if j else 1 - fr[:, 1]) * (fr[:, 2] if k else 1 - fr[:, 2])
            res += w * vals[:, n]
        out[idx] = res
        return out.reshape(sh)
