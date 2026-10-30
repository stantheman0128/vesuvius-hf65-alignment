"""Shared helpers for the HF65 vs tifxyz registration (paris4reg).

Every network read goes through fetch(), which caches under DATA/cache and appends
the byte count of each real download to DATA/ledger.tsv."""
import os, sys, hashlib, threading, time, struct, json, re
from urllib.parse import quote
import numpy as np
import httpx

DATA = os.environ.get("HF65_DATA", os.path.abspath("hf65_data"))
CACHE = os.path.join(DATA, "cache")
LEDGER = os.path.join(DATA, "ledger.tsv")
HF = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
MIRROR = os.environ.get("HF65_MIRROR", "/nonexistent")
_lock = threading.Lock()
_redir = {}
_cli = httpx.Client(follow_redirects=True, timeout=180,
                    limits=httpx.Limits(max_connections=4, max_keepalive_connections=4))


def hf(path):
    return HF + quote(path, safe="")


def s3(path):
    return S3 + path


def fetch(url, rng=None, cache=True, allow_short=False):
    """GET url (optionally bytes rng=(a, b) inclusive). None on 404."""
    key = hashlib.sha1(f"{url}|{rng}".encode()).hexdigest()
    p = os.path.join(CACHE, key[:2], key)
    if cache and os.path.exists(p):
        with open(p, "rb") as f:
            return f.read()
    headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    last = None
    for attempt in range(20):
        # audit copy: reuse the CDN redirect of an HF resolve URL for range reads (the resolver is
        # rate-limited to 3000 requests / 300 s per IP; the CDN is not), and wait out 429 windows.
        target = url
        if rng is not None and url.startswith(HF):
            with _lock:
                hit = _redir.get(url)
            if hit and time.time() - hit[1] < 600:
                target = hit[0]
        try:
            r = _cli.get(target, headers=headers)
        except httpx.HTTPError as e:
            last = e
            time.sleep(2 * (attempt + 1))
            continue
        if target != url and r.status_code in (401, 403, 410):
            with _lock:
                _redir.pop(url, None)
            continue
        if target == url and url.startswith(HF) and r.history and r.status_code in (200, 206):
            with _lock:
                _redir[url] = (str(r.url), time.time())
        if r.status_code == 429:
            last = 429
            m = re.search(r";t=(\d+)", r.headers.get("RateLimit", ""))
            time.sleep((int(m.group(1)) + 2) if m else 30 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        if r.status_code in (200, 206):
            d = r.content
            if rng is not None:
                if r.status_code != 206:
                    raise RuntimeError(f"range ignored for {url}")
                if len(d) != rng[1] - rng[0] + 1 and not allow_short:
                    raise RuntimeError(f"short range read {len(d)} for {url} {rng}")
            with _lock:
                with open(LEDGER, "a") as f:
                    f.write(f"{len(d)}\t{url}\t{rng}\n")
            if cache:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                tmp = p + f".tmp{threading.get_ident()}"
                with open(tmp, "wb") as f:
                    f.write(d)
                os.replace(tmp, p)
            return d
        if r.status_code in (500, 502, 503, 504):
            last = r.status_code
            time.sleep(3 * (attempt + 1))
            continue
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    raise RuntimeError(f"retries exhausted for {url}: {last}")


def ledger_total():
    if not os.path.exists(LEDGER):
        return 0
    with open(LEDGER) as f:
        return sum(int(l.split("\t", 1)[0]) for l in f if l.strip())


# ---------------- TIFF (uncompressed float32 strips) ----------------
class Src:
    """A byte source: local path or URL."""

    def __init__(self, loc):
        self.loc = loc

    def read(self, a, b, allow_short=False):
        if self.loc.startswith("http"):
            return fetch(self.loc, (a, b), allow_short=allow_short)
        with open(self.loc, "rb") as f:
            f.seek(a)
            return f.read(b - a + 1)


TT = {1: ("B", 1), 2: ("c", 1), 3: ("H", 2), 4: ("I", 4), 16: ("Q", 8), 12: ("d", 8), 11: ("f", 4)}


class Tiff:
    def __init__(self, loc):
        self.src = Src(loc)
        head = self.src.read(0, 15)
        bo = "<" if head[:2] == b"II" else ">"
        big = struct.unpack(bo + "H", head[2:4])[0] == 43
        off = struct.unpack(bo + ("Q" if big else "I"), head[8:16] if big else head[4:8])[0]
        ifd = self.src.read(off, off + 4095, allow_short=True)
        if big:
            n, p, esz, vo, ps = struct.unpack(bo + "Q", ifd[:8])[0], 8, 20, 12, 8
        else:
            n, p, esz, vo, ps = struct.unpack(bo + "H", ifd[:2])[0], 2, 12, 8, 4
        t = {}
        for i in range(n):
            e = ifd[p + i * esz: p + (i + 1) * esz]
            tag, typ = struct.unpack(bo + "HH", e[:4])
            cnt = struct.unpack(bo + ("Q" if big else "I"), e[4:vo])[0]
            if typ not in TT:
                continue
            fmt, sz = TT[typ]
            if cnt * sz <= ps:
                vals = struct.unpack(bo + fmt * cnt, e[vo:vo + cnt * sz])
            else:
                vp = struct.unpack(bo + ("Q" if big else "I"), e[vo:vo + ps])[0]
                vals = struct.unpack(bo + fmt * cnt, self.src.read(vp, vp + cnt * sz - 1))
            t[tag] = vals
        self.bo = bo
        self.W, self.H = t[256][0], t[257][0]
        assert t.get(259, (1,))[0] == 1, f"compressed tiff {loc}"
        assert t[258][0] == 32 and t.get(339, (1,))[0] == 3, f"not float32 {loc}"
        assert 322 not in t, "tiled tiff not supported"
        self.rps = t.get(278, (self.H,))[0]
        self.so = t[273]
        self.dt = np.dtype(bo + "f4")

    def rows(self, r0, r1, c0, c1, threads=4):
        """float32 array [r0:r1, c0:c1] (clipped reads outside -> -1)."""
        out = np.full((r1 - r0, c1 - c0), -1.0, np.float32)
        rr0, rr1 = max(r0, 0), min(r1, self.H)
        cc0, cc1 = max(c0, 0), min(c1, self.W)
        if rr0 >= rr1 or cc0 >= cc1:
            return out
        jobs = []
        for r in range(rr0, rr1):
            s = r // self.rps
            a = self.so[s] + ((r - s * self.rps) * self.W + cc0) * 4
            jobs.append((r, a, a + (cc1 - cc0) * 4 - 1))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(lambda j: self.src.read(j[1], j[2]), jobs))
        for (r, _, _), d in zip(jobs, res):
            out[r - r0, cc0 - c0:cc1 - c0] = np.frombuffer(d, self.dt)
        return out
