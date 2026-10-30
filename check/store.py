"""Independent byte store. Reads raw bytes from the other agent's cache (looked up through
its ledger: url|rng -> sha1 file), downloads anything missing into my own dir with a byte cap."""
import os, hashlib, ast, threading, time
import numpy as np
import httpx
OLD = os.environ.get("HF65_DATA", os.path.abspath("hf65_data"))
NEW = os.environ.get("HF65_CHECK_DATA", os.path.abspath("hf65_check_data"))
CAP = int(os.environ.get("CAP_MB", "240")) * 1000000
_lock = threading.Lock()
_idx = None
_cli = httpx.Client(follow_redirects=True, timeout=180)
STATS = {"hit": 0, "dl": 0}


def _key(url, rng):
    return hashlib.sha1(f"{url}|{rng}".encode()).hexdigest()


def _build():
    global _idx
    _idx = {}
    for root in (OLD, NEW):
        led = os.path.join(root, "ledger.tsv")
        if not os.path.exists(led):
            continue
        sizes = {}
        cdir = os.path.join(root, "cache")
        for sub in os.scandir(cdir):
            if sub.is_dir():
                for e in os.scandir(sub.path):
                    if ".tmp" not in e.name:
                        sizes[e.name] = e.stat().st_size
        seen = set()
        for line in open(led):
            n, url, rng = line.rstrip("\n").split("\t")
            r = None if rng == "None" else ast.literal_eval(rng)
            k = _key(url, r)
            if k in sizes and k not in seen:
                seen.add(k)
                a = 0 if r is None else r[0]
                _idx.setdefault(url, []).append((a, a + sizes[k] - 1, os.path.join(cdir, k[:2], k), r is None))


def new_bytes():
    led = os.path.join(NEW, "ledger.tsv")
    if not os.path.exists(led):
        return 0
    return sum(int(l.split("\t")[0]) for l in open(led))


def _download(url, r):
    if new_bytes() > CAP:
        raise RuntimeError("download cap reached")
    h = {"Range": f"bytes={r[0]}-{r[1]}"} if r else {}
    for att in range(6):
        try:
            resp = _cli.get(url, headers=h)
        except httpx.HTTPError:
            time.sleep(2 * (att + 1)); continue
        if resp.status_code == 404:
            return None
        if resp.status_code in (200, 206):
            if r and resp.status_code != 206:
                raise RuntimeError("range ignored")
            d = resp.content
            k = _key(url, r)
            p = os.path.join(NEW, "cache", k[:2], k)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(d)
            with _lock:
                with open(os.path.join(NEW, "ledger.tsv"), "a") as f:
                    f.write(f"{len(d)}\t{url}\t{r}\n")
                a = 0 if r is None else r[0]
                _idx.setdefault(url, []).append((a, a + len(d) - 1, p, r is None))
                STATS["dl"] += len(d)
            return d
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(3 * (att + 1)); continue
        raise RuntimeError(f"HTTP {resp.status_code} {url}")
    raise RuntimeError("retries")


def get(url):
    if _idx is None:
        _build()
    for a, b, p, full in _idx.get(url, []):
        if full:
            STATS["hit"] += 1
            return open(p, "rb").read()
    return _download(url, None)


def get_range(url, a, b, short_ok=False):
    """bytes [a, b] inclusive, assembled from cached pieces; downloads only if not covered."""
    if _idx is None:
        _build()
    n = b - a + 1
    out = np.zeros(n, np.uint8)
    cov = np.zeros(n, bool)
    for pa, pb, p, full in _idx.get(url, []):
        lo, hi = max(a, pa), min(b, pb)
        if lo > hi:
            continue
        with open(p, "rb") as f:
            f.seek(lo - pa)
            out[lo - a:hi - a + 1] = np.frombuffer(f.read(hi - lo + 1), np.uint8)
        cov[lo - a:hi - a + 1] = True
        if cov.all():
            STATS["hit"] += 1
            return out.tobytes()
    if short_ok and cov[0]:
        k = n if cov.all() else int(np.argmin(cov))
        return out[:k].tobytes()
    return _download(url, (a, b))
