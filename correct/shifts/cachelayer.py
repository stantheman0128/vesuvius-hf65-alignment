"""Two-level byte cache for the validation runs: read the audit's cache first, put new downloads in a fresh folder.

audit/code/common.fetch caches by the exact (url, range) pair under $HF65_DATA/cache and logs each real
download to $HF65_DATA/ledger.tsv. install(old_data) wraps it so that
  1. an exact hit in old_data/cache is returned from there,
  2. a range read is stitched from every range already cached for that URL, in old_data (indexed from its
     ledger.tsv) or in $HF65_DATA (this run's own ledger), and only the uncovered gaps go to the network,
  3. everything new goes through the original fetch into $HF65_DATA, so $HF65_DATA/ledger.tsv counts
     exactly the bytes this work downloaded.
The bytes returned are the same bytes either way (same idea as correct/rangecache.py).
"""
import os, re, hashlib, threading
import common
import lib

_orig = common.fetch
_idx = {}
_lock = threading.Lock()
OLD_CACHE = None
STATS = {"old_exact": 0, "new_exact": 0, "stitched": 0, "partial": 0, "network": 0, "gap_bytes": 0}
_RNG = re.compile(r"^\((\d+), (\d+)\)$")


def _key(url, rng):
    return hashlib.sha1(f"{url}|{rng}".encode()).hexdigest()


def _index(data_dir):
    cache = os.path.join(data_dir, "cache")
    led = os.path.join(data_dir, "ledger.tsv")
    if not (os.path.isdir(cache) and os.path.exists(led)):
        return 0
    have = set()
    for d in os.listdir(cache):
        dd = os.path.join(cache, d)
        if os.path.isdir(dd):
            have.update(f for f in os.listdir(dd) if ".tmp" not in f)
    n = 0
    for line in open(led):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        m = _RNG.match(parts[2])
        if not m:
            continue
        k = _key(parts[1], parts[2])
        if k not in have:
            continue
        a = int(m.group(1))
        _idx.setdefault(parts[1], []).append([a, a + int(parts[0]) - 1, os.path.join(cache, k[:2], k)])
        n += 1
    return n


def _register(url, a, b, path):
    with _lock:
        _idx.setdefault(url, []).append([a, b, path])


def fetch(url, rng=None, cache=True, allow_short=False):
    k = _key(url, rng)
    p_old = os.path.join(OLD_CACHE, k[:2], k)
    if os.path.exists(p_old):
        STATS["old_exact"] += 1
        with open(p_old, "rb") as f:
            return f.read()
    p_new = os.path.join(common.CACHE, k[:2], k)
    if os.path.exists(p_new) or rng is None or allow_short:
        STATS["new_exact" if os.path.exists(p_new) else "network"] += 1
        d = _orig(url, rng, cache, allow_short)
        if d is not None and rng is not None:
            _register(url, rng[0], rng[0] + len(d) - 1, p_new)
        return d
    a, b = int(rng[0]), int(rng[1])
    with _lock:
        ivs = list(_idx.get(url, ()))
    pieces, pos, gaps = [], a, 0
    while pos <= b:
        best = None
        for iv in ivs:
            if iv[0] <= pos <= iv[1] and (best is None or iv[1] > best[1]):
                best = iv
        if best is not None:
            end = min(best[1], b)
            with open(best[2], "rb") as f:
                f.seek(pos - best[0])
                c = f.read(end - pos + 1)
            if len(c) != end - pos + 1:
                raise RuntimeError(f"cache file shorter than its ledger line: {best[2]}")
            pieces.append(c)
            pos = end + 1
            continue
        nxt = min((iv[0] for iv in ivs if iv[0] > pos), default=b + 1)
        g1 = min(b, nxt - 1)
        d = _orig(url, (pos, g1), cache, False)
        if d is None:
            return None
        h = _key(url, (pos, g1))
        gp = os.path.join(common.CACHE, h[:2], h)
        _register(url, pos, g1, gp)
        ivs.append([pos, g1, gp])
        pieces.append(d)
        STATS["gap_bytes"] += len(d)
        gaps += 1
        pos = g1 + 1
    STATS["partial" if gaps else "stitched"] += 1
    out = b"".join(pieces)
    assert len(out) == b - a + 1
    return out


def install(old_data):
    global OLD_CACHE
    OLD_CACHE = os.path.join(old_data, "cache")
    if os.path.abspath(old_data) == os.path.abspath(common.DATA):
        raise SystemExit("HF65_DATA must be a separate, fresh folder for new downloads")
    os.makedirs(common.CACHE, exist_ok=True)
    n_old = _index(old_data)
    n_new = _index(common.DATA)
    common.fetch = fetch
    lib.fetch = fetch
    return n_old, n_new
