"""Serve HTTP range reads from bytes already in the HF65_DATA cache, downloading only the gaps.

measure/common.fetch caches by the exact (url, range) pair, so a shifted sampling lattice re-downloads
bytes that are already on disk under other ranges. install() wraps fetch: an exact cache hit is
unchanged; otherwise the ranges cached for that URL (indexed from the ledgers) are stitched together
and only the uncovered parts go to the network through the original fetch, which caches them and
writes them to the ledger as before. The bytes returned are the same bytes; nothing about the
sampling changes.

Ledger: when HF65_LEDGER_TAG is set, this process writes its downloads to ledger_<tag>.tsv in
HF65_DATA (several processes appending to one file on /mnt/c can lose lines); merge_ledgers()
appends them to ledger.tsv afterwards."""
import os, re, glob, hashlib, threading
import common
import lib

_orig_fetch = common.fetch
_idx = {}            # url -> list of [a, b, path, file_offset]
_lock = threading.Lock()
STATS = {"exact": 0, "covered": 0, "partial": 0, "uncached": 0, "gap_bytes": 0, "stitched_bytes": 0}
_RNG = re.compile(r"^\((\d+), (\d+)\)$")


def _path(url, rng_str):
    k = hashlib.sha1(f"{url}|{rng_str}".encode()).hexdigest()
    return os.path.join(common.CACHE, k[:2], k)


def _load():
    have = set()
    for d in os.listdir(common.CACHE):
        dd = os.path.join(common.CACHE, d)
        if os.path.isdir(dd):
            have.update(f for f in os.listdir(dd) if ".tmp" not in f)
    n = 0
    for led in [common.LEDGER] + sorted(glob.glob(os.path.join(common.DATA, "ledger_*.tsv"))):
        if not os.path.exists(led):
            continue
        for line in open(led):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            m = _RNG.match(parts[2])
            if not m:
                continue
            url, rs = parts[1], parts[2]
            k = hashlib.sha1(f"{url}|{rs}".encode()).hexdigest()
            if k not in have:
                continue
            a = int(m.group(1))
            b = a + int(parts[0]) - 1          # bytes actually stored
            _idx.setdefault(url, []).append([a, b, os.path.join(common.CACHE, k[:2], k), 0])
            n += 1
    for v in _idx.values():
        v.sort()
    return n


def _register(url, a, b, path):
    with _lock:
        lst = _idx.setdefault(url, [])
        lst.append([a, b, path, 0])
        lst.sort()


def fetch(url, rng=None, cache=True, allow_short=False):
    if rng is None or not cache:
        return _orig_fetch(url, rng, cache, allow_short)
    a, b = int(rng[0]), int(rng[1])
    p = _path(url, str(rng))
    if os.path.exists(p):
        STATS["exact"] += 1
        return _orig_fetch(url, rng, cache, allow_short)
    with _lock:
        ivs = list(_idx.get(url, ()))
    if not ivs or allow_short:
        STATS["uncached"] += 1
        d = _orig_fetch(url, rng, cache, allow_short)
        if d is not None:
            _register(url, a, a + len(d) - 1, p)
        return d
    pieces = []          # (start, bytes)
    pos = a
    gaps = 0
    while pos <= b:
        best = None
        for iv in ivs:
            if iv[0] <= pos <= iv[1] and (best is None or iv[1] > best[1]):
                best = iv
        if best is not None:
            end = min(best[1], b)
            with open(best[2], "rb") as f:
                f.seek(pos - best[0])
                chunk = f.read(end - pos + 1)
            if len(chunk) != end - pos + 1:
                raise RuntimeError(f"cache file shorter than ledger says: {best[2]}")
            pieces.append(chunk)
            STATS["stitched_bytes"] += len(chunk)
            pos = end + 1
            continue
        nxt = min((iv[0] for iv in ivs if iv[0] > pos), default=b + 1)
        g1 = min(b, nxt - 1)
        d = _orig_fetch(url, (pos, g1), cache, False)
        if d is None:
            return None
        gp = _path(url, str((pos, g1)))
        _register(url, pos, g1, gp)
        ivs.append([pos, g1, gp, 0])
        pieces.append(d)
        STATS["gap_bytes"] += len(d)
        gaps += 1
        pos = g1 + 1
    STATS["partial" if gaps else "covered"] += 1
    out = b"".join(pieces)
    assert len(out) == b - a + 1
    return out


def install():
    tag = os.environ.get("HF65_LEDGER_TAG")
    if tag:
        common.LEDGER = os.path.join(common.DATA, f"ledger_{tag}.tsv")
    n = _load()
    common.fetch = fetch
    lib.fetch = fetch
    return n


def ledger_bytes(path):
    if not os.path.exists(path):
        return 0
    return sum(int(l.split("\t", 1)[0]) for l in open(path) if l.strip())
