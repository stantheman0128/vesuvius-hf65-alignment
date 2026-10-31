"""Small HF bucket reader for the w029 mask check.

listing(prefix): non-recursive tree call (JSON only).
get(path, rng=None): GET one file (or byte range) from the bucket; bytes are saved under DL and
counted in DL/ledger.tsv. Refuses to go past CAP_BYTES in total."""
import os, json, time, re
from urllib.parse import quote
import httpx

DL = os.environ["W029_DL"]
CAP_BYTES = 200 * 1000 * 1000
API = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/"
RES = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
LEDGER = os.path.join(DL, "ledger.tsv")
cli = httpx.Client(follow_redirects=True, timeout=180,
                   limits=httpx.Limits(max_connections=2, max_keepalive_connections=2))


def spent():
    if not os.path.exists(LEDGER):
        return 0
    with open(LEDGER) as f:
        return sum(int(l.split("\t")[0]) for l in f if l.strip())


def _log(n, url, note=""):
    with open(LEDGER, "a") as f:
        f.write(f"{n}\t{url}\t{note}\n")


def _get(url, headers=None):
    for attempt in range(12):
        try:
            r = cli.get(url, headers=headers or {})
        except httpx.HTTPError:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 429:
            m = re.search(r";t=(\d+)", r.headers.get("RateLimit", ""))
            time.sleep((int(m.group(1)) + 2) if m else 30)
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(5 * (attempt + 1))
            continue
        return r
    raise RuntimeError("retries exhausted " + url)


def listing(prefix):
    url = API + quote(prefix, safe="/") + "?recursive=false&limit=1000"
    out = []
    while url:
        r = _get(url)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {url}")
        _log(len(r.content), url, "listing")
        out += r.json()
        url = r.links.get("next", {}).get("url")
    return out


def get(path, rng=None):
    """Return bytes or None on 404. Cached under DL/files/<path>."""
    local = os.path.join(DL, "files", path.replace("/", os.sep))
    if rng:
        local += f".range{rng[0]}-{rng[1]}"
    if os.path.exists(local):
        with open(local, "rb") as f:
            return f.read()
    if spent() >= CAP_BYTES:
        raise RuntimeError(f"download cap reached: {spent()} bytes")
    url = RES + quote(path, safe="/")
    headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else None
    r = _get(url, headers)
    if r.status_code == 404:
        _log(0, url, "404")
        return None
    if r.status_code not in (200, 206):
        raise RuntimeError(f"{r.status_code} {url}")
    d = r.content
    _log(len(d), url, "range" if rng else "file")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "wb") as f:
        f.write(d)
    return d
