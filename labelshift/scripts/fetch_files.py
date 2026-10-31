"""Download files from the public scrollprize/datasets HF bucket over plain HTTPS, in parallel.

usage: python -I fetch_files.py DST_ROOT LEDGER (--listing LISTING.json [--levels 3,4,5] [--bbox L Y0 Y1 X0 X1] | --path BUCKET_PATH)...

Metadata files (.zarray, .zattrs, .zgroup) of every level are always kept.

LISTING.json is the output of `hf buckets ls hf://buckets/scrollprize/datasets/<zarr> -R --json`.
Each file lands at DST_ROOT/<bucket path without the leading "ink/">. Files already present with the
listed size are skipped. Every real download appends "<bytes>\t<path>" to LEDGER.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import httpx

BASE = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
_lock = threading.Lock()
_client = httpx.Client(follow_redirects=True, timeout=120,
                       limits=httpx.Limits(max_connections=32, max_keepalive_connections=32))


def fetch(path, size, dst_root, ledger):
    rel = path[len("ink/"):] if path.startswith("ink/") else path
    dst = os.path.join(dst_root, rel)
    if os.path.exists(dst) and (os.path.getsize(dst) == size if size is not None else os.path.getsize(dst) > 0):
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = BASE + quote(path, safe="/")
    for attempt in range(12):
        try:
            with _client.stream("GET", url) as r:
                if r.status_code == 429:
                    wait = float(r.headers.get("retry-after") or 0) or 5 * (attempt + 1)
                    time.sleep(min(wait, 120))
                    raise RuntimeError("HTTP 429")
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                tmp = dst + f".part{threading.get_ident()}"
                n = 0
                with open(tmp, "wb") as f:
                    for block in r.iter_bytes(1 << 20):
                        f.write(block)
                        n += len(block)
            if size is not None and n != size:
                raise RuntimeError(f"size {n} != listed {size}")
            os.replace(tmp, dst)
            with _lock, open(ledger, "a") as f:
                f.write(f"{n}\t{path}\n")
            return n
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{path}: {last}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dst_root")
    p.add_argument("ledger")
    p.add_argument("--listing", action="append", default=[])
    p.add_argument("--levels", default=None, help="comma list of pyramid levels to keep (listings only)")
    p.add_argument("--path", action="append", default=[], help="single bucket path (size unknown)")
    p.add_argument("--bbox", type=int, nargs=5, metavar=("LEVEL", "Y0", "Y1", "X0", "X1"),
                   help="also keep chunks of LEVEL intersecting this level-LEVEL box (chunks 128 in y, x)")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    keep = None if a.levels is None else set(a.levels.split(","))
    jobs = []
    for listing in a.listing:
        for e in json.load(open(listing)):
            if e["type"] != "file":
                continue
            inner = e["path"].split(".zarr/", 1)[1]
            top = inner.split("/", 1)[0]
            meta = os.path.basename(inner).startswith(".")
            if keep is not None and top not in keep and not meta:
                continue
            if a.bbox and not meta:
                lvl, y0, y1, x0, x1 = a.bbox
                parts = inner.split("/")
                if parts[0] != str(lvl) or len(parts) != 4:
                    if keep is None or top not in keep:
                        continue
                else:
                    cy, cx = int(parts[2]), int(parts[3])
                    if not (y0 // 128 <= cy <= (y1 - 1) // 128 and x0 // 128 <= cx <= (x1 - 1) // 128):
                        continue
            jobs.append((e["path"], int(e["size"])))
    jobs += [(path, None) for path in a.path]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        got = list(ex.map(lambda j: fetch(j[0], j[1], a.dst_root, a.ledger), jobs))
    print(f"{len(jobs)} files, {sum(1 for g in got if g)} downloaded, {sum(got)} bytes, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
