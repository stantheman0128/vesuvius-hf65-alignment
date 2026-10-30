"""Non-recursive walk of hf://buckets/scrollprize/datasets/ink/ : scroll dirs, segment dirs, and any
non-.zarr subdirectory below a segment (depth <= 3). .zarr directories are not descended (their
metadata is read separately). Writes results/listing.jsonl; response bytes go to
$HF65_DATA/ledger_listing.tsv (default ./hf65_data, as in common.py)."""
import os, json, time, sys
from urllib.parse import quote
import httpx

AUDIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/"
cli = httpx.Client(follow_redirects=True, timeout=120)
DATA = os.environ.get("HF65_DATA", os.path.abspath("hf65_data"))
os.makedirs(DATA, exist_ok=True)
LED = os.path.join(DATA, "ledger_listing.tsv")


def ls(prefix):
    url = API + quote(prefix, safe="/") + "?recursive=false&limit=1000"
    out = []
    while url:
        for attempt in range(8):
            r = cli.get(url)
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"{r.status_code} {url}")
        else:
            raise RuntimeError("retries exhausted " + url)
        with open(LED, "a") as f:
            f.write(f"{len(r.content)}\t{url}\n")
        out += r.json()
        nxt = r.links.get("next", {}).get("url")
        url = nxt
    return out


def main():
    rows = []
    top = ls("ink")
    rows += top
    scrolls = [e["path"] for e in top if e["type"] == "directory"]
    print("scroll dirs:", scrolls, flush=True)
    for sd in scrolls:
        ent = ls(sd)
        rows += ent
        segs = [e["path"] for e in ent if e["type"] == "directory" and not e["path"].lower().endswith(".zarr")]
        print(sd, len(ent), "entries", len(segs), "subdirs", flush=True)
        stack = [(s, 1) for s in segs]
        while stack:
            d, depth = stack.pop()
            ent = ls(d)
            rows += ent
            for e in ent:
                if e["type"] == "directory" and not e["path"].lower().endswith(".zarr") and depth < 3:
                    stack.append((e["path"], depth + 1))
    with open(os.path.join(AUDIT, "results", "listing.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("entries", len(rows))


main()
