"""Verifier I/O: reuse the audit's fetch()/cache, but count our own cache misses separately."""
import os, sys, hashlib, json
AUDIT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # audit/
sys.path.insert(0, os.path.join(AUDIT, "code"))
import common
from common import hf, s3, CACHE, DATA

# cache misses of the independent check only; the shared ledger is $HF65_DATA/ledger.tsv
MYLEDGER = os.path.join(DATA, "independent_downloads.tsv")


def cached(url, rng=None):
    key = hashlib.sha1(f"{url}|{rng}".encode()).hexdigest()
    return os.path.exists(os.path.join(CACHE, key[:2], key))


def fetch_counted(url, rng=None):
    hit = cached(url, rng)
    d = common.fetch(url, rng)
    if not hit and d is not None:
        with open(MYLEDGER, "a") as f:
            f.write(f"{len(d)}\t{url}\t{rng}\n")
    return d


def mine():
    if not os.path.exists(MYLEDGER):
        return 0
    return sum(int(l.split("\t")[0]) for l in open(MYLEDGER) if l.strip())
