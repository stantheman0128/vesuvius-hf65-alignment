"""Is the local MAN5 tree complete for levels 1 and 2 (absent chunk = not stored = all zero)?
a) remote .zarray of every level used equals the local one;
b) level 1 and level 2 slice 32, max-pooled to level 5, against local level 5 slice 32
   (a missing level-1 chunk under level-5 ink would show as level-5 ink with no level-1 ink);
c) a few chunks absent locally, chosen next to present ones, fetched remotely: expect 404.
HTTP through measure/common.fetch (cached, ledgered under HF65_LEDGER_TAG)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "measure"))
import numpy as np
import common
if os.environ.get("HF65_LEDGER_TAG"):
    common.LEDGER = os.path.join(common.DATA, f"ledger_{os.environ['HF65_LEDGER_TAG']}.tsv")
import place
from place import Lab, LOCAL, HFDIR, ZNAME

b0 = common.ledger_total()
for nm in ("inklabels", "front", "segment"):
    for lev in (0, 1, 2, 5):
        lp = os.path.join(LOCAL, ZNAME[nm], str(lev), ".zarray")
        rem = json.loads(common.fetch(common.hf(f"{HFDIR}/{ZNAME[nm]}/{lev}/.zarray")))
        loc = json.load(open(lp)) if os.path.exists(lp) else None
        print(f"a) {nm} L{lev}: remote shape {rem['shape']} chunks {rem['chunks']}; local {'same' if loc == rem else ('absent' if loc is None else 'DIFFERENT')}")
for nm in ("inklabels", "front"):
    L5 = Lab(nm, 5)
    A5 = L5.slice2d(32, 0, L5.shape[1], 0, L5.shape[2]) > 0
    for lev in (1, 2):
        L = Lab(nm, lev)
        A = L.slice2d(32, 0, L.shape[1], 0, L.shape[2]) > 0
        f = 2 ** (5 - lev)
        H, W = A5.shape
        P = np.zeros((H * f, W * f), bool)
        P[:min(A.shape[0], H * f), :min(A.shape[1], W * f)] = A[:H * f, :W * f]
        P = P.reshape(H, f, W, f).any(axis=(1, 3))
        only5 = int((A5 & ~P).sum())
        onlyk = int((P & ~A5).sum())
        print(f"b) {nm}: level-5 ink px {int(A5.sum())}; level {lev} pooled to level 5: {int(P.sum())}; level-5 ink with no level-{lev} ink {only5}; level-{lev} ink with no level-5 ink {onlyk}")
# c) absent chunks next to present ones
for nm in ("inklabels", "front"):
    L = Lab(nm, 1)
    d = os.path.join(LOCAL, ZNAME[nm], "1", "0")
    present = {(int(a), int(b)) for a in os.listdir(d) for b in os.listdir(os.path.join(d, a))}
    ny, nx = -(-L.shape[1] // L.ch[1]), -(-L.shape[2] // L.ch[2])
    nb = sorted({(a + da, b + db) for a, b in present for da, db in ((0, 1), (1, 0), (0, -1), (-1, 0))
                 if 0 <= a + da < ny and 0 <= b + db < nx} - present)
    pick = nb[:: max(1, len(nb) // 6)][:6]
    res = []
    for a, b in pick:
        r = common.fetch(common.hf(f"{HFDIR}/{ZNAME[nm]}/1/0/{a}/{b}"))
        res.append((a, b, "404" if r is None else f"{len(r)} bytes"))
    print(f"c) {nm} level 1: {len(present)} chunks present locally, {len(nb)} absent neighbours; remote check of {len(pick)}: {res}")
print(f"new bytes {common.ledger_total() - b0}")
