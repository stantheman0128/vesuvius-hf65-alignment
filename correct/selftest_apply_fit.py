"""Checks of the apply_fit.py resample CLI (run as a subprocess, files in a temp dir).
1) Real tile: the level-2 HF65 slice of a before-run (npz) resampled by the CLI equals the array
   method B produced in regcorr.py for the same lattice.
2) Point labels: single pixels of a level-0 label tile land at round(fwd(y, x)) in the output.
3) Round trip: canvas -> tifxyz -> canvas with order 0 returns the tile where both maps are inside.
argv: none (uses npz/before_w01_new0.npz and npz/before_c018_new0.npz)"""
import os, sys, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import apply_fit

PY = sys.executable
CLI = os.path.join(HERE, "apply_fit.py")
tmp = tempfile.mkdtemp()
ok_all = True


def cli(*args):
    r = subprocess.run([PY, "-I", CLI, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        raise SystemExit("CLI failed")
    return r.stdout.strip()


# 1) real tile vs method B
for key, label, y0, x0 in (("w01", "new0", 13312, 28672), ("c018", "new0", 6144, 89600)):
    z = np.load(os.path.join(HERE, "..", "npz", f"before_{key}_{label}.npz"))
    I, It, Itm = z["I"], z["It"], z["Itm"]
    oy, ox = int(z["oy"]), int(z["ox"])
    gi0, gj0 = y0 // 4, x0 // 4
    np.save(os.path.join(tmp, "tile.npy"), I)
    msg = cli("resample", key, os.path.join(tmp, "tile.npy"), os.path.join(tmp, "out.npy"), "--origin", gi0, gj0,
              "--level", 2, "--order", 1, "--out-origin", gi0 + oy, gj0 + ox)
    O = np.load(os.path.join(tmp, "out.npy"))
    d = np.abs(O[Itm] - It[Itm]).max()
    same_mask = np.array_equal(Itm, (O > 0) & Itm)
    print(f"1) {key} {label}: CLI level-2 bilinear vs method B array: max |diff| {d:.2e} over {int(Itm.sum())} px; {msg}")
    ok_all &= d < 1e-3 and same_mask

# 2) point labels at level 0
for key in ("w01", "w00", "w05", "c018"):
    M = apply_fit.model(key)
    Y0, X0 = 40000 if key != "w00" else 20000, 30000
    n = 512
    T = np.zeros((n, n), np.uint8)
    pts = [(10, 20), (100, 400), (255, 255), (500, 7), (37, 491)]
    for r, c in pts:
        T[r, c] = 255
    np.save(os.path.join(tmp, "lab.npy"), T)
    msg = cli("resample", key, os.path.join(tmp, "lab.npy"), os.path.join(tmp, "lab_t.npy"), "--origin", Y0, X0,
              "--order", 0, "--shape", n + 400, n + 400)
    O = np.load(os.path.join(tmp, "lab_t.npy"))
    oy0, ox0 = (int(v) for v in msg.split("tifxyz level-0 origin ")[1].split(")")[0].split(","))
    got = sorted(zip(*np.nonzero(O)))
    want = []
    for r, c in pts:
        ty, tx = M.fwd(Y0 + r, X0 + c)
        want.append((int(np.rint(ty)) - oy0, int(np.rint(tx)) - ox0))
    want.sort()
    good = [tuple(map(int, g)) for g in got] == want
    print(f"2) {key}: {len(pts)} single-pixel labels land at round(fwd(y, x)): {good}"
          + ("" if good else f" got {got} want {want}"))
    ok_all &= good

# 3) round trip at level 0, order 0: forward with the CLI, then back with apply_fit.resample on the inverse model
key = "w01"
M = apply_fit.model(key)
rng = np.random.default_rng(1)
T = (rng.random((300, 300)) < 0.2).astype(np.uint8) * 255
np.save(os.path.join(tmp, "rt.npy"), T)
msg = cli("resample", key, os.path.join(tmp, "rt.npy"), os.path.join(tmp, "rt_t.npy"), "--origin", 50000, 20000, "--order", 0)
O = np.load(os.path.join(tmp, "rt_t.npy"))
oy0, ox0 = (int(v) for v in msg.split("tifxyz level-0 origin ")[1].split(")")[0].split(","))
# back: canvas px p takes tifxyz value at fwd(p): resample with the inverse of fwd, i.e. a model whose inv is fwd
class Fwd:
    def inv(self, y, x):
        return M.fwd(y, x)
B, okb = apply_fit.resample(O, Fwd(), (oy0, ox0), 1, (50000, 20000), 1, T.shape, order=0)
sel = okb.copy()
sel[:4, :] = sel[-4:, :] = False
sel[:, :4] = sel[:, -4:] = False
agree = float((B[sel] == T[sel]).mean())
print(f"3) {key}: canvas -> tifxyz -> canvas (nearest both ways), agreement over {int(sel.sum())} interior px: {agree:.4f}")
ok_all &= agree > 0.99
print("ALL OK" if ok_all else "SOME CHECK FAILED")
