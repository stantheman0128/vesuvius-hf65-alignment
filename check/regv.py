"""Independent registration. argv: scriptdir key y0 x0 n R [oy ox]  (y0,x0,n level-0 canvas px; R, oy, ox
level-2 lattice px). Finds d maximising NCC(I(i), S(i + d)); I = canvas slice on a level-2 lattice,
S = CT sampled at tifxyz full-res cell = lattice centre. d != 0: canvas content at canvas px p is at
tifxyz full-res cell p + 4d. env CENTRE=0 -> CT corner convention; NOPYR=1 -> ignore w00 row shift."""
import sys, os, json, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import store
from canvas import Canvas, SEGS, hf
from geom import Mesh, CTVol, CT, S3

key = sys.argv[2]
y0, x0, n, R = (int(v) for v in sys.argv[3:7])
oy, ox = (int(v) for v in sys.argv[7:9]) if len(sys.argv) > 8 else (0, 0)
centre = os.environ.get("CENTRE", "1") == "1"
SL = int(os.environ.get("SLICE", "-1"))
h = n // 4
gi0, gj0 = y0 // 4, x0 // 4
t0 = time.time(); b0 = store.new_bytes()

S3SEG = {"c018s3": (S3 + "PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/",
                    "mesh/20240304144031-on-20251217075048-2.399um.tifxyz",
                    "surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr", "p1667"),
         "w00s3": (S3 + "PHercParis4/segments/20231016151002/",
                   "mesh/20231016151002-on-20260411134726-2.4um.tifxyz",
                   "surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr", "p4")}


def s3slice(base, level, k, i0, i1, j0, j1):
    za = json.loads(store.get(f"{base}/{level}/.zarray"))
    C = za["chunks"][1]
    out = np.zeros((i1 - i0, j1 - j0), np.uint8)
    for cy in range(i0 // C, (i1 - 1) // C + 1):
        for cx in range(j0 // C, (j1 - 1) // C + 1):
            d = store.get_range(f"{base}/{level}/0/{cy}/{cx}", k * C * C, (k + 1) * C * C - 1)
            c = np.frombuffer(d, np.uint8).reshape(C, C)
            a0, a1 = max(i0, cy * C), min(i1, cy * C + C)
            c0, c1 = max(j0, cx * C), min(j1, cx * C + C)
            out[a0 - i0:a1 - i0, c0 - j0:c1 - j0] = c[a0 - cy * C:a1 - cy * C, c0 - cx * C:c1 - cx * C]
    return out


rowshift = 0.0
if key in S3SEG:
    seg, meshp, svp, vol = S3SEG[key]
    mesh = Mesh(lambda nm: seg + meshp + "/" + nm)
    sl = 54 if SL < 0 else SL
    I = s3slice(seg + svp, 2, sl, gi0, gi0 + h, gj0, gj0 + h).astype(np.float32)
    depth = {k: s3slice(seg + svp, 2, k, gi0, gi0 + h, gj0, gj0 + h).astype(np.float32) for k in (50, 52, 56, 58)}
else:
    d, z = SEGS[key]
    mesh = Mesh(lambda nm: hf(d + "/" + nm))
    vol = "p1667" if key == "w018" else "p4"
    sl = 32 if SL < 0 else SL
    dz = [28, 30, 31, 33, 34, 36]
    if key == "w018":   # level 1: 2D-downsampled, depth-truncated to slices 0..32, shift 0 (verified)
        zl = [sl] + [k for k in dz if k < 33]
        cv = Canvas(key, 1)
        A = cv.read(zl, 2 * gi0, 2 * gi0 + 2 * h, 2 * gj0, 2 * gj0 + 2 * h).astype(np.float32)
        A = A.reshape(A.shape[0], h, 2, h, 2).mean(axis=(2, 4))
    else:               # level 2; w00 rows shifted by -3 (verified), w01 0 (verified)
        zl = [sl] + dz
        cv = Canvas(key, 2)
        A = cv.read(zl, gi0, gi0 + h, gj0, gj0 + h).astype(np.float32)
        if key == "w00" and os.environ.get("NOPYR", "0") != "1":
            rowshift = -3.0
    I = A[0]
    depth = {k: A[i] for i, k in enumerate(zl) if i > 0}
# canvas level-2 pixel i covers level-0 rows [4i + rowshift, 4i + rowshift + 4): centre 4i + rowshift + 1.5
ii = np.arange(gi0 + oy - R, gi0 + oy + h + R)
jj = np.arange(gj0 + ox - R, gj0 + ox + h + R)
P = mesh.points(4.0 * ii + 1.5 + rowshift, 4.0 * jj + 1.5)
S = CTVol(CT[vol], 2, centre).sample(P)
Sm = np.isfinite(S) & (S > 0)
S = np.where(Sm, S, 0).astype(np.float32)
Im = I > 0


def hp(a, m, s=3.0):
    mf = m.astype(np.float32)
    den = np.maximum(gaussian_filter(mf, s), 1e-3)
    mu = gaussian_filter(np.where(m, a, 0), s) / den
    g = np.where(m, a - mu, 0)
    sd = np.sqrt(gaussian_filter(g * g, s) / den)
    return np.where(m, g / np.maximum(sd, 1e-3), 0).astype(np.float32)


def nccmap(Ia, Ima, Sa, Sma):
    M = np.full((2 * R + 1, 2 * R + 1), np.nan)
    nI = Ima.sum()
    for a in range(2 * R + 1):
        for b in range(2 * R + 1):
            m = Ima & Sma[a:a + h, b:b + h]
            if m.sum() < 0.5 * nI:
                continue
            p = Ia[m].astype(np.float64); q = Sa[a:a + h, b:b + h][m].astype(np.float64)
            p -= p.mean(); q -= q.mean()
            dd = np.sqrt((p * p).sum() * (q * q).sum())
            M[a, b] = (p * q).sum() / dd if dd > 0 else np.nan
    return M  # M[a, b] <-> d = (a - R + oy, b - R + ox) level-2 px


def peak(M):
    A_ = np.where(np.isfinite(M), M, -np.inf)
    i, j = np.unravel_index(np.argmax(A_), A_.shape)
    def sub(l, c, r):
        den = l - 2 * c + r
        return 0.5 * (l - r) / den if np.isfinite(l) and np.isfinite(r) and den < 0 else np.nan
    sy = sub(A_[i - 1, j], A_[i, j], A_[i + 1, j]) if 0 < i < A_.shape[0] - 1 else np.nan
    sx = sub(A_[i, j - 1], A_[i, j], A_[i, j + 1]) if 0 < j < A_.shape[1] - 1 else np.nan
    lm = (A_ == maximum_filter(A_, 3, mode="constant", cval=-np.inf)) & np.isfinite(A_)
    yy, xx = np.mgrid[:A_.shape[0], :A_.shape[1]]
    c2 = np.where(lm & ((yy - i) ** 2 + (xx - j) ** 2 > 16), A_, -np.inf)
    i2, j2 = np.unravel_index(np.argmax(c2), c2.shape)
    edge = i in (0, A_.shape[0] - 1) or j in (0, A_.shape[1] - 1)
    return dict(dy=float(4 * (i - R + oy + sy)), dx=float(4 * (j - R + ox + sx)), ncc=float(A_[i, j]),
                second=float(c2[i2, j2]), second_at=[int(4 * (i2 - R + oy)), int(4 * (j2 - R + ox))],
                edge=bool(edge), ij=[int(i), int(j)])


Ih, Sh = hp(I, Im), hp(S, Sm)
M = nccmap(Ih, Im, Sh, Sm)
pk = peak(M)
pr = peak(nccmap(I, Im, S, Sm))
i, j = pk["ij"]
prof = {}
for k, Ak in sorted(list(depth.items()) + [(sl, I)]):
    m = Im & Sm[i:i + h, j:j + h] & (Ak > 0)
    p = hp(Ak, Ak > 0)[m].astype(float); q = Sh[i:i + h, j:j + h][m].astype(float)
    p -= p.mean(); q -= q.mean()
    prof[int(k)] = round(float((p * q).sum() / np.sqrt((p * p).sum() * (q * q).sum())), 3)
# explicit sign check: value of the map at d_opposite = -d_found (in level-2 px)
da, db = i - R + oy, j - R + ox
a2, b2 = -da - oy + R, -db - ox + R
opp = float(M[a2, b2]) if 0 <= a2 < 2 * R + 1 and 0 <= b2 < 2 * R + 1 and np.isfinite(M[a2, b2]) else None
idv = float(M[R - oy, R - ox]) if 0 <= R - oy < 2 * R + 1 and 0 <= R - ox < 2 * R + 1 else None
row = dict(key=key, y0=y0, x0=x0, n=n, R=R, win_centre_l2=[oy, ox], centre_conv=centre, rowshift=rowshift,
           slice=sl, hp=pk, raw=pr, identity_hp=idv, opposite_sign_hp=opp, depth_profile=prof,
           surf_valid=round(float(Sm.mean()), 3), img_valid=round(float(Im.mean()), 3),
           mesh_scale=[mesh.sx, mesh.sy], new_bytes=store.new_bytes() - b0, sec=round(time.time() - t0, 1))
print(json.dumps(row))
with open(os.path.join(sys.argv[1], "myresults.jsonl"), "a") as f:
    f.write(json.dumps(row) + "\n")
np.savez_compressed(os.environ.get("HF65_CHECK_DATA", os.path.abspath("hf65_check_data")) + f"/m_{key}_{y0}_{x0}_{int(centre)}_{int(rowshift)}.npz", I=I, S=S, M=M)
