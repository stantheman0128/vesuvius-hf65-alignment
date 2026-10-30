"""Before/after numbers for MAN5_outer_3_inklabels placed on the segment grid, against
front_inklabels (slice 32, value > 0 = ink), at level 1 and level 2.

For level k a table over integer level-k shifts holds |front & placed| and |placed| (placed = padded
and cropped to the grid, as place.py does). A level-0 offset (dy, dx) is a shift of (dy, dx) / 2^k
level-k px; when that is fractional, its counts are the bilinear interpolation of the table. That is
exactly the count obtained by upsampling both level-k labels to level-0 px (nearest) and shifting by
the level-0 offset (a fraction r/2^k of the sub-rows takes the next whole shift). Integer level-k
shifts are cross-checked against place.place() output.

Level 0: a window around dense ink is read at level 0 (HTTP range reads, cached, ledgered) and the
IoU is computed over level-0 offsets near the best.
argv: [--window-size N] [--no-level0]"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "measure"))
import numpy as np
import place
from place import Lab, grid_shape

Z = 32
CAND = [("no shift", 0, 0), ("centred crop", 29, 29), ("centred crop", 29, 30), ("centred crop", 30, 29),
        ("centred crop", 30, 30), ("offset.py best (level-1 15, 17)", 30, 34)]


def load(k):
    GH, GW = grid_shape(k)
    F = Lab("front", k).slice2d(Z, 0, GH, 0, GW) > 0
    Lm = Lab("inklabels", k)
    M = Lm.slice2d(Z, 0, Lm.shape[1], 0, Lm.shape[2]) > 0
    return F, M, (GH, GW), Lm.shape[1:]


def shift_table(F, M, grid, lo, hi):
    """T[sy - lo, sx - lo] = (|F & placed M|, |placed M|) for integer shifts lo..hi."""
    GH, GW = grid
    ys, xs = np.nonzero(M)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    Mb = M[y0:y1, x0:x1]
    pad = max(abs(lo), abs(hi)) + 1
    Fp = np.zeros((GH + 2 * pad, GW + 2 * pad), bool)
    Fp[pad:pad + GH, pad:pad + GW] = F
    Gp = np.zeros_like(Fp)
    Gp[pad:pad + GH, pad:pad + GW] = True
    n = hi - lo + 1
    inter = np.zeros((n, n), np.int64)
    mp = np.zeros((n, n), np.int64)
    rows_in = None
    for i, sy in enumerate(range(lo, hi + 1)):
        for j, sx in enumerate(range(lo, hi + 1)):
            a, b = y0 + sy + pad, x0 + sx + pad
            inter[i, j] = np.count_nonzero(Mb & Fp[a:a + Mb.shape[0], b:b + Mb.shape[1]])
            mp[i, j] = np.count_nonzero(Mb & Gp[a:a + Mb.shape[0], b:b + Mb.shape[1]])
    return inter, mp


def interp(T, lo, sy, sx):
    qy, qx = int(np.floor(sy)), int(np.floor(sx))
    fy, fx = sy - qy, sx - qx
    i, j = qy - lo, qx - lo
    v =(1 - fy) * (1 - fx) * T[i, j]
    if fy:
        v += fy * (1 - fx) * T[i + 1, j]
    if fx:
        v += (1 - fy) * fx * T[i, j + 1]
    if fy and fx:
        v += fy * fx * T[i + 1, j + 1]
    return float(v)


def metrics(inter, mp, nf):
    return {"iou": inter / (nf + mp - inter), "share_man5_on_front": inter / mp, "share_front_on_man5": inter / nf,
            "inter": inter, "man5_px": mp, "front_px": nf}


def main(do_l0=True, wsize=2048):
    out = {"levels": {}}
    best0 = {}
    for k, (lo, hi) in ((1, (-4, 24)), (2, (-2, 12))):
        t0 = time.time()
        F, M, grid, mshape = load(k)
        nf = int(F.sum())
        inter, mp = shift_table(F, M, grid, lo, hi)
        f = 2 ** k
        print(f"\n## level {k}: grid {grid}, MAN5 label {mshape}; front ink {nf} px, MAN5 ink {int(M.sum())} px"
              f" ({M.sum() / nf:.2f} of front); table {hi - lo + 1}^2 shifts in {time.time() - t0:.0f} s; chunks {place.STATS}")
        ys_, xs_ = np.nonzero(M)
        mg = 80 // f
        box = (max(ys_.min() - mg, 0), min(ys_.max() + 1 + hi + mg, grid[0]), max(xs_.min() - mg, 0), min(xs_.max() + 1 + hi + mg, grid[1]))
        nf_box = int(F[box[0]:box[1], box[2]:box[3]].sum())
        print(f"   box for iou_box (level-{k} px): y {box[0]}..{box[1]}, x {box[2]}..{box[3]}; front ink inside {nf_box} px ({nf_box / nf:.3f} of all)")
        iou_t = inter / (nf + mp - inter)
        i, j = np.unravel_index(np.argmax(iou_t), iou_t.shape)
        print(f"   best integer level-{k} shift ({i + lo}, {j + lo}) = level-0 ({f * (i + lo)}, {f * (j + lo)}): IoU {iou_t[i, j]:.4f}")
        # every level-0 integer offset the table covers
        best = (-1, 0, 0)
        for dy in range(f * lo, f * hi + 1):
            for dx in range(f * lo, f * hi + 1):
                v = metrics(interp(inter, lo, dy / f, dx / f), interp(mp, lo, dy / f, dx / f), nf)["iou"]
                if v > best[0]:
                    best = (v, dy, dx)
        best0[k] = best
        print(f"   best level-0 offset with level-{k} data (upsampled, interpolated table): ({best[1]}, {best[2]}) IoU {best[0]:.4f}")
        # IoU profile along each axis through the best, level-0 offsets
        print("   IoU vs dy at dx=%d: " % best[2] + " ".join(f"{dy}:{metrics(interp(inter, lo, dy / f, best[2] / f), interp(mp, lo, dy / f, best[2] / f), nf)['iou']:.4f}" for dy in range(best[1] - 6, best[1] + 7, 2)))
        print("   IoU vs dx at dy=%d: " % best[1] + " ".join(f"{dx}:{metrics(interp(inter, lo, best[1] / f, dx / f), interp(mp, lo, best[1] / f, dx / f), nf)['iou']:.4f}" for dx in range(best[2] - 6, best[2] + 7, 2)))
        rows = []
        cands = CAND + [(f"best at level {k}", best[1], best[2])]
        for nm, dy, dx in cands:
            m = metrics(interp(inter, lo, dy / f, dx / f), interp(mp, lo, dy / f, dx / f), nf)
            # IoU inside a fixed box on the grid: MAN5 ink bounding box (label coords) widened by
            # 80 level-0 px and by the largest shift tested; all placed MAN5 ink lies inside it,
            # front ink outside it is not counted (the window idea of the offset check in villa#2369)
            m["iou_box"] = m["inter"] / (nf_box + m["man5_px"] - m["inter"])
            exact = (dy % f == 0 and dx % f == 0)
            chk = ""
            if exact:
                P, info = place.place("inklabels", k, dy, dx, z=Z)
                Pb = P > 0
                mi = int((Pb & F).sum())
                mpp = int(Pb.sum())
                chk = f"place.py: inter {mi} placed {mpp} ({'same' if (mi, mpp) == (m['inter'], m['man5_px']) else 'DIFFERENT'})"
            rows.append({"name": nm, "dy": dy, "dx": dx, "levelk_shift": [dy / f, dx / f], **m, "check": chk})
            print(f"   {nm:34s} level-0 ({dy:2d}, {dx:2d}) level-{k} ({dy / f:g}, {dx / f:g}): IoU {m['iou']:.4f}, IoU in box {m['iou_box']:.4f},"
                  f" MAN5 ink on front ink {m['share_man5_on_front']:.4f}, front ink on MAN5 ink {m['share_front_on_man5']:.4f}"
                  f" (inter {m['inter']:.1f}, MAN5 {m['man5_px']:.1f}, front {nf}) {chk}")
        out["levels"][k] = {"front_px": nf, "man5_px": int(M.sum()), "rows": rows,
                            "best_levelk_integer": [int(i + lo), int(j + lo), float(iou_t[i, j])],
                            "best_level0": list(best)}
    if do_l0:
        out["level0"] = level0(best0[1], wsize)
    json.dump(out, open(os.path.join(HERE, "..", "results", "man5_eval.json"), "w"), indent=1, default=float)
    print("\nchunks read", place.STATS)


def level0(best, n):
    """IoU at level 0 in an n x n window of dense MAN5 ink, over offsets near the level-1 best."""
    F2, M2, grid2, _ = load(2)
    # densest n x n (level-0) block of MAN5 ink at level 2 (on the placed best position)
    b = n // 4
    S = np.cumsum(np.cumsum(np.pad(M2.astype(np.int64), ((1, 0), (1, 0))), 0), 1)
    dens = S[b:, b:] - S[:-b, b:] - S[b:, :-b] + S[:-b, :-b]
    i, j = np.unravel_index(np.argmax(dens[::8, ::8]), dens[::8, ::8].shape)
    my0, mx0 = 4 * 8 * i, 4 * 8 * j            # level-0 corner in label coordinates
    _, bdy, bdx = best
    Y0, X0 = my0 + bdy, mx0 + bdx               # the same block on the segment grid
    W = (Y0, Y0 + n, X0, X0 + n)
    t0 = time.time()
    import common
    if os.environ.get("HF65_LEDGER_TAG"):
        common.LEDGER = os.path.join(common.DATA, f"ledger_{os.environ['HF65_LEDGER_TAG']}.tsv")
    l0 = common.ledger_total()
    F = Lab("front", 0).slice2d(Z, *W) > 0
    R = 64
    Lm = Lab("inklabels", 0)
    M = Lm.slice2d(Z, Y0 - R - 1, Y0 + n + R, X0 - R - 1, X0 + n + R) > 0   # label coords = grid - offset
    dl = common.ledger_total() - l0
    print(f"\n## level 0 window: segment grid y {Y0}..{Y0 + n}, x {X0}..{X0 + n} (label block at {my0},{mx0});"
          f" front ink {int(F.sum())} px; read in {time.time() - t0:.0f} s, {dl / 1e6:.2f} MB new; chunks {place.STATS}")
    nf = int(F.sum())

    def m_at(dy, dx):
        # placed[Y, X] = M[Y - dy, X - dx]; M array index of label row (Y0 - R - 1)
        a, c = R + 1 - dy, R + 1 - dx
        P = M[a:a + n, c:c + n]
        inter = int((P & F).sum())
        mp = int(P.sum())
        return metrics(inter, mp, nf)

    rows = []
    grid = {}
    for dy in range(-4, R + 2):
        for dx in range(-4, R + 2):
            grid[(dy, dx)] = m_at(dy, dx)["iou"]
    (by, bx), bv = max(grid.items(), key=lambda t: t[1])
    edge = by in (-4, R + 1) or bx in (-4, R + 1)
    print(f"   best level-0 offset in the window (search -4..{R + 1} on both axes): ({by}, {bx}) IoU {bv:.4f}, on search edge: {edge}")
    print("   IoU vs dy at dx=%d: " % bx + " ".join(f"{dy}:{grid[(dy, bx)]:.4f}" for dy in range(-4, R + 2, 4)))
    print("   IoU vs dx at dy=%d: " % by + " ".join(f"{dx}:{grid[(by, dx)]:.4f}" for dx in range(-4, R + 2, 4)))
    for nm, dy, dx in CAND + [("best in window", by, bx)]:
        if not (-R <= dy <= R + 1 and -R <= dx <= R + 1):
            continue
        m = m_at(dy, dx)
        rows.append({"name": nm, "dy": dy, "dx": dx, **m})
        print(f"   {nm:34s} level-0 ({dy:2d}, {dx:2d}): IoU {m['iou']:.4f}, MAN5 ink on front ink {m['share_man5_on_front']:.4f}"
              f" (inter {m['inter']}, MAN5 {m['man5_px']}, front {nf})")
    return {"window": W, "best": [by, bx, bv], "rows": rows, "new_bytes": dl}


if __name__ == "__main__":
    a = sys.argv[1:]
    ws = int(a[a.index("--window-size") + 1]) if "--window-size" in a else 2048
    main(do_l0="--no-level0" not in a, wsize=ws)
