"""Tables from results_<key>.jsonl: per-region offsets, and per segment a constant model vs an
affine model  d = a + b*y + c*x  (y, x = region centre, level-0 canvas px), each axis separately.
argv: key ..."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(key):
    rows = {}
    for line in open(os.path.join(os.environ.get("HF65_RESULTS", os.path.join(HERE, "..", "results")), f"results_{key}.jsonl")):
        r = json.loads(line)
        rows[r["region"]] = r  # last run of a region wins
    return [rows[k] for k in sorted(rows)]


for key in sys.argv[1:]:
    rs = load(key)
    print(f"\n## {key}  ({len(rs)} regions)")
    print("| region | centre (y, x) | dy | dx | peak NCC | 2nd peak NCC (at dy,dx, level-2 px) | raw NCC | best slice |")
    print("|---|---|---|---|---|---|---|---|")
    Y, X, DY, DX = [], [], [], []
    for r in rs:
        cy, cx = r["y0"] + r["size"] / 2, r["x0"] + r["size"] / 2
        h = r["hp"]
        Y.append(cy); X.append(cx); DY.append(r["level0_dy"]); DX.append(r["level0_dx"])
        print(f"| {r['region']} | {cy:.0f}, {cx:.0f} | {r['level0_dy']:+.1f} | {r['level0_dx']:+.1f} | {h['ncc']:.3f} | "
              f"{h['second']:.3f} ({h['second_dy']},{h['second_dx']}) | {r['raw']['ncc']:.3f} | {r['best_slice']} |")
    Y, X, DY, DX = map(np.array, (Y, X, DY, DX))
    print(f"constant model: dy mean {DY.mean():+.2f} sd {DY.std(ddof=1):.2f} range [{DY.min():+.1f}, {DY.max():+.1f}];"
          f" dx mean {DX.mean():+.2f} sd {DX.std(ddof=1):.2f} range [{DX.min():+.1f}, {DX.max():+.1f}]")
    if len(rs) >= 4:
        A = np.stack([np.ones_like(Y), Y, X], 1)
        for nm, d in (("dy", DY), ("dx", DX)):
            coef, *_ = np.linalg.lstsq(A, d, rcond=None)
            res = d - A @ coef
            print(f"affine {nm} = {coef[0]:+.2f} {coef[1]:+.6f}*y {coef[2]:+.6f}*x ; residuals {np.round(res, 2).tolist()}"
                  f" (rms {np.sqrt((res ** 2).mean()):.2f})")
