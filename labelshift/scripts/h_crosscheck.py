"""Independent recompute of the Amendment H w00 reading, written apart from decide_h.py, to hand-check results_H.json.

usage: py -I h_crosscheck.py F_RUNS H_RUNS RESULTS_H_JSON
Uses statistics.variance for the per-arm seed variance (pooled over arms with two seeds), a separate t table
(df 1 to 6) and numpy-free least squares, then prints both readings side by side and the largest difference.
"""
import json
import math
import os
import statistics
import sys

f_runs, h_runs, res_path = sys.argv[1:4]
T = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469}
SRC = {"C0": f_runs, "AS": f_runs, "C8y": f_runs, "C16y": h_runs, "C32y": h_runs}
PX = {"C0": 0, "C8y": 32, "C16y": 64, "C32y": 128}

w00 = {}
for arm, d in SRC.items():
    for seed in (27, 28):
        p = os.path.join(d, f"{arm}_s{seed}", "eval.json")
        if os.path.exists(p):
            w00.setdefault(arm, []).append(json.load(open(p))["metrics"]["w00_20231016151002"]["auc"])

two = [v for v in w00.values() if len(v) == 2]
df = len(two)
s = math.sqrt(statistics.mean(statistics.variance(v) for v in two))  # variance of 2 values = (a-b)^2/2
t = T[df]
mean = {a: statistics.mean(v) for a, v in w00.items()}
mine = {"df": df, "s": s, "t": t, "c0_lower": mean["C0"] - t * s / math.sqrt(len(w00["C0"]))}
for a in ("C16y", "C32y"):
    if a in mean:
        d = mean["C0"] - mean[a]
        se = s * math.sqrt(1 / len(w00["C0"]) + 1 / len(w00[a]))
        mine[a] = {"d": d, "lo": d - t * se, "hi": d + t * se}
pts = [(PX[a], mean[a]) for a in PX if a in mean]
if len(pts) >= 3:
    n = len(pts)
    sx, sy = sum(p for p, _ in pts), sum(q for _, q in pts)
    sxx, sxy = sum(p * p for p, _ in pts), sum(p * q for p, q in pts)
    mine["slope_per_32px"] = 32 * (n * sxy - sx * sy) / (n * sxx - sx * sx)

R = json.load(open(res_path))["w00"]
theirs = {"df": R.get("df"), "s": R.get("s"), "t": R.get("t"), "c0_lower": R.get("c0_lower"),
          "slope_per_32px": R.get("slope_per_32px")}
for a in ("C16y", "C32y"):
    v = R.get("arms", {}).get(a, {})
    if "d" in v:
        theirs[a] = {k: v[k] for k in ("d", "lo", "hi")}

print("w00 per run:", {a: [round(x, 4) for x in v] for a, v in w00.items()})
print("means:", {a: round(v, 4) for a, v in mean.items()})
worst = 0.0
for k in mine:
    a, b = mine[k], theirs.get(k)
    if isinstance(a, dict):
        for kk in a:
            diff = abs(a[kk] - b[kk]) if b else float("nan")
            worst = max(worst, diff) if b else worst
            print(f"{k}.{kk}: mine {a[kk]:+.5f}  decide_h {b[kk] if b else 'MISSING'}  diff {diff:.1e}")
    else:
        diff = abs(a - b) if b is not None else float("nan")
        worst = max(worst, diff) if b is not None else worst
        print(f"{k}: mine {a:+.5f}  decide_h {b}  diff {diff:.1e}")
print(f"largest difference {worst:.1e} (t tables differ in the 4th decimal, so up to about 1e-4 on bounds is expected)")
