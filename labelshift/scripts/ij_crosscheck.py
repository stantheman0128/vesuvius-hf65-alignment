"""Independent recompute of the Amendment I (seeds 27 to 29) or J (seeds 27 to 30) w00 reading, written apart from
decide_i.py and decide_j.py, to hand-check results_I.json or results_J.json.

usage: python -I ij_crosscheck.py F_RUNS H_RUNS I_RUNS J_RUNS_OR_DASH RESULTS_JSON
Seeds 27 and 28 of C0, AS and C8y come from F_RUNS, C16y and C32y seeds 27 and 28 from H_RUNS (H's C0_s28 session
check is not used), every seed 29 run from I_RUNS and seed 30 from J_RUNS (pass - for the three-seed reading). On the
pod H's and I's runs share one folder, so H_RUNS and I_RUNS can be the same path. The seed variance is
pooled as the summed squared deviations of every arm with two or more seeds over the summed df. The t table is its
own, the slope is numpy-free least squares on the arm means, and both readings are printed side by side.
"""
import json
import math
import os
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
f_runs, h_runs, i_runs, j_runs, res_path = sys.argv[1:6]
T = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622,
     10: 2.2281, 11: 2.2010, 12: 2.1788}
SEG = "w00_20231016151002"
PX = {"C0": 0, "C8y": 32, "C16y": 64, "C32y": 128}

src = {(a, s): f_runs for a in ("C0", "AS", "C8y") for s in (27, 28)}
src.update({(a, s): h_runs for a in ("C16y", "C32y") for s in (27, 28)})
src.update({(a, 29): i_runs for a in ("C0", "C16y", "C32y")})
if j_runs != "-":
    src.update({(a, 30): j_runs for a in ("C0", "C16y", "C32y")})

vals, used = {}, []
for (a, s), d in sorted(src.items()):
    p = os.path.join(d, f"{a}_s{s}", "eval.json")
    if os.path.exists(p):
        vals.setdefault(a, []).append(json.load(open(p))["metrics"][SEG]["auc"])
        used.append(f"{a}_s{s}")

multi = [v for v in vals.values() if len(v) >= 2]
df = sum(len(v) - 1 for v in multi)
s = math.sqrt(sum(sum((x - statistics.mean(v)) ** 2 for x in v) for v in multi) / df)
t = T[df]
mean = {a: statistics.mean(v) for a, v in vals.items()}
head = mean["C0"] - 0.5
mine = {"df": df, "s": s, "t": t, "c0_lower": mean["C0"] - t * s / math.sqrt(len(vals["C0"])), "headroom": head}
for a in ("AS", "C8y", "C16y", "C32y"):
    if a in mean:
        d = mean["C0"] - mean[a]
        se = s * math.sqrt(1 / len(vals["C0"]) + 1 / len(vals[a]))
        mine[a] = {"d": d, "lo": d - t * se, "hi": d + t * se}
pts = [(PX[a], mean[a]) for a in PX if a in mean]
n = len(pts)
sx, sy = sum(p for p, _ in pts), sum(q for _, q in pts)
sxx, sxy = sum(p * p for p, _ in pts), sum(p * q for p, q in pts)
mine["slope_per_32px"] = 32 * (n * sxy - sx * sy) / (n * sxx - sx * sx)

R = json.load(open(res_path, encoding="utf-8"))["w00"]
theirs = {k: R.get(k) for k in ("df", "s", "t", "c0_lower", "headroom", "slope_per_32px")}
for a in ("AS", "C8y", "C16y", "C32y"):
    v = R.get("arms", {}).get(a, {})
    if "d" in v:
        theirs[a] = {k: v[k] for k in ("d", "lo", "hi")}

print("runs used:", " ".join(used))
print("seeds per arm:", {a: len(v) for a, v in vals.items()})
print("means:", {a: round(v, 4) for a, v in mean.items()})
worst = 0.0
for k, a in mine.items():
    b = theirs.get(k)
    if isinstance(a, dict):
        for kk in a:
            diff = abs(a[kk] - b[kk]) if b else float("nan")
            worst = max(worst, diff) if b else worst
            print(f"{k}.{kk}: mine {a[kk]:+.5f}  readout {b[kk] if b else 'MISSING'}  diff {diff:.1e}")
    else:
        diff = abs(a - b) if b is not None else float("nan")
        worst = max(worst, diff) if b is not None else worst
        print(f"{k}: mine {a:+.5f}  readout {b}  diff {diff:.1e}")
for a in ("C16y", "C32y"):
    if a in mine:
        print(f"{a}: upper end is {mine[a]['hi'] / head:.0%} of the headroom; lower end above 0: {mine[a]['lo'] > 0}")
print(f"largest difference {worst:.1e} (t tables differ in the 4th decimal, so up to about 1e-4 on bounds is expected)")
