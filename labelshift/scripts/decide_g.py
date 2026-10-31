"""Apply the AMENDMENT-G reading (each segment on its own, 95% t intervals from seed noise) to finished runs.

usage: python -I decide_g.py RUNS_DIR OUT_MD [L1_EVAL L2_EVAL]
RUNS_DIR/<ARM>_s<SEED>/eval.json comes from eval_heldout.py. L1_EVAL and L2_EVAL are the C0 seed 27 eval.json
files of Amendment E's L1 and L2 (optional; they give the doubling-training yardstick). Writes a markdown summary
and the same as JSON next to OUT_MD. The t quantiles are tabulated (checked against numerical integration of the
t density) so the script needs no scipy.
"""
import json
import math
import os
import sys

SEGS = {"w00": "w00_20231016151002", "w018": "w018_20240304144031_2um", "pooled": "pooled"}
ARMS, SEEDS, CHANCE = ["C0", "AS", "C8y"], [27, 28], 0.5
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}
T80 = {1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896, 8: 0.889, 9: 0.883, 10: 0.879,
       11: 0.876, 12: 0.873, 13: 0.870, 14: 0.868, 15: 0.866}
RUN_MIN, NTD_PER_H = 71, 30.29  # L2 wall clock per run (4276 s) and the RTX 5090 price

runs, out_md = sys.argv[1], sys.argv[2]
yard = sys.argv[3:5] if len(sys.argv) >= 5 else None


def aucs(path):
    m = json.load(open(path))["metrics"]
    return {seg: m[key]["auc"] for seg, key in SEGS.items()}


rows = {}
for arm in ARMS:
    for seed in SEEDS:
        try:
            rows[f"{arm}_s{seed}"] = aucs(os.path.join(runs, f"{arm}_s{seed}", "eval.json"))
        except (OSError, ValueError, KeyError):
            pass
gain = None
if yard:
    try:
        l1, l2 = aucs(yard[0]), aucs(yard[1])
        gain = {seg: l2[seg] - l1[seg] for seg in SEGS}
    except (OSError, ValueError, KeyError):
        pass


def reading(seg):
    x = {a: [rows[f"{a}_s{s}"][seg] for s in SEEDS if f"{a}_s{s}" in rows] for a in ARMS}
    m = {a: sum(v) / len(v) for a, v in x.items() if v}
    pairs = [v for v in x.values() if len(v) == 2]
    out = {"mean": m, "n": {a: len(v) for a, v in x.items()}, "df": len(pairs)}
    if not pairs or "C0" not in m:
        out["verdict"] = "not computable: needs C0 and at least one arm with both seeds"
        return out
    df = len(pairs)
    s = math.sqrt(sum((v[0] - v[1]) ** 2 / 2 for v in pairs) / df)
    t = T975[df]
    c0_low = m["C0"] - t * s / math.sqrt(len(x["C0"]))
    head = m["C0"] - CHANCE
    out.update(s=s, t=t, c0_lower=c0_low, measures=c0_low > CHANCE, headroom=head)
    out["verdict"] = ("C0 is above chance on this segment" if out["measures"] else
                      "C0 is not distinguishable from chance on this segment: arm differences get no verdict")
    out["arms"] = {}
    for a in ("AS", "C8y"):
        if a not in m:
            out["arms"][a] = {"verdict": "missing"}
            continue
        d = m["C0"] - m[a]
        se = s * math.sqrt(1 / len(x["C0"]) + 1 / len(x[a]))
        lo, hi = d - t * se, d + t * se
        if not out["measures"]:
            v = "no verdict (C0 not above chance)"
        elif lo > 0:
            v = "cost detected"
        elif hi < 0:
            v = f"{a} scored above C0 (not explained by the design)"
        else:
            v = f"no cost detected; the data rule out a cost above {hi:.4f} AUC"
        out["arms"][a] = {"d": d, "se": se, "lo": lo, "hi": hi,
                          "hi_over_headroom": hi / head if head > 0 else None, "verdict": v}
    out["mde"] = {n: (T975[3 * (n - 1)] + T80[3 * (n - 1)]) * s * math.sqrt(2 / n) for n in range(2, 7)}
    return out


res = {seg: reading(seg) for seg in SEGS}
extra = {n: {"runs": 3 * (n - 2), "hours": 3 * (n - 2) * RUN_MIN / 60,
             "ntd": round(3 * (n - 2) * RUN_MIN / 60 * NTD_PER_H)} for n in range(2, 7)}
json.dump({"amendment": "G", "exploratory": True, "runs": rows, "doubling_gain_C0_s27": gain,
           "segments": res, "extra_runs_for_n_seeds": extra},
          open(os.path.splitext(out_md)[0] + ".json", "w", encoding="utf-8"), indent=1)

f4 = lambda v: "n/a" if v is None else f"{v:.4f}"
lines = ["# Amendment G reading (exploratory; written after G0 failed, before Amendment F's new runs finished)", "",
         "Each segment is read on its own. Intervals are 95% Student t intervals from the seed to seed noise, so the",
         "only convention is the 95% level. Amendment F's own readout (results_F.md) comes first and is unchanged.", "",
         "| run | w00 AUC | w018 AUC | pooled AUC |", "|---|---|---|---|"]
for k, r in rows.items():
    lines.append(f"| {k} | {r['w00']:.4f} | {r['w018']:.4f} | {r['pooled']:.4f} |")
for seg, r in res.items():
    lines += ["", f"## {seg}" + (" (listed for continuity, not judged)" if seg == "pooled" else ""), ""]
    lines.append("Means: " + (", ".join(f"{a} {v:.4f} (n={r['n'][a]})" for a, v in r["mean"].items()) or "none"))
    if "s" not in r:
        lines.append(r["verdict"])
        continue
    lines += [f"Seed noise s = {r['s']:.4f} (df {r['df']}), t = {r['t']:.3f}. C0 lower bound {r['c0_lower']:.4f},"
              f" headroom above chance {r['headroom']:.4f}." + (f" Doubling training (C0 s27, L2 minus L1) gained"
              f" {gain[seg]:.4f}." if gain else ""), r["verdict"] + "."]
    if seg == "pooled":
        continue
    for a, v in r["arms"].items():
        if "d" not in v:
            lines.append(f"- {a} vs C0: {v['verdict']}")
            continue
        lines.append(f"- {a} vs C0: d = C0 minus {a} = {v['d']:.4f}, 95% interval {v['lo']:.4f} to {v['hi']:.4f};"
                     f" {v['verdict']}" + (f" (upper end is {v['hi_over_headroom']:.0%} of the headroom)"
                                           if r["measures"] and v["hi"] > 0 else ""))
    lines.append("Smallest cost detectable at 80% power with n seeds per arm: " +
                 ", ".join(f"n={n} {f4(v)}" for n, v in r["mde"].items()))
lines += ["", "Extra GPU time for n seeds per arm (three arms, 71 min per run, NT$30.29/h): " +
          ", ".join(f"n={n} {e['runs']} runs {e['hours']:.1f} h NT${e['ntd']}" for n, e in extra.items() if n > 2)]
open(out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
