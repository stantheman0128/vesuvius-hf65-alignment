"""Apply the AMENDMENT-I reading (Amendment H's w00 reading with a third seed for C0, C16y and C32y) to finished runs.

usage: python -I decide_i.py F_RUNS_DIR H_RUNS_DIR OUT_MD
F_RUNS_DIR holds Amendment F's C0, AS and C8y runs (seeds 27 and 28), H_RUNS_DIR this pod session's runs: C16y and
C32y seeds 27, 28 and 29, and C0 seed 29, each as <ARM>_s<SEED>/eval.json from eval_heldout.py. The C0_s28 run in
H_RUNS_DIR is H's session check: it is listed and enters no interval. With no seed 29 run present the numbers equal
decide_h.py's. Amendment H's own two-seed reading stays in results_H.md from decide_h.py. Writes a markdown summary
and the same as JSON next to OUT_MD. The t quantiles are the table of decide_g.py, so the script needs no scipy.
"""
import json
import math
import os
import sys

SEGS = {"w00": "w00_20231016151002", "w018": "w018_20240304144031_2um", "pooled": "pooled"}
F_ARMS, H_ARMS, CHANCE = ["C0", "AS", "C8y"], ["C16y", "C32y"], 0.5
SEEDS = [27, 28, 29]
# where each arm's seeds come from: F's folder for C0, AS and C8y seeds 27 and 28; this session for the rest
SOURCE = {("C0", 27): "f", ("C0", 28): "f", ("C0", 29): "h",
          ("AS", 27): "f", ("AS", 28): "f", ("C8y", 27): "f", ("C8y", 28): "f",
          **{(a, s): "h" for a in H_ARMS for s in SEEDS}}
SHIFT_PX = {"C0": 0, "C8y": 32, "C16y": 64, "C32y": 128}  # level-0 px in y, 4 px per cell
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

f_runs, h_runs, out_md = sys.argv[1:4]


def aucs(path):
    m = json.load(open(path))["metrics"]
    return {seg: m[key]["auc"] for seg, key in SEGS.items()}


rows = {}
for (arm, seed), src in SOURCE.items():
    try:
        rows[f"{arm}_s{seed}"] = aucs(os.path.join(f_runs if src == "f" else h_runs, f"{arm}_s{seed}", "eval.json"))
    except (OSError, ValueError, KeyError):
        pass
try:
    bridge = aucs(os.path.join(h_runs, "C0_s28", "eval.json"))
except (OSError, ValueError, KeyError):
    bridge = None


def means(seg):
    x = {a: [rows[f"{a}_s{s}"][seg] for s in SEEDS if f"{a}_s{s}" in rows] for a in F_ARMS + H_ARMS}
    return x, {a: sum(v) / len(v) for a, v in x.items() if v}


def reading(seg):
    x, m = means(seg)
    multi = {a: v for a, v in x.items() if len(v) >= 2}
    df = sum(len(v) - 1 for v in multi.values())
    out = {"mean": m, "n": {a: len(v) for a, v in x.items()}, "df": df}
    if not multi or "C0" not in m:
        out["verdict"] = "not computable: needs C0 and at least one arm with two or more seeds"
        return out
    # an arm with two seeds adds (x1 - x2)^2 / 2, written as in decide_h.py so two-seed readings match it bit for bit
    ss = sum((v[0] - v[1]) ** 2 / 2 if len(v) == 2 else sum((q - m[a]) ** 2 for q in v) for a, v in multi.items())
    s = math.sqrt(ss / df)
    t = T975[df]
    c0_low = m["C0"] - t * s / math.sqrt(len(x["C0"]))
    head = m["C0"] - CHANCE
    out.update(s=s, t=t, c0_lower=c0_low, measures=c0_low > CHANCE, headroom=head)
    out["verdict"] = ("C0 is above chance on this segment" if out["measures"] else
                      "C0 is not distinguishable from chance on this segment: arm differences get no verdict")
    out["arms"] = {}
    for a in H_ARMS:
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
        out["arms"][a] = {"d": d, "se": se, "lo": lo, "hi": hi, "n": len(x[a]),
                          "hi_over_headroom": hi / head if head > 0 else None, "verdict": v}
    pts = [(SHIFT_PX[a], m[a]) for a in SHIFT_PX if a in m]
    if len(pts) >= 3:  # dose response, reported only
        mx, my = sum(p for p, _ in pts) / len(pts), sum(q for _, q in pts) / len(pts)
        b = sum((p - mx) * (q - my) for p, q in pts) / sum((p - mx) ** 2 for p, _ in pts)
        out.update(slope_per_32px=32 * b, slope_points=pts)
    return out


res = reading("w00")
listed = {seg: means(seg)[1] for seg in ("w018", "pooled")}
json.dump({"amendment": "I", "exploratory": True, "runs": rows, "w00": res, "listed_not_judged": listed,
           "session_check_C0_s28": bridge},
          open(os.path.splitext(out_md)[0] + ".json", "w", encoding="utf-8"), indent=1)

lines = ["# Amendment I reading (exploratory; a third seed for C0, C16y and C32y, decided before any H result)", "",
         "Only w00 is judged. C0 seeds 27 and 28, AS and C8y are Amendment F's runs; C0 seed 29, C16y and C32y are",
         "this session's. Intervals are 95% Student t intervals from the pooled seed to seed variance of every arm with",
         "two or more seeds. Amendment H's two-seed reading is in results_H.md and is not replaced.", "",
         "| run | w00 AUC | w018 AUC | pooled AUC |", "|---|---|---|---|"]
for k, r in rows.items():
    lines.append(f"| {k} | {r['w00']:.4f} | {r['w018']:.4f} | {r['pooled']:.4f} |")
lines += ["", "## w00", "",
          "Means: " + (", ".join(f"{a} {v:.4f} (n={res['n'][a]})" for a, v in res["mean"].items()) or "none")]
if "s" not in res:
    lines.append(res["verdict"])
else:
    lines += [f"Seed noise s = {res['s']:.4f} (df {res['df']}), t = {res['t']:.3f}. C0 lower bound"
              f" {res['c0_lower']:.4f}, headroom above chance {res['headroom']:.4f}.", res["verdict"] + "."]
    for a, v in res["arms"].items():
        if "d" not in v:
            lines.append(f"- {a} vs C0: {v['verdict']}")
            continue
        lines.append(f"- {a} vs C0: d = C0 minus {a} = {v['d']:.4f}, 95% interval {v['lo']:.4f} to {v['hi']:.4f};"
                     f" {v['verdict']} (n = {res['n']['C0']} and {v['n']})" +
                     (f" (upper end is {v['hi_over_headroom']:.0%} of the headroom)"
                      if res["measures"] and v["hi"] > 0 else ""))
    if "slope_per_32px" in res:
        lines.append(f"Dose response (reported only): AUC change per 32 px of shift {res['slope_per_32px']:+.4f},"
                     " least squares over the arm means at " +
                     ", ".join(f"{p} px" for p, _ in res["slope_points"]) + ".")
lines += ["", "## w018 and pooled (listed, not judged)", ""]
for seg, m in listed.items():
    lines.append(f"{seg} means: " + (", ".join(f"{a} {v:.4f}" for a, v in m.items()) or "none"))
if bridge:
    ref = rows.get("C0_s28")
    lines += ["", "Session check (reported only): C0 seed 28 rerun in this session scored w00 "
              f"{bridge['w00']:.4f}" + (f" against {ref['w00']:.4f} in Amendment F (difference"
                                         f" {bridge['w00'] - ref['w00']:+.4f})." if ref else ".")]
open(out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
