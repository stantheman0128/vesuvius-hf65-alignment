"""Apply the AMENDMENT-F rules (exploratory, w00 only) to finished runs.

usage: python -I decide_f.py RUNS_DIR OUT_MD
RUNS_DIR/<ARM>_s<SEED>/eval.json comes from eval_heldout.py. Primary metric: the w00 held-out AUC; pooled
and w018 AUCs are reported as secondary. Writes a markdown summary and the same as JSON next to OUT_MD.
"""
import json
import math
import os
import sys

W00, W018 = "w00_20231016151002", "w018_20240304144031_2um"
ARMS, SEEDS, FLOOR = ["C0", "AS", "C8y"], [27, 28], 0.01

runs, out_md = sys.argv[1], sys.argv[2]
rows = {}
for arm in ARMS:
    for seed in SEEDS:
        try:
            m = json.load(open(os.path.join(runs, f"{arm}_s{seed}", "eval.json")))["metrics"]
            rows[f"{arm}_s{seed}"] = {"arm": arm, "seed": seed, "w00": m[W00]["auc"],
                                       "pooled": m["pooled"]["auc"], "w018": m[W018]["auc"]}
        except (OSError, ValueError, KeyError):
            pass

complete = [a for a in ARMS if all(f"{a}_s{s}" in rows for s in SEEDS)]
missing = [a for a in ARMS if a not in complete]
mean = {a: sum(rows[f"{a}_s{s}"]["w00"] for s in SEEDS) / len(SEEDS) for a in complete}
sigma = (math.sqrt(sum((rows[f"{a}_s27"]["w00"] - rows[f"{a}_s28"]["w00"]) ** 2 / 2 for a in complete) / len(complete))
         if complete else None)
delta = max(2 * sigma, FLOOR) if sigma is not None else None


def versus_c0(arm):
    if delta is None or "C0" not in mean or arm not in mean:
        return "missing"
    d = mean["C0"] - mean[arm]
    if d > delta:
        return "COST"
    if -d > delta:
        return "unexplained (above C0)"
    return "NO COST"


r1, ras = versus_c0("C8y"), versus_c0("AS")
out = {"amendment": "F", "exploratory": True, "g0": "failed in attempt 0, L1 and L2; waived under Amendment F",
       "primary": "w00 held-out voxel ROC AUC", "runs": rows, "complete_arms": complete, "missing_arms": missing,
       "mean_w00": mean, "sigma": sigma, "delta": delta, "C8y_vs_C0": r1, "AS_vs_C0": ras}

lines = ["# Amendment F readout (exploratory, post hoc)", "",
         "G0 failed in every attempt (attempt 0, L1, L2). Amendment F was added afterwards and waives it. w018 did not",
         "learn (AUC near 0.5), so the primary metric is the w00 held-out AUC. Pooled and w018 are secondary.", "",
         "| run | w00 AUC | pooled AUC | w018 AUC |", "|---|---|---|---|"]
for k, r in rows.items():
    lines.append(f"| {k} | {r['w00']:.4f} | {r['pooled']:.4f} | {r['w018']:.4f} |")
lines += ["", "Means of the w00 AUC over seeds 27 and 28: " + (", ".join(f"{a} {v:.4f}" for a, v in mean.items()) or "none"),
          f"Missing arms (left out): {', '.join(missing) or 'none'}",
          f"sigma = {sigma:.4f}, delta = {delta:.4f}" if delta is not None else "sigma and delta: not computable", "",
          f"- C8y vs C0: {r1}" + (" (no measurable cost of a 32 px shift on w00 in this setting, with a base model near"
                                  " 0.69 AUC)" if r1 == "NO COST" else ""),
          f"- AS vs C0 (reported, does not decide COST): {ras}"]
open(out_md, "w").write("\n".join(lines) + "\n")
json.dump(out, open(os.path.splitext(out_md)[0] + ".json", "w"), indent=1)
print("\n".join(lines))
