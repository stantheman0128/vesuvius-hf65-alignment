"""Apply PREREG.md sections 5 and 7 to the finished w01 runs.

usage: python -I decide_w01.py RUNS_DIR OUT_MD

RUNS_DIR holds <ARM>_s<SEED>/eval.json (eval_heldout.py against the C0 labels in the held-out boxes,
the primary readout) and <ARM>_s<SEED>/eval_AS.json (the same against the labels as shipped, the
secondary readout) for ARM in C0, AS and SEED in 27 to 30. Writes OUT_MD and the same as JSON next
to it. The t quantiles are the table of labelshift/scripts/decide_g.py, so no scipy is needed.

Rules:
  gate     C0 seed 27 primary AUC >= 0.60, else "w01 does not learn in this setup" and no reading.
  reading  needs all four seeds in both arms. d = mean(C0) - mean(AS); s^2 = pooled seed variance
           (sum of squared deviations from each arm's mean over n_C0 + n_AS - 2 degrees of freedom);
           95% interval d +- t * s * sqrt(1/n_C0 + 1/n_AS). A cost is measured if the lower end is
           above 0. With a missing run no decision is made; the interval over the seeds present is
           shown as descriptive only.
  secondary the same means and difference on the AS-scored AUC, listed with no verdict.
"""
import json
import math
import os
import sys

SEG = "w01_20230702185753_r15"
ARMS, SEEDS, GATE, CHANCE = ("C0", "AS"), (27, 28, 29, 30), 0.60, 0.5
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201}

runs, out_md = sys.argv[1:3]


def auc(arm, seed, name):
    try:
        return json.load(open(os.path.join(runs, f"{arm}_s{seed}", name)))["metrics"][SEG]["auc"]
    except (OSError, ValueError, KeyError):
        return None


prim = {(a, s): auc(a, s, "eval.json") for a in ARMS for s in SEEDS}
sec = {(a, s): auc(a, s, "eval_AS.json") for a in ARMS for s in SEEDS}


def reading(vals):
    x = {a: [vals[(a, s)] for s in SEEDS if vals[(a, s)] is not None] for a in ARMS}
    out = {"n": {a: len(v) for a, v in x.items()}, "mean": {a: sum(v) / len(v) for a, v in x.items() if v}}
    if any(len(v) < 2 for v in x.values()):
        return out
    m = out["mean"]
    df = sum(len(v) - 1 for v in x.values())
    ss = sum((q - m[a]) ** 2 for a, v in x.items() for q in v)
    s = math.sqrt(ss / df)
    t = T975[df]
    d = m["C0"] - m["AS"]
    se = s * math.sqrt(1 / len(x["C0"]) + 1 / len(x["AS"]))
    out.update(d=d, s=s, df=df, t=t, se=se, lo=d - t * se, hi=d + t * se,
               c0_lower=m["C0"] - t * s / math.sqrt(len(x["C0"])), headroom=m["C0"] - CHANCE)
    return out


g = prim[("C0", 27)]
res = {"segment": SEG, "gate": GATE, "c0_s27": g, "primary_runs": {f"{a}_s{s}": v for (a, s), v in prim.items()},
       "secondary_runs": {f"{a}_s{s}": v for (a, s), v in sec.items()}}
lines = ["# A: w01 reading (PREREG.md sections 5 and 7)", "",
         "| run | w01 AUC on C0 labels (primary) | w01 AUC on labels as shipped (secondary) |", "|---|---|---|"]
for a in ARMS:
    for s in SEEDS:
        p, q = prim[(a, s)], sec[(a, s)]
        lines.append(f"| {a} s{s} | {'missing' if p is None else f'{p:.4f}'} | {'missing' if q is None else f'{q:.4f}'} |")
lines.append("")
if g is None:
    res["outcome"] = "gate not evaluated: C0 seed 27 has no eval.json"
elif g < GATE:
    res["outcome"] = f"w01 does not learn in this setup (C0 seed 27 AUC {g:.4f} < {GATE})"
else:
    r = reading(prim)
    res["primary"] = r
    complete = all(r["n"][a] == len(SEEDS) for a in ARMS)
    if "d" not in r:
        res["outcome"] = f"gate passed ({g:.4f}); not enough runs for an interval: n = {r['n']}"
    else:
        span = (f"d = mean(C0) - mean(AS) = {r['d']:.4f}, 95% interval {r['lo']:.4f} to {r['hi']:.4f} "
                f"(s = {r['s']:.4f}, df {r['df']}, t = {r['t']:.3f}, n = {r['n']['C0']} and {r['n']['AS']})")
        if not complete:
            res["outcome"] = f"gate passed ({g:.4f}); incomplete, no decision. Descriptive only: {span}"
        elif r["lo"] > 0:
            res["outcome"] = f"cost measured: {span}"
        elif r["hi"] < 0:
            res["outcome"] = f"AS scored above C0 (not explained by the design): {span}"
        else:
            res["outcome"] = f"no cost measured; the data rule out a cost above {r['hi']:.4f} AUC: {span}"
        res["primary_complete"] = complete
    lines += [f"Means on C0 labels: " + ", ".join(f"{a} {v:.4f} (n={r['n'][a]})" for a, v in r["mean"].items())]
    if "c0_lower" in r:
        lines.append(f"C0 lower bound {r['c0_lower']:.4f}, headroom above chance {r['headroom']:.4f} (reported only).")
    r2 = reading(sec)
    res["secondary"] = r2
    lines.append("Secondary, labels as shipped (reported only, no verdict): " +
                 ", ".join(f"{a} {v:.4f} (n={r2['n'][a]})" for a, v in r2["mean"].items()) +
                 (f"; d = {r2['d']:.4f}, interval {r2['lo']:.4f} to {r2['hi']:.4f}" if "d" in r2 else ""))
lines += ["", f"Outcome: {res['outcome']}."]
json.dump(res, open(os.path.splitext(out_md)[0] + ".json", "w", encoding="utf-8"), indent=1)
open(out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
