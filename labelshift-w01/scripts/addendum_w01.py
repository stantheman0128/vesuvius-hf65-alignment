"""Exploratory addendum to A: the four pre-registered seeds plus seeds 31 and 32.

usage: python -I addendum_w01.py RUNS_DIR EXTRA_DIR OUT_MD

RUNS_DIR is results/runs (seeds 27 to 30, the runs PREREG.md section 7 decided on) and EXTRA_DIR is
results/extra (seeds 31 and 32, run after the result was published). Both hold
<ARM>_s<SEED>/eval.json (primary, AUC against the C0 labels) and eval_AS.json (secondary, AUC against
the labels as shipped in the held-out boxes).

The interval uses the formula of decide_w01.py: pooled seed variance over n_C0 + n_AS - 2 degrees of
freedom and d +- t * s * sqrt(1/n_C0 + 1/n_AS). The four-seed line reproduces results.md. The six-seed
line is descriptive only: seeds 31 and 32 were added after the outcome was known, so they do not change
the section 7 decision. Writes OUT_MD and the same as JSON next to it.
"""
import json
import math
import os
import sys

SEG = "w01_20230702185753_r15"
ARMS = ("C0", "AS")
PREREG_SEEDS, EXTRA_SEEDS = (27, 28, 29, 30), (31, 32)
T975 = {6: 2.447, 8: 2.306, 10: 2.228}

runs, extra, out_md = sys.argv[1:4]


def folder(seed):
    return runs if seed in PREREG_SEEDS else extra


def read(arm, seed, name):
    e = json.load(open(os.path.join(folder(seed), f"{arm}_s{seed}", name)))
    m = e["metrics"][SEG]
    return {"auc": m["auc"], "n_patches": e["n_patches"], "n_pos": m["n_pos"], "n_neg": m["n_neg"]}


seeds = PREREG_SEEDS + EXTRA_SEEDS
prim = {(a, s): read(a, s, "eval.json") for a in ARMS for s in seeds}
sec = {(a, s): read(a, s, "eval_AS.json") for a in ARMS for s in seeds}


def reading(vals, use):
    x = {a: [vals[(a, s)]["auc"] for s in use] for a in ARMS}
    m = {a: sum(v) / len(v) for a, v in x.items()}
    df = sum(len(v) - 1 for v in x.values())
    s = math.sqrt(sum((q - m[a]) ** 2 for a, v in x.items() for q in v) / df)
    t = T975[df]
    d = m["C0"] - m["AS"]
    se = s * math.sqrt(1 / len(x["C0"]) + 1 / len(x["AS"]))
    return {"seeds": list(use), "mean": m, "d": d, "s": s, "df": df, "t": t, "lo": d - t * se, "hi": d + t * se}


res = {"segment": SEG, "primary": {}, "secondary": {}}
lines = ["# A, exploratory addendum: seeds 27 to 32", "",
         "| run | primary: AUC on C0 labels | secondary: AUC on labels as shipped | primary n_pos / n_neg |",
         "|---|---|---|---|"]
for a in ARMS:
    for s in seeds:
        p, q = prim[(a, s)], sec[(a, s)]
        tag = "" if s in PREREG_SEEDS else " (added)"
        lines.append(f"| {a} s{s}{tag} | {p['auc']:.4f} | {q['auc']:.4f} | {p['n_pos']} / {p['n_neg']} |")
lines.append("")
for key, vals in (("primary", prim), ("secondary", sec)):
    for label, use in (("seeds 27 to 30, pre-registered", PREREG_SEEDS), ("seeds 27 to 32, descriptive", seeds)):
        r = reading(vals, use)
        res[key][label] = r
        lines.append(f"{key.capitalize()}, {label}: C0 {r['mean']['C0']:.4f}, AS {r['mean']['AS']:.4f}, "
                     f"d = {r['d']:.4f}, 95% interval {r['lo']:.4f} to {r['hi']:.4f} "
                     f"(s = {r['s']:.4f}, df {r['df']}, t = {r['t']:.3f})")
same_seed = {s: prim[("C0", s)]["auc"] - prim[("AS", s)]["auc"] for s in seeds}
res["same_seed_d_primary"] = same_seed
lines += ["", "Same-seed difference C0 - AS on the primary AUC: " +
          ", ".join(f"s{s} {v:.4f}" for s, v in same_seed.items()) + ".",
          f"Lowest C0 run {min(prim[('C0', s)]['auc'] for s in seeds):.4f}, "
          f"highest AS run {max(prim[('AS', s)]['auc'] for s in seeds):.4f}."]
json.dump(res, open(os.path.splitext(out_md)[0] + ".json", "w", encoding="utf-8", newline="\n"), indent=1)
open(out_md, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("\n".join(lines))
