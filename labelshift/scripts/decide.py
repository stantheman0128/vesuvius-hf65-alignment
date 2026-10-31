"""Apply the PREREG.md section 7 rules to finished runs.

usage: python -I decide.py RUNS_DIR OUT_MD [--core AS,C0,C2y,C4y,C8y] [--xarms C2x,C4x,C8x]
RUNS_DIR/<ARM>_s<SEED>/eval.json comes from eval_heldout.py; validation_metrics.jsonl from the trainer.
Writes a markdown table and the rule outcomes, and the same as JSON next to OUT_MD.
"""
import argparse
import glob
import json
import math
import os

GATE_AUC = 0.70
FLOOR = 0.01
SHIFT_PX = {"C0": 0, "C2": 8, "C4": 16, "C8": 32}


def load(runs):
    res = {}
    for d in sorted(glob.glob(os.path.join(runs, "*_s*"))):
        name = os.path.basename(d)
        if name.startswith("eval_") or not os.path.isdir(d):
            continue
        arm, seed = name.rsplit("_s", 1)
        row = {"arm": arm, "seed": int(seed)}
        try:
            e = json.load(open(os.path.join(d, "eval.json")))
            row["auc"] = e["metrics"]["pooled"].get("auc")
            row["f05_best"] = e["metrics"]["pooled"].get("f05_best")
            row["per_segment_auc"] = {k: v.get("auc") for k, v in e["metrics"].items() if k != "pooled"}
        except (OSError, KeyError, ValueError):
            row["auc"] = None
        try:
            lines = open(os.path.join(d, "out", "validation_metrics.jsonl")).read().strip().splitlines()
            row["trainer_val_balanced_accuracy"] = json.loads(lines[-1])["val_balanced_accuracy"]
        except (OSError, IndexError, KeyError, ValueError):
            row["trainer_val_balanced_accuracy"] = None
        res.setdefault(arm, {})[int(seed)] = row
    return res


def curve_outcome(res, arms, axis):
    complete = {a: [res[a][s]["auc"] for s in sorted(res.get(a, {})) if res[a][s]["auc"] is not None]
                for a in arms}
    complete = {a: v for a, v in complete.items() if len(v) >= 2}
    out = {"complete_arms": sorted(complete)}
    if not complete:
        out["status"] = "no complete arms"
        return out
    sigma = math.sqrt(sum((v[0] - v[1]) ** 2 / 2 for v in complete.values()) / len(complete))
    delta = max(2 * sigma, FLOOR)
    m = {a: sum(v[:2]) / 2 for a, v in complete.items()}
    out.update(sigma=sigma, delta=delta, means=m)
    c8 = f"C8{axis}"
    if "C0" not in m or c8 not in m:
        out["R1"] = "not decidable (C0 or C8 missing)"
        return out
    d8 = m["C0"] - m[c8]
    if d8 > delta:
        out["R1"] = f"COST: m(C0) - m({c8}) = {d8:.4f} > delta {delta:.4f}"
        seq = [a for a in ("C0", f"C2{axis}", f"C4{axis}", c8) if a in m]
        mono = all(m[seq[i + 1]] <= m[seq[i]] + delta for i in range(len(seq) - 1))
        out["R2"] = f"{'monotone' if mono else 'not monotone'} within delta over {seq}"
    elif abs(d8) <= delta:
        out["R1"] = f"NO COST: |m(C0) - m({c8})| = {abs(d8):.4f} <= delta {delta:.4f}; no measurable cost of a 32 px shift in this setting"
    else:
        out["R1"] = f"UNEXPLAINED: m({c8}) - m(C0) = {-d8:.4f} > delta; other arms not interpreted"
        return out
    if axis == "y" and "AS" in m:
        d_as = m["C0"] - m["AS"]
        if abs(d_as) <= delta:
            out["R3"] = f"d_AS = {d_as:.4f}, |d_AS| <= delta {delta:.4f}: cost of the shipped offset not distinguishable from zero here"
        else:
            pts = [(SHIFT_PX[a[:2]], m[a]) for a in ("C0", "C2y", "C4y", "C8y") if a in m]
            eq = None
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if (y0 - m["AS"]) * (y1 - m["AS"]) <= 0 and y0 != y1:
                    eq = x0 + (m["AS"] - y0) * (x1 - x0) / (y1 - y0)
                    break
            if eq is None:
                eq_txt = "more than 32 px equivalent" if m["AS"] < pts[-1][1] else "not on the curve (better than C2y or curve not monotone)"
            else:
                eq_txt = f"{eq:.1f} px equivalent y shift"
            out["R3"] = (f"d_AS = {d_as:.4f} (> delta {delta:.4f}); {eq_txt}. The shipped offset is two-axis on w018 "
                         "and position-dependent on w00, so this is a y-only yardstick")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs")
    p.add_argument("out_md")
    p.add_argument("--core", default="AS,C0,C2y,C4y,C8y")
    p.add_argument("--xarms", default="C2x,C4x,C8x")
    a = p.parse_args()
    res = load(a.runs)
    pilot = res.get("C0", {}).get(27, {}).get("auc")
    out = {"gate_G0": None if pilot is None else {"c0_seed27_auc": pilot, "pass": pilot >= GATE_AUC}}
    if pilot is not None and pilot < GATE_AUC:
        out["outcome"] = f"G0 failed: C0 seed 27 AUC {pilot:.4f} < {GATE_AUC}; inconclusive"
    else:
        out["y_curve"] = curve_outcome(res, a.core.split(","), "y")
        if any(x in res for x in a.xarms.split(",")):
            out["x_curve"] = curve_outcome(res, ["C0"] + a.xarms.split(","), "x")
    lines = ["# Label-shift results (rules from PREREG.md section 7)", "",
             "| arm | seed | pooled AUC | best F0.5 | per-segment AUC | trainer val balanced acc |", "|---|---|---|---|---|---|"]
    for arm in sorted(res):
        for s, r in sorted(res[arm].items()):
            seg = ", ".join(f"{k.split('_')[0]} {v:.4f}" for k, v in (r.get("per_segment_auc") or {}).items() if v is not None)
            f = lambda v: "missing" if v is None else f"{v:.4f}"  # noqa: E731
            lines.append(f"| {arm} | {s} | {f(r.get('auc'))} | {f(r.get('f05_best'))} | {seg} | {f(r.get('trainer_val_balanced_accuracy'))} |")
    lines += ["", "```", json.dumps(out, indent=1), "```"]
    open(a.out_md, "w").write("\n".join(lines) + "\n")
    json.dump({"runs": res, "decision": out}, open(os.path.splitext(a.out_md)[0] + ".json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
