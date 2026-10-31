"""Readout of the eval offset scan (SCAN-2026-11-01.md, "Readout").

usage: python -I decide_scan.py SCAN_OUT_DIR F_RUNS H_RUNS OUT_MD [CHECK_TXT]

SCAN_OUT_DIR holds <run>__<folder>.json from scan.sh. A checkpoint's E0 score is its own eval.json: F_RUNS for
C0, AS and C8y (Amendment F), H_RUNS for C16y and C32y (Amendment H). Writes two tables (question 1 and
question 2) for w00, the segment that is read, with pooled and w018 next to it, and the wording check.
"""
import json
import os
import sys

out_dir, f_runs, h_runs, out_md = sys.argv[1:5]
check = open(sys.argv[5]).read().strip() if len(sys.argv) > 5 and os.path.exists(sys.argv[5]) else "(no check file)"
SEGS = (("w00", "w00_20231016151002"), ("pooled", "pooled"), ("w018", "w018_20240304144031_2um"))
OWN = {"C8y": "E8y", "C16y": "E16y", "C32y": "E32y", "AS": "EAS"}


def auc(path, seg):
    if not os.path.exists(path):
        return None
    m = json.load(open(path))["metrics"]
    return m.get(seg, {}).get("auc")


def score(run, folder, seg):
    if folder == "E0":
        base = h_runs if run.startswith(("C16y", "C32y")) else f_runs
        return auc(os.path.join(base, run, "eval.json"), seg)
    return auc(os.path.join(out_dir, f"{run}__{folder}.json"), seg)


def fmt(v, signed=False):
    return "not run" if v is None else (f"{v:+.4f}" if signed else f"{v:.4f}")


lines = ["# Eval offset scan readout", "", f"Pair 1 check: {check}", ""]
lines += ["## Question 1: C0 checkpoints, held-out labels moved down", "",
          "| checkpoint | segment | E0 (0 px) | E4y (16) | E8y (32) | E16y (64) | E32y (128) |", "|---|---|---|---|---|---|---|"]
for run in ("C0_s27", "C0_s28"):
    for name, seg in SEGS:
        e0 = score(run, "E0", seg)
        cells = [fmt(e0)]
        for fo in ("E4y", "E8y", "E16y", "E32y"):
            v = score(run, fo, seg)
            cells.append("not run" if v is None else f"{v:.4f} ({v - e0:+.4f})")
        lines.append(f"| {run} | {name} | " + " | ".join(cells) + " |")
lines += ["", "Differences in brackets are against E0.", "",
          "## Question 2: trained on moved labels, scored on the same move", "",
          "| checkpoint | own move | w00 on E0 | w00 on own move | difference | C0 same seed: difference there | pooled difference | w018 difference |",
          "|---|---|---|---|---|---|---|---|"]
gains, c0_losses, complete = [], [], True
for arm, fo in OWN.items():
    for seed in ("s27", "s28"):
        run = f"{arm}_{seed}"
        d = {}
        for name, seg in SEGS:
            a0, a1 = score(run, "E0", seg), score(run, fo, seg)
            d[name] = None if a0 is None or a1 is None else a1 - a0
        c0a, c0b = score(f"C0_{seed}", "E0", SEGS[0][1]), score(f"C0_{seed}", fo, SEGS[0][1])
        c0d = None if c0a is None or c0b is None else c0b - c0a
        if d["w00"] is None:
            lines.append(f"| {run} | {fo} | {fmt(score(run, 'E0', SEGS[0][1]))} | not run | | | | |")
            continue
        gains.append(d["w00"] > 0)
        if c0d is None:
            complete = False
        else:
            c0_losses.append(c0d < 0)
        lines.append(f"| {run} | {fo} | {fmt(score(run, 'E0', SEGS[0][1]))} | {fmt(score(run, fo, SEGS[0][1]))} | "
                     f"{fmt(d['w00'], True)} | {fmt(c0d, True)} | {fmt(d['pooled'], True)} | {fmt(d['w018'], True)} |")
allowed = bool(gains) and all(gains) and complete and all(c0_losses)
lines += ["", "Wording check (SCAN-2026-11-01.md): \"models trained on moved labels predict moved ink\" is "
          + ("allowed: every trained checkpoint that ran gains on its own move and C0 of the same seed loses there."
             if allowed else
             f"not allowed. Trained checkpoints that ran: {len(gains)}, gaining on their own move: {sum(gains)}; "
             f"C0 comparisons available: {len(c0_losses)}, losing there: {sum(c0_losses)}"
             + ("" if complete else "; at least one C0 comparison did not run") + ".")]
with open(out_md, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
