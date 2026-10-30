"""Before/after table from results/before_<key>.jsonl and results/after_<key>.jsonl (last row per
region wins). Prints markdown and a summary of |residual| per group.  argv: [results_dir]"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
SEG = {"w00": "w00", "w01": "w01_r15", "w05": "w05_4424", "c018": "w018"}


def load(mode, key):
    p = os.path.join(RD, f"{mode}_{key}.jsonl")
    rows = {}
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            rows[r["region"]] = r
    return rows


def f2(v, fmt="+.2f"):
    return "n/a" if v is None else format(v, fmt)


print("| segment | region | sample | centre (y, x) | before dy, dx | before peak NCC | before NCC at 0 | README predicted dy, dx |"
      " after residual dy, dx | after peak NCC | after NCC at 0 | method B residual (README) | method B residual (3-term) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
groups = {"in": [], "out": []}
groupsB = {"in": [], "out": []}
for key in ("w00", "w01", "w05", "c018"):
    B, A = load("before", key), load("after", key)
    order = sorted(B, key=lambda s: (not s.startswith("r"), s))
    for lab in order:
        b = B[lab]
        a = A.get(lab)
        g = "in" if b["sample"].startswith("in-sample") else "out"
        mb = b["methodB"]["readme"]
        m3 = b["methodB"].get("affine3") or b["methodB"].get("mean")
        groupsB[g].append((abs(mb["level0_dy"]), abs(mb["level0_dx"])))
        if a:
            groups[g].append((abs(a["level0_dy"]), abs(a["level0_dx"])))
        pr = b["readme_pred_at_centre"]
        print(f"| {SEG[key]} | {lab} | {'in' if g == 'in' else 'OUT'} | {b['centre'][0]:.0f}, {b['centre'][1]:.0f} |"
              f" {b['level0_dy']:+.2f}, {b['level0_dx']:+.2f} | {b['hp']['ncc']:.3f} | {f2(b['ncc_zero_shift_hp'], '.3f')} |"
              f" {pr[0]:+.2f}, {pr[1]:+.2f} |"
              + (f" {a['level0_dy']:+.2f}, {a['level0_dx']:+.2f} | {a['hp']['ncc']:.3f} | {a['ncc_zero_shift_hp']:.3f} |" if a else " not run | | |")
              + f" {mb['level0_dy']:+.2f}, {mb['level0_dx']:+.2f} | {m3['level0_dy']:+.2f}, {m3['level0_dx']:+.2f} |")
for g in ("in", "out"):
    for nm, G in (("after (CT sampled at corrected cells)", groups[g]), ("method B (canvas resampled into tifxyz frame)", groupsB[g])):
        if G:
            Ar = np.array(G)
            print(f"\n{g}-sample, {nm}: {len(G)} regions, max |dy| {Ar[:, 0].max():.2f}, max |dx| {Ar[:, 1].max():.2f},"
                  f" mean |dy| {Ar[:, 0].mean():.2f}, mean |dx| {Ar[:, 1].mean():.2f}")
