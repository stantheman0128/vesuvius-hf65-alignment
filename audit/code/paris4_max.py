"""Largest |offset| over the canvas implied by d = a + (full/canvas - 1) * position per axis, with a the mean
implied intercept of this audit's regions (paris4_check.json). Evaluated at canvas px 0 and canvas size.
-> results/phase2/paris4_maxoffset.json"""
import os, json
AUDIT = os.environ.get("AUDIT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P1 = {r["path"]: r for r in json.load(open(os.path.join(AUDIT, "results", "phase1.json")))}
C = json.load(open(os.path.join(AUDIT, "results", "phase2", "paris4_check.json")))
out = {}
for seg, es in C.items():
    r = P1[seg]
    (Hf, Wf), (Hc, Wc) = r["full_hw"], r["canvas_hw"]
    if (Hf, Wf) == (Hc, Wc):
        continue
    ay = sum(e["implied_intercept"][0] for e in es) / len(es)
    ax = sum(e["implied_intercept"][1] for e in es) / len(es)
    my = max(abs(ay), abs(ay + Hf - Hc))
    mx = max(abs(ax), abs(ax + Wf - Wc))
    out[seg] = {"scale": r["meta_scale"], "n_regions": len(es), "intercept": [ay, ax], "full_minus_canvas": [Hf - Hc, Wf - Wc],
                "max_abs_offset_px": [my, mx]}
    print(f"{seg}: scale {r['meta_scale'][0]:.4f}, regions {len(es)}, intercept ({ay:+.1f},{ax:+.1f}), full-canvas ({Hf-Hc:+d},{Wf-Wc:+d}) -> max |offset| rows {my:.0f} px, cols {mx:.0f} px")
json.dump(out, open(os.path.join(AUDIT, "results", "phase2", "paris4_maxoffset.json"), "w"), indent=1)
