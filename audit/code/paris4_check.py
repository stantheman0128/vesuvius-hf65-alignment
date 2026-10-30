"""For every registered segment whose tifxyz full-res size differs from the canvas (flag b), compare the
measured offsets with the size-ratio model of villa#2370: d = a + (full/canvas - 1) * position.
Per region it prints the size-ratio term at the region centre and the implied intercept a = measured - term;
with two or more regions, a constant implied intercept (spread <= 3 px) supports the #2370 type.
Also prints the published #2370 fit for the four segments it covers, evaluated at the same centres.
-> results/phase2/paris4_check.json"""
import os, json, glob

AUDIT = os.environ.get("AUDIT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P2 = os.path.join(AUDIT, "results", "phase2")
# README fits of stantheman0128/vesuvius-hf65-alignment (affine3, level-0 canvas px)
FITS = {
    "ink/phercparis4/w00_20231016151002": ((-20.6991, 0.0009926, 0.0000013), (11.0864, 0.0000224, -0.0006630)),
    "ink/phercparis4/w01_20230702185753_r15": ((11.5100, 0.0042631, 0.0000060), (11.8180, 0.0000025, -0.0038761)),
    "ink/phercparis4/w05_4424": ((-42.3946, 0.0007890, 0.0000008), (22.9348, -0.0000042, -0.0004539)),
    "ink/1667/w018_20240304144031_2um": ((-19.64, 0, 0), (-19.64, 0, 0)),
}
out = {}
for p in sorted(glob.glob(os.path.join(P2, "reg_*.jsonl"))):
    rows = {}
    for l in open(p):
        r = json.loads(l)
        rows[r["region"]] = r
    for ri, r in sorted(rows.items()):
        cy, cx = r["y0"] + r["size"] / 2, r["x0"] + r["size"] / 2
        ty, tx = r["size_ratio_pred_at_centre"]
        e = {"key": r["key"], "region": ri, "centre": [cy, cx], "measured": [r["level0_dy"], r["level0_dx"]],
             "ncc": r["hp"]["ncc"], "second": r["hp"]["second"], "size_ratio_term": [ty, tx],
             "implied_intercept": [r["level0_dy"] - ty, r["level0_dx"] - tx]}
        if r["seg"] in FITS:
            (a, b, c), (a2, b2, c2) = FITS[r["seg"]]
            e["published_fit"] = [a + b * cy + c * cx, a2 + b2 * cy + c2 * cx]
        out.setdefault(r["seg"], []).append(e)
for seg, es in out.items():
    print(seg)
    for e in es:
        pf = e.get("published_fit")
        print(f"   {e['key']} r{e['region']} centre ({e['centre'][0]:.0f},{e['centre'][1]:.0f}) measured ({e['measured'][0]:+.1f},{e['measured'][1]:+.1f})"
              f" ncc {e['ncc']:.3f} | size-ratio term ({e['size_ratio_term'][0]:+.1f},{e['size_ratio_term'][1]:+.1f})"
              f" implied intercept ({e['implied_intercept'][0]:+.1f},{e['implied_intercept'][1]:+.1f})"
              + (f" | published fit ({pf[0]:+.1f},{pf[1]:+.1f})" if pf else ""))
    if len(es) >= 2:
        sy = max(e["implied_intercept"][0] for e in es) - min(e["implied_intercept"][0] for e in es)
        sx = max(e["implied_intercept"][1] for e in es) - min(e["implied_intercept"][1] for e in es)
        print(f"   implied intercept spread over {len(es)} regions: rows {sy:.1f} px, cols {sx:.1f} px")
json.dump(out, open(os.path.join(P2, "paris4_check.json"), "w"), indent=1)
