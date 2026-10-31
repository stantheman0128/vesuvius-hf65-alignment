"""Check pair 1 of the eval offset scan: C0 seed 27 on E0 must give its eval.json back.

usage: python -I scan_check.py REFERENCE_EVAL_JSON SCAN_EVAL_JSON

Prints CHECK_OK when the patch counts match and pooled and per-segment AUC differ by at most 0.001,
CHECK_FAILED otherwise. Only the differences are printed.
"""
import json
import sys

ref, new = (json.load(open(p)) for p in sys.argv[1:3])
bad = []
for k in ("n_patches", "n_patches_scored"):
    if ref[k] != new[k]:
        bad.append(f"{k} {ref[k]} vs {new[k]}")
diffs = {}
for seg in ref["metrics"]:
    a, b = ref["metrics"][seg].get("auc"), new["metrics"].get(seg, {}).get("auc")
    if a is None or b is None:
        bad.append(f"{seg}: auc missing")
        continue
    diffs[seg] = abs(a - b)
    if diffs[seg] > 1e-3:
        bad.append(f"{seg}: |auc diff| {diffs[seg]:.6f}")
print(("CHECK_OK" if not bad else "CHECK_FAILED") + " max |auc diff| per segment "
      + ", ".join(f"{k} {v:.2e}" for k, v in diffs.items()) + ("" if not bad else "; " + "; ".join(bad)))
