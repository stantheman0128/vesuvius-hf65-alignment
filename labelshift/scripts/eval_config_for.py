"""Write the eval offset scan's config for one eval label folder (SCAN-2026-11-01.md).

usage: python -I eval_config_for.py C0_EVAL_CONFIG ARM OUT_JSON

The C0 eval config (runs/eval_C0_s<seed>/config.json) with its two segments_path values moved from
labels/full/C0/ to labels/full_eval/ARM/, and out_dir moved to <dir of OUT_JSON>/out. out_dir is where villa's
InkDataset keeps its patch cache, which it overwrites when the labels differ, so each folder gets its own.
Nothing else changes.
"""
import json
import os
import sys

src, arm, out = sys.argv[1:4]
text = open(src).read()
old = "/labels/full/C0/"
assert text.count(old) == 2, f"{src}: {text.count(old)} label roots, expected 2"
cfg = json.loads(text.replace(old, f"/labels/full_eval/{arm}/"))
assert cfg["out_dir"].endswith("/out"), cfg["out_dir"]
cfg["out_dir"] = os.path.join(os.path.dirname(os.path.abspath(out)), "out")
os.makedirs(cfg["out_dir"], exist_ok=True)
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
