#!/bin/bash
# Pack what desk_launch.sh and pod_run.sh need into one tarball that unpacks as labelshift-w01/
# (for the GPU machine: copy it to the WSL home and run tar -xzf there, which gives ~/labelshift-w01).
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
cd "$W"
tar -czf w01_bundle.tar.gz --exclude=__pycache__ --transform 's,^,labelshift-w01/,' \
  desk_launch.sh pod_run.sh PREREG.md regions.json \
  labels/manifest_full.json labels/manifest_eval.json logs/raw_sha256.txt \
  scripts/fetch_files.py scripts/build_labels.py scripts/build_eval_labels.py scripts/labellib.py \
  scripts/check_eval_labels.py scripts/mkconfig.py scripts/eval_config_for.py scripts/launch.py \
  scripts/eval_heldout.py scripts/check_gpu_config.py scripts/count_val_patches.py scripts/decide_w01.py \
  scripts/pod_requirements.txt scripts/hf65
ls -la w01_bundle.tar.gz
tar -tzf w01_bundle.tar.gz
