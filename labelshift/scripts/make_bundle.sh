#!/bin/bash
# Pack what pod_run.sh needs into one tarball Stan can copy to the pod (scp / pod upload).
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
cd "$W"
tar -czf pod_bundle.tar.gz --exclude=__pycache__ --transform 's,^,labelshift/,' \
  pod_run.sh PREREG.md regions.json labels/manifest_full.json \
  scripts/fetch_files.py scripts/build_labels.py scripts/labellib.py scripts/mkconfig.py scripts/launch.py \
  scripts/eval_heldout.py scripts/decide.py scripts/pod_requirements.txt scripts/hf65
ls -la pod_bundle.tar.gz
tar -tzf pod_bundle.tar.gz
