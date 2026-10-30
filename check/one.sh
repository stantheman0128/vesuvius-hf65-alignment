#!/bin/bash
VS=$(cd "$(dirname "$0")" && pwd)
cd /tmp  # keep cwd outside the data and script dirs
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
${PYTHON:-python3} -I $VS/regv.py $VS "$@"
