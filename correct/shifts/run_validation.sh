#!/bin/bash
# The validation batch as run on 2026-10-31, one process at a time (parallel runs hit HTTP 429 on the HF files).
# AUDIT_DATA: the audit's download cache (read first). HF65_DATA: a fresh, empty folder for new downloads.
# PY: a Python with numpy, scipy, httpx, numcodecs. Results append to correct/shifts/validation/results.jsonl.
set -u
W="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-python3}
export AUDIT_DATA=${AUDIT_DATA:?set AUDIT_DATA to the audit data folder} HF65_DATA=${HF65_DATA:?set HF65_DATA to a fresh folder}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
V="$PY -I $W/validate.py"
cd /tmp
echo "== start $(date -Is)"
# the audit picks for these three all sit within 1024 px of their fit regions; pick a far block instead
for k in c028 t028 t043; do [ -f "$W/validation/regions_far_$k.json" ] || $V farregion $k 0.0; done
# controls and one in-sample region first (cached), then one held-out region per row
$V run p500:0 c013:0
$V run n841:0 n841a:0 n841b:0 man5:0 \
  c018:new1 t040:2 p4w07:2 \
  c013:2 c023:1 c028:far0 c029v:1 c031:1 b0009:2 t016:2 t017:1 t028:far0 t029:1 t030:1 \
  t035:1 t039:1 t041:1 t043:far0 t044:1 \
  w00:new1 w01:new1 p4w02:2 p4w03:2 w05:new1 p4w06:2 p4w09:2
$V table > "$W/validation/table.md"
echo "== end $(date -Is)"
