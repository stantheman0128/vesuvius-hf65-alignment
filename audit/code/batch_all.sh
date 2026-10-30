#!/bin/bash
# usage: batch_all.sh <region idx> key[:pred] ...  (serial: pyramid model, region pick, registration)
# skips steps whose output already exists; stops when the download ledger passes LIMIT_MB (default 950)
# PY is the interpreter (default python3). HF65_DATA (default ./hf65_data) and HF65_RESULTS
# (default results/phase2 next to this folder) are read as in reg2.py.
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
export HF65_DATA=${HF65_DATA:-$PWD/hf65_data}
OUT=${HF65_RESULTS:-$HERE/../results/phase2}
mkdir -p "$HF65_DATA"
IDX="$1"; shift
LIMIT=${LIMIT_MB:-950}
for kk in "$@"; do
  k=${kk%%:*}; opt=""; [ "$kk" != "$k" ] && opt="--pred"
  mb=0
  [ -f "$HF65_DATA/ledger.tsv" ] && mb=$(awk -F'\t' '{s+=$1} END {printf "%d", s/1e6}' "$HF65_DATA/ledger.tsv")
  if [ "$mb" -ge "$LIMIT" ]; then echo "STOP: ledger ${mb} MB >= ${LIMIT} MB before $k"; break; fi
  [ -f "$OUT/pyr_$k.json" ] || "$PY" -I "$HERE/reg2.py" pyr "$k"
  [ -f "$OUT/regions_$k.json" ] || "$PY" -I "$HERE/reg2.py" regions "$k" 4
  if [ -f "$OUT/reg_$k.jsonl" ] && grep -q "\"region\": $IDX," "$OUT/reg_$k.jsonl"; then echo "skip $k r$IDX (done)"; continue; fi
  "$PY" -I "$HERE/reg2.py" reg "$k" "$IDX" $opt
done
echo "BATCH END"
