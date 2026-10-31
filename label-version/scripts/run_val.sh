#!/bin/bash
W=${W:-$(cd "$(dirname "$0")/.." && pwd)}
mkdir -p $W/out/arr $W/dl
for s in w018_20240304144031 w023_20240304161941 w028_20251208130119156 w031_2025122323; do
  o=$W/out/arr/${s%%_*}_validation_mask_v2_L5.npz
  [ -f $o ] && continue
  env W029_DL=$W/dl OMP_NUM_THREADS=2 ${PY:-python3} -I $W/scripts/readlevel.py ink/1667/${s}_2um/${s}_validation_mask_v2.zarr 5 $o >> $W/out/readlevel.jsonl
done
echo VALDONE
