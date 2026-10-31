#!/bin/bash
W=${W:-$(cd "$(dirname "$0")/.." && pwd)}
mkdir -p $W/out/arr $W/dl
SEGS="w013_20240304141531 w018_20240304144031 w023_20240304161941 w028_20251208130119156 w031_2025122323"
run() { # seg-prefix array level
  o=$W/out/arr/${1%%_*}_${2}_L${3}.npz
  [ -f $o ] && return
  env W029_DL=$W/dl OMP_NUM_THREADS=2 ${PY:-python3} -I $W/scripts/readlevel.py ink/1667/${1}_2um/${1}_${2}.zarr $3 $o >> $W/out/readlevel.jsonl
}
for s in $SEGS; do
  for a in 2um_supervision_mask supervision_mask supervision_mask_v2 2um_inklabels inklabels_v2; do run $s $a 5; done
done
echo L5DONE
for s in $SEGS; do
  for a in 2um_supervision_mask 2um_inklabels; do run $s $a 4; done
done
echo DONE
