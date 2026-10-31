#!/bin/bash
W=${W:-$(cd "$(dirname "$0")/.." && pwd)}
mkdir -p $W/out/arr $W/dl
S=ink/1667/w029_20251212185248662_2um
P=w029_20251212185248662
for lv in 5 4; do
for a in 2um_supervision_mask supervision_mask supervision_mask_v2 validation_mask_v2 2um_inklabels inklabels inklabels_v2; do
  o=$W/out/arr/w029_${a}_L${lv}.npz
  [ -f $o ] && continue
  env W029_DL=$W/dl OMP_NUM_THREADS=2 ${PY:-python3} -I $W/scripts/readlevel.py $S/${P}_${a}.zarr $lv $o >> $W/out/readlevel.jsonl
done
done
echo DONE
