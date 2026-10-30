#!/bin/bash
# 1) cover-cache check on a run whose CT bytes were mostly stitched (held-out w01 after-run),
# 2) second new region per Paris4 segment, 3) the in-sample after-runs that hit HTTP 429.
# Each run is skipped once the new downloads of this work (ledger_*.tsv) pass LIMIT_MB.
W="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-python3}
export HF65_DATA=${HF65_DATA:?set HF65_DATA to the cache folder} OMP_NUM_THREADS=1
LIMIT_MB=${LIMIT_MB:-275}
cd /tmp
used() { cat $HF65_DATA/ledger_*.tsv | awk -F'\t' '{s+=$1} END {printf "%d", s/1e6}'; }
run() { tag=$1; shift; u=$(used); if [ "$u" -ge "$LIMIT_MB" ]; then echo "== SKIP $* (new downloads $u MB >= $LIMIT_MB)"; return; fi
        echo "== $(date +%T) $* (new downloads so far $u MB)"; HF65_LEDGER_TAG=$tag $PY -I $W/correct/regcorr.py "$@" 2>&1 | grep -v "^rangecache:"; }
mkdir -p $W/covercheck/results $W/covercheck/npz
echo "== $(date +%T) covercheck2 w01 after 45504,26496 (HF65_COVER=0)"
HF65_COVER=0 HF65CORR_OUT=$W/covercheck HF65_LEDGER_TAG=covercheck2 $PY -I $W/correct/regcorr.py w01 after 45504,26496 2>&1
$PY -I -c "
import numpy as np
a = np.load('$W/npz/after_w01_at45504_26496.npz'); b = np.load('$W/covercheck/npz/after_w01_at45504_26496.npz')
for k in ('S', 'Sm', 'Mh'):
    print('covercheck2', k, 'identical' if np.array_equal(a[k], b[k], equal_nan=True) else 'DIFFERENT', a[k].shape)
"
for k in w01 w05 w00; do
  run new_$k $k before new1
  run new_$k $k after new1
done
run after2_c018 c018 after 5
for r in 3 4 5; do run after2_w01 w01 after $r; done
run after2_w05 w05 after 5
for r in 3 4 5; do run after2_w00 w00 after $r; done
echo "== $(date +%T) done, new downloads $(used) MB"
