#!/bin/bash
# Out-of-sample before/after runs, one process at a time (parallel runs hit HTTP 429 on the HF tifxyz).
W="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-python3}
export HF65_DATA=${HF65_DATA:?set HF65_DATA to the cache folder} OMP_NUM_THREADS=1
cd /tmp
run() { tag=$1; shift; echo "== $(date +%T) $*"; HF65_LEDGER_TAG=$tag $PY -I $W/correct/regcorr.py "$@" 2>&1 | grep -v "^rangecache:"; }
# held-out w01 region, centre (46016, 27008)
run hold_w01 w01 before 45504,26496
run hold_w01 w01 after 45504,26496
# first new region per Paris4 segment
for k in w00 w01 w05; do
  run new_$k $k before new0
  run new_$k $k after new0
done
# cover-cache check: same after-run with plain common.fetch, into a separate output dir
mkdir -p $W/covercheck/results $W/covercheck/npz
echo "== $(date +%T) covercheck w01 after 0 (HF65_COVER=0)"
HF65_COVER=0 HF65CORR_OUT=$W/covercheck HF65_LEDGER_TAG=covercheck $PY -I $W/correct/regcorr.py w01 after 0 2>&1
# w018 new region
run new_c018 c018 before new0
run new_c018 c018 after new0
echo "== $(date +%T) done"
