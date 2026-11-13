# w01 end-to-end check: transfer single with the affine3 refit constant 11.3181 11.1463 (not the
# shifts.csv w01 row), then a no-offset control reusing the UV cache (the cache does not depend on
# the offset). Then pixel agreement and region shifts against apply_fit affine3 w01.
#
# usage: PY=python VILLA_SRC=... VC_READER=... W01_DIR=... RUN_DIR=... bash run_w01.sh
# VILLA_SRC, VC_READER: see ../villa_boot.py (leave VC_READER unset to use an installed vc_reader).
# W01_DIR: the segment folder; RUN_DIR: where the outputs go (large TIFFs and the UV cache).
set -u
: "${PY:?}" "${VILLA_SRC:?}" "${W01_DIR:?}" "${RUN_DIR:?}"
export VILLA_SRC W01_DIR RUN_DIR
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN=$HERE/../run_transfer.py
OUT=$RUN_DIR
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
COMMON="--source-tifxyz $W01_DIR --target-tifxyz $W01_DIR --label $W01_DIR/w01_20230702185753_r15_inklabels.tif --workers 4 --uv-cache $OUT/uvcache/w01"
mkdir -p $OUT/offset $OUT/control $OUT/uvcache
date -Is; echo "== offset run"
/usr/bin/time -v -o $OUT/offset/time.txt taskset -c 0-3 $PY -I $RUN single $COMMON \
  --label-canvas-offset 11.3181 11.1463 --output $OUT/offset/w01_inklabels_offset.tif
echo "exit $?"; date -Is; echo "== control run (no offset)"
/usr/bin/time -v -o $OUT/control/time.txt taskset -c 0-3 $PY -I $RUN single $COMMON \
  --output $OUT/control/w01_inklabels_nooffset.tif
echo "exit $?"; date -Is; echo "== compare (pixel agreement)"
/usr/bin/time -v -o $OUT/compare_time.txt taskset -c 0-3 $PY -I $HERE/compare.py
echo "exit $?"; date -Is; echo "== cc (region shifts, cross-correlation)"
/usr/bin/time -v -o $OUT/cc_time.txt taskset -c 0-3 $PY -I $HERE/cc.py
echo "exit $?"; date -Is; echo "== cc2 (region shifts, ink overlap)"
/usr/bin/time -v -o $OUT/cc2_time.txt taskset -c 0-3 $PY -I $HERE/cc2.py
echo "exit $?"; date -Is; echo ALL_DONE
