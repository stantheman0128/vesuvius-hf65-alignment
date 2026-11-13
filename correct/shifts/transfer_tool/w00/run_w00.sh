# w00 end-to-end check: transfer single with the table's tool-frame offset, a no-offset control,
# and the table's loader-frame offset. The UV cache is built once and reused (it does not depend
# on the offset). Then pixel agreement and region shifts against apply_fit "readme" w00, which is
# the shifts.csv w00 row.
#
# usage: PY=python VILLA_SRC=... VC_READER=... W00_DIR=... RUN_DIR=... bash run_w00.sh
# VILLA_SRC, VC_READER: see ../villa_boot.py (leave VC_READER unset to use an installed vc_reader).
# W00_DIR: the segment folder; RUN_DIR: where the outputs go (large TIFFs and the UV cache).
# The runs are pinned to 4 cores and timed with GNU time, as when the published numbers were made.
set -u
: "${PY:?}" "${VILLA_SRC:?}" "${W00_DIR:?}" "${RUN_DIR:?}"
export VILLA_SRC W00_DIR RUN_DIR
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN=$HERE/../run_transfer.py
OUT=$RUN_DIR
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
COMMON="--source-tifxyz $W00_DIR --target-tifxyz $W00_DIR --label $W00_DIR/w00_20231016151002_inklabels.tif --workers 4 --uv-cache $OUT/uvcache/w00"
mkdir -p $OUT/offset $OUT/control $OUT/loader $OUT/uvcache
date -Is; echo "== offset run (tool frame, table value)"
/usr/bin/time -v -o $OUT/offset/time.txt taskset -c 0-3 $PY -I $RUN single $COMMON \
  --label-canvas-offset -21.3771 11.0061 --output $OUT/offset/w00_inklabels_offset.tif
echo "exit $?"; date -Is; echo "== control run (no offset)"
/usr/bin/time -v -o $OUT/control/time.txt taskset -c 0-3 $PY -I $RUN single $COMMON \
  --output $OUT/control/w00_inklabels_nooffset.tif
echo "exit $?"; date -Is; echo "== loader-frame run (table loader_DY loader_DX)"
/usr/bin/time -v -o $OUT/loader/time.txt taskset -c 0-3 $PY -I $RUN single $COMMON \
  --label-canvas-offset -20.6889 11.5579 --output $OUT/loader/w00_inklabels_loaderframe.tif
echo "exit $?"; date -Is; echo "== compare (pixel agreement)"
/usr/bin/time -v -o $OUT/compare_time.txt taskset -c 0-3 $PY -I $HERE/compare.py
echo "exit $?"; date -Is; echo "== cc (region shifts, cross-correlation)"
/usr/bin/time -v -o $OUT/cc_time.txt taskset -c 0-3 $PY -I $HERE/cc.py
echo "exit $?"; date -Is; echo "== cc2 (region shifts, ink overlap)"
/usr/bin/time -v -o $OUT/cc2_time.txt taskset -c 0-3 $PY -I $HERE/cc2.py
echo "exit $?"; date -Is; echo ALL_DONE
