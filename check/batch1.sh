#!/bin/bash
VS=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$VS/batch1.log}
r() { echo "== $*" >> $O; env "$@" >> $O 2>&1; }
P="bash $VS/one.sh"
r CENTRE=1 $P w018 25600 1536 1024 40
r CENTRE=1 $P c018s3 20480 48128 1024 40
r CENTRE=0 $P c018s3 20480 48128 1024 40
r CENTRE=1 $P w00 17408 2048 1024 40
r CENTRE=1 $P w00 6656 36864 1024 40
r NOPYR=1 $P w00 6656 36864 1024 40
r CENTRE=1 $P w01 9408 42688 640 80
r CENTRE=1 $P w01 56512 18112 640 80
echo ALLDONE >> $O
