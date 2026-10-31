# copy Amendment J's small results every 10 min, and each finished seed 30 checkpoint once, to /vault/labelshift_J
# (/vault/labelshift_E, _F, _H and _I stay untouched)
set -u
V=/vault/labelshift_J; W=~/labelshift
mkdir -p $V
while :; do
  cd $W
  tar cf - go_j.log prepare_J.log run_J*.log vault_sync_j.log AMENDMENT-I-2026-10-31.md AMENDMENT-J-2026-11-01.md \
      runs/*/config.json runs/*/eval.json runs/*/*.log runs/*/train.done runs/*/out/*.jsonl runs/*/out/*.json 2>/dev/null \
      | (cd $V && tar xf -)
  for d in runs/*_s30; do
    [ -f $d/train.done ] || continue
    ck=$(ls $d/out/ckpt_*.pth 2>/dev/null | tail -1); [ -n "$ck" ] || continue
    [ -f $V/$ck ] && [ $(stat -c %s $ck) = $(stat -c %s $V/$ck) ] && continue
    mkdir -p $V/$d/out; cp $ck $V/$ck.part && mv $V/$ck.part $V/$ck
  done
  echo "$(date -Is) synced" > $V/last_sync.txt
  [ -f $W/ALL_DONE_J ] && { echo "$(date -Is) final sync" >> $V/last_sync.txt; break; }
  sleep 600
done
