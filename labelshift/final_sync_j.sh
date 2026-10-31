# Last copy before the second pod stops (run by the local autostop_j.sh). Copies Amendment J's files and every
# finished seed 30 checkpoint to /vault/labelshift_J. No readout is written here: decide_j.py needs the first pod's
# runs and is run locally.
set -u
cd ~/labelshift
touch ALL_DONE_J
V=/vault/labelshift_J
mkdir -p $V
tar cf - go_j.log prepare_J.log run_J*.log vault_sync_j.log AMENDMENT-I-2026-10-31.md AMENDMENT-J-2026-11-01.md \
    runs/*/config.json runs/*/eval.json runs/*/*.log runs/*/train.done runs/*/out/*.jsonl runs/*/out/*.json 2>/dev/null \
    | (cd $V && tar xf -)
for d in runs/*_s30; do
  [ -f $d/train.done ] || continue
  ck=$(ls $d/out/ckpt_*.pth 2>/dev/null | tail -1); [ -n "$ck" ] || continue
  [ -f $V/$ck ] && [ $(stat -c %s $ck) = $(stat -c %s $V/$ck) ] && continue
  mkdir -p $V/$d/out; cp $ck $V/$ck.part && mv $V/$ck.part $V/$ck
done
echo "$(date -Is) autostop final sync" >> $V/last_sync.txt
echo "eval.json on vault: J $(ls $V/runs/*_s30/eval.json 2>/dev/null | wc -l)"
du -sh $V 2>/dev/null
