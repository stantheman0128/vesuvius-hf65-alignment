#!/usr/bin/env bash
# Eval offset scan, second driver (SCAN-2026-11-01.md, revision of 2026-11-01 02:40 +0800). The pairs, configs,
# checkpoints and eval command are those of scan.sh. Logistics changes only: it may run while H trains (started with
# nice 10), a pair whose output json already exists is skipped, pairs other than pair 1 run only after pair 1's check
# printed CHECK_OK, and a lock keeps two calls from running at once. It is called while H trains, again by
# scan_wait_h.sh once C32y_s28 has trained, and by go9c.sh after ALL_DONE_H (CAP_MIN=35 FINAL=1) for any pair that is
# still not scored.
set -u
cd ~/labelshift
exec 9> scan/scan.lock
flock 9
PY="venv/bin/python -I"
SRC=villa/vesuvius/src
CAP_MIN=${CAP_MIN:-150}
FINAL=${FINAL:-0}
V=/vault/labelshift_scan
mkdir -p scan/out scan/cfg $V
log() { echo "[$(date -Is)] $*"; }
t0=$(date +%s)
log "scan2 start, cap $CAP_MIN min, final $FINAL"
# written by hand once the label checks (check_eval_labels.py ALL_OK) and E0's patch counts (676) passed
[ -f scan/PRECHECK_OK ] || { log "scan/PRECHECK_OK is missing: scan not run"; [ "$FINAL" = 1 ] && touch scan/SCAN_DONE; exit 0; }
for seed in 27 28; do
    for arm in E0 E4y E8y E16y E32y EAS; do
        $PY scripts/eval_config_for.py runs/eval_C0_s$seed/config.json $arm scan/cfg/${arm}_s$seed/config.json \
            < /dev/null || { log "config $arm s$seed could not be written: scan stopped"; exit 0; }
    done
done
while read -r n run src arm seed; do
    { [ -z "$n" ] || [ "${n:0:1}" = "#" ]; } && continue
    o=scan/out/${run}__${arm}
    [ -s $o.json ] && continue
    if [ "$n" != 1 ] && ! grep -q '^CHECK_OK' scan/check.txt 2>/dev/null; then
        log "pair $n: pair 1 has no CHECK_OK: scan stopped"; break
    fi
    el=$(( ($(date +%s) - t0) / 60 ))
    if [ $el -ge "$CAP_MIN" ]; then log "pair $n $run $arm: skipped (cap reached at $el min)"; continue; fi
    ck=""
    case $src in
        F) ck=$(ls /vault/labelshift_F/runs/$run/out/ckpt_*.pth 2>/dev/null | tail -1) ;;
        H) [ -f runs/$run/train.done ] && ck=$(ls runs/$run/out/ckpt_*.pth 2>/dev/null | tail -1) ;;
    esac
    [ -n "$ck" ] || { log "pair $n $run $arm: no checkpoint yet"; continue; }
    log "pair $n $run on $arm (config seed $seed): $ck"
    timeout 900 $PY scripts/eval_heldout.py $SRC "$ck" scan/cfg/${arm}_s$seed/config.json $o.json \
        --batch 2 --workers 16 > $o.stdout.log 2> $o.stderr.log < /dev/null
    log "pair $n returned $? at $(( ($(date +%s) - t0) / 60 )) min"
    if [ "$n" = 1 ]; then
        $PY scripts/scan_check.py prior_F/runs/C0_s27/eval.json $o.json > scan/check.txt 2>&1 < /dev/null
        log "$(cat scan/check.txt)"
        grep -q '^CHECK_OK' scan/check.txt || { log "pair 1 did not reproduce eval.json: scan stopped"; break; }
    fi
    cp scan/out/*.json scan/check.txt "$V/" 2>/dev/null
done < scan/pairs.tsv
cp scan/out/* scan/check.txt scan.log "$V/" 2>/dev/null
[ "$FINAL" = 1 ] && touch scan/SCAN_DONE
log "scan2 end, $(ls scan/out/*.json 2>/dev/null | wc -l) of 18 pairs scored"
