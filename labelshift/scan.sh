#!/usr/bin/env bash
# Eval offset scan (SCAN-2026-11-01.md): score finished checkpoints against the held-out annotation moved by
# 0 to 128 px (labels/full_eval, scripts/build_eval_labels.py). go9b.sh runs it after ALL_DONE_H and before
# Amendment I's prepare, with the GPU to itself. Pairs run in the frozen order of scan/pairs.tsv. A pair starts
# only while fewer than CAP_MIN minutes have passed and has a 15 minute timeout, so Amendment I starts at most
# about CAP_MIN + 15 minutes late. Pair 1 (C0 s27 on E0) must reproduce C0 s27's eval.json, else the scan stops.
set -u
cd ~/labelshift
PY="venv/bin/python -I"
SRC=villa/vesuvius/src
CAP_MIN=${CAP_MIN:-35}
V=/vault/labelshift_scan
mkdir -p scan/out scan/cfg $V
log() { echo "[$(date -Is)] $*"; }
t0=$(date +%s)
log "scan start, cap $CAP_MIN min"
# written by hand once the label checks (check_eval_labels.py ALL_OK) and E0's patch counts (676) passed
[ -f scan/PRECHECK_OK ] || { log "scan/PRECHECK_OK is missing: scan not run"; touch scan/SCAN_DONE; exit 0; }
for seed in 27 28; do
    for arm in E0 E4y E8y E16y E32y EAS; do
        $PY scripts/eval_config_for.py runs/eval_C0_s$seed/config.json $arm scan/cfg/${arm}_s$seed/config.json \
            < /dev/null || { log "config $arm s$seed could not be written: scan stopped"; exit 0; }
    done
done
while read -r n run src arm seed; do
    { [ -z "$n" ] || [ "${n:0:1}" = "#" ]; } && continue
    el=$(( ($(date +%s) - t0) / 60 ))
    if [ $el -ge "$CAP_MIN" ]; then log "pair $n $run $arm: skipped (cap reached at $el min)"; continue; fi
    ck=""
    case $src in
        F) ck=$(ls /vault/labelshift_F/runs/$run/out/ckpt_*.pth 2>/dev/null | tail -1) ;;
        H) [ -f runs/$run/train.done ] && ck=$(ls runs/$run/out/ckpt_*.pth 2>/dev/null | tail -1) ;;
    esac
    [ -n "$ck" ] || { log "pair $n $run $arm: no checkpoint"; continue; }
    o=scan/out/${run}__${arm}
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
touch scan/SCAN_DONE
log "scan end"
