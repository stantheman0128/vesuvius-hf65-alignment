#!/usr/bin/env bash
# Extra seeds for experiment A after its pre-registered result (exploratory, outside PREREG section 7).
# Own work tree so A's runs/ and results.md stay as published: data/raw, labels and volume_cache are
# symlinks into ~/labelshift-w01, runs/ is new. One run at a time, started only when it can end before the
# next busy window (Mon, Wed, Fri 09:00 to 20:00 +0800); a run still going 5 min before that window is
# stopped and retried in the next free window. Touch ~/labelshift-w01-extra/STOP to stop before the next run.
set -uo pipefail
A=$HOME/labelshift-w01
XW=$HOME/labelshift-w01-extra
QUEUE=${QUEUE:-"C0:31 AS:31 C0:32 AS:32"}
export WORK=$XW VILLA=$HOME/labelshift/villa VENV=$HOME/labelshift/venv
export WORKERS=8 STEPS=12000 MAXSTEPS_MODE=anneal SEEDS="31 32"
export AWS_RETRY_MODE=legacy AWS_MAX_ATTEMPTS=14
EST=25400      # 5% over A's longest run (C0 s30, 24,188 s)
MARGIN=300
NET='ConnectTimeoutError|ConnectionTimeoutError|ReadTimeoutError|EndpointConnectionError|ClientConnectorError|ServerDisconnectedError'
SEG=w01_20230702185753_r15
LOG=$XW/extra.log
mkdir -p "$XW"
exec >> "$LOG" 2>&1
log() { echo "[$(TZ=Asia/Taipei date '+%F %T %z')] $*"; }
auc() { "$VENV/bin/python" -I -c 'import json, sys; print("%.4f" % json.load(open(sys.argv[1]))["metrics"][sys.argv[2]]["auc"])' "$1" "$SEG" 2>/dev/null || echo "?"; }

busy_window() {  # "start end" of the first Mon/Wed/Fri 09:00-20:00 window that has not ended at $1
    local now=$1 base d day s e
    base=$(TZ=Asia/Taipei date -d "@$now" +%F)
    for d in 0 1 2 3 4 5 6 7; do
        day=$(TZ=Asia/Taipei date -d "$base +$d day" +%F)
        case $(TZ=Asia/Taipei date -d "$day" +%u) in
            1|3|5)
                s=$(TZ=Asia/Taipei date -d "$day 09:00" +%s)
                e=$(TZ=Asia/Taipei date -d "$day 20:00" +%s)
                [ "$e" -gt "$now" ] && { echo "$s $e"; return; } ;;
        esac
    done
}

DEADLINE=0
wait_for_slot() {
    local now s e
    while :; do
        [ -f "$XW/STOP" ] && { log "STOP found; exiting"; exit 0; }
        now=$(date +%s)
        read -r s e < <(busy_window "$now")
        if [ "$now" -ge "$s" ] || [ $((now + EST)) -gt "$s" ]; then
            log "a run needs up to $((EST / 60)) min and would reach the busy window at $(TZ=Asia/Taipei date -d "@$s" '+%a %F %R'); waiting until $(TZ=Asia/Taipei date -d "@$e" '+%a %F %R')"
            sleep $((e - now + 60))
            continue
        fi
        if pgrep -f "[l]aunch.py|[e]val_heldout.py|[r]ender_beforeafter" > /dev/null; then
            log "another GPU job: $(pgrep -af '[l]aunch.py|[e]val_heldout.py|[r]ender_beforeafter' | head -2 | tr '\n' ';'); checking again in 10 min"
            sleep 600
            continue
        fi
        DEADLINE=$((s - MARGIN))
        return 0
    done
}

stash_training() {  # dir reason: move a training that did not finish to failed/
    local d=$1 dest
    dest=$XW/failed/$(basename "$d")_$(date +%Y%m%d-%H%M%S)
    mkdir -p "$dest"
    for f in out train.start stdout.log stderr.log; do [ -e "$d/$f" ] && mv "$d/$f" "$dest/"; done
    log "$(basename "$d"): $2; moved its training to $dest"
}

run_one() {  # arm seed; returns 0 done, 1 retry, 2 stop
    local arm=$1 seed=$2 d=$XW/runs/${1}_s$2 rc
    [ -f "$d/eval.json" ] && [ -f "$d/eval_AS.json" ] && { log "$arm s$seed already finished"; return 0; }
    wait_for_slot
    log "$arm s$seed start; stopped at $(TZ=Asia/Taipei date -d "@$DEADLINE" '+%a %F %R') if not done"
    timeout --kill-after=120 $((DEADLINE - $(date +%s))) bash "$A/pod_run.sh" one "$arm" "$seed" >> "$XW/run.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        log "$arm s$seed done in $(( $(stat -c %Y "$d/eval_AS.json") - $(cat "$d/train.start") )) s: AUC on C0 labels $(auc "$d/eval.json"), on labels as shipped $(auc "$d/eval_AS.json")"
        return 0
    fi
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        [ -f "$d/train.done" ] || stash_training "$d" "stopped before the busy window (rc $rc)"
        return 1
    fi
    if [ ! -f "$d/train.done" ]; then
        grep -qE "$NET" "$d/stderr.log" 2>/dev/null && { stash_training "$d" "died on a network error"; return 1; }
        log "STOP: $arm s$seed training failed without a network error (rc $rc); see $d/stderr.log"
        return 2
    fi
    grep -qsE "$NET" "$d/eval_stderr.log" "$d/eval_AS_stderr.log" && { log "$arm s$seed eval died on a network error"; return 1; }
    log "STOP: $arm s$seed eval failed (rc $rc); see $d/eval_stderr.log and $d/eval_AS_stderr.log"
    return 2
}

log "extra_seeds start: queue $QUEUE, STEPS $STEPS, WORKERS $WORKERS, AWS_MAX_ATTEMPTS $AWS_MAX_ATTEMPTS"
[ -d "$A/data/raw" ] && [ -d "$A/labels/full" ] && [ -d "$A/volume_cache" ] || { log "STOP: A's tree is incomplete"; exit 2; }
mkdir -p "$XW/data" "$XW/runs" "$XW/failed"
[ -e "$XW/data/raw" ] || ln -s "$A/data/raw" "$XW/data/raw"
[ -e "$XW/labels" ] || ln -s "$A/labels" "$XW/labels"
[ -e "$XW/volume_cache" ] || ln -s "$A/volume_cache" "$XW/volume_cache"
if [ ! -f "$XW/PREPARED" ]; then
    log "prepare: configs for seeds $SEEDS, inputs checked again"
    bash "$A/pod_run.sh" prepare > "$XW/prepare.log" 2>&1 || { log "STOP: prepare failed: $(tail -3 "$XW/prepare.log" | tr '\n' ' ')"; exit 2; }
    touch "$XW/PREPARED"
    log "prepare done"
fi
for item in $QUEUE; do
    arm=${item%%:*} seed=${item##*:}
    for attempt in 1 2; do
        run_one "$arm" "$seed"
        rc=$?
        [ $rc -eq 0 ] && break
        [ $rc -eq 2 ] && exit 2
        [ $attempt -eq 2 ] && { log "STOP: $arm s$seed did not finish in 2 attempts"; exit 2; }
        log "$arm s$seed attempt 2 in 15 min"
        sleep 900
    done
done
log "queue done"
