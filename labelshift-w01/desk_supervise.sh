#!/usr/bin/env bash
# Runs desk_launch.sh and starts it again when a run died on a network error, at most MAX_RESTARTS times.
# Added on 2026-11-05 after the PREREG freeze (see DEVIATIONS.md): the first AS s27 training died at step
# 2314 when an S3 read ran out of retries. The volume cache does not keep chunks that S3 reports missing,
# so every run keeps reading from S3. Nothing here changes what a run computes: a training that died is
# moved to $WORK/failed/ and the run starts again from step 0 with the same config and seed.
#
# Start it the way desk_launch.sh is started, for example:
#   wsl.exe -d Ubuntu-24.04 -- bash -lc "bash ~/labelshift-w01/desk_supervise.sh"
#
# Env: STEPS (12000), AWS_RETRY_MODE (legacy) and AWS_MAX_ATTEMPTS (14): botocore's retry settings for the
# trainer's S3 reads. The default of 5 attempts rides out about 30 s without a connection. Legacy mode waits
# rand * 2^(n-1) s after attempt n and has no retry quota, so with 14 attempts one read keeps trying for about
# an hour on average. MAX_RESTARTS (2), RESTART_WAIT (900 s).
set -uo pipefail

WORK=${WORK:-$HOME/labelshift-w01}
LAUNCH=${LAUNCH:-$WORK/desk_launch.sh}
MAX_RESTARTS=${MAX_RESTARTS:-2}
RESTART_WAIT=${RESTART_WAIT:-900}
export STEPS=${STEPS:-12000}
export AWS_RETRY_MODE=${AWS_RETRY_MODE:-legacy}
export AWS_MAX_ATTEMPTS=${AWS_MAX_ATTEMPTS:-14}
NET='ConnectTimeoutError|ConnectionTimeoutError|ReadTimeoutError|EndpointConnectionError|ClientConnectorError|ServerDisconnectedError'
LOG=$WORK/desk_launch.log

log() { echo "[$(TZ=Asia/Taipei date '+%F %T %z')] supervise: $*" >> "$LOG"; }

restarts=0
while :; do
    log "start desk_launch.sh (AWS_RETRY_MODE=$AWS_RETRY_MODE AWS_MAX_ATTEMPTS=$AWS_MAX_ATTEMPTS, restart $restarts of $MAX_RESTARTS)"
    bash "$LAUNCH"
    rc=$?
    [ $rc -eq 0 ] && exit 0
    [ -f "$WORK/A_STOP" ] && { log "A_STOP found; not restarting"; exit $rc; }
    [ "$restarts" -ge "$MAX_RESTARTS" ] && { log "desk_launch.sh stopped (rc $rc) after $restarts restarts; not restarting"; exit $rc; }

    # a training that died: move it away if it died on the network, else leave everything for a person
    moved= evalnet=
    for d in "$WORK"/runs/C0_s* "$WORK"/runs/AS_s*; do
        [ -f "$d/train.start" ] && [ ! -f "$d/train.done" ] || continue
        grep -qE "$NET" "$d/stderr.log" 2>/dev/null || { log "$(basename "$d") stopped without a network error; not restarting"; exit $rc; }
        dest=$WORK/failed/$(basename "$d")_$(date +%Y%m%d-%H%M%S)
        mkdir -p "$dest"
        for f in out train.start stdout.log stderr.log; do
            [ -e "$d/$f" ] && mv "$d/$f" "$dest/"
        done
        log "$(basename "$d") died on a network error; moved its training to $dest"
        moved="$moved $(basename "$d")"
    done
    # an eval that died on the network starts again as it is (pod_run.sh skips the parts already done)
    if [ -z "$moved" ]; then
        for d in "$WORK"/runs/C0_s* "$WORK"/runs/AS_s*; do
            [ -f "$d/train.done" ] && [ ! -f "$d/eval_AS.json" ] || continue
            grep -qsE "$NET" "$d/eval_stderr.log" "$d/eval_AS_stderr.log" && evalnet="$evalnet $(basename "$d")"
        done
        [ -n "$evalnet" ] || { log "desk_launch.sh stopped (rc $rc) without a network error in a run; not restarting"; exit $rc; }
        log "eval of$evalnet died on a network error"
    fi
    restarts=$((restarts + 1))
    log "restart $restarts of $MAX_RESTARTS in $((RESTART_WAIT / 60)) min"
    sleep "$RESTART_WAIT"
done
