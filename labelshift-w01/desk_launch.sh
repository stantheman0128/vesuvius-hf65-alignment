#!/usr/bin/env bash
# Run experiment A (PREREG.md) on the GPU machine: WSL Ubuntu-24.04, one RTX 3070 Ti.
# Written for that machine's layout and not run there yet.
#
# Layout it expects:
#   ~/labelshift/villa   villa 37d1897fb sparse checkout   (reused read-only; no fetch, no checkout)
#   ~/labelshift/venv    Python 3.14.6 venv with scripts/pod_requirements.txt
#   ~/labelshift-w01     this folder (HERE); also the work tree (WORK): data, labels, runs, volume cache
# The October tree in ~/labelshift (its data, labels, runs and caches) is not touched.
#
# Start it in the foreground of a wsl.exe that outlives the SSH session (the WMI Win32_Process.Create
# route recorded in PROGRESS 2026-11-04), for example with STEPS=12000:
#   wsl.exe -d Ubuntu-24.04 -- bash -lc "STEPS=12000 bash ~/labelshift-w01/desk_launch.sh"
#
# What it does: preflight checks, pod_run.sh prepare, then the runs in PREREG order (C0 s27, AS s27, gate,
# then C0 and AS for seeds 28, 29, 30), then pod_run.sh summarize. Before prepare and before each run it
# estimates the step's length and starts it only if it ends before the next busy window (Mon, Wed, Fri
# 10:00 to 19:00 +0800, when the machine is in use); otherwise it waits for that window to end. Finished steps are skipped on a
# restart. Touch ~/labelshift-w01/A_STOP to stop before the next step.
#
# Env: STEPS (required, 12000 per PREREG section 4), WORKERS (8; keep it the same for every run of A), SEC_PER_STEP
# (2.6, measured on this machine with a cold cache on 2026-11-04), EVAL_MIN (40, both evals of one run, a guess
# until the first run is measured), PREP_MIN (60).
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WORK=${WORK:-$HOME/labelshift-w01}
export VILLA=${VILLA:-$HOME/labelshift/villa}
export VENV=${VENV:-$HOME/labelshift/venv}
export WORKERS=${WORKERS:-8}
export MAXSTEPS_MODE=anneal
export SEEDS="27 28 29 30"
SEC_PER_STEP=${SEC_PER_STEP:-2.6}
EVAL_MIN=${EVAL_MIN:-40}
PREP_MIN=${PREP_MIN:-60}
LOG=$WORK/desk_launch.log
mkdir -p "$WORK"
exec >> "$LOG" 2>&1

log() { echo "[$(TZ=Asia/Taipei date '+%F %T %z')] $*"; }
die() { log "STOP: $*"; exit 2; }

case ${STEPS:-} in
    12000) export STEPS ;;
    *) die "STEPS must be 12000 (PREREG section 4); got '${STEPS:-}'" ;;
esac

# --- busy windows -----------------------------------------------------------------------------------
# prints "start end" (epoch seconds) of the first busy window that has not ended at time $1
busy_window() {
    local now=$1 base day d
    base=$(TZ=Asia/Taipei date -d "@$now" +%F)
    for d in 0 1 2 3 4 5 6 7; do
        day=$(date -d "$base +$d day" +%F)
        case $(date -d "$day" +%u) in
            1|3|5)
                local s e
                s=$(TZ=Asia/Taipei date -d "$day 10:00" +%s)
                e=$(TZ=Asia/Taipei date -d "$day 19:00" +%s)
                [ "$e" -gt "$now" ] && { echo "$s $e"; return; } ;;
        esac
    done
}

# "go" if a step of $2 seconds started at time $1 ends before the next busy window starts,
# else "wait <end of that window>"
slot_decision() {
    local now=$1 need=$2 s e
    read -r s e < <(busy_window "$now")
    if [ "$now" -ge "$s" ] || [ $((now + need)) -gt "$s" ]; then echo "wait $e"; else echo go; fi
}

# waits until a step of $1 seconds can start and end outside busy hours
wait_for_slot() {
    local need=$1 now verdict until
    while :; do
        [ -f "$WORK/A_STOP" ] && die "A_STOP found"
        now=$(date +%s)
        read -r verdict until < <(slot_decision "$now" "$need")
        if [ "$verdict" = wait ]; then
            log "next step needs $((need / 60)) min and would overlap busy hours; waiting until $(TZ=Asia/Taipei date -d "@$until" '+%a %F %R')"
            sleep $((until - now + 60))
            continue
        fi
        # nothing else may hold the GPU (tonight's tests, other jobs)
        if pgrep -f "[l]aunch.py|[e]val_heldout.py|[d]esk_tonight" > /dev/null; then
            log "another GPU job is running: $(pgrep -af "[l]aunch.py|[e]val_heldout.py|[d]esk_tonight" | head -3 | tr '\n' ';'); checking again in 10 min"
            sleep 600
            continue
        fi
        return 0
    done
}

# --- estimates ------------------------------------------------------------------------------------------
est_run() {  # seconds for one run: training plus both evals, +10%, or 5% over the longest measured run
    local guess measured=0 d
    guess=$(awk -v s="$STEPS" -v r="$SEC_PER_STEP" -v e="$EVAL_MIN" 'BEGIN {printf "%d", (s * r + e * 60) * 1.10}')
    for d in "$WORK"/runs/C0_s* "$WORK"/runs/AS_s*; do
        [ -f "$d/train.start" ] && [ -f "$d/eval_AS.json" ] || continue
        local t=$(( $(stat -c %Y "$d/eval_AS.json") - $(cat "$d/train.start") ))
        [ "$t" -gt "$measured" ] && measured=$t
    done
    if [ "$measured" -gt 0 ]; then
        awk -v m="$measured" 'BEGIN {printf "%d", m * 1.05}'
    else
        echo "$guess"
    fi
}

# --- preflight ------------------------------------------------------------------------------------------
log "desk_launch start: STEPS=$STEPS WORKERS=$WORKERS"
head=$(git -C "$VILLA" rev-parse HEAD 2>/dev/null) || die "no git checkout at $VILLA"
case $head in 37d1897fb*) log "villa $head" ;; *) die "villa is at $head, not 37d1897fb" ;; esac
[ "$("$VENV/bin/python" -I -c 'import sys; print(sys.version.split()[0])')" = 3.14.6 ] || die "$VENV is not Python 3.14.6"
mem_gb=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
[ "$mem_gb" -ge 20 ] || die "WSL has ${mem_gb} GB RAM; 8 workers were killed by the OOM killer at 15 GB on 2026-11-04 (.wslconfig memory=24GB fixes it)"
free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 80 ] || die "only ${free_gb} GB free on the WSL disk"
r=$(est_run)
total=$(( PREP_MIN * 60 + 8 * r ))
log "estimate: $((r / 60)) min per run (STEPS x ${SEC_PER_STEP} s + ${EVAL_MIN} min of eval, +10%); seed 27 pair $((2 * r / 3600)) h; all 8 runs plus prepare $((total / 3600)) h of machine time, spread over the hours outside the busy windows"

# --- prepare --------------------------------------------------------------------------------------------
if [ ! -f "$WORK/A_PREPARED" ]; then
    wait_for_slot $((PREP_MIN * 60))
    log "prepare"
    bash "$HERE/pod_run.sh" prepare > "$WORK/prepare.log" 2>&1 || die "prepare failed: $(tail -3 "$WORK/prepare.log" | tr '\n' ' ')"
    touch "$WORK/A_PREPARED"
    log "prepare done"
fi
# every run of A uses one setting: a restart with another STEPS or WORKERS would mix two settings
for c in "$WORK"/runs/*/config.json; do
    got=$("$VENV/bin/python" -I -c 'import json, sys; c = json.load(open(sys.argv[1])); print(c["num_iterations"], c["dataloader_workers"])' "$c") \
        || die "cannot read $c"
    [ "$got" = "$STEPS $WORKERS" ] || die "$c has num_iterations and workers '$got', not '$STEPS $WORKERS'"
done

# --- runs -----------------------------------------------------------------------------------------------
run_step() {  # arm seed
    local d=$WORK/runs/${1}_s$2
    [ -f "$d/eval.json" ] && [ -f "$d/eval_AS.json" ] && { log "$1 s$2 already finished"; return 0; }
    [ -f "$d/train.start" ] && [ ! -f "$d/train.done" ] && die "$1 s$2 has a training that did not finish ($d); look at $d/stderr.log, move $d/out away and restart"
    wait_for_slot "$(est_run)"
    log "$1 s$2 start (estimate $(( $(est_run) / 60 )) min)"
    bash "$HERE/pod_run.sh" one "$1" "$2" >> "$WORK/run.log" 2>&1 || die "$1 s$2 failed: $(tail -3 "$WORK/run.log" | tr '\n' ' ')"
    log "$1 s$2 done in $(( $(stat -c %Y "$d/eval_AS.json") - $(cat "$d/train.start") )) s"
}

run_step C0 27
run_step AS 27
bash "$HERE/pod_run.sh" gate >> "$WORK/run.log" 2>&1
rc=$?
log "gate: $(grep '^gate:' "$WORK/run.log" | tail -1) (rc $rc)"
if [ $rc -ne 0 ]; then
    bash "$HERE/pod_run.sh" summarize > /dev/null 2>&1
    touch "$WORK/A_DONE"
    log "A ends: w01 does not learn in this setup (gate not passed); seeds 28 to 30 not run. results.md written"
    exit 0
fi
for seed in 28 29 30; do
    run_step C0 $seed
    run_step AS $seed
done
bash "$HERE/pod_run.sh" summarize > /dev/null 2>&1
touch "$WORK/A_DONE"
log "A done: $(grep '^Outcome' "$WORK/results.md")"
