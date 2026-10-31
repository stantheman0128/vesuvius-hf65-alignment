#!/usr/bin/env bash
# go9.sh (Amendment I) with the eval offset scan (SCAN-2026-11-01.md) between ALL_DONE_H and Amendment I's
# prepare. It replaces go9.sh, which was stopped while it waited for ALL_DONE_H. Everything after the scan is
# go9.sh unchanged. The scan is capped in scan.sh, and a scan failure does not stop Amendment I.
set -u
cd ~/labelshift
export UV_CACHE_DIR=$HOME/labelshift/uv_cache
PY="venv/bin/python -I"
log() { echo "[$(date -Is)] $*"; }
finish() { log "$*"; touch ALL_DONE_I; exit 0; }

log "go9b start (Amendment I with the eval offset scan first), waiting for ALL_DONE_H"
for f in AMENDMENT-I-2026-10-31.md scripts/decide_i.py prior_F/runs/C0_s27/eval.json prior_F/runs/C0_s28/eval.json; do
    [ -f "$f" ] || finish "$f is missing: stop"
done
until [ -f ALL_DONE_H ]; do sleep 60; done
log "ALL_DONE_H seen"
setsid nohup bash vault_sync_i.sh > vault_sync_i.log 2>&1 < /dev/null &
log "eval offset scan (scan.sh, log in scan.log)"
bash scan.sh > scan.log 2>&1 < /dev/null
log "scan.sh returned $?"
export STEPS=12000 SEEDS=29 MAXSTEPS_MODE=anneal WORKERS=16 G0_WAIVED=1
export BUILD_ARMS=AS,C0,C2y,C4y,C8y,C2x,C4x,C8x,C16y,C32y
log "prepare (configs for seed 29)"
ARMS=C0,C32y,C16y bash pod_run.sh prepare > prepare_I.log 2>&1 || finish "prepare failed: $(tail -3 prepare_I.log | tr '\n' ' ')"
log "run"
ARMS=C0,C32y,C16y FORCE=1 bash pod_run.sh run > run_I.log 2>&1
log "pod_run.sh run exited with status $?"
$PY scripts/decide_i.py prior_F/runs runs results_I.md > decide_i.log 2>&1
finish "go9 end; $(grep -E '^- C(16|32)y vs C0' results_I.md 2>/dev/null | tr '\n' ' ')"
