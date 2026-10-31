#!/usr/bin/env bash
# go9b.sh with the eval offset scan run by scan2.sh (SCAN-2026-11-01.md, revision of 2026-11-01 02:40 +0800).
# The scan's first pass runs while H trains; after ALL_DONE_H, scan2.sh scores any pair still missing, capped at 35
# minutes, and then Amendment I runs as in go9.sh. It replaces go9b.sh, which was stopped while it waited for
# ALL_DONE_H. A scan failure does not stop Amendment I.
set -u
cd ~/labelshift
export UV_CACHE_DIR=$HOME/labelshift/uv_cache
PY="venv/bin/python -I"
log() { echo "[$(date -Is)] $*"; }
finish() { log "$*"; touch ALL_DONE_I; exit 0; }

log "go9c start (Amendment I after the rest of the eval offset scan), waiting for ALL_DONE_H"
for f in AMENDMENT-I-2026-10-31.md scripts/decide_i.py prior_F/runs/C0_s27/eval.json prior_F/runs/C0_s28/eval.json; do
    [ -f "$f" ] || finish "$f is missing: stop"
done
until [ -f ALL_DONE_H ]; do sleep 60; done
log "ALL_DONE_H seen"
setsid nohup bash vault_sync_i.sh > vault_sync_i.log 2>&1 < /dev/null &
log "eval offset scan, pairs not scored yet (scan2.sh, log appended to scan.log)"
CAP_MIN=35 FINAL=1 bash scan2.sh >> scan.log 2>&1 < /dev/null
log "scan2.sh returned $?"
export STEPS=12000 SEEDS=29 MAXSTEPS_MODE=anneal WORKERS=16 G0_WAIVED=1
export BUILD_ARMS=AS,C0,C2y,C4y,C8y,C2x,C4x,C8x,C16y,C32y
log "prepare (configs for seed 29)"
ARMS=C0,C32y,C16y bash pod_run.sh prepare > prepare_I.log 2>&1 || finish "prepare failed: $(tail -3 prepare_I.log | tr '\n' ' ')"
log "run"
ARMS=C0,C32y,C16y FORCE=1 bash pod_run.sh run > run_I.log 2>&1
log "pod_run.sh run exited with status $?"
$PY scripts/decide_i.py prior_F/runs runs results_I.md > decide_i.log 2>&1
finish "go9 end; $(grep -E '^- C(16|32)y vs C0' results_I.md 2>/dev/null | tr '\n' ' ')"
