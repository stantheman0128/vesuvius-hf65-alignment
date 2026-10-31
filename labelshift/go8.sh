#!/usr/bin/env bash
# Amendment H: C16y and C32y (64 and 128 px) with seeds 27 and 28 on Amendment F's settings, from a fresh tree
# (pod_bundle.tar.gz plus h_extra.tar.gz). C0, AS and C8y are not rerun: their Amendment F eval.json files are in
# prior_F/runs. If BRIDGE_OK exists when the four runs end, C0 seed 28 is retrained as the session check.
# ALL_DONE_H marks the end.
set -u
cd ~/labelshift
export UV_CACHE_DIR=$HOME/labelshift/uv_cache
PY="venv/bin/python -I"
log() { echo "[$(date -Is)] $*"; }
finish() { log "$*"; touch ALL_DONE_H; exit 0; }

log "go8 start (Amendment H)"
for f in AMENDMENT-H-2026-10-31.md scripts/decide_h.py prior_F/runs/C0_s27/eval.json prior_F/runs/C0_s28/eval.json; do
    [ -f "$f" ] || finish "$f is missing: stop"
done
grep -q BUILD_ARMS pod_run.sh || finish "pod_run.sh has no BUILD_ARMS switch: stop"
setsid nohup bash vault_sync_h.sh > vault_sync_h.log 2>&1 < /dev/null &
export STEPS=12000 SEEDS="27 28" MAXSTEPS_MODE=anneal WORKERS=16 G0_WAIVED=1
export BUILD_ARMS=AS,C0,C2y,C4y,C8y,C2x,C4x,C8x,C16y,C32y
log "prepare (configs for C16y, C32y and the C0 session check)"
ARMS=C16y,C32y,C0 bash pod_run.sh prepare > prepare_H.log 2>&1 || finish "prepare failed: $(tail -3 prepare_H.log | tr '\n' ' ')"
log "run"
ARMS=C16y,C32y FORCE=1 bash pod_run.sh run > run_H.log 2>&1
log "pod_run.sh run exited with status $?"
if [ -f BRIDGE_OK ]; then
    log "session check: C0 seed 28"
    ARMS=C0 SEEDS=28 FORCE=1 bash pod_run.sh run > run_H_bridge.log 2>&1
    log "session check exited with status $?"
fi
$PY scripts/decide_h.py prior_F/runs runs results_H.md > decide_h.log 2>&1
finish "go8 end; $(grep -E '^- C(16|32)y vs C0' results_H.md 2>/dev/null | tr '\n' ' ')"
