#!/usr/bin/env bash
# Amendment J: seed 30 of C0, C32y and C16y on a second, fresh pod, with Amendment I's settings (go9c.sh from its
# Amendment I part on, with SEEDS=30). The pod is fresh, so nothing is awaited and prepare builds the venv, data,
# label arms and configs. The four-seed reading (decide_j.py) needs the first pod's runs too and is run locally after
# both pods' results are pulled, so it is not run here. ALL_DONE_J marks the end.
set -u
cd ~/labelshift
export UV_CACHE_DIR=$HOME/labelshift/uv_cache
PY="venv/bin/python -I"
log() { echo "[$(date -Is)] $*"; }
finish() { log "$*"; touch ALL_DONE_J; exit 0; }

log "go_j start (Amendment J, fresh pod, nothing to wait for)"
for f in AMENDMENT-I-2026-10-31.md AMENDMENT-J-2026-11-01.md scripts/decide_i.py scripts/decide_j.py; do
    [ -f "$f" ] || finish "$f is missing: stop"
done
grep -q BUILD_ARMS pod_run.sh || finish "pod_run.sh has no BUILD_ARMS switch: stop"
setsid nohup bash vault_sync_j.sh > vault_sync_j.log 2>&1 < /dev/null &
export STEPS=12000 SEEDS=30 MAXSTEPS_MODE=anneal WORKERS=16 G0_WAIVED=1
export BUILD_ARMS=AS,C0,C2y,C4y,C8y,C2x,C4x,C8x,C16y,C32y
log "prepare (venv, data, label arms, configs for seed 30)"
ARMS=C0,C32y,C16y bash pod_run.sh prepare > prepare_J.log 2>&1 || finish "prepare failed: $(tail -3 prepare_J.log | tr '\n' ' ')"
log "run"
ARMS=C0,C32y,C16y FORCE=1 bash pod_run.sh run > run_J.log 2>&1
log "pod_run.sh run exited with status $?"
finish "go_j end; eval.json present for: $(ls -d runs/*_s30 2>/dev/null | while read d; do [ -f $d/eval.json ] && printf '%s ' "${d#runs/}"; done)"
