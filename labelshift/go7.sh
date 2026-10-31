#!/usr/bin/env bash
# Amendment F after the unexpected stop (F_RESTART_NOTE.md). The tree was rebuilt from pod_bundle.tar.gz with
# Amendment F's files on top. f_choice.txt (written 05:22:36 UTC) is reused unchanged: L2 setting, C0 seed 27 from
# L2, restored into runs/. Runs AS s27, C8y s27, C0 s28, AS s28, C8y s28 from scratch. ALL_DONE_F marks the end.
set -u
cd ~/labelshift
export UV_CACHE_DIR=$HOME/labelshift/uv_cache
PY="venv/bin/python -I"
log() { echo "[$(date -Is)] $*"; }
finish() { log "$*"; touch ALL_DONE_F; exit 0; }

log "go7 start (Amendment F restart)"
grep -q '^setting: L2 (STEPS=12000' f_choice.txt || finish "f_choice.txt does not hold the L2 choice: stop"
[ -f runs/C0_s27/eval.json ] && [ -f runs/C0_s27/train.done ] || finish "runs/C0_s27 was not restored: stop"
grep -q G0_WAIVED pod_run.sh || finish "pod_run.sh has no G0_WAIVED switch: stop"
setsid nohup bash vault_sync_f.sh > vault_sync_f.log 2>&1 < /dev/null &
export STEPS=12000 ARMS=C0,AS,C8y SEEDS="27 28" MAXSTEPS_MODE=anneal WORKERS=16 G0_WAIVED=1
log "prepare (fresh tree)"
bash pod_run.sh prepare > prepare_F2.log 2>&1 || finish "prepare failed: $(tail -3 prepare_F2.log | tr '\n' ' ')"
log "run"
FORCE=1 bash pod_run.sh run > run_F2.log 2>&1
log "pod_run.sh run exited with status $?"
cp -f results.md results_F_prereg_rules.md 2>/dev/null
cp -f results.json results_F_prereg_rules.json 2>/dev/null
$PY scripts/decide_f.py runs results_F.md > decide_f.log 2>&1
finish "go7 end; $(grep -E '^- (C8y|AS) vs C0' results_F.md 2>/dev/null | tr '\n' ' ')"
