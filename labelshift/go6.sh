#!/usr/bin/env bash
# Amendment F on the pod. Waits for go5.sh to exit. If no pilot passed G0, runs the exploratory w00-only
# comparison: C0 seed 27 reused from L1 (or from L2 under the amendment's exception), new runs AS s27, C8y s27,
# C0 s28, AS s28, C8y s28. If a pilot passed G0, Amendment E's plan ran inside go5 and F is void.
# ALL_DONE_F marks the end in every case; the local autostop_f.sh watches for it.
set -u
cd ~/labelshift
export UV_CACHE_DIR=$HOME/labelshift/uv_cache
PY="venv/bin/python -I"
HARD=$(date -d "2026-10-31 15:20:00 UTC" +%s)   # 23:20 +0800
log() { echo "[$(date -Is)] $*"; }
finish() { log "$*"; touch ALL_DONE_F; exit 0; }
w00() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['metrics']['w00_20231016151002']['auc'])" "$1" 2>/dev/null; }

log "go6 start, waiting for go5.sh to exit"
while [ $(ps -eo args | grep -c "^bash go5\.sh") -gt 0 ]; do sleep 60; done
log "go5.sh exited; last go5 line: $(tail -1 go5.log)"
[ -f amendment_e_choice.txt ] && finish "a pilot passed G0, so Amendment E's plan ran in go5: Amendment F is void"
[ -f runs_L1/C0_s27/eval.json ] && [ -f runs_L2/C0_s27/eval.json ] || finish "L1 or L2 has no eval.json: Amendment F has nothing to build on"

A1=$(w00 runs_L1/C0_s27/eval.json); A2=$(w00 runs_L2/C0_s27/eval.json)
PER2=$(awk '$1 == "pilot" && $2 == "L2" {print $6}' pilots.txt | tail -1)
NOW=$(date +%s); END2=$(( NOW + 5 * ${PER2:-999999} ))
if $PY -c "import sys; sys.exit(0 if float(sys.argv[2]) - float(sys.argv[1]) >= 0.01 else 1)" "$A1" "$A2" && [ $END2 -lt $HARD ]; then
    SET=L2; STEPS=12000
else
    SET=L1; STEPS=6000
fi
{ echo "Amendment F choice, written $(date -Is) before any Amendment F run"
  echo "C0 seed 27 w00 AUC: L1 $A1, L2 $A2 (L2 setting needs L2 - L1 >= 0.01)"
  echo "L2 wall clock ${PER2:-missing}s; five runs at that pace from now would end $(date -d @$END2 -Is), hard stop $(date -d @$HARD -Is)"
  echo "setting: $SET (STEPS=$STEPS, MAXSTEPS_MODE=anneal, WORKERS=16); C0 seed 27 reused from runs_$SET/C0_s27"
  echo "new runs, in order: AS s27, C8y s27, C0 s28, AS s28, C8y s28; G0 waived; primary readout: w00 AUC"; } | tee f_choice.txt

cp -p pod_run.sh pod_run.sh.E && mv pod_run.sh.F pod_run.sh || finish "could not install the Amendment F pod_run.sh"
grep -q G0_WAIVED pod_run.sh || finish "pod_run.sh has no G0_WAIVED switch"
[ -e runs ] && mv runs runs_before_F
mkdir -p runs && cp -a runs_$SET/C0_s27 runs/C0_s27
[ -f results.md ] && cp -f results.md results_E_last_pilot.md
setsid nohup bash vault_sync_f.sh > vault_sync_f.log 2>&1 < /dev/null &
export STEPS ARMS=C0,AS,C8y SEEDS="27 28" MAXSTEPS_MODE=anneal WORKERS=16 G0_WAIVED=1
log "Amendment F: prepare ($SET setting)"
bash pod_run.sh prepare > prepare_F.log 2>&1 || finish "prepare for Amendment F failed: $(tail -3 prepare_F.log | tr '\n' ' ')"
log "Amendment F: run"
FORCE=1 bash pod_run.sh run > run_F.log 2>&1
log "pod_run.sh run exited with status $?"
cp -f results.md results_F_prereg_rules.md 2>/dev/null
cp -f results.json results_F_prereg_rules.json 2>/dev/null
$PY scripts/decide_f.py runs results_F.md > decide_f.log 2>&1
finish "go6 end; $(grep -E '^- (C8y|AS) vs C0' results_F.md 2>/dev/null | tr '\n' ' ')"
