# Amendment E: anneal-schedule ladder for the C0 seed 27 pilot, then the comparison plan that fits.
# Writes ALL_DONE when nothing more will run on this pod.
set -u
cd ~/labelshift
export MAXSTEPS_MODE=anneal WORKERS=16
PY="venv/bin/python -I"
HARD=$(date -d "2026-10-31 15:20:00 UTC" +%s)   # 23:20 +0800
log() { echo "[$(date -Is)] $*"; }
auc() { $PY -c "import json; print(json.load(open('runs/C0_s27/eval.json'))['metrics']['pooled']['auc'])" 2>/dev/null; }
pass() { $PY -c "import sys; sys.exit(0 if float('$1') >= 0.70 else 1)" 2>/dev/null; }

pilot() {  # name steps -> sets PER (seconds) and AUC
    export STEPS=$2 ARMS=C0 SEEDS=27
    log "pilot $1: STEPS=$2 MAXSTEPS_MODE=$MAXSTEPS_MODE"
    bash pod_run.sh prepare > prepare_$1.log 2>&1 || { log "prepare $1 failed"; tail -5 prepare_$1.log; return 2; }
    local t0=$(date +%s)
    FORCE=1 bash pod_run.sh run > run_$1.log 2>&1
    PER=$(( $(date +%s) - t0 )); AUC=$(auc)
    echo "pilot $1 steps $2 seconds $PER pooled_auc ${AUC:-missing}" | tee -a pilots.txt
}

log "go5 start"
chosen=""
for step in "L1 6000" "L2 12000"; do
    set -- $step
    pilot $1 $2 || { log "prepare failed before pilot $1: nothing trained, stop"; touch ALL_DONE; exit 1; }
    if [ -n "$AUC" ] && pass "$AUC"; then chosen=$1; STEPS_OK=$2; break; fi
    log "pilot $1 did not pass G0 (AUC ${AUC:-missing})"
    mv runs runs_$1
done
if [ -z "$chosen" ]; then log "no pilot passed G0: stop (Amendment E)"; touch ALL_DONE; exit 0; fi

T=$(( HARD - $(date +%s) - 1080 ))
plan=none
if   [ $(( 9 * PER )) -le $T ]; then plan=P0; ARMS=C0,AS,C8y,C4y,C2y
elif [ $(( 5 * PER )) -le $T ]; then plan=P1; ARMS=C0,AS,C8y
elif [ $(( 3 * PER )) -le $T ]; then plan=P2; ARMS=C0,AS
fi
{ echo "Amendment E choice, written $(date -Is) before the comparison runs"
  echo "passing pilot $chosen: STEPS=$STEPS_OK, pooled AUC $AUC, wall clock ${PER}s (train + eval)"
  echo "T = ${T}s to the 23:20 +0800 hard stop minus 0.3 h; P0 needs $((9*PER))s, P1 $((5*PER))s, P2 $((3*PER))s"
  echo "chosen: $plan ${ARMS:-} SEEDS=\"27 28\" STEPS=$STEPS_OK MAXSTEPS_MODE=anneal"; } | tee amendment_e_choice.txt
[ $plan = none ] && { log "no plan fits"; touch ALL_DONE; exit 0; }

export STEPS=$STEPS_OK ARMS SEEDS="27 28"
bash pod_run.sh prepare > prepare_cmp.log 2>&1 || { log "prepare for the comparison failed"; tail -5 prepare_cmp.log; touch ALL_DONE; exit 1; }
FORCE=1 bash pod_run.sh run > run_cmp.log 2>&1
cp -f results.md results_E_$plan.md 2>/dev/null
log "comparison $plan done"

if [ $plan = P0 ]; then   # Amendment D: x arms, N from the measured per-run time
    avail=$(( HARD - $(date +%s) - 1080 )); N=$(( avail > 0 ? avail / PER : 0 ))
    if [ $N -ge 6 ]; then XS="27 28"; elif [ $N -ge 3 ]; then XS="27"; else XS=""; fi
    log "Amendment D: N=$N, x seeds '${XS}'"
    if [ -n "$XS" ]; then
        export ARMS=C0,C8x,C4x,C2x SEEDS="$XS"
        bash pod_run.sh prepare > prepare_x.log 2>&1 && FORCE=1 bash pod_run.sh run > run_x.log 2>&1
        cp -f results.md results_E_with_x.md 2>/dev/null
    fi
fi
log "go5 end"
touch ALL_DONE
