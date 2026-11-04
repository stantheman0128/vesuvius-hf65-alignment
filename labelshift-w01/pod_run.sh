#!/usr/bin/env bash
# Experiment A (PREREG.md): labels as shipped (AS) against HF65-corrected labels (C0) on Paris4 w01, villa's
# full_3d trainer at 37d1897fb. labelshift/pod_run.sh adapted to one segment and two arms; differences from it:
#   - inputs and labels: w01 only; label arms AS and C0; eval-only folders E0 and EAS for the secondary readout
#   - every run is scored twice: eval.json against C0 labels in the held-out boxes (primary) and eval_AS.json
#     against the labels as shipped there (secondary)
#   - run order C0 s27, AS s27, then the gate (C0 s27 AUC >= 0.60), then C0 and AS for seeds 28, 29, 30
#   - STEPS has no default and must be 12000 (PREREG section 4); MAXSTEPS_MODE defaults to anneal
#   - VILLA and VENV can point at an existing villa 37d1897fb checkout and venv (the GPU machine's ~/labelshift)
#   - CPU_DRYRUN=1 runs prepare without a GPU, a venv install or a villa clone (VILLA and VENV must exist)
# Idempotent: rerunning skips what is already there, finished runs are not redone.
#
#   bash pod_run.sh prepare        # env checks, w01 public data, AS/C0 arms, E0/EAS eval folders, configs
#   bash pod_run.sh counts         # CPU: trainer config staging and training/validation patch counts per arm
#   bash pod_run.sh one ARM SEED   # train and score one run
#   bash pod_run.sh gate           # exit 0 if C0 s27 passed the gate, 1 if it failed, 2 if not scored yet
#   bash pod_run.sh run            # every run in PREREG order with the gate, then summarize
#   bash pod_run.sh summarize      # PREREG section 7 -> $WORK/results.md
#
# Env: WORK (~/labelshift-w01), VILLA ($WORK/villa), VENV ($WORK/venv), STEPS (required), SEEDS ("27 28 29 30"),
# WORKERS (8), MAXSTEPS_MODE (anneal), CPU_DRYRUN (0).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-$HOME/labelshift-w01}
VILLA_COMMIT=${VILLA_COMMIT:-37d1897fb}
SEEDS=${SEEDS:-"27 28 29 30"}
WORKERS=${WORKERS:-8}
MAXSTEPS_MODE=${MAXSTEPS_MODE:-anneal}
CPU_DRYRUN=${CPU_DRYRUN:-0}
PYVER=3.14.6
VILLA=${VILLA:-$WORK/villa}
SRC=$VILLA/vesuvius/src
TEACHER=$SRC/vesuvius/ink_detection/configs/dino_guided_teacher.json
VENV=${VENV:-$WORK/venv}
PY="$VENV/bin/python -I"
RAW=$WORK/data/raw
CACHE=$WORK/volume_cache
LABELS=$WORK/labels/full
EVAL_LABELS=$WORK/labels/full_eval
RUNS=$WORK/runs
SEG=w01_20230702185753_r15
GATE=0.60
export PATH="$HOME/.local/bin:$PATH"

log() { echo "[$(date +%T)] $*"; }
need_steps() { [ "${STEPS:-}" = 12000 ] || { echo "STEPS must be 12000 (PREREG section 4); got '${STEPS:-}'"; exit 2; }; }

# same content as the reference build: nonzero pixel counts per arm, array and level must match
check_manifest() {  # built reference
    $PY - "$1" "$2" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = [(k, arm) for k in b["segments"] for arm in b["segments"][k]["nonzero_per_level"]
       if a["segments"][k]["nonzero_per_level"][arm] != b["segments"][k]["nonzero_per_level"][arm]]
print(f"{sys.argv[1]}: label arms match the reference build" if not bad else f"MISMATCH {bad}")
sys.exit(1 if bad else 0)
EOF
}

prepare() {
    need_steps
    mkdir -p "$WORK" "$RUNS" "$CACHE"
    if [ "$CPU_DRYRUN" = 1 ]; then
        log "CPU_DRYRUN: no GPU or driver check, no venv install, no villa clone"
        [ -x "$VENV/bin/python" ] || { echo "CPU_DRYRUN needs an existing VENV ($VENV)"; exit 2; }
        [ -d "$SRC/vesuvius/ink_detection" ] || { echo "CPU_DRYRUN needs an existing villa tree ($SRC)"; exit 2; }
        [ -f "$VILLA/REV" ] && echo "villa $(cat "$VILLA/REV") (archive)"
    else
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
        # torch 2.12.1 from PyPI is a CUDA 13.0 build; CUDA 13.x runs on drivers >= 580 only
        local drv; drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
        [ "$drv" -ge 580 ] || { echo "NVIDIA driver $drv < 580: CUDA 13 needs 580 or newer"; exit 2; }
        if [ ! -d "$VILLA/.git" ]; then
            git clone -q --filter=blob:none --sparse https://github.com/ScrollPrize/villa "$VILLA"
            (cd "$VILLA" && git sparse-checkout set vesuvius/src && git checkout -q "$VILLA_COMMIT")
        fi
        local head; head=$(git -C "$VILLA" rev-parse HEAD)
        case $head in "$VILLA_COMMIT"*) echo "villa $head";; *) echo "villa is at $head, not $VILLA_COMMIT"; exit 2;; esac
        if [ ! -x "$VENV/bin/python" ]; then
            command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
            uv venv -q --python "$PYVER" "$VENV"
            uv pip install -q --python "$VENV/bin/python" -r "$HERE/scripts/pod_requirements.txt"
        fi
        $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported())"
        $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available: driver too old for this torch build?')"
    fi
    $PY -c "import sys; print('python', sys.version.split()[0])"
    # public inputs: tifxyz + label TIFFs of w01 (about 500 MB); fetch_files.py skips files already present
    local P="--path ink/phercparis4/$SEG"
    $PY "$HERE/scripts/fetch_files.py" "$RAW" "$WORK/data/ledger.tsv" \
        $P/meta.json $P/x.tif $P/y.tif $P/z.tif $P/${SEG}_inklabels.tif $P/${SEG}_supervision_mask.tif
    (cd "$RAW" && sha256sum -c "$HERE/logs/raw_sha256.txt")
    if [ ! -f "$LABELS/manifest.json" ]; then
        log "building label arms AS, C0"
        $PY "$HERE/scripts/build_labels.py" "$RAW" "$HERE/regions.json" "$SRC" "$LABELS" --arms AS,C0
    fi
    check_manifest "$LABELS/manifest.json" "$HERE/labels/manifest_full.json"
    if [ ! -f "$EVAL_LABELS/manifest.json" ]; then
        log "building eval label folders E0, EAS"
        $PY "$HERE/scripts/build_eval_labels.py" "$RAW" "$HERE/regions.json" "$SRC" "$EVAL_LABELS" --arms E0,EAS
    fi
    check_manifest "$EVAL_LABELS/manifest.json" "$HERE/labels/manifest_eval.json"
    $PY "$HERE/scripts/check_eval_labels.py" "$EVAL_LABELS" "$HERE/labels/manifest_full.json" "$RAW" "$HERE/regions.json" | tail -1 \
        | grep -qx ALL_OK || { echo "check_eval_labels.py did not print ALL_OK"; exit 1; }
    log "eval label folders checked (E0 equals C0 byte for byte, EAS equals the shipped labels in its boxes)"
    for seed in $SEEDS; do
        for arm in C0 AS; do
            mkdir -p "$RUNS/${arm}_s$seed"
            $PY "$HERE/scripts/mkconfig.py" gpu "$TEACHER" "$LABELS/$arm" "$RUNS/${arm}_s$seed/out" "$seed" \
                "$RUNS/${arm}_s$seed/config.json" --steps "$STEPS" --cache-dir "$CACHE" --workers "$WORKERS" --max-steps-mode "$MAXSTEPS_MODE"
        done
        mkdir -p "$RUNS/eval_C0_s$seed" "$RUNS/eval_EAS_s$seed"
        $PY "$HERE/scripts/mkconfig.py" gpu "$TEACHER" "$LABELS/C0" "$RUNS/eval_C0_s$seed/out" "$seed" \
            "$RUNS/eval_C0_s$seed/config.json" --steps "$STEPS" --cache-dir "$CACHE" --workers "$WORKERS" --max-steps-mode "$MAXSTEPS_MODE"
        $PY "$HERE/scripts/eval_config_for.py" "$RUNS/eval_C0_s$seed/config.json" EAS "$RUNS/eval_EAS_s$seed/config.json"
    done
    du -sh "$RAW" "$LABELS" "$EVAL_LABELS"
}

counts() {  # CPU only: the trainer stages each config, builds the model, and finds its patches
    for arm in C0 AS; do
        log "counts ${arm}_s27"
        $PY "$HERE/scripts/check_gpu_config.py" "$SRC" "$RUNS/${arm}_s27/config.json"
    done
    $PY "$HERE/scripts/count_val_patches.py" "$SRC" "$RUNS/eval_C0_s27/config.json" "$RUNS/eval_EAS_s27/config.json"
}

train_one() {  # dir
    local d=$1
    [ -f "$d/train.done" ] && return 0
    log "train $(basename "$d")"
    date +%s > "$d/train.start"
    $PY "$HERE/scripts/launch.py" "$SRC" "$d/config.json" > "$d/stdout.log" 2> "$d/stderr.log"
    date +%s > "$d/train.done"
}

eval_one() {  # dir seed
    local d=$1 seed=$2 ck
    ck=$(ls "$d"/out/ckpt_*.pth | tail -1)
    if [ ! -f "$d/eval.json" ]; then
        log "eval $(basename "$d") on C0 labels $ck"
        $PY "$HERE/scripts/eval_heldout.py" "$SRC" "$ck" "$RUNS/eval_C0_s$seed/config.json" "$d/eval.json" \
            --batch 2 --workers "$WORKERS" > "$d/eval_stdout.log" 2> "$d/eval_stderr.log"
    fi
    if [ ! -f "$d/eval_AS.json" ]; then
        log "eval $(basename "$d") on labels as shipped $ck"
        $PY "$HERE/scripts/eval_heldout.py" "$SRC" "$ck" "$RUNS/eval_EAS_s$seed/config.json" "$d/eval_AS.json" \
            --batch 2 --workers "$WORKERS" > "$d/eval_AS_stdout.log" 2> "$d/eval_AS_stderr.log"
    fi
}

one() {  # arm seed
    local d=$RUNS/${1}_s$2
    [ -f "$d/config.json" ] || { echo "$d/config.json missing: run prepare with this seed"; exit 2; }
    train_one "$d"
    eval_one "$d" "$2"
}

gate() {
    [ -f "$RUNS/C0_s27/eval.json" ] || { echo "gate: C0 s27 not scored yet"; return 2; }
    $PY - "$RUNS/C0_s27/eval.json" "$SEG" "$GATE" <<'EOF'
import json, sys
auc = json.load(open(sys.argv[1]))["metrics"][sys.argv[2]]["auc"]
print(f"gate: C0 seed 27 w01 AUC {auc:.4f} (gate {sys.argv[3]})")
sys.exit(0 if auc >= float(sys.argv[3]) else 1)
EOF
}

run() {
    one C0 27
    one AS 27
    local rc=0; gate || rc=$?
    if [ $rc -ne 0 ]; then
        log "gate not passed (rc $rc): w01 does not learn in this setup; seeds 28 to 30 are not run (PREREG section 5)"
        summarize; return 0
    fi
    for seed in $SEEDS; do
        [ "$seed" = 27 ] && continue
        one C0 "$seed"
        one AS "$seed"
    done
    summarize
}

summarize() {
    $PY "$HERE/scripts/decide_w01.py" "$RUNS" "$WORK/results.md"
}

case ${1:-} in
    prepare) prepare ;;
    counts) counts ;;
    one) one "${2:?arm}" "${3:?seed}" ;;
    gate) gate ;;
    run) run ;;
    summarize) summarize ;;
    *) echo "usage: $0 prepare|counts|one ARM SEED|gate|run|summarize"; exit 2 ;;
esac
