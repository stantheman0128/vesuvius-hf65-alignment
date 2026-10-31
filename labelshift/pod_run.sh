#!/usr/bin/env bash
# Label-shift experiment (PREREG.md) on one rented Linux CUDA pod: Ubuntu, one NVIDIA GPU with bf16
# (RTX 3090 or newer) on NVIDIA driver 580 or newer (the PyPI torch 2.12.1 wheel is a CUDA 13.0 build),
# at least 8 vCPUs, 32 GB RAM and 60 GB disk. Copy pod_bundle.tar.gz (scripts/make_bundle.sh) to the
# pod and untar it; it holds this script, PREREG.md, regions.json, labels/manifest_full.json and the
# scripts. The label arms are rebuilt on the pod from public data and checked against that manifest.
# Idempotent: rerunning skips what is already there, finished runs are not redone.
#
#   bash pod_run.sh prepare     # uv + Python 3.14 env, villa 37d1897fb, public data, label arms, configs
#   bash pod_run.sh smoke       # 200 steps of C0 seed 27 on the GPU, prints s/step and projected GPU hours
#   bash pod_run.sh run         # pilot C0 s27, gate G0, then every arm and seed in PREREG order, scored
#   bash pod_run.sh summarize   # PREREG section 7 rules -> $WORK/results.md
#
# Env: WORK (default ~/labelshift), STEPS (6000), ARMS ("C0,AS,C8y,C4y,C2y"; append ",C8x,C4x,C2x" for
# the optional x curve), SEEDS ("27 28"), WORKERS (8), CAP_HOURS (45), FORCE=1 skips the cap check,
# MAXSTEPS_MODE (teacher; anneal ends the cosine schedule at STEPS, Amendment E), G0_WAIVED=1 skips the
# G0 gate (Amendment F).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-$HOME/labelshift}
VILLA_COMMIT=${VILLA_COMMIT:-37d1897fb}
STEPS=${STEPS:-6000}
ARMS=${ARMS:-C0,AS,C8y,C4y,C2y}
SEEDS=${SEEDS:-"27 28"}
WORKERS=${WORKERS:-8}
CAP_HOURS=${CAP_HOURS:-45}
MAXSTEPS_MODE=${MAXSTEPS_MODE:-teacher}  # teacher (PREREG) or anneal (Amendment E)
PYVER=3.14.6
VILLA=$WORK/villa
SRC=$VILLA/vesuvius/src
TEACHER=$SRC/vesuvius/ink_detection/configs/dino_guided_teacher.json
VENV=$WORK/venv
PY="$VENV/bin/python -I"
RAW=$WORK/data/raw
CACHE=$WORK/volume_cache
LABELS=$WORK/labels/full
RUNS=$WORK/runs
export PATH="$HOME/.local/bin:$PATH"

log() { echo "[$(date +%T)] $*"; }

prepare() {
    mkdir -p "$WORK" "$RUNS" "$CACHE"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    # torch 2.12.1 from PyPI is a CUDA 13.0 build; CUDA 13.x runs on drivers >= 580 only
    local drv; drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    [ "$drv" -ge 580 ] || { echo "NVIDIA driver $drv < 580: pick a pod image with driver 580 or newer (CUDA 13)"; exit 2; }
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ ! -d "$VILLA/.git" ]; then
        git clone -q --filter=blob:none --sparse https://github.com/ScrollPrize/villa "$VILLA"
        (cd "$VILLA" && git sparse-checkout set vesuvius/src && git checkout -q "$VILLA_COMMIT")
    fi
    (cd "$VILLA" && git log -1 --format='villa %H %cd')
    # The vesuvius "models" extra pulls monorepo C++ packages (volume-cartographer, vc-object-store) that
    # training does not import; install the package set the CPU smoke test ran with instead
    # (scripts/pod_requirements.txt, torch 2.12.1 from PyPI, which ships CUDA on Linux x86_64), and put
    # the villa tree on sys.path through scripts/launch.py, as the smoke test did.
    if [ ! -x "$VENV/bin/python" ]; then
        uv venv -q --python "$PYVER" "$VENV"
        uv pip install -q --python "$VENV/bin/python" -r "$HERE/scripts/pod_requirements.txt"
    fi
    $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported())"
    $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available: driver too old for this torch build?')"
    # public inputs: tifxyz + label TIFFs of the two segments (about 440 MB)
    local P="--path ink/phercparis4/w00_20231016151002" Q="--path ink/1667/w018_20240304144031_2um"
    $PY "$HERE/scripts/fetch_files.py" "$RAW" "$WORK/data/ledger.tsv" \
        $P/meta.json $P/x.tif $P/y.tif $P/z.tif $P/w00_20231016151002_inklabels.tif $P/w00_20231016151002_supervision_mask.tif \
        $Q/meta.json $Q/x.tif $Q/y.tif $Q/z.tif $Q/w018_20240304144031_inklabels.tif $Q/w018_20240304144031_supervision_mask.tif
    if [ ! -f "$LABELS/manifest.json" ]; then
        log "building label arms (about 20 min on 8 cores)"
        $PY "$HERE/scripts/build_labels.py" "$RAW" "$HERE/regions.json" "$SRC" "$LABELS" --arms "AS,C0,C2y,C4y,C8y,C2x,C4x,C8x"
    fi
    # same content as the local build: nonzero pixel counts per arm, array and level must match
    $PY - "$LABELS/manifest.json" "$HERE/labels/manifest_full.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = [(k, arm) for k in b["segments"] for arm in b["segments"][k]["nonzero_per_level"]
       if a["segments"][k]["nonzero_per_level"][arm] != b["segments"][k]["nonzero_per_level"][arm]]
print("label arms match the local build" if not bad else f"MISMATCH {bad}")
sys.exit(1 if bad else 0)
EOF
    for seed in $SEEDS; do
        for arm in ${ARMS//,/ }; do
            mkdir -p "$RUNS/${arm}_s$seed"
            $PY "$HERE/scripts/mkconfig.py" gpu "$TEACHER" "$LABELS/$arm" "$RUNS/${arm}_s$seed/out" "$seed" \
                "$RUNS/${arm}_s$seed/config.json" --steps "$STEPS" --cache-dir "$CACHE" --workers "$WORKERS" --max-steps-mode "$MAXSTEPS_MODE"
        done
        mkdir -p "$RUNS/eval_C0_s$seed"
        $PY "$HERE/scripts/mkconfig.py" gpu "$TEACHER" "$LABELS/C0" "$RUNS/eval_C0_s$seed/out" "$seed" \
            "$RUNS/eval_C0_s$seed/config.json" --steps "$STEPS" --cache-dir "$CACHE" --workers "$WORKERS" --max-steps-mode "$MAXSTEPS_MODE"
    done
    du -sh "$RAW" "$LABELS"
}

train_one() {  # dir
    local d=$1
    [ -f "$d/train.done" ] && return 0
    log "train $(basename "$d")"
    $PY "$HERE/scripts/launch.py" "$SRC" "$d/config.json" > "$d/stdout.log" 2> "$d/stderr.log"
    touch "$d/train.done"
}

eval_one() {  # dir seed
    local d=$1 seed=$2 ck
    [ -f "$d/eval.json" ] && return 0
    ck=$(ls "$d"/out/ckpt_*.pth | tail -1)
    log "eval $(basename "$d") $ck"
    $PY "$HERE/scripts/eval_heldout.py" "$SRC" "$ck" "$RUNS/eval_C0_s$seed/config.json" "$d/eval.json" \
        --batch 2 --workers "$WORKERS" > "$d/eval_stdout.log" 2> "$d/eval_stderr.log"
}

smoke() {
    local d=$RUNS/gpusmoke
    mkdir -p "$d"
    $PY "$HERE/scripts/mkconfig.py" gpu "$TEACHER" "$LABELS/C0" "$d/out" 27 "$d/config.json" \
        --steps 200 --cache-dir "$CACHE" --workers "$WORKERS" --benchmark-warmup 50 --max-steps-mode "$MAXSTEPS_MODE"
    rm -f "$d/train.done"
    train_one "$d"
    local nruns=$(( $(echo "${ARMS//,/ }" | wc -w) * $(echo $SEEDS | wc -w) ))
    $PY - "$d/out/benchmark_summary.json" "$STEPS" "$nruns" "$CAP_HOURS" <<'EOF' | tee "$d/projection.txt"
import json, sys
b = json.load(open(sys.argv[1])); steps, n, cap = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
sps = b["steps_per_second"]
per_run = steps / sps / 3600 * 1.10  # +10%: validation passes and the held-out eval
print(f"measured {sps:.3f} steps/s ({1 / sps:.2f} s/step, data wait {b.get('data_wait_seconds_per_step', float('nan')):.2f} s/step)")
print(f"projected {per_run:.2f} GPU h per run, {2 * per_run:.2f} per arm (2 seeds), {n * per_run:.1f} for {n} runs; cap {cap}")
print("PROJECTION_OK" if n * per_run <= cap else "PROJECTION_OVER_CAP")
EOF
}

run() {
    if [ "${FORCE:-0}" != 1 ]; then
        grep -q PROJECTION_OK "$RUNS/gpusmoke/projection.txt" 2>/dev/null \
            || { echo "run 'smoke' first, or the projection is over CAP_HOURS=$CAP_HOURS (FORCE=1 to override)"; exit 2; }
    fi
    local first=1
    for seed in $SEEDS; do
        for arm in ${ARMS//,/ }; do
            local d=$RUNS/${arm}_s$seed
            train_one "$d"
            eval_one "$d" "$seed"
            if [ $first = 1 ] && [ "${G0_WAIVED:-0}" = 1 ]; then
                first=0; log "G0 waived (Amendment F): $(basename "$d") is not gated"
            fi
            if [ $first = 1 ]; then
                first=0
                [ "$arm" = C0 ] && [ "$seed" = 27 ] || { echo "ARMS must start with C0 and SEEDS with 27 (pilot)"; exit 2; }
                $PY - "$d/eval.json" <<'EOF' || { log "gate G0 failed, stopping (PREREG section 7)"; summarize; exit 0; }
import json, sys
auc = json.load(open(sys.argv[1]))["metrics"]["pooled"]["auc"]
print(f"G0: C0 seed 27 pooled AUC {auc:.4f} (gate 0.70)")
sys.exit(0 if auc >= 0.70 else 1)
EOF
            fi
        done
    done
    summarize
}

summarize() {
    $PY "$HERE/scripts/decide.py" "$RUNS" "$WORK/results.md"
}

case ${1:-} in
    prepare) prepare ;;
    smoke) smoke ;;
    run) run ;;
    summarize) summarize ;;
    *) echo "usage: $0 prepare|smoke|run|summarize"; exit 2 ;;
esac
