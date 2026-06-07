#!/bin/bash
# run_cpguard_openloop.sh
#
# Automates all six open-loop conditions (C1–C6) for CP-Guard evaluation
# on the Morecav 5-CAV dataset.
#
# Usage:
#   bash run_cpguard_openloop.sh [C1] [C2] [C3] [C4] [C5] [C6]
#   (no args = run all conditions)
#
# Results are written to:
#   $MODEL_DIR/AP_cpguard030507.yaml         (C1 – upper bound)
#   $MODEL_DIR/AP_cpguard_eps{e}.txt         (C2 – clean + defence)
#   $MODEL_DIR/AP_attack_*_nodef*.txt        (C3, C5 – no-defence attacked)
#   $MODEL_DIR/AP_attack_*_eps{e}.txt        (C4, C6 – defended attacked)

set -e

# ── Hardware ────────────────────────────────────────────────────────────────
# Restrict thread spawning to prevent OOM / CPU kills (≤1600% CPU, ≤30 GiB)
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ── Parameters ───────────────────────────────────────────────────────────────
MODEL_DIR="opencood/logs/codriving_morecav_finetune_2026_05_29"
EPS_LIST="0.08 0.04 0.02 0.01 0.005"
N_UPPER=5
MAX_FRAMES=1000
PGD_STEPS=10
ATK_EPS=0.2

# ── Condition selector (default: all) ─────────────────────────────────────
RUN_ALL=true
RUN=()
for arg in "$@"; do
    RUN+=("$arg")
    RUN_ALL=false
done
should_run() { $RUN_ALL || [[ " ${RUN[*]} " =~ " $1 " ]]; }

echo "Starting CP-Guard Open Loop Evaluation..."
echo "MODEL_DIR : $MODEL_DIR"
echo "EPS_LIST  : $EPS_LIST"
echo "GPU       : $CUDA_VISIBLE_DEVICES"
echo ""

# ── C1: No defence, no attack (upper-bound baseline) ─────────────────────
if should_run C1; then
    echo "=========================================="
    echo "C1 — No defence, no attack (upper bound)"
    echo "=========================================="
    python opencood/tools/inference_multiclass_cpguard.py \
        --model_dir "$MODEL_DIR" \
        --max_frames $MAX_FRAMES \
        --eps $EPS_LIST --n_upper $N_UPPER \
        --no_pasac
fi

# ── C2: CP-Guard defence, no attack ────────────────────────────────────────
if should_run C2; then
    echo "=========================================="
    echo "C2 — CP-Guard defence, no attack"
    echo "=========================================="
    python opencood/tools/inference_multiclass_cpguard.py \
        --model_dir "$MODEL_DIR" \
        --max_frames $MAX_FRAMES \
        --eps $EPS_LIST --n_upper $N_UPPER \
        --log_frames
fi

# ── C3: No defence + FP attack ─────────────────────────────────────────────
if should_run C3; then
    echo "=========================================="
    echo "C3 — No defence + FP attack"
    echo "=========================================="
    python opencood/tools/inference_multiclass_attack_cpguard.py \
        --model_dir "$MODEL_DIR" \
        --max_frames $MAX_FRAMES --pgd_steps $PGD_STEPS --attack_eps $ATK_EPS \
        --attack_goal fp --no_pasac \
        --eps $EPS_LIST --n_upper $N_UPPER
fi

# ── C4: CP-Guard defence + FP attack ───────────────────────────────────────
if should_run C4; then
    echo "=========================================="
    echo "C4 — CP-Guard defence + FP attack"
    echo "=========================================="
    python opencood/tools/inference_multiclass_attack_cpguard.py \
        --model_dir "$MODEL_DIR" \
        --attack_goal fp \
        --max_frames $MAX_FRAMES --pgd_steps $PGD_STEPS --attack_eps $ATK_EPS \
        --eps $EPS_LIST --n_upper $N_UPPER \
        --log_frames
fi

# ── C5: No defence + FN attack ─────────────────────────────────────────────
if should_run C5; then
    echo "=========================================="
    echo "C5 — No defence + FN attack"
    echo "=========================================="
    python opencood/tools/inference_multiclass_attack_cpguard.py \
        --model_dir "$MODEL_DIR" \
        --max_frames $MAX_FRAMES --pgd_steps $PGD_STEPS --attack_eps $ATK_EPS \
        --attack_goal fn --no_pasac \
        --eps $EPS_LIST --n_upper $N_UPPER
fi

# ── C6: CP-Guard defence + FN attack ───────────────────────────────────────
if should_run C6; then
    echo "=========================================="
    echo "C6 — CP-Guard defence + FN attack"
    echo "=========================================="
    python opencood/tools/inference_multiclass_attack_cpguard.py \
        --model_dir "$MODEL_DIR" \
        --attack_goal fn \
        --max_frames $MAX_FRAMES --pgd_steps $PGD_STEPS --attack_eps $ATK_EPS \
        --eps $EPS_LIST --n_upper $N_UPPER \
        --log_frames
fi

echo ""
echo "Open Loop Evaluation Complete!"
