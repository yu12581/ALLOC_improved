#!/bin/bash
# [S1D 40ep-Oracle] 50-run matrix
# S1 bottleneck + L1-hinge distortion + D outclass — extended to 40 epochs.
#
# Usage:
#   bash run_s1d_40ep_oracle.sh --smoke    # digit=1 seed=42 only (~8 min)
#   bash run_s1d_40ep_oracle.sh           # full 50 runs (~90 min)

set -euo pipefail

source "$(dirname "$0")/_server_env.sh"

SMOKE=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

SEEDS=(42 1337 2026 7 123)
DIGITS=(0 1 2 3 4 5 6 7 8 9)

if $SMOKE; then
    SEEDS=(42)
    DIGITS=(1)
fi

OUTPUT_ROOT="${ALOCC_S1D_OUTPUT:-$PAPER_DIR}"
LOGS_DIR="$OUTPUT_ROOT/_logs"
mkdir -p "$LOGS_DIR"

TOTAL=$(( ${#SEEDS[@]} * ${#DIGITS[@]} ))
CELL_IDX=0
TOTAL_START=$(date +%s)
FAILED=()

echo "[S1D-40EP-ORACLE] total=$TOTAL runs"

get_class_config() {
    local c=$1
    if [ "$c" -eq 7 ]; then
        echo "4 0.5"
    else
        echo "8 0.3"
    fi
}

for C in "${DIGITS[@]}"; do
    read -r RANK P <<< "$(get_class_config "$C")"

    OUTLIERS=""
    for d in {0..9}; do
        [ "$d" -ne "$C" ] && OUTLIERS="$OUTLIERS $d"
    done
    OUTLIERS="${OUTLIERS# }"

    for SEED in "${SEEDS[@]}"; do
        CELL_IDX=$((CELL_IDX + 1))
        OUT_DIR="$OUTPUT_ROOT/s1d_40ep_c${C}_seed${SEED}_oracle"
        LOG_FILE="$LOGS_DIR/c${C}_s${SEED}_40ep.log"

        echo "=== [$CELL_IDX/$TOTAL] c=$C r=$RANK p=$P seed=$SEED ==="
        if $DRY_RUN; then
            echo "  DRYRUN: $OUT_DIR"
            continue
        fi

        rm -rf "$OUT_DIR"
        START_RUN=$(date +%s)

        set +e
        "$PYTHON" "$RUNNER" \
            --output-dir "$OUT_DIR" \
            --variant alocc_loss \
            --specific "$C" \
            --seed "$SEED" \
            --epochs 40 \
            --train-count 4096 \
            --test-inlier-count 200 \
            --batch-size 64 \
            --eval-batch-size 128 \
            --noise-std 0.31 \
            --r-alpha 0.2 \
            --lr 0.002 \
            --d-outclass-loss-scale 0.1 \
            --out-per-class-count 300 \
            --selection-strategy refined_auc \
            --selection-epoch-start 1 \
            --selection-epoch-end 40 \
            --distortion-alpha 1.0 \
            --distortion-beta 1.0 \
            --paper-score-normalization relative \
            --bottleneck-rank "$RANK" \
            --bottleneck-dropout "$P" \
            --bottleneck-noise-type dropout \
            --triplet-count 12 \
            --figure7-sample-count 40 \
            --outlier-labels $OUTLIERS \
            > "$LOG_FILE" 2> "${LOG_FILE}.err"
        EXIT_CODE=$?
        set -e

        DUR=$(( $(date +%s) - START_RUN ))
        echo "=== [$CELL_IDX/$TOTAL] END dur=${DUR}s exit=$EXIT_CODE ==="

        SUM_PATH="$OUT_DIR/experiment/summary.json"
        if [ $EXIT_CODE -ne 0 ] || [ ! -f "$SUM_PATH" ]; then
            echo "WARNING: FAILED c=$C seed=$SEED"
            FAILED+=("c${C}_s${SEED}")
        fi
    done
done

TOTAL_DUR=$(( ($(date +%s) - TOTAL_START) / 60 ))
echo "[S1D-40EP-ORACLE] done. elapsed=${TOTAL_DUR}min failed=${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi
