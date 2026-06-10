#!/bin/bash
# [BASELINE B-default 20ep-Oracle] 50-run matrix
# Linux version — sources _server_env.sh for paths.
#
# Usage:
#   bash run_baseline_b_default_20ep.sh --smoke   # digit=1 seed=42 only
#   bash run_baseline_b_default_20ep.sh           # full 50 runs

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

SEEDS=(42 43 44 45 46)
DIGITS=(0 1 2 3 4 5 6 7 8 9)

if $SMOKE; then
    SEEDS=(42)
    DIGITS=(1)
fi

OUTPUT_ROOT="${ALOCC_BL_OUTPUT:-$REPO_DIR/baseline_b_default_20ep}"
mkdir -p "$OUTPUT_ROOT"
LOGS_DIR="$OUTPUT_ROOT/_logs"
mkdir -p "$LOGS_DIR"

TOTAL=$(( ${#SEEDS[@]} * ${#DIGITS[@]} ))
INDEX=0
STARTED=$(date +%s)
FAILED=()

echo "[BL-B-DEFAULT-20EP] output=$OUTPUT_ROOT  total=$TOTAL runs"

for SEED in "${SEEDS[@]}"; do
    for DIGIT in "${DIGITS[@]}"; do
        INDEX=$((INDEX + 1))

        OUTLIERS=""
        for d in {0..9}; do
            [ "$d" -ne "$DIGIT" ] && OUTLIERS="$OUTLIERS $d"
        done
        OUTLIERS="${OUTLIERS# }"

        RUN_DIR="$OUTPUT_ROOT/d${DIGIT}_s${SEED}"
        LOG_FILE="$LOGS_DIR/d${DIGIT}_s${SEED}.log"

        echo "[$INDEX/$TOTAL] digit=$DIGIT seed=$SEED"
        if $DRY_RUN; then
            echo "  DRYRUN: $PYTHON $BASELINE_RUNNER ..."
            continue
        fi

        mkdir -p "$RUN_DIR"

        START_RUN=$(date +%s)
        set +e
        "$PYTHON" "$BASELINE_RUNNER" \
            --variant alocc \
            --output-dir "$RUN_DIR" \
            --specific "$DIGIT" \
            --seed "$SEED" \
            --epochs 20 \
            --train-count 10000 \
            --test-inlier-count 200 \
            --batch-size 128 \
            --noise-std 0.31 \
            --r-alpha 0.2 \
            --lr 0.002 \
            --bottleneck-rank 0 \
            --selection-strategy refined_auc \
            --selection-epoch-start 1 \
            --selection-epoch-end 20 \
            --test-outlier-labels $OUTLIERS \
            > "$LOG_FILE" 2> "${LOG_FILE}.err"
        EXIT_CODE=$?
        set -e

        DUR=$(( $(date +%s) - START_RUN ))
        echo "  [$INDEX/$TOTAL] END dur=${DUR}s exit=$EXIT_CODE"

        if [ $EXIT_CODE -ne 0 ]; then
            echo "WARNING: FAILED digit=$DIGIT seed=$SEED exit=$EXIT_CODE"
            FAILED+=("d${DIGIT}_s${SEED}")
        fi
    done
done

ELAPSED=$(( $(date +%s) - STARTED ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
echo "[BL-B-DEFAULT-20EP] done. elapsed=${ELAPSED_MIN}min failed=${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi
