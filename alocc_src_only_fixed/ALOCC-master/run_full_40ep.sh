#!/bin/bash
set -e

PYTHON="C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/python.exe"
RUNNER="mnist_experiment.py"
BASE_DIR="runs/full_40ep_bestauc"
LOG="runs/full_40ep_bestauc_run.log"

mkdir -p "$BASE_DIR"
> "$LOG"

SEEDS=(42 1337 2026)
DIGITS=(0 1 2 3 4 5 6 7 8 9)
total=$(( ${#SEEDS[@]} * ${#DIGITS[@]} ))
done=0
skipped=0
failed=()

for seed in "${SEEDS[@]}"; do
    for digit in "${DIGITS[@]}"; do
        done=$(( done + 1 ))
        out="${BASE_DIR}/d${digit}_s${seed}"

        if [ -f "${out}/summary.json" ]; then
            echo "[$done/$total] SKIP d${digit}_s${seed} (already done)"
            skipped=$(( skipped + 1 ))
            continue
        fi

        echo "[$done/$total] digit=$digit seed=$seed -> $out"
        echo "[$done/$total] digit=$digit seed=$seed" >> "$LOG"

        "$PYTHON" "$RUNNER" \
            --variant alocc_loss \
            --specific "$digit" \
            --epochs 40 \
            --seed "$seed" \
            --train-count 4096 \
            --batch-size 64 \
            --out-per-class-count 300 \
            --noise-std 0.31 \
            --r-alpha 0.2 \
            --d-outclass-loss-scale 0.1 \
            --selection-strategy best_auc \
            --output-dir "$out" \
            && echo "OK: d${digit}_s${seed}" >> "$LOG" \
            || { echo "FAILED: d${digit}_s${seed}" >> "$LOG"; failed+=("d${digit}_s${seed}"); }
    done
done

echo ""
echo "=== 完成 $done/$total，跳过 $skipped，失败 ${#failed[@]} ==="
if [ ${#failed[@]} -gt 0 ]; then
    echo "失败列表: ${failed[*]}"
fi
