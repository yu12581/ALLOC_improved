#!/bin/bash
# 消融实验：CBAM vs S1 vs CBAM+S1
# 4组 × 3 digits(0/2/8) × 2 seeds(42/2026) = 24 runs，约 20 分钟
set -e

PYTHON="C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/python.exe"
RUNNER="mnist_experiment.py"
BASE="runs/ablation_cbam_s1"
LOG="runs/ablation_cbam_s1.log"

mkdir -p "$BASE"
> "$LOG"

DIGITS=(0 2 8)
SEEDS=(42 2026)
COMMON="--variant alocc_loss --epochs 40 --train-count 4096 --batch-size 64
        --out-per-class-count 300 --noise-std 0.31 --r-alpha 0.2
        --d-outclass-loss-scale 0.1 --selection-strategy best_auc"

total=$(( ${#DIGITS[@]} * ${#SEEDS[@]} * 4 ))
n=0
failed=()

run_one() {
    local tag=$1; local digit=$2; local seed=$3; shift 3
    local extra="$@"
    local out="$BASE/${tag}_d${digit}_s${seed}"
    n=$(( n + 1 ))

    if [ -f "${out}/summary.json" ]; then
        echo "[$n/$total] SKIP ${tag}_d${digit}_s${seed}"
        return
    fi

    echo "[$n/$total] ${tag} digit=${digit} seed=${seed}"
    echo "[$n/$total] ${tag} d${digit}_s${seed}" >> "$LOG"

    "$PYTHON" $RUNNER $COMMON \
        --specific "$digit" --seed "$seed" \
        --output-dir "$out" \
        $extra \
        && echo "OK ${tag}_d${digit}_s${seed}" >> "$LOG" \
        || { echo "FAIL ${tag}_d${digit}_s${seed}" >> "$LOG"
             failed+=("${tag}_d${digit}_s${seed}"); }
}

for seed in "${SEEDS[@]}"; do
    for digit in "${DIGITS[@]}"; do
        # A: 无 S1，无 CBAM（对照基线）
        run_one "A_base" "$digit" "$seed"

        # B: S1 only（rank=8, dropout=0.3）
        run_one "B_s1" "$digit" "$seed" \
            --bottleneck-rank 8 --bottleneck-dropout 0.3

        # C: CBAM only
        run_one "C_cbam" "$digit" "$seed" \
            --use-cbam

        # D: S1 + CBAM
        run_one "D_s1_cbam" "$digit" "$seed" \
            --bottleneck-rank 8 --bottleneck-dropout 0.3 --use-cbam
    done
done

echo ""
echo "=== 完成 $n/$total  失败 ${#failed[@]} ==="
[ ${#failed[@]} -gt 0 ] && echo "失败: ${failed[*]}"
