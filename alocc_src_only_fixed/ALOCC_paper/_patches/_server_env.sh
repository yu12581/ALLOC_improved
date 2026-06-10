#!/bin/bash
# _server_env.sh — Linux server path auto-configuration
# Source this from any runner script:
#   source "$(dirname "$0")/_server_env.sh"
#
# Override any path by setting the corresponding env var before sourcing.
# Example:
#   export ALOCC_PYTHON=/home/guyf/.miniconda3/envs/alocc/bin/python

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PAPER_DIR="${ALOCC_PAPER_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export REPO_DIR="${ALOCC_REPO_DIR:-$(cd "$PAPER_DIR/../ALOCC-master" && pwd)}"
export PYTHON="${ALOCC_PYTHON:-/home/guyf/.miniconda3/envs/alocc/bin/python}"
export RUNNER="${ALOCC_RUNNER:-$REPO_DIR/run_paper_mnist_figure6_7.py}"
export BASELINE_RUNNER="${ALOCC_BASELINE_RUNNER:-$REPO_DIR/mnist_experiment_runner.py}"

# Verify key paths exist
if [ ! -f "$PYTHON" ]; then
    echo "[_server_env] WARNING: Python not found at $PYTHON"
    echo "[_server_env] Set ALOCC_PYTHON env var to the correct path"
fi
if [ ! -d "$REPO_DIR" ]; then
    echo "[_server_env] ERROR: REPO_DIR not found: $REPO_DIR"
    exit 1
fi
for f in "$RUNNER" "$BASELINE_RUNNER"; do
    if [ ! -f "$f" ]; then
        echo "[_server_env] WARNING: runner not found: $f"
    fi
done

echo "[_server_env] PAPER_DIR=$PAPER_DIR"
echo "[_server_env] REPO_DIR=$REPO_DIR"
echo "[_server_env] PYTHON=$PYTHON"
