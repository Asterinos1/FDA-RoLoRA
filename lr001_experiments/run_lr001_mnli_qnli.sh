#!/usr/bin/env bash
# ==============================================================================
# GPU Server Runner for LR=0.01 RoLoRA vs. FDA-RoLoRA Experiments
#
# Datasets: MNLI, QNLI
# Methods:  RoLoRA vs. FDA-RoLoRA (DualTrack, Old Controller, CRATIO=0.5)
# Budget:   P2 only (layers 7-11)
# Epochs:   E=5, 10, 20 (FDA tiers: [3, 10], [3, 20], [3, 40])
# Clients:  C=20 (R=50), C=50 (R=30)
# Seeds:    42, 43, 44
# Total:    72 configurations (or subset via flags)
#
# Usage Examples:
#   ./run_lr001_mnli_qnli.sh --dry-run                          # Preview full 72-run queue
#   ./run_lr001_mnli_qnli.sh                                    # Run all 72 experiments
#   ./run_lr001_mnli_qnli.sh --dataset mnli --clients 20        # Run pilot wave (18 runs)
#   ./run_lr001_mnli_qnli.sh --dataset mnli                     # Run MNLI only (36 runs)
#   ./run_lr001_mnli_qnli.sh --clients 20                       # Run C=20 only (36 runs)
#   ./run_lr001_mnli_qnli.sh --gpus 0,1 --jobs-per-gpu 3        # Explicit GPU & lane config
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve REPO_ROOT and RUNNER_PY whether script is in lr001_experiments/ or repo root
if [ -f "${SCRIPT_DIR}/thesis_experiments.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
    RUNNER_PY="${SCRIPT_DIR}/lr001_experiments/run_lr001_mnli_qnli.py"
    if [ ! -f "${RUNNER_PY}" ]; then
        RUNNER_PY="${SCRIPT_DIR}/run_lr001_mnli_qnli.py"
    fi
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    RUNNER_PY="${SCRIPT_DIR}/run_lr001_mnli_qnli.py"
fi

# Select Python binary (respects $PYTHON or active conda/virtualenv)
if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "ERROR: Neither python3 nor python found in PATH." >&2
    exit 1
fi

# Detect Windows/WSL path mapping if using Windows Python binary
PY_REPO_ROOT="${REPO_ROOT}"
PY_RUNNER="${RUNNER_PY}"
if [[ "$("${PYTHON_BIN}" -c "import sys; print(sys.platform)" 2>/dev/null || true)" == "win32" ]] && command -v wslpath >/dev/null 2>&1; then
    PY_REPO_ROOT="$(wslpath -w "${REPO_ROOT}")"
    PY_RUNNER="$(wslpath -w "${RUNNER_PY}")"
fi

# Environment variables for PyTorch & CUDA optimization
export PYTHONPATH="${PY_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

# Handle operator commands: status, stop, dry-run, help
ACTION="${1:-}"
if [ "${ACTION}" = "status" ]; then
    shift
    exec "${PYTHON_BIN}" "${PY_RUNNER}" --status "$@"
elif [ "${ACTION}" = "stop" ]; then
    shift
    if [ $# -eq 0 ]; then
        echo "Usage: ./run_lr001_mnli_qnli.sh stop <CONTROL_ID or RUN_ID>" >&2
        exit 1
    fi
    exec "${PYTHON_BIN}" "${PY_RUNNER}" --stop "$@"
fi

for arg in "$@"; do
    if [[ "${arg}" == "--dry-run" || "${arg}" == "-h" || "${arg}" == "--help" || "${arg}" == "--status" || "${arg}" == "--stop" ]]; then
        exec "${PYTHON_BIN}" "${PY_RUNNER}" "$@"
    fi
done

# Ensure logs directory exists for real execution
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/run_lr001_${TIMESTAMP}.log"

echo "========================================================================"
echo " Starting LR=0.01 RoLoRA vs. FDA-RoLoRA Experiment Suite"
echo " Date:         $(date)"
echo " Repo Root:    ${REPO_ROOT}"
echo " Runner:       ${RUNNER_PY}"
echo " Python:       $("${PYTHON_BIN}" --version 2>&1)"
echo " Log File:     ${LOG_FILE}"
echo " Arguments:    $*"
echo "========================================================================"

# Execute with stdout + stderr tee'd to console and timestamped log file
"${PYTHON_BIN}" "${PY_RUNNER}" "$@" 2>&1 | tee -a "${LOG_FILE}"
