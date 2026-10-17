#!/usr/bin/env bash
# ==============================================================================
# GPU Server Runner for RoLoRA vs. FDA-RoLoRA Threshold Controller Experiments
#
# Datasets:   MNLI (default), QNLI (optional)
# Client:     C=20, R=50 (default), C=50, R=30 (optional)
# Methods:    1. RoLoRA baselines: E=15, E=20
#             2. FDA-RoLoRA kappa=0.5 (Emax/2): E in [3, 30], [3, 40]
#             3. FDA-RoLoRA kappa=1.0 (Emax/1): E in [3, 30], [3, 40] (at very end)
# Learning:   eta = 0.01 (client SGD), server SGD (lr = 1.0)
# Stopping:   Strict Early Stopping (ES0.60 for MNLI, MR=6, P=3, D=0.005, W=3)
#
# Usage Examples:
#   ./run_threshold_experiments.sh --dry-run                     # Preview full queue
#   ./run_threshold_experiments.sh --stage rolora --dry-run     # Preview RoLoRA only
#   ./run_threshold_experiments.sh --stage rolora               # Execute RoLoRA baselines only
#   ./run_threshold_experiments.sh --stage fda_k05              # Execute FDA (kappa=0.5) only
#   ./run_threshold_experiments.sh --stage fda_k1               # Execute FDA (kappa=1.0) only
#   ./run_threshold_experiments.sh                              # Execute all 3 stages in order
#   ./run_threshold_experiments.sh --seeds 42,43                # 2 seeds instead of 3
#   ./run_threshold_experiments.sh --gpus 0,1 --jobs-per-gpu 3  # Custom GPU & lane setup
#   ./run_threshold_experiments.sh status                       # Check active runs
#   ./run_threshold_experiments.sh stop <CONTROL_ID or RUN_ID>  # Stop a running job
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve REPO_ROOT and RUNNER_PY whether script is in threshold_controller_experiments/ or repo root
if [ -f "${SCRIPT_DIR}/thesis_experiments.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
    RUNNER_PY="${SCRIPT_DIR}/threshold_controller_experiments/run_threshold_experiments.py"
    if [ ! -f "${RUNNER_PY}" ]; then
        RUNNER_PY="${SCRIPT_DIR}/run_threshold_experiments.py"
    fi
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    RUNNER_PY="${SCRIPT_DIR}/run_threshold_experiments.py"
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

# Environment variables for PyTorch & CUDA execution
export PYTHONPATH="${PY_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

# Handle operator commands: start, run, plan, status, stop, dry-run, help
ACTION="${1:-}"

case "${ACTION}" in
    plan)
        shift || true
        exec "${PYTHON_BIN}" "${PY_RUNNER}" --dry-run "$@"
        ;;
    start|run)
        shift || true
        # Continues execution below
        ;;
    status)
        shift || true
        exec "${PYTHON_BIN}" "${PY_RUNNER}" --status "$@"
        ;;
    stop)
        shift || true
        if [ $# -eq 0 ]; then
            echo "Usage: ./run_threshold_experiments.sh stop <CONTROL_ID or RUN_ID>" >&2
            exit 1
        fi
        exec "${PYTHON_BIN}" "${PY_RUNNER}" --stop "$@"
        ;;
esac

for arg in "$@"; do
    if [[ "${arg}" == "--dry-run" || "${arg}" == "-h" || "${arg}" == "--help" || "${arg}" == "--status" || "${arg}" == "--stop" ]]; then
        exec "${PYTHON_BIN}" "${PY_RUNNER}" "$@"
    fi
done

# Ensure logs directory exists
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/run_threshold_experiments_${TIMESTAMP}.log"

echo "========================================================================"
echo " Starting RoLoRA vs. FDA-RoLoRA Threshold Controller Experiments"
echo " Date:         $(date)"
echo " Repo Root:    ${REPO_ROOT}"
echo " Runner:       ${RUNNER_PY}"
echo " Python:       $("${PYTHON_BIN}" --version 2>&1)"
echo " Log File:     ${LOG_FILE}"
echo " Arguments:    $*"
echo "========================================================================"

# Execute and tee output to stdout and logfile
exec "${PYTHON_BIN}" "${PY_RUNNER}" "$@" 2>&1 | tee -a "${LOG_FILE}"
