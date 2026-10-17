#!/usr/bin/env bash
# ==============================================================================
# Complete Master Thesis Experiment Runner (Polytechnix GPU Scheduler)
#
# Protocol:
#   - Stage 1: Batch-32 IID tasks (QQP, QNLI, MNLI across C=20, C=50, Seeds 42, 43)
#   - Stage 2: Batch-32 Non-IID tasks (QNLI, MNLI alpha=0.5 across C=20, C=50, Seeds 42, 43)
#   - Stage 3: Batch-64 IID draining (SST-2 across C=20, C=50, Seeds 42, 43)
#
# Concurrency:
#   - Up to 6 parallel jobs per GPU (12 parallel slots total on 2 A10 GPUs)
#   - Strict safety cap of 2 SST-2 (batch-64) jobs per GPU to eliminate OOM
#
# Usage:
#   ./run_master_experiments.sh plan
#   ./run_master_experiments.sh start
#   ./run_master_experiments.sh status
#   ./run_master_experiments.sh stop <CONTROL_ID>
#   ./run_master_experiments.sh drop <CONTROL_ID>
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect REPO_ROOT whether script is in master_experiments/ or repo root
if [ -f "${SCRIPT_DIR}/thesis_experiments.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
    EXPERIMENT_PY="${SCRIPT_DIR}/master_experiments/thesis_master_experiments.py"
    if [ ! -f "${EXPERIMENT_PY}" ]; then
        EXPERIMENT_PY="${SCRIPT_DIR}/thesis_master_experiments.py"
    fi
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    EXPERIMENT_PY="${SCRIPT_DIR}/thesis_master_experiments.py"
fi

# Auto-detect conda environment 'gpu_server_env' if available
PYTHON_BIN=""
if [ -n "${CONDA_PREFIX:-}" ] && [[ "${CONDA_PREFIX}" == *"gpu_server_env"* ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "${CONDA_BASE}" ] && [ -x "${CONDA_BASE}/envs/gpu_server_env/bin/python" ]; then
        PYTHON_BIN="${CONDA_BASE}/envs/gpu_server_env/bin/python"
    fi
fi
if [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(which python3 || which python)"
fi

# Handle WSL / Git-Bash path conversions
PY_REPO_ROOT="${REPO_ROOT}"
PY_EXPERIMENT_PY="${EXPERIMENT_PY}"
if [[ "${PYTHON_BIN}" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
    PY_REPO_ROOT="$(wslpath -w "${REPO_ROOT}")"
    PY_EXPERIMENT_PY="$(wslpath -w "${EXPERIMENT_PY}")"
fi

export PYTHONPATH="${PY_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Concurrency defaults for Polytechnix A10 GPUs (3 jobs/GPU batch-32, 2 jobs/GPU batch-64)
export THESIS_MASTER_JOBS_PER_GPU="${THESIS_MASTER_JOBS_PER_GPU:-3}"
export THESIS_MASTER_BATCH64_JOBS_PER_GPU="${THESIS_MASTER_BATCH64_JOBS_PER_GPU:-2}"

ACTION="${1:-plan}"
shift || true

case "${ACTION}" in
    plan)
        echo "=== Displaying Complete Master Thesis Experiment Plan ==="
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group plan "$@"
        ;;
    run|start)
        echo "=== Launching Complete Master Thesis Experiment Scheduler ==="
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group run "$@"
        ;;
    status)
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group status "$@"
        ;;
    stop)
        if [ $# -lt 1 ]; then
            echo "Error: Run ID or Control ID required to stop."
            echo "Usage: $0 stop <ID> [reason]"
            exit 1
        fi
        TARGET_ID="$1"
        shift || true
        REASON="${*:-operator stopped run}"
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group stop --run-id "${TARGET_ID}" --reason "${REASON}"
        ;;
    drop)
        if [ $# -lt 1 ]; then
            echo "Error: Run ID or Control ID required to drop."
            echo "Usage: $0 drop <ID> [reason]"
            exit 1
        fi
        TARGET_ID="$1"
        shift || true
        REASON="${*:-operator dropped run permanently}"
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group drop --run-id "${TARGET_ID}" --reason "${REASON}"
        ;;
    *)
        echo "Usage: $0 {plan|start|run|status|stop|drop} [options...]"
        exit 1
        ;;
esac
