#!/usr/bin/env bash
# ==============================================================================
# Runner for Pure FDA-RoLoRA Toy Model Experiments (Polytechnix GPU Scheduler)
#
# Usage:
#   ./run_toy_experiments.sh plan
#   ./run_toy_experiments.sh start
#   ./run_toy_experiments.sh status
#   ./run_toy_experiments.sh stop <CONTROL_ID>
#   ./run_toy_experiments.sh drop <CONTROL_ID>
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect REPO_ROOT whether script is in toy/ or repo root
if [ -f "${SCRIPT_DIR}/thesis_experiments.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
    EXPERIMENT_PY="${SCRIPT_DIR}/toy/thesis_toy_experiments.py"
    if [ ! -f "${EXPERIMENT_PY}" ]; then
        EXPERIMENT_PY="${SCRIPT_DIR}/thesis_toy_experiments.py"
    fi
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    EXPERIMENT_PY="${SCRIPT_DIR}/thesis_toy_experiments.py"
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

# Concurrency defaults for Polytechnix A10 GPUs (4 jobs per GPU = 8 parallel slots across 2 GPUs)
export TOY_JOBS_PER_GPU="${TOY_JOBS_PER_GPU:-4}"

ACTION="${1:-plan}"
shift || true

case "${ACTION}" in
    plan)
        echo "=== Displaying Pure FDA-RoLoRA Toy Model Plan ==="
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group plan "$@"
        ;;
    plan-emax10|emax10)
        echo "=== Displaying FDA Pure (Emax=10, k=1.0 & k=0.5) Plan ==="
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group plan --emax 10 --ratios 1.0 0.5 "$@"
        ;;
    run|start)
        echo "=== Launching Pure FDA-RoLoRA Toy Model Scheduler ==="
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group run "$@"
        ;;
    start-emax10|run-emax10)
        echo "=== Launching FDA Pure (Emax=10, k=1.0 & k=0.5) Multi-GPU Scheduler ==="
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group run --emax 10 --ratios 1.0 0.5 "$@"
        ;;
    status)
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group status "$@"
        ;;
    status-emax10)
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group status --emax 10 --ratios 1.0 0.5 "$@"
        ;;
    stop)
        if [ $# -lt 1 ]; then
            echo "Error: Run ID or Control ID required to stop."
            echo "Usage: $0 stop <ID> [reason]"
            exit 1
        fi
        TARGET_ID="$1"
        shift || true
        REASON="${*:-operator stopped toy pure run}"
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
        REASON="${*:-operator dropped toy pure run permanently}"
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group drop --run-id "${TARGET_ID}" --reason "${REASON}"
        ;;
    *)
        echo "Usage: $0 {plan|start|run|status|stop|drop} [options...]"
        exit 1
        ;;
esac
