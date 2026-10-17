#!/usr/bin/env bash
# ==============================================================================
# Runner for Non-IID Dirichlet (alpha=0.5) Experiments (Polytechnix GPU Scheduler)
#
# Protocol:
#   - Dirichlet Non-IID alpha=0.5 across C=20 clients, R=50 rounds
#   - Methods: FedAvg-RoLoRA (E10, E20) & FDA-RoLoRA-DualTrack-Pure (Emax=20, 40 with k=1.0)
#   - Early stopping >= 60% of rounds (Round >= 30), 5-round window
#   - 6-lane continuous multi-GPU scheduler (3 jobs/GPU on Polytechnix A10s)
#
# Usage:
#   ./run_non_iid_experiments.sh plan
#   ./run_non_iid_experiments.sh start
#   ./run_non_iid_experiments.sh status
#   ./run_non_iid_experiments.sh stop <CONTROL_ID>
#   ./run_non_iid_experiments.sh drop <CONTROL_ID>
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect REPO_ROOT whether script is in non_iid_experiments/ or repo root
if [ -f "${SCRIPT_DIR}/thesis_experiments.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
    EXPERIMENT_PY="${SCRIPT_DIR}/non_iid_experiments/thesis_non_iid_experiments.py"
    if [ ! -f "${EXPERIMENT_PY}" ]; then
        EXPERIMENT_PY="${SCRIPT_DIR}/thesis_non_iid_experiments.py"
    fi
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    EXPERIMENT_PY="${SCRIPT_DIR}/thesis_non_iid_experiments.py"
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

# Concurrency defaults for Polytechnix A10 GPUs (3 jobs per GPU = 6 parallel jobs on 2 GPUs)
export THESIS_NONIID_JOBS_PER_GPU="${THESIS_NONIID_JOBS_PER_GPU:-3}"
export THESIS_NONIID_BATCH64_JOBS_PER_GPU="${THESIS_NONIID_BATCH64_JOBS_PER_GPU:-2}"

ACTION="${1:-plan}"
shift || true

case "${ACTION}" in
    plan)
        echo "=== Displaying Non-IID (alpha=0.5) Experiment Plan ==="
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group plan "$@"
        ;;
    run|start)
        echo "=== Launching Non-IID Continuous 6-Lane Scheduler ==="
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
        REASON="${*:-operator stopped non-iid run}"
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
        REASON="${*:-operator dropped non-iid run permanently}"
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group drop --run-id "${TARGET_ID}" --reason "${REASON}"
        ;;
    *)
        echo "Usage: $0 {plan|start|run|status|stop|drop} [options...]"
        exit 1
        ;;
esac
