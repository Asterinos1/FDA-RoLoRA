#!/usr/bin/env bash
# ==============================================================================
# Runner for FDA-RoLoRA Dual-Track Calibrated Controller Experiments
#
# Supports:
#   ./run_calibration_experiments.sh plan
#   ./run_calibration_experiments.sh run
#   ./run_calibration_experiments.sh status
#   ./run_calibration_experiments.sh stop <CONTROL_ID>
#   ./run_calibration_experiments.sh drop <CONTROL_ID>
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect REPO_ROOT whether script is in fixed_calibration/ or root
if [ -f "${SCRIPT_DIR}/thesis_experiments.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
    EXPERIMENT_PY="${SCRIPT_DIR}/fixed_calibration/thesis_calibration_experiments.py"
    if [ ! -f "${EXPERIMENT_PY}" ]; then
        EXPERIMENT_PY="${SCRIPT_DIR}/thesis_calibration_experiments.py"
    fi
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    EXPERIMENT_PY="${SCRIPT_DIR}/thesis_calibration_experiments.py"
fi

PYTHON_BIN="$(which python3 || which python)"

# Detect if Windows python is called from WSL/Git-Bash
PY_REPO_ROOT="${REPO_ROOT}"
PY_EXPERIMENT_PY="${EXPERIMENT_PY}"
if [[ "${PYTHON_BIN}" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
    PY_REPO_ROOT="$(wslpath -w "${REPO_ROOT}")"
    PY_EXPERIMENT_PY="$(wslpath -w "${EXPERIMENT_PY}")"
fi

export PYTHONPATH="${PY_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Concurrency defaults (override via environment if needed)
export THESIS_CALIB_JOBS_PER_GPU="${THESIS_CALIB_JOBS_PER_GPU:-3}"
export THESIS_CALIB_BATCH64_JOBS_PER_GPU="${THESIS_CALIB_BATCH64_JOBS_PER_GPU:-2}"

ACTION="${1:-plan}"
shift || true

case "${ACTION}" in
    plan)
        echo "=== Displaying Calibration Experiment Plan ==="
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group plan "$@"
        ;;
    run|start)
        echo "=== Launching Calibration Experiment Scheduler ==="
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
        REASON="${*:-operator stopped calibration run}"
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
        REASON="${*:-operator dropped calibration run permanently}"
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group drop --run-id "${TARGET_ID}" --reason "${REASON}"
        ;;
    dropped)
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group dropped "$@"
        ;;
    *)
        echo "Usage: $0 {plan|start|run|status|stop|drop|dropped} [options...]"
        exit 1
        ;;
esac
