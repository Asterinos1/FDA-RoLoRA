#!/usr/bin/env bash
# ==============================================================================
# Complete Thesis Filler Experiments Runner (Polytechnix GPU Scheduler)
#
# Scope:
#   - Tier 1: C=50 Non-IID missing k=1.0 runs (MNLI, QNLI, Seeds 42, 43) [4 runs]
#   - Tier 2: C=20 Non-IID deep-compute replication on Seed 43 (Emax=40, RoLoRA E20) [5 runs]
#   - Tier 3: Additional Non-IID tasks (QQP & SST-2 alpha=0.5 across C=20 & C=50) [24 runs]
#
# Optimized Early Stopping & Round Caps:
#   - Realistic Non-IID entry gates: QNLI C=50: 65%, MNLI C=50: 48%, QQP: 65%, SST2: 75-80%
#   - Activation round guard (MR): MR=12 for C=50, MR=20 for C=20
#   - Tighter Patience=3, Window=3, MinDelta=0.005
#   - Max round cap for C=50: R=24 (eliminates 40%+ unnecessary compute post-peak)
#
# Concurrency:
#   - Up to 3 parallel jobs per GPU (6 parallel slots total on 2 A10 GPUs)
#   - Strict safety cap of 2 SST-2 (batch-64) jobs per GPU to eliminate OOM
#
# Usage:
#   ./run_filler_experiments.sh plan [tier]       # tier: tier1, tier2, tier1-2, tier3, all (default: all)
#   ./run_filler_experiments.sh start [tier]      # launch experiments
#   ./run_filler_experiments.sh status            # inspect running & finished jobs
#   ./run_filler_experiments.sh stop <ID> [reason]
#   ./run_filler_experiments.sh drop <ID> [reason]
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect REPO_ROOT whether script is in fillers/ or repo root
if [ -f "${SCRIPT_DIR}/thesis_experiments.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
    EXPERIMENT_PY="${SCRIPT_DIR}/fillers/thesis_filler_experiments.py"
    if [ ! -f "${EXPERIMENT_PY}" ]; then
        EXPERIMENT_PY="${SCRIPT_DIR}/thesis_filler_experiments.py"
    fi
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    EXPERIMENT_PY="${SCRIPT_DIR}/thesis_filler_experiments.py"
fi

# Auto-detect conda environment 'gpu_server_env' or fallback python
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [[ "${CONDA_PREFIX}" == *"gpu_server_env"* ]]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
        if [ -n "${CONDA_BASE}" ] && [ -x "${CONDA_BASE}/envs/gpu_server_env/bin/python" ]; then
            PYTHON_BIN="${CONDA_BASE}/envs/gpu_server_env/bin/python"
        fi
    fi
fi
if [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; then
    if command -v python.exe >/dev/null 2>&1; then
        PYTHON_BIN="$(which python.exe)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(which python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(which python)"
    fi
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
export THESIS_FILLER_JOBS_PER_GPU="${THESIS_FILLER_JOBS_PER_GPU:-3}"
export THESIS_FILLER_BATCH64_JOBS_PER_GPU="${THESIS_FILLER_BATCH64_JOBS_PER_GPU:-2}"

ACTION="${1:-plan}"
shift || true

# Check if next argument is a tier specification
TIER_ARG="all"
if [ $# -gt 0 ]; then
    case "$1" in
        tier1|tier2|tier1-2|tier3|all)
            TIER_ARG="$1"
            shift || true
            ;;
    esac
fi

case "${ACTION}" in
    plan)
        echo "=== Displaying Thesis Filler Experiment Plan (Tier: ${TIER_ARG}) ==="
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group plan --tier "${TIER_ARG}" "$@"
        ;;
    run|start)
        echo "=== Launching Thesis Filler Experiment Scheduler (Tier: ${TIER_ARG}) ==="
        "${PYTHON_BIN}" "${PY_EXPERIMENT_PY}" --group run --tier "${TIER_ARG}" "$@"
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
        echo "Usage: $0 {plan|start|run|status|stop|drop} [tier: tier1|tier2|tier1-2|tier3|all] [options...]"
        exit 1
        ;;
esac
