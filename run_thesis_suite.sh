#!/usr/bin/env bash
# ==============================================================================
# Unified Runner for All FDA-LoRA Benchmark Suites (Polytechnix GPU Scheduler)
#
# Available Suites:
#   - pure                 (Pure FDA-RoLoRA Dual-Track Suite)
#   - calibration / calib  (Calibration Suite with Gated Discounting)
#   - non_iid              (Non-IID Dirichlet Robustness Suite)
#   - master               (Master Cross-Silo Benchmark Suite)
#   - fillers              (Filler / Profile Evaluation Suite)
#   - threshold / thresh   (Threshold Controller Experiments kappa=0.5 vs 1.0)
#   - toy                  (MNIST Toy Model Experiments Emax=10/15)
#   - all                  (Global Full Benchmark Run)
#
# Usage Examples:
#   ./run_thesis_suite.sh pure plan
#   ./run_thesis_suite.sh threshold plan
#   ./run_thesis_suite.sh threshold --stage rolora
#   ./run_thesis_suite.sh toy plan
#   ./run_thesis_suite.sh toy start
#   ./run_thesis_suite.sh master start --devices cuda:0,cuda:1
#   ./run_thesis_suite.sh all plan
#   ./run_thesis_suite.sh status
#   ./run_thesis_suite.sh stop <RUN_ID>
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

PYTHON_BIN="python"
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "${CONDA_BASE}" ] && [ -x "${CONDA_BASE}/envs/gpu_server_env/bin/python" ]; then
        PYTHON_BIN="${CONDA_BASE}/envs/gpu_server_env/bin/python"
    fi
fi

# Detect Windows/WSL path mapping if using Windows Python binary
PY_REPO_ROOT="${REPO_ROOT}"
PY_RUNNER="${REPO_ROOT}/run_thesis_suite.py"
if [[ "$("${PYTHON_BIN}" -c "import sys; print(sys.platform)" 2>/dev/null || true)" == "win32" ]] && command -v wslpath >/dev/null 2>&1; then
    PY_REPO_ROOT="$(wslpath -w "${REPO_ROOT}")"
    PY_RUNNER="$(wslpath -w "${REPO_ROOT}/run_thesis_suite.py")"
fi

export PYTHONPATH="${PY_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

exec "${PYTHON_BIN}" "${PY_RUNNER}" "$@"
