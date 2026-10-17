#!/usr/bin/env bash
# ==============================================================================
# Root Launcher for Non-IID Dirichlet (alpha=0.5) Experiments
# Dispatches to non_iid_experiments/run_non_iid_experiments.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/non_iid_experiments/run_non_iid_experiments.sh" "$@"
