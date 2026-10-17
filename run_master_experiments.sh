#!/usr/bin/env bash
# ==============================================================================
# Root Launcher for Complete Master Thesis Experiments
# Dispatches to master_experiments/run_master_experiments.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/master_experiments/run_master_experiments.sh" "$@"
