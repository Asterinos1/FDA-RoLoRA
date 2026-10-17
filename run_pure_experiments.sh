#!/usr/bin/env bash
# ==============================================================================
# Root Launcher for Pure FDA-RoLoRA Dual-Track Experiments
# Dispatches to pure_experiments/run_pure_experiments.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/pure_experiments/run_pure_experiments.sh" "$@"
