#!/usr/bin/env bash
# ==============================================================================
# Root Launcher for Pure FDA-RoLoRA Toy Model Experiments
# Dispatches to toy/run_toy_experiments.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/toy/run_toy_experiments.sh" "$@"
