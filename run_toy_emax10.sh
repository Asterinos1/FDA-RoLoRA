#!/usr/bin/env bash
# ==============================================================================
# Root Launcher for Pure FDA-RoLoRA Toy Model Experiments (Emax=10, k=0.5, Target=5.0)
# Dispatches to toy/run_toy_emax10.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/toy/run_toy_emax10.sh" "$@"
