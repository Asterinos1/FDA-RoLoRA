#!/usr/bin/env bash
# Forwarding wrapper to fillers/run_filler_experiments.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/fillers/run_filler_experiments.sh" "$@"
