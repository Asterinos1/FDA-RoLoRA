#!/usr/bin/env python3
"""
Unified Master Scheduler and Benchmark Suite Runner for FDA-LoRA Experiments.

Consolidates all experimental tracks into a single authoritative orchestrator:
  1. Pure FDA-RoLoRA Dual-Track Suite (`--suite pure`)
  2. Calibration Suite with Gated Discounting (`--suite calibration`)
  3. Non-IID Dirichlet Robustness Suite (`--suite non_iid`)
  4. Master Cross-Silo Benchmark Suite (`--suite master`)
  5. Filler / Profile Evaluation Suite (`--suite fillers`)
  6. Threshold Controller Experiments (`--suite threshold` / `threshold_controller`)
  7. MNIST Toy Model Experiments (`--suite toy`)
  8. Global Full Benchmark Run (`--suite all`)

Usage Examples:
    python run_thesis_suite.py pure plan
    python run_thesis_suite.py threshold plan
    python run_thesis_suite.py threshold --stage rolora --dry-run
    python run_thesis_suite.py toy plan
    python run_thesis_suite.py toy --emax 10 plan
    python run_thesis_suite.py master start --devices cuda:0,cuda:1
    python run_thesis_suite.py all plan
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import fcntl
except ImportError:
    fcntl = None

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import thesis_experiments as base
from fdaopt import utils
from fdaopt.training.fda_opt import fda_rolora_dual_track_pure_opt
from fdaopt.training.fed_opt import fedavg_rolora

# Register all variant methods into base.METHODS and base.METHOD_SLUGS dictionaries
for m, slug in [
    ("FDA-RoLoRA-DualTrack-Pure", "fda-pure"),
    ("FDA-RoLoRA-DualTrack-Calibrated", "fda-calib"),
    ("FDA-RoLoRA-DualTrack", "fda-standard"),
    ("FedAvg-RoLoRA", "rolora"),
]:
    base.METHODS[m] = {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
    }
    base.METHOD_SLUGS[m] = slug

RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "round_metrics"

DEFAULT_SEEDS = [42, 43]
DEFAULT_CLIENT_COUNTS = [20, 50]
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")

# Early stopping defaults
EARLY_STOPPING_MIN_ROUNDS = 6
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.005
EARLY_STOPPING_WINDOW = 3

DATASET_COST_ORDER = ("qqp", "qnli", "mnli", "sst2")
DATASET_SETTINGS = {
    "qqp": {"batch_size": 32, "layers": list(range(7, 12)), "rounds": {20: 50, 50: 30}, "learning_rate": 0.10},
    "qnli": {"batch_size": 32, "layers": list(range(7, 12)), "rounds": {20: 50, 50: 30}, "learning_rate": 0.10},
    "mnli": {"batch_size": 32, "layers": list(range(7, 12)), "rounds": {20: 50, 50: 30}, "learning_rate": 0.10},
    "sst2": {"batch_size": 64, "layers": list(range(9, 12)), "rounds": {20: 50, 50: 30}, "learning_rate": 0.10},
}

EARLY_STOPPING_GATES = {
    "qqp": {20: 0.72, 50: 0.72},
    "qnli": {20: 0.80, 50: 0.80},
    "mnli": {20: 0.60, 50: 0.55},
    "sst2": {20: 0.85, 50: 0.85},
}

# Toy Model Scenarios
TOY_SCENARIOS = [
    {"scenario": "10-Clients-1-Class", "linear": False, "num_clients": 10, "arch": "NONLIN"},
    {"scenario": "10-Clients-1-Class", "linear": True,  "num_clients": 10, "arch": "LIN"},
    {"scenario": "5-Clients-2-Classes", "linear": False, "num_clients": 5,  "arch": "NONLIN"},
    {"scenario": "5-Clients-2-Classes", "linear": True,  "num_clients": 5,  "arch": "LIN"},
]

# Paired Epoch Tiers for Threshold Controller experiments
THRESHOLD_EPOCH_TIERS = [
    {"baseline_epochs": 15, "fda_e_min": 3, "fda_e_max": 30},
    {"baseline_epochs": 20, "fda_e_min": 3, "fda_e_max": 40},
]


def _as_list(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


# =============================================================================
# Configuration Builders
# =============================================================================

def build_pure_configs(
    datasets: List[str],
    client_counts: List[int],
    seeds: List[int],
    ratios: List[float],
    e_min: int = 1,
    e_max: int = 20,
) -> List[Dict[str, Any]]:
    """Build isolated Pure FDA-RoLoRA configurations."""
    configs = []
    for dataset in datasets:
        settings = DATASET_SETTINGS[dataset]
        for num_clients in client_counts:
            for seed in seeds:
                rounds = settings["rounds"][num_clients]
                gate = EARLY_STOPPING_GATES[dataset][num_clients]
                for ratio in ratios:
                    cfg = base.make_config(
                        stage="pure-calib",
                        dataset=dataset,
                        method="FDA-RoLoRA-DualTrack",
                        seed=seed,
                        num_clients=num_clients,
                        total_rounds=rounds,
                        alpha=1000.0,
                        learning_rate=settings["learning_rate"],
                        rank=LORA_RANK,
                        layers=settings["layers"],
                        batch_size=settings["batch_size"],
                        clients_per_round=num_clients,
                        local_epochs=1,
                        lora_alpha=LORA_ALPHA,
                        target_modules=TARGET_MODULES,
                    )
                    cfg["METHOD_NAME"] = "FDA-RoLoRA-DualTrack-Pure"
                    cfg.update({
                        "server_opt_name": "SGD",
                        "server_opt_lr": 1.0,
                        "client_opt_name": "SGD",
                        "num_workers": 0,
                        "pin_memory": False,
                        "train_head": False,
                        "early_stopping": True,
                        "threshold": False,
                        "paper_parameter_budget": "P2",
                        "e_min": e_min,
                        "max_epochs": e_max,
                        "theta": 0.0,
                        "controller_ratio": ratio,
                        "enable_b_discount": False,
                        "b_discount_factor": 1.0,
                        "early_stopping_start_accuracy": gate,
                        "plateau_min_accuracy": gate,
                        "early_stopping_min_rounds": EARLY_STOPPING_MIN_ROUNDS,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                        "early_stopping_window": EARLY_STOPPING_WINDOW,
                        "stagnation_round": 10000,
                        "stagnation_tolerance": 0.0,
                        "start_matrix": "B",
                        "first_round_guard": False,
                    })
                    ratio_tag = f"CRATIO{ratio:g}"
                    extra_id = (
                        f"THESIS-RBASE-PURE-DS-{dataset}-C{num_clients}-R{rounds}"
                        f"-M-fda-pure-LR{settings['learning_rate']:g}-RK{LORA_RANK}"
                        f"-E{e_min}TO{e_max}-{ratio_tag}-NODISC-BSTART-NOGUARD-S{seed}"
                        f"-ES{gate:g}-MR{EARLY_STOPPING_MIN_ROUNDS}-P{EARLY_STOPPING_PATIENCE}"
                        f"-D{EARLY_STOPPING_MIN_DELTA:g}-W{EARLY_STOPPING_WINDOW}"
                    )
                    cfg["extras"] = extra_id
                    configs.append(cfg)
    return configs


def build_calib_configs(
    datasets: List[str],
    client_counts: List[int],
    seeds: List[int],
    ratios: List[float],
    discount_factors: Optional[List[float]] = None,
    discount_modes: Optional[List[bool]] = None,
    e_min: int = 3,
    e_max: int = 20,
) -> List[Dict[str, Any]]:
    """Build calibrated configurations with optional B-matrix discounting."""
    if discount_factors is None:
        discount_factors = [1.0]
    if discount_modes is None:
        discount_modes = [False]

    configs = []
    for dataset in datasets:
        settings = DATASET_SETTINGS[dataset]
        for num_clients in client_counts:
            rounds = settings["rounds"][num_clients]
            gate = EARLY_STOPPING_GATES[dataset][num_clients]
            for seed in seeds:
                for ratio in ratios:
                    ratio_tag = f"CRATIO{ratio:g}"
                    for disc_mode in discount_modes:
                        factor_list = discount_factors if disc_mode else [1.0]
                        for discount_factor in factor_list:
                            cfg = base.make_config(
                                stage="pure-calib",
                                dataset=dataset,
                                method="FDA-RoLoRA-DualTrack",
                                seed=seed,
                                num_clients=num_clients,
                                total_rounds=rounds,
                                alpha=1000.0,
                                learning_rate=settings["learning_rate"],
                                rank=LORA_RANK,
                                layers=settings["layers"],
                                batch_size=settings["batch_size"],
                                clients_per_round=num_clients,
                                local_epochs=1,
                                lora_alpha=LORA_ALPHA,
                                target_modules=TARGET_MODULES,
                            )
                            cfg["METHOD_NAME"] = "FDA-RoLoRA-DualTrack-Calibrated"
                            cfg.update({
                                "server_opt_name": "SGD",
                                "server_opt_lr": 1.0,
                                "client_opt_name": "SGD",
                                "num_workers": 0,
                                "pin_memory": False,
                                "train_head": False,
                                "early_stopping": True,
                                "threshold": False,
                                "paper_parameter_budget": "P2",
                                "e_min": e_min,
                                "max_epochs": e_max,
                                "theta": 0.0,
                                "controller_ratio": ratio,
                                "enable_b_discount": disc_mode,
                                "b_discount_factor": discount_factor,
                                "early_stopping_start_accuracy": gate,
                                "plateau_min_accuracy": gate,
                                "early_stopping_min_rounds": EARLY_STOPPING_MIN_ROUNDS,
                                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                                "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                                "early_stopping_window": EARLY_STOPPING_WINDOW,
                                "stagnation_round": 10000,
                                "stagnation_tolerance": 0.0,
                                "start_matrix": "B",
                                "first_round_guard": False,
                            })
                            disc_tag = f"BDISC{discount_factor:g}" if disc_mode else "NODISC"
                            extra_id = (
                                f"THESIS-RBASE-CALIB-DS-{dataset}-C{num_clients}-R{rounds}"
                                f"-M-fda-calib-LR{settings['learning_rate']:g}-RK{LORA_RANK}"
                                f"-E{e_min}TO{e_max}-{ratio_tag}-{disc_tag}-S{seed}"
                                f"-ES{gate:g}-MR{EARLY_STOPPING_MIN_ROUNDS}"
                                f"-P{EARLY_STOPPING_PATIENCE}"
                                f"-D{EARLY_STOPPING_MIN_DELTA:g}"
                                f"-W{EARLY_STOPPING_WINDOW}"
                            )
                            cfg["extras"] = extra_id
                            configs.append(cfg)
    return configs


def build_non_iid_configs(
    datasets: List[str],
    client_counts: List[int],
    seeds: List[int],
    alpha: float = 0.5,
    rounds: Optional[int] = None,
    methods: Optional[List[str]] = None,
    local_epochs: Optional[List[int]] = None,
    fda_configs: Optional[List[tuple]] = None,
    e_min: int = 1,
) -> List[Dict[str, Any]]:
    """Build Dirichlet Non-IID robustness comparison configurations."""
    if methods is None:
        methods = ["FedAvg-RoLoRA", "FDA-RoLoRA-DualTrack-Pure"]
    if local_epochs is None:
        local_epochs = [10]
    if fda_configs is None:
        fda_configs = [(20, 1.0), (20, 0.5)]

    configs = []
    for dataset in datasets:
        settings = DATASET_SETTINGS[dataset]
        gate = EARLY_STOPPING_GATES.get(dataset, {20: 0.80, 50: 0.80})[20]
        total_rounds = rounds or settings["rounds"][20]
        min_rounds = int(0.6 * total_rounds)

        for num_clients in client_counts:
            for seed in seeds:
                if "FedAvg-RoLoRA" in methods:
                    for epochs in local_epochs:
                        cfg = base.make_config(
                            stage="non-iid",
                            dataset=dataset,
                            method="FedAvg-RoLoRA",
                            seed=seed,
                            num_clients=num_clients,
                            total_rounds=total_rounds,
                            alpha=alpha,
                            rank=LORA_RANK,
                            learning_rate=settings["learning_rate"],
                            layers=settings["layers"],
                            batch_size=settings["batch_size"],
                            clients_per_round=num_clients,
                            local_epochs=epochs,
                            lora_alpha=LORA_ALPHA,
                            target_modules=TARGET_MODULES,
                        )
                        cfg.update({
                            "server_opt_name": "SGD",
                            "server_opt_lr": 1.0,
                            "client_opt_name": "SGD",
                            "num_workers": 0,
                            "pin_memory": False,
                            "train_head": False,
                            "early_stopping": True,
                            "threshold": False,
                            "paper_parameter_budget": "P2",
                            "start_matrix": "B",
                            "early_stopping_start_accuracy": gate,
                            "plateau_min_accuracy": gate,
                            "early_stopping_min_rounds": min_rounds,
                            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                            "early_stopping_window": 5,
                        })
                        extra_id = (
                            f"THESIS-NONIID-A{alpha:g}-DS-{dataset}-C{num_clients}-R{total_rounds}"
                            f"-M-rolora-E{epochs}-LR{settings['learning_rate']:g}-RK{LORA_RANK}"
                            f"-BSTART-S{seed}-ES{gate:g}-MR{min_rounds}"
                            f"-P{EARLY_STOPPING_PATIENCE}-D{EARLY_STOPPING_MIN_DELTA:g}-W5"
                        )
                        cfg["extras"] = extra_id
                        configs.append(cfg)

                if "FDA-RoLoRA-DualTrack-Pure" in methods:
                    for fda_emax, ratio in fda_configs:
                        cfg = base.make_config(
                            stage="non-iid",
                            dataset=dataset,
                            method="FDA-RoLoRA-DualTrack-Pure",
                            seed=seed,
                            num_clients=num_clients,
                            total_rounds=total_rounds,
                            alpha=alpha,
                            rank=LORA_RANK,
                            learning_rate=settings["learning_rate"],
                            layers=settings["layers"],
                            batch_size=settings["batch_size"],
                            clients_per_round=num_clients,
                            local_epochs=1,
                            lora_alpha=LORA_ALPHA,
                            target_modules=TARGET_MODULES,
                        )
                        cfg.update({
                            "server_opt_name": "SGD",
                            "server_opt_lr": 1.0,
                            "client_opt_name": "SGD",
                            "num_workers": 0,
                            "pin_memory": False,
                            "train_head": False,
                            "early_stopping": True,
                            "threshold": False,
                            "paper_parameter_budget": "P2",
                            "e_min": e_min,
                            "max_epochs": fda_emax,
                            "theta": 0.0,
                            "controller_ratio": ratio,
                            "enable_b_discount": False,
                            "b_discount_factor": 1.0,
                            "start_matrix": "B",
                            "first_round_guard": False,
                            "early_stopping_start_accuracy": gate,
                            "plateau_min_accuracy": gate,
                            "early_stopping_min_rounds": min_rounds,
                            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                            "early_stopping_window": 5,
                        })
                        ratio_tag = f"CRATIO{ratio:g}"
                        extra_id = (
                            f"THESIS-NONIID-A{alpha:g}-DS-{dataset}-C{num_clients}-R{total_rounds}"
                            f"-M-fda-pure-LR{settings['learning_rate']:g}-RK{LORA_RANK}"
                            f"-E{e_min}TO{fda_emax}-{ratio_tag}-NODISC-BSTART-NOGUARD-S{seed}"
                            f"-ES{gate:g}-MR{min_rounds}-P{EARLY_STOPPING_PATIENCE}"
                            f"-D{EARLY_STOPPING_MIN_DELTA:g}-W5"
                        )
                        cfg["extras"] = extra_id
                        configs.append(cfg)
    return configs


def build_master_configs() -> List[Dict[str, Any]]:
    """Build all 48 master cross-silo comparison benchmarks."""
    configs = []
    # Part A: Pure IID Suite (32 configs)
    for ds in DATASET_COST_ORDER:
        settings = DATASET_SETTINGS[ds]
        for c in DEFAULT_CLIENT_COUNTS:
            rounds = settings["rounds"][c]
            gate = EARLY_STOPPING_GATES[ds][c]
            for s in DEFAULT_SEEDS:
                for r in [1.0, 0.5]:
                    cfg = base.make_config(
                        stage="pure-calib",
                        dataset=ds,
                        method="FDA-RoLoRA-DualTrack",
                        seed=s,
                        num_clients=c,
                        total_rounds=rounds,
                        alpha=1000.0,
                        learning_rate=settings["learning_rate"],
                        rank=LORA_RANK,
                        layers=settings["layers"],
                        batch_size=settings["batch_size"],
                        clients_per_round=c,
                        local_epochs=1,
                        lora_alpha=LORA_ALPHA,
                        target_modules=TARGET_MODULES,
                    )
                    cfg["METHOD_NAME"] = "FDA-RoLoRA-DualTrack-Pure"
                    cfg.update({
                        "server_opt_name": "SGD",
                        "server_opt_lr": 1.0,
                        "client_opt_name": "SGD",
                        "num_workers": 0,
                        "pin_memory": False,
                        "train_head": False,
                        "early_stopping": True,
                        "threshold": False,
                        "paper_parameter_budget": "P2",
                        "e_min": 1,
                        "max_epochs": 20,
                        "theta": 0.0,
                        "controller_ratio": r,
                        "enable_b_discount": False,
                        "b_discount_factor": 1.0,
                        "early_stopping_start_accuracy": gate,
                        "plateau_min_accuracy": gate,
                        "early_stopping_min_rounds": 6,
                        "early_stopping_patience": 3,
                        "early_stopping_min_delta": 0.005,
                        "early_stopping_window": 3,
                        "stagnation_round": 10_000,
                        "stagnation_tolerance": 0.0,
                        "start_matrix": "B",
                        "first_round_guard": False,
                    })
                    ratio_tag = f"CRATIO{r:g}"
                    extra_id = (
                        f"THESIS-RBASE-PURE-DS-{ds}-C{c}-R{rounds}"
                        f"-M-fda-pure-LR{settings['learning_rate']:g}-RK{LORA_RANK}"
                        f"-E1TO20-{ratio_tag}-NODISC-BSTART-NOGUARD-S{s}"
                        f"-ES{gate:g}-MR6-P3-D0.005-W3"
                    )
                    cfg["extras"] = extra_id
                    configs.append(cfg)

    # Part B: Lean Non-IID Suite (16 configs)
    for ds in ["qnli", "mnli"]:
        settings = DATASET_SETTINGS[ds]
        for c in DEFAULT_CLIENT_COUNTS:
            rounds = settings["rounds"][c]
            gate = 0.80 if ds == "qnli" else (0.60 if c == 20 else 0.55)
            mr = int(rounds * 0.6)
            for s in DEFAULT_SEEDS:
                cfg_ro = base.make_config(
                    stage="non-iid",
                    dataset=ds,
                    method="FedAvg-RoLoRA",
                    seed=s,
                    num_clients=c,
                    total_rounds=rounds,
                    alpha=0.5,
                    learning_rate=0.10,
                    rank=LORA_RANK,
                    layers=settings["layers"],
                    batch_size=32,
                    clients_per_round=c,
                    local_epochs=10,
                    lora_alpha=LORA_ALPHA,
                    target_modules=TARGET_MODULES,
                )
                cfg_ro["METHOD_NAME"] = "FedAvg-RoLoRA"
                cfg_ro.update({
                    "server_opt_name": "SGD",
                    "server_opt_lr": 1.0,
                    "client_opt_name": "SGD",
                    "num_workers": 0,
                    "pin_memory": False,
                    "train_head": False,
                    "early_stopping": True,
                    "threshold": False,
                    "paper_parameter_budget": "P2",
                    "start_matrix": "B",
                    "first_round_guard": False,
                    "early_stopping_start_accuracy": gate,
                    "plateau_min_accuracy": gate,
                    "early_stopping_min_rounds": mr,
                    "early_stopping_patience": 5,
                    "early_stopping_min_delta": 0.005,
                    "early_stopping_window": 5,
                })
                extra_ro = (
                    f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                    f"-M-rolora-E10-LR0.1-RK{LORA_RANK}-BSTART-S{s}"
                    f"-ES{gate:g}-MR{mr}-P5-D0.005-W5"
                )
                cfg_ro["extras"] = extra_ro
                configs.append(cfg_ro)

                ratios_noniid = [0.5, 1.0] if c == 20 else [0.5]
                for r in ratios_noniid:
                    cfg_fda = base.make_config(
                        stage="non-iid",
                        dataset=ds,
                        method="FDA-RoLoRA-DualTrack",
                        seed=s,
                        num_clients=c,
                        total_rounds=rounds,
                        alpha=0.5,
                        learning_rate=0.10,
                        rank=LORA_RANK,
                        layers=settings["layers"],
                        batch_size=32,
                        clients_per_round=c,
                        local_epochs=1,
                        lora_alpha=LORA_ALPHA,
                        target_modules=TARGET_MODULES,
                    )
                    cfg_fda["METHOD_NAME"] = "FDA-RoLoRA-DualTrack-Pure"
                    cfg_fda.update({
                        "server_opt_name": "SGD",
                        "server_opt_lr": 1.0,
                        "client_opt_name": "SGD",
                        "num_workers": 0,
                        "pin_memory": False,
                        "train_head": False,
                        "early_stopping": True,
                        "threshold": False,
                        "paper_parameter_budget": "P2",
                        "e_min": 1,
                        "max_epochs": 20,
                        "theta": 0.0,
                        "controller_ratio": r,
                        "enable_b_discount": False,
                        "b_discount_factor": 1.0,
                        "start_matrix": "B",
                        "first_round_guard": False,
                        "early_stopping_start_accuracy": gate,
                        "plateau_min_accuracy": gate,
                        "early_stopping_min_rounds": mr,
                        "early_stopping_patience": 5,
                        "early_stopping_min_delta": 0.005,
                        "early_stopping_window": 5,
                    })
                    r_tag = f"CRATIO{r:g}"
                    extra_fda = (
                        f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                        f"-M-fda-pure-LR0.1-RK{LORA_RANK}-E1TO20-{r_tag}-NODISC-BSTART-NOGUARD-S{s}"
                        f"-ES{gate:g}-MR{mr}-P5-D0.005-W5"
                    )
                    cfg_fda["extras"] = extra_fda
                    configs.append(cfg_fda)
    return configs


def build_filler_configs(tier: str = "all") -> List[Dict[str, Any]]:
    """Build filler benchmark configurations filtered by tier."""
    configs = []
    # Tier 1: C=50 Non-IID Missing k=1.0 runs
    if tier in ("tier1", "tier1-2", "all"):
        for ds in ["qnli", "mnli"]:
            settings = DATASET_SETTINGS[ds]
            c = 50
            rounds = 30
            gate = 0.80 if ds == "qnli" else 0.55
            mr = 18
            for s in DEFAULT_SEEDS:
                cfg = base.make_config(
                    stage="non-iid",
                    dataset=ds,
                    method="FDA-RoLoRA-DualTrack",
                    seed=s,
                    num_clients=c,
                    total_rounds=rounds,
                    alpha=0.5,
                    learning_rate=settings["learning_rate"],
                    rank=LORA_RANK,
                    layers=settings["layers"],
                    batch_size=settings["batch_size"],
                    clients_per_round=c,
                    local_epochs=1,
                    lora_alpha=LORA_ALPHA,
                    target_modules=TARGET_MODULES,
                )
                cfg["METHOD_NAME"] = "FDA-RoLoRA-DualTrack-Pure"
                cfg.update({
                    "server_opt_name": "SGD",
                    "server_opt_lr": 1.0,
                    "client_opt_name": "SGD",
                    "num_workers": 0,
                    "pin_memory": False,
                    "train_head": False,
                    "early_stopping": True,
                    "threshold": False,
                    "paper_parameter_budget": "P2",
                    "e_min": 1,
                    "max_epochs": 20,
                    "theta": 0.0,
                    "controller_ratio": 1.0,
                    "enable_b_discount": False,
                    "b_discount_factor": 1.0,
                    "start_matrix": "B",
                    "first_round_guard": False,
                    "early_stopping_start_accuracy": gate,
                    "plateau_min_accuracy": gate,
                    "early_stopping_min_rounds": mr,
                    "early_stopping_patience": 5,
                    "early_stopping_min_delta": 0.005,
                    "early_stopping_window": 5,
                })
                extra_id = (
                    f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                    f"-M-fda-pure-LR{settings['learning_rate']:g}-RK{LORA_RANK}"
                    f"-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S{s}"
                    f"-ES{gate:g}-MR{mr}-P5-D0.005-W5"
                )
                cfg["extras"] = extra_id
                configs.append(cfg)
    return configs


def build_threshold_configs(
    datasets: Optional[List[str]] = None,
    client_counts: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    stage_filter: str = "all",
    learning_rate: float = 0.10,
    paired_tiers: Optional[List[Dict[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """Build RoLoRA vs. FDA-RoLoRA Threshold Controller experiment configurations (kappa=0.5 vs 1.0).
    
    Stages:
      Stage 1: RoLoRA baselines (fixed epochs E=15, 20)
      Stage 2: FDA-RoLoRA with kappa = 0.5 (E_max / 2) -> E in [3, 30] and [3, 40]
      Stage 3: FDA-RoLoRA with kappa = 1.0 (E_max / 1) -> E in [3, 30] and [3, 40] (at very end)
    """
    if datasets is None:
        datasets = ["mnli"]
    if client_counts is None:
        client_counts = [20]
    if seeds is None:
        seeds = [42, 43]
    if paired_tiers is None:
        paired_tiers = THRESHOLD_EPOCH_TIERS

    p2_layers = list(range(7, 12))

    stage1_rolora = []
    stage2_fda_k05 = []
    stage3_fda_k1 = []

    for dataset in datasets:
        settings = DATASET_SETTINGS.get(dataset, {"rounds": {20: 50, 50: 30}, "batch_size": 32})
        for clients in client_counts:
            rounds = settings["rounds"].get(clients, 50 if clients == 20 else 30)
            gate = EARLY_STOPPING_GATES.get(dataset, {}).get(clients, 0.60 if dataset == "mnli" else 0.80)
            mr = EARLY_STOPPING_MIN_ROUNDS
            p = EARLY_STOPPING_PATIENCE
            d = EARLY_STOPPING_MIN_DELTA
            w = EARLY_STOPPING_WINDOW
            es_tag = f"-ES{gate:g}-MR{mr}-P{p}-D{d:g}-W{w}"

            for tier in paired_tiers:
                base_e = tier["baseline_epochs"]
                fda_min = tier["fda_e_min"]
                fda_max = tier["fda_e_max"]

                for seed in seeds:
                    # Stage 1: RoLoRA Baseline
                    rolora_extras = (
                        f"THESIS-{base.MODEL_SLUG}-MAIN-DS-{dataset}"
                        f"-C{clients}-R{rounds}-M-rolora"
                        f"-LR{learning_rate:g}-RK4-S{seed}-E{base_e}{es_tag}"
                    )
                    rolora_cfg = base.make_config(
                        stage="main",
                        dataset=dataset,
                        method="FedAvg-RoLoRA",
                        seed=seed,
                        num_clients=clients,
                        total_rounds=rounds,
                        alpha=1000.0,
                        learning_rate=learning_rate,
                        rank=LORA_RANK,
                        layers=copy.deepcopy(p2_layers),
                        batch_size=32,
                        clients_per_round=clients,
                        local_epochs=base_e,
                        lora_alpha=LORA_ALPHA,
                        target_modules=TARGET_MODULES,
                    )
                    rolora_cfg.update({
                        "server_opt_name": "SGD",
                        "server_opt_lr": 1.0,
                        "client_opt_name": "SGD",
                        "client_opt_lr": learning_rate,
                        "early_stopping": True,
                        "early_stopping_start_accuracy": gate,
                        "plateau_min_accuracy": gate,
                        "early_stopping_min_rounds": mr,
                        "early_stopping_patience": p,
                        "early_stopping_min_delta": d,
                        "early_stopping_window": w,
                        "stagnation_round": 10000,
                        "stagnation_tolerance": 0.0,
                        "paper_parameter_budget": "P2",
                        "extras": rolora_extras,
                        "_stage_label": "Stage 1: RoLoRA Baseline",
                    })
                    stage1_rolora.append(rolora_cfg)

                    # Stage 2: FDA-RoLoRA kappa=0.5
                    fda_k05_extras = (
                        f"THESIS-{base.MODEL_SLUG}-MAIN-DS-{dataset}"
                        f"-C{clients}-R{rounds}-M-fda-k05"
                        f"-LR{learning_rate:g}-RK4-E{fda_min}TO{fda_max}"
                        f"-S{seed}{es_tag}"
                    )
                    fda_k05_cfg = base.make_config(
                        stage="main",
                        dataset=dataset,
                        method="FDA-RoLoRA-DualTrack",
                        seed=seed,
                        num_clients=clients,
                        total_rounds=rounds,
                        alpha=1000.0,
                        learning_rate=learning_rate,
                        rank=LORA_RANK,
                        layers=copy.deepcopy(p2_layers),
                        batch_size=32,
                        clients_per_round=clients,
                        local_epochs=1,
                        lora_alpha=LORA_ALPHA,
                        target_modules=TARGET_MODULES,
                    )
                    fda_k05_cfg.update({
                        "server_opt_name": "SGD",
                        "server_opt_lr": 1.0,
                        "client_opt_name": "SGD",
                        "client_opt_lr": learning_rate,
                        "e_min": fda_min,
                        "max_epochs": fda_max,
                        "gamma": 0.3,
                        "theta": 0.0,
                        "controller_type": "standard",
                        "controller_ratio": 0.5,
                        "enable_b_discount": False,
                        "b_discount_factor": 3.0,
                        "early_stopping": True,
                        "early_stopping_start_accuracy": gate,
                        "plateau_min_accuracy": gate,
                        "early_stopping_min_rounds": mr,
                        "early_stopping_patience": p,
                        "early_stopping_min_delta": d,
                        "early_stopping_window": w,
                        "stagnation_round": 10000,
                        "stagnation_tolerance": 0.0,
                        "paper_parameter_budget": "P2",
                        "extras": fda_k05_extras,
                        "_stage_label": "Stage 2: FDA-RoLoRA (kappa=0.5, Emax/2)",
                    })
                    stage2_fda_k05.append(fda_k05_cfg)

                    # Stage 3: FDA-RoLoRA kappa=1.0
                    fda_k1_extras = (
                        f"THESIS-{base.MODEL_SLUG}-MAIN-DS-{dataset}"
                        f"-C{clients}-R{rounds}-M-fda-k1"
                        f"-LR{learning_rate:g}-RK4-E{fda_min}TO{fda_max}"
                        f"-S{seed}{es_tag}"
                    )
                    fda_k1_cfg = base.make_config(
                        stage="main",
                        dataset=dataset,
                        method="FDA-RoLoRA-DualTrack",
                        seed=seed,
                        num_clients=clients,
                        total_rounds=rounds,
                        alpha=1000.0,
                        learning_rate=learning_rate,
                        rank=LORA_RANK,
                        layers=copy.deepcopy(p2_layers),
                        batch_size=32,
                        clients_per_round=clients,
                        local_epochs=1,
                        lora_alpha=LORA_ALPHA,
                        target_modules=TARGET_MODULES,
                    )
                    fda_k1_cfg.update({
                        "server_opt_name": "SGD",
                        "server_opt_lr": 1.0,
                        "client_opt_name": "SGD",
                        "client_opt_lr": learning_rate,
                        "e_min": fda_min,
                        "max_epochs": fda_max,
                        "gamma": 0.3,
                        "theta": 0.0,
                        "controller_type": "standard",
                        "controller_ratio": 1.0,
                        "enable_b_discount": False,
                        "b_discount_factor": 3.0,
                        "early_stopping": True,
                        "early_stopping_start_accuracy": gate,
                        "plateau_min_accuracy": gate,
                        "early_stopping_min_rounds": mr,
                        "early_stopping_patience": p,
                        "early_stopping_min_delta": d,
                        "early_stopping_window": w,
                        "stagnation_round": 10000,
                        "stagnation_tolerance": 0.0,
                        "paper_parameter_budget": "P2",
                        "extras": fda_k1_extras,
                        "_stage_label": "Stage 3: FDA-RoLoRA (kappa=1.0, Emax/1)",
                    })
                    stage3_fda_k1.append(fda_k1_cfg)

    if stage_filter == "rolora":
        return stage1_rolora
    elif stage_filter == "fda_k05":
        return stage2_fda_k05
    elif stage_filter == "fda_k1":
        return stage3_fda_k1
    elif stage_filter == "fda_all":
        return stage2_fda_k05 + stage3_fda_k1
    else:
        return stage1_rolora + stage2_fda_k05 + stage3_fda_k1


def build_toy_configs(
    ratios: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    e_max: int = 15,
    scenarios: Optional[List[Dict[str, Any]]] = None,
    total_rounds: int = 100,
) -> List[Dict[str, Any]]:
    """Build Pure FDA-RoLoRA Toy Model experiment configurations on MNIST."""
    if ratios is None:
        ratios = [1.0, 0.5]
    if seeds is None:
        seeds = [42, 43, 44]
    if scenarios is None:
        scenarios = TOY_SCENARIOS

    configs = []
    for ratio in ratios:
        for scen in scenarios:
            for seed in seeds:
                emax_tag = f"_EMAX{e_max}" if e_max != 15 else ""
                run_id = f"TOY-PURE-FDA_PURE_FDA-RoLoRA-DualTrack-Pure_{scen['scenario']}_{scen['arch']}{emax_tag}_CRATIO{ratio}_S{seed}"
                cfg = {
                    "scenario": scen["scenario"],
                    "arch": scen["arch"],
                    "linear": scen["linear"],
                    "num_clients": scen["num_clients"],
                    "controller_ratio": ratio,
                    "seed": seed,
                    "run_id": run_id,
                    "extras": run_id,
                    "total_rounds": total_rounds,
                    "e_min": 1,
                    "e_max": e_max,
                    "max_epochs": e_max,
                    "batch_size": 64,
                    "lora_r": 16,
                    "rank": 16,
                    "checkpoint": "toy-model",
                    "ds_path": "ylecun/mnist",
                    "ds_name": "mnist",
                    "method": "FDA-RoLoRA-DualTrack-Pure",
                    "METHOD_NAME": "FDA-RoLoRA-DualTrack-Pure",
                    "_is_toy": True,
                }
                configs.append(cfg)
    return configs


# =============================================================================
# Execution Engine & Worker Isolation
# =============================================================================

def is_completed(config: dict) -> bool:
    """Check if the experiment run has already completed metrics in results/round_metrics."""
    if config.get("_is_toy") or config.get("extras", "").startswith("TOY-"):
        scenario = config["scenario"]
        arch = config["arch"]
        ratio = config["controller_ratio"]
        seed = config["seed"]
        e_max = config.get("e_max", 15)
        emax_tag = f"_EMAX{e_max}" if e_max != 15 else ""
        if e_max == 15:
            pattern = str(METRICS_DIR / f"TOY-PURE-FDA_PURE*_{scenario}_{arch}_CRATIO{ratio}_S{seed}_*.csv")
        else:
            pattern = str(METRICS_DIR / f"TOY-PURE-FDA_PURE*_{scenario}_{arch}{emax_tag}_CRATIO{ratio}_S{seed}_*.csv")
        for f in glob.glob(pattern):
            if e_max == 15 and "_EMAX" in Path(f).name and "_EMAX15" not in Path(f).name:
                continue
            try:
                import pandas as pd
                df = pd.read_csv(f)
                if len(df) >= config.get("total_rounds", 100):
                    return True
            except Exception:
                pass
        return False

    pattern = str(METRICS_DIR / f"{config['extras']}_*.csv")
    for path in glob.glob(pattern):
        try:
            import pandas as pd
            frame = pd.read_csv(path)
            if not frame.empty and "round" in frame.columns:
                return True
        except Exception:
            continue
    return False


def toggle_rolora_matrices(model, active_matrix: str):
    """Toggle LoRA active factor requires_grad flag."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = f"lora_{active_matrix}" in name


def execute_toy_worker(config: dict, device_str: str) -> None:
    """Execute Toy Model alternating optimization loop on MNIST."""
    import torch
    from fdaopt import utils
    from fdaopt.models.toy_model import ToyModel
    from fdaopt.datasets.fed_data_prep import prepare_federated_datasets
    from fdaopt.models.ops import (
        compute_client_drifts,
        variance,
        compute_pseudo_gradients,
        copy_parameters,
        set_gradients,
        get_save_device,
    )
    from fdaopt.training.optimizers import server_client_optimizers
    from fdaopt.metrics.mentrics_handler import MetricsHandler

    scenario = config["scenario"]
    arch = config["arch"]
    linear = config["linear"]
    num_clients = config["num_clients"]
    controller_ratio = config["controller_ratio"]
    seed = config["seed"]
    total_rounds = config.get("total_rounds", 100)
    e_min = config.get("e_min", 1)
    e_max = config.get("e_max", 15)
    batch_size = config.get("batch_size", 64)
    rank = config.get("lora_r", 16)

    device = torch.device(device_str)
    utils.set_device(device)
    utils.set_seed(seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    emax_tag = f"_EMAX{e_max}" if e_max != 15 else ""
    extras = f"TOY-PURE-FDA_PURE-{timestamp}_FDA-RoLoRA-DualTrack-Pure_{scenario}_{arch}{emax_tag}_CRATIO{controller_ratio}_S{seed}"

    hyperparams = {
        "checkpoint": "toy-model",
        "ds_path": "ylecun/mnist",
        "ds_name": "mnist",
        "num_labels": 10,
        "num_clients": num_clients,
        "clients_per_round": num_clients,
        "total_rounds": total_rounds,
        "local_epochs": 1,
        "e_min": e_min,
        "max_epochs": e_max,
        "controller_ratio": controller_ratio,
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": True,
        "alpha": 0.0,
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
        "client_opt_lr": 0.01,
        "lora_r": rank,
        "linear": linear,
        "extras": extras,
        "method": "FDA-RoLoRA-DualTrack-Pure",
        "METHOD_NAME": "FDA-RoLoRA-DualTrack-Pure",
        "early_stopping": False,
        "seed": seed,
        "enable_b_discount": False,
        "b_discount_factor": 1.0,
        "start_matrix": "B",
        "first_round_guard": False,
    }

    # 1. Dataset loading & Partition
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams["ds_path"], hyperparams["ds_name"], hyperparams["checkpoint"],
        hyperparams["num_clients"], hyperparams["alpha"], hyperparams["batch_size"],
        hyperparams["num_workers"], hyperparams["pin_memory"]
    )

    # 2. In-VRAM pre-caching with pre-flattened float tensors
    client_gpu_data = {}
    steps_per_epoch = {}
    for c_id in range(num_clients):
        dl = fed_ds.client_dataloaders[c_id]
        all_imgs, all_lbls = [], []
        for b in dl:
            all_imgs.append(b["image"])
            all_lbls.append(b["labels"])
        imgs_cat = torch.cat(all_imgs, dim=0).to(device)
        lbls_cat = torch.cat(all_lbls, dim=0).to(device)
        client_gpu_data[c_id] = (
            imgs_cat.view(imgs_cat.size(0), -1).float() / 255.0,
            lbls_cat.long(),
        )
        steps_per_epoch[c_id] = len(dl)

    test_dl = test_ds["mnist"]
    test_imgs, test_lbls = [], []
    for b in test_dl:
        test_imgs.append(b["image"])
        test_lbls.append(b["labels"])
    t_imgs = torch.cat(test_imgs, dim=0).to(device)
    t_lbls = torch.cat(test_lbls, dim=0).to(device)
    gpu_test_x = t_imgs.view(t_imgs.size(0), -1).float() / 255.0
    gpu_test_y = t_lbls.long()

    # 3. Model & Optimizers
    model = ToyModel(rank=rank, linear=linear).to(device)
    params_A = [p for n, p in model.named_parameters() if "lora_A" in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad]
    num_lora = len(params_A)
    params_A = params_A + params_other
    params_B = params_B + params_other

    server_opt_A, _ = server_client_optimizers(params_A, hyperparams, num_lora_params=num_lora)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams, num_lora_params=num_lora)

    client_dict_A = {c_id: [p.detach().clone().to(get_save_device()) for p in params_A] for c_id in range(num_clients)}
    client_dict_B = {c_id: [p.detach().clone().to(get_save_device()) for p in params_B] for c_id in range(num_clients)}

    metrics_handler = MetricsHandler(hyperparams)
    theta_A = 0.0
    theta_B = 0.0

    print(f"[{time.strftime('%H:%M:%S')}] >>> TOY WORKER STARTED: {extras} on {device_str} <<<", flush=True)
    t_start = time.time()
    rng_gen = torch.Generator(device=device)
    rng_gen.manual_seed(seed)

    for r in range(total_rounds):
        r_start = time.time()
        active_matrix = "B" if r % 2 == 0 else "A"
        toggle_rolora_matrices(model, active_matrix)

        active_params = params_A if active_matrix == "A" else params_B
        active_server_opt = server_opt_A if active_matrix == "A" else server_opt_B
        active_client_dict = client_dict_A if active_matrix == "A" else client_dict_B
        current_theta = theta_A if active_matrix == "A" else theta_B

        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = list(range(num_clients))
        for c_id in sampled_clients:
            active_client_dict[c_id] = [p.detach().clone().to(get_save_device()) for p in round_start_active_params]

        training_loss = 0.0
        training_accuracy = 0.0
        var_subspace = 0.0
        avg_norm_sq_drifts = 0.0
        norm_sq_avg_drift = 0.0
        round_epochs = 0

        train_start = time.time()
        while True:
            round_epochs += 1
            for client_id in sampled_clients:
                _, client_opt = server_client_optimizers(active_params, hyperparams, num_lora_params=num_lora)
                model.train()
                copy_parameters(from_params=active_client_dict[client_id], to_params=active_params)

                cx, cy = client_gpu_data[client_id]
                N = cx.size(0)
                perm = torch.randperm(N, generator=rng_gen, device=device)
                px, py = cx[perm], cy[perm]

                running_loss = torch.tensor(0.0, device=device)
                running_correct = torch.tensor(0, dtype=torch.long, device=device)
                n_batches = 0

                for i in range(0, N, batch_size):
                    bx = px[i : i + batch_size]
                    by = py[i : i + batch_size]
                    client_opt.zero_grad(set_to_none=True)

                    hidden = bx @ model.lora_A @ model.lora_B
                    if not model.linear:
                        hidden = torch.relu(hidden)
                    logits = hidden @ model.W_out
                    loss = torch.nn.functional.cross_entropy(logits, by)

                    preds = torch.argmax(logits, dim=-1)
                    running_correct += (preds == by).sum()
                    loss.backward()
                    client_opt.step()
                    running_loss += loss.detach()
                    n_batches += 1

                copy_parameters(from_params=active_params, to_params=active_client_dict[client_id])
                training_loss += (running_loss.item() / n_batches)
                training_accuracy += (running_correct.item() / N)

            # Evaluate client drift variance on active factor
            sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
            client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
            var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)

            # Stopping condition: breaks if variance exceeds threshold or hits e_max
            if var_subspace > current_theta or round_epochs >= e_max:
                break

        train_end = time.time()
        agg_start = time.time()

        training_loss = training_loss / round_epochs
        training_accuracy = training_accuracy / round_epochs
        pseudo_gradients = compute_pseudo_gradients(client_drifts)

        with torch.no_grad():
            update_l2_norm = float(torch.sqrt(sum(torch.sum(g**2) for g in pseudo_gradients)).item())

        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        set_gradients(active_params, pseudo_gradients)
        active_server_opt.step()
        active_server_opt.zero_grad()

        agg_end = time.time()

        # Update specific phase theta with proportional setpoint
        target_epochs = controller_ratio * e_max
        next_theta = (target_epochs / round_epochs) * var_subspace
        if active_matrix == "A":
            theta_A = next_theta
        else:
            theta_B = next_theta

        # Global test set evaluation
        model.eval()
        with torch.no_grad():
            hidden = gpu_test_x @ model.lora_A @ model.lora_B
            if not model.linear:
                hidden = torch.relu(hidden)
            t_logits = hidden @ model.W_out
            test_loss = torch.nn.functional.cross_entropy(t_logits, gpu_test_y).item()
            test_preds = torch.argmax(t_logits, dim=-1)
            test_acc = (test_preds == gpu_test_y).sum().item() / gpu_test_y.size(0)

        round_end = time.time()

        metrics = {
            "method": "FDA-RoLoRA-DualTrack-Pure",
            "round": r + 1,
            "active_matrix": active_matrix,
            "start_matrix": "B",
            "first_round_guard": False,
            "e_min": e_min,
            "e_max": e_max,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end - r_start,
            "update_l2_norm": update_l2_norm,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * steps_per_epoch[0],
            "variance": var_subspace,
            "effective_variance": var_subspace,
            "avg_norm_sq_drifts": avg_norm_sq_drifts,
            "norm_sq_avg_drift": norm_sq_avg_drift,
            "theta_phase": current_theta,
            "controller_ratio": controller_ratio,
            "enable_b_discount": False,
            "b_discount_factor": 1.0,
            "accuracy": test_acc,
            "testing_loss": test_loss,
            "training_loss": training_loss / num_clients,
            "training_accuracy": training_accuracy / num_clients,
        }
        metrics_handler.append_round_metrics(metrics)

        print(
            f"[METRIC] round={r+1:3d}/{total_rounds} matrix={active_matrix} epochs={round_epochs:2d} "
            f"acc={test_acc*100:5.2f}% loss={test_loss:7.4f} var={var_subspace:7.3f} "
            f"theta={next_theta:7.3f} time={round_end-r_start:5.2f}s",
            flush=True,
        )

    metrics_handler.save_metrics()
    elapsed = time.time() - t_start
    print(f"[{time.strftime('%H:%M:%S')}] [DONE] finished in {elapsed/60:.2f} min ({elapsed:.1f}s) -> {metrics_handler.filename}", flush=True)


def execute_run(config_json: str) -> None:
    """Child-worker target invoked by scheduler via --run-config."""
    import torch
    config = json.loads(config_json)
    device = config.pop("TARGET_DEVICE", "cuda:0")

    # If it is a toy experiment configuration:
    if config.get("_is_toy") or config.get("checkpoint") == "toy-model":
        execute_toy_worker(config, device)
        return

    # Standard RoBERTa-base experiments:
    base.install_runtime_patches(config)

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    utils.DEVICE_RAM_PROGRAM = "performance"
    utils.set_device(device)
    utils.set_seed(config["seed"])

    method = config.get("METHOD_NAME", config.get("method", ""))
    print(f"[{time.strftime('%H:%M:%S')}] Starting {config['extras']} ({method}) on {device}...")

    if "rolora" in method.lower() and "fda" not in method.lower():
        fedavg_rolora(config)
    else:
        fda_rolora_dual_track_pure_opt(config)

    print(f"[{time.strftime('%H:%M:%S')}] Execution finished for {config['extras']}.")


def run_config_on_device(config: dict, device: str) -> bool:
    """Execute one configuration as an isolated child process via run_thesis_suite.py."""
    run_id = config["extras"]
    temporary_dir = None
    registry_path = None
    request_path = None
    try:
        if is_completed(config):
            print(f"SKIP {run_id}")
            return True

        run_config = dict(config)
        run_config["TARGET_DEVICE"] = device
        base.LOG_DIR.mkdir(parents=True, exist_ok=True)
        stdout_path = base.LOG_DIR / f"{run_id}.out"
        stderr_path = base.LOG_DIR / f"{run_id}.err"

        temp_base = "/tmp" if os.path.exists("/tmp") else tempfile.gettempdir()
        temporary_dir = tempfile.mkdtemp(prefix=f"{run_id[:20]}-", dir=temp_base)

        environment = os.environ.copy()
        environment["HF_METRICS_CACHE"] = os.path.join(temporary_dir, "metrics")
        environment["HF_DATASETS_CACHE"] = os.path.join(temporary_dir, "datasets")
        environment["TMPDIR"] = temporary_dir
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        environment["PYTHONPATH"] = f"{REPO_ROOT}:{environment.get('PYTHONPATH', '')}"

        runner_script = os.path.abspath(__file__)

        with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    runner_script,
                    "--run-config",
                    json.dumps(run_config),
                ],
                stdout=stdout_file,
                stderr=stderr_file,
                env=environment,
                start_new_session=(sys.platform != "win32"),
            )
            control_id = base.get_control_id(run_id)
            registry_path = base.active_registry_path(control_id)
            request_path = base.stop_request_path(control_id)
            request_path.unlink(missing_ok=True)

            record = {
                "batch_size": int(config.get("batch_size", 32)),
                "control_id": control_id,
                "device": device,
                "pid": process.pid,
                "process_start_ticks": base.process_start_ticks(process.pid),
                "run_id": run_id,
                "started_at": time.time(),
                "stderr_path": str(stderr_path.resolve()),
                "stdout_path": str(stdout_path.resolve()),
            }
            base.atomic_write_json(registry_path, record)
            print(f"START [{control_id}] pid={process.pid} {run_id} on {device}")
            return_code = process.wait()

        if request_path.exists():
            print(f"STOPPED [{control_id}] {run_id} by operator request")
            return True

        if return_code != 0:
            print(f"ERROR {run_id}: see {stderr_path}")
            return False

        final_metric = base.read_latest_metric_line(stdout_path)
        print(f"DONE [{control_id}] {run_id}; {final_metric}")
        return True

    finally:
        if registry_path is not None:
            registry_path.unlink(missing_ok=True)
        if request_path is not None:
            request_path.unlink(missing_ok=True)
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)


# =============================================================================
# Main Suite Dispatcher
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Master Scheduler for FDA-LoRA Thesis Experiments")
    parser.add_argument("suite_pos", nargs="?", default=None, help="Optional positional suite name (e.g. pure, toy, threshold)")
    parser.add_argument("group_pos", nargs="?", default=None, choices=["plan", "start", "status", "stop", "drop", "summary"], help="Optional positional group action")
    parser.add_argument(
        "--suite",
        choices=[
            "pure",
            "calibration",
            "calib",
            "non_iid",
            "master",
            "fillers",
            "threshold",
            "threshold_controller",
            "thresh",
            "toy",
            "all",
        ],
        default=None,
        help="Target benchmark suite",
    )
    parser.add_argument(
        "--group",
        choices=["plan", "start", "status", "stop", "drop", "summary"],
        default=None,
        help="Scheduler command",
    )
    parser.add_argument("--dry-run", action="store_true", help="Alias for --group plan")
    parser.add_argument("--status", action="store_true", help="Alias for --group status")
    parser.add_argument("--stop", type=str, default=None, metavar="RUN_ID", help="Stop an active run by ID")
    parser.add_argument("--stage", choices=["all", "rolora", "fda_k05", "fda_k1", "fda_all"], default="all", help="Stage filter for threshold experiments")
    parser.add_argument("--lr", type=float, default=0.10, help="Client learning rate override (default: 0.10)")
    parser.add_argument("--emax", "--e-max", type=int, default=15, help="Max local epoch bound e_max for toy experiments (default: 15)")
    parser.add_argument("--run-config", help="Internal child worker invocation JSON string")
    parser.add_argument("--run-id", help="Target run ID for stop/drop commands")
    parser.add_argument("--reason", default="operator stopped run", help="Reason for stopping/dropping run")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Datasets to run (e.g. qnli mnli sst2 qqp)")
    parser.add_argument("--clients", nargs="+", type=int, default=[20, 50], help="Client counts (e.g. 20 50)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds (e.g. 42 43 or 42 43 44)")
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.5], help="Controller setpoint ratios")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter for non-IID")
    parser.add_argument("--tier", default="all", help="Filler tier (tier1, tier2, all)")
    parser.add_argument("--devices", default="cuda:0,cuda:1", help="Compute devices (e.g. cuda:0,cuda:1)")
    parser.add_argument("--all", action="store_true", help="Include completed runs in plan view")
    args = parser.parse_args()

    # 1. Child Worker Target
    if args.run_config:
        execute_run(args.run_config)
        return

    # Resolve positional vs flagged suite and group
    suite_str = args.suite or args.suite_pos or "pure"
    suite = suite_str.lower()

    if args.stop:
        group = "stop"
        args.run_id = args.stop
    elif args.status:
        group = "status"
    elif args.dry_run:
        group = "plan"
    else:
        group = args.group or args.group_pos or "plan"

    # Operator commands: status / stop / drop
    if group == "status":
        base.list_active_runs()
        return

    if group == "stop":
        target = args.run_id
        if not target:
            print("Error: --run-id or --stop <RUN_ID> is required to stop a run.")
            sys.exit(1)
        base.stop_active_run(target, args.reason)
        return

    # Configure base runner environment
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    base.DEVICES = devices

    # Parse datasets
    selected_datasets = list(DATASET_COST_ORDER) if "all" in args.datasets else args.datasets

    # Resolve seeds defaults
    if args.seeds is None:
        if suite == "toy":
            seeds = [42, 43, 44]
        else:
            seeds = DEFAULT_SEEDS
    else:
        seeds = args.seeds

    # 2. Assemble target suite configurations
    configs = []

    if suite in ("pure", "all"):
        configs.extend(build_pure_configs(selected_datasets, args.clients, seeds, args.ratios))
    if suite in ("calibration", "calib", "all"):
        configs.extend(build_calib_configs(selected_datasets, args.clients, seeds, args.ratios))
    if suite in ("non_iid", "all"):
        configs.extend(build_non_iid_configs(selected_datasets, args.clients, seeds, alpha=args.alpha))
    if suite in ("master", "all"):
        configs.extend(build_master_configs())
    if suite in ("fillers", "all"):
        configs.extend(build_filler_configs(tier=args.tier))
    if suite in ("threshold", "threshold_controller", "thresh", "all"):
        thresh_datasets = ["mnli"] if "all" in args.datasets else selected_datasets
        thresh_clients = [20] if args.clients == [20, 50] else args.clients
        configs.extend(build_threshold_configs(
            datasets=thresh_datasets,
            client_counts=thresh_clients,
            seeds=seeds,
            stage_filter=args.stage,
            learning_rate=args.lr,
        ))
    if suite in ("toy", "all"):
        configs.extend(build_toy_configs(
            ratios=args.ratios,
            seeds=seeds,
            e_max=args.emax,
        ))

    # Remove duplicates preserving order
    seen_ids = set()
    unique_configs = []
    for cfg in configs:
        if cfg["extras"] not in seen_ids:
            seen_ids.add(cfg["extras"])
            unique_configs.append(cfg)
    configs = unique_configs

    # 3. Handle commands
    if group == "plan":
        print(f"\n{'='*70}")
        print(f"EXPERIMENT PLAN: Suite='{suite.upper()}' ({len(configs)} total configurations)")
        print(f"{'='*70}")
        completed_count = 0
        for i, cfg in enumerate(configs, start=1):
            done = is_completed(cfg)
            if done:
                completed_count += 1
            if not done or args.all:
                status_str = "[COMPLETED]" if done else "[PENDING]  "
                print(f" {i:3d}. {status_str} {cfg['extras']}")
        print(f"{'='*70}")
        print(f"Summary: {completed_count}/{len(configs)} completed, {len(configs) - completed_count} pending.")
        print(f"{'='*70}\n")
        return

    if group == "start":
        print(f"\nStarting Suite '{suite.upper()}' on devices: {devices}")
        pending = [cfg for cfg in configs if not is_completed(cfg)]
        print(f"Total runnable jobs: {len(pending)} (skipping {len(configs) - len(pending)} already completed)\n")
        if not pending:
            print("All runs in this suite are already completed!")
            return

        # Wire runner into base main phase using our custom runner
        base.MAIN_METHODS = [
            "FDA-RoLoRA-DualTrack-Pure",
            "FDA-RoLoRA-DualTrack-Calibrated",
            "FDA-RoLoRA-DualTrack",
            "FedAvg-RoLoRA",
        ]
        base.run_main_phase(pending, job_runner=run_config_on_device)
        print(f"\n=== SUITE '{suite.upper()}' COMPLETED ===")


if __name__ == "__main__":
    main()
