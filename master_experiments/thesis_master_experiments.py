#!/usr/bin/env python3
"""
Master Orchestrator for the Complete Thesis Benchmark Suite (Polytechnix Deployment).

Unified Execution Matrix:
  1. Part A (Pure IID Suite):
     - Tasks: QQP, QNLI, MNLI, SST-2 across C in {20, 50}, k in {1.0, 0.5}, Seeds in {42, 43}
     - Protocol: Matrix B first, NODISC (gamma=1.0), e_min=1, no round-1 guard
  2. Part B (Lean Non-IID Suite):
     - Tasks: QNLI, MNLI across C in {20, 50}, Dirichlet alpha=0.5, Seeds in {42, 43}
     - Methods: FDA Pure Emax=20 (k=0.5, k=1.0) & FedAvg-RoLoRA (Fixed E=10)
     - Rescues / completes the crashed MNLI C=20 k=1.0 run!
  3. Staged Memory-Safe Scheduling:
     - Tier 1: Batch-32 IID Heavyweights (QQP, QNLI, MNLI) at 6 jobs/GPU (12 slots total)
     - Tier 2: Batch-32 Non-IID Heavyweights (QNLI, MNLI) dynamically backfill
     - Tier 3: Batch-64 IID (SST-2) held at end, strictly capped at 2 jobs/GPU to prevent OOM
  4. Interactive Operator Control:
     - Each round's metrics are saved to CSV on the fly
     - 'status' shows live progress and latest accuracy
     - 'stop <CONTROL_ID>' cleanly halts plateaued runs and immediately dispatches the next job!
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import fcntl
except ImportError:
    fcntl = None

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR if (CURRENT_DIR / "thesis_experiments.py").exists() else CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import thesis_experiments as base
except ImportError as error:
    raise RuntimeError(
        f"Unable to import thesis_experiments.py from {REPO_ROOT}. "
        "Verify repository layout."
    ) from error


def _as_list(value: Any) -> list:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


ROLORA_METHOD = "FedAvg-RoLoRA"
PURE_METHOD = "FDA-RoLoRA-DualTrack-Pure"

DEFAULT_SEEDS = [42, 43]
DEFAULT_CLIENT_COUNTS = [20, 50]
DEFAULT_E_MIN = 1
DEFAULT_E_MAX = 20
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")

# Early stopping configurations
IID_EARLY_STOPPING_PATIENCE = 3
IID_EARLY_STOPPING_MIN_DELTA = 0.005
IID_EARLY_STOPPING_WINDOW = 3
IID_GATES = {
    "qqp": {20: 0.72, 50: 0.72},
    "qnli": {20: 0.80, 50: 0.80},
    "mnli": {20: 0.60, 50: 0.55},
    "sst2": {20: 0.85, 50: 0.85},
}

NONIID_EARLY_STOPPING_PATIENCE = 5
NONIID_EARLY_STOPPING_MIN_DELTA = 0.005
NONIID_EARLY_STOPPING_WINDOW = 5
NONIID_GATES = {
    "qnli": {20: 0.80, 50: 0.80},
    "mnli": {20: 0.60, 50: 0.55},
}

DATASET_ORDER = ("qqp", "qnli", "mnli", "sst2")
DATASET_SETTINGS = {
    "qqp": {
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "rounds": {20: 50, 50: 30},
        "learning_rate": 0.10,
    },
    "qnli": {
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "rounds": {20: 50, 50: 30},
        "learning_rate": 0.10,
    },
    "mnli": {
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "rounds": {20: 50, 50: 30},
        "learning_rate": 0.10,
    },
    "sst2": {
        "batch_size": 64,
        "layers": list(range(9, 12)),
        "rounds": {20: 50, 50: 30},
        "learning_rate": 0.10,
    },
}

RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "round_metrics"
LOG_DIR = RESULTS_DIR / "logs" / "thesis_master"
RUN_CONTROL_DIR = RESULTS_DIR / "run_control_master"
ACTIVE_RUNS_DIR = RUN_CONTROL_DIR / "active"
STOP_REQUESTS_DIR = RUN_CONTROL_DIR / "stop_requests"
DROPPED_RUNS_DIR = RUN_CONTROL_DIR / "dropped"
OPERATOR_EVENTS_FILE = RUN_CONTROL_DIR / "operator_events.jsonl"
SCHEDULER_LOCK_FILE = RUN_CONTROL_DIR / "scheduler.lock"

JOBS_PER_GPU = max(1, int(os.environ.get("THESIS_MASTER_JOBS_PER_GPU", "3")))
BATCH64_JOBS_PER_GPU = max(
    1,
    min(JOBS_PER_GPU, int(os.environ.get("THESIS_MASTER_BATCH64_JOBS_PER_GPU", "2"))),
)


def configure_base_runner() -> None:
    """Wire custom paths, scheduler settings, and method registration into base runner."""
    base.METHODS[ROLORA_METHOD] = {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
    }
    base.METHODS[PURE_METHOD] = {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
    }
    base.METHOD_SLUGS[PURE_METHOD] = "fda-pure"
    base.METHOD_SLUGS[ROLORA_METHOD] = "rolora"
    base.MAIN_METHODS = [PURE_METHOD, ROLORA_METHOD]
    base.MAIN_DATASET_ORDER = DATASET_ORDER
    base.MAIN_CLIENT_PRIORITY = (20, 50)
    base.MAIN_INTERRUPTED_SNAPSHOT_RUN_IDS = frozenset()
    base.MAIN_JOBS_PER_GPU = JOBS_PER_GPU
    base.MAIN_BATCH64_JOBS_PER_GPU = BATCH64_JOBS_PER_GPU
    base.REPO_ROOT = REPO_ROOT
    base.RESULTS_DIR = RESULTS_DIR
    base.METRICS_DIR = METRICS_DIR
    base.is_completed = is_master_completed
    base.LOG_DIR = LOG_DIR
    base.RUN_CONTROL_DIR = RUN_CONTROL_DIR
    base.ACTIVE_RUNS_DIR = ACTIVE_RUNS_DIR
    base.STOP_REQUESTS_DIR = STOP_REQUESTS_DIR
    base.OPERATOR_EVENTS_FILE = OPERATOR_EVENTS_FILE
    base.main_config_priority = master_config_priority
    base.wait_for_registered_run = wait_for_registered_master_run
    base.run_config_on_device = run_master_config_on_device


def master_config_priority(config: dict) -> tuple:
    """
    Tiered scheduling priority:
      Tier 0: Batch-32 IID tasks (QQP, QNLI, MNLI) - maximum concurrency sprint (6 jobs/GPU).
      Tier 1: Batch-32 Non-IID tasks (QNLI, MNLI alpha=0.5) - continuous follow-through.
      Tier 2: Batch-64 IID task (SST-2) - backfills dynamically, strictly capped at 2 jobs/GPU.
    """
    is_batch64 = int(config.get("batch_size", 32)) >= 64
    is_non_iid = float(config.get("alpha", 1000.0)) < 10.0

    if is_batch64:
        tier = 2
    elif is_non_iid:
        tier = 1
    else:
        tier = 0

    ds_idx = DATASET_ORDER.index(config["ds_name"]) if config["ds_name"] in DATASET_ORDER else 99
    client_order = int(config["num_clients"])
    ratio_val = float(config.get("controller_ratio", 1.0))
    ratio_order = 0 if ratio_val == 1.0 else 1
    seed_order = int(config["seed"])

    return (tier, ds_idx, client_order, ratio_order, seed_order)


def build_master_configs() -> list[dict]:
    """Build all 48 master benchmark configurations (IID + Non-IID across seeds 42 and 43)."""
    configs = []

    # ==========================================
    # 1. PART A: PURE IID SUITE (32 Configurations)
    # ==========================================
    for ds in DATASET_ORDER:
        settings = DATASET_SETTINGS[ds]
        for c in DEFAULT_CLIENT_COUNTS:
            rounds = settings["rounds"][c]
            gate = IID_GATES[ds][c]
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
                    cfg["METHOD_NAME"] = PURE_METHOD
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
                        "e_min": DEFAULT_E_MIN,
                        "max_epochs": DEFAULT_E_MAX,
                        "theta": 0.0,
                        "controller_ratio": r,
                        "enable_b_discount": False,
                        "b_discount_factor": 1.0,
                        "early_stopping_start_accuracy": gate,
                        "plateau_min_accuracy": gate,
                        "early_stopping_min_rounds": 6,
                        "early_stopping_patience": IID_EARLY_STOPPING_PATIENCE,
                        "early_stopping_min_delta": IID_EARLY_STOPPING_MIN_DELTA,
                        "early_stopping_window": IID_EARLY_STOPPING_WINDOW,
                        "stagnation_round": 10_000,
                        "stagnation_tolerance": 0.0,
                        "start_matrix": "B",
                        "first_round_guard": False,
                    })
                    ratio_tag = f"CRATIO{r:g}"
                    extra_id = (
                        f"THESIS-RBASE-PURE-DS-{ds}-C{c}-R{rounds}"
                        f"-M-fda-pure-LR{settings['learning_rate']:g}-RK{LORA_RANK}"
                        f"-E{DEFAULT_E_MIN}TO{DEFAULT_E_MAX}-{ratio_tag}-NODISC-BSTART-NOGUARD-S{s}"
                        f"-ES{gate:g}-MR6-P{IID_EARLY_STOPPING_PATIENCE}-D{IID_EARLY_STOPPING_MIN_DELTA:g}-W{IID_EARLY_STOPPING_WINDOW}"
                    )
                    cfg["extras"] = extra_id
                    configs.append(cfg)

    # ==========================================
    # 2. PART B: LEAN NON-IID SUITE (16 Configurations)
    # ==========================================
    for ds in ["qnli", "mnli"]:
        settings = DATASET_SETTINGS[ds]
        for c in DEFAULT_CLIENT_COUNTS:
            rounds = settings["rounds"][c]
            gate = NONIID_GATES[ds][c]
            mr = int(rounds * 0.6)  # 30 rounds for R=50, 18 rounds for R=30
            for s in DEFAULT_SEEDS:
                # B1: RoLoRA E10 Baseline
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
                cfg_ro["METHOD_NAME"] = ROLORA_METHOD
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
                    "early_stopping_patience": NONIID_EARLY_STOPPING_PATIENCE,
                    "early_stopping_min_delta": NONIID_EARLY_STOPPING_MIN_DELTA,
                    "early_stopping_window": NONIID_EARLY_STOPPING_WINDOW,
                })
                extra_ro = (
                    f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                    f"-M-rolora-E10-LR0.1-RK{LORA_RANK}-BSTART-S{s}"
                    f"-ES{gate:g}-MR{mr}-P{NONIID_EARLY_STOPPING_PATIENCE}-D{NONIID_EARLY_STOPPING_MIN_DELTA:g}-W{NONIID_EARLY_STOPPING_WINDOW}"
                )
                cfg_ro["extras"] = extra_ro
                configs.append(cfg_ro)

                # B2: FDA Pure Emax=20 (k=0.5 and k=1.0)
                # Note: k=1.0 on C=20 rescues the crashed Round 4 run!
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
                    cfg_fda["METHOD_NAME"] = PURE_METHOD
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
                        "e_min": DEFAULT_E_MIN,
                        "max_epochs": DEFAULT_E_MAX,
                        "theta": 0.0,
                        "controller_ratio": r,
                        "enable_b_discount": False,
                        "b_discount_factor": 1.0,
                        "start_matrix": "B",
                        "first_round_guard": False,
                        "early_stopping_start_accuracy": gate,
                        "plateau_min_accuracy": gate,
                        "early_stopping_min_rounds": mr,
                        "early_stopping_patience": NONIID_EARLY_STOPPING_PATIENCE,
                        "early_stopping_min_delta": NONIID_EARLY_STOPPING_MIN_DELTA,
                        "early_stopping_window": NONIID_EARLY_STOPPING_WINDOW,
                    })
                    r_tag = f"CRATIO{r:g}"
                    extra_fda = (
                        f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                        f"-M-fda-pure-LR0.1-RK{LORA_RANK}-E{DEFAULT_E_MIN}TO{DEFAULT_E_MAX}-{r_tag}-NODISC-BSTART-NOGUARD-S{s}"
                        f"-ES{gate:g}-MR{mr}-P{NONIID_EARLY_STOPPING_PATIENCE}-D{NONIID_EARLY_STOPPING_MIN_DELTA:g}-W{NONIID_EARLY_STOPPING_WINDOW}"
                    )
                    cfg_fda["extras"] = extra_fda
                    configs.append(cfg_fda)

    return sorted(configs, key=master_config_priority)


def is_master_completed(config: dict) -> bool:
    """Check if run is finished. Aborted/crashed runs with <5 rounds are flagged incomplete."""
    pattern = str(METRICS_DIR / f"{config['extras']}_*.csv")
    for path in glob.glob(pattern):
        try:
            import pandas as pd
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not frame.empty and "round" in frame.columns:
            # If terminated in fewer than 5 rounds without early stopping gate met, re-run!
            if len(frame) < 5:
                return False
            return True
    return False


def is_dropped(config: dict) -> bool:
    cid = base.get_control_id(config["extras"])
    return (DROPPED_RUNS_DIR / f"{cid}.json").exists()


def run_master_config_on_device(config: dict, device: str) -> bool:
    """Spawns this runner script itself with --run-config for child worker isolation."""
    run_id = config["extras"]
    registry_path = None
    request_path = None
    try:
        if is_master_completed(config):
            print(f"SKIP {run_id}")
            return True
        stdout_path = LOG_DIR / f"{run_id}.stdout.log"
        stderr_path = LOG_DIR / f"{run_id}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

        run_config = dict(config)
        run_config["TARGET_DEVICE"] = device

        environment = os.environ.copy()
        current_pythonpath = environment.get("PYTHONPATH", "")
        extra_paths = [
            str(REPO_ROOT),
            str(REPO_ROOT / "pure_experiments"),
            str(REPO_ROOT / "non_iid_experiments"),
        ]
        if current_pythonpath:
            extra_paths.append(current_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(extra_paths)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    os.path.abspath(__file__),
                    "--run-config",
                    json.dumps(run_config),
                ],
                stdout=stdout_file,
                stderr=stderr_file,
                env=environment,
                start_new_session=True,
            )
            control_id = base.get_control_id(run_id)
            registry_path = base.active_registry_path(control_id)
            request_path = base.stop_request_path(control_id)
            request_path.unlink(missing_ok=True)
            record = {
                "batch_size": int(config["batch_size"]),
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
            base.append_operator_event({
                "action": "worker_spawned",
                "control_id": control_id,
                "device": device,
                "pid": process.pid,
                "run_id": run_id,
            })
            print(f"START [{control_id}] pid={process.pid} {run_id} on {device}")
            return_code = process.wait()

        if request_path.exists():
            print(f"STOPPED [{control_id}] {run_id} by operator request")
            return True
        if return_code != 0:
            print(f"ERROR {run_id}: {stderr_path}")
            return False
        final_metric = base.read_latest_metric_line(stdout_path)
        print(f"DONE [{control_id}] {run_id}; {final_metric}")
        return True
    finally:
        if registry_path is not None:
            registry_path.unlink(missing_ok=True)
        if request_path is not None:
            request_path.unlink(missing_ok=True)


def wait_for_registered_master_run(
    config: dict | None,
    record: dict,
    poll_seconds: int = 5,
) -> bool:
    run_id = record["run_id"]
    control_id = record["control_id"]
    print(f"ADOPT [{control_id}] pid={record['pid']} {run_id} on {record['device']}")
    while base.process_matches_registry(record):
        time.sleep(poll_seconds)
    base.active_registry_path(control_id).unlink(missing_ok=True)
    request_path = base.stop_request_path(control_id)
    was_stopped = request_path.exists()
    request_path.unlink(missing_ok=True)
    if was_stopped or (config is not None and is_dropped(config)):
        print(f"ADOPTED STOPPED [{control_id}] {run_id}")
        return True
    if config is None:
        print(f"RELEASE [{control_id}] detached worker finished")
        return True
    if is_master_completed(config):
        final_metric = base.read_latest_metric_line(record["stdout_path"])
        print(f"ADOPTED DONE [{control_id}] {run_id}; {final_metric}")
        return True
    print(f"ADOPTED ERROR [{control_id}] {run_id}: worker exited without a valid metrics CSV")
    return False


def resolve_master_optimizers():
    """Dynamically resolve fedavg_rolora_opt and fda_rolora_dual_track_pure_opt."""
    fedavg_opt = None
    pure_opt = None

    try:
        from non_iid_experiments.fda_rolora_non_iid import fedavg_rolora_opt
        fedavg_opt = fedavg_rolora_opt
    except ImportError:
        pass

    try:
        from pure_experiments.fda_rolora_pure import fda_rolora_dual_track_pure_opt
        pure_opt = fda_rolora_dual_track_pure_opt
    except ImportError:
        pass

    if pure_opt is None:
        try:
            from fda_rolora_pure import fda_rolora_dual_track_pure_opt
            pure_opt = fda_rolora_dual_track_pure_opt
        except ImportError:
            pass

    return fedavg_opt, pure_opt


def execute_run(config_json: str) -> None:
    """Child worker execution target."""
    import torch
    from fdaopt import utils
    from fdaopt.training import fda_opt, fed_opt

    fedavg_opt, pure_opt = resolve_master_optimizers()
    if pure_opt is not None:
        fda_opt.fda_rolora_dual_track_pure_opt = pure_opt
        fed_opt.fda_rolora_dual_track_pure_opt = pure_opt
    if fedavg_opt is not None:
        fed_opt.rolora_fedavg_opt = fedavg_opt

    config = json.loads(config_json)
    device = config.pop("TARGET_DEVICE", "cuda:0")
    base.install_runtime_patches(config)

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    utils.DEVICE_RAM_PROGRAM = "performance"
    utils.set_device(device)
    utils.set_seed(config["seed"])

    method = config.get("METHOD_NAME", config.get("method"))
    print(f"[{time.strftime('%H:%M:%S')}] Starting {config['extras']} ({method}) on {device}...")

    if method == ROLORA_METHOD:
        if fedavg_opt is None:
            raise RuntimeError("fedavg_rolora_opt could not be imported.")
        fedavg_opt(config)
    elif method == PURE_METHOD:
        if pure_opt is None:
            raise RuntimeError("fda_rolora_dual_track_pure_opt could not be imported.")
        pure_opt(config)
    else:
        raise ValueError(f"Unknown method {method}")

    print(f"[{time.strftime('%H:%M:%S')}] Execution finished for {config['extras']}.")


def print_plan(configs: list[dict], show_all: bool = False) -> None:
    completed = [cfg for cfg in configs if is_master_completed(cfg)]
    pending = [cfg for cfg in configs if not is_master_completed(cfg)]
    displayed = configs if show_all else pending

    sep = "=" * 120
    print(f"\n{sep}")
    if show_all:
        print(f"MASTER THESIS EXPERIMENT AUDIT (Total: {len(configs)} | Done: {len(completed)} | Pending: {len(pending)})")
    else:
        print(f"MASTER THESIS EXECUTION PLAN (Pending: {len(pending)} to run | {len(completed)} already completed omitted)")
    print(f"Scope: Pure IID (QQP, QNLI, MNLI, SST-2) + Lean Non-IID (MNLI, QNLI alpha=0.5) across Seeds 42, 43")
    print(f"Stage 1: Batch-32 IID tasks execute first at {JOBS_PER_GPU} lanes/GPU.")
    print(f"Stage 2: Batch-32 Non-IID tasks follow through into available lanes.")
    print(f"Stage 3: Batch-64 (SST-2) queues at end, strictly capped at {BATCH64_JOBS_PER_GPU} jobs/GPU to prevent OOM.")
    print(sep)
    header = f"{'#':<3} | {'Control ID':<10} | {'DS':<5} | {'Alpha':<6} | {'BS':<3} | {'C':<3} | {'R':<3} | {'Method / Ratio':<20} | {'Gate':<5} | {'Seed':<4} | {'Status':<10}"
    print(header)
    print("-" * len(header))
    current_tier = None
    for idx, cfg in enumerate(displayed, 1):
        is_b64 = int(cfg.get("batch_size", 32)) >= 64
        is_niid = float(cfg.get("alpha", 1000.0)) < 10.0
        tier_label = "STAGE 3 (SST-2)" if is_b64 else ("STAGE 2 (NON-IID)" if is_niid else "STAGE 1 (IID)")
        if current_tier is not None and tier_label != current_tier:
            print("-" * len(header))
        current_tier = tier_label

        cid = base.get_control_id(cfg["extras"])
        alpha_str = f"{cfg.get('alpha', 1000.0):g}"
        method = cfg.get("METHOD_NAME", "")
        if method == ROLORA_METHOD:
            m_str = f"RoLoRA (E{cfg.get('local_epochs', 10)})"
        else:
            m_str = f"FDA Pure (k={cfg.get('controller_ratio', 1.0)})"

        status_str = "DONE" if is_master_completed(cfg) else "PENDING"
        line = (
            f"{idx:<3} | {cid:<10} | {cfg['ds_name'].upper():<5} | {alpha_str:<6} | {cfg['batch_size']:<3} | {cfg['num_clients']:<3} | "
            f"{cfg['total_rounds']:<3} | {m_str:<20} | {cfg['early_stopping_start_accuracy']:<5.0%} | {cfg['seed']:<4} | {status_str:<10}"
        )
        print(line)
    print(f"{sep}\n")


def run_master_phase(configs: list[dict], job_runner=None) -> None:
    job_runner = run_master_config_on_device if job_runner is None else job_runner
    active_records = base.load_active_run_records()
    active_run_ids = {record["run_id"] for record in active_records}
    configs_by_run_id = {config["extras"]: config for config in configs}
    adopted_configs = {
        run_id: configs_by_run_id[run_id]
        for run_id in active_run_ids
        if run_id in configs_by_run_id
    }
    pending = [
        config
        for config in configs
        if config["extras"] not in active_run_ids and not is_master_completed(config)
    ]
    pending_queue = sorted(pending, key=master_config_priority)

    print(
        f"Scheduled {len(pending)} queued and "
        f"{len(adopted_configs)} adopted configurations"
    )
    print(
        f"Skipped {len(configs) - len(pending) - len(adopted_configs)} "
        "completed configurations"
    )
    print(
        f"Master scheduler: {JOBS_PER_GPU} dynamic lanes/GPU across {len(base.DEVICES)} GPUs; "
        f"batch-64 cap={BATCH64_JOBS_PER_GPU}/GPU"
    )

    if not pending and not adopted_configs:
        return

    if pending:
        base.print_restart_manifest(pending_queue)

    lanes = [
        (device, lane_index)
        for device in base.DEVICES
        for lane_index in range(JOBS_PER_GPU)
    ]
    active_batch64 = {device: 0 for device in base.DEVICES}

    def take_allowed(device: str):
        for index, config in enumerate(pending_queue):
            is_batch64 = int(config.get("batch_size", 32)) >= 64
            if is_batch64 and active_batch64[device] >= BATCH64_JOBS_PER_GPU:
                continue
            return pending_queue.pop(index)
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        future_to_lane = {}
        idle_lanes = list(lanes)

        for record in active_records:
            device = record.get("device")
            matching_lanes = [lane for lane in idle_lanes if lane[0] == device]
            if not matching_lanes:
                continue
            lane = matching_lanes[0]
            config = configs_by_run_id.get(record["run_id"])
            future = executor.submit(wait_for_registered_master_run, config, record)
            future_to_lane[future] = (lane, config, record)
            idle_lanes.remove(lane)
            batch_size = (
                config["batch_size"]
                if config is not None
                else int(record.get("batch_size", 64))
            )
            if batch_size >= 64:
                active_batch64[device] += 1

        def launch_available():
            made_progress = True
            while made_progress:
                made_progress = False
                for lane in list(idle_lanes):
                    device, _lane_index = lane
                    config = take_allowed(device)
                    if config is None:
                        continue
                    future = executor.submit(job_runner, config, device)
                    future_to_lane[future] = (lane, config, None)
                    idle_lanes.remove(lane)
                    if int(config.get("batch_size", 32)) >= 64:
                        active_batch64[device] += 1
                    made_progress = True

        launch_available()

        while future_to_lane:
            completed, _ = concurrent.futures.wait(
                future_to_lane,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in completed:
                lane, completed_config, adopted_record = future_to_lane.pop(future)
                idle_lanes.append(lane)
                device = lane[0]
                if completed_config is not None:
                    if int(completed_config.get("batch_size", 32)) >= 64:
                        active_batch64[device] = max(0, active_batch64[device] - 1)
                elif adopted_record is not None:
                    if int(adopted_record.get("batch_size", 64)) >= 64:
                        active_batch64[device] = max(0, active_batch64[device] - 1)
                launch_available()


def run_all(configs: list[dict]) -> None:
    runnable = [config for config in configs if not is_dropped(config)]
    RUN_CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with SCHEDULER_LOCK_FILE.open("w", encoding="utf-8") as lock_file:
        if fcntl is not None:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SystemExit(
                    "Another Master Thesis scheduler is already active. "
                    "Use the status command instead of starting a second one."
                ) from error
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()

        print_plan(configs)
        print(f"Launching master scheduler across {JOBS_PER_GPU} lanes/GPU (max concurrent: {len(base.DEVICES) * JOBS_PER_GPU})...")
        print(f"Batch-64 safety cap: max {BATCH64_JOBS_PER_GPU} SST-2 runs per GPU concurrently.\n")
        run_master_phase(runnable)
        print("\n=== ALL MASTER THESIS EXPERIMENTS COMPLETED ===")


def list_runs() -> None:
    """Lists currently active runs with live round progress and test accuracy from stdout logs."""
    active_records = []
    for p in ACTIVE_RUNS_DIR.glob("*.json"):
        if p.is_file():
            try:
                active_records.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    if not active_records:
        print("No active master experiments currently running.")
        return
    print(f"\nActive Master Experiments ({len(active_records)}):")
    for rec in active_records:
        cid = rec.get("control_id", "unknown")
        pid = rec.get("pid", "unknown")
        dev = rec.get("device", "unknown")
        rid = rec.get("run_id", "unknown")
        elapsed = (time.time() - rec.get("started_at", time.time())) / 60.0
        stdout_path = Path(rec.get("stdout_path", ""))
        metric_line = ""
        if stdout_path.exists():
            try:
                lines = stdout_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                for line in reversed(lines[-20:]):
                    if "Round" in line and "finished" in line:
                        metric_line = f" -> {line.strip()}"
                        break
            except Exception:
                pass
        print(f"  [{cid}] PID={pid} DEV={dev} Elapsed={elapsed:.1f}m{metric_line}\n      ID: {rid}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Complete Master Thesis Experiment Orchestrator")
    parser.add_argument(
        "--group",
        choices=["plan", "run", "start", "status", "stop", "drop", "dropped"],
        default="plan",
        help="Action to execute (default: plan)",
    )
    parser.add_argument(
        "--run-config",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Target Run ID or Control ID for stop/drop",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="operator request",
        help="Reason for stop/drop",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include already completed configurations in plan output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_config:
        execute_run(args.run_config)
        return

    configure_base_runner()
    configs = build_master_configs()

    if args.group == "plan":
        print_plan(configs, show_all=args.all)
    elif args.group in ("run", "start"):
        run_all(configs)
    elif args.group == "status":
        list_runs()
    elif args.group in {"stop", "drop"}:
        if not args.run_id:
            raise SystemExit(f"--run-id is required for {args.group}")
        if args.group == "stop":
            base.stop_active_run(args.run_id, args.reason)
        elif args.group == "drop":
            base.drop_active_run(args.run_id, args.reason)


if __name__ == "__main__":
    main()
