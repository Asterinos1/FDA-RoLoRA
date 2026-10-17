#!/usr/bin/env python3
"""
Master Orchestrator for Non-IID Dirichlet (alpha=0.5) Experiments.

Protocol:
  1. Methods:
     - FedAvg-RoLoRA (Fixed E=10, E=20)
     - FDA-RoLoRA-DualTrack-Pure (Adaptive E=1..20/40, k=1.0, NODISC, Matrix B first)
  2. Non-IID Dirichlet Partition: alpha = 0.5 (configurable)
  3. Client scale: C=20, R=50 (Full participation)
  4. Datasets: QNLI and MNLI
  5. 6-Lane Continuous Multi-GPU Scheduler:
     - Saturates all 6 slots at all times across 2 GPUs (3 jobs/GPU on A10s).
     - QNLI and MNLI queue continuously: QNLI takes initial 4 slots, MNLI immediately
       fills remaining 2 slots, and backfills dynamically as runs complete.
  6. Robust Non-IID Early Stopping:
     - 60% minimum round threshold (Round >= 30 for R=50).
     - 5-round moving average window.
     - Patience = 3, min_delta = 0.005.
     - Gates: QNLI >= 0.80, MNLI >= 0.60.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
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

# Resolve REPO_ROOT: support running inside non_iid_experiments/ or repo root
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

DEFAULT_SEEDS = [42]
DEFAULT_CLIENT_COUNTS = [20]
DEFAULT_ALPHA = 0.5
DEFAULT_ROUNDS = 50
DEFAULT_E_MIN = 1
DEFAULT_LOCAL_EPOCHS = [10, 20]

# Paired FDA configurations:
# 1. E_max = 20, k = 1.0 -> Convexity compensated matching baseline E=10 (realizes e_bar ≈ 11.4)
# 2. E_max = 40, k = 1.0 -> Convexity compensated matching baseline E=20 (realizes e_bar ≈ 22.8)
DEFAULT_FDA_CONFIGS = [
    (20, 1.0),
    (40, 1.0),
]
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")

# Non-IID Early Stopping Gates
EARLY_STOPPING_MIN_ROUNDS = 30  # 60% of 50 rounds
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.005
EARLY_STOPPING_WINDOW = 5
EARLY_STOPPING_GATES = {
    "qnli": 0.80,
    "mnli": 0.60,
}

DATASET_COST_ORDER = ("qnli", "mnli")
DATASET_SETTINGS = {
    "qnli": {
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "rounds": 50,
        "learning_rate": 0.10,
    },
    "mnli": {
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "rounds": 50,
        "learning_rate": 0.10,
    },
}

RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "round_metrics"
LOG_DIR = RESULTS_DIR / "logs" / "thesis_non_iid"
RUN_CONTROL_DIR = RESULTS_DIR / "run_control_non_iid"
ACTIVE_RUNS_DIR = RUN_CONTROL_DIR / "active"
STOP_REQUESTS_DIR = RUN_CONTROL_DIR / "stop_requests"
DROPPED_RUNS_DIR = RUN_CONTROL_DIR / "dropped"
OPERATOR_EVENTS_FILE = RUN_CONTROL_DIR / "operator_events.jsonl"
SCHEDULER_LOCK_FILE = RUN_CONTROL_DIR / "scheduler.lock"

JOBS_PER_GPU = max(1, int(os.environ.get("THESIS_NONIID_JOBS_PER_GPU", "3")))
BATCH64_JOBS_PER_GPU = max(
    1,
    min(JOBS_PER_GPU, int(os.environ.get("THESIS_NONIID_BATCH64_JOBS_PER_GPU", "2"))),
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
    base.MAIN_METHODS = [ROLORA_METHOD, PURE_METHOD]
    base.MAIN_DATASET_ORDER = DATASET_COST_ORDER
    base.MAIN_CLIENT_PRIORITY = (20,)
    base.MAIN_INTERRUPTED_SNAPSHOT_RUN_IDS = frozenset()
    base.MAIN_JOBS_PER_GPU = JOBS_PER_GPU
    base.MAIN_BATCH64_JOBS_PER_GPU = BATCH64_JOBS_PER_GPU
    base.LOG_DIR = LOG_DIR
    base.RUN_CONTROL_DIR = RUN_CONTROL_DIR
    base.ACTIVE_RUNS_DIR = ACTIVE_RUNS_DIR
    base.STOP_REQUESTS_DIR = STOP_REQUESTS_DIR
    base.OPERATOR_EVENTS_FILE = OPERATOR_EVENTS_FILE
    base.main_config_priority = non_iid_config_priority
    base.wait_for_registered_run = wait_for_registered_non_iid_run
    base.run_config_on_device = run_non_iid_config_on_device

    # Allow custom device list via environment
    devices_env = os.environ.get("THESIS_NONIID_DEVICES", "")
    if devices_env:
        base.DEVICES = [d.strip() for d in devices_env.split(",") if d.strip()]


def non_iid_config_priority(config: dict) -> tuple:
    """
    Sort configurations to ensure smooth, continuous 6-lane saturation:
      1. Dataset: QNLI (faster turnaround) before MNLI.
      2. Compute Parity Tiers:
         - Tier 1 (rank 10): RoLoRA E10 and FDA Pure Emax=20 (k=1.0, e_bar ≈ 11.4)
         - Tier 2 (rank 20): RoLoRA E20 and FDA Pure Emax=40 (k=1.0, e_bar ≈ 22.8)
      3. Seed order.
    """
    ds_order = DATASET_COST_ORDER.index(config["ds_name"])
    method = config.get("METHOD_NAME", "")
    if method == ROLORA_METHOD:
        epochs = int(config.get("local_epochs", 10))
        tier_rank = 10 if epochs == 10 else 20
    else:
        e_max = int(config.get("max_epochs", 20))
        ratio_val = float(config.get("controller_ratio", 1.0))
        if e_max == 20 and ratio_val == 1.0:
            tier_rank = 10
        elif e_max == 40 and ratio_val == 1.0:
            tier_rank = 20
        else:
            tier_rank = 30

    return (
        ds_order,
        tier_rank,
        int(config.get("seed", 42)),
    )


def build_non_iid_configs(
    datasets: list[str],
    client_counts: list[int],
    seeds: list[int],
    alpha: float = DEFAULT_ALPHA,
    rounds: int = DEFAULT_ROUNDS,
    methods: list[str] | None = None,
    local_epochs: list[int] | None = None,
    fda_configs: list[tuple[int, float]] | None = None,
    e_min: int = DEFAULT_E_MIN,
) -> list[dict]:
    """Generate isolated Non-IID configurations with distinct EXTRA identifiers."""
    if methods is None:
        methods = [ROLORA_METHOD, PURE_METHOD]
    if local_epochs is None:
        local_epochs = DEFAULT_LOCAL_EPOCHS
    if fda_configs is None:
        fda_configs = DEFAULT_FDA_CONFIGS

    configs = []
    for dataset in datasets:
        settings = DATASET_SETTINGS[dataset]
        gate = EARLY_STOPPING_GATES.get(dataset, 0.80)
        total_rounds = rounds or settings["rounds"]
        min_rounds = int(0.6 * total_rounds)

        for num_clients in client_counts:
            for seed in seeds:
                # 1. FedAvg-RoLoRA Fixed Epoch Baselines
                if ROLORA_METHOD in methods:
                    for epochs in local_epochs:
                        config = base.make_config(
                            stage="non-iid",
                            dataset=dataset,
                            method=ROLORA_METHOD,
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
                        config.update(
                            {
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
                                "early_stopping_window": EARLY_STOPPING_WINDOW,
                            }
                        )
                        extra_id = (
                            f"THESIS-NONIID-A{alpha:g}-DS-{dataset}-C{num_clients}-R{total_rounds}"
                            f"-M-rolora-E{epochs}-LR{settings['learning_rate']:g}-RK{LORA_RANK}"
                            f"-BSTART-S{seed}-ES{gate:g}-MR{min_rounds}"
                            f"-P{EARLY_STOPPING_PATIENCE}-D{EARLY_STOPPING_MIN_DELTA:g}-W{EARLY_STOPPING_WINDOW}"
                        )
                        config["extras"] = extra_id
                        configs.append(config)

                # 2. FDA-RoLoRA-DualTrack-Pure Closed-Loop Runs
                if PURE_METHOD in methods:
                    for fda_emax, ratio in fda_configs:
                        config = base.make_config(
                            stage="non-iid",
                            dataset=dataset,
                            method=PURE_METHOD,
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
                        config.update(
                            {
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
                                "early_stopping_window": EARLY_STOPPING_WINDOW,
                            }
                        )
                        ratio_tag = f"CRATIO{ratio:g}"
                        extra_id = (
                            f"THESIS-NONIID-A{alpha:g}-DS-{dataset}-C{num_clients}-R{total_rounds}"
                            f"-M-fda-pure-LR{settings['learning_rate']:g}-RK{LORA_RANK}"
                            f"-E{e_min}TO{fda_emax}-{ratio_tag}-NODISC-BSTART-NOGUARD-S{seed}"
                            f"-ES{gate:g}-MR{min_rounds}-P{EARLY_STOPPING_PATIENCE}"
                            f"-D{EARLY_STOPPING_MIN_DELTA:g}-W{EARLY_STOPPING_WINDOW}"
                        )
                        config["extras"] = extra_id
                        configs.append(config)

    return sorted(configs, key=non_iid_config_priority)


def dropped_marker_path(run_id: str) -> Path:
    control_id = base.get_control_id(run_id)
    return DROPPED_RUNS_DIR / f"{control_id}.marker"


def is_dropped(config: dict) -> bool:
    return dropped_marker_path(config["extras"]).exists()


def is_non_iid_completed(config: dict) -> bool:
    """Accept run as completed only if total rounds reached or early stopping triggered."""
    pattern = str(METRICS_DIR / f"{config['extras']}_*.csv")
    for path in glob.glob(pattern):
        try:
            import pandas as pd
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "round" not in frame.columns:
            continue
        max_round = int(frame["round"].max())
        if max_round >= int(config["total_rounds"]):
            return True
        if "early_stopped" in frame.columns and frame["early_stopped"].dropna().any():
            return True
        # Check stdout log
        stdout_path = LOG_DIR / f"{config['extras']}.stdout.log"
        if stdout_path.exists():
            try:
                log_text = stdout_path.read_text(encoding="utf-8", errors="replace")
                if "Early stopping triggered" in log_text or "Completed execution" in log_text:
                    return True
            except Exception:
                pass
    return False


def run_non_iid_config_on_device(config: dict, device: str) -> bool:
    """Spawns this runner script itself with --run-config for child worker isolation."""
    run_id = config["extras"]
    registry_path = None
    request_path = None
    try:
        if is_non_iid_completed(config):
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
        non_iid_dir = str(REPO_ROOT / "non_iid_experiments")
        extra_paths = [str(REPO_ROOT), non_iid_dir]
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
            base.append_operator_event(
                {
                    "action": "worker_spawned",
                    "control_id": control_id,
                    "device": device,
                    "pid": process.pid,
                    "run_id": run_id,
                }
            )
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


def wait_for_registered_non_iid_run(
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
    if is_non_iid_completed(config):
        final_metric = base.read_latest_metric_line(record["stdout_path"])
        print(f"ADOPTED DONE [{control_id}] {run_id}; {final_metric}")
        return True
    print(f"ADOPTED ERROR [{control_id}] {run_id}: worker exited without completed metrics CSV")
    return False


def resolve_non_iid_optimizers():
    """Dynamically resolve fedavg_rolora_opt and fda_rolora_dual_track_pure_opt."""
    try:
        from non_iid_experiments.fda_rolora_non_iid import (
            fda_rolora_dual_track_pure_opt,
            fedavg_rolora_opt,
        )
        return fedavg_rolora_opt, fda_rolora_dual_track_pure_opt
    except ImportError:
        pass
    try:
        from fda_rolora_non_iid import (
            fda_rolora_dual_track_pure_opt,
            fedavg_rolora_opt,
        )
        return fedavg_rolora_opt, fda_rolora_dual_track_pure_opt
    except ImportError:
        pass

    candidates = [
        REPO_ROOT / "non_iid_experiments" / "fda_rolora_non_iid.py",
        CURRENT_DIR / "fda_rolora_non_iid.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("fda_non_iid_module", str(candidate))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return getattr(mod, "fedavg_rolora_opt"), getattr(mod, "fda_rolora_dual_track_pure_opt")

    raise RuntimeError("Unable to load fda_rolora_non_iid module.")


def execute_run(config_json: str) -> None:
    """Child worker execution target."""
    import torch
    from fdaopt import utils

    fedavg_opt, pure_opt = resolve_non_iid_optimizers()

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
        fedavg_opt(config)
    elif method == PURE_METHOD:
        pure_opt(config)
    else:
        raise ValueError(f"Unknown method {method}")

    print(f"[{time.strftime('%H:%M:%S')}] Execution finished for {config['extras']}.")


def print_plan(configs: list[dict]) -> None:
    sep = "=" * 115
    print(f"\n{sep}")
    print(f"NON-IID EXPERIMENT PLAN (Total Runs: {len(configs)})")
    print(f"Protocol: Dirichlet alpha={DEFAULT_ALPHA} | C=20 | R=50 | Matrix B First | Early Stopping >= 60% (R>=30)")
    print(f"Continuous 6-Lane Scheduling: 4 QNLI runs start immediately, 2 MNLI runs fill lanes 5-6, and backfill dynamically")
    print(sep)
    header = f"{'#':<3} | {'Control ID':<10} | {'DS':<5} | {'Method':<20} | {'Setting':<10} | {'Alpha':<5} | {'C/R':<7} | {'Gate':<5} | {'MinR':<4} | {'Seed':<4}"
    print(header)
    print("-" * len(header))
    current_ds = None
    for idx, cfg in enumerate(configs, 1):
        if current_ds is not None and cfg["ds_name"] != current_ds:
            print("-" * len(header))
        current_ds = cfg["ds_name"]
        cid = base.get_control_id(cfg["extras"])
        method = cfg["METHOD_NAME"]
        if method == ROLORA_METHOD:
            setting_str = f"E={cfg['local_epochs']}"
        else:
            setting_str = f"k={cfg['controller_ratio']} (1..{cfg['max_epochs']})"
        cr_str = f"{cfg['num_clients']}/{cfg['total_rounds']}"
        gate = cfg.get("plateau_min_accuracy", 0.80)
        min_r = cfg.get("early_stopping_min_rounds", 30)
        line = (
            f"{idx:<3} | {cid:<10} | {cfg['ds_name'].upper():<5} | {method:<20} | "
            f"{setting_str:<10} | {cfg['alpha']:<5g} | {cr_str:<7} | {gate:<5.0%} | {min_r:<4} | {cfg['seed']:<4}"
        )
        print(line)
    print(f"{sep}\n")


def run_all(configs: list[dict]) -> None:
    """
    Invokes base multi-GPU scheduler to dispatch jobs across GPUs:
    All 6 slots are saturated at all times:
      - 4 QNLI runs start immediately on lanes 1-4.
      - 2 MNLI runs immediately start on lanes 5-6.
      - As soon as any lane finishes, it immediately pulls the next pending run.
    """
    runnable = [config for config in configs if not is_dropped(config)]
    RUN_CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with SCHEDULER_LOCK_FILE.open("w", encoding="utf-8") as lock_file:
        if fcntl is not None:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SystemExit(
                    "Another Non-IID scheduler is already active. "
                    "Use the status command instead of starting a second one."
                ) from error
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()

        print_plan(configs)
        print(f"Launching Non-IID scheduler across {JOBS_PER_GPU} lanes/GPU (6 parallel slots)...")
        print("Continuous multi-stage saturation: all 6 parallel lanes active from round 1 until queue drains.\n")
        base.run_main_phase(runnable)
        print("\n=== ALL NON-IID EXPERIMENTS COMPLETED ===")


def list_runs() -> None:
    """Lists currently active runs from the registry."""
    active_records = []
    for p in ACTIVE_RUNS_DIR.glob("*.json"):
        if p.is_file():
            try:
                active_records.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    if not active_records:
        print("No active non-IID experiments running.")
        return
    print(f"\nActive Non-IID Experiments ({len(active_records)}):")
    for rec in active_records:
        cid = rec.get("control_id", "unknown")
        pid = rec.get("pid", "unknown")
        dev = rec.get("device", "unknown")
        rid = rec.get("run_id", "unknown")
        elapsed = (time.time() - rec.get("started_at", time.time())) / 60.0
        latest_metric = base.read_latest_metric_line(rec.get("stdout_path", ""))
        print(f"  [{cid}] PID={pid} DEV={dev} Elapsed={elapsed:.1f}m")
        print(f"        Run:    {rid}")
        print(f"        Latest: {latest_metric}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Non-IID Closed-Loop FDA-RoLoRA & Baseline Runner")
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
        help=argparse.SUPPRESS,  # internal child worker flag
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
        "--datasets",
        nargs="+",
        default=["qnli", "mnli"],
        help="Datasets to evaluate (default: qnli mnli)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[ROLORA_METHOD, PURE_METHOD],
        help=f"Methods to evaluate (default: {ROLORA_METHOD} {PURE_METHOD})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=DEFAULT_LOCAL_EPOCHS,
        help="Local epochs for RoLoRA baseline (default: 10 20)",
    )
    parser.add_argument(
        "--fda-configs",
        nargs="+",
        default=["20:1.0", "40:1.0"],
        help="FDA Pure settings as Emax:ratio pairs (default: 20:1.0 40:1.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Dirichlet distribution alpha parameter (default: 0.5)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        nargs="+",
        default=_as_list(DEFAULT_CLIENT_COUNTS),
        help="Client scales to test (default: 20)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Total federated rounds (default: 50)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=_as_list(DEFAULT_SEEDS),
        help="Seeds to run (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_config:
        execute_run(args.run_config)
        return

    configure_base_runner()

    selected_datasets = list(DATASET_COST_ORDER) if "all" in args.datasets else args.datasets

    parsed_fda_configs = []
    for item in args.fda_configs:
        emax_str, ratio_str = item.split(":")
        parsed_fda_configs.append((int(emax_str), float(ratio_str)))

    configs = build_non_iid_configs(
        datasets=selected_datasets,
        client_counts=args.clients,
        seeds=args.seeds,
        alpha=args.alpha,
        rounds=args.rounds,
        methods=args.methods,
        local_epochs=args.epochs,
        fda_configs=parsed_fda_configs,
        e_min=DEFAULT_E_MIN,
    )

    if args.group == "plan":
        print_plan(configs)
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
