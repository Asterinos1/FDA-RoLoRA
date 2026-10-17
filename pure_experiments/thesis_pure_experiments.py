#!/usr/bin/env python3
"""
Master Orchestrator for Pure FDA-RoLoRA Dual-Track Experiments.

Experimental Protocol:
  1. Starts with Matrix B first (Round 1: B, Round 2: A, ...)
  2. Pure un-discounted drift variance (NODISC, gamma=1.0)
  3. No e_min clamp (e_min = 1)
  4. Discard first-round guard (natural cold-start bootstrap from initial Theta = 0.0)
  5. Controller setpoints: k = 1.0 (target E_max) vs. k = 0.5 (target E_max / 2)
  6. Datasets: QNLI and MNLI across both C=20 (R=50) and C=50 (R=30)
  7. Multi-GPU parallel scheduler (6 parallel slots on Polytechnix, 3 jobs/GPU)
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

try:
    import fcntl
except ImportError:
    fcntl = None

# Resolve REPO_ROOT: support running inside pure_experiments/ or repo root
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


def _as_list(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


PURE_METHOD = "FDA-RoLoRA-DualTrack-Pure"
STAGE = "pure-calib"
DEFAULT_SEEDS = [42, 43]
DEFAULT_CLIENT_COUNTS = [20, 50]
PARTITION_ALPHA = 1000.0
DEFAULT_E_MIN = 1
DEFAULT_E_MAX = 20
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")

# Exact production early stopping gates
EARLY_STOPPING_MIN_ROUNDS = 6
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.005
EARLY_STOPPING_WINDOW = 3
EARLY_STOPPING_GATES = {
    "qqp": {20: 0.72, 50: 0.72},
    "qnli": {20: 0.80, 50: 0.80},
    "mnli": {20: 0.60, 50: 0.55},
    "sst2": {20: 0.85, 50: 0.85},
}

DATASET_COST_ORDER = ("qqp", "qnli", "mnli", "sst2")
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
LOG_DIR = RESULTS_DIR / "logs" / "thesis_pure"
RUN_CONTROL_DIR = RESULTS_DIR / "run_control_pure"
ACTIVE_RUNS_DIR = RUN_CONTROL_DIR / "active"
STOP_REQUESTS_DIR = RUN_CONTROL_DIR / "stop_requests"
DROPPED_RUNS_DIR = RUN_CONTROL_DIR / "dropped"
OPERATOR_EVENTS_FILE = RUN_CONTROL_DIR / "operator_events.jsonl"
SCHEDULER_LOCK_FILE = RUN_CONTROL_DIR / "scheduler.lock"

JOBS_PER_GPU = max(1, int(os.environ.get("THESIS_PURE_JOBS_PER_GPU", "6")))
BATCH64_JOBS_PER_GPU = max(
    1,
    min(JOBS_PER_GPU, int(os.environ.get("THESIS_PURE_BATCH64_JOBS_PER_GPU", "2"))),
)


def configure_base_runner() -> None:
    """Wire custom paths, scheduler settings, and method registration into base runner."""
    base.METHODS[PURE_METHOD] = {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
    }
    base.MAIN_METHODS = [PURE_METHOD]
    base.MAIN_DATASET_ORDER = DATASET_COST_ORDER
    base.MAIN_CLIENT_PRIORITY = (20, 50)
    base.MAIN_INTERRUPTED_SNAPSHOT_RUN_IDS = frozenset()
    base.MAIN_JOBS_PER_GPU = JOBS_PER_GPU
    base.MAIN_BATCH64_JOBS_PER_GPU = BATCH64_JOBS_PER_GPU
    base.LOG_DIR = LOG_DIR
    base.RUN_CONTROL_DIR = RUN_CONTROL_DIR
    base.ACTIVE_RUNS_DIR = ACTIVE_RUNS_DIR
    base.STOP_REQUESTS_DIR = STOP_REQUESTS_DIR
    base.OPERATOR_EVENTS_FILE = OPERATOR_EVENTS_FILE
    base.main_config_priority = pure_config_priority
    base.wait_for_registered_run = wait_for_registered_pure_run
    base.run_config_on_device = run_pure_config_on_device


def pure_config_priority(config: dict) -> tuple:
    # 1) Stage 1 priority: All batch-32 tasks (batch_tier=0) strictly before batch-64 tasks (batch_tier=1)
    is_batch64 = int(config.get("batch_size", 32)) >= 64
    batch_tier = 1 if is_batch64 else 0

    # 2) Within tier, follow dataset cost order: qqp -> qnli -> mnli -> sst2
    ds_order = DATASET_COST_ORDER.index(config["ds_name"]) if config["ds_name"] in DATASET_COST_ORDER else 99

    # 3) C=20 before C=50 (faster runs complete milestones earlier)
    client_order = int(config["num_clients"])

    # 4) Ratio 1.0 before 0.5
    ratio_val = float(config.get("controller_ratio", 1.0))
    ratio_order = 0 if ratio_val == 1.0 else 1

    # 5) Seed 42 before 43
    seed_order = int(config["seed"])

    return (batch_tier, ds_order, client_order, ratio_order, seed_order)


def build_pure_configs(
    datasets: list[str],
    client_counts: list[int],
    seeds: list[int],
    ratios: list[float],
    e_min: int = DEFAULT_E_MIN,
    e_max: int = DEFAULT_E_MAX,
) -> list[dict]:
    """Generate isolated pure configurations with distinct EXTRA identifier."""
    configs = []
    for dataset in datasets:
        settings = DATASET_SETTINGS[dataset]
        for num_clients in client_counts:
            for seed in seeds:
                rounds = settings["rounds"][num_clients]
                gate = EARLY_STOPPING_GATES[dataset][num_clients]

                for ratio in ratios:
                    config = base.make_config(
                        stage=STAGE,
                        dataset=dataset,
                        method="FDA-RoLoRA-DualTrack",
                        seed=seed,
                        num_clients=num_clients,
                        total_rounds=rounds,
                        alpha=PARTITION_ALPHA,
                        learning_rate=settings["learning_rate"],
                        rank=LORA_RANK,
                        layers=settings["layers"],
                        batch_size=settings["batch_size"],
                        clients_per_round=num_clients,
                        local_epochs=1,
                        lora_alpha=LORA_ALPHA,
                        target_modules=TARGET_MODULES,
                    )
                    config["METHOD_NAME"] = PURE_METHOD

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
                            "stagnation_round": 10_000,
                            "stagnation_tolerance": 0.0,
                            "start_matrix": "B",
                            "first_round_guard": False,
                        }
                    )

                    ratio_tag = f"CRATIO{ratio:g}"
                    extra_id = (
                        f"THESIS-RBASE-PURE-DS-{dataset}-C{num_clients}-R{rounds}"
                        f"-M-fda-pure-LR{settings['learning_rate']:g}-RK{LORA_RANK}"
                        f"-E{e_min}TO{e_max}-{ratio_tag}-NODISC-BSTART-NOGUARD-S{seed}"
                        f"-ES{gate:g}-MR{EARLY_STOPPING_MIN_ROUNDS}"
                        f"-P{EARLY_STOPPING_PATIENCE}"
                        f"-D{EARLY_STOPPING_MIN_DELTA:g}"
                        f"-W{EARLY_STOPPING_WINDOW}"
                    )
                    config["extras"] = extra_id
                    configs.append(config)

    return sorted(configs, key=pure_config_priority)


def dropped_marker_path(run_id: str) -> Path:
    return DROPPED_RUNS_DIR / f"{base.get_control_id(run_id)}.json"


def is_dropped(config: dict) -> bool:
    return dropped_marker_path(config["extras"]).exists()


def is_pure_completed(config: dict) -> bool:
    pattern = str(METRICS_DIR / f"{config['extras']}_*.csv")
    for path in glob.glob(pattern):
        try:
            import pandas as pd
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not frame.empty and "round" in frame.columns:
            return True
    return False


def run_pure_config_on_device(config: dict, device: str) -> bool:
    """Spawns this runner script itself with --run-config for child worker isolation."""
    run_id = config["extras"]
    registry_path = None
    request_path = None
    try:
        if is_pure_completed(config):
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
        pure_dir = str(REPO_ROOT / "pure_experiments")
        extra_paths = [str(REPO_ROOT), pure_dir]
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


def wait_for_registered_pure_run(
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
    if is_pure_completed(config):
        final_metric = base.read_latest_metric_line(record["stdout_path"])
        print(f"ADOPTED DONE [{control_id}] {run_id}; {final_metric}")
        return True
    print(f"ADOPTED ERROR [{control_id}] {run_id}: worker exited without a readable metrics CSV")
    return False


def resolve_pure_optimizer():
    """Dynamically resolve fda_rolora_dual_track_pure_opt."""
    try:
        from pure_experiments.fda_rolora_pure import fda_rolora_dual_track_pure_opt
        return fda_rolora_dual_track_pure_opt
    except ImportError:
        pass
    try:
        from fda_rolora_pure import fda_rolora_dual_track_pure_opt
        return fda_rolora_dual_track_pure_opt
    except ImportError:
        pass

    candidates = [
        REPO_ROOT / "pure_experiments" / "fda_rolora_pure.py",
        CURRENT_DIR / "fda_rolora_pure.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("fda_pure_module", str(candidate))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "fda_rolora_dual_track_pure_opt"):
                    return getattr(mod, "fda_rolora_dual_track_pure_opt")

    # Built-in standalone fallback
    return _builtin_fda_rolora_dual_track_pure_opt


def _builtin_fda_rolora_dual_track_pure_opt(hyperparams: dict):
    """Built-in standalone fallback for Pure FDA-RoLoRA dual-track."""
    import gc
    import math
    import time
    import torch
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import AutoModelForSequenceClassification
    from fdaopt import utils
    from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
    from fdaopt.metrics.mentrics_handler import MetricsHandler
    from fdaopt.models.ops import (
        compute_client_drifts,
        compute_pseudo_gradients,
        set_gradients,
        copy_parameters,
        get_updated_client_parameters,
        variance,
        compute_metrics,
        check_dynamic_early_stopping,
    )
    from fdaopt.training.fed_train import client_round_train
    from fdaopt.training.optimizers import server_client_optimizers
    from fdaopt.training import fda_opt

    def get_save_device() -> str:
        if utils.DEVICE_RAM_PROGRAM in ("performance", "moderate"):
            return utils.DEVICE
        return "cpu"

    def toggle_rolora_matrices(model, active_matrix: str):
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = f"lora_{active_matrix}" in name

    e_min = int(hyperparams.get("e_min", 1))
    e_max = int(hyperparams.get("max_epochs", 20))
    controller_ratio = float(hyperparams.get("controller_ratio", 1.0))

    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams["ds_path"],
        hyperparams["ds_name"],
        hyperparams["checkpoint"],
        hyperparams["num_clients"],
        hyperparams["alpha"],
        hyperparams["batch_size"],
        hyperparams.get("num_workers", 4),
        hyperparams.get("pin_memory", True),
    )

    if hyperparams.get("ds_name") == "mnist":
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(
            rank=hyperparams.get("lora_r", 16),
            linear=hyperparams.get("linear", False)
        ).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams["checkpoint"], num_labels=hyperparams["num_labels"]
        )
        lora_kwargs = {
            "task_type": TaskType.SEQ_CLS,
            "inference_mode": False,
            "r": hyperparams.get("lora_r", 8),
            "lora_alpha": hyperparams.get("lora_alpha", 16),
            "lora_dropout": 0.1,
            "target_modules": hyperparams.get("target_modules", None),
        }
        target_layers = hyperparams.get("target_layer_indices", hyperparams.get("layers"))
        if target_layers is not None:
            lora_kwargs["layers_to_transform"] = target_layers
            lora_kwargs["layers_pattern"] = "layer"
        peft_config = LoraConfig(**lora_kwargs)
        model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
        if not hyperparams.get("train_head", True):
            for name, param in model.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False

    params_A = [p for n, p in model.named_parameters() if "lora_A" in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad]
    num_lora = len(params_A)
    params_A = params_A + params_other
    params_B = params_B + params_other

    server_opt_A, _ = server_client_optimizers(params_A, hyperparams, num_lora_params=num_lora)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams, num_lora_params=num_lora)

    client_sampler = ClientSampler(list(range(hyperparams["num_clients"])), hyperparams["clients_per_round"])
    client_dict_A = {
        c_id: [param.detach().clone().to(get_save_device()) for param in params_A]
        for c_id in range(hyperparams["num_clients"])
    }
    client_dict_B = {
        c_id: [param.detach().clone().to(get_save_device()) for param in params_B]
        for c_id in range(hyperparams["num_clients"])
    }

    metrics_handler = MetricsHandler(hyperparams)
    theta_A = float(hyperparams.get("theta", 0.0))
    theta_B = float(hyperparams.get("theta", 0.0))

    for r in range(hyperparams["total_rounds"]):
        round_start_time = time.time()
        active_matrix = "B" if r % 2 == 0 else "A"
        toggle_rolora_matrices(model, active_matrix)

        active_params = params_A if active_matrix == "A" else params_B
        active_server_opt = server_opt_A if active_matrix == "A" else server_opt_B
        active_client_dict = client_dict_A if active_matrix == "A" else client_dict_B
        current_theta = theta_A if active_matrix == "A" else theta_B

        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(
            active_client_dict, sampled_clients, round_start_active_params
        )
        round_steps = hyperparams["local_epochs"] * fed_ds.epoch_steps(sampled_clients)

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
                loss, acc = client_round_train(
                    model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps
                )
                training_loss += loss
                training_accuracy += acc

            if round_epochs >= e_min:
                sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)

                if var_subspace > current_theta or round_epochs >= e_max:
                    break
            elif round_epochs >= e_max:
                sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                break

        train_end = time.time()
        agg_start = time.time()

        training_loss = training_loss / round_epochs
        training_accuracy = training_accuracy / round_epochs
        pseudo_gradients = compute_pseudo_gradients(client_drifts)

        with torch.no_grad():
            update_l2_norm = math.sqrt(sum(torch.sum(g**2).item() for g in pseudo_gradients))

        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        set_gradients(active_params, pseudo_gradients)
        active_server_opt.step()
        active_server_opt.zero_grad()

        agg_end = time.time()
        round_end_time = time.time()

        target_epochs = controller_ratio * e_max
        next_theta = (target_epochs / round_epochs) * var_subspace
        if active_matrix == "A":
            theta_A = next_theta
        else:
            theta_B = next_theta

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
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * round_steps,
            "variance": var_subspace,
            "effective_variance": var_subspace,
            "avg_norm_sq_drifts": avg_norm_sq_drifts,
            "norm_sq_avg_drift": norm_sq_avg_drift,
            "theta_phase": current_theta,
            "controller_ratio": controller_ratio,
            "enable_b_discount": False,
            "b_discount_factor": 1.0,
        } | compute_metrics(model, hyperparams["ds_path"], hyperparams["ds_name"], test_ds)

        metrics["training_loss"] = training_loss / hyperparams["clients_per_round"]
        metrics["training_accuracy"] = training_accuracy / hyperparams["clients_per_round"]
        metrics_handler.append_round_metrics(metrics)
        metrics_handler.save_metrics()

        print(
            f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished "
            f"(Matrix {active_matrix}, Epochs: {round_epochs}, Ratio: {controller_ratio}, Pure NODISC)."
        )
        gc.collect()

        if hyperparams.get("early_stopping"):
            early_stop_fn = getattr(fda_opt, "check_dynamic_early_stopping", check_dynamic_early_stopping)
            if early_stop_fn(hyperparams, metrics_handler.round_metrics):
                print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
                break

    metrics_handler.save_metrics()


def execute_run(config_json: str) -> None:
    """Child worker execution target."""
    import torch
    from fdaopt import utils
    from fdaopt.training import fda_opt, fed_opt

    opt_fn = resolve_pure_optimizer()

    # Register into method tables
    fda_opt.fda_rolora_dual_track_pure_opt = opt_fn
    fed_opt.fda_rolora_dual_track_pure_opt = opt_fn

    config = json.loads(config_json)
    device = config.pop("TARGET_DEVICE", "cuda:0")
    base.install_runtime_patches(config)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    utils.DEVICE_RAM_PROGRAM = "performance"
    utils.set_device(device)
    utils.set_seed(config["seed"])

    print(f"[{time.strftime('%H:%M:%S')}] Starting {config['extras']} on {device}...")
    opt_fn(config)
    print(f"[{time.strftime('%H:%M:%S')}] Execution finished for {config['extras']}.")


def print_plan(configs: list[dict]) -> None:
    sep = "=" * 115
    print(f"\n{sep}")
    print(f"PURE FDA-RoLoRA EXPERIMENT PLAN (Total Runs: {len(configs)})")
    print(f"Features: Matrix B first | e_min=1 | Zero first-round guard | NODISC (gamma=1.0)")
    print(f"Staged Scheduling: Batch-32 tasks (QQP, QNLI, MNLI) execute first across {JOBS_PER_GPU} lanes/GPU.")
    print(f"Draining / Backfilling: Batch-64 (SST-2) queues at the end (capped at {BATCH64_JOBS_PER_GPU} jobs/GPU) to prevent OOM.")
    print(sep)
    header = f"{'#':<3} | {'Control ID':<10} | {'DS':<5} | {'BS':<3} | {'C':<3} | {'R':<3} | {'Ratio (k)':<9} | {'E-Range':<7} | {'Start':<5} | {'LR':<4} | {'Gate':<5} | {'Seed':<4}"
    print(header)
    print("-" * len(header))
    current_tier = None
    for idx, cfg in enumerate(configs, 1):
        tier = "BATCH-64" if int(cfg.get("batch_size", 32)) >= 64 else "BATCH-32"
        if current_tier is not None and tier != current_tier:
            print("-" * len(header))
        current_tier = tier
        cid = base.get_control_id(cfg["extras"])
        ratio_str = f"k={cfg['controller_ratio']}"
        e_range = f"E{cfg['e_min']}..{cfg['max_epochs']}"
        start_mat = cfg.get("start_matrix", "B")
        line = (
            f"{idx:<3} | {cid:<10} | {cfg['ds_name'].upper():<5} | {cfg['batch_size']:<3} | {cfg['num_clients']:<3} | "
            f"{cfg['total_rounds']:<3} | {ratio_str:<9} | {e_range:<7} | {start_mat:<5} | "
            f"{cfg['client_opt_lr']:<4.2g} | {cfg['early_stopping_start_accuracy']:<5.0%} | {cfg['seed']:<4}"
        )
        print(line)
    print(f"{sep}\n")


def run_all(configs: list[dict]) -> None:
    """Invokes base scheduler to dispatch jobs across GPUs:
    Batch-32 tasks (QQP, QNLI, MNLI) are prioritized first across dynamic lanes per GPU.
    Batch-64 tasks (SST-2) are held at the end and dynamically backfill into freed lanes,
    strictly abiding by BATCH64_JOBS_PER_GPU cap to prevent CUDA OOM.
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
                    "Another Pure FDA-RoLoRA scheduler is already active. "
                    "Use the status command instead of starting a second one."
                ) from error
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()

        print_plan(configs)
        print(f"Launching scheduler across {JOBS_PER_GPU} dynamic lanes/GPU (max concurrent: {len(base.DEVICES) * JOBS_PER_GPU})...")
        print(f"Batch-64 safety cap: max {BATCH64_JOBS_PER_GPU} SST-2 runs per GPU concurrently.\n")
        base.run_main_phase(runnable)
        print("\n=== ALL PURE FDA-RoLoRA EXPERIMENTS COMPLETED ===")


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
        print("No active pure experiments running.")
        return
    print(f"\nActive Pure Experiments ({len(active_records)}):")
    for rec in active_records:
        cid = rec.get("control_id", "unknown")
        pid = rec.get("pid", "unknown")
        dev = rec.get("device", "unknown")
        rid = rec.get("run_id", "unknown")
        elapsed = (time.time() - rec.get("started_at", time.time())) / 60.0
        print(f"  [{cid}] PID={pid} DEV={dev} Elapsed={elapsed:.1f}m -> {rid}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Pure FDA-RoLoRA Dual-Track Runner")
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
        default=["qqp", "qnli", "mnli", "sst2"],
        help="Datasets to evaluate (default: qqp qnli mnli sst2)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        nargs="+",
        default=_as_list(DEFAULT_CLIENT_COUNTS),
        help="Client scales to test (default: 20 50)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=_as_list(DEFAULT_SEEDS),
        help="Seeds to run (default: 42)",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[1.0, 0.5],
        help="Controller ratios k (default: 1.0 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_config:
        execute_run(args.run_config)
        return

    configure_base_runner()

    selected_datasets = list(DATASET_COST_ORDER) if "all" in args.datasets else args.datasets

    configs = build_pure_configs(
        datasets=selected_datasets,
        client_counts=args.clients,
        seeds=args.seeds,
        ratios=args.ratios,
        e_min=DEFAULT_E_MIN,
        e_max=DEFAULT_E_MAX,
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
