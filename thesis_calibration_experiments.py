"""Run the FDA-RoLoRA Dual-Track Calibrated Controller Experiments.

Evaluates:
1. Controller setpoint calibrations: ratio in {1.0, 0.75, 0.5} where
   proposed_target = ((controller_ratio * E_max) / round_epochs) * var_subspace.
   - ratio=1.0: Targets E_max directly (pushes clients toward maximum compute).
   - ratio=0.75: Targets 0.75 * E_max.
   - ratio=0.5: Standard midpoint behavior (E_max / 2).
2. Matrix B variance discount factor:
   - Optional toggle `--enable-discount` (with `--discount-factor 3.0`).
   - When enabled, discounts Matrix B's variance: effective_var = var_subspace / discount_factor.
   - Dormant by default (`enable_b_discount=False`).

Applies exact GLUE early stopping accuracy gates, 6-round minimum, 3-round patience,
and multi-GPU parallel execution identical to thesis_sma_experiments.py.
"""
from __future__ import annotations
import argparse
try:
    import fcntl
except ImportError:
    fcntl = None
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure root repository is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
for candidate in [Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent]:
    if (candidate / "thesis_experiments.py").exists() and (candidate / "fdaopt").is_dir():
        REPO_ROOT = candidate
        break

for p in [str(REPO_ROOT), str(SCRIPT_DIR), str(REPO_ROOT / "fixed_calibration")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import thesis_experiments as base
except Exception as error:
    raise SystemExit(
        f"Failed to import thesis_experiments ({error}). "
        f"Ensure dependencies (like pandas, torch) are installed and "
        "thesis_calibration_experiments.py is placed inside ARIS-FDA-LoRA/fixed_calibration/."
    ) from error

def _as_list(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]

CALIB_METHOD = "FDA-RoLoRA-DualTrack-Calibrated"
STAGE = "fixed-calib"
DEFAULT_SEEDS = [42]  # Use [42] or (42,) for single seed, or [42, 43] for multiple
DEFAULT_CLIENT_COUNTS = [50, 20]
PARTITION_ALPHA = 1000.0
DEFAULT_E_MIN = 3
DEFAULT_E_MAX = 30
E_MIN = DEFAULT_E_MIN
E_MAX = DEFAULT_E_MAX
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")

# Exact early stopping gates from production protocol
EARLY_STOPPING_MIN_ROUNDS = 6
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.005
EARLY_STOPPING_WINDOW = 3
EARLY_STOPPING_GATES = {
    "rte": {20: 0.60, 50: 0.60},
    "mrpc": {20: 0.70, 50: 0.70},
    "qnli": {20: 0.80, 50: 0.80},
    "mnli": {20: 0.60, 50: 0.55},
    "qqp": {20: 0.72, 50: 0.75},
    "sst2": {20: 0.85, 50: 0.85},
}

DATASET_COST_ORDER = ("qnli", "mnli", "qqp", "sst2")
DATASET_SETTINGS = {
    "rte": {
        "batch_size": 32,
        "layers": list(range(8, 12)),
        "rounds": {20: 30, 50: 12},
        "learning_rate": 0.05,
    },
    "mrpc": {
        "batch_size": 32,
        "layers": list(range(8, 12)),
        "rounds": {20: 30, 50: 12},
        "learning_rate": 0.05,
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
    "qqp": {
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
LOG_DIR = RESULTS_DIR / "logs" / "thesis_fixed_calibration"
RUN_CONTROL_DIR = RESULTS_DIR / "run_control_fixed_calibration"
ACTIVE_RUNS_DIR = RUN_CONTROL_DIR / "active"
STOP_REQUESTS_DIR = RUN_CONTROL_DIR / "stop_requests"
DROPPED_RUNS_DIR = RUN_CONTROL_DIR / "dropped"
OPERATOR_EVENTS_FILE = RUN_CONTROL_DIR / "operator_events.jsonl"
SCHEDULER_LOCK_FILE = RUN_CONTROL_DIR / "scheduler.lock"

JOBS_PER_GPU = max(1, int(os.environ.get("THESIS_CALIB_JOBS_PER_GPU", "3")))
BATCH64_JOBS_PER_GPU = max(
    1,
    min(JOBS_PER_GPU, int(os.environ.get("THESIS_CALIB_BATCH64_JOBS_PER_GPU", "2"))),
)


def configure_base_runner() -> None:
    """Wire up custom paths, scheduler settings, and method registration into base runner."""
    base.METHODS[CALIB_METHOD] = {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
    }
    base.MAIN_METHODS = [CALIB_METHOD]
    base.MAIN_DATASET_ORDER = DATASET_COST_ORDER
    base.MAIN_CLIENT_PRIORITY = (50, 20)
    base.MAIN_INTERRUPTED_SNAPSHOT_RUN_IDS = frozenset()
    base.MAIN_JOBS_PER_GPU = JOBS_PER_GPU
    base.MAIN_BATCH64_JOBS_PER_GPU = BATCH64_JOBS_PER_GPU
    base.LOG_DIR = LOG_DIR
    base.RUN_CONTROL_DIR = RUN_CONTROL_DIR
    base.ACTIVE_RUNS_DIR = ACTIVE_RUNS_DIR
    base.STOP_REQUESTS_DIR = STOP_REQUESTS_DIR
    base.OPERATOR_EVENTS_FILE = OPERATOR_EVENTS_FILE
    base.main_config_priority = calib_config_priority
    base.wait_for_registered_run = wait_for_registered_calib_run
    base.run_config_on_device = run_calib_config_on_device


def calib_config_priority(config: dict) -> tuple:
    # 1) Ratio 1.0 before 0.75
    ratio_val = float(config.get("controller_ratio", 0.5))
    ratio_order = 0 if ratio_val == 1.0 else (1 if ratio_val == 0.75 else 2)
    # 2) NoDisc (0) before B-Disc (1)
    disc_order = 1 if config.get("enable_b_discount", False) else 0

    return (
        int(config["total_rounds"]),
        DATASET_COST_ORDER.index(config["ds_name"]),
        int(config["num_clients"]),
        disc_order,
        ratio_order,
        int(config["seed"]),
    )


def build_calib_configs(
    datasets: list[str],
    client_counts: list[int],
    seeds: list[int],
    ratios: list[float],
    enable_discount: bool,
    discount_factor: float,
    e_min: int = DEFAULT_E_MIN,
    e_max: int = DEFAULT_E_MAX,
    b_discount_only: bool = False,
) -> list[dict]:
    """Generate isolated calibration configurations with distinct EXTRA identifier."""
    configs = []
    for dataset in datasets:
        settings = DATASET_SETTINGS[dataset]
        for num_clients in client_counts:
            for seed in seeds:
                rounds = settings["rounds"][num_clients]
                gate = EARLY_STOPPING_GATES[dataset][num_clients]

                for ratio in ratios:
                    # Test both discount states if requested, or the single setting
                    if b_discount_only:
                        discount_modes = [True]
                    elif enable_discount:
                        discount_modes = [False, True]
                    else:
                        discount_modes = [False]
                    for disc_mode in discount_modes:
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
                        config["METHOD_NAME"] = CALIB_METHOD

                        # Inject custom calibration parameters
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
                                "enable_b_discount": disc_mode,
                                "b_discount_factor": discount_factor if disc_mode else 1.0,
                                "early_stopping_start_accuracy": gate,
                                "plateau_min_accuracy": gate,
                                "early_stopping_min_rounds": EARLY_STOPPING_MIN_ROUNDS,
                                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                                "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                                "early_stopping_window": EARLY_STOPPING_WINDOW,
                                "stagnation_round": 10_000,
                                "stagnation_tolerance": 0.0,
                            }
                        )

                        # Construct unambiguous distinct run identifier
                        ratio_tag = f"CRATIO{ratio:g}"
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
                        config["extras"] = extra_id
                        configs.append(config)

    return sorted(configs, key=calib_config_priority)


def dropped_marker_path(run_id: str) -> Path:
    return DROPPED_RUNS_DIR / f"{base.get_control_id(run_id)}.json"


def is_dropped(config: dict) -> bool:
    return dropped_marker_path(config["extras"]).exists()


def is_calib_completed(config: dict) -> bool:
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


def run_calib_config_on_device(config: dict, device: str) -> bool:
    """Spawns this runner script itself with --run-config for child worker isolation."""
    run_id = config["extras"]
    registry_path = None
    request_path = None
    try:
        if is_calib_completed(config):
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
        calib_dir = str(REPO_ROOT / "fixed_calibration")
        extra_paths = [str(REPO_ROOT), calib_dir]
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


def wait_for_registered_calib_run(
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
    if is_calib_completed(config):
        final_metric = base.read_latest_metric_line(record["stdout_path"])
        print(f"ADOPTED DONE [{control_id}] {run_id}; {final_metric}")
        return True
    print(f"ADOPTED ERROR [{control_id}] {run_id}: worker exited without a readable metrics CSV")
    return False


def _builtin_fda_rolora_dual_track_calibrated_opt(hyperparams: dict):
    """Built-in standalone fallback for FDA-RoLoRA calibrated controller."""
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

    try:
        from fdaopt.training.fda_opt import get_save_device, toggle_rolora_matrices
    except ImportError:
        def get_save_device() -> str:
            if utils.DEVICE_RAM_PROGRAM in ("performance", "moderate"):
                return utils.DEVICE
            return "cpu"

        def toggle_rolora_matrices(model, active_matrix: str):
            for name, param in model.named_parameters():
                if "lora_" in name:
                    if f"lora_{active_matrix}" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

    e_min = hyperparams.get("e_min", 3)
    e_max = hyperparams.get("max_epochs", 20)
    controller_ratio = float(hyperparams.get("controller_ratio", 0.5))
    enable_b_discount = bool(hyperparams.get("enable_b_discount", False))
    b_discount_factor = float(hyperparams.get("b_discount_factor", 3.0))

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

        model = ToyModel(rank=hyperparams.get("lora_r", 8)).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams["checkpoint"], num_labels=hyperparams["num_labels"]
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=hyperparams.get("lora_r", 8),
            lora_alpha=hyperparams.get("lora_alpha", 16),
            lora_dropout=0.1,
            target_modules=hyperparams.get("target_modules", None),
        )
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
        c_id: [p.detach().clone().to(get_save_device()) for p in params_A]
        for c_id in range(hyperparams["num_clients"])
    }
    client_dict_B = {
        c_id: [p.detach().clone().to(get_save_device()) for p in params_B]
        for c_id in range(hyperparams["num_clients"])
    }

    metrics_handler = MetricsHandler(hyperparams)
    theta_A = hyperparams.get("theta", 0.0)
    theta_B = hyperparams.get("theta", 0.0)
    first_round_A = True
    first_round_B = True

    for r in range(hyperparams["total_rounds"]):
        round_start_time = time.time()
        active_matrix = "A" if r % 2 == 0 else "B"
        toggle_rolora_matrices(model, active_matrix)

        active_params = params_A if active_matrix == "A" else params_B
        active_server_opt = server_opt_A if active_matrix == "A" else server_opt_B
        active_client_dict = client_dict_A if active_matrix == "A" else client_dict_B

        current_theta = theta_A if active_matrix == "A" else theta_B
        is_first_round = first_round_A if active_matrix == "A" else first_round_B

        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(
            active_client_dict, sampled_clients, round_start_active_params
        )
        round_steps = hyperparams["local_epochs"] * fed_ds.epoch_steps(sampled_clients)

        training_loss = 0.0
        training_accuracy = 0.0
        var_subspace = 0.0
        effective_variance = 0.0
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

                if is_first_round:
                    if active_matrix == "A":
                        first_round_A = False
                    else:
                        first_round_B = False
                    break

                if enable_b_discount and active_matrix == "B":
                    effective_variance = var_subspace / b_discount_factor
                else:
                    effective_variance = var_subspace

                if effective_variance > current_theta or round_epochs >= e_max:
                    break
            elif round_epochs >= e_max:
                sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                effective_variance = (
                    var_subspace / b_discount_factor
                    if (enable_b_discount and active_matrix == "B")
                    else var_subspace
                )
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
            "method": "FDA-RoLoRA-DualTrack-Calibrated",
            "round": r + 1,
            "active_matrix": active_matrix,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * round_steps,
            "variance": var_subspace,
            "effective_variance": effective_variance,
            "avg_norm_sq_drifts": avg_norm_sq_drifts,
            "norm_sq_avg_drift": norm_sq_avg_drift,
            "theta_phase": current_theta,
            "controller_ratio": controller_ratio,
            "enable_b_discount": enable_b_discount,
            "b_discount_factor": b_discount_factor,
        } | compute_metrics(model, hyperparams["ds_path"], hyperparams["ds_name"], test_ds)

        metrics["training_loss"] = training_loss / hyperparams["clients_per_round"]
        metrics["training_accuracy"] = training_accuracy / hyperparams["clients_per_round"]
        metrics_handler.append_round_metrics(metrics)

        print(
            f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished "
            f"(Matrix {active_matrix}, Epochs: {round_epochs}, Ratio: {controller_ratio}, B-Disc: {enable_b_discount})."
        )
        gc.collect()

        if hyperparams.get("early_stopping") and check_dynamic_early_stopping(
            hyperparams, metrics_handler.round_metrics
        ):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break

    metrics_handler.save_metrics()


def resolve_calibrated_optimizer():
    """Dynamically resolve fda_rolora_dual_track_calibrated_opt under any setup."""
    # 1. Directly in fdaopt.training.fda_opt
    try:
        from fdaopt.training.fda_opt import fda_rolora_dual_track_calibrated_opt

        return fda_rolora_dual_track_calibrated_opt
    except (ImportError, AttributeError):
        pass

    # 2. In fixed_calibration package
    try:
        from fixed_calibration.fda_rolora_dual_track_calibrated import (
            fda_rolora_dual_track_calibrated_opt,
        )

        return fda_rolora_dual_track_calibrated_opt
    except (ImportError, ModuleNotFoundError):
        pass

    # 3. Flat directory import
    try:
        from fda_rolora_dual_track_calibrated import (
            fda_rolora_dual_track_calibrated_opt,
        )

        return fda_rolora_dual_track_calibrated_opt
    except (ImportError, ModuleNotFoundError):
        pass

    # 4. Search file candidates via importlib
    import importlib.util

    candidates = [
        REPO_ROOT / "fixed_calibration" / "fda_rolora_dual_track_calibrated.py",
        REPO_ROOT / "fda_rolora_dual_track_calibrated.py",
        Path(__file__).resolve().parent / "fixed_calibration" / "fda_rolora_dual_track_calibrated.py",
        Path(__file__).resolve().parent / "fda_rolora_dual_track_calibrated.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("fda_calib_module", str(candidate))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "fda_rolora_dual_track_calibrated_opt"):
                    return getattr(mod, "fda_rolora_dual_track_calibrated_opt")

    # 5. Standalone built-in implementation fallback
    return _builtin_fda_rolora_dual_track_calibrated_opt


def execute_run(config_json: str):
    """Child worker target: dispatches to fda_rolora_dual_track_calibrated_opt."""
    import torch
    from fdaopt import utils
    from fdaopt.training import fda_opt, fed_opt

    opt_fn = resolve_calibrated_optimizer()

    config = json.loads(config_json)
    device = config.pop("TARGET_DEVICE", "cuda:0")
    base.install_runtime_patches(config)
    torch.cuda.set_device(torch.device(device))
    utils.DEVICE_RAM_PROGRAM = "performance"
    utils.set_device(device)
    utils.set_seed(config["seed"])

    print(f"[{time.strftime('%H:%M:%S')}] Starting {config['extras']} on {device}...")
    opt_fn(config)
    print(f"[{time.strftime('%H:%M:%S')}] Execution finished for {config['extras']}.")


def print_plan(configs: list[dict]) -> None:
    complete = sum(is_calib_completed(config) for config in configs)
    dropped = sum(is_dropped(config) for config in configs)
    active_ids = {record["run_id"] for record in base.load_active_run_records()}
    active = sum(config["extras"] in active_ids for config in configs)

    print("\nFDA-RoLoRA DUAL-TRACK CALIBRATION STUDY PLAN")
    print("=" * 95)
    print(f"Configurations: {len(configs)} (complete={complete}, active={active}, dropped={dropped})")
    print("Method: FDA-RoLoRA-DualTrack-Calibrated")
    e_min_disp = configs[0]["e_min"] if configs else E_MIN
    e_max_disp = configs[0]["max_epochs"] if configs else E_MAX
    print(f"Window: E{e_min_disp}--E{e_max_disp}; LoRA Rank={LORA_RANK}, Alpha={LORA_ALPHA}")
    print(f"Scheduler: {JOBS_PER_GPU} jobs/GPU; batch-64 cap {BATCH64_JOBS_PER_GPU}/GPU")
    print("Early Stopping: Acc-gated with 6-round minimum, 3-round patience, 0.5% min-delta")
    print("-" * 95)
    for index, config in enumerate(configs, start=1):
        state = "DONE" if is_calib_completed(config) else "DROP" if is_dropped(config) else "TODO"
        disc_str = f"B-Disc={config['b_discount_factor']}x" if config['enable_b_discount'] else "NoDisc"
        print(
            f"{index:02d} {state:4s} "
            f"{config['ds_name'].upper():5s} C{config['num_clients']:<2d} "
            f"R{config['total_rounds']:<2d} S{config['seed']} "
            f"Ratio={config['controller_ratio']:<4g} {disc_str:<12s} "
            f"Gate={config['early_stopping_start_accuracy']:.0%}"
        )
    print("=" * 95)


def list_runs() -> None:
    records = base.load_active_run_records()
    if not records:
        print("No active FDA-Calibrated runs.")
        return
    print(f"Active FDA-Calibrated runs: {len(records)}")
    for record in records:
        elapsed = base.format_elapsed(time.time() - float(record["started_at"]))
        latest = base.read_latest_metric_line(record["stdout_path"])
        print(f"[{record['control_id']}] pid={record['pid']} gpu={record['device']} elapsed={elapsed}")
        print(f"  run:    {record['run_id']}")
        print(f"  latest: {latest}")
        print(f"  log:    {record['stdout_path']}")


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
                    "Another FDA-Calibration scheduler is already active. "
                    "Use the status command instead of starting a second one."
                ) from error
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        print_plan(configs)
        base.run_main_phase(runnable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FDA-RoLoRA Dual-Track Calibrated Experiment Orchestrator")
    parser.add_argument(
        "--group",
        default="plan",
        choices=("plan", "run", "start", "status", "stop", "drop", "dropped", "restore"),
        help="Action to perform",
    )
    parser.add_argument("--run-config", help="Internal child-worker invocation flag")
    parser.add_argument("--run-id", help="Target run ID or control ID for stop/drop commands")
    parser.add_argument("--reason", default="operator stopped run after inspecting live metrics")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_COST_ORDER),
        choices=list(DATASET_COST_ORDER) + ["all"],
        help="Datasets to evaluate (default: all 6 GLUE tasks)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        nargs="+",
        default=_as_list(DEFAULT_CLIENT_COUNTS),
        help="Client scales to test (default: 50 20)",
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
        default=[1.0],
        help="Controller ratio setpoints to evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--disable-discount",
        action="store_true",
        help="Disable Matrix B variance discount evaluation (runs only NoDisc)",
    )
    parser.add_argument(
        "--enable-nodisc",
        action="store_true",
        help="Also evaluate without Matrix B discount (default runs B-Disc only)",
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=3.0,
        help="Discount divisor for Matrix B when discount is enabled (default: 3.0)",
    )
    parser.add_argument(
        "--emax",
        type=int,
        default=DEFAULT_E_MAX,
        help=f"Maximum local epochs ceiling (default: {DEFAULT_E_MAX}, e.g. 30 for E3TO30)",
    )
    parser.add_argument(
        "--emin",
        type=int,
        default=DEFAULT_E_MIN,
        help=f"Minimum local epochs floor (default: {DEFAULT_E_MIN})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Worker dispatch mode
    if args.run_config:
        execute_run(args.run_config)
        return

    configure_base_runner()

    selected_datasets = list(DATASET_COST_ORDER) if "all" in args.datasets else args.datasets

    configs = build_calib_configs(
        datasets=selected_datasets,
        client_counts=args.clients,
        seeds=args.seeds,
        ratios=args.ratios,
        enable_discount=not args.disable_discount,
        discount_factor=args.discount_factor,
        e_min=args.emin,
        e_max=args.emax,
        b_discount_only=not args.enable_nodisc and not args.disable_discount,
    )

    if args.group == "plan":
        print_plan(configs)
    elif args.group in ("run", "start"):
        run_all(configs)
    elif args.group == "status":
        list_runs()
    elif args.group == "dropped":
        # list dropped
        pass
    elif args.group in {"stop", "drop", "restore"}:
        if not args.run_id:
            raise SystemExit(f"--run-id is required for {args.group}")
        if args.group == "stop":
            base.stop_active_run(args.run_id, args.reason)
        elif args.group == "drop":
            base.drop_active_run(args.run_id, args.reason)
        else:
            # restore
            pass


if __name__ == "__main__":
    main()
