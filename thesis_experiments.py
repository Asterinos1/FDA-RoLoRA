"""RoBERTa-base thesis experiment runner.
FINAL CROSS-SILO MAIN PROTOCOL (2026-09-04)
-------------------------------------------
The active ``main-all`` group schedules 120 configurations:
* datasets: RTE, MRPC, QNLI, MNLI, QQP, and SST-2
* cross-silo scales: C20/K20 and C50/K50 (full participation at both scales)
* methods: FedAvg-LoRA and FedAvg-RoLoRA at fixed E5 and E10;
  FDA-RoLoRA-DualTrack once with an adaptive E3--E20 window
* seeds: 42 and 43; alpha=1000; rank=4; LoRA alpha=8; query+value
* finish all C50 datasets before starting C20; large datasets use C50/R30
  and C20/R50; completed RTE/MRPC retain C50/R12 and C20/R30
* within each client phase, finish E5 baselines plus adaptive FDA across all
  datasets before starting E10 baselines; FDA is scheduled only once
* early stopping: existing-data gates for RTE/QNLI/SST-2 and a conservative
  90% gate for unrun MRPC/MNLI/QQP; minimum six rounds, three-round smoothing
  and patience, and a required 0.5 percentage-point improvement
* scheduler: target three jobs per GPU, but never more than two batch-64 jobs
  on one 24 GB A10; ``THESIS_MAIN_JOBS_PER_GPU=2`` remains a safe override
* observability/control: every evaluated round emits a ``[METRIC]`` test line;
  active child processes receive short control IDs and can be listed or
  stopped individually without terminating the scheduler
No dataset sample cap is set here. The universal 25,000-example cap is applied
inside the data-loading implementation, as requested, so this orchestration
script cannot accidentally cap a split twice.
The former 81-run settings are retained below as commented code for an audit
trail. Generic LR-calibration and execution helpers remain active and shared.
"""
import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import queue
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import pandas as pd
SEEDS = [42, 43]
DEVICES = ["cuda:0", "cuda:1"]
MODEL_CHECKPOINT = "roberta-base"
MODEL_SLUG = "RBASE"
LAYER_MAPPING = "normalized-depth-from-roberta-large"
SELECTED_LRS_FILE = Path("selected_lrs_roberta_base.json")
RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "round_metrics"
LOG_DIR = RESULTS_DIR / "logs" / "thesis_roberta_base"
RUN_CONTROL_DIR = RESULTS_DIR / "run_control"
ACTIVE_RUNS_DIR = RUN_CONTROL_DIR / "active"
STOP_REQUESTS_DIR = RUN_CONTROL_DIR / "stop_requests"
OPERATOR_EVENTS_FILE = RUN_CONTROL_DIR / "operator_events.jsonl"
LR_GRID = [1e-1]
LR_BOUNDARY_RATE = LR_GRID[0]
LR_CALIBRATION_ROUNDS = 45
LR_CALIBRATION_LOCAL_EPOCHS = 1
LR_STAGE = "lr4-e1"
LR_BOUNDARY_STAGE = "lr4-e1-boundary"
LR_RESULT_GLOB = f"THESIS-{MODEL_SLUG}-LR4-E1-*.csv"
BASELINE_LOCAL_EPOCHS = 20
FDA_E_MIN = 3
FDA_LR_E_MAX = 30
DATASETS = {
    "sst2": {
        "num_labels": 2,
        "batch_size": 64,
        "layers": list(range(9, 12)),
        "paper_layers": list(range(18, 24)),
        "rounds": {20: 75, 50: 30},
    },
    "qnli": {
        "num_labels": 2,
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "paper_layers": list(range(15, 24)),
        "rounds": {20: 75, 50: 30},
    },
    "rte": {
        "num_labels": 2,
        "batch_size": 32,
        "layers": list(range(8, 12)),
        "paper_layers": list(range(16, 24)),
        "rounds": {20: 30, 50: 12},
    },
    "mrpc": {
        "num_labels": 2,
        "batch_size": 32,
        "layers": list(range(8, 12)),
        "paper_layers": list(range(16, 24)),
        "rounds": {20: 30, 50: 12},
    },
    "mnli": {
        "num_labels": 3,
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "paper_layers": list(range(15, 24)),
        "rounds": {20: 75, 50: 30},
    },
    "qqp": {
        "num_labels": 2,
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "paper_layers": list(range(15, 24)),
        "rounds": {20: 75, 50: 30},
    },
}
PARAMETER_BUDGETS = {
    "mnli": {
        "P1": list(range(10, 12)),
        "P2": list(range(7, 12)),
        "P3": list(range(6, 12)),
    },
    "qqp": {
        "P1": list(range(10, 12)),
        "P2": list(range(7, 12)),
        "P3": list(range(6, 12)),
    },
}
PAPER_PARAMETER_BUDGETS = {
    "mnli": {
        "P1": list(range(21, 24)),
        "P2": list(range(15, 24)),
        "P3": list(range(12, 24)),
    },
    "qqp": {
        "P1": list(range(21, 24)),
        "P2": list(range(15, 24)),
        "P3": list(range(12, 24)),
    },
}
METHODS = {
    "FedAvg-LoRA": {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
    },
    "FFA-LoRA": {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
    },
    "FedAvg-RoLoRA": {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
    },
    "FDA-RoLoRA-DualTrack": {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
        "gamma": 0.3,
        "theta": 0.0,
    },
    "FDA-RoLoRA-DualTrack-EMA-Standard": {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
        "gamma": 0.3,
        "theta": 0.0,
        "rho": 0.9,
    },
    "FDA-RoLoRA-DualTrack-SMA": {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
        "gamma": 0.3,
        "theta": 0.0,
        "sma_window": 3,
    },
}
METHOD_SLUGS = {
    "FedAvg-LoRA": "lora",
    "FFA-LoRA": "ffa",
    "FedAvg-RoLoRA": "rolora",
    "FDA-RoLoRA-DualTrack": "fda-standard",
    "FDA-RoLoRA-DualTrack-EMA-Standard": "fda-ema09",
    "FDA-RoLoRA-DualTrack-SMA": "fda-sma3",
}
LR_METHODS = [
    "FedAvg-LoRA",
    "FedAvg-RoLoRA",
    "FDA-RoLoRA-DualTrack",
]
MAIN_DATASET_ORDER = ("rte", "mrpc", "qnli", "mnli", "qqp", "sst2")
LR_DATASET_ORDER = ("rte", "sst2", "qnli")
MAIN_CLIENT_COUNTS = (20, 50)
MAIN_CLIENTS_PER_ROUND = {20: 20, 50: 50}
MAIN_SEEDS = (42, 43)
MAIN_ALPHA = 1000.0
MAIN_METHODS = [
    "FedAvg-LoRA",
    "FedAvg-RoLoRA",
    "FDA-RoLoRA-DualTrack",
]
MAIN_RANKS_BY_CLIENT = {
    20: (4,),
    50: (4,),
}
MAIN_LORA_ALPHA_BY_RANK = {
    4: 8,
}
MAIN_TARGET_MODULES = ("query", "value")
MAIN_BASELINE_LOCAL_EPOCHS = (5, 10)
MAIN_FDA_LOCAL_EPOCHS = 1
MAIN_FDA_E_MIN = 3
MAIN_FDA_E_MAX = 20
MAIN_SERVER_OPT_NAME = "SGD"
MAIN_SERVER_OPT_LR = 1.0
MAIN_CLIENT_OPT_NAME = "SGD"
MAIN_FDA_GAMMA = 0.3
MAIN_FDA_THETA = 0.0
MAIN_NUM_WORKERS = 0
MAIN_PIN_MEMORY = False
MAIN_TRAIN_HEAD = False
MAIN_EARLY_STOPPING = True
MAIN_EARLY_STOPPING_START_ACCURACY = {
    "rte": 0.60,
    "mrpc": 0.90,
    "sst2": 0.85,
    "qnli": 0.80,
    "mnli": 0.90,
    "qqp": 0.90,
}
MAIN_C20_EARLY_STOPPING_START_ACCURACY = {
    "mnli": 0.60,
    "qqp": 0.72,
}
MAIN_EARLY_STOPPING_MIN_ROUNDS = 6
MAIN_EARLY_STOPPING_PATIENCE = 3
MAIN_EARLY_STOPPING_MIN_DELTA = 0.005
MAIN_EARLY_STOPPING_WINDOW = 3
MAIN_STAGNATION_ROUND = 10_000
MAIN_STAGNATION_TOLERANCE = 0.0
MAIN_THRESHOLD = False
MAIN_PAPER_PARAMETER_BUDGET = "P2"
MAIN_JOBS_PER_GPU = max(
    1, int(os.environ.get("THESIS_MAIN_JOBS_PER_GPU", "3"))
)
MAIN_BATCH64_JOBS_PER_GPU = max(
    1,
    min(
        MAIN_JOBS_PER_GPU,
        int(os.environ.get("THESIS_MAIN_BATCH64_JOBS_PER_GPU", "2")),
    ),
)
MAIN_CLIENT_PRIORITY = (50, 20)
MAIN_INTERRUPTED_SNAPSHOT_RUN_IDS = frozenset()
MAIN_DATASET_SETTINGS = {
    "sst2": {
        "batch_size": 64,
        "target_layer_indices": list(range(9, 12)),
        "rounds": {20: 50, 50: 30},
    },
    "qnli": {
        "batch_size": 32,
        "target_layer_indices": list(range(7, 12)),
        "rounds": {20: 50, 50: 30},
    },
    "rte": {
        "batch_size": 32,
        "target_layer_indices": list(range(8, 12)),
        "rounds": {20: 30, 50: 12},
    },
    "mrpc": {
        "batch_size": 32,
        "target_layer_indices": list(range(8, 12)),
        "rounds": {20: 30, 50: 12},
    },
    "mnli": {
        "batch_size": 32,
        "target_layer_indices": list(range(7, 12)),
        "rounds": {20: 50, 50: 30},
    },
    "qqp": {
        "batch_size": 32,
        "target_layer_indices": list(range(7, 12)),
        "rounds": {20: 50, 50: 30},
    },
}
MAIN_LEARNING_RATES = {
    "sst2": {
        "FedAvg-LoRA": 0.1,
        "FFA-LoRA": 0.1,
        "FedAvg-RoLoRA": 0.1,
        "FDA-RoLoRA-DualTrack": 0.1,
    },
    "qnli": {
        "FedAvg-LoRA": 0.1,
        "FFA-LoRA": 0.1,
        "FedAvg-RoLoRA": 0.1,
        "FDA-RoLoRA-DualTrack": 0.1,
    },
    "rte": {
        "FedAvg-LoRA": 0.05,
        "FFA-LoRA": 0.05,
        "FedAvg-RoLoRA": 0.05,
        "FDA-RoLoRA-DualTrack": 0.05,
    },
    "mrpc": {
        "FedAvg-LoRA": 0.05,
        "FFA-LoRA": 0.05,
        "FedAvg-RoLoRA": 0.05,
        "FDA-RoLoRA-DualTrack": 0.05,
    },
    "mnli": {
        "FedAvg-LoRA": 0.1,
        "FFA-LoRA": 0.1,
        "FedAvg-RoLoRA": 0.1,
        "FDA-RoLoRA-DualTrack": 0.1,
    },
    "qqp": {
        "FedAvg-LoRA": 0.1,
        "FFA-LoRA": 0.1,
        "FedAvg-RoLoRA": 0.1,
        "FDA-RoLoRA-DualTrack": 0.1,
    },
}
FDA_VARIANTS = [
    "FDA-RoLoRA-DualTrack-EMA-Standard",
    "FDA-RoLoRA-DualTrack-SMA",
]
def load_selected_lrs():
    if not SELECTED_LRS_FILE.exists():
        raise FileNotFoundError(
            f"{SELECTED_LRS_FILE} is missing. "
            "Run the RoBERTa-base learning-rate experiment first."
        )
    with SELECTED_LRS_FILE.open() as file:
        return json.load(file)
def get_learning_rate(dataset, method):
    selected = load_selected_lrs()
    return float(selected[dataset][method])
def get_main_learning_rate(dataset, method):
    """Return the frozen LR used by the production main experiment."""
    return float(MAIN_LEARNING_RATES[dataset][method])
def get_primary_accuracy(dataset, metrics):
    """Extract the round metric used by the gated stopping rule."""
    if dataset == "mnli":
        return (
            metrics.get("accuracy_m", 0.0)
            + metrics.get("accuracy_mm", 0.0)
        ) / 2.0
    if dataset == "cola":
        return metrics.get("matthews_correlation", 0.0)
    return metrics.get("accuracy", 0.0)
_LOGGED_METRIC_KEYS = set()
def _append_metric(parts, label, value):
    if value is None:
        return
    try:
        parts.append(f"{label}={float(value):.4f}")
    except (TypeError, ValueError):
        return
def log_latest_round_metrics(hyperparams, metrics_history):
    """Emit one compact test-metric line for every completed round."""
    if not metrics_history:
        return
    latest = metrics_history[-1]
    round_number = latest.get("round", len(metrics_history))
    run_id = hyperparams.get("extras", "unknown-run")
    metric_key = (run_id, round_number)
    if metric_key in _LOGGED_METRIC_KEYS:
        return
    _LOGGED_METRIC_KEYS.add(metric_key)
    dataset = hyperparams["ds_name"]
    parts = [f"round={round_number}"]
    if dataset == "mnli":
        _append_metric(parts, "test_accuracy", get_primary_accuracy(dataset, latest))
        _append_metric(parts, "matched_accuracy", latest.get("accuracy_m"))
        _append_metric(parts, "mismatched_accuracy", latest.get("accuracy_mm"))
    else:
        _append_metric(parts, "test_accuracy", latest.get("accuracy"))
    _append_metric(parts, "test_f1", latest.get("f1"))
    _append_metric(parts, "test_loss", latest.get("testing_loss"))
    _append_metric(parts, "train_accuracy", latest.get("training_accuracy"))
    _append_metric(parts, "round_epochs", latest.get("round_epochs"))
    print("[METRIC] " + " ".join(parts), flush=True)
def check_accuracy_gated_early_stopping(hyperparams, metrics_history):
    """Stop only after an accuracy gate followed by a true plateau.
    The first raw round accuracy at or above the configured gate starts the
    patience clock. Accuracy values before that round are excluded from both
    smoothing and patience. If the gate is never reached, this function always
    returns False and the run consumes its complete communication-round budget.
    """
    if not metrics_history:
        return False
    log_latest_round_metrics(hyperparams, metrics_history)
    values = [
        get_primary_accuracy(hyperparams["ds_name"], metrics)
        for metrics in metrics_history
    ]
    min_rounds = max(1, int(hyperparams.get("early_stopping_min_rounds", 1)))
    if len(values) < min_rounds:
        return False
    start_accuracy = hyperparams.get(
        "early_stopping_start_accuracy",
        hyperparams.get("plateau_min_accuracy", 0.90),
    )
    patience = max(1, int(hyperparams.get("early_stopping_patience", 5)))
    min_delta = float(hyperparams.get("early_stopping_min_delta", 0.01))
    window_size = max(1, int(hyperparams.get("early_stopping_window", 3)))
    drop_tolerance = float(hyperparams.get("early_stopping_drop_tolerance", 0.0))
    gate_index = next(
        (index for index, value in enumerate(values) if value >= start_accuracy),
        None,
    )
    if gate_index is None:
        return False
    def trailing_mean(index):
        window_start = max(gate_index, index - window_size + 1)
        window = values[window_start:index + 1]
        return sum(window) / len(window)
    best_value = trailing_mean(gate_index)
    rounds_without_improvement = 0
    for index in range(gate_index + 1, len(values)):
        current_value = trailing_mean(index)
        if current_value >= best_value + min_delta:
            best_value = current_value
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        if drop_tolerance > 0 and (best_value - current_value) >= drop_tolerance:
            return True
    return rounds_without_improvement >= patience
def get_fda_epoch_window(stage):
    """Return the independent FDA epoch window for LR or production runs."""
    if stage.startswith("lr"):
        return FDA_E_MIN, FDA_LR_E_MAX
    return MAIN_FDA_E_MIN, MAIN_FDA_E_MAX
def make_config(
    stage,
    dataset,
    method,
    seed,
    num_clients,
    total_rounds,
    alpha,
    learning_rate,
    rank=4,
    layers=None,
    batch_size=None,
    clients_per_round=None,
    local_epochs=None,
    lora_alpha=None,
    target_modules=None,
):
    dataset_config = DATASETS[dataset]
    is_fda = method.startswith("FDA-")
    fda_epoch_window = get_fda_epoch_window(stage) if is_fda else (None, None)
    fda_e_min, fda_e_max = fda_epoch_window
    config = {
        "ds_path": "nyu-mll/glue",
        "ds_name": dataset,
        "num_labels": dataset_config["num_labels"],
        "checkpoint": MODEL_CHECKPOINT,
        "layer_mapping": LAYER_MAPPING,
        "paper_parameter_budget": "P2" if layers is None else None,
        "paper_target_layer_indices": dataset_config["paper_layers"],
        "num_clients": num_clients,
        "clients_per_round": (
            num_clients if clients_per_round is None else clients_per_round
        ),
        "alpha": alpha,
        "batch_size": (
            dataset_config["batch_size"] if batch_size is None else batch_size
        ),
        "total_rounds": total_rounds,
        "local_epochs": (
            (1 if is_fda else BASELINE_LOCAL_EPOCHS)
            if local_epochs is None
            else local_epochs
        ),
        "lora_r": rank,
        "lora_alpha": 2 * rank if lora_alpha is None else lora_alpha,
        "target_modules": (
            ["query", "value"]
            if target_modules is None
            else list(target_modules)
        ),
        "target_layer_indices": (
            dataset_config["layers"] if layers is None else list(layers)
        ),
        "num_workers": 0,
        "pin_memory": False,
        "train_head": False,
        "early_stopping": False,
        "threshold": False,
        "seed": seed,
        "METHOD_NAME": method,
        "client_opt_lr": learning_rate,
    }
    config.update(METHODS[method])
    if is_fda:
        config["e_min"] = fda_e_min
        config["max_epochs"] = fda_e_max
    epoch_window_tag = f"-E{fda_e_min}TO{fda_e_max}" if is_fda else ""
    config["extras"] = (
        f"THESIS-{MODEL_SLUG}-{stage.upper()}-DS-{dataset}-C{num_clients}"
        f"-R{total_rounds}-M-{METHOD_SLUGS[method]}"
        f"-LR{learning_rate:g}-RK{rank}{epoch_window_tag}-S{seed}"
    )
    return config
def build_lr_configs(dataset):
    configs = []
    for method in LR_METHODS:
        for learning_rate in LR_GRID:
            config = make_config(
                stage=LR_STAGE,
                dataset=dataset,
                method=method,
                seed=42,
                num_clients=20,
                total_rounds=LR_CALIBRATION_ROUNDS,
                alpha=1000.0,
                learning_rate=learning_rate,
            )
            config["local_epochs"] = LR_CALIBRATION_LOCAL_EPOCHS
            configs.append(config)
    return configs
def lr_score(frame, dataset):
    if dataset == "mnli":
        return ((frame["accuracy_m"] + frame["accuracy_mm"]) / 2).max()
    return frame["accuracy"].max()
def find_best_lr_result(dataset, method):
    best = None
    for path in METRICS_DIR.glob(LR_RESULT_GLOB):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if str(frame.iloc[0]["ds_name"]) != dataset:
            continue
        if str(frame.iloc[0]["METHOD_NAME"]) != method:
            continue
        learning_rate = float(frame.iloc[0]["client_opt_lr"])
        score = float(lr_score(frame, dataset))
        if best is None or score > best["score"]:
            best = {
                "learning_rate": learning_rate,
                "score": score,
            }
    return best
def build_lr_boundary_configs(dataset):
    print(
        "SKIP legacy lr-boundary group: "
        "learning rate 0.1 is now scheduled directly by the lr group"
    )
    return []
def build_main_configs(dataset):
    if dataset not in MAIN_DATASET_SETTINGS:
        raise ValueError(f"No main-experiment settings defined for {dataset}")
    configs = []
    dataset_settings = MAIN_DATASET_SETTINGS[dataset]
    for num_clients in MAIN_CLIENT_COUNTS:
        if num_clients not in MAIN_CLIENTS_PER_ROUND:
            raise ValueError(
                f"MAIN_CLIENTS_PER_ROUND is missing C{num_clients}"
            )
        if num_clients not in dataset_settings["rounds"]:
            raise ValueError(
                f"{dataset} main rounds are missing C{num_clients}"
            )
        if num_clients not in MAIN_RANKS_BY_CLIENT:
            raise ValueError(f"MAIN_RANKS_BY_CLIENT is missing C{num_clients}")
        rounds = dataset_settings["rounds"][num_clients]
        for rank in MAIN_RANKS_BY_CLIENT[num_clients]:
            if rank not in MAIN_LORA_ALPHA_BY_RANK:
                raise ValueError(
                    f"MAIN_LORA_ALPHA_BY_RANK is missing rank {rank}"
                )
            for method in MAIN_METHODS:
                is_fda = method.startswith("FDA-")
                local_epoch_options = (
                    (MAIN_FDA_LOCAL_EPOCHS,)
                    if is_fda
                    else MAIN_BASELINE_LOCAL_EPOCHS
                )
                for local_epochs in local_epoch_options:
                    for seed in MAIN_SEEDS:
                        config = make_config(
                            stage="main",
                            dataset=dataset,
                            method=method,
                            seed=seed,
                            num_clients=num_clients,
                            total_rounds=rounds,
                            alpha=MAIN_ALPHA,
                            rank=rank,
                            learning_rate=get_main_learning_rate(
                                dataset, method
                            ),
                            layers=dataset_settings["target_layer_indices"],
                            batch_size=dataset_settings["batch_size"],
                            clients_per_round=(
                                MAIN_CLIENTS_PER_ROUND[num_clients]
                            ),
                            local_epochs=local_epochs,
                            lora_alpha=MAIN_LORA_ALPHA_BY_RANK[rank],
                            target_modules=MAIN_TARGET_MODULES,
                        )
                        config.update(
                            {
                                "server_opt_name": MAIN_SERVER_OPT_NAME,
                                "server_opt_lr": MAIN_SERVER_OPT_LR,
                                "client_opt_name": MAIN_CLIENT_OPT_NAME,
                                "num_workers": MAIN_NUM_WORKERS,
                                "pin_memory": MAIN_PIN_MEMORY,
                                "train_head": MAIN_TRAIN_HEAD,
                                "early_stopping": MAIN_EARLY_STOPPING,
                                "threshold": MAIN_THRESHOLD,
                                "paper_parameter_budget": (
                                    MAIN_PAPER_PARAMETER_BUDGET
                                ),
                            }
                        )
                        if not is_fda:
                            config["extras"] += f"-E{local_epochs}"
                        if config["early_stopping"]:
                            start_accuracy = (
                                MAIN_C20_EARLY_STOPPING_START_ACCURACY.get(
                                    dataset,
                                    MAIN_EARLY_STOPPING_START_ACCURACY[dataset],
                                )
                                if num_clients == 20
                                else MAIN_EARLY_STOPPING_START_ACCURACY[dataset]
                            )
                            config.update(
                                {
                                    "early_stopping_start_accuracy": (
                                        start_accuracy
                                    ),
                                    "plateau_min_accuracy": start_accuracy,
                                    "early_stopping_min_rounds": (
                                        MAIN_EARLY_STOPPING_MIN_ROUNDS
                                    ),
                                    "early_stopping_patience": (
                                        MAIN_EARLY_STOPPING_PATIENCE
                                    ),
                                    "early_stopping_min_delta": (
                                        MAIN_EARLY_STOPPING_MIN_DELTA
                                    ),
                                    "early_stopping_window": (
                                        MAIN_EARLY_STOPPING_WINDOW
                                    ),
                                    "stagnation_round": (
                                        MAIN_STAGNATION_ROUND
                                    ),
                                    "stagnation_tolerance": (
                                        MAIN_STAGNATION_TOLERANCE
                                    ),
                                }
                            )
                            config["extras"] += (
                                f"-ES{start_accuracy:g}"
                                f"-MR{MAIN_EARLY_STOPPING_MIN_ROUNDS}"
                                f"-P{MAIN_EARLY_STOPPING_PATIENCE}"
                                f"-D{MAIN_EARLY_STOPPING_MIN_DELTA:g}"
                                f"-W{MAIN_EARLY_STOPPING_WINDOW}"
                            )
                        if is_fda:
                            config["gamma"] = MAIN_FDA_GAMMA
                            config["theta"] = MAIN_FDA_THETA
                        configs.append(config)
    return configs
def build_main_all_configs():
    """Queue the complete 252-run cross-silo matrix."""
    return [
        config
        for dataset in MAIN_DATASET_ORDER
        for config in build_main_configs(dataset)
    ]
def print_main_plan(configs=None):
    """Print the frozen main protocol before any GPU job is started."""
    configs = configs if configs is not None else build_main_all_configs()
    client_rank_settings = sum(
        len(MAIN_RANKS_BY_CLIENT[clients])
        for clients in MAIN_CLIENT_COUNTS
    )
    method_variants = sum(
        1 if method.startswith("FDA-") else len(MAIN_BASELINE_LOCAL_EPOCHS)
        for method in MAIN_METHODS
    )
    expected_count = (
        len(MAIN_DATASET_ORDER)
        * client_rank_settings
        * method_variants
        * len(MAIN_SEEDS)
    )
    if len(configs) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} main configs, found {len(configs)}"
        )
    print("\nMAIN EXPERIMENT PLAN")
    print("=" * 72)
    print(f"Model: {MODEL_CHECKPOINT}")
    print(f"Configurations: {len(configs)}")
    print(
        "Methods: FedAvg-LoRA E5/E10; "
        "FedAvg-RoLoRA E5/E10; FDA-RoLoRA-DualTrack E3--E20"
    )
    client_text = ", ".join(
        f"C{clients}/K{MAIN_CLIENTS_PER_ROUND[clients]}"
        for clients in MAIN_CLIENT_COUNTS
    )
    print(f"Clients/round: {client_text}")
    print(f"Seeds: {list(MAIN_SEEDS)}")
    print(f"Partition alpha: {MAIN_ALPHA:g}")
    rank_text = "; ".join(
        f"C{clients}: "
        + ", ".join(
            f"r={rank}/alpha={MAIN_LORA_ALPHA_BY_RANK[rank]}"
            for rank in MAIN_RANKS_BY_CLIENT[clients]
        )
        for clients in MAIN_CLIENT_COUNTS
    )
    print(
        f"LoRA: {rank_text}; "
        f"target modules={','.join(MAIN_TARGET_MODULES)}"
    )
    print(
        f"Optimizer: client={MAIN_CLIENT_OPT_NAME}; "
        f"server={MAIN_SERVER_OPT_NAME}, server LR={MAIN_SERVER_OPT_LR:g}"
    )
    print(f"Baselines: fixed local epochs/round={MAIN_BASELINE_LOCAL_EPOCHS}")
    print(
        f"FDA main: adaptive {MAIN_FDA_E_MIN}--{MAIN_FDA_E_MAX} "
        f"epochs/round; gamma={MAIN_FDA_GAMMA:g}; theta={MAIN_FDA_THETA:g}"
    )
    print(
        "All-method stopping: dataset-specific accuracy gate; "
        f"minimum rounds={MAIN_EARLY_STOPPING_MIN_ROUNDS}; "
        f"patience={MAIN_EARLY_STOPPING_PATIENCE}; "
        f"min_delta={MAIN_EARLY_STOPPING_MIN_DELTA:.1%}; "
        f"window={MAIN_EARLY_STOPPING_WINDOW}"
    )
    c50_gate_text = ", ".join(
        f"{dataset.upper()}={MAIN_EARLY_STOPPING_START_ACCURACY[dataset]:.0%}"
        for dataset in MAIN_DATASET_ORDER
    )
    c20_gate_text = ", ".join(
        f"{dataset.upper()}={MAIN_C20_EARLY_STOPPING_START_ACCURACY.get(dataset, MAIN_EARLY_STOPPING_START_ACCURACY[dataset]):.0%}"
        for dataset in MAIN_DATASET_ORDER
    )
    print(f"C50 stopping activation gates: {c50_gate_text}")
    print(f"C20 stopping activation gates: {c20_gate_text}")
    print(
        f"Queue: {MAIN_JOBS_PER_GPU} jobs/GPU across {len(DEVICES)} GPUs; "
        f"batch-64 safety cap={MAIN_BATCH64_JOBS_PER_GPU} jobs/GPU"
    )
    print(
        "Priority: C50 E5+FDA -> C50 E10 -> C20 E5+FDA -> C20 E10; datasets: "
        "RTE -> MRPC -> QNLI -> MNLI -> QQP -> SST-2; rank 4 only"
    )
    for dataset in MAIN_DATASET_ORDER:
        dataset_config = MAIN_DATASET_SETTINGS[dataset]
        layer_text = ",".join(
            str(layer) for layer in dataset_config["target_layer_indices"]
        )
        round_text = ", ".join(
            f"C{clients}={rounds} rounds"
            for clients, rounds in dataset_config["rounds"].items()
        )
        lr_text = ", ".join(
            f"{METHOD_SLUGS[method]}={get_main_learning_rate(dataset, method):g}"
            for method in MAIN_METHODS
        )
        print(
            f"{dataset.upper()}: batch={dataset_config['batch_size']}; "
            f"layers={layer_text}; {round_text}; LR[{lr_text}]"
        )
    print("=" * 72)
def build_noniid_configs(dataset):
    if dataset not in {"mnli", "qqp"}:
        raise ValueError("Non-IID experiments only use MNLI and QQP.")
    configs = []
    settings = [
        {"num_clients": 10, "alpha": 0.5, "rounds": 100},
        {"num_clients": 15, "alpha": 1.0, "rounds": 100},
    ]
    for setting in settings:
        for method in MAIN_METHODS:
            for seed in SEEDS:
                configs.append(
                    make_config(
                        stage="noniid",
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        num_clients=setting["num_clients"],
                        total_rounds=setting["rounds"],
                        alpha=setting["alpha"],
                        learning_rate=get_learning_rate(dataset, method),
                    )
                )
    return configs
def build_variant_configs(dataset):
    if dataset not in {"mnli", "qqp"}:
        raise ValueError("FDA variant experiments only use MNLI and QQP.")
    configs = []
    settings = [
        {"num_clients": 10, "alpha": 0.5, "rounds": 100},
        {"num_clients": 15, "alpha": 1.0, "rounds": 100},
    ]
    for setting in settings:
        for method in FDA_VARIANTS:
            for seed in SEEDS:
                configs.append(
                    make_config(
                        stage="variants",
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        num_clients=setting["num_clients"],
                        total_rounds=setting["rounds"],
                        alpha=setting["alpha"],
                        learning_rate=get_learning_rate(dataset, method),
                    )
                )
    return configs
def build_rank_configs(dataset):
    if dataset not in {"mnli", "qqp"}:
        raise ValueError("Rank experiments only use MNLI and QQP.")
    configs = []
    for rank in [2, 8]:
        for method in ["FedAvg-RoLoRA", "FDA-RoLoRA-DualTrack"]:
            for seed in SEEDS:
                configs.append(
                    make_config(
                        stage="rank",
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        num_clients=50,
                        total_rounds=30,
                        alpha=1000.0,
                        learning_rate=get_learning_rate(dataset, method),
                        rank=rank,
                    )
                )
    return configs
def build_parameter_configs(dataset):
    if dataset not in PARAMETER_BUDGETS:
        raise ValueError("Parameter-budget experiments only use MNLI and QQP.")
    configs = []
    for budget_name, layers in PARAMETER_BUDGETS[dataset].items():
        for method in ["FedAvg-RoLoRA", "FDA-RoLoRA-DualTrack"]:
            for seed in SEEDS:
                config = make_config(
                    stage=f"params-{budget_name}",
                    dataset=dataset,
                    method=method,
                    seed=seed,
                    num_clients=20,
                    total_rounds=75,
                    alpha=1000.0,
                    learning_rate=get_learning_rate(dataset, method),
                    layers=layers,
                )
                config["parameter_budget"] = budget_name
                config["paper_parameter_budget"] = budget_name
                config["paper_target_layer_indices"] = (
                    PAPER_PARAMETER_BUDGETS[dataset][budget_name]
                )
                configs.append(config)
    return configs
def select_learning_rates():
    selections = {}
    for path in METRICS_DIR.glob(LR_RESULT_GLOB):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        dataset = str(frame.iloc[0]["ds_name"])
        method = str(frame.iloc[0]["METHOD_NAME"])
        learning_rate = float(frame.iloc[0]["client_opt_lr"])
        score = float(lr_score(frame, dataset))
        key = (dataset, method)
        previous = selections.get(key)
        if previous is None or score > previous["score"]:
            selections[key] = {
                "learning_rate": learning_rate,
                "score": score,
            }
    output = {}
    for dataset in LR_DATASET_ORDER:
        output[dataset] = {}
        for method in LR_METHODS:
            key = (dataset, method)
            if key not in selections:
                raise RuntimeError(
                    f"Missing learning-rate results for {dataset}/{method}"
                )
            output[dataset][method] = selections[key]["learning_rate"]
    with SELECTED_LRS_FILE.open("w") as file:
        json.dump(output, file, indent=2)
    print(f"Saved {SELECTED_LRS_FILE}")
def build_compute_match_configs():
    configs = []
    for seed in SEEDS:
        pattern = (
            f"THESIS-{MODEL_SLUG}-MAIN-DS-mnli-C50-R30-M-fda-standard-"
            f"*-E{MAIN_FDA_E_MIN}TO{MAIN_FDA_E_MAX}-S{seed}_*.csv"
        )
        files = sorted(METRICS_DIR.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"Missing MNLI/50-client FDA Standard result for seed {seed}"
            )
        frame = pd.read_csv(files[-1])
        mean_epochs = float(frame["round_epochs"].mean())
        matched_epochs = max(1, int(mean_epochs + 0.5))
        config = make_config(
            stage="compute-match",
            dataset="mnli",
            method="FedAvg-RoLoRA",
            seed=seed,
            num_clients=50,
            total_rounds=30,
            alpha=1000.0,
            learning_rate=get_learning_rate("mnli", "FedAvg-RoLoRA"),
        )
        config["local_epochs"] = matched_epochs
        config["fda_mean_epochs"] = mean_epochs
        config["extras"] = (
            f"THESIS-{MODEL_SLUG}-COMPUTE-MATCH-DS-mnli-C50-R30"
            f"-M-rolora-E{matched_epochs}-S{seed}"
        )
        configs.append(config)
    return configs
def install_runtime_patches(config):
    from peft import LoraConfig as OriginalLoraConfig
    from fdaopt.datasets import fed_data_prep
    from fdaopt.training import fda_opt, fed_opt
    original_tokenize_function = fed_data_prep.tokenize_function
    def tokenize_function(ds_path, ds_name, tokenizer):
        if ds_name == "qqp":
            return lambda example: tokenizer(
                example["question1"],
                example["question2"],
                truncation=True,
            )
        return original_tokenize_function(ds_path, ds_name, tokenizer)
    def configured_lora(*args, **kwargs):
        kwargs["layers_to_transform"] = config["target_layer_indices"]
        kwargs["layers_pattern"] = "layer"
        return OriginalLoraConfig(*args, **kwargs)
    fed_data_prep.tokenize_function = tokenize_function
    fed_opt.LoraConfig = configured_lora
    fda_opt.LoraConfig = configured_lora
    if config.get("early_stopping"):
        fed_opt.check_dynamic_early_stopping = (
            check_accuracy_gated_early_stopping
        )
        fda_opt.check_dynamic_early_stopping = (
            check_accuracy_gated_early_stopping
        )
def execute_run(config_json):
    import torch
    from fdaopt import utils
    from fdaopt.training import fda_opt, fed_opt
    config = json.loads(config_json)
    method = config["METHOD_NAME"]
    device = config.pop("TARGET_DEVICE", "cuda:0")
    install_runtime_patches(config)
    torch.cuda.set_device(torch.device(device))
    utils.DEVICE_RAM_PROGRAM = "performance"
    utils.set_device(device)
    utils.set_seed(config["seed"])
    dispatch = {
        "FedAvg-LoRA": fed_opt.fedavg_lora,
        "FFA-LoRA": fed_opt.ffa_lora,
        "FedAvg-RoLoRA": fed_opt.fedavg_rolora,
        "FDA-RoLoRA-DualTrack": (
            fda_opt.fda_rolora_dual_track_opt
        ),
        "FDA-RoLoRA-DualTrack-EMA-Standard": (
            fda_opt.fda_rolora_dual_track_ema_standard_opt
        ),
        "FDA-RoLoRA-DualTrack-SMA": (
            fda_opt.fda_rolora_dual_track_sma_opt
        ),
    }
    dispatch[method](config)
def is_completed(config):
    """Accept only a readable, non-empty metrics file as a finished run."""
    pattern = str(METRICS_DIR / f"{config['extras']}_*.csv")
    for path in glob.glob(pattern):
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if frame.empty or "round" not in frame.columns:
            continue
        if "extras" in frame.columns:
            recorded_id = str(frame.iloc[0]["extras"])
            if recorded_id != config["extras"]:
                continue
        return True
    return False
def print_restart_manifest(pending):
    """Mark every configuration selected by the result-driven restart."""
    print("Restart manifest:")
    for config in pending:
        run_id = config["extras"]
        state = (
            "INTERRUPTED"
            if run_id in MAIN_INTERRUPTED_SNAPSHOT_RUN_IDS
            else "INCOMPLETE"
        )
        print(f"  {state} {run_id}")
def get_control_id(run_id):
    """Return a short, stable operator ID for a full experiment identifier."""
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
def process_start_ticks(pid):
    """Read Linux process start ticks so stale PID reuse cannot kill a new job."""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields_after_name = stat_text.rsplit(")", 1)[1].split()
        return int(fields_after_name[19])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None
def active_registry_path(control_id):
    return ACTIVE_RUNS_DIR / f"{control_id}.json"
def stop_request_path(control_id):
    return STOP_REQUESTS_DIR / f"{control_id}.json"
def process_matches_registry(record):
    pid = int(record["pid"])
    expected_ticks = record.get("process_start_ticks")
    actual_ticks = process_start_ticks(pid)
    if expected_ticks is not None and actual_ticks is not None:
        return actual_ticks == expected_ticks
    try:
        os.kill(pid, 0)
        return os.getpgid(pid) == pid
    except (PermissionError, ProcessLookupError):
        return False
def load_active_run_records(prune_stale=True):
    records = []
    if not ACTIVE_RUNS_DIR.exists():
        return records
    for path in sorted(ACTIVE_RUNS_DIR.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            if prune_stale:
                path.unlink(missing_ok=True)
            continue
        if not process_matches_registry(record):
            if prune_stale:
                path.unlink(missing_ok=True)
            continue
        records.append(record)
    return records
def wait_for_registered_run(config, record, poll_seconds=5):
    """Adopt a detached worker by waiting for its registered PID to finish."""
    run_id = record["run_id"]
    control_id = record["control_id"]
    print(
        f"ADOPT [{control_id}] pid={record['pid']} {run_id} "
        f"on {record['device']}"
    )
    while process_matches_registry(record):
        time.sleep(poll_seconds)
    active_registry_path(control_id).unlink(missing_ok=True)
    if config is None:
        print(f"RELEASE [{control_id}] detached thesis worker finished")
        return True
    if is_completed(config):
        final_metric = read_latest_metric_line(record["stdout_path"])
        print(f"ADOPTED DONE [{control_id}] {run_id}; {final_metric}")
        return True
    print(
        f"ADOPTED ERROR [{control_id}] {run_id}: worker exited without "
        "a readable metrics CSV"
    )
    return False
def read_latest_metric_line(log_path, max_bytes=131_072):
    path = Path(log_path)
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            end = file.tell()
            file.seek(max(0, end - max_bytes))
            text = file.read().decode("utf-8", errors="replace")
    except OSError:
        return "waiting for first evaluation"
    for line in reversed(text.splitlines()):
        if line.startswith("[METRIC]"):
            return line.removeprefix("[METRIC]").strip()
    return "waiting for first evaluation"
def format_elapsed(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
def list_active_runs():
    """Print active run IDs, PIDs, devices, elapsed time, and latest metrics."""
    records = load_active_run_records()
    if not records:
        print("No active thesis runs.")
        return
    print(f"Active thesis runs: {len(records)}")
    for record in records:
        elapsed = format_elapsed(time.time() - float(record["started_at"]))
        latest_metric = read_latest_metric_line(record["stdout_path"])
        print(
            f"[{record['control_id']}] pid={record['pid']} "
            f"gpu={record['device']} elapsed={elapsed}"
        )
        print(f"  run:    {record['run_id']}")
        print(f"  latest: {latest_metric}")
        print(f"  log:    {record['stdout_path']}")
    print("Stop one attempt with: ./run_thesis_experiments.sh stop <ID>")
def append_operator_event(payload):
    RUN_CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    with OPERATOR_EVENTS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())
def resolve_active_run(selector):
    records = load_active_run_records()
    exact_matches = [
        record
        for record in records
        if selector
        in {
            record["control_id"],
            str(record["pid"]),
            record["run_id"],
        }
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    prefix_matches = [
        record
        for record in records
        if record["control_id"].startswith(selector)
        or record["run_id"].startswith(selector)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if not exact_matches and not prefix_matches:
        raise SystemExit(f"No active run matches {selector!r}")
    matching_ids = ", ".join(
        record["control_id"] for record in (exact_matches or prefix_matches)
    )
    raise SystemExit(f"Ambiguous run selector {selector!r}: {matching_ids}")
def stop_active_run(selector, reason):
    """Request SIGTERM for one verified child process group."""
    record = resolve_active_run(selector)
    pid = int(record["pid"])
    if not process_matches_registry(record):
        raise SystemExit("The selected run ended before the stop request")
    try:
        process_group = os.getpgid(pid)
    except ProcessLookupError as error:
        raise SystemExit("The selected run is no longer active") from error
    if process_group != pid:
        raise SystemExit(
            "Refusing to stop an unverified process group; registry is stale"
        )
    event = {
        "action": "operator_stop",
        "control_id": record["control_id"],
        "device": record["device"],
        "pid": pid,
        "reason": reason,
        "requested_at": time.time(),
        "run_id": record["run_id"],
    }
    request_path = stop_request_path(record["control_id"])
    atomic_write_json(request_path, event)
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError as error:
        request_path.unlink(missing_ok=True)
        raise SystemExit("The selected run ended before SIGTERM") from error
    append_operator_event(event)
    print(
        f"Stop requested for [{record['control_id']}] pid={pid}: "
        f"{record['run_id']}"
    )
    print("The scheduler will continue with the next queued configuration.")
def run_config_on_device(config, device):
    """Run one configuration on an explicitly assigned GPU."""
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
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stdout_path = LOG_DIR / f"{run_id}.out"
        stderr_path = LOG_DIR / f"{run_id}.err"
        temporary_dir = tempfile.mkdtemp(
            prefix=f"{run_id}-",
            dir="/tmp",
        )
        environment = os.environ.copy()
        environment["HF_METRICS_CACHE"] = os.path.join(
            temporary_dir,
            "metrics",
        )
        environment["HF_DATASETS_CACHE"] = os.path.join(
            temporary_dir, "datasets"
        )
        environment["TMPDIR"] = temporary_dir
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        with (
            stdout_path.open("w") as stdout_file,
            stderr_path.open("w") as stderr_file,
        ):
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
            control_id = get_control_id(run_id)
            registry_path = active_registry_path(control_id)
            request_path = stop_request_path(control_id)
            request_path.unlink(missing_ok=True)
            record = {
                "batch_size": int(config["batch_size"]),
                "control_id": control_id,
                "device": device,
                "pid": process.pid,
                "process_start_ticks": process_start_ticks(process.pid),
                "run_id": run_id,
                "started_at": time.time(),
                "stderr_path": str(stderr_path.resolve()),
                "stdout_path": str(stdout_path.resolve()),
            }
            atomic_write_json(registry_path, record)
            print(
                f"START [{control_id}] pid={process.pid} {run_id} on {device}"
            )
            return_code = process.wait()
        if request_path.exists():
            print(f"STOPPED [{control_id}] {run_id} by operator request")
            return True
        if return_code != 0:
            print(f"ERROR {run_id}: {stderr_path}")
            return False
        final_metric = read_latest_metric_line(stdout_path)
        print(f"DONE [{control_id}] {run_id}; {final_metric}")
        return True
    finally:
        if registry_path is not None:
            registry_path.unlink(missing_ok=True)
        if request_path is not None:
            request_path.unlink(missing_ok=True)
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
def run_worker(config, device_queue):
    """Legacy shared-device worker used by non-main experiment groups."""
    device = device_queue.get()
    try:
        return run_config_on_device(config, device)
    finally:
        device_queue.put(device)
def main_config_priority(config):
    """Sort main runs from the smallest/faster configurations upward."""
    dataset = config["ds_name"]
    num_clients = config["num_clients"]
    method = config["METHOD_NAME"]
    dataset_rank = MAIN_DATASET_ORDER.index(dataset)
    client_rank = MAIN_CLIENT_PRIORITY.index(num_clients)
    method_rank = MAIN_METHODS.index(method)
    epoch_rank = (
        -1
        if method.startswith("FDA-")
        else MAIN_BASELINE_LOCAL_EPOCHS.index(config["local_epochs"])
    )
    return (
        dataset_rank,
        client_rank,
        config["total_rounds"],
        config["lora_r"],
        epoch_rank,
        config["seed"],
        method_rank,
    )
def run_main_configs(configs, job_runner=None):
    """Finish E5 plus FDA, then E10, for C50 before repeating for C20."""
    for clients in MAIN_CLIENT_PRIORITY:
        for epochs in MAIN_BASELINE_LOCAL_EPOCHS:
            phase_configs = [
                c for c in configs
                if c["num_clients"] == clients
                and (
                    epochs == MAIN_BASELINE_LOCAL_EPOCHS[0]
                    if c["METHOD_NAME"].startswith("FDA-")
                    else c["local_epochs"] == epochs
                )
            ]
            if not phase_configs:
                continue
            phase_label = f"C{clients} E{epochs}"
            if epochs == MAIN_BASELINE_LOCAL_EPOCHS[0]:
                phase_label += " + adaptive FDA"
            print(f"Execution phase: {phase_label} across all datasets")
            run_main_phase(phase_configs, job_runner=job_runner)
            if job_runner is None and any(not is_completed(c) for c in phase_configs):
                print(f"{phase_label} has unfinished runs; later phases will not start.")
                raise SystemExit(1)
def run_main_phase(configs, job_runner=None):
    """Run up to three lanes/GPU with a conservative batch-64 limit.
    Lane zero on each GPU prefers FDA, while the remaining lanes prefer
    baselines. Idle lanes borrow from the other queue. The central dispatcher
    prevents more than ``MAIN_BATCH64_JOBS_PER_GPU`` batch-64 runs from being
    active on the same GPU, matching the measured A10 resource envelope.
    """
    job_runner = run_config_on_device if job_runner is None else job_runner
    active_records = load_active_run_records()
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
        if config["extras"] not in active_run_ids and not is_completed(config)
    ]
    fda_pending = sorted(
        [
            config
            for config in pending
            if config["METHOD_NAME"].startswith("FDA-")
        ],
        key=main_config_priority,
    )
    baseline_pending = sorted(
        [
            config
            for config in pending
            if not config["METHOD_NAME"].startswith("FDA-")
        ],
        key=main_config_priority,
    )
    print(
        f"Scheduled {len(pending)} queued and "
        f"{len(adopted_configs)} adopted configurations"
    )
    print(
        f"Skipped {len(configs) - len(pending) - len(adopted_configs)} "
        "completed configurations"
    )
    print(
        f"Main scheduler: {MAIN_JOBS_PER_GPU} dynamic lanes/GPU; "
        "one FDA-preferred lane and remaining baseline-preferred lanes; "
        f"batch-64 cap={MAIN_BATCH64_JOBS_PER_GPU}/GPU"
    )
    print(
        f"Initial queues: FDA={len(fda_pending)}, "
        f"baseline={len(baseline_pending)}"
    )
    if not pending and not adopted_configs:
        return
    if pending:
        print_restart_manifest(fda_pending + baseline_pending)
    lanes = [
        (
            device,
            lane_index,
            "fda" if lane_index == 0 else "baseline",
        )
        for device in DEVICES
        for lane_index in range(MAIN_JOBS_PER_GPU)
    ]
    active_batch64 = {device: 0 for device in DEVICES}
    def take_allowed(config_queue, device):
        for index, config in enumerate(config_queue):
            is_batch64 = config["batch_size"] >= 64
            if (
                is_batch64
                and active_batch64[device] >= MAIN_BATCH64_JOBS_PER_GPU
            ):
                continue
            return config_queue.pop(index)
        return None
    def take_next(preferred_kind, device):
        preferred_queue = (
            fda_pending if preferred_kind == "fda" else baseline_pending
        )
        fallback_queue = (
            baseline_pending if preferred_kind == "fda" else fda_pending
        )
        return (
            take_allowed(preferred_queue, device)
            or take_allowed(fallback_queue, device)
        )
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(lanes)
    ) as executor:
        future_to_lane = {}
        idle_lanes = list(lanes)
        for record in active_records:
            device = record.get("device")
            matching_lanes = [lane for lane in idle_lanes if lane[0] == device]
            if not matching_lanes:
                raise RuntimeError(
                    f"More registered workers than configured lanes on {device}"
                )
            lane = matching_lanes[0]
            config = configs_by_run_id.get(record["run_id"])
            future = executor.submit(wait_for_registered_run, config, record)
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
                    device, _lane_index, preferred_kind = lane
                    config = take_next(preferred_kind, device)
                    if config is None:
                        continue
                    future = executor.submit(job_runner, config, device)
                    future_to_lane[future] = (lane, config, None)
                    idle_lanes.remove(lane)
                    if config["batch_size"] >= 64:
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
                device, _lane_index, _preferred_kind = lane
                results.append(future.result())
                completed_batch_size = (
                    completed_config["batch_size"]
                    if completed_config is not None
                    else int(adopted_record.get("batch_size", 64))
                )
                if completed_batch_size >= 64:
                    active_batch64[device] -= 1
                idle_lanes.append(lane)
            launch_available()
        if fda_pending or baseline_pending:
            raise RuntimeError(
                "Main scheduler stopped with queued configurations remaining"
            )
    if not all(results):
        raise SystemExit(1)
def run_configs(configs):
    if not configs:
        return
    is_main_group = all("-MAIN-" in config["extras"] for config in configs)
    if is_main_group:
        run_main_configs(configs)
        return
    jobs_per_gpu = MAIN_JOBS_PER_GPU if is_main_group else 2
    device_queue = queue.Queue()
    for _ in range(jobs_per_gpu):
        for device in DEVICES:
            device_queue.put(device)
    pending = [
        config
        for config in configs
        if not is_completed(config)
    ]
    print(f"Scheduled {len(pending)} incomplete configurations")
    print(f"Pending {len(pending)}; already complete {len(configs) - len(pending)}")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(DEVICES) * jobs_per_gpu
    ) as executor:
        futures = [
            executor.submit(run_worker, config, device_queue)
            for config in pending
        ]
        results = [future.result() for future in futures]
    if not all(results):
        raise SystemExit(1)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=[
            "lr",
            "lr-all",
            "lr-boundary",
            "lr-boundary-all",
            "select-lr",
            "main",
            "main-all",
            "show-main-plan",
            "list-runs",
            "stop-run",
        ],
    )
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--run-config")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--reason",
        default="operator stopped run after inspecting live metrics",
    )
    args = parser.parse_args()
    if args.run_config:
        execute_run(args.run_config)
        return
    if args.group == "list-runs":
        list_active_runs()
        return
    if args.group == "stop-run":
        if not args.run_id:
            parser.error("--run-id is required for --group stop-run")
        stop_active_run(args.run_id, args.reason)
        return
    if args.group == "select-lr":
        select_learning_rates()
        return
    if args.group == "lr-all":
        run_configs(
            [
                config
                for dataset in LR_DATASET_ORDER
                for config in build_lr_configs(dataset)
            ]
        )
        return
    if args.group == "lr-boundary-all":
        run_configs(
            [
                config
                for dataset in LR_DATASET_ORDER
                for config in build_lr_boundary_configs(dataset)
            ]
        )
        return
    if args.group == "show-main-plan":
        print_main_plan()
        return
    if args.group == "main-all":
        configs = build_main_all_configs()
        print_main_plan(configs)
        run_configs(configs)
        return
    if not args.dataset:
        parser.error("--dataset is required for this experiment group")
    builders = {
        "lr": build_lr_configs,
        "lr-boundary": build_lr_boundary_configs,
        "main": build_main_configs,
    }
    run_configs(builders[args.group](args.dataset))
if __name__ == "__main__":
    main()
