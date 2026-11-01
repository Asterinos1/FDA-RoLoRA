# Repository Structure: `ARIS-FDA-LoRA`

```text
ARIS-FDA-LoRA/
    controller2.py
    llm_summary.json
    thesis_experiments.py
    check.py
    sub_exp_1_simulation.py
    simulator.py
    thesis_sma_experiments.py
    run_threshold_experiments.py
    run_lr001_mnli_qnli.py
    thesis_pure_experiments.py
    thesis_calibration_experiments.py
    parallel_evaluation.py
    toy_model_simulation.py
    simulation.py
    roberta_large_pilot.py
    controller.py
    thesis_ema_pure_experiments.py
    toy_experiment_runner.py
    sub_exp_2_simulation.py
    master_experiments/
        README.md
        thesis_master_experiments.py
    non_iid_experiments/
        README.md
        thesis_non_iid_experiments.py
        fda_rolora_non_iid.py
    results.local-backup-20260812-161442/
        output/
        logs/
        round_metrics/
    ready_master_runs/
    pure_experiments/
        README.md
        fda_rolora_pure.py
        thesis_pure_experiments.py
    logs/
    toy/
        README.md
        analyze_toy.py
        thesis_toy_experiments.py
    fdaopt/
        utils.py
        main.py
        models/
            toy_model.py
            ops.py
        training/
            optimizers.py
            fed_train.py
            fed_opt.py
            fda_opt.py
            sketch.py
        miscellaneous/
            parameter_tools.py
            create_hyperparameters.py
        datasets/
            fed_data_prep.py
        metrics/
            mentrics_handler.py
    fillers/
        thesis_filler_experiments.py
        README.md
        profile_tf32.py
        profile_optimizations.py
```

# File Contents


## File: `controller2.py`

```py
"""Run the isolated FDA 10--50 epoch controller ablation on SST-2.
This runner reuses the frozen C50, rank-4 production configuration from
``thesis_experiments.py``. It keeps the MAX50/MID25 controller experiment
unchanged except for increasing the minimum local epochs from 3 to 10. The
run is capped at 15 communication rounds and uses the same 90%-gated plateau
stopping configuration as the baseline experiments.
Examples:
    python controller_min10_max50_experiment.py plan
    python controller_min10_max50_experiment.py run --seed 42
    python controller_min10_max50_experiment.py run --seed 42 43 44
"""
import argparse
from pathlib import Path
import thesis_experiments as thesis
METHOD = "FDA-RoLoRA-DualTrack"
DATASET = "sst2"
NUM_CLIENTS = 50
CLIENTS_PER_ROUND = 50
TOTAL_ROUNDS = 15
LORA_RANK = 4
FDA_E_MIN = 10
FDA_E_MAX = 50
FDA_MIDPOINT_EPOCHS = FDA_E_MAX / 2
DEFAULT_SEEDS = (42,)
LOG_DIR = Path("results/logs/controller_min10_max50")
EARLY_STOPPING_START_ACCURACY = 0.90
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.01
EARLY_STOPPING_WINDOW = 3
def build_config(seed: int) -> dict:
    """Derive one MIN10/MAX50 FDA run from the production configuration."""
    candidates = [
        config
        for config in thesis.build_main_configs(DATASET)
        if config["METHOD_NAME"] == METHOD
        and config["num_clients"] == NUM_CLIENTS
        and config["clients_per_round"] == CLIENTS_PER_ROUND
        and config["lora_r"] == LORA_RANK
        and config["seed"] == seed
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one production template for {DATASET}/seed-{seed}, "
            f"found {len(candidates)}."
        )
    config = dict(candidates[0])
    config["total_rounds"] = TOTAL_ROUNDS
    config["e_min"] = FDA_E_MIN
    config["max_epochs"] = FDA_E_MAX
    config["controller_midpoint_epochs"] = FDA_MIDPOINT_EPOCHS
    config["early_stopping"] = True
    config["early_stopping_start_accuracy"] = EARLY_STOPPING_START_ACCURACY
    config["plateau_min_accuracy"] = EARLY_STOPPING_START_ACCURACY
    config["early_stopping_patience"] = EARLY_STOPPING_PATIENCE
    config["early_stopping_min_delta"] = EARLY_STOPPING_MIN_DELTA
    config["early_stopping_window"] = EARLY_STOPPING_WINDOW
    config["experiment_group"] = "controller-min10-max50"
    config["extras"] = (
        f"THESIS-{thesis.MODEL_SLUG}-CTRL-MIN10-MAX50"
        f"-DS-{DATASET}-C{NUM_CLIENTS}-R{TOTAL_ROUNDS}"
        f"-M-{thesis.METHOD_SLUGS[METHOD]}"
        f"-LR{config['client_opt_lr']:g}-RK{LORA_RANK}"
        f"-E{FDA_E_MIN}TO{FDA_E_MAX}-MID{FDA_MIDPOINT_EPOCHS:g}"
        f"-S{seed}"
        f"-ES{EARLY_STOPPING_START_ACCURACY:g}"
        f"-P{EARLY_STOPPING_PATIENCE}"
        f"-D{EARLY_STOPPING_MIN_DELTA:g}"
        f"-W{EARLY_STOPPING_WINDOW}"
    )
    return config
def print_plan(configs: list[dict], device: str) -> None:
    print("\nFDA CONTROLLER MIN-10 / MAX-50 ABLATION")
    print("=" * 72)
    print(f"Model: {thesis.MODEL_CHECKPOINT}")
    print(f"Configurations: {len(configs)}")
    print(f"Device: {device}; execution: sequential, one process at a time")
    print(f"Dataset: {DATASET.upper()}")
    print(f"Method: {METHOD}")
    print(f"Clients/round: C{NUM_CLIENTS}/K{CLIENTS_PER_ROUND}")
    print(f"LoRA: r={LORA_RANK}, alpha={2 * LORA_RANK}; query,value")
    print(
        f"FDA window: {FDA_E_MIN}--{FDA_E_MAX} epochs; "
        f"threshold midpoint={FDA_MIDPOINT_EPOCHS:g} epochs"
    )
    print(
        "Stopping: activates at 90% accuracy; patience=5; "
        "min_delta=1%; window=3; hard cap=15 rounds"
    )
    for config in configs:
        print(
            f"{config['ds_name'].upper()}: batch={config['batch_size']}; "
            f"layers={','.join(map(str, config['target_layer_indices']))}; "
            f"rounds={config['total_rounds']}; seed={config['seed']}; "
            f"LR={config['client_opt_lr']:g}"
        )
        print(f"  {config['extras']}")
    print("=" * 72)
def run_configs(configs: list[dict], device: str) -> None:
    thesis.LOG_DIR = LOG_DIR
    for config in configs:
        if not thesis.run_config_on_device(config, device):
            raise SystemExit(1)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated SST-2 FDA e_min=10, max_epochs=50 ablation."
    )
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="One or more seeds, executed sequentially.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used by the single experiment process.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    configs = [build_config(seed) for seed in args.seed]
    print_plan(configs, args.device)
    if args.action == "run":
        run_configs(configs, args.device)
if __name__ == "__main__":
    main()

```


## File: `thesis_experiments.py`

```py
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

```


## File: `check.py`

```py
import os
import glob
import pandas as pd
import json
def parse_metrics_for_llm(results_dir="results/round_metrics"):
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    allowed_groups = [
        "TOY-BASELINES",
        "TOY-FDA_STANDARD",
        "TOY-FDA_EMA_STANDARD",
        "TOY-FDA_EMA_PURE",
        "TOY-FDA_SMA"
    ]
    summaries = []
    for file in csv_files:
        filename = os.path.basename(file)
        if not any(group in filename for group in allowed_groups):
            continue
        try:
            df = pd.read_csv(file)
            if df.empty:
                continue
            first_row = df.iloc[0]
            method = first_row.get('METHOD_NAME', 'Unknown')
            dataset = first_row.get('ds_name', 'Unknown')
            seed = first_row.get('seed', None)
            extras = first_row.get('extras', None)
            target_metric = 'accuracy'
            if dataset == 'cola':
                target_metric = 'matthews_correlation'
            elif dataset == 'mnli':
                if 'accuracy_m' in df.columns and 'accuracy_mm' in df.columns:
                    df['mnli_avg_acc'] = (df['accuracy_m'] + df['accuracy_mm']) / 2
                    target_metric = 'mnli_avg_acc'
            has_metric = target_metric in df.columns
            total_rounds = df['round'].max() if 'round' in df.columns else len(df)
            if has_metric:
                best_round_idx = df[target_metric].idxmax()
                best_round = df.loc[best_round_idx, 'round'] if 'round' in df.columns else best_round_idx + 1
                best_metric_val = df.loc[best_round_idx, target_metric]
                final_metric_val = df.iloc[-1][target_metric]
            else:
                best_round, best_metric_val, final_metric_val = None, None, None
            final_train_loss = df.iloc[-1]['training_loss'] if 'training_loss' in df.columns else None
            final_test_loss = df.iloc[-1]['testing_loss'] if 'testing_loss' in df.columns else None
            time_cols = [c for c in df.columns if 'time_round_total' in c]
            total_time = df[time_cols[0]].sum() if time_cols else None
            summary = {
                "Experiment_ID": extras,
                "Method": method,
                "Dataset": dataset,
                "Seed": int(seed) if pd.notna(seed) else None,
                "Total_Rounds": int(total_rounds),
                "Best_Round": int(best_round) if best_round else None,
                f"Best_{target_metric}": round(float(best_metric_val), 4) if best_metric_val else None,
                f"Final_{target_metric}": round(float(final_metric_val), 4) if final_metric_val else None,
                "Final_Train_Loss": round(float(final_train_loss), 4) if final_train_loss else None,
                "Final_Test_Loss": round(float(final_test_loss), 4) if final_test_loss else None,
                "Total_Time_Seconds": round(float(total_time), 2) if total_time else None
            }
            summary = {k: v for k, v in summary.items() if pd.notna(v)}
            summaries.append(summary)
        except Exception:
            continue
    output_file = "llm_summary.json"
    with open(output_file, "w") as f:
        json.dump(summaries, f, indent=2)
if __name__ == "__main__":
    parse_metrics_for_llm()

```


## File: `sub_exp_1_simulation.py`

```py
import os
import sys
import time
import subprocess
import argparse
import json
import concurrent.futures
import queue
import threading
try:
    import torch
    num_gpus = torch.cuda.device_count()
    AVAILABLE_DEVICES = [f'cuda:{i}' for i in range(num_gpus)] if num_gpus > 0 else ['cpu']
except Exception:
    AVAILABLE_DEVICES = ['cpu']
JOBS_PER_DEVICE = 5
MAX_CONCURRENT_JOBS = len(AVAILABLE_DEVICES) * JOBS_PER_DEVICE
progress_lock = threading.Lock()
completed_jobs = 0
total_scheduled_jobs = 0
BASE_HYPERPARAMS = {
    'checkpoint': "roberta-large",
    'num_clients': 50,
    'clients_per_round': 10,
    'alpha': 0.5,
    'total_rounds': 100,
    'batch_size': 32,
    'lora_r': 8,
    'lora_alpha': 16,
    'target_modules': ["query", "value"],
    'num_workers': 4,
    'pin_memory': True,
    'early_stopping': True,
    'early_stopping_patience': 45,
    'train_head': False
}
METHODS = {
    "FedAvg-LoRA": [{'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01}],
    "FedAvg-RoLoRA": [{'server_opt_name': 'SGD', 'server_opt_lr': 1.0, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01}],
    "FFA-LoRA": [{'server_opt_name': 'SGD', 'server_opt_lr': 1.0, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01}],
    "FDA-RoLoRA-DualTrack": [{'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'gamma': 0.3, 'theta': 0.0}]
}
def execute_run(config_json):
    from fdaopt import utils
    from fdaopt.training.fed_opt import fedavg_lora, fedavg_rolora, ffa_lora
    from fdaopt.training.fda_opt import fda_rolora_dual_track_opt
    hyperparams = json.loads(config_json)
    method_name = hyperparams.pop('METHOD_NAME')
    target_device = hyperparams.pop('TARGET_DEVICE', 'cuda:0')
    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(target_device)
    utils.set_seed(hyperparams.get('seed', 42))
    if method_name == "FedAvg-LoRA":
        fedavg_lora(hyperparams)
    elif method_name == "FedAvg-RoLoRA":
        fedavg_rolora(hyperparams)
    elif method_name == "FFA-LoRA":
        ffa_lora(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack":
        fda_rolora_dual_track_opt(hyperparams)
    else:
        raise ValueError(f"Unknown method requested: {method_name}")
def _worker_task(run_config, device_queue, log_dir):
    global completed_jobs, total_scheduled_jobs
    device = device_queue.get()
    run_identifier = run_config['extras']
    out_file = os.path.join(log_dir, f"{run_identifier}.out")
    err_file = os.path.join(log_dir, f"{run_identifier}.err")
    try:
        run_config['TARGET_DEVICE'] = device
        config_str = json.dumps(run_config)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--run_config", config_str]
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        start_time = time.time()
        with open(out_file, 'w') as stdout_f, open(err_file, 'w') as stderr_f:
            subprocess.run(cmd, env=env, stdout=stdout_f, stderr=stderr_f, check=True)
        duration = (time.time() - start_time) / 60
        with progress_lock:
            completed_jobs += 1
            print(f"[{completed_jobs}/{total_scheduled_jobs}] SUCCESS | {run_identifier} ({duration:.2f}m)")
    except subprocess.CalledProcessError:
        with progress_lock:
            completed_jobs += 1
            print(f"[{completed_jobs}/{total_scheduled_jobs}] ERROR | {run_identifier} | Check {err_file}")
    finally:
        time.sleep(3)
        device_queue.put(device)
def orchestrate(dataset_name):
    global total_scheduled_jobs
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    all_runs = []
    labels_map = {"mnli": 3, "sst2": 2, "qnli": 2, "qqp": 2, "rte": 2, "mrpc": 2}
    num_labels = labels_map.get(dataset_name, 2)
    epochs_configs = [
        {'local_epochs': 5, 'e_min': 2, 'max_epochs': 10},
        {'local_epochs': 10, 'e_min': 2, 'max_epochs': 20}
    ]
    for method_name, method_variants in METHODS.items():
        for method_cfg in method_variants:
            for seed in [42, 43, 44]:
                for ep_cfg in epochs_configs:
                    run_config = BASE_HYPERPARAMS.copy()
                    run_config['ds_path'] = "nyu-mll/glue"
                    run_config['ds_name'] = dataset_name
                    run_config['num_labels'] = num_labels
                    run_config.update(method_cfg)
                    run_config['seed'] = seed
                    run_config['METHOD_NAME'] = method_name
                    if "FDA" in method_name:
                        run_config['local_epochs'] = 1
                        run_config['e_min'] = ep_cfg['e_min']
                        run_config['max_epochs'] = ep_cfg['max_epochs']
                        ep_str = f"EMIN{ep_cfg['e_min']}-EMAX{ep_cfg['max_epochs']}"
                    else:
                        run_config['local_epochs'] = ep_cfg['local_epochs']
                        ep_str = f"FEPOCH{ep_cfg['local_epochs']}"
                    run_config['extras'] = f"SUB1_{method_name}_DS-{dataset_name}_{ep_str}_S{seed}"
                    all_runs.append(run_config)
    total_scheduled_jobs = len(all_runs)
    device_queue = queue.Queue()
    for dev in AVAILABLE_DEVICES:
        for _ in range(JOBS_PER_DEVICE):
            device_queue.put(dev)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = [executor.submit(_worker_task, run_config, device_queue, log_dir) for run_config in all_runs]
        concurrent.futures.wait(futures)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, default=None)
    parser.add_argument('--dataset', type=str, help="The GLUE dataset to run (e.g., mrpc, sst2)")
    args = parser.parse_args()
    if args.run_config:
        execute_run(args.run_config)
    elif args.dataset:
        orchestrate(args.dataset)
    else:
        print("Please provide a --dataset argument or a --run_config.")

```


## File: `simulator.py`

```py
import subprocess
import signal
import os
import argparse
from collections import namedtuple
import time
script_directory = os.path.dirname(os.path.abspath(__file__))
HYPERPARAM_DIR = os.path.normpath(os.path.join(script_directory, './hyperparameters'))
OUTPUT_DIR = os.path.normpath(os.path.join(script_directory, './results/output'))
Task = namedtuple('Task', ['hyperparam_file', 'hyperparam_id', 'stdout_filename', 'stderr_filename'])
processes_devices, task_queue = [], []
device_counts, device_limits = {}, {}
def signal_handler(signum, frame):
    """
    Signal handler to kill all child processes.
    """
    global processes_devices
    for (proc, device) in processes_devices:
        if not proc.poll():
            proc.kill()
    print("All processes terminated. Exiting.")
    exit()
def set_device_limits(device_limit_args):
    global device_limits
    device_limits = {}
    for arg in device_limit_args:
        device, num = arg.split('=')
        device_limits[device] = int(num)
def get_free_device():
    global device_counts, device_limits
    for device, count in device_counts.items():
        if count < device_limits[device]:
            return device
    return None
def check_finished_processes():
    global processes_devices, device_counts
    for (proc, device) in processes_devices[:]:
        if proc.poll() is not None:
            device_counts[device] -= 1
            processes_devices.remove((proc, device))
def terminate_and_quit():
    global processes_devices
    print("Terminating all processes based on user input.")
    for (proc, device) in processes_devices:
        if proc.poll() is None:
            proc.kill()
    print("All processes terminated. Exiting.")
    exit()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_limits', nargs='*', required=True,
                        help="How many tests can be run in-parallel in each device (e.g., --device_limit cuda:0=1 cuda:1=1 cpu=2")
    args = parser.parse_args()
    set_device_limits(args.device_limits)
    device_counts = {device: 0 for device in device_limits}
    hyperparameter_files = [filename for filename in os.listdir(HYPERPARAM_DIR) if filename.endswith('json')]
    hyperparameter_files = sorted(hyperparameter_files, key=lambda x: int(x.split('.')[0]))
    for hyperparam_file in hyperparameter_files:
        user_response = input(
            f"Press 'Enter' to queue the test for {hyperparam_file} or 'q' to terminate all and quit: "
        ).strip().lower()
        if user_response == "q":
            terminate_and_quit()
        hyperparam_id = hyperparam_file.split('.')[0]
        task = Task(
            hyperparam_file=hyperparam_file,
            hyperparam_id=hyperparam_id,
            stdout_filename=f'{OUTPUT_DIR}/{hyperparam_id}.out',
            stderr_filename=f'{OUTPUT_DIR}/{hyperparam_id}.err'
        )
        task_queue.append(task)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    while task_queue:
        check_finished_processes()
        device = get_free_device()
        if device:
            task = task_queue.pop(0)
            cmd = ['python', '-u', '-m', 'fdaopt.main', f'--filename={task.hyperparam_file}', f'--device={device}']
            env = os.environ.copy()
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            with open(task.stdout_filename, 'w') as stdout_file:
                with open(task.stderr_filename, 'w') as stderr_file:
                    print(f"Running on {device} the hyperparameter file {task.hyperparam_file}.")
                    process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, env=env)
                    processes_devices.append((process, device))
            device_counts[device] += 1
        else:
            time.sleep(10)
    while processes_devices:
        check_finished_processes()
        time.sleep(10)
    print("All processes terminated. Exiting.")

```


## File: `thesis_sma_experiments.py`

```py
"""Run the final MAIN FDA-RoLoRA SMA W=3, E3--E20 replication.
This is a narrow orchestration layer over the canonical
``thesis_experiments.py`` runner. It reuses the existing training dispatch,
metric writer, process registry, and GPU worker implementation while keeping
SMA logs and control state separate from the completed MAIN study.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
try:
    import thesis_experiments as base
except ModuleNotFoundError as error:
    raise SystemExit(
        "Place thesis_sma_experiments.py beside thesis_experiments.py "
        "in the ARIS-FDA-LoRA repository."
    ) from error
SMA_METHOD = "FDA-RoLoRA-DualTrack-SMA"
STAGE = "main-sma-w3"
SEEDS = (42, 43)
CLIENT_COUNTS = (20, 50)
PARTITION_ALPHA = 1000.0
E_MIN = 3
E_MAX = 20
SMA_WINDOW = 3
GAMMA = 0.3
THETA = 0.0
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")
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
DATASET_COST_ORDER = ("rte", "mrpc", "qnli", "mnli", "qqp", "sst2")
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
RESULTS_DIR = Path("results")
LOG_DIR = RESULTS_DIR / "logs" / "thesis_roberta_base_sma_w3"
RUN_CONTROL_DIR = RESULTS_DIR / "run_control_sma_w3"
ACTIVE_RUNS_DIR = RUN_CONTROL_DIR / "active"
STOP_REQUESTS_DIR = RUN_CONTROL_DIR / "stop_requests"
DROPPED_RUNS_DIR = RUN_CONTROL_DIR / "dropped"
OPERATOR_EVENTS_FILE = RUN_CONTROL_DIR / "operator_events.jsonl"
SCHEDULER_LOCK_FILE = RUN_CONTROL_DIR / "scheduler.lock"
JOBS_PER_GPU = max(
    1, int(os.environ.get("THESIS_SMA_JOBS_PER_GPU", "3"))
)
BATCH64_JOBS_PER_GPU = max(
    1,
    min(
        JOBS_PER_GPU,
        int(os.environ.get("THESIS_SMA_BATCH64_JOBS_PER_GPU", "2")),
    ),
)
def configure_base_runner() -> None:
    required = (
        "make_config",
        "run_main_phase",
        "run_config_on_device",
        "check_accuracy_gated_early_stopping",
        "load_active_run_records",
        "stop_active_run",
    )
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        raise SystemExit(
            "The canonical thesis_experiments.py is missing required support: "
            + ", ".join(missing)
        )
    if SMA_METHOD not in getattr(base, "METHODS", {}):
        raise SystemExit(
            f"{SMA_METHOD} is not registered in thesis_experiments.py"
        )
    base.MAIN_FDA_E_MIN = E_MIN
    base.MAIN_FDA_E_MAX = E_MAX
    base.MAIN_METHODS = [SMA_METHOD]
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
    base.main_config_priority = sma_config_priority
    base.wait_for_registered_run = wait_for_registered_sma_run
def sma_config_priority(config: dict) -> tuple:
    return (
        int(config["total_rounds"]),
        DATASET_COST_ORDER.index(config["ds_name"]),
        int(config["num_clients"]),
        int(config["seed"]),
    )
def build_sma_configs() -> list[dict]:
    configs = []
    for dataset in DATASET_COST_ORDER:
        settings = DATASET_SETTINGS[dataset]
        for num_clients in CLIENT_COUNTS:
            for seed in SEEDS:
                rounds = settings["rounds"][num_clients]
                gate = EARLY_STOPPING_GATES[dataset][num_clients]
                config = base.make_config(
                    stage=STAGE,
                    dataset=dataset,
                    method=SMA_METHOD,
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
                        "e_min": E_MIN,
                        "max_epochs": E_MAX,
                        "gamma": GAMMA,
                        "theta": THETA,
                        "sma_window": SMA_WINDOW,
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
                config["extras"] += (
                    f"-ES{gate:g}"
                    f"-MR{EARLY_STOPPING_MIN_ROUNDS}"
                    f"-P{EARLY_STOPPING_PATIENCE}"
                    f"-D{EARLY_STOPPING_MIN_DELTA:g}"
                    f"-W{EARLY_STOPPING_WINDOW}"
                )
                configs.append(config)
    return sorted(configs, key=sma_config_priority)
def validate_configs(configs: list[dict]) -> None:
    expected = (
        len(DATASET_COST_ORDER) * len(CLIENT_COUNTS) * len(SEEDS)
    )
    if len(configs) != expected:
        raise RuntimeError(f"Expected {expected} SMA configs, found {len(configs)}")
    run_ids = [config["extras"] for config in configs]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("Duplicate SMA experiment identifiers detected")
    for config in configs:
        if config["METHOD_NAME"] != SMA_METHOD:
            raise RuntimeError("A non-SMA method entered the SMA matrix")
        if config["seed"] not in SEEDS:
            raise RuntimeError("Unexpected seed in SMA matrix")
        if config["clients_per_round"] != config["num_clients"]:
            raise RuntimeError("SMA main replication must use full participation")
        if config["e_min"] != E_MIN or config["max_epochs"] != E_MAX:
            raise RuntimeError("Unexpected adaptive epoch window")
        if config["sma_window"] != SMA_WINDOW:
            raise RuntimeError("Unexpected SMA window")
    priorities = [sma_config_priority(config) for config in configs]
    if priorities != sorted(priorities):
        raise RuntimeError("SMA configurations are not cost ordered")
def dropped_marker_path(run_id: str) -> Path:
    return DROPPED_RUNS_DIR / f"{base.get_control_id(run_id)}.json"
def is_dropped(config: dict) -> bool:
    return dropped_marker_path(config["extras"]).exists()
def print_plan(configs: list[dict]) -> None:
    complete = sum(base.is_completed(config) for config in configs)
    dropped = sum(is_dropped(config) for config in configs)
    active_ids = {
        record["run_id"] for record in base.load_active_run_records()
    }
    active = sum(config["extras"] in active_ids for config in configs)
    print("\nFDA-SMA W=3 MAIN REPLICATION")
    print("=" * 88)
    print(
        f"Configurations: {len(configs)} "
        f"(complete={complete}, active={active}, dropped={dropped})"
    )
    print("Method: FDA-RoLoRA-DualTrack-SMA")
    print("Window: E3--E20; SMA W=3; gamma=0.3; theta=0")
    print("Model: roberta-base; rank=4; LoRA alpha=8; query+value")
    print("Clients: C20/K20 and C50/K50; seeds: 42,43; Dirichlet alpha=1000")
    print(
        "Stopping: minimum 6 rounds; patience 3; min_delta 0.5%; "
        "accuracy window 3"
    )
    print(
        f"Scheduler: {JOBS_PER_GPU} jobs/GPU; "
        f"batch-64 cap {BATCH64_JOBS_PER_GPU}/GPU"
    )
    print("Queue order: total rounds first, then estimated dataset cost, then seed")
    print("-" * 88)
    for index, config in enumerate(configs, start=1):
        state = "DONE" if base.is_completed(config) else "DROP" if is_dropped(config) else "TODO"
        print(
            f"{index:02d} {state:4s} "
            f"{config['ds_name'].upper():5s} C{config['num_clients']:<2d} "
            f"R{config['total_rounds']:<2d} S{config['seed']} "
            f"B{config['batch_size']:<2d} "
            f"gate={config['early_stopping_start_accuracy']:.0%}"
        )
    print("=" * 88)
def wait_for_registered_sma_run(
    config: dict | None,
    record: dict,
    poll_seconds: int = 5,
) -> bool:
    run_id = record["run_id"]
    control_id = record["control_id"]
    print(
        f"ADOPT [{control_id}] pid={record['pid']} {run_id} "
        f"on {record['device']}"
    )
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
        print(f"RELEASE [{control_id}] detached SMA worker finished")
        return True
    if base.is_completed(config):
        final_metric = base.read_latest_metric_line(record["stdout_path"])
        print(f"ADOPTED DONE [{control_id}] {run_id}; {final_metric}")
        return True
    print(
        f"ADOPTED ERROR [{control_id}] {run_id}: worker exited without "
        "a readable metrics CSV"
    )
    return False
def list_runs() -> None:
    records = base.load_active_run_records()
    if not records:
        print("No active FDA-SMA W=3 runs.")
        return
    print(f"Active FDA-SMA W=3 runs: {len(records)}")
    for record in records:
        elapsed = base.format_elapsed(
            time.time() - float(record["started_at"])
        )
        latest = base.read_latest_metric_line(record["stdout_path"])
        print(
            f"[{record['control_id']}] pid={record['pid']} "
            f"gpu={record['device']} elapsed={elapsed}"
        )
        print(f"  run:    {record['run_id']}")
        print(f"  latest: {latest}")
        print(f"  log:    {record['stdout_path']}")
    print("Stop one attempt: ./run_thesis_sma_experiments.sh stop <ID>")
    print("Stop and skip permanently: ./run_thesis_sma_experiments.sh drop <ID>")
def drop_active_run(selector: str, reason: str) -> None:
    record = base.resolve_active_run(selector)
    marker = {
        "action": "operator_drop",
        "control_id": record["control_id"],
        "reason": reason,
        "requested_at": time.time(),
        "run_id": record["run_id"],
    }
    base.atomic_write_json(dropped_marker_path(record["run_id"]), marker)
    base.stop_active_run(selector, reason)
    print("This configuration is marked dropped and will not restart on resume.")
def list_dropped_runs() -> None:
    markers = (
        sorted(DROPPED_RUNS_DIR.glob("*.json"))
        if DROPPED_RUNS_DIR.exists()
        else []
    )
    if not markers:
        print("No FDA-SMA configurations are marked dropped.")
        return
    print(f"Dropped FDA-SMA configurations: {len(markers)}")
    for path in markers:
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        print(f"[{marker['control_id']}] {marker['run_id']}")
        print(f"  reason: {marker['reason']}")
def restore_dropped_run(selector: str) -> None:
    matches = []
    if DROPPED_RUNS_DIR.exists():
        for path in DROPPED_RUNS_DIR.glob("*.json"):
            try:
                marker = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                marker["control_id"].startswith(selector)
                or marker["run_id"].startswith(selector)
            ):
                matches.append((path, marker))
    if len(matches) != 1:
        if not matches:
            raise SystemExit(f"No dropped SMA run matches {selector!r}")
        ids = ", ".join(marker["control_id"] for _, marker in matches)
        raise SystemExit(f"Ambiguous dropped-run selector {selector!r}: {ids}")
    path, marker = matches[0]
    path.unlink()
    print(f"Restored [{marker['control_id']}] {marker['run_id']}")
def run_all(configs: list[dict]) -> None:
    runnable = [config for config in configs if not is_dropped(config)]
    RUN_CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    with SCHEDULER_LOCK_FILE.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(
                "Another FDA-SMA scheduler is already active. "
                "Use the status command instead of starting a second one."
            ) from error
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        print_plan(configs)
        base.run_main_phase(runnable)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        required=True,
        choices=("plan", "run", "status", "stop", "drop", "dropped", "restore"),
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--reason",
        default="operator stopped FDA-SMA run after inspecting live metrics",
    )
    return parser.parse_args()
def main() -> None:
    configure_base_runner()
    configs = build_sma_configs()
    validate_configs(configs)
    args = parse_args()
    if args.group == "plan":
        print_plan(configs)
    elif args.group == "run":
        run_all(configs)
    elif args.group == "status":
        list_runs()
    elif args.group == "dropped":
        list_dropped_runs()
    elif args.group in {"stop", "drop", "restore"}:
        if not args.run_id:
            raise SystemExit(f"--run-id is required for {args.group}")
        if args.group == "stop":
            base.stop_active_run(args.run_id, args.reason)
        elif args.group == "drop":
            drop_active_run(args.run_id, args.reason)
        else:
            restore_dropped_run(args.run_id)
if __name__ == "__main__":
    main()

```


## File: `run_threshold_experiments.py`

```py
"""Run RoLoRA vs. FDA-RoLoRA Threshold Controller Experiments (kappa=0.5 vs. kappa=1.0).
Experimental Protocol:
- Baselines:
    RoLoRA fixed local epochs E in {15, 20}
- FDA-RoLoRA Dynamic Windows:
    Paired with baseline*2 -> E in [3, 30] (for E=15) and E in [3, 40] (for E=20)
    Controller Setpoints:
      * kappa = 0.5 (E_max / 2, standard midpoint threshold)
      * kappa = 1.0 (E_max / 1, full upper threshold — executed at the VERY END)
- Datasets:
    Primary: MNLI (default)
    Optional: QNLI (via --dataset qnli or --dataset all)
- Client Scales:
    Primary: C=20, R=50 (default)
    Optional: C=50, R=30 (via --clients 50 or --clients all)
- Learning Rate:
    Literature consensus eta = 0.1 (client SGD), server SGD (lr = 1.0)
- Strict Early Stopping:
    Acc-gated (ES0.60 for MNLI, ES0.80 for QNLI), MR=6, P=3, D=0.005 (0.5%), W=3
- Execution Ordering:
    Stage 1: RoLoRA baselines (E=15, E=20)
    Stage 2: FDA-RoLoRA kappa = 0.5 ([3, 30], [3, 40])
    Stage 3: FDA-RoLoRA kappa = 1.0 ([3, 30], [3, 40]) -> at the VERY END
"""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR if (CURRENT_DIR / "thesis_experiments.py").exists() else CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import thesis_experiments as te
CLIENT_LEARNING_RATE = 0.1
SERVER_LEARNING_RATE = 1.0
DATASETS = ("mnli", "qnli")
CLIENT_SETTINGS = {
    20: 50,
    50: 30,
}
DEFAULT_SEEDS = (42, 43, 44)
ALPHA = 1000.0
EPOCH_TIERS = [
    {"baseline_epochs": 15, "fda_e_min": 3, "fda_e_max": 30},
    {"baseline_epochs": 20, "fda_e_min": 3, "fda_e_max": 40},
]
STRICT_EARLY_STOPPING_PARAMS = {
    "early_stopping": True,
    "early_stopping_min_rounds": 6,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.005,
    "early_stopping_window": 3,
    "early_stopping_gates": {
        "mnli": 0.60,
        "qnli": 0.80,
    },
}
P2_LAYERS = list(range(7, 12))
def build_experiment_configs(
    stage_filter: str = "all",
    dataset_filter: str = "mnli",
    clients_filter: str = "20",
    seeds_filter: str = "42,43,44",
    learning_rate: float = CLIENT_LEARNING_RATE,
) -> list[dict]:
    """Build the ordered experiment queue adhering strictly to the staging constraints:
    Stage 1: RoLoRA baselines (E=15, 20)
    Stage 2: FDA-RoLoRA with kappa = 0.5 (E_max / 2)
    Stage 3: FDA-RoLoRA with kappa = 1.0 (E_max / 1) -> at the VERY END
    """
    datasets = (
        DATASETS
        if dataset_filter == "all"
        else [ds.strip().lower() for ds in dataset_filter.split(",")]
    )
    client_counts = (
        list(CLIENT_SETTINGS.keys())
        if clients_filter == "all"
        else [int(c.strip()) for c in clients_filter.split(",")]
    )
    seeds = [int(s.strip()) for s in seeds_filter.split(",")]
    stage1_rolora: list[dict] = []
    stage2_fda_k05: list[dict] = []
    stage3_fda_k1: list[dict] = []
    for dataset in datasets:
        if dataset not in te.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        gate = STRICT_EARLY_STOPPING_PARAMS["early_stopping_gates"][dataset]
        mr = STRICT_EARLY_STOPPING_PARAMS["early_stopping_min_rounds"]
        p = STRICT_EARLY_STOPPING_PARAMS["early_stopping_patience"]
        d = STRICT_EARLY_STOPPING_PARAMS["early_stopping_min_delta"]
        w = STRICT_EARLY_STOPPING_PARAMS["early_stopping_window"]
        es_tag = f"-ES{gate:g}-MR{mr}-P{p}-D{d:g}-W{w}"
        for clients in client_counts:
            rounds = CLIENT_SETTINGS[clients]
            for tier in EPOCH_TIERS:
                base_e = tier["baseline_epochs"]
                fda_min = tier["fda_e_min"]
                fda_max = tier["fda_e_max"]
                for seed in seeds:
                    rolora_extras = (
                        f"THESIS-{te.MODEL_SLUG}-MAIN-DS-{dataset}"
                        f"-C{clients}-R{rounds}-M-rolora"
                        f"-LR{learning_rate:g}-RK4-S{seed}-E{base_e}{es_tag}"
                    )
                    rolora_cfg = te.make_config(
                        stage="main",
                        dataset=dataset,
                        method="FedAvg-RoLoRA",
                        seed=seed,
                        num_clients=clients,
                        total_rounds=rounds,
                        alpha=ALPHA,
                        learning_rate=learning_rate,
                        rank=4,
                        layers=copy.deepcopy(P2_LAYERS),
                        batch_size=32,
                        clients_per_round=clients,
                        local_epochs=base_e,
                        lora_alpha=8,
                        target_modules=["query", "value"],
                    )
                    rolora_cfg.update(
                        {
                            "server_opt_name": "SGD",
                            "server_opt_lr": SERVER_LEARNING_RATE,
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
                        }
                    )
                    stage1_rolora.append(rolora_cfg)
                    fda_k05_extras = (
                        f"THESIS-{te.MODEL_SLUG}-MAIN-DS-{dataset}"
                        f"-C{clients}-R{rounds}-M-fda-k05"
                        f"-LR{learning_rate:g}-RK4-E{fda_min}TO{fda_max}"
                        f"-S{seed}{es_tag}"
                    )
                    fda_k05_cfg = te.make_config(
                        stage="main",
                        dataset=dataset,
                        method="FDA-RoLoRA-DualTrack",
                        seed=seed,
                        num_clients=clients,
                        total_rounds=rounds,
                        alpha=ALPHA,
                        learning_rate=learning_rate,
                        rank=4,
                        layers=copy.deepcopy(P2_LAYERS),
                        batch_size=32,
                        clients_per_round=clients,
                        local_epochs=1,
                        lora_alpha=8,
                        target_modules=["query", "value"],
                    )
                    fda_k05_cfg.update(
                        {
                            "server_opt_name": "SGD",
                            "server_opt_lr": SERVER_LEARNING_RATE,
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
                        }
                    )
                    stage2_fda_k05.append(fda_k05_cfg)
                    fda_k1_extras = (
                        f"THESIS-{te.MODEL_SLUG}-MAIN-DS-{dataset}"
                        f"-C{clients}-R{rounds}-M-fda-k1"
                        f"-LR{learning_rate:g}-RK4-E{fda_min}TO{fda_max}"
                        f"-S{seed}{es_tag}"
                    )
                    fda_k1_cfg = te.make_config(
                        stage="main",
                        dataset=dataset,
                        method="FDA-RoLoRA-DualTrack",
                        seed=seed,
                        num_clients=clients,
                        total_rounds=rounds,
                        alpha=ALPHA,
                        learning_rate=learning_rate,
                        rank=4,
                        layers=copy.deepcopy(P2_LAYERS),
                        batch_size=32,
                        clients_per_round=clients,
                        local_epochs=1,
                        lora_alpha=8,
                        target_modules=["query", "value"],
                    )
                    fda_k1_cfg.update(
                        {
                            "server_opt_name": "SGD",
                            "server_opt_lr": SERVER_LEARNING_RATE,
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
                        }
                    )
                    stage3_fda_k1.append(fda_k1_cfg)
    if stage_filter == "rolora":
        return stage1_rolora
    elif stage_filter == "fda_k05":
        return stage2_fda_k05
    elif stage_filter == "fda_k1":
        return stage3_fda_k1
    elif stage_filter == "fda_all":
        return stage2_fda_k05 + stage3_fda_k1
    elif stage_filter == "all":
        return stage1_rolora + stage2_fda_k05 + stage3_fda_k1
    else:
        raise ValueError(f"Unknown stage filter: {stage_filter}")
def execute_run(config_json: str):
    """Child worker execution target with dynamic dispatch for RoLoRA and Calibrated FDA."""
    import torch
    from fdaopt import utils
    from fdaopt.training import fda_opt, fed_opt
    config = json.loads(config_json)
    method = config["METHOD_NAME"]
    device = config.pop("TARGET_DEVICE", "cuda:0")
    te.install_runtime_patches(config)
    torch.cuda.set_device(torch.device(device))
    utils.DEVICE_RAM_PROGRAM = "performance"
    utils.set_device(device)
    utils.set_seed(config["seed"])
    print(f"[{time.strftime('%H:%M:%S')}] Launching {config['extras']} on {device}...")
    if method == "FedAvg-RoLoRA":
        fed_opt.fedavg_rolora(config)
    elif method == "FDA-RoLoRA-DualTrack":
        if hasattr(fda_opt, "fda_rolora_dual_track_calibrated_opt"):
            fda_opt.fda_rolora_dual_track_calibrated_opt(config)
        elif config.get("controller_ratio", 0.5) == 0.5:
            fda_opt.fda_rolora_dual_track_opt(config)
        else:
            raise RuntimeError(
                "fda_opt.fda_rolora_dual_track_calibrated_opt not found for controller_ratio != 0.5"
            )
    else:
        raise ValueError(f"Unsupported method in threshold controller runner: {method}")
    print(f"[{time.strftime('%H:%M:%S')}] Execution finished for {config['extras']}.")
def run_config_on_device(config: dict, device: str) -> bool:
    """Execute one configuration as an isolated child process with full logging and control."""
    run_id = config["extras"]
    temporary_dir = None
    registry_path = None
    request_path = None
    try:
        if te.is_completed(config):
            print(f"SKIP {run_id}")
            return True
        run_config = dict(config)
        run_config["TARGET_DEVICE"] = device
        te.LOG_DIR.mkdir(parents=True, exist_ok=True)
        stdout_path = te.LOG_DIR / f"{run_id}.out"
        stderr_path = te.LOG_DIR / f"{run_id}.err"
        temp_base = "/tmp" if os.path.exists("/tmp") else tempfile.gettempdir()
        temporary_dir = tempfile.mkdtemp(prefix=f"{run_id}-", dir=temp_base)
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
            control_id = te.get_control_id(run_id)
            registry_path = te.active_registry_path(control_id)
            request_path = te.stop_request_path(control_id)
            request_path.unlink(missing_ok=True)
            record = {
                "batch_size": int(config["batch_size"]),
                "control_id": control_id,
                "device": device,
                "pid": process.pid,
                "process_start_ticks": te.process_start_ticks(process.pid),
                "run_id": run_id,
                "started_at": time.time(),
                "stderr_path": str(stderr_path.resolve()),
                "stdout_path": str(stdout_path.resolve()),
            }
            te.atomic_write_json(registry_path, record)
            print(f"START [{control_id}] pid={process.pid} {run_id} on {device}")
            return_code = process.wait()
        if request_path.exists():
            print(f"STOPPED [{control_id}] {run_id} by operator request")
            return True
        if return_code != 0:
            print(f"ERROR {run_id}: see {stderr_path}")
            return False
        final_metric = te.read_latest_metric_line(stdout_path)
        print(f"DONE [{control_id}] {run_id}; {final_metric}")
        return True
    finally:
        if registry_path is not None:
            registry_path.unlink(missing_ok=True)
        if request_path is not None:
            request_path.unlink(missing_ok=True)
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
def run_configs_pool(
    configs: list[dict],
    devices: list[str],
    jobs_per_gpu: int,
) -> None:
    """Execute queue across GPUs with dynamic lane replenishment."""
    pending = [c for c in configs if not te.is_completed(c)]
    if not pending:
        print("All configurations already completed! Nothing to run.")
        return
    lanes = [
        (device, lane_idx)
        for device in devices
        for lane_idx in range(jobs_per_gpu)
    ]
    print(f"\nExecuting {len(pending)} pending runs on {len(devices)} GPUs with "
          f"{jobs_per_gpu} lanes/GPU (Total concurrent: {len(lanes)})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        future_to_lane = {}
        idle_lanes = list(lanes)
        config_queue = list(pending)
        def launch_available():
            made_progress = True
            while made_progress and idle_lanes and config_queue:
                made_progress = False
                for lane in list(idle_lanes):
                    if not config_queue:
                        break
                    device, lane_idx = lane
                    config = config_queue.pop(0)
                    future = executor.submit(run_config_on_device, config, device)
                    future_to_lane[future] = (lane, config)
                    idle_lanes.remove(lane)
                    made_progress = True
        launch_available()
        while future_to_lane:
            completed, _ = concurrent.futures.wait(
                future_to_lane,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in completed:
                lane, completed_config = future_to_lane.pop(future)
                idle_lanes.append(lane)
                try:
                    success = future.result()
                    if not success:
                        print(f"[FAILED] {completed_config['extras']} on GPU {lane[0]} (Lane {lane[1]})")
                except Exception as e:
                    print(f"[EXCEPTION] {completed_config['extras']} on GPU {lane[0]}: {e}")
            launch_available()
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RoLoRA vs. FDA-RoLoRA Threshold Controller Experiments (kappa=0.5 vs kappa=1.0)"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "rolora", "fda_k05", "fda_k1", "fda_all"],
        default="all",
        help="Stage filter: 'all' (ordered: rolora -> fda_k05 -> fda_k1), "
             "'rolora' (only baseline fixed epochs 15,20), "
             "'fda_k05' (only FDA kappa=0.5 / Emax/2), "
             "'fda_k1' (only FDA kappa=1.0 / Emax/1), "
             "'fda_all' (both FDA stages)",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "mnli", "qnli"],
        default="mnli",
        help="Dataset: 'mnli' (default), 'qnli', or 'all'",
    )
    parser.add_argument(
        "--clients",
        choices=["all", "20", "50"],
        default="20",
        help="Client scale: '20' (default C=20, R=50), '50' (C=50, R=30), or 'all'",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43",
        help="Comma-separated seed list (default: '42,43', or e.g. '42,43,44' / '42')",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=CLIENT_LEARNING_RATE,
        help=f"Client learning rate (default: {CLIENT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU device IDs (e.g. '0,1' or '0')",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=3,
        help="Maximum concurrent jobs per GPU (default: 3)",
    )
    parser.add_argument(
        "--stage-barrier",
        action="store_true",
        help="Enforce strict barrier wait between stages (leaves lanes idle if stage runs < available lanes)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="List all currently active runs, GPU allocations, and live metrics",
    )
    parser.add_argument(
        "--stop",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Stop/kill a specific active worker by its control_id, PID, or run_id",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned run queue and ordering without executing",
    )
    parser.add_argument(
        "--run-config",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()
def main():
    args = parse_args()
    if args.run_config:
        execute_run(args.run_config)
        return
    if args.status:
        te.list_active_runs()
        return
    if args.stop:
        te.stop_active_run(args.stop, reason="operator stopped run via threshold controller runner")
        return
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        devices = [f"cuda:{i}" for i in gpu_ids]
    else:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(count)]
        else:
            devices = ["cuda:0"]
    configs = build_experiment_configs(
        stage_filter=args.stage,
        dataset_filter=args.dataset,
        clients_filter=args.clients,
        seeds_filter=args.seeds,
        learning_rate=args.lr,
    )
    completed = [c for c in configs if te.is_completed(c)]
    pending = [c for c in configs if not te.is_completed(c)]
    print("=" * 85)
    print("  RoLoRA vs. FDA-RoLoRA Threshold Controller Experiments (kappa=0.5 vs kappa=1.0)")
    print("=" * 85)
    print(f"Stage Filter:    {args.stage.upper()}")
    print(f"Datasets:        {args.dataset.upper()}")
    print(f"Client Counts:   {args.clients}")
    print(f"Seeds:           {args.seeds}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Devices:         {devices} ({len(devices)} GPUs)")
    print(f"Lanes per GPU:   {args.jobs_per_gpu} (Max concurrent: {len(devices) * args.jobs_per_gpu})")
    print(f"Queue Mode:      {'Strict Stage Barrier' if args.stage_barrier else 'Continuous Dynamic Pool (100% GPU Saturation)'}")
    print(f"Early Stopping:  MR={STRICT_EARLY_STOPPING_PARAMS['early_stopping_min_rounds']}, "
          f"P={STRICT_EARLY_STOPPING_PARAMS['early_stopping_patience']}, "
          f"D={STRICT_EARLY_STOPPING_PARAMS['early_stopping_min_delta']:.1%}, "
          f"W={STRICT_EARLY_STOPPING_PARAMS['early_stopping_window']}")
    print("-" * 85)
    print(f"Total Planned:   {len(configs)} runs")
    print(f"Already Done:    {len(completed)} runs")
    print(f"To Execute:      {len(pending)} runs")
    print("=" * 85)
    if args.dry_run:
        print("\n[DRY RUN] Queue List (Strict Planned Execution Order):")
        current_stage = None
        for idx, c in enumerate(configs, 1):
            stage_label = c.get("_stage_label", "")
            if stage_label != current_stage:
                print(f"\n--- {stage_label} ---")
                current_stage = stage_label
            status = "DONE" if te.is_completed(c) else "PENDING"
            r = c["total_rounds"]
            e = f"E{c['local_epochs']}" if c["METHOD_NAME"] == "FedAvg-RoLoRA" else f"E{c['e_min']}-{c['max_epochs']}"
            ratio = c.get("controller_ratio", "-")
            ratio_str = f"k={ratio}" if ratio != "-" else "fixed"
            print(f"  {idx:02d}. [{status}] [{c['ds_name'].upper()} C{c['num_clients']:02d} R{r:02d} S{c['seed']}] [{e:>6}] [{ratio_str:>5}] {c['extras']}")
        return
    if not pending:
        print("All configurations already completed! Nothing to run.")
        return
    if args.stage_barrier and args.stage == "all":
        stages = [
            ("Stage 1: RoLoRA Baseline", "rolora"),
            ("Stage 2: FDA-RoLoRA (kappa=0.5, Emax/2)", "fda_k05"),
            ("Stage 3: FDA-RoLoRA (kappa=1.0, Emax/1) [AT THE VERY END]", "fda_k1"),
        ]
        for stage_title, stage_name in stages:
            stage_configs = build_experiment_configs(
                stage_filter=stage_name,
                dataset_filter=args.dataset,
                clients_filter=args.clients,
                seeds_filter=args.seeds,
                learning_rate=args.lr,
            )
            stage_pending = [c for c in stage_configs if not te.is_completed(c)]
            if stage_pending:
                print("\n" + "=" * 85)
                print(f"  STARTING {stage_title.upper()} ({len(stage_pending)} runs pending)")
                print("=" * 85)
                run_configs_pool(stage_configs, devices, args.jobs_per_gpu)
                print(f"\n>>> FINISHED {stage_title.upper()}.\n")
            else:
                print(f"All runs already completed for {stage_title}.")
    else:
        print(f"\nStarting execution of {len(pending)} runs with continuous dynamic pool (100% GPU saturation)...")
        run_configs_pool(configs, devices, args.jobs_per_gpu)
if __name__ == "__main__":
    main()

```


## File: `run_lr001_mnli_qnli.py`

```py
"""Run the LR=0.01 RoLoRA vs. FDA-RoLoRA (Old Controller, Midpoint Target) Experiments.
Focuses exclusively on:
- Tasks: MNLI, QNLI
- Methods: RoLoRA vs. FDA-RoLoRA (DualTrack, Old Controller, CRATIO=0.5)
- Budget: P2 only (layers 7-11 for RoBERTa-base)
- Epoch Tiers: E=5, E=10, E=20 (paired with FDA [3, 10], [3, 20], [3, 40])
- Client Settings: C=20 (R=50), C=50 (R=30)
- Learning Rate: 0.01 (client SGD), 1.0 (server SGD)
- Seeds: 42, 43, 44
- Early Stopping: Lighter configuration (P=5, D=0.002, MR=10, W=3)
- Dynamic Run Scheduling: 3 concurrent jobs/GPU asynchronously refilled upon completion
- Cheapest Runs First: Orders queue by estimated compute cost (fewer rounds/epochs first)
Usage:
    python run_lr001_mnli_qnli.py
    python run_lr001_mnli_qnli.py --clients 20
    python run_lr001_mnli_qnli.py --dataset mnli
    python run_lr001_mnli_qnli.py --dataset mnli --clients 20
    python run_lr001_mnli_qnli.py --dry-run
    python run_lr001_mnli_qnli.py --gpus 0,1
"""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import os
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR if (CURRENT_DIR / "thesis_experiments.py").exists() else CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import thesis_experiments as te
CLIENT_LEARNING_RATE = 0.01
SERVER_LEARNING_RATE = 1.0
DATASETS = ("mnli", "qnli")
CLIENT_SETTINGS = {
    20: 50,
    50: 30,
}
SEEDS = (42, 43, 44)
ALPHA = 1000.0
EPOCH_TIERS = [
    {"baseline_epochs": 5,  "fda_e_min": 3, "fda_e_max": 10},
    {"baseline_epochs": 10, "fda_e_min": 3, "fda_e_max": 20},
    {"baseline_epochs": 20, "fda_e_min": 3, "fda_e_max": 40},
]
EARLY_STOPPING_PARAMS = {
    "early_stopping": True,
    "early_stopping_min_rounds": 10,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.002,
    "early_stopping_window": 3,
    "early_stopping_gates": {
        "mnli": 0.60,
        "qnli": 0.80,
    },
}
P2_LAYERS = list(range(7, 12))
def build_experiment_configs(
    dataset_filter: str = "all",
    clients_filter: str = "all",
    sort_by_cost: bool = True,
) -> list[dict]:
    """Build the exact queue of experiment configs for LR=0.01 using the server schema."""
    datasets = (
        DATASETS
        if dataset_filter == "all"
        else [ds.strip().lower() for ds in dataset_filter.split(",")]
    )
    client_counts = (
        list(CLIENT_SETTINGS.keys())
        if clients_filter == "all"
        else [int(c.strip()) for c in clients_filter.split(",")]
    )
    configs: list[dict] = []
    for dataset in datasets:
        if dataset not in te.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        gate = EARLY_STOPPING_PARAMS["early_stopping_gates"][dataset]
        mr = EARLY_STOPPING_PARAMS["early_stopping_min_rounds"]
        p = EARLY_STOPPING_PARAMS["early_stopping_patience"]
        d = EARLY_STOPPING_PARAMS["early_stopping_min_delta"]
        w = EARLY_STOPPING_PARAMS["early_stopping_window"]
        es_tag = f"-ES{gate:g}-MR{mr}-P{p}-D{d:g}-W{w}"
        for clients in client_counts:
            rounds = CLIENT_SETTINGS[clients]
            for tier in EPOCH_TIERS:
                base_e = tier["baseline_epochs"]
                fda_min = tier["fda_e_min"]
                fda_max = tier["fda_e_max"]
                for seed in SEEDS:
                    rolora_extras = (
                        f"THESIS-{te.MODEL_SLUG}-MAIN-DS-{dataset}"
                        f"-C{clients}-R{rounds}-M-rolora"
                        f"-LR{CLIENT_LEARNING_RATE:g}-RK4-S{seed}-E{base_e}{es_tag}"
                    )
                    rolora_cfg = te.make_config(
                        stage="main",
                        dataset=dataset,
                        method="FedAvg-RoLoRA",
                        seed=seed,
                        num_clients=clients,
                        total_rounds=rounds,
                        alpha=ALPHA,
                        learning_rate=CLIENT_LEARNING_RATE,
                        rank=4,
                        layers=copy.deepcopy(P2_LAYERS),
                        batch_size=32,
                        clients_per_round=clients,
                        local_epochs=base_e,
                        lora_alpha=8,
                        target_modules=["query", "value"],
                    )
                    rolora_cfg.update(
                        {
                            "server_opt_name": "SGD",
                            "server_opt_lr": SERVER_LEARNING_RATE,
                            "client_opt_name": "SGD",
                            "client_opt_lr": CLIENT_LEARNING_RATE,
                            "early_stopping": True,
                            "early_stopping_start_accuracy": gate,
                            "plateau_min_accuracy": gate,
                            "early_stopping_min_rounds": mr,
                            "early_stopping_patience": p,
                            "early_stopping_min_delta": d,
                            "early_stopping_window": w,
                            "paper_parameter_budget": "P2",
                            "extras": rolora_extras,
                        }
                    )
                    configs.append(rolora_cfg)
                    fda_extras = (
                        f"THESIS-{te.MODEL_SLUG}-MAIN-DS-{dataset}"
                        f"-C{clients}-R{rounds}-M-fda-standard"
                        f"-LR{CLIENT_LEARNING_RATE:g}-RK4-E{fda_min}TO{fda_max}"
                        f"-S{seed}{es_tag}"
                    )
                    fda_cfg = te.make_config(
                        stage="main",
                        dataset=dataset,
                        method="FDA-RoLoRA-DualTrack",
                        seed=seed,
                        num_clients=clients,
                        total_rounds=rounds,
                        alpha=ALPHA,
                        learning_rate=CLIENT_LEARNING_RATE,
                        rank=4,
                        layers=copy.deepcopy(P2_LAYERS),
                        batch_size=32,
                        clients_per_round=clients,
                        local_epochs=1,
                        lora_alpha=8,
                        target_modules=["query", "value"],
                    )
                    fda_cfg.update(
                        {
                            "server_opt_name": "SGD",
                            "server_opt_lr": SERVER_LEARNING_RATE,
                            "client_opt_name": "SGD",
                            "client_opt_lr": CLIENT_LEARNING_RATE,
                            "e_min": fda_min,
                            "max_epochs": fda_max,
                            "gamma": 0.3,
                            "theta": 0.0,
                            "controller_type": "standard",
                            "controller_ratio": 0.5,
                            "early_stopping": True,
                            "early_stopping_start_accuracy": gate,
                            "plateau_min_accuracy": gate,
                            "early_stopping_min_rounds": mr,
                            "early_stopping_patience": p,
                            "early_stopping_min_delta": d,
                            "early_stopping_window": w,
                            "paper_parameter_budget": "P2",
                            "extras": fda_extras,
                        }
                    )
                    configs.append(fda_cfg)
    if sort_by_cost:
        def est_cost(c: dict) -> float:
            r = c["total_rounds"]
            if c["METHOD_NAME"] == "FDA-RoLoRA-DualTrack":
                e = (c["e_min"] + c["max_epochs"]) / 2.0
            else:
                e = float(c["local_epochs"])
            return r * e
        configs.sort(
            key=lambda c: (
                est_cost(c),
                c["total_rounds"],
                c["local_epochs"],
                c["ds_name"],
                c["METHOD_NAME"],
                c["seed"],
            )
        )
    return configs
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LR=0.01 RoLoRA vs FDA-RoLoRA Experiments (Budget P2)"
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "mnli", "qnli"],
        default="all",
        help="Filter datasets: 'all', 'mnli', 'qnli' (or comma-separated)",
    )
    parser.add_argument(
        "--clients",
        choices=["all", "20", "50"],
        default="all",
        help="Filter client counts: 'all', '20', '50' (or comma-separated)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU device IDs (e.g., '0,1' or '0,1,2,3')",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=3,
        help="Maximum concurrent jobs per GPU (default: 3 for batch_size=32)",
    )
    parser.add_argument(
        "--no-cheapest-first",
        action="store_true",
        help="Disable sorting by compute cost (run in default nested loop order)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="List all currently active runs, GPU allocations, and live metrics",
    )
    parser.add_argument(
        "--stop",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Stop/kill a specific active worker by its control_id, PID, or run_id",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing",
    )
    return parser.parse_args()
def run_configs_pool(
    configs: list[dict],
    devices: list[str],
    jobs_per_gpu: int,
) -> None:
    """Run configs asynchronously across GPUs with dynamic worker replenishment.
    Directly invokes te.run_config_on_device(config, device) to execute
    isolated child workers with full process control, metric logging, and
    cheapest-first queue progression. Bypasses the legacy rigid phase table.
    """
    pending = [c for c in configs if not te.is_completed(c)]
    if not pending:
        print("All configurations already completed! Nothing to run.")
        return
    lanes = [
        (device, lane_idx)
        for device in devices
        for lane_idx in range(jobs_per_gpu)
    ]
    print(f"Executing {len(pending)} pending runs on {len(devices)} GPUs with "
          f"{jobs_per_gpu} lanes/GPU (Total concurrent: {len(lanes)})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        future_to_lane = {}
        idle_lanes = list(lanes)
        config_queue = list(pending)
        def launch_available():
            made_progress = True
            while made_progress and idle_lanes and config_queue:
                made_progress = False
                for lane in list(idle_lanes):
                    if not config_queue:
                        break
                    device, lane_idx = lane
                    config = config_queue.pop(0)
                    future = executor.submit(te.run_config_on_device, config, device)
                    future_to_lane[future] = (lane, config)
                    idle_lanes.remove(lane)
                    made_progress = True
        launch_available()
        while future_to_lane:
            completed, _ = concurrent.futures.wait(
                future_to_lane,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in completed:
                lane, completed_config = future_to_lane.pop(future)
                idle_lanes.append(lane)
                try:
                    success = future.result()
                    if not success:
                        print(f"[FAILED] {completed_config['extras']} on GPU {lane[0]} (Lane {lane[1]})")
                except Exception as e:
                    print(f"[EXCEPTION] {completed_config['extras']} on GPU {lane[0]}: {e}")
            launch_available()
def main():
    args = parse_args()
    if args.status:
        te.list_active_runs()
        return
    if args.stop:
        te.stop_active_run(args.stop, reason="operator stopped run via lr001 runner")
        return
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        devices = [f"cuda:{i}" for i in gpu_ids]
    else:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(count)]
        else:
            devices = ["cuda:0"]
    configs = build_experiment_configs(
        dataset_filter=args.dataset,
        clients_filter=args.clients,
        sort_by_cost=not args.no_cheapest_first,
    )
    completed = [c for c in configs if te.is_completed(c)]
    pending = [c for c in configs if not te.is_completed(c)]
    print("=" * 80)
    print("  LR=0.01 RoLoRA vs. FDA-RoLoRA Runner (Budget P2, Old Controller)")
    print("=" * 80)
    print(f"Datasets:        {args.dataset}")
    print(f"Client Counts:   {args.clients}")
    print(f"Learning Rate:   {CLIENT_LEARNING_RATE}")
    print(f"Order:           {'Cheapest First (fewer rounds/epochs first)' if not args.no_cheapest_first else 'Sequential'}")
    print(f"Devices:         {devices} ({len(devices)} GPUs)")
    print(f"Lanes per GPU:   {args.jobs_per_gpu} (Max concurrent: {len(devices) * args.jobs_per_gpu})")
    print(f"Early Stopping:  MR={EARLY_STOPPING_PARAMS['early_stopping_min_rounds']}, "
          f"P={EARLY_STOPPING_PARAMS['early_stopping_patience']}, "
          f"D={EARLY_STOPPING_PARAMS['early_stopping_min_delta']:.1%}, "
          f"W={EARLY_STOPPING_PARAMS['early_stopping_window']}")
    print("-" * 80)
    print(f"Total Planned:   {len(configs)} runs")
    print(f"Already Done:    {len(completed)} runs")
    print(f"To Execute:      {len(pending)} runs")
    print("=" * 80)
    if args.dry_run:
        print("\n[DRY RUN] Queue List (Planned Execution Order):")
        for idx, c in enumerate(configs, 1):
            status = "DONE" if te.is_completed(c) else "PENDING"
            r = c["total_rounds"]
            e = f"E{c['local_epochs']}" if c["METHOD_NAME"] == "FedAvg-RoLoRA" else f"E{c['e_min']}-{c['max_epochs']}"
            print(f"  {idx:02d}. [{status}] [R={r:02d}, {e:>6}] {c['extras']}")
        return
    if not pending:
        print("All configurations already completed! Nothing to run.")
        return
    print(f"\nStarting execution of {len(pending)} runs with dynamic slot allocation...")
    run_configs_pool(configs, devices, args.jobs_per_gpu)
if __name__ == "__main__":
    main()

```


## File: `thesis_pure_experiments.py`

```py
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
DEFAULT_SEEDS = [42]
DEFAULT_CLIENT_COUNTS = [20, 50]
PARTITION_ALPHA = 1000.0
DEFAULT_E_MIN = 1
DEFAULT_E_MAX = 20
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")
EARLY_STOPPING_MIN_ROUNDS = 6
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.005
EARLY_STOPPING_WINDOW = 3
EARLY_STOPPING_GATES = {
    "qnli": {20: 0.80, 50: 0.80},
    "mnli": {20: 0.60, 50: 0.55},
}
DATASET_COST_ORDER = ("qnli", "mnli")
DATASET_SETTINGS = {
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
JOBS_PER_GPU = max(1, int(os.environ.get("THESIS_PURE_JOBS_PER_GPU", "3")))
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
    ratio_val = float(config.get("controller_ratio", 1.0))
    ratio_order = 0 if ratio_val == 1.0 else 1
    return (
        int(config["num_clients"]),
        DATASET_COST_ORDER.index(config["ds_name"]),
        ratio_order,
        int(config["seed"]),
    )
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
    raise RuntimeError("Unable to locate fda_rolora_dual_track_pure_opt!")
def execute_run(config_json: str) -> None:
    """Child worker execution target."""
    from fdaopt import utils
    from fdaopt.training import fda_opt, fed_opt
    opt_fn = resolve_pure_optimizer()
    fda_opt.fda_rolora_dual_track_pure_opt = opt_fn
    fed_opt.fda_rolora_dual_track_pure_opt = opt_fn
    config = json.loads(config_json)
    target_device = config.pop("TARGET_DEVICE", None)
    if target_device:
        utils.DEVICE = target_device
        os.environ["CUDA_VISIBLE_DEVICES"] = target_device.replace("cuda:", "")
    opt_fn(config)
def print_plan(configs: list[dict]) -> None:
    sep = "=" * 105
    print(f"\n{sep}")
    print(f"PURE FDA-RoLoRA EXPERIMENT PLAN (Total Runs: {len(configs)})")
    print(f"Features: Matrix B first | e_min=1 | Zero first-round guard | NODISC")
    print(f"Queue Priority: All C=20 queued first (slots 1-4), C=50 fills remaining slots & backfill")
    print(sep)
    header = f"{'#':<3} | {'Control ID':<10} | {'DS':<5} | {'C':<3} | {'R':<3} | {'Ratio (k)':<9} | {'E-Range':<7} | {'Start':<5} | {'Guard':<5} | {'LR':<4} | {'Gate':<5} | {'Seed':<4}"
    print(header)
    print("-" * len(header))
    current_c = None
    for idx, cfg in enumerate(configs, 1):
        if current_c is not None and cfg["num_clients"] != current_c:
            print("-" * len(header))
        current_c = cfg["num_clients"]
        cid = base.get_control_id(cfg["extras"])
        ratio_str = f"k={cfg['controller_ratio']}"
        e_range = f"E{cfg['e_min']}..{cfg['max_epochs']}"
        start_mat = cfg.get("start_matrix", "B")
        guard_str = "None" if not cfg.get("first_round_guard", False) else "Round1"
        line = (
            f"{idx:<3} | {cid:<10} | {cfg['ds_name'].upper():<5} | {cfg['num_clients']:<3} | "
            f"{cfg['total_rounds']:<3} | {ratio_str:<9} | {e_range:<7} | {start_mat:<5} | {guard_str:<5} | "
            f"{cfg['client_opt_lr']:<4.2g} | {cfg['early_stopping_start_accuracy']:<5.0%} | {cfg['seed']:<4}"
        )
        print(line)
    print(f"{sep}\n")
def run_all(configs: list[dict]) -> None:
    """Invokes base scheduler to dispatch jobs across GPUs with 6 parallel slots:
    All C=20 runs are queued first (taking slots 1-4), remaining 2 slots are filled
    with C=50, and as slots become free they immediately backfill with the remaining C=50 runs.
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
        print(f"Launching scheduler across {JOBS_PER_GPU} lanes/GPU (6 parallel slots)...")
        print("Initial allocation: 4 slots -> C=20 (Runs 1-4), 2 slots -> C=50 (Runs 5-6).")
        print("Free slots will dynamically backfill with remaining C=50 runs (Runs 7-8).\n")
        base.run_main_phase(runnable)
        print("\n=== ALL PURE FDA-RoLoRA EXPERIMENTS COMPLETED ===")
def list_runs() -> None:
    """Lists currently active runs from the registry."""
    active_records = [
        base.load_json_dict(p)
        for p in ACTIVE_RUNS_DIR.glob("*.json")
        if p.is_file()
    ]
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
        "--datasets",
        nargs="+",
        default=["qnli", "mnli"],
        help="Datasets to evaluate (default: qnli mnli)",
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

```


## File: `thesis_calibration_experiments.py`

```py
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
DEFAULT_SEEDS = [42]
DEFAULT_CLIENT_COUNTS = [50, 20]
PARTITION_ALPHA = 1000.0
DEFAULT_E_MIN = 3
DEFAULT_E_MAX = 30
E_MIN = DEFAULT_E_MIN
E_MAX = DEFAULT_E_MAX
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")
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
    ratio_val = float(config.get("controller_ratio", 0.5))
    ratio_order = 0 if ratio_val == 1.0 else (1 if ratio_val == 0.75 else 2)
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
    try:
        from fdaopt.training.fda_opt import fda_rolora_dual_track_calibrated_opt
        return fda_rolora_dual_track_calibrated_opt
    except (ImportError, AttributeError):
        pass
    try:
        from fixed_calibration.fda_rolora_dual_track_calibrated import (
            fda_rolora_dual_track_calibrated_opt,
        )
        return fda_rolora_dual_track_calibrated_opt
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from fda_rolora_dual_track_calibrated import (
            fda_rolora_dual_track_calibrated_opt,
        )
        return fda_rolora_dual_track_calibrated_opt
    except (ImportError, ModuleNotFoundError):
        pass
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
        pass
    elif args.group in {"stop", "drop", "restore"}:
        if not args.run_id:
            raise SystemExit(f"--run-id is required for {args.group}")
        if args.group == "stop":
            base.stop_active_run(args.run_id, args.reason)
        elif args.group == "drop":
            base.drop_active_run(args.run_id, args.reason)
        else:
            pass
if __name__ == "__main__":
    main()

```


## File: `parallel_evaluation.py`

```py
import os
import sys
import time
import subprocess
import argparse
import json
import concurrent.futures
import queue
import threading
from datetime import datetime
import torch
try:
    num_gpus = torch.cuda.device_count()
    AVAILABLE_DEVICES = [f'cuda:{i}' for i in range(num_gpus)] if num_gpus > 0 else ['cpu']
except Exception:
    AVAILABLE_DEVICES = ['cpu']
JOBS_PER_DEVICE = 4
MAX_CONCURRENT_JOBS = len(AVAILABLE_DEVICES) * JOBS_PER_DEVICE
progress_lock = threading.Lock()
completed_jobs = 0
total_scheduled_jobs = 0
TOY_BASE_HYPERPARAMS = {
    'ds_path': "ylecun/mnist",
    'ds_name': "mnist",
    'num_labels': 10,
    'checkpoint': "toy-model",
    'total_rounds': 100,
    'batch_size': 64,
    'lora_r': 16,
    'num_workers': 2,
    'pin_memory': True,
    'early_stopping': False,
    'threshold': False
}
SCENARIOS = {
    "5-Clients-2-Classes": {'num_clients': 5, 'clients_per_round': 5, 'alpha': 0.0},
    "10-Clients-1-Class": {'num_clients': 10, 'clients_per_round': 10, 'alpha': 0.0}
}
METHODS = {
    "FedAvg-LoRA": [{'server_opt_name': 'SGD', 'server_opt_lr': 1.0, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'local_epochs': 5}],
    "FFA-LoRA": [{'server_opt_name': 'SGD', 'server_opt_lr': 1.0, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'local_epochs': 5}],
    "FedAvg-RoLoRA": [{'server_opt_name': 'SGD', 'server_opt_lr': 1.0, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'local_epochs': 5}],
    "FDA-RoLoRA-DualTrack": [
        {'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'gamma': 0.3, 'e_min': 1, 'max_epochs': 15, 'theta': 0.0}
    ]
}
def execute_run(config_json):
    from fdaopt import utils
    from fdaopt.training.fed_opt import fedavg_rolora, fedavg_lora, ffa_lora
    from fdaopt.training.fda_opt import fda_rolora_dual_track_opt
    hyperparams = json.loads(config_json)
    method_name = hyperparams.pop('METHOD_NAME')
    target_device = hyperparams.pop('TARGET_DEVICE', 'cuda:0')
    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(target_device)
    utils.set_seed(hyperparams.get('seed', 42))
    if method_name == "FedAvg-LoRA":
        fedavg_lora(hyperparams)
    elif method_name == "FedAvg-RoLoRA":
        fedavg_rolora(hyperparams)
    elif method_name == "FFA-LoRA":
        ffa_lora(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack":
        fda_rolora_dual_track_opt(hyperparams)
    else:
        raise ValueError(f"Unknown method: {method_name}")
def _worker_task(run_config, device_queue, log_dir):
    global completed_jobs, total_scheduled_jobs
    device = device_queue.get()
    run_identifier = run_config['extras']
    out_file = os.path.join(log_dir, f"{run_identifier}.out")
    err_file = os.path.join(log_dir, f"{run_identifier}.err")
    try:
        run_config['TARGET_DEVICE'] = device
        config_str = json.dumps(run_config)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--run_config", config_str]
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        start_time = time.time()
        with open(out_file, 'w') as stdout_f, open(err_file, 'w') as stderr_f:
            subprocess.run(cmd, env=env, stdout=stdout_f, stderr=stderr_f, check=True)
        duration = (time.time() - start_time) / 60
        with progress_lock:
            completed_jobs += 1
            print(f"[{completed_jobs}/{total_scheduled_jobs}] SUCCESS | {run_identifier} ({duration:.2f}m)")
    except subprocess.CalledProcessError:
        with progress_lock:
            completed_jobs += 1
            print(f"[{completed_jobs}/{total_scheduled_jobs}] ERROR | {run_identifier} | Check {err_file}")
    finally:
        device_queue.put(device)
def orchestrate():
    global total_scheduled_jobs
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    all_runs = []
    batch_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for scen_name, scen_cfg in SCENARIOS.items():
        for method_name, method_variants in METHODS.items():
            for method_cfg in method_variants:
                for seed in [42, 43, 44]:
                    run_config = TOY_BASE_HYPERPARAMS.copy()
                    run_config.update(scen_cfg)
                    run_config.update(method_cfg)
                    run_config['seed'] = seed
                    run_config['METHOD_NAME'] = method_name
                    run_config['extras'] = f"TOY-{batch_timestamp}_{method_name}_{scen_name}_S{seed}"
                    all_runs.append(run_config)
    total_scheduled_jobs = len(all_runs)
    device_queue = queue.Queue()
    for dev in AVAILABLE_DEVICES:
        for _ in range(JOBS_PER_DEVICE):
            device_queue.put(dev)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = [executor.submit(_worker_task, run_config, device_queue, log_dir) for run_config in all_runs]
        concurrent.futures.wait(futures)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, default=None)
    args = parser.parse_args()
    if args.run_config:
        execute_run(args.run_config)
    else:
        orchestrate()

```


## File: `toy_model_simulation.py`

```py
import os
import sys
import time
import subprocess
import argparse
import json
import concurrent.futures
import queue
import threading
from datetime import datetime
import torch
try:
    num_gpus = torch.cuda.device_count()
    AVAILABLE_DEVICES = [f'cuda:{i}' for i in range(num_gpus)] if num_gpus > 0 else ['cpu']
except Exception:
    AVAILABLE_DEVICES = ['cpu']
JOBS_PER_DEVICE = 4
MAX_CONCURRENT_JOBS = len(AVAILABLE_DEVICES) * JOBS_PER_DEVICE
progress_lock = threading.Lock()
completed_jobs = 0
total_scheduled_jobs = 0
TOY_BASE_HYPERPARAMS = {
    'ds_path': "ylecun/mnist",
    'ds_name': "mnist",
    'num_labels': 10,
    'checkpoint': "toy-model",
    'total_rounds': 100,
    'batch_size': 64,
    'lora_r': 16,
    'num_workers': 2,
    'pin_memory': True,
    'early_stopping': False,
    'threshold': False
}
SCENARIOS = {
    "5-Clients-2-Classes": {'num_clients': 5, 'clients_per_round': 5, 'alpha': 0.0},
    "10-Clients-1-Class": {'num_clients': 10, 'clients_per_round': 10, 'alpha': 0.0}
}
METHODS = {
    "FDA-RoLoRA-DualTrack": [
        {'variant_name': 'Standard', 'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'gamma': 0.3, 'e_min': 1, 'max_epochs': 15, 'theta': 0.0}
    ],
    "FDA-RoLoRA-DualTrack-EMA-Standard": [
        {'variant_name': f'rho{rho}', 'rho': rho, 'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'gamma': 0.3, 'e_min': 1, 'max_epochs': 15, 'theta': 0.0}
        for rho in [0.8, 0.9]
    ],
    "FDA-RoLoRA-DualTrack-EMA-Pure": [
        {'variant_name': f'alpha{a_ema}', 'rho': 0.8, 'alpha_ema': a_ema, 'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'gamma': 0.3, 'e_min': 1, 'max_epochs': 15, 'theta': 0.0}
        for a_ema in [1.15, 1.30]
    ],
    "FDA-RoLoRA-DualTrack-SMA": [
        {'variant_name': f'window{w}', 'sma_window': w, 'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'gamma': 0.3, 'e_min': 1, 'max_epochs': 15, 'theta': 0.0}
        for w in [3, 5]
    ]
}
def execute_run(config_json):
    from fdaopt import utils
    from fdaopt.training.fed_opt import fedavg_rolora, fedavg_lora, ffa_lora
    from fdaopt.training.fda_opt import (
        fda_rolora_dual_track_opt,
        fda_rolora_dual_track_ema_standard_opt,
        fda_rolora_dual_track_ema_pure_opt,
        fda_rolora_dual_track_sma_opt
    )
    hyperparams = json.loads(config_json)
    method_name = hyperparams.pop('METHOD_NAME')
    target_device = hyperparams.pop('TARGET_DEVICE', 'cuda:0')
    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(target_device)
    utils.set_seed(hyperparams.get('seed', 42))
    if method_name == "FedAvg-LoRA":
        fedavg_lora(hyperparams)
    elif method_name == "FedAvg-RoLoRA":
        fedavg_rolora(hyperparams)
    elif method_name == "FFA-LoRA":
        ffa_lora(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack":
        fda_rolora_dual_track_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-EMA-Standard":
        fda_rolora_dual_track_ema_standard_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-EMA-Pure":
        fda_rolora_dual_track_ema_pure_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-SMA":
        fda_rolora_dual_track_sma_opt(hyperparams)
    else:
        raise ValueError(f"Unknown method: {method_name}")
def _worker_task(run_config, device_queue, log_dir):
    global completed_jobs, total_scheduled_jobs
    device = device_queue.get()
    run_identifier = run_config['extras']
    out_file = os.path.join(log_dir, f"{run_identifier}.out")
    err_file = os.path.join(log_dir, f"{run_identifier}.err")
    try:
        run_config['TARGET_DEVICE'] = device
        config_str = json.dumps(run_config)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--run_config", config_str]
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        start_time = time.time()
        with open(out_file, 'w') as stdout_f, open(err_file, 'w') as stderr_f:
            subprocess.run(cmd, env=env, stdout=stdout_f, stderr=stderr_f, check=True)
        duration = (time.time() - start_time) / 60
        with progress_lock:
            completed_jobs += 1
            print(f"[{completed_jobs}/{total_scheduled_jobs}] SUCCESS | {run_identifier} ({duration:.2f}m)")
    except subprocess.CalledProcessError:
        with progress_lock:
            completed_jobs += 1
            print(f"[{completed_jobs}/{total_scheduled_jobs}] ERROR | {run_identifier} | Check {err_file}")
    finally:
        device_queue.put(device)
def orchestrate():
    global total_scheduled_jobs
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    all_runs = []
    batch_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for linear_setting in [False, True]:
        for scen_name, scen_cfg in SCENARIOS.items():
            for method_name, method_variants in METHODS.items():
                for method_cfg in method_variants:
                    for seed in [42, 43, 44]:
                        run_config = TOY_BASE_HYPERPARAMS.copy()
                        run_config.update(scen_cfg)
                        run_config.update(method_cfg)
                        run_config['linear'] = linear_setting
                        run_config['seed'] = seed
                        run_config['METHOD_NAME'] = method_name
                        if "FDA" in method_name:
                            run_config['local_epochs'] = 1
                        lin_str = "LIN" if linear_setting else "NONLIN"
                        run_config['extras'] = f"TOY-FDA-{batch_timestamp}_{method_name}_{scen_name}_{lin_str}_S{seed}"
                        all_runs.append(run_config)
    total_scheduled_jobs = len(all_runs)
    device_queue = queue.Queue()
    for dev in AVAILABLE_DEVICES:
        for _ in range(JOBS_PER_DEVICE):
            device_queue.put(dev)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = [executor.submit(_worker_task, run_config, device_queue, log_dir) for run_config in all_runs]
        concurrent.futures.wait(futures)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, default=None)
    args = parser.parse_args()
    if args.run_config:
        execute_run(args.run_config)
    else:
        orchestrate()

```


## File: `simulation.py`

```py
import os
import sys
import time
import subprocess
import argparse
import json
import concurrent.futures
import queue
import threading
AVAILABLE_DEVICES = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
JOBS_PER_DEVICE = 2
MAX_CONCURRENT_JOBS = len(AVAILABLE_DEVICES) * JOBS_PER_DEVICE
BASE_HYPERPARAMS = {
    'ds_path': "glue",
    'total_rounds': 100,
    'batch_size': 16,
    'target_modules': ("query", "value"),
    'num_workers': 4,
    'pin_memory': True,
    'early_stopping': True,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.005,
    'early_stopping_window': 3,
    'train_head': True
}
SEEDS = [42, 43, 44]
CHECKPOINTS = ["roberta-base", "roberta-large"]
DATASETS = [
    {'ds_name': 'mnli', 'num_labels': 3},
    {'ds_name': 'sst2', 'num_labels': 2},
    {'ds_name': 'qnli', 'num_labels': 2},
    {'ds_name': 'rte', 'num_labels': 2}
]
LORA_CONFIGS = [
    {'lora_r': 4, 'lora_alpha': 8},
    {'lora_r': 8, 'lora_alpha': 16},
    {'lora_r': 16, 'lora_alpha': 32}
]
SCENARIOS = {
    "Cross-Device": {'num_clients': 50, 'clients_per_round': 10, 'alpha': 1.0},
    "Cross-Silo": {'num_clients': 10, 'clients_per_round': 10, 'alpha': 0.5}
}
FEDAVG_EPOCHS = [1, 5, 10, 20]
FDA_EPOCH_BOUNDS = [
    {'e_min': 2, 'max_epochs': 10},
    {'e_min': 2, 'max_epochs': 20}
]
CLIENT_LRS = [0.001, 0.005, 0.01]
METHODS_MECHANICS = {
    "FedAvg-LoRA": [
        {'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD'}
    ],
    "FedAvg-RoLoRA": [
        {'server_opt_name': 'SGD', 'server_opt_lr': 1.0, 'client_opt_name': 'SGD'}
    ],
    "FDA-RoLoRA-DualTrack": [
        {'variant_name': 'Standard', 'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'gamma': 0.3, 'theta': 0.0}
    ],
    "FDA-RoLoRA-DualTrack-EMA-Standard": [
        {'variant_name': f'rho{rho}', 'rho': rho, 'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'gamma': 0.3}
        for rho in [0.7, 0.8, 0.9]
    ],
    "FDA-RoLoRA-DualTrack-EMA-Pure": [
        {'variant_name': f'rho{rho}-alpha{a_ema}', 'rho': rho, 'alpha_ema': a_ema, 'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'gamma': 0.3}
        for rho in [0.7, 0.8, 0.9] for a_ema in [1.05, 1.15, 1.25]
    ],
    "FDA-RoLoRA-DualTrack-SMA": [
        {'variant_name': f'window{w}', 'sma_window': w, 'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'gamma': 0.3}
        for w in [3, 5, 10]
    ]
}
progress_lock = threading.Lock()
completed_jobs = 0
total_scheduled_jobs = 0
def execute_run(config_json):
    import torch
    from fdaopt import utils
    from fdaopt.training.fed_opt import fedavg_lora, fedavg_rolora
    from fdaopt.training.fda_opt import (
        fda_rolora_dual_track_opt,
        fda_rolora_dual_track_ema_standard_opt,
        fda_rolora_dual_track_ema_pure_opt,
        fda_rolora_dual_track_sma_opt
    )
    hyperparams = json.loads(config_json)
    method_name = hyperparams.pop('METHOD_NAME')
    run_name = hyperparams['extras']
    target_device = hyperparams.pop('TARGET_DEVICE', 'cuda:0')
    run_seed = hyperparams.pop('seed', 42)
    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(target_device)
    utils.set_seed(run_seed)
    if method_name == "FedAvg-LoRA":
        fedavg_lora(hyperparams)
    elif method_name == "FedAvg-RoLoRA":
        fedavg_rolora(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack":
        fda_rolora_dual_track_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-EMA-Standard":
        fda_rolora_dual_track_ema_standard_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-EMA-Pure":
        fda_rolora_dual_track_ema_pure_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-SMA":
        fda_rolora_dual_track_sma_opt(hyperparams)
    else:
        raise ValueError(f"Unknown method requested: {method_name}")
def _worker_task(run_config, device_queue, log_dir):
    global completed_jobs, total_scheduled_jobs
    device = device_queue.get()
    run_identifier = run_config['extras']
    out_file = os.path.join(log_dir, f"{run_identifier}.out")
    err_file = os.path.join(log_dir, f"{run_identifier}.err")
    try:
        run_config['TARGET_DEVICE'] = device
        config_str = json.dumps(run_config)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--run_config", config_str]
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        with open(out_file, 'w') as stdout_f, open(err_file, 'w') as stderr_f:
            subprocess.run(cmd, env=env, stdout=stdout_f, stderr=stderr_f, check=True)
        with progress_lock:
            completed_jobs += 1
            current = completed_jobs
        print(f"[{current}/{total_scheduled_jobs}] SUCCESS | {run_identifier}")
    except subprocess.CalledProcessError:
        with progress_lock:
            completed_jobs += 1
            current = completed_jobs
        print(f"[{current}/{total_scheduled_jobs}] ERROR | {run_identifier} | Check {err_file}")
    finally:
        time.sleep(3)
        device_queue.put(device)
def orchestrate():
    global total_scheduled_jobs
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    all_runs = []
    for chkpt in CHECKPOINTS:
        for ds in DATASETS:
            for lora_cfg in LORA_CONFIGS:
                for scen_name, scen_cfg in SCENARIOS.items():
                    for method_name, method_variants in METHODS_MECHANICS.items():
                        for method_cfg in method_variants:
                            for cl_lr in CLIENT_LRS:
                                for current_seed in SEEDS:
                                    epoch_iters = FDA_EPOCH_BOUNDS if "FDA" in method_name else [{'local_epochs': e} for e in FEDAVG_EPOCHS]
                                    for ep_cfg in epoch_iters:
                                        run_config = BASE_HYPERPARAMS.copy()
                                        run_config['checkpoint'] = chkpt
                                        run_config.update(ds)
                                        run_config.update(lora_cfg)
                                        run_config.update(scen_cfg)
                                        run_config.update(method_cfg)
                                        run_config.update(ep_cfg)
                                        run_config['client_opt_lr'] = cl_lr
                                        run_config['seed'] = current_seed
                                        run_config['METHOD_NAME'] = method_name
                                        if "FDA" in method_name:
                                            run_config['local_epochs'] = 1
                                        variant_name = run_config.pop('variant_name', None)
                                        method_str = f"{method_name}-{variant_name}" if variant_name else method_name
                                        chkpt_short = chkpt.split('-')[-1]
                                        ep_str = f"EMIN{ep_cfg.get('e_min')}-EMAX{ep_cfg.get('max_epochs')}" if "FDA" in method_name else f"E{ep_cfg.get('local_epochs')}"
                                        run_config['extras'] = f"ALLINONE_{chkpt_short}_{method_str}_{ds['ds_name']}_R{lora_cfg['lora_r']}_{scen_name}_LR{cl_lr}_{ep_str}_S{current_seed}"
                                        all_runs.append(run_config)
    total_scheduled_jobs = len(all_runs)
    device_queue = queue.Queue()
    for dev in AVAILABLE_DEVICES:
        for _ in range(JOBS_PER_DEVICE):
            device_queue.put(dev)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = [executor.submit(_worker_task, run_config, device_queue, log_dir) for run_config in all_runs]
        concurrent.futures.wait(futures)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, default=None)
    args = parser.parse_args()
    if args.run_config:
        execute_run(args.run_config)
    else:
        orchestrate()

```


## File: `roberta_large_pilot.py`

```py
import argparse
import concurrent.futures
import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
DEFAULT_METHODS = [
    "FedAvg-LoRA",
    "FedAvg-RoLoRA",
    "FFA-LoRA",
    "FDA-RoLoRA-DualTrack",
    "FDA-RoLoRA-DualTrack-EMA-Standard",
]
METHOD_CONFIGS = {
    "FedAvg-LoRA": {
        "server_opt_name": "AdamW",
        "server_opt_lr": 1e-3,
        "client_opt_name": "SGD",
        "client_opt_lr": 0.01,
        "local_epochs": 5,
    },
    "FedAvg-RoLoRA": {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
        "client_opt_lr": 0.01,
        "local_epochs": 5,
    },
    "FFA-LoRA": {
        "server_opt_name": "SGD",
        "server_opt_lr": 1.0,
        "client_opt_name": "SGD",
        "client_opt_lr": 0.01,
        "local_epochs": 5,
    },
    "FDA-RoLoRA-DualTrack": {
        "server_opt_name": "AdamW",
        "server_opt_lr": 1e-3,
        "client_opt_name": "SGD",
        "client_opt_lr": 0.01,
        "local_epochs": 1,
        "e_min": 2,
        "max_epochs": 10,
        "gamma": 0.3,
        "theta": 0.0,
    },
    "FDA-RoLoRA-DualTrack-EMA-Standard": {
        "server_opt_name": "AdamW",
        "server_opt_lr": 1e-3,
        "client_opt_name": "SGD",
        "client_opt_lr": 0.01,
        "local_epochs": 1,
        "e_min": 2,
        "max_epochs": 10,
        "rho": 0.8,
        "theta": 0.0,
    },
}
LABEL_COUNTS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
}
progress_lock = threading.Lock()
completed_jobs = 0
failed_jobs = 0
def execute_run(config_json):
    from fdaopt import utils
    from fdaopt.training.fda_opt import (
        fda_rolora_dual_track_ema_standard_opt,
        fda_rolora_dual_track_opt,
    )
    from fdaopt.training.fed_opt import fedavg_lora, fedavg_rolora, ffa_lora
    hyperparams = json.loads(config_json)
    method_name = hyperparams.pop("METHOD_NAME")
    target_device = hyperparams.pop("TARGET_DEVICE")
    utils.DEVICE_RAM_PROGRAM = "performance"
    utils.set_device(target_device)
    utils.set_seed(hyperparams["seed"])
    runners = {
        "FedAvg-LoRA": fedavg_lora,
        "FedAvg-RoLoRA": fedavg_rolora,
        "FFA-LoRA": ffa_lora,
        "FDA-RoLoRA-DualTrack": fda_rolora_dual_track_opt,
        "FDA-RoLoRA-DualTrack-EMA-Standard": fda_rolora_dual_track_ema_standard_opt,
    }
    runners[method_name](hyperparams)
def run_worker(run_config, device_queue, log_dir, total_jobs):
    global completed_jobs, failed_jobs
    device = device_queue.get()
    run_identifier = run_config["extras"]
    stdout_path = log_dir / f"{run_identifier}.out"
    stderr_path = log_dir / f"{run_identifier}.err"
    success = False
    start_time = time.time()
    try:
        child_config = copy.deepcopy(run_config)
        child_config["TARGET_DEVICE"] = device
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--run-config",
            json.dumps(child_config),
        ]
        env = os.environ.copy()
        env.update(
            {
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
            subprocess.run(
                command,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                check=True,
            )
        success = True
    except subprocess.CalledProcessError:
        pass
    except Exception as error:
        stderr_path.write_text(f"Pilot launcher error: {error}\n")
    finally:
        device_queue.put(device)
        duration_minutes = (time.time() - start_time) / 60
        with progress_lock:
            completed_jobs += 1
            if not success:
                failed_jobs += 1
            status = "SUCCESS" if success else "ERROR"
            print(
                f"[{completed_jobs}/{total_jobs}] {status} | {device} | "
                f"{run_identifier} | {duration_minutes:.1f} min",
                flush=True,
            )
            if not success:
                print(f"  See: {stderr_path}", flush=True)
def build_runs(args, batch_id):
    base_config = {
        "checkpoint": "roberta-large",
        "ds_path": "nyu-mll/glue",
        "ds_name": args.dataset,
        "num_labels": LABEL_COUNTS[args.dataset],
        "num_clients": 10,
        "clients_per_round": 10,
        "alpha": 0.5,
        "total_rounds": args.rounds,
        "batch_size": 16,
        "lora_r": 8,
        "lora_alpha": 16,
        "target_modules": ["query", "value"],
        "num_workers": 0,
        "pin_memory": False,
        "early_stopping": False,
        "train_head": False,
        "seed": args.seed,
    }
    runs = []
    for method_name in args.methods:
        config = copy.deepcopy(base_config)
        config.update(METHOD_CONFIGS[method_name])
        config["METHOD_NAME"] = method_name
        config["extras"] = (
            f"ROBERTA-LARGE-PILOT-{batch_id}_{method_name}_"
            f"DS-{args.dataset}_A0.5_S{args.seed}"
        )
        runs.append(config)
    return runs
def parse_devices(value):
    devices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        devices.append(item if item.startswith("cuda:") else f"cuda:{item}")
    if not devices:
        raise argparse.ArgumentTypeError("Provide at least one CUDA device.")
    return devices
def orchestrate(args):
    import torch
    devices = parse_devices(args.devices)
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count == 0:
        raise RuntimeError("No CUDA devices are visible.")
    invalid_devices = [
        device
        for device in devices
        if int(device.split(":", maxsplit=1)[1]) >= visible_gpu_count
    ]
    if invalid_devices:
        raise RuntimeError(
            f"Unavailable devices: {invalid_devices}; visible GPU count: {visible_gpu_count}."
        )
    batch_id = time.strftime("%Y%m%d-%H%M%S")
    runs = build_runs(args, batch_id)
    print(f"Pilot batch: {batch_id}")
    print(f"Devices: {', '.join(devices)}")
    print(f"Concurrent jobs per device: {args.jobs_per_device}")
    print(f"Scheduled runs: {len(runs)}")
    for config in runs:
        print(f"  - {config['extras']}")
    if args.dry_run:
        return 0
    log_dir = Path("results") / "logs" / f"roberta_large_pilot_{batch_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    device_queue = queue.Queue()
    for device in devices:
        for _ in range(args.jobs_per_device):
            device_queue.put(device)
    max_workers = len(devices) * args.jobs_per_device
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_worker, config, device_queue, log_dir, len(runs))
            for config in runs
        ]
        concurrent.futures.wait(futures)
    print(f"Pilot complete: {len(runs) - failed_jobs} succeeded, {failed_jobs} failed.")
    print(f"Logs: {log_dir}")
    return 1 if failed_jobs else 0
def build_parser():
    parser = argparse.ArgumentParser(description="RoBERTa-Large FDA pilot experiment")
    parser.add_argument("--run-config", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=sorted(LABEL_COUNTS), default="mrpc")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--jobs-per-device", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser
if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    if cli_args.run_config:
        execute_run(cli_args.run_config)
    else:
        raise SystemExit(orchestrate(cli_args))

```


## File: `controller.py`

```py
"""Run the isolated FDA 3--50 epoch controller ablation.
This runner reuses the frozen production configuration from
``thesis_experiments.py`` and changes only the FDA maximum local epochs from
30 to 50. Under the current FDA threshold rule, this changes the intended
midpoint from 15 to 25 local epochs per round.
Examples:
    python controller_max50_experiment.py plan
    python controller_max50_experiment.py run --dataset sst2 --seed 42
    python controller_max50_experiment.py run --dataset sst2 --seed 42 43 44
    python controller_max50_experiment.py run --dataset qnli --seed 42
"""
import argparse
from pathlib import Path
import thesis_experiments as thesis
METHOD = "FDA-RoLoRA-DualTrack"
NUM_CLIENTS = 50
CLIENTS_PER_ROUND = 50
LORA_RANK = 4
FDA_E_MIN = 3
FDA_E_MAX = 50
FDA_MIDPOINT_EPOCHS = FDA_E_MAX / 2
SUPPORTED_DATASETS = ("sst2", "qnli")
DEFAULT_SEEDS = (42,)
LOG_DIR = Path("results/logs/controller_max50")
def build_config(dataset: str, seed: int, rounds: int | None = None) -> dict:
    """Derive one max-50 FDA run from the matching production configuration."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; choose from {SUPPORTED_DATASETS}."
        )
    candidates = [
        config
        for config in thesis.build_main_configs(dataset)
        if config["METHOD_NAME"] == METHOD
        and config["num_clients"] == NUM_CLIENTS
        and config["clients_per_round"] == CLIENTS_PER_ROUND
        and config["lora_r"] == LORA_RANK
        and config["seed"] == seed
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one production template for {dataset}/seed-{seed}, "
            f"found {len(candidates)}."
        )
    config = dict(candidates[0])
    if rounds is not None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        config["total_rounds"] = rounds
    config["e_min"] = FDA_E_MIN
    config["max_epochs"] = FDA_E_MAX
    config["controller_midpoint_epochs"] = FDA_MIDPOINT_EPOCHS
    config["experiment_group"] = "controller-max50"
    config["extras"] = (
        f"THESIS-{thesis.MODEL_SLUG}-CTRL-MAX50"
        f"-DS-{dataset}-C{NUM_CLIENTS}-R{config['total_rounds']}"
        f"-M-{thesis.METHOD_SLUGS[METHOD]}"
        f"-LR{config['client_opt_lr']:g}-RK{LORA_RANK}"
        f"-E{FDA_E_MIN}TO{FDA_E_MAX}-MID{FDA_MIDPOINT_EPOCHS:g}"
        f"-S{seed}"
        f"-ES{config['early_stopping_start_accuracy']:g}"
        f"-P{config['early_stopping_patience']}"
        f"-D{config['early_stopping_min_delta']:g}"
        f"-W{config['early_stopping_window']}"
    )
    return config
def print_plan(configs: list[dict], device: str) -> None:
    print("\nFDA CONTROLLER MAX-50 ABLATION")
    print("=" * 72)
    print(f"Model: {thesis.MODEL_CHECKPOINT}")
    print(f"Configurations: {len(configs)}")
    print(f"Device: {device}; execution: sequential, one process at a time")
    print(f"Method: {METHOD}")
    print(f"Clients/round: C{NUM_CLIENTS}/K{CLIENTS_PER_ROUND}")
    print(f"LoRA: r={LORA_RANK}, alpha={2 * LORA_RANK}; query,value")
    print(
        f"FDA window: {FDA_E_MIN}--{FDA_E_MAX} epochs; "
        f"threshold midpoint={FDA_MIDPOINT_EPOCHS:g} epochs"
    )
    print(
        "Stopping: activates at 90% accuracy; patience=5; "
        "min_delta=1%; window=3"
    )
    for config in configs:
        print(
            f"{config['ds_name'].upper()}: batch={config['batch_size']}; "
            f"layers={','.join(map(str, config['target_layer_indices']))}; "
            f"rounds={config['total_rounds']}; seed={config['seed']}; "
            f"LR={config['client_opt_lr']:g}"
        )
        print(f"  {config['extras']}")
    print("=" * 72)
def run_configs(configs: list[dict], device: str) -> None:
    thesis.LOG_DIR = LOG_DIR
    for config in configs:
        if not thesis.run_config_on_device(config, device):
            raise SystemExit(1)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated FDA max_epochs=50 controller ablation."
    )
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="sst2",
        help="Dataset to test. QNLI is optional and is never queued implicitly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="One or more seeds, executed sequentially.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override the production C50 round budget (default: 30).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used by the single experiment process.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    configs = [
        build_config(args.dataset, seed, args.rounds)
        for seed in args.seed
    ]
    print_plan(configs, args.device)
    if args.action == "run":
        run_configs(configs, args.device)
if __name__ == "__main__":
    main()

```


## File: `thesis_ema_pure_experiments.py`

```py
"""Run the final MAIN FDA-RoLoRA SMA W=3, E3--E20 replication.
This is a narrow orchestration layer over the canonical
``thesis_experiments.py`` runner. It reuses the existing training dispatch,
metric writer, process registry, and GPU worker implementation while keeping
SMA logs and control state separate from the completed MAIN study.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
try:
    import thesis_experiments as base
except ModuleNotFoundError as error:
    raise SystemExit(
        "Place thesis_sma_experiments.py beside thesis_experiments.py "
        "in the ARIS-FDA-LoRA repository."
    ) from error
SMA_METHOD = "FDA-RoLoRA-DualTrack-SMA"
STAGE = "main-sma-w3"
SEEDS = (42, 43)
CLIENT_COUNTS = (20, 50)
PARTITION_ALPHA = 1000.0
E_MIN = 3
E_MAX = 30
SMA_WINDOW = 3
GAMMA = 0.3
THETA = 0.0
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")
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
DATASET_COST_ORDER = ("rte", "mrpc", "qnli", "mnli", "qqp", "sst2")
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
RESULTS_DIR = Path("results")
LOG_DIR = RESULTS_DIR / "logs" / "thesis_roberta_base_sma_w3"
RUN_CONTROL_DIR = RESULTS_DIR / "run_control_sma_w3"
ACTIVE_RUNS_DIR = RUN_CONTROL_DIR / "active"
STOP_REQUESTS_DIR = RUN_CONTROL_DIR / "stop_requests"
DROPPED_RUNS_DIR = RUN_CONTROL_DIR / "dropped"
OPERATOR_EVENTS_FILE = RUN_CONTROL_DIR / "operator_events.jsonl"
SCHEDULER_LOCK_FILE = RUN_CONTROL_DIR / "scheduler.lock"
JOBS_PER_GPU = max(
    1, int(os.environ.get("THESIS_SMA_JOBS_PER_GPU", "3"))
)
BATCH64_JOBS_PER_GPU = max(
    1,
    min(
        JOBS_PER_GPU,
        int(os.environ.get("THESIS_SMA_BATCH64_JOBS_PER_GPU", "2")),
    ),
)
def configure_base_runner() -> None:
    required = (
        "make_config",
        "run_main_phase",
        "run_config_on_device",
        "check_accuracy_gated_early_stopping",
        "load_active_run_records",
        "stop_active_run",
    )
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        raise SystemExit(
            "The canonical thesis_experiments.py is missing required support: "
            + ", ".join(missing)
        )
    if SMA_METHOD not in getattr(base, "METHODS", {}):
        raise SystemExit(
            f"{SMA_METHOD} is not registered in thesis_experiments.py"
        )
    base.MAIN_FDA_E_MIN = E_MIN
    base.MAIN_FDA_E_MAX = E_MAX
    base.MAIN_METHODS = [SMA_METHOD]
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
    base.main_config_priority = sma_config_priority
    base.wait_for_registered_run = wait_for_registered_sma_run
def sma_config_priority(config: dict) -> tuple:
    return (
        int(config["total_rounds"]),
        DATASET_COST_ORDER.index(config["ds_name"]),
        int(config["num_clients"]),
        int(config["seed"]),
    )
def build_sma_configs() -> list[dict]:
    configs = []
    for dataset in DATASET_COST_ORDER:
        settings = DATASET_SETTINGS[dataset]
        for num_clients in CLIENT_COUNTS:
            for seed in SEEDS:
                rounds = settings["rounds"][num_clients]
                gate = EARLY_STOPPING_GATES[dataset][num_clients]
                config = base.make_config(
                    stage=STAGE,
                    dataset=dataset,
                    method=SMA_METHOD,
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
                        "e_min": E_MIN,
                        "max_epochs": E_MAX,
                        "gamma": GAMMA,
                        "theta": THETA,
                        "sma_window": SMA_WINDOW,
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
                config["extras"] += (
                    f"-ES{gate:g}"
                    f"-MR{EARLY_STOPPING_MIN_ROUNDS}"
                    f"-P{EARLY_STOPPING_PATIENCE}"
                    f"-D{EARLY_STOPPING_MIN_DELTA:g}"
                    f"-W{EARLY_STOPPING_WINDOW}"
                )
                configs.append(config)
    return sorted(configs, key=sma_config_priority)
def validate_configs(configs: list[dict]) -> None:
    expected = (
        len(DATASET_COST_ORDER) * len(CLIENT_COUNTS) * len(SEEDS)
    )
    if len(configs) != expected:
        raise RuntimeError(f"Expected {expected} SMA configs, found {len(configs)}")
    run_ids = [config["extras"] for config in configs]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("Duplicate SMA experiment identifiers detected")
    for config in configs:
        if config["METHOD_NAME"] != SMA_METHOD:
            raise RuntimeError("A non-SMA method entered the SMA matrix")
        if config["seed"] not in SEEDS:
            raise RuntimeError("Unexpected seed in SMA matrix")
        if config["clients_per_round"] != config["num_clients"]:
            raise RuntimeError("SMA main replication must use full participation")
        if config["e_min"] != E_MIN or config["max_epochs"] != E_MAX:
            raise RuntimeError("Unexpected adaptive epoch window")
        if config["sma_window"] != SMA_WINDOW:
            raise RuntimeError("Unexpected SMA window")
    priorities = [sma_config_priority(config) for config in configs]
    if priorities != sorted(priorities):
        raise RuntimeError("SMA configurations are not cost ordered")
def dropped_marker_path(run_id: str) -> Path:
    return DROPPED_RUNS_DIR / f"{base.get_control_id(run_id)}.json"
def is_dropped(config: dict) -> bool:
    return dropped_marker_path(config["extras"]).exists()
def print_plan(configs: list[dict]) -> None:
    complete = sum(base.is_completed(config) for config in configs)
    dropped = sum(is_dropped(config) for config in configs)
    active_ids = {
        record["run_id"] for record in base.load_active_run_records()
    }
    active = sum(config["extras"] in active_ids for config in configs)
    print("\nFDA-SMA W=3 MAIN REPLICATION")
    print("=" * 88)
    print(
        f"Configurations: {len(configs)} "
        f"(complete={complete}, active={active}, dropped={dropped})"
    )
    print("Method: FDA-RoLoRA-DualTrack-SMA")
    print("Window: E3--E20; SMA W=3; gamma=0.3; theta=0")
    print("Model: roberta-base; rank=4; LoRA alpha=8; query+value")
    print("Clients: C20/K20 and C50/K50; seeds: 42,43; Dirichlet alpha=1000")
    print(
        "Stopping: minimum 6 rounds; patience 3; min_delta 0.5%; "
        "accuracy window 3"
    )
    print(
        f"Scheduler: {JOBS_PER_GPU} jobs/GPU; "
        f"batch-64 cap {BATCH64_JOBS_PER_GPU}/GPU"
    )
    print("Queue order: total rounds first, then estimated dataset cost, then seed")
    print("-" * 88)
    for index, config in enumerate(configs, start=1):
        state = "DONE" if base.is_completed(config) else "DROP" if is_dropped(config) else "TODO"
        print(
            f"{index:02d} {state:4s} "
            f"{config['ds_name'].upper():5s} C{config['num_clients']:<2d} "
            f"R{config['total_rounds']:<2d} S{config['seed']} "
            f"B{config['batch_size']:<2d} "
            f"gate={config['early_stopping_start_accuracy']:.0%}"
        )
    print("=" * 88)
def wait_for_registered_sma_run(
    config: dict | None,
    record: dict,
    poll_seconds: int = 5,
) -> bool:
    run_id = record["run_id"]
    control_id = record["control_id"]
    print(
        f"ADOPT [{control_id}] pid={record['pid']} {run_id} "
        f"on {record['device']}"
    )
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
        print(f"RELEASE [{control_id}] detached SMA worker finished")
        return True
    if base.is_completed(config):
        final_metric = base.read_latest_metric_line(record["stdout_path"])
        print(f"ADOPTED DONE [{control_id}] {run_id}; {final_metric}")
        return True
    print(
        f"ADOPTED ERROR [{control_id}] {run_id}: worker exited without "
        "a readable metrics CSV"
    )
    return False
def list_runs() -> None:
    records = base.load_active_run_records()
    if not records:
        print("No active FDA-SMA W=3 runs.")
        return
    print(f"Active FDA-SMA W=3 runs: {len(records)}")
    for record in records:
        elapsed = base.format_elapsed(
            time.time() - float(record["started_at"])
        )
        latest = base.read_latest_metric_line(record["stdout_path"])
        print(
            f"[{record['control_id']}] pid={record['pid']} "
            f"gpu={record['device']} elapsed={elapsed}"
        )
        print(f"  run:    {record['run_id']}")
        print(f"  latest: {latest}")
        print(f"  log:    {record['stdout_path']}")
    print("Stop one attempt: ./run_thesis_sma_experiments.sh stop <ID>")
    print("Stop and skip permanently: ./run_thesis_sma_experiments.sh drop <ID>")
def drop_active_run(selector: str, reason: str) -> None:
    record = base.resolve_active_run(selector)
    marker = {
        "action": "operator_drop",
        "control_id": record["control_id"],
        "reason": reason,
        "requested_at": time.time(),
        "run_id": record["run_id"],
    }
    base.atomic_write_json(dropped_marker_path(record["run_id"]), marker)
    base.stop_active_run(selector, reason)
    print("This configuration is marked dropped and will not restart on resume.")
def list_dropped_runs() -> None:
    markers = (
        sorted(DROPPED_RUNS_DIR.glob("*.json"))
        if DROPPED_RUNS_DIR.exists()
        else []
    )
    if not markers:
        print("No FDA-SMA configurations are marked dropped.")
        return
    print(f"Dropped FDA-SMA configurations: {len(markers)}")
    for path in markers:
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        print(f"[{marker['control_id']}] {marker['run_id']}")
        print(f"  reason: {marker['reason']}")
def restore_dropped_run(selector: str) -> None:
    matches = []
    if DROPPED_RUNS_DIR.exists():
        for path in DROPPED_RUNS_DIR.glob("*.json"):
            try:
                marker = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                marker["control_id"].startswith(selector)
                or marker["run_id"].startswith(selector)
            ):
                matches.append((path, marker))
    if len(matches) != 1:
        if not matches:
            raise SystemExit(f"No dropped SMA run matches {selector!r}")
        ids = ", ".join(marker["control_id"] for _, marker in matches)
        raise SystemExit(f"Ambiguous dropped-run selector {selector!r}: {ids}")
    path, marker = matches[0]
    path.unlink()
    print(f"Restored [{marker['control_id']}] {marker['run_id']}")
def run_all(configs: list[dict]) -> None:
    runnable = [config for config in configs if not is_dropped(config)]
    RUN_CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    with SCHEDULER_LOCK_FILE.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(
                "Another FDA-SMA scheduler is already active. "
                "Use the status command instead of starting a second one."
            ) from error
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        print_plan(configs)
        base.run_main_phase(runnable)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        required=True,
        choices=("plan", "run", "status", "stop", "drop", "dropped", "restore"),
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--reason",
        default="operator stopped FDA-SMA run after inspecting live metrics",
    )
    return parser.parse_args()
def main() -> None:
    configure_base_runner()
    configs = build_sma_configs()
    validate_configs(configs)
    args = parse_args()
    if args.group == "plan":
        print_plan(configs)
    elif args.group == "run":
        run_all(configs)
    elif args.group == "status":
        list_runs()
    elif args.group == "dropped":
        list_dropped_runs()
    elif args.group in {"stop", "drop", "restore"}:
        if not args.run_id:
            raise SystemExit(f"--run-id is required for {args.group}")
        if args.group == "stop":
            base.stop_active_run(args.run_id, args.reason)
        elif args.group == "drop":
            drop_active_run(args.run_id, args.reason)
        else:
            restore_dropped_run(args.run_id)
if __name__ == "__main__":
    main()

```


## File: `toy_experiment_runner.py`

```py
import os
import sys
import time
import subprocess
import argparse
import json
import concurrent.futures
import queue
import threading
from datetime import datetime
import torch
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    raise RuntimeError(
        f"Two GPUs are required, but PyTorch detects only {num_gpus}. "
        "Check CUDA_VISIBLE_DEVICES."
    )
AVAILABLE_DEVICES = ['cuda:0', 'cuda:1']
JOBS_PER_DEVICE = 10
MAX_CONCURRENT_JOBS = len(AVAILABLE_DEVICES) * JOBS_PER_DEVICE
progress_lock = threading.Lock()
completed_jobs = 0
total_scheduled_jobs = 0
TOY_BASE_HYPERPARAMS = {
    'ds_path': "ylecun/mnist",
    'ds_name': "mnist",
    'num_labels': 10,
    'checkpoint': "toy-model",
    'total_rounds': 100,
    'batch_size': 64,
    'lora_r': 16,
    'num_workers': 0,
    'pin_memory': False,
    'early_stopping': False,
    'threshold': False,
    'local_epochs': 1
}
SCENARIOS = {
    "5-Clients-2-Classes": {'num_clients': 5, 'clients_per_round': 5, 'alpha': 0.0},
    "10-Clients-1-Class": {'num_clients': 10, 'clients_per_round': 10, 'alpha': 0.0}
}
EXPERIMENT_GROUPS = {
    "fda_sma": {
        "FDA-RoLoRA-DualTrack-SMA": [
            {'variant_name': f'SGD_window{w}_E3', 'sma_window': w, 'server_opt_name': 'SGD', 'server_opt_lr': 1.0, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'e_min': 3, 'max_epochs': 15, 'theta': 0.0}
            for w in [3]
        ]
    }
}
def execute_run(config_json):
    from fdaopt import utils
    from fdaopt.training.fda_opt import (
        fda_rolora_dual_track_opt,
        fda_rolora_dual_track_ema_standard_opt,
        fda_rolora_dual_track_ema_pure_opt,
        fda_rolora_dual_track_sma_opt
    )
    hyperparams = json.loads(config_json)
    method_name = hyperparams['METHOD_NAME']
    target_device = hyperparams.pop('TARGET_DEVICE', 'cuda:0')
    if target_device.startswith('cuda'):
        torch.cuda.set_device(torch.device(target_device))
    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(target_device)
    utils.set_seed(hyperparams.get('seed', 42))
    if method_name == "FDA-RoLoRA-DualTrack":
        fda_rolora_dual_track_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-EMA-Standard":
        fda_rolora_dual_track_ema_standard_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-EMA-Pure":
        fda_rolora_dual_track_ema_pure_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-SMA":
        fda_rolora_dual_track_sma_opt(hyperparams)
    else:
        raise ValueError(f"Unknown method: {method_name}")
def _worker_task(run_config, device_queue, log_dir):
    global completed_jobs, total_scheduled_jobs
    device = device_queue.get()
    run_identifier = run_config['extras']
    out_file = os.path.join(log_dir, f"{run_identifier}.out")
    err_file = os.path.join(log_dir, f"{run_identifier}.err")
    try:
        run_config['TARGET_DEVICE'] = device
        config_str = json.dumps(run_config)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--run_config", config_str]
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        start_time = time.time()
        with open(out_file, 'w') as stdout_f, open(err_file, 'w') as stderr_f:
            subprocess.run(cmd, env=env, stdout=stdout_f, stderr=stderr_f, check=True)
        duration = (time.time() - start_time) / 60
        with progress_lock:
            completed_jobs += 1
            print(f"[{completed_jobs}/{total_scheduled_jobs}] SUCCESS | {run_identifier} ({duration:.2f}m)")
    except subprocess.CalledProcessError:
        with progress_lock:
            completed_jobs += 1
            print(f"[{completed_jobs}/{total_scheduled_jobs}] ERROR | {run_identifier} | Check {err_file}")
    finally:
        device_queue.put(device)
def orchestrate(experiment_group):
    global total_scheduled_jobs
    if experiment_group not in EXPERIMENT_GROUPS:
        raise ValueError(f"Invalid experiment group. Choose from: {list(EXPERIMENT_GROUPS.keys())}")
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    all_runs = []
    batch_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    methods = EXPERIMENT_GROUPS[experiment_group]
    for linear_setting in [False]:
        for scen_name, scen_cfg in SCENARIOS.items():
            for method_name, method_variants in methods.items():
                for method_cfg in method_variants:
                    for seed in [42, 43, 44]:
                        run_config = TOY_BASE_HYPERPARAMS.copy()
                        run_config.update(scen_cfg)
                        run_config.update(method_cfg)
                        run_config['linear'] = linear_setting
                        run_config['seed'] = seed
                        run_config['METHOD_NAME'] = method_name
                        lin_str = "LIN" if linear_setting else "NONLIN"
                        var_str = method_cfg.get('variant_name', 'VAR')
                        run_config['extras'] = f"TOY-NEW3FDA-{experiment_group.upper()}-{batch_timestamp}_{method_name}_{var_str}_{scen_name}_{lin_str}_S{seed}"
                        all_runs.append(run_config)
    total_scheduled_jobs = len(all_runs)
    device_queue = queue.Queue()
    for _ in range(JOBS_PER_DEVICE):
        for dev in AVAILABLE_DEVICES:
            device_queue.put(dev)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = [executor.submit(_worker_task, run_config, device_queue, log_dir) for run_config in all_runs]
        concurrent.futures.wait(futures)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, default=None)
    parser.add_argument('--experiment_group', type=str, choices=list(EXPERIMENT_GROUPS.keys()), help="Group of methods to run.")
    args = parser.parse_args()
    if args.run_config:
        execute_run(args.run_config)
    elif args.experiment_group:
        orchestrate(args.experiment_group)
    else:
        parser.print_help()

```


## File: `sub_exp_2_simulation.py`

```py
import os
import sys
import time
import subprocess
import argparse
import json
import concurrent.futures
import queue
import threading
try:
    import torch
    num_gpus = torch.cuda.device_count()
    AVAILABLE_DEVICES = [f'cuda:{i}' for i in range(num_gpus)] if num_gpus > 0 else ['cpu']
except Exception:
    AVAILABLE_DEVICES = ['cpu']
JOBS_PER_DEVICE = 5
MAX_CONCURRENT_JOBS = len(AVAILABLE_DEVICES) * JOBS_PER_DEVICE
progress_lock = threading.Lock()
completed_jobs = 0
total_scheduled_jobs = 0
BASE_HYPERPARAMS = {
    'checkpoint': "roberta-large",
    'total_rounds': 100,
    'batch_size': 32,
    'lora_r': 8,
    'lora_alpha': 16,
    'target_modules': ["query", "value"],
    'num_workers': 4,
    'pin_memory': True,
    'early_stopping': True,
    'early_stopping_patience': 45,
    'train_head': False,
    'local_epochs': 5
}
SCENARIOS = {
    "Cross-Silo": {'num_clients': 10, 'clients_per_round': 10},
    "Cross-Device": {'num_clients': 100, 'clients_per_round': 10}
}
ALPHAS = [0.1, 0.5, 1.0, 10.0]
METHODS = {
    "FedAvg-LoRA": [{'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01}],
    "FedAvg-RoLoRA": [{'server_opt_name': 'SGD', 'server_opt_lr': 1.0, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01}],
    "FFA-LoRA": [{'server_opt_name': 'SGD', 'server_opt_lr': 1.0, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01}],
    "FDA-RoLoRA-DualTrack": [{'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3, 'client_opt_name': 'SGD', 'client_opt_lr': 0.01, 'gamma': 0.3, 'theta': 0.0, 'e_min': 2, 'max_epochs': 10}]
}
def execute_run(config_json):
    from fdaopt import utils
    from fdaopt.training.fed_opt import fedavg_lora, fedavg_rolora, ffa_lora
    from fdaopt.training.fda_opt import fda_rolora_dual_track_opt
    hyperparams = json.loads(config_json)
    method_name = hyperparams.pop('METHOD_NAME')
    target_device = hyperparams.pop('TARGET_DEVICE', 'cuda:0')
    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(target_device)
    utils.set_seed(hyperparams.get('seed', 42))
    if method_name == "FedAvg-LoRA":
        fedavg_lora(hyperparams)
    elif method_name == "FedAvg-RoLoRA":
        fedavg_rolora(hyperparams)
    elif method_name == "FFA-LoRA":
        ffa_lora(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack":
        fda_rolora_dual_track_opt(hyperparams)
    else:
        raise ValueError(f"Unknown method requested: {method_name}")
def _worker_task(run_config, device_queue, log_dir):
    global completed_jobs, total_scheduled_jobs
    device = device_queue.get()
    run_identifier = run_config['extras']
    out_file = os.path.join(log_dir, f"{run_identifier}.out")
    err_file = os.path.join(log_dir, f"{run_identifier}.err")
    try:
        run_config['TARGET_DEVICE'] = device
        config_str = json.dumps(run_config)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--run_config", config_str]
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        start_time = time.time()
        with open(out_file, 'w') as stdout_f, open(err_file, 'w') as stderr_f:
            subprocess.run(cmd, env=env, stdout=stdout_f, stderr=stderr_f, check=True)
        duration = (time.time() - start_time) / 60
        with progress_lock:
            completed_jobs += 1
            print(f"[{completed_jobs}/{total_scheduled_jobs}] SUCCESS | {run_identifier} ({duration:.2f}m)")
    except subprocess.CalledProcessError:
        with progress_lock:
            completed_jobs += 1
            print(f"[{completed_jobs}/{total_scheduled_jobs}] ERROR | {run_identifier} | Check {err_file}")
    finally:
        time.sleep(3)
        device_queue.put(device)
def orchestrate(dataset_name):
    global total_scheduled_jobs
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    all_runs = []
    labels_map = {"mnli": 3, "sst2": 2, "qnli": 2, "qqp": 2, "rte": 2, "mrpc": 2}
    num_labels = labels_map.get(dataset_name, 2)
    for scen_name, scen_cfg in SCENARIOS.items():
        for alpha_val in ALPHAS:
            for method_name, method_variants in METHODS.items():
                for method_cfg in method_variants:
                    for seed in [42, 43, 44]:
                        run_config = BASE_HYPERPARAMS.copy()
                        run_config['ds_path'] = "nyu-mll/glue"
                        run_config['ds_name'] = dataset_name
                        run_config['num_labels'] = num_labels
                        run_config.update(scen_cfg)
                        run_config['alpha'] = alpha_val
                        run_config.update(method_cfg)
                        run_config['seed'] = seed
                        run_config['METHOD_NAME'] = method_name
                        if "FDA" in method_name:
                            run_config['local_epochs'] = 1
                        run_config['extras'] = f"SUB2_{method_name}_DS-{dataset_name}_SCEN-{scen_name}_A{alpha_val}_S{seed}"
                        all_runs.append(run_config)
    total_scheduled_jobs = len(all_runs)
    device_queue = queue.Queue()
    for dev in AVAILABLE_DEVICES:
        for _ in range(JOBS_PER_DEVICE):
            device_queue.put(dev)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = [executor.submit(_worker_task, run_config, device_queue, log_dir) for run_config in all_runs]
        concurrent.futures.wait(futures)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, default=None)
    parser.add_argument('--dataset', type=str, help="The GLUE dataset to run")
    args = parser.parse_args()
    if args.run_config:
        execute_run(args.run_config)
    elif args.dataset:
        orchestrate(args.dataset)

```


## File: `master_experiments/thesis_master_experiments.py`

```py
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
    for ds in ["qnli", "mnli"]:
        settings = DATASET_SETTINGS[ds]
        for c in DEFAULT_CLIENT_COUNTS:
            rounds = settings["rounds"][c]
            gate = NONIID_GATES[ds][c]
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

```


## File: `non_iid_experiments/thesis_non_iid_experiments.py`

```py
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
DEFAULT_FDA_CONFIGS = [
    (20, 1.0),
    (40, 1.0),
]
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")
EARLY_STOPPING_MIN_ROUNDS = 30
EARLY_STOPPING_PATIENCE = 5
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

```


## File: `non_iid_experiments/fda_rolora_non_iid.py`

```py
"""
Non-IID Experiment Module for Closed-Loop FDA-RoLoRA Dual-Track and FedAvg-RoLoRA.
Implements:
  1. FDA-RoLoRA-DualTrack-Pure:
     - Alternating rank subspace updates starting with Matrix B on Round 1.
     - Pure physical inter-client drift metric (NODISC, gamma=1.0).
     - Adaptive local epoch bounds [e_min=1, e_max=20] with proportional setpoint ratio k.
     - Natural cold-start bootstrap from initial Theta = 0.0 (no first-round guard).
  2. FedAvg-RoLoRA:
     - Fixed local epoch baseline (E=10, E=20) starting with Matrix B on Round 1.
     - Logs inter-client drift variance on the active factor for direct comparison.
  3. Robust Non-IID Early Stopping:
     - 60% minimum round threshold (disabled for Rounds 1..30 of R=50).
     - 5-round rolling moving average to filter out Dirichlet client sampling variance.
     - Stagnation patience = 3, delta = 0.005, gated by plateau_min_accuracy.
  4. Per-round incremental metrics persistence via MetricsHandler.
"""
from __future__ import annotations
import gc
import math
import os
import time
from typing import Any, Dict, List
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification
from fdaopt import utils
from fdaopt.datasets.fed_data_prep import ClientSampler, prepare_federated_datasets
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (
    average_client_parameters,
    compute_client_drifts,
    compute_metrics,
    compute_pseudo_gradients,
    copy_parameters,
    get_target_metric_value,
    get_updated_client_parameters,
    set_gradients,
    variance,
)
from fdaopt.training.fed_train import client_round_train
from fdaopt.training.optimizers import (
    get_opt_hyperparams,
    get_optimizer_class,
    server_client_optimizers,
)
def get_save_device() -> str:
    if utils.DEVICE_RAM_PROGRAM in ("performance", "moderate"):
        return utils.DEVICE
    return "cpu"
def toggle_rolora_matrices(model: torch.nn.Module, active_matrix: str) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = f"lora_{active_matrix}" in name
def check_non_iid_early_stopping(
    hyperparams: Dict[str, Any],
    metrics_history: List[Dict[str, Any]],
) -> bool:
    """
    Evaluate early stopping conditions tailored for Dirichlet Non-IID federated training:
      1. Strictly disabled until 60% of total rounds have elapsed (e.g. Round 30 of 50).
      2. 5-round moving average window to smooth out heterogeneous client sampling variance.
      3. Patience = 3 consecutive window evaluations without meaningful accuracy gain (min_delta=0.005).
      4. Requires best smoothed value to exceed the plateau baseline gate.
    """
    if not metrics_history:
        return False
    total_rounds = int(hyperparams.get("total_rounds", 50))
    min_rounds = int(
        hyperparams.get(
            "early_stopping_min_rounds",
            int(0.6 * total_rounds),
        )
    )
    current_round = len(metrics_history)
    if current_round < min_rounds:
        return False
    window_size = int(hyperparams.get("early_stopping_window", 5))
    if current_round < window_size:
        return False
    patience = int(hyperparams.get("early_stopping_patience", 3))
    min_delta = float(hyperparams.get("early_stopping_min_delta", 0.005))
    plateau_min_acc = float(
        hyperparams.get(
            "plateau_min_accuracy",
            hyperparams.get(
                "early_stopping_start_accuracy",
                0.80 if hyperparams.get("ds_name") == "qnli" else 0.60,
            ),
        )
    )
    values = [
        get_target_metric_value(hyperparams["ds_name"], m)
        for m in metrics_history
    ]
    smoothed_values = [
        sum(values[i : i + window_size]) / window_size
        for i in range(current_round - window_size + 1)
    ]
    best_value = -float("inf")
    rounds_without_improvement = 0
    for val in smoothed_values:
        if val > best_value + min_delta:
            best_value = val
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
    if best_value >= plateau_min_acc and rounds_without_improvement >= patience:
        return True
    return False
def fedavg_rolora_opt(hyperparams: Dict[str, Any]) -> None:
    """
    Execute standard Federated Averaging with Alternating Low-Rank Adaptation (FedAvg-RoLoRA).
    Operates at fixed local epochs (e.g. E=10 or E=20) with Matrix B starting at Round 1.
    Evaluates and records inter-client subspace drift variance every round for parity comparison.
    """
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
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams["checkpoint"], num_labels=hyperparams["num_labels"]
    )
    lora_kwargs = {
        "task_type": TaskType.SEQ_CLS,
        "inference_mode": False,
        "r": hyperparams.get("lora_r", 4),
        "lora_alpha": hyperparams.get("lora_alpha", 8),
        "lora_dropout": 0.1,
        "target_modules": hyperparams.get("target_modules", ["query", "value"]),
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
    group_A = params_A + params_other
    group_B = params_B + params_other
    client_opt_class = get_optimizer_class(hyperparams["client_opt_name"])
    client_opt_kwargs = get_opt_hyperparams("client", hyperparams)
    base_lr = client_opt_kwargs.get("lr", hyperparams.get("client_opt_lr", 0.10))
    other_opt_kwargs = {k: v for k, v in client_opt_kwargs.items() if k != "lr"}
    client_sampler = ClientSampler(list(range(hyperparams["num_clients"])), hyperparams["clients_per_round"])
    client_dict_A = {
        c_id: [p.detach().clone().to(get_save_device()) for p in group_A]
        for c_id in range(hyperparams["num_clients"])
    }
    client_dict_B = {
        c_id: [p.detach().clone().to(get_save_device()) for p in group_B]
        for c_id in range(hyperparams["num_clients"])
    }
    metrics_handler = MetricsHandler(hyperparams)
    local_epochs = int(hyperparams.get("local_epochs", 10))
    print(
        f"[{time.strftime('%H:%M:%S')}] [FedAvg-RoLoRA] Starting {hyperparams['total_rounds']} rounds "
        f"on {hyperparams['ds_name'].upper()} (C={hyperparams['num_clients']}, E={local_epochs}, alpha={hyperparams['alpha']})."
    )
    for r in range(hyperparams["total_rounds"]):
        round_start_time = time.time()
        active_matrix = "B" if r % 2 == 0 else "A"
        toggle_rolora_matrices(model, active_matrix)
        active_params = group_A if active_matrix == "A" else group_B
        active_lora_params = params_A if active_matrix == "A" else params_B
        active_client_dict = client_dict_A if active_matrix == "A" else client_dict_B
        round_start_active_params = [p.detach().clone() for p in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(
            active_client_dict, sampled_clients, round_start_active_params
        )
        round_steps = local_epochs * fed_ds.epoch_steps(sampled_clients)
        training_loss = 0.0
        training_accuracy = 0.0
        train_start = time.time()
        for client_id in sampled_clients:
            param_groups = [{"params": active_lora_params, "lr": base_lr}]
            if params_other:
                param_groups.append({"params": params_other, "lr": base_lr * 0.1})
            client_opt = client_opt_class(param_groups, **other_opt_kwargs)
            loss, acc = client_round_train(
                model,
                active_params,
                active_client_dict,
                client_opt,
                fed_ds,
                client_id,
                round_steps,
            )
            training_loss += loss
            training_accuracy += acc
        train_end = time.time()
        agg_start = time.time()
        sampled_client_params = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
        client_drifts = compute_client_drifts(round_start_active_params, sampled_client_params)
        var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
        averaged_params = average_client_parameters(sampled_client_params)
        with torch.no_grad():
            update_l2_norm = math.sqrt(
                sum(
                    torch.sum((p_new.to(utils.DEVICE) - p_old.to(utils.DEVICE)) ** 2).item()
                    for p_new, p_old in zip(averaged_params, round_start_active_params)
                )
            )
        copy_parameters(from_params=averaged_params, to_params=active_params)
        agg_end = time.time()
        round_end_time = time.time()
        eval_results = compute_metrics(model, hyperparams["ds_path"], hyperparams["ds_name"], test_ds)
        metrics = {
            "method": "FedAvg-RoLoRA",
            "round": r + 1,
            "active_matrix": active_matrix,
            "start_matrix": "B",
            "alpha": hyperparams.get("alpha", 0.5),
            "num_clients": hyperparams["num_clients"],
            "clients_per_round": hyperparams["clients_per_round"],
            "local_epochs": local_epochs,
            "round_epochs": local_epochs,
            "round_steps": round_steps,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "variance": var_subspace,
            "effective_variance": var_subspace,
            "avg_norm_sq_drifts": avg_norm_sq_drifts,
            "norm_sq_avg_drift": norm_sq_avg_drift,
        } | eval_results
        metrics["training_loss"] = training_loss / hyperparams["clients_per_round"]
        metrics["training_accuracy"] = training_accuracy / hyperparams["clients_per_round"]
        metrics_handler.append_round_metrics(metrics)
        metrics_handler.save_metrics()
        primary_acc = get_target_metric_value(hyperparams["ds_name"], metrics)
        print(
            f"[METRIC] round={r + 1}/{hyperparams['total_rounds']} matrix={active_matrix} "
            f"epochs={local_epochs} train_loss={metrics['training_loss']:.4f} "
            f"acc={primary_acc:.4f} var={var_subspace:.4e}"
        )
        gc.collect()
        if hyperparams.get("early_stopping", True) and check_non_iid_early_stopping(
            hyperparams, metrics_handler.round_metrics
        ):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            metrics["early_stopped"] = True
            metrics_handler.save_metrics()
            break
    metrics_handler.save_metrics()
    print(f"[{time.strftime('%H:%M:%S')}] [FedAvg-RoLoRA] Completed execution. Metrics saved to CSV.")
def fda_rolora_dual_track_pure_opt(hyperparams: Dict[str, Any]) -> None:
    """
    Execute Pure Closed-Loop FDA-RoLoRA Dual-Track with dynamic inter-client drift regularization:
      1. Starts with Matrix B first on Round 1 (r=0).
      2. Pure un-discounted subspace drift variance (NODISC, gamma=1.0).
      3. Dynamic bounds [e_min=1, e_max=20] with zero first-round forced epoch guard.
      4. Adaptive controller ratio setpoint: target_epochs = controller_ratio * e_max.
      5. Robust Non-IID early stopping (Round >= 30, window=5, delta=0.005).
    """
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
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams["checkpoint"], num_labels=hyperparams["num_labels"]
    )
    lora_kwargs = {
        "task_type": TaskType.SEQ_CLS,
        "inference_mode": False,
        "r": hyperparams.get("lora_r", 4),
        "lora_alpha": hyperparams.get("lora_alpha", 8),
        "lora_dropout": 0.1,
        "target_modules": hyperparams.get("target_modules", ["query", "value"]),
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
        c_id: [p.detach().clone().to(get_save_device()) for p in params_A]
        for c_id in range(hyperparams["num_clients"])
    }
    client_dict_B = {
        c_id: [p.detach().clone().to(get_save_device()) for p in params_B]
        for c_id in range(hyperparams["num_clients"])
    }
    metrics_handler = MetricsHandler(hyperparams)
    theta_A = 0.0
    theta_B = 0.0
    print(
        f"[{time.strftime('%H:%M:%S')}] [FDA-RoLoRA-Pure] Starting {hyperparams['total_rounds']} rounds "
        f"on {hyperparams['ds_name'].upper()} (C={hyperparams['num_clients']}, k={controller_ratio}, "
        f"E={e_min}..{e_max}, alpha={hyperparams['alpha']})."
    )
    for r in range(hyperparams["total_rounds"]):
        round_start_time = time.time()
        active_matrix = "B" if r % 2 == 0 else "A"
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == "A" else params_B
        active_server_opt = server_opt_A if active_matrix == "A" else server_opt_B
        active_client_dict = client_dict_A if active_matrix == "A" else client_dict_B
        current_theta = theta_A if active_matrix == "A" else theta_B
        round_start_active_params = [p.detach().clone() for p in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(
            active_client_dict, sampled_clients, round_start_active_params
        )
        round_steps = hyperparams.get("local_epochs", 1) * fed_ds.epoch_steps(sampled_clients)
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
                _, client_opt = server_client_optimizers(
                    active_params, hyperparams, num_lora_params=num_lora
                )
                loss, acc = client_round_train(
                    model,
                    active_params,
                    active_client_dict,
                    client_opt,
                    fed_ds,
                    client_id,
                    round_steps,
                )
                training_loss += loss
                training_accuracy += acc
            sampled_active_client_dict = {
                c_id: active_client_dict[c_id] for c_id in sampled_clients
            }
            client_drifts = compute_client_drifts(
                round_start_active_params, sampled_active_client_dict
            )
            var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
            if var_subspace > current_theta or round_epochs >= e_max:
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
        eval_results = compute_metrics(model, hyperparams["ds_path"], hyperparams["ds_name"], test_ds)
        metrics = {
            "method": "FDA-RoLoRA-DualTrack-Pure",
            "round": r + 1,
            "active_matrix": active_matrix,
            "start_matrix": "B",
            "first_round_guard": False,
            "alpha": hyperparams.get("alpha", 0.5),
            "num_clients": hyperparams["num_clients"],
            "clients_per_round": hyperparams["clients_per_round"],
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
        } | eval_results
        metrics["training_loss"] = training_loss / hyperparams["clients_per_round"]
        metrics["training_accuracy"] = training_accuracy / hyperparams["clients_per_round"]
        metrics_handler.append_round_metrics(metrics)
        metrics_handler.save_metrics()
        primary_acc = get_target_metric_value(hyperparams["ds_name"], metrics)
        print(
            f"[METRIC] round={r + 1}/{hyperparams['total_rounds']} matrix={active_matrix} "
            f"epochs={round_epochs} train_loss={metrics['training_loss']:.4f} "
            f"acc={primary_acc:.4f} var={var_subspace:.4e}"
        )
        gc.collect()
        if hyperparams.get("early_stopping", True) and check_non_iid_early_stopping(
            hyperparams, metrics_handler.round_metrics
        ):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            metrics["early_stopped"] = True
            metrics_handler.save_metrics()
            break
    metrics_handler.save_metrics()
    print(f"[{time.strftime('%H:%M:%S')}] [FDA-RoLoRA-Pure] Completed execution. Metrics saved to CSV.")

```


## File: `pure_experiments/fda_rolora_pure.py`

```py
import os
import gc
import time
import math
from typing import Dict, Any
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
        if utils.DEVICE_RAM_PROGRAM in ('performance', 'moderate'):
            return utils.DEVICE
        return 'cpu'
    def toggle_rolora_matrices(model, active_matrix: str):
        for name, param in model.named_parameters():
            if 'lora_' in name:
                if f'lora_{active_matrix}' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
def fda_rolora_dual_track_pure_opt(hyperparams: dict):
    """
    Pure Closed-Loop FDA-RoLoRA Dual-Track Optimizer:
      1. Starts with Matrix B first (Round 1: B, Round 2: A, ...)
      2. No B-discount (NODISC): evaluations use true physical client drift variance.
      3. No e_min floor (e_min = 1).
      4. No first-round guard (natural cold-start bootstrap from initial Theta = 0.0).
      5. Configurable controller ratio setpoints: k = 1.0 (E_max) and k = 0.5 (E_max / 2).
    """
    e_min = int(hyperparams.get('e_min', 1))
    e_max = int(hyperparams.get('max_epochs', 20))
    controller_ratio = float(hyperparams.get('controller_ratio', 1.0))
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    if hyperparams.get('ds_name') == 'mnist':
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(rank=hyperparams.get('lora_r', 8)).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
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
        if not hyperparams.get('train_head', True):
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    num_lora = len(params_A)
    params_A = params_A + params_other
    params_B = params_B + params_other
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams, num_lora_params=num_lora)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams, num_lora_params=num_lora)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [p.detach().clone().to(get_save_device()) for p in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [p.detach().clone().to(get_save_device()) for p in params_B] for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    theta_A = 0.0
    theta_B = 0.0
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        current_theta = theta_A if active_matrix == 'A' else theta_B
        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
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
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
            client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
            var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
            if var_subspace > current_theta or round_epochs >= e_max:
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
        if active_matrix == 'A':
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
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics_handler.append_round_metrics(metrics)
        metrics_handler.save_metrics()
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished (Matrix {active_matrix}, Epochs: {round_epochs}, Ratio: {controller_ratio}, Pure NODISC).")
        gc.collect()
        if hyperparams.get('early_stopping'):
            from fdaopt.training import fda_opt
            early_stop_fn = getattr(fda_opt, 'check_dynamic_early_stopping', check_dynamic_early_stopping)
            if early_stop_fn(hyperparams, metrics_handler.round_metrics):
                print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
                break
    metrics_handler.save_metrics()

```


## File: `pure_experiments/thesis_pure_experiments.py`

```py
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
DEFAULT_SEEDS = [42]
DEFAULT_CLIENT_COUNTS = [20, 50]
PARTITION_ALPHA = 1000.0
DEFAULT_E_MIN = 1
DEFAULT_E_MAX = 20
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")
EARLY_STOPPING_MIN_ROUNDS = 6
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.005
EARLY_STOPPING_WINDOW = 3
EARLY_STOPPING_GATES = {
    "qnli": {20: 0.80, 50: 0.80},
    "mnli": {20: 0.60, 50: 0.55},
}
DATASET_COST_ORDER = ("qnli", "mnli")
DATASET_SETTINGS = {
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
JOBS_PER_GPU = max(1, int(os.environ.get("THESIS_PURE_JOBS_PER_GPU", "3")))
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
    ratio_val = float(config.get("controller_ratio", 1.0))
    ratio_order = 0 if ratio_val == 1.0 else 1
    return (
        int(config["num_clients"]),
        DATASET_COST_ORDER.index(config["ds_name"]),
        ratio_order,
        int(config["seed"]),
    )
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
        model = ToyModel(rank=hyperparams.get("lora_r", 8)).to(utils.DEVICE)
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
    sep = "=" * 105
    print(f"\n{sep}")
    print(f"PURE FDA-RoLoRA EXPERIMENT PLAN (Total Runs: {len(configs)})")
    print(f"Features: Matrix B first | e_min=1 | Zero first-round guard | NODISC")
    print(f"Queue Priority: All C=20 queued first (slots 1-4), C=50 fills remaining slots & backfill")
    print(sep)
    header = f"{'#':<3} | {'Control ID':<10} | {'DS':<5} | {'C':<3} | {'R':<3} | {'Ratio (k)':<9} | {'E-Range':<7} | {'Start':<5} | {'Guard':<5} | {'LR':<4} | {'Gate':<5} | {'Seed':<4}"
    print(header)
    print("-" * len(header))
    current_c = None
    for idx, cfg in enumerate(configs, 1):
        if current_c is not None and cfg["num_clients"] != current_c:
            print("-" * len(header))
        current_c = cfg["num_clients"]
        cid = base.get_control_id(cfg["extras"])
        ratio_str = f"k={cfg['controller_ratio']}"
        e_range = f"E{cfg['e_min']}..{cfg['max_epochs']}"
        start_mat = cfg.get("start_matrix", "B")
        guard_str = "None" if not cfg.get("first_round_guard", False) else "Round1"
        line = (
            f"{idx:<3} | {cid:<10} | {cfg['ds_name'].upper():<5} | {cfg['num_clients']:<3} | "
            f"{cfg['total_rounds']:<3} | {ratio_str:<9} | {e_range:<7} | {start_mat:<5} | {guard_str:<5} | "
            f"{cfg['client_opt_lr']:<4.2g} | {cfg['early_stopping_start_accuracy']:<5.0%} | {cfg['seed']:<4}"
        )
        print(line)
    print(f"{sep}\n")
def run_all(configs: list[dict]) -> None:
    """Invokes base scheduler to dispatch jobs across GPUs with 6 parallel slots:
    All C=20 runs are queued first (taking slots 1-4), remaining 2 slots are filled
    with C=50, and as slots become free they immediately backfill with the remaining C=50 runs.
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
        print(f"Launching scheduler across {JOBS_PER_GPU} lanes/GPU (6 parallel slots)...")
        print("Initial allocation: 4 slots -> C=20 (Runs 1-4), 2 slots -> C=50 (Runs 5-6).")
        print("Free slots will dynamically backfill with remaining C=50 runs (Runs 7-8).\n")
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
        "--datasets",
        nargs="+",
        default=["qnli", "mnli"],
        help="Datasets to evaluate (default: qnli mnli)",
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

```


## File: `toy/analyze_toy.py`

```py
"""
Automated Telemetry Analysis & Comparative Benchmarking for Pure Toy Model Experiments.
Parses:
  - 36 Baseline runs from results/toy_experiments/ (LoRA, FFA-LoRA, RoLoRA, E=5)
  - Pure FDA-RoLoRA Dual-Track runs from results/round_metrics/
  - Legacy FDA variants for historical comparison
Generates:
  - Markdown summary table with Mean ± Std of Final & Peak Accuracy, Total Epochs
  - High-resolution trajectory plots (Accuracy vs. Communication Rounds with error bands)
  - Accuracy vs. Cumulative Epochs efficiency curves
  - Detailed CSV metrics export
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.autolayout': True
})
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR if (CURRENT_DIR / "thesis_experiments.py").exists() else CURRENT_DIR.parent
RESULTS_DIRS = [
    REPO_ROOT / "results" / "toy_experiments",
    REPO_ROOT / "results" / "round_metrics"
]
OUT_DIR = REPO_ROOT / "analysis" / "toy_pure_analysis"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_SEEDS = {42, 43, 44}
def parse_run_file(filepath: Path) -> dict | None:
    filename = filepath.name
    if not filename.endswith('.csv'):
        return None
    arch = "LIN" if "_LIN_" in filename else "NONLIN"
    if "5-Clients-2-Classes" in filename:
        scenario = "5c2"
    elif "10-Clients-1-Class" in filename:
        scenario = "10c1"
    else:
        return None
    seed = None
    for part in filename.split('_'):
        if part.startswith('S') and part[1:].isdigit():
            seed = int(part[1:])
            break
    if seed not in ALLOWED_SEEDS:
        return None
    method = None
    if filename.startswith("TOY-PURE-FDA_PURE"):
        if "EMAX10" in filename:
            ratio_str = "k=0.5" if "CRATIO0.5" in filename else "k=1.0"
            method = f"FDA Pure ({ratio_str}, Emax=10)"
        elif "CRATIO0.5" in filename:
            method = "FDA Pure (k=0.5)"
        else:
            method = "FDA Pure (k=1.0)"
    elif filename.startswith("TOY-BASELINES"):
        if "FedAvg-LoRA" in filename:
            method = "LoRA"
        elif "FFA-LoRA" in filename:
            method = "FFA-LoRA"
        elif "FedAvg-RoLoRA" in filename:
            method = "RoLoRA"
    elif filename.startswith("TOY-NEW2FDA-FDA_STANDARD"):
        method = "FDA Standard (Legacy)"
    elif filename.startswith("TOY-NEW2FDA-FDA_EMA_STANDARD"):
        method = "FDA EMA (Legacy)"
    elif filename.startswith("TOY-NEW3FDA-FDA_SMA"):
        method = "FDA SMA (Legacy)"
    if method is None:
        return None
    run_hash = filename.split('_')[-1].replace('.csv', '')
    return {
        "filepath": filepath,
        "filename": filename,
        "arch": arch,
        "scenario": scenario,
        "seed": seed,
        "method": method,
        "run_hash": run_hash
    }
def main():
    print("=" * 80)
    print("ANALYZING TOY MODEL EXPERIMENTS: FDA PURE VS BASELINES")
    print("=" * 80)
    all_files = []
    for d in RESULTS_DIRS:
        if d.exists():
            all_files.extend(d.glob("*.csv"))
    parsed_runs = []
    seen_hashes = set()
    for f in all_files:
        meta = parse_run_file(f)
        if meta and meta["run_hash"] not in seen_hashes:
            seen_hashes.add(meta["run_hash"])
            try:
                df = pd.read_csv(meta["filepath"])
                if len(df) >= 100:
                    meta["df"] = df
                    parsed_runs.append(meta)
            except Exception as e:
                print(f"Error reading {f.name}: {e}")
    print(f"Loaded {len(parsed_runs)} verified complete 100-round runs.")
    groups = {}
    for r in parsed_runs:
        key = (r["scenario"], r["arch"], r["method"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    method_order = [
        "LoRA", "RoLoRA",
        "FDA Pure (k=0.5, Emax=10)", "FDA Pure (k=1.0, Emax=10)",
        "FDA Pure (k=0.5)", "FDA Pure (k=1.0)",
        "FFA-LoRA", "FDA Standard (Legacy)", "FDA EMA (Legacy)", "FDA SMA (Legacy)"
    ]
    scenarios = ["10c1", "5c2"]
    archs = ["NONLIN", "LIN"]
    summary_rows = []
    for sc in scenarios:
        for ar in archs:
            for m in method_order:
                key = (sc, ar, m)
                if key not in groups:
                    continue
                group_runs = groups[key]
                seeds_found = sorted([r["seed"] for r in group_runs])
                final_accs = [r["df"]["accuracy"].iloc[-1] * 100 for r in group_runs]
                max_accs = [r["df"]["accuracy"].max() * 100 for r in group_runs]
                total_epochs = [r["df"]["round_epochs"].sum() for r in group_runs]
                mean_epochs = [r["df"]["round_epochs"].mean() for r in group_runs]
                mean_a_epochs = [
                    r["df"][r["df"]["active_matrix"] == 'A']["round_epochs"].mean()
                    if 'active_matrix' in r["df"].columns and (r["df"]["active_matrix"] == 'A').any()
                    else r["df"]["round_epochs"].mean()
                    for r in group_runs
                ]
                mean_b_epochs = [
                    r["df"][r["df"]["active_matrix"] == 'B']["round_epochs"].mean()
                    if 'active_matrix' in r["df"].columns and (r["df"]["active_matrix"] == 'B').any()
                    else r["df"]["round_epochs"].mean()
                    for r in group_runs
                ]
                summary_rows.append({
                    "Scenario": sc,
                    "Arch": ar,
                    "Method": m,
                    "Seeds": f"{len(seeds_found)}/3 ({','.join(map(str, seeds_found))})",
                    "Final Acc Mean (%)": np.mean(final_accs),
                    "Final Acc Std (%)": np.std(final_accs) if len(final_accs) > 1 else 0.0,
                    "Max Acc Mean (%)": np.mean(max_accs),
                    "Max Acc Std (%)": np.std(max_accs) if len(max_accs) > 1 else 0.0,
                    "Mean Ep/Rd": np.mean(mean_epochs),
                    "Mean Ep (A)": np.mean(mean_a_epochs),
                    "Mean Ep (B)": np.mean(mean_b_epochs),
                    "Total Epochs Mean": np.mean(total_epochs),
                    "Total Epochs Std": np.std(total_epochs) if len(total_epochs) > 1 else 0.0,
                })
    summary_df = pd.DataFrame(summary_rows)
    csv_out = OUT_DIR / "toy_pure_summary_metrics.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"Saved summary CSV: {csv_out}")
    md_lines = [
        "# Toy Model Experiments: FDA Pure vs. Baselines Summary Report\n",
        "Authoritative benchmark comparison evaluating **FDA Pure** (`FDA-RoLoRA-DualTrack-Pure`) "
        "against the static federated baselines (**LoRA**, **FFA-LoRA**, **RoLoRA** with $E=5$) "
        "and legacy FDA variants across 4 extreme non-IID scenarios on MNIST ($d=784, r=16, c=10$).\n",
        "| Scenario | Arch | Method | Seeds | Final Acc (Mean ± Std %) | Max Acc (Mean ± Std %) | Mean Ep/Rd | Mean Ep (A) | Mean Ep (B) | Total Epochs |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| `{row['Scenario']}` | `{row['Arch']}` | **{row['Method']}** | {row['Seeds']} | "
            f"{row['Final Acc Mean (%)']:.2f} ± {row['Final Acc Std (%)']:.2f}% | "
            f"{row['Max Acc Mean (%)']:.2f} ± {row['Max Acc Std (%)']:.2f}% | "
            f"{row['Mean Ep/Rd']:.2f} | {row['Mean Ep (A)']:.2f} | {row['Mean Ep (B)']:.2f} | "
            f"{row['Total Epochs Mean']:.1f} ± {row['Total Epochs Std']:.1f} |"
        )
    md_out = OUT_DIR / "README.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Generated Markdown report: {md_out}")
    colors = {
        "LoRA": "#1f77b4",
        "FFA-LoRA": "#8c564b",
        "RoLoRA": "#ff7f0e",
        "FDA Pure (k=1.0)": "#d62728",
        "FDA Pure (k=0.5)": "#9467bd",
        "FDA Pure (k=0.5, Emax=10)": "#2ca02c",
        "FDA Pure (k=1.0, Emax=10)": "#17becf",
        "FDA Standard (Legacy)": "#bcbd22",
        "FDA EMA (Legacy)": "#e377c2",
        "FDA SMA (Legacy)": "#7f7f7f"
    }
    for sc in scenarios:
        for ar in archs:
            plt.figure(figsize=(9, 6))
            plotted = False
            for m in method_order:
                key = (sc, ar, m)
                if key not in groups:
                    continue
                group_runs = groups[key]
                all_accs = np.array([r["df"]["accuracy"].values[:100] * 100 for r in group_runs])
                mean_acc = np.mean(all_accs, axis=0)
                std_acc = np.std(all_accs, axis=0) if len(group_runs) > 1 else np.zeros_like(mean_acc)
                rounds = np.arange(1, len(mean_acc) + 1)
                c = colors.get(m, "#333333")
                lw = 2.5 if "Pure" in m else 1.8
                ls = '-' if "Pure" in m or m in ["LoRA", "RoLoRA", "FFA-LoRA"] else '--'
                plt.plot(rounds, mean_acc, label=f"{m} (N={len(group_runs)})", color=c, linewidth=lw, linestyle=ls)
                plt.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, color=c, alpha=0.15)
                plotted = True
            if plotted:
                plt.title(f"Toy Model Accuracy vs Communication Rounds ({sc.upper()} - {ar})", fontsize=13, fontweight='bold')
                plt.xlabel("Communication Round", fontsize=11)
                plt.ylabel("Test Accuracy (%)", fontsize=11)
                plt.ylim(0, 100)
                plt.legend(loc="lower right", frameon=True)
                fig_path = FIG_DIR / f"trajectory_{sc}_{ar}.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved figure: {fig_path.name}")
    print("\nAnalysis suite executed successfully!")
if __name__ == '__main__':
    main()

```


## File: `toy/thesis_toy_experiments.py`

```py
"""
Master Orchestrator for Pure FDA-RoLoRA Dual-Track Toy Model Experiments.
Experimental Protocol (matching rolora_verr4.pdf & pure design):
  1. Alternation starts with Matrix B first at Round 1 (r=0 -> B, r=1 -> A).
  2. B initialized ~ N(0, 1e-4) to prevent dead ReLU neurons; A ~ N(0, 1e-4); W_out frozen.
  3. Pure physical inter-client drift variance (NODISC, gamma=1.0, zero heuristic discount).
  4. Dynamic local epoch budget: e_min = 1 to e_max = 15.
  5. Natural cold-start bootstrap from initial Theta_A = 0.0, Theta_B = 0.0 (NOGUARD).
  6. Closed-loop proportional controller setpoint: target = kappa * e_max (kappa in {1.0, 0.5}).
  7. 4 Non-IID Scenarios on MNIST (d=784, r=16, c=10):
     - 10-Clients-1-Class_NONLIN
     - 10-Clients-1-Class_LIN
     - 5-Clients-2-Classes_NONLIN
     - 5-Clients-2-Classes_LIN
  8. Seeds: {42, 43, 44}. Total rounds: 100.
  9. Multi-GPU parallel scheduler (auto-detects GPUs, manages run registry, process control, logs).
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import torch
import pandas as pd
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR if (CURRENT_DIR / "thesis_experiments.py").exists() else CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from fdaopt import utils
from fdaopt.models.toy_model import ToyModel
from fdaopt.datasets.fed_data_prep import prepare_federated_datasets
from fdaopt.models.ops import (
    compute_client_drifts,
    variance,
    compute_pseudo_gradients,
    copy_parameters,
    set_gradients,
    get_save_device
)
from fdaopt.training.optimizers import server_client_optimizers
from fdaopt.metrics.mentrics_handler import MetricsHandler
RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "round_metrics"
LOG_DIR = RESULTS_DIR / "logs" / "thesis_toy"
RUN_CONTROL_DIR = RESULTS_DIR / "run_control_toy"
ACTIVE_RUNS_DIR = RUN_CONTROL_DIR / "active"
STOP_REQUESTS_DIR = RUN_CONTROL_DIR / "stop_requests"
DROPPED_RUNS_DIR = RUN_CONTROL_DIR / "dropped"
OPERATOR_EVENTS_FILE = RUN_CONTROL_DIR / "operator_events.jsonl"
SCHEDULER_LOCK_FILE = RUN_CONTROL_DIR / "scheduler.lock"
for d in (METRICS_DIR, LOG_DIR, ACTIVE_RUNS_DIR, STOP_REQUESTS_DIR, DROPPED_RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)
METHOD_NAME = "FDA-RoLoRA-DualTrack-Pure"
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_RATIOS = [1.0, 0.5]
DEFAULT_E_MIN = 1
DEFAULT_E_MAX = 15
TOTAL_ROUNDS = 100
BATCH_SIZE = 64
LORA_RANK = 16
SCENARIOS = [
    {"scenario": "10-Clients-1-Class", "linear": False, "num_clients": 10, "arch": "NONLIN"},
    {"scenario": "10-Clients-1-Class", "linear": True,  "num_clients": 10, "arch": "LIN"},
    {"scenario": "5-Clients-2-Classes", "linear": False, "num_clients": 5,  "arch": "NONLIN"},
    {"scenario": "5-Clients-2-Classes", "linear": True,  "num_clients": 5,  "arch": "LIN"},
]
JOBS_PER_GPU = max(1, int(os.environ.get("TOY_JOBS_PER_GPU", "4")))
def get_control_id(run_id: str) -> str:
    """Return a short, stable 10-char operator control ID."""
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
def toggle_rolora_matrices(model, active_matrix: str):
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = f'lora_{active_matrix}' in name
def build_run_id(scenario: str, arch: str, ratio: float, seed: int, e_max: int = DEFAULT_E_MAX) -> str:
    emax_tag = f"_EMAX{e_max}" if e_max != 15 else ""
    return f"TOY-PURE-FDA_PURE_FDA-RoLoRA-DualTrack-Pure_{scenario}_{arch}{emax_tag}_CRATIO{ratio}_S{seed}"
def build_configs(ratios: list[float] = DEFAULT_RATIOS, seeds: list[int] = DEFAULT_SEEDS, e_max: int = DEFAULT_E_MAX) -> list[dict]:
    configs = []
    for ratio in ratios:
        for scen in SCENARIOS:
            for seed in seeds:
                run_id = build_run_id(scen["scenario"], scen["arch"], ratio, seed, e_max=e_max)
                cid = get_control_id(run_id)
                configs.append({
                    "scenario": scen["scenario"],
                    "arch": scen["arch"],
                    "linear": scen["linear"],
                    "num_clients": scen["num_clients"],
                    "controller_ratio": ratio,
                    "seed": seed,
                    "run_id": run_id,
                    "control_id": cid,
                    "total_rounds": TOTAL_ROUNDS,
                    "e_min": DEFAULT_E_MIN,
                    "e_max": e_max,
                    "batch_size": BATCH_SIZE,
                    "lora_r": LORA_RANK
                })
    return configs
def find_completed_csv(config: dict) -> Path | None:
    scenario = config["scenario"]
    arch = config["arch"]
    ratio = config["controller_ratio"]
    seed = config["seed"]
    e_max = config.get("e_max", DEFAULT_E_MAX)
    emax_tag = f"_EMAX{e_max}" if e_max != 15 else ""
    if e_max == 15:
        pattern = str(METRICS_DIR / f"TOY-PURE-FDA_PURE*_{scenario}_{arch}_CRATIO{ratio}_S{seed}_*.csv")
    else:
        pattern = str(METRICS_DIR / f"TOY-PURE-FDA_PURE*_{scenario}_{arch}{emax_tag}_CRATIO{ratio}_S{seed}_*.csv")
    for f in glob.glob(pattern):
        if e_max == 15 and "_EMAX" in Path(f).name and "_EMAX15" not in Path(f).name:
            continue
        try:
            df = pd.read_csv(f)
            if len(df) >= config["total_rounds"]:
                return Path(f)
        except Exception:
            pass
    return None
def is_dropped(control_id: str) -> bool:
    return (DROPPED_RUNS_DIR / f"{control_id}.json").exists()
def read_latest_metric_line(log_path: Path, max_bytes: int = 131072) -> str:
    if not log_path.exists():
        return "waiting for first evaluation"
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            f.seek(max(0, end - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return "waiting for first evaluation"
    for line in reversed(text.splitlines()):
        if "[METRIC]" in line:
            return line[line.index("[METRIC]"):].strip()
        if "Round " in line and "finished" in line:
            return line.strip()
    return "running"
def load_active_records() -> list[dict]:
    records = []
    if not ACTIVE_RUNS_DIR.exists():
        return records
    for path in sorted(ACTIVE_RUNS_DIR.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            pid = int(record["pid"])
            os.kill(pid, 0)
            records.append(record)
        except (OSError, ProcessLookupError, ValueError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
    return records
def print_plan(configs: list[dict]):
    print("=" * 96)
    print("PURE FDA-RoLoRA TOY MODEL EXPERIMENT PLAN (Polytechnix Deployment)")
    print("=" * 96)
    completed_count = 0
    dropped_count = 0
    pending_count = 0
    for cfg in configs:
        cid = cfg["control_id"]
        csv_path = find_completed_csv(cfg)
        if csv_path:
            status = f"DONE ({csv_path.name})"
            completed_count += 1
        elif is_dropped(cid):
            status = "DROPPED"
            dropped_count += 1
        else:
            status = "PENDING"
            pending_count += 1
        target_setpoint = cfg['controller_ratio'] * cfg['e_max']
        print(f"  [{cid}] [{status:12s}] Ratio={cfg['controller_ratio']:3.1f} | Emax={cfg['e_max']:2d} (Target={target_setpoint:4.1f}) | "
              f"{cfg['scenario']:19s} | {cfg['arch']:6s} | Seed {cfg['seed']}")
    print("-" * 96)
    print(f"Total: {len(configs)} runs | Completed: {completed_count} | Dropped: {dropped_count} | Pending: {pending_count}")
    print(f"Estimated Execution Time on Polytechnix (2x A10): ~{pending_count * 2.5:.1f} minutes")
    print("=" * 96)
def print_status(configs: list[dict]):
    print("=" * 96)
    print("PURE FDA-RoLoRA TOY MODEL RUN STATUS")
    print("=" * 96)
    active = {r["control_id"]: r for r in load_active_records()}
    for cfg in configs:
        cid = cfg["control_id"]
        csv_path = find_completed_csv(cfg)
        if csv_path:
            state = f"DONE ({csv_path.name})"
        elif is_dropped(cid):
            state = "DROPPED"
        elif cid in active:
            rec = active[cid]
            log_p = Path(rec.get("stdout_path", ""))
            metric = read_latest_metric_line(log_p)
            state = f"RUNNING [pid={rec['pid']} on {rec['device']}]: {metric}"
        else:
            state = "QUEUED"
        target_setpoint = cfg['controller_ratio'] * cfg['e_max']
        print(f"  [{cid}] Ratio={cfg['controller_ratio']:3.1f} | Emax={cfg['e_max']:2d} (Target={target_setpoint:4.1f}) | {cfg['scenario']:19s} | {cfg['arch']:6s} | S{cfg['seed']} -> {state}")
    print("=" * 96)
def request_stop(run_or_control_id: str, reason: str = "operator requested stop"):
    cid = run_or_control_id if len(run_or_control_id) == 10 else get_control_id(run_or_control_id)
    reg_path = ACTIVE_RUNS_DIR / f"{cid}.json"
    if not reg_path.exists():
        print(f"No active run found for ID: {cid}")
        return
    try:
        record = json.loads(reg_path.read_text(encoding="utf-8"))
        pid = int(record["pid"])
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to process {pid} [{cid}] ({reason})")
    except Exception as e:
        print(f"Error stopping {cid}: {e}")
    reg_path.unlink(missing_ok=True)
def request_drop(run_or_control_id: str, reason: str = "operator dropped run"):
    cid = run_or_control_id if len(run_or_control_id) == 10 else get_control_id(run_or_control_id)
    request_stop(cid, reason=f"drop: {reason}")
    drop_file = DROPPED_RUNS_DIR / f"{cid}.json"
    drop_file.write_text(json.dumps({"control_id": cid, "reason": reason, "timestamp": time.time()}))
    print(f"Marked {cid} as DROPPED permanently.")
def execute_worker(config: dict, device_str: str):
    cid = config["control_id"]
    scenario = config["scenario"]
    arch = config["arch"]
    linear = config["linear"]
    num_clients = config["num_clients"]
    controller_ratio = config["controller_ratio"]
    seed = config["seed"]
    total_rounds = config["total_rounds"]
    e_min = config["e_min"]
    e_max = config["e_max"]
    batch_size = config["batch_size"]
    rank = config["lora_r"]
    device = torch.device(device_str)
    utils.set_device(device)
    utils.set_seed(seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    emax_tag = f"_EMAX{e_max}" if e_max != 15 else ""
    extras = f"TOY-PURE-FDA_PURE-{timestamp}_FDA-RoLoRA-DualTrack-Pure_{scenario}_{arch}{emax_tag}_CRATIO{controller_ratio}_S{seed}"
    hyperparams = {
        'checkpoint': 'toy-model',
        'ds_path': 'ylecun/mnist',
        'ds_name': 'mnist',
        'num_labels': 10,
        'num_clients': num_clients,
        'clients_per_round': num_clients,
        'total_rounds': total_rounds,
        'local_epochs': 1,
        'e_min': e_min,
        'max_epochs': e_max,
        'controller_ratio': controller_ratio,
        'batch_size': batch_size,
        'num_workers': 0,
        'pin_memory': True,
        'alpha': 0.0,
        'server_opt_name': 'SGD',
        'server_opt_lr': 1.0,
        'client_opt_name': 'SGD',
        'client_opt_lr': 0.01,
        'lora_r': rank,
        'linear': linear,
        'extras': extras,
        'method': METHOD_NAME,
        'METHOD_NAME': METHOD_NAME,
        'early_stopping': False,
        'seed': seed,
        'enable_b_discount': False,
        'b_discount_factor': 1.0,
        'start_matrix': 'B',
        'first_round_guard': False,
    }
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams['num_workers'], hyperparams['pin_memory']
    )
    client_gpu_data = {}
    steps_per_epoch = {}
    for c_id in range(num_clients):
        dl = fed_ds.client_dataloaders[c_id]
        all_imgs, all_lbls = [], []
        for b in dl:
            all_imgs.append(b['image'])
            all_lbls.append(b['labels'])
        imgs_cat = torch.cat(all_imgs, dim=0).to(device)
        lbls_cat = torch.cat(all_lbls, dim=0).to(device)
        client_gpu_data[c_id] = (
            imgs_cat.view(imgs_cat.size(0), -1).float() / 255.0,
            lbls_cat.long()
        )
        steps_per_epoch[c_id] = len(dl)
    test_dl = test_ds['mnist']
    test_imgs, test_lbls = [], []
    for b in test_dl:
        test_imgs.append(b['image'])
        test_lbls.append(b['labels'])
    t_imgs = torch.cat(test_imgs, dim=0).to(device)
    t_lbls = torch.cat(test_lbls, dim=0).to(device)
    gpu_test_x = t_imgs.view(t_imgs.size(0), -1).float() / 255.0
    gpu_test_y = t_lbls.long()
    model = ToyModel(rank=rank, linear=linear).to(device)
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
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
    print(f"[{time.strftime('%H:%M:%S')}] >>> WORKER STARTED [{cid}]: {extras} on {device_str} <<<", flush=True)
    t_start = time.time()
    rng_gen = torch.Generator(device=device)
    rng_gen.manual_seed(seed)
    for r in range(total_rounds):
        r_start = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        current_theta = theta_A if active_matrix == 'A' else theta_B
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
            sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
            client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
            var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
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
        target_epochs = controller_ratio * e_max
        next_theta = (target_epochs / round_epochs) * var_subspace
        if active_matrix == 'A':
            theta_A = next_theta
        else:
            theta_B = next_theta
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
            "method": METHOD_NAME,
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
            "training_accuracy": training_accuracy / num_clients
        }
        metrics_handler.append_round_metrics(metrics)
        print(f"[METRIC] round={r+1:3d}/{total_rounds} matrix={active_matrix} epochs={round_epochs:2d} "
              f"acc={test_acc*100:5.2f}% loss={test_loss:7.4f} var={var_subspace:7.3f} "
              f"theta={next_theta:7.3f} time={round_end-r_start:5.2f}s", flush=True)
    metrics_handler.save_metrics()
    elapsed = time.time() - t_start
    print(f"[{time.strftime('%H:%M:%S')}] [DONE] [{cid}] finished in {elapsed/60:.2f} min ({elapsed:.1f}s) -> {metrics_handler.filename}", flush=True)
def get_available_devices() -> list[str]:
    if not torch.cuda.is_available():
        return ["cpu"]
    count = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(count)]
def run_scheduler(configs: list[dict]):
    devices = get_available_devices()
    print(f"=== Starting Toy Model Scheduler across devices: {devices} (max {JOBS_PER_GPU} jobs/GPU) ===", flush=True)
    queue_configs = []
    for cfg in configs:
        cid = cfg["control_id"]
        if find_completed_csv(cfg):
            continue
        if is_dropped(cid):
            continue
        queue_configs.append(cfg)
    if not queue_configs:
        print("All configurations are already completed or dropped!", flush=True)
        return
    print(f"Queued {len(queue_configs)} pending runs for dispatch.", flush=True)
    device_slots = {dev: [] for dev in devices}
    cfg_idx = 0
    while cfg_idx < len(queue_configs) or any(len(slots) > 0 for slots in device_slots.values()):
        for dev, slots in device_slots.items():
            surviving = []
            for item in slots:
                proc = item["proc"]
                cfg = item["config"]
                cid = cfg["control_id"]
                ret = proc.poll()
                if ret is None:
                    surviving.append(item)
                else:
                    item["log_file"].close()
                    (ACTIVE_RUNS_DIR / f"{cid}.json").unlink(missing_ok=True)
                    if ret == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] Worker [{cid}] on {dev} COMPLETED successfully.", flush=True)
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] Worker [{cid}] on {dev} FAILED with return code {ret}!", flush=True)
            device_slots[dev] = surviving
        for dev in devices:
            while len(device_slots[dev]) < JOBS_PER_GPU and cfg_idx < len(queue_configs):
                cfg = queue_configs[cfg_idx]
                cid = cfg["control_id"]
                cfg_idx += 1
                log_path = LOG_DIR / f"{cid}.stdout.log"
                log_file = log_path.open("w", encoding="utf-8")
                cmd = [
                    sys.executable, "-u", str(Path(__file__).resolve()),
                    "--worker",
                    "--config-json", json.dumps(cfg),
                    "--device", dev
                ]
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                record = {
                    "control_id": cid,
                    "run_id": cfg["run_id"],
                    "pid": proc.pid,
                    "device": dev,
                    "started_at": time.time(),
                    "stdout_path": str(log_path)
                }
                (ACTIVE_RUNS_DIR / f"{cid}.json").write_text(json.dumps(record, indent=2))
                device_slots[dev].append({
                    "proc": proc,
                    "config": cfg,
                    "log_file": log_file,
                    "start_time": time.time()
                })
                print(f"[{time.strftime('%H:%M:%S')}] Dispatched [{cid}] to {dev} (pid={proc.pid}) -> {log_path.name}", flush=True)
        time.sleep(2)
    print("\n=== All queued Toy Model configurations have completed! ===", flush=True)
def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator for Pure FDA-RoLoRA Toy Model Experiments")
    parser.add_argument("--group", choices=["plan", "run", "start", "status", "stop", "drop"], default="plan", help="Action to execute")
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS, help="Controller ratios (default: 1.0 0.5)")
    parser.add_argument("--emax", "--e-max", type=int, default=DEFAULT_E_MAX, help=f"Max local epochs ceiling e_max (default: {DEFAULT_E_MAX})")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Seeds to evaluate (default: 42 43 44)")
    parser.add_argument("--run-id", type=str, default="", help="Target Run ID or Control ID for stop/drop")
    parser.add_argument("--reason", type=str, default="", help="Reason for stop/drop action")
    parser.add_argument("--worker", action="store_true", help="Internal flag: launch as single worker")
    parser.add_argument("--config-json", type=str, default="", help="Internal flag: worker configuration payload")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device for worker")
    args = parser.parse_args()
    if args.worker:
        if not args.config_json:
            print("Error: --config-json payload required in --worker mode.")
            sys.exit(1)
        cfg = json.loads(args.config_json)
        execute_worker(cfg, args.device)
        return
    configs = build_configs(ratios=args.ratios, seeds=args.seeds, e_max=args.emax)
    if args.group == "plan":
        print_plan(configs)
    elif args.group in ["run", "start"]:
        run_scheduler(configs)
    elif args.group == "status":
        print_status(configs)
    elif args.group == "stop":
        if not args.run_id:
            print("Error: --run-id required for stop.")
            sys.exit(1)
        request_stop(args.run_id, args.reason or "operator stop")
    elif args.group == "drop":
        if not args.run_id:
            print("Error: --run-id required for drop.")
            sys.exit(1)
        request_drop(args.run_id, args.reason or "operator drop")
if __name__ == "__main__":
    main()

```


## File: `fdaopt/utils.py`

```py
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()
import evaluate
DEVICE = None
DEVICE_RAM_PROGRAM = 'moderate'
def set_seed(seed):
    """
    Set the seed for reproducibility.
    Args:
        seed (int): The seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
def set_device(device):
    global DEVICE
    DEVICE = device

```


## File: `fdaopt/main.py`

```py
import pprint
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help="The filename of the hyperparameters.")
    parser.add_argument('--device', type=str, help="The device to use (e.g. 'cuda:1').")
    parser.add_argument('--seed', type=int, default=42, help="The seed for reproducibility.")
    args = parser.parse_args()
    from fdaopt import utils
    utils.set_seed(args.seed)
    utils.set_device(args.device)
    from fdaopt.training.fed_opt import fed_opt
    from fdaopt.training.fda_opt import fda_opt
    from fdaopt.miscellaneous.parameter_tools import get_hyperparameters
    hyperparams = get_hyperparameters(args.filename)
    print(f"Using device: {utils.DEVICE}")
    pprint.pprint(hyperparams, sort_dicts=False)
    print()
    if hyperparams['fda']:
        fda_opt(hyperparams)
    else:
        fed_opt(hyperparams)
    print("Finished Testing!")

```


## File: `fdaopt/models/toy_model.py`

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
class ToyModelOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits
class ToyModel(nn.Module):
    """
    Two-layer toy model matching the paper's formulation:
        f_i(x_i) = ReLU(x_i @ A @ B) @ W_out
    where A in R^{d x r}, B in R^{r x d}, W_out in R^{d x c} is fixed.
    Fixes vs. the original implementation:
      1. `lora_B` is no longer initialized to exact zeros. With ReLU active
         (linear=False), an all-zero pre-activation makes ReLU'(0) = 0
         everywhere, which zeroes out the gradient for both lora_A and
         lora_B on every step -- the model can never leave initialization.
         A small-scale random init preserves LoRA's "near-identity at
         init" spirit without killing the gradient.
      2. `lora_B` now maps back up to `input_dim` (not `output_dim`), and
         `W_out` is a genuine fixed (d x c) projection rather than a
         (c x c) identity, matching the paper's two-stage
         bottleneck -> full representation -> class logits pipeline.
    """
    def __init__(self, input_dim=784, rank=16, output_dim=10, linear=False):
        super().__init__()
        self.linear = linear
        self.lora_A = nn.Parameter(torch.randn(input_dim, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, input_dim) * 1e-4)
        self.W_out = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=False)
    def forward(self, image=None, input_ids=None, labels=None, **kwargs):
        x = image if image is not None else input_ids
        x = x.view(x.size(0), -1).float() / 255.0
        hidden = x @ self.lora_A @ self.lora_B
        if not self.linear:
            hidden = F.relu(hidden)
        logits = hidden @ self.W_out
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())
        return ToyModelOutput(loss, logits)

```


## File: `fdaopt/models/ops.py`

```py
from fdaopt import utils
import torch
import evaluate
import math
_METRICS_CACHE = {}
def get_save_device():
    if utils.DEVICE_RAM_PROGRAM == 'performance':
        return utils.DEVICE
    elif utils.DEVICE_RAM_PROGRAM == 'moderate' or utils.DEVICE_RAM_PROGRAM == 'low':
        return 'cpu'
    return 'cpu'
@torch.no_grad
def copy_parameters(from_params, to_params):
    """
    Copies the values from one set of parameters to another.
    This function operates in-place and modifies the `to_parameters` directly.
    The @torch.no_grad() decorator ensures that this operation is not tracked
    by autograd, preventing unnecessary computation and memory usage.
    Args:
        from_params (list of torch.nn.Parameter): An iterable of source parameters to copy from.
        to_params (list of torch.nn.Parameter): An iterable of destination parameters to copy to.
    """
    for from_param, to_param in zip(from_params, to_params):
        device = to_param.device
        to_param.copy_(from_param.to(device))
@torch.no_grad
def average_client_parameters(client_train_params_dict):
    """
    Efficiently averages parameters from multiple clients on GPU.
    This function uses an incremental mean calculation to avoid creating large
    intermediate tensors, performing in-place additions to keep memory usage low.
    Args:
        client_train_params_dict (dict): Dictionary where keys are client IDs and values are lists of parameter tensors.
    Returns:
        list: A list of averaged parameters, stored on `utils.DEVICE`.
    """
    num_clients = len(client_train_params_dict)
    average_params = None
    for train_params in client_train_params_dict.values():
        if not average_params:
            average_params = [param.clone().to(utils.DEVICE) / num_clients for param in train_params]
        else:
            for param, avg_param in zip(train_params, average_params):
                avg_param.add_(param.to(utils.DEVICE) / num_clients)
    return average_params
@torch.no_grad
def average_client_parameters2(client_train_params_dict):
    """
    Averages the parameters from multiple clients.
    This function computes the mean of the parameters from all clients. It stacks
    the parameters for each layer across clients, computes the mean, and returns
    the averaged parameters. The @torch.no_grad decorator ensures that this operation
    is not tracked by autograd.
    Args:
        client_train_params_dict (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
    Returns:
        list: A list of averaged parameters which lies on `utils.DEVICE`.
    """
    average_params = [
        torch.mean(torch.stack(param_list), dim=0)
        for param_list in zip(*client_train_params_dict.values())
    ]
    return average_params
@torch.no_grad
def compute_metrics(model, ds_path, ds_name, test_ds_dict):
    def _compute_metrics(_model, _ds_path, _config_name, _test_ds):
        is_mnist = 'mnist' in _ds_path or 'mnist' in _config_name
        if not is_mnist:
            cache_key = f"{_ds_path}_{_config_name}"
            if cache_key not in _METRICS_CACHE:
                actual_metric_path = "glue" if "glue" in _ds_path else _ds_path
                _METRICS_CACHE[cache_key] = evaluate.load(
                    path=actual_metric_path,
                    config_name=_config_name,
                    keep_in_memory=True
                )
            metric = _METRICS_CACHE[cache_key]
        testing_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_batches = len(_test_ds)
        _model.eval()
        for batch in _test_ds:
            batch = {key: value.to(utils.DEVICE) for key, value in batch.items()}
            outputs = _model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            testing_loss += outputs.loss.item()
            if is_mnist:
                correct_predictions += (
                    predictions == batch["labels"]
                ).sum().item()
                total_predictions += batch["labels"].size(0)
            else:
                metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"]
                )
        average_test_loss = testing_loss / num_batches
        if is_mnist:
            return {
                "accuracy": correct_predictions / total_predictions,
                "testing_loss": average_test_loss
            }
        metrics = metric.compute()
        metrics["testing_loss"] = average_test_loss
        return metrics
    if ds_path == 'glue' or ds_path == 'nyu-mll/glue':
        if ds_name == 'mnli':
            metrics_m = _compute_metrics(model, ds_path, 'mnli_matched', test_ds_dict['mnli_matched'])
            metrics_m = {k + "_m": v for k, v in metrics_m.items()}
            metrics_mm = _compute_metrics(model, ds_path, 'mnli_mismatched', test_ds_dict['mnli_mismatched'])
            metrics_mm = {k + "_mm": v for k, v in metrics_mm.items()}
            return {**metrics_m, **metrics_mm}
        else:
            return _compute_metrics(model, ds_path, ds_name, test_ds_dict[ds_name])
    else:
        target_name = ds_name if ds_name in test_ds_dict else list(test_ds_dict.keys())[0]
        return _compute_metrics(model, ds_path, target_name, test_ds_dict[target_name])
@torch.no_grad
def compute_drifts(old_params, new_params):
    """
    Calculate the drift (difference) between old and new parameters.
    Args:
        old_params (list of torch.nn.Parameter): The original parameters.
        new_params (list of torch.nn.Parameter): The updated parameters.
    Returns:
        list of torch.Tensor: The computed drifts for each parameter stored in SAVE_DEVICE.
    """
    drifts = []
    for old_param, new_param in zip(old_params, new_params):
        old_param = old_param.to(utils.DEVICE)
        new_param = new_param.to(utils.DEVICE)
        drifts.append(
            (new_param - old_param).to(get_save_device())
        )
    return drifts
@torch.no_grad
def compute_client_drifts(old_params, client_train_params_dict):
    """
    Compute the drifts for all clients based on the original parameters.
    Args:
        old_params (list of torch.nn.Parameter): The original parameters.
        client_train_params_dict (dict): Dictionary of client IDs and their corresponding parameters.
    Returns:
        dict: A dictionary where keys are client IDs and values are lists of drifts for each parameter stored in DEVICE_SAVE.
    """
    return {
        client_id: compute_drifts(old_params, client_params)
        for client_id, client_params in client_train_params_dict.items()
    }
@torch.no_grad
def compute_pseudo_gradients(client_drifts):
    """
    Compute the pseudo-gradient based on the drifts between old and client parameters.
    Args:
        client_drifts (dict): Dictionary of client IDs and their corresponding parameter drifts.
    Returns:
        list of torch.Tensor: The computed pseudo-gradient with the result lying on utils.DEVICE.
    """
    average_drifts = average_client_parameters(client_drifts)
    pseudo_gradient = [-drift for drift in average_drifts]
    return pseudo_gradient
@torch.no_grad
def set_gradients(train_params, gradients):
    """
    Set gradients for trainable parameters.
    This function assigns the provided gradients to the .grad attribute of the corresponding trainable parameters.
    Args:
        train_params (list of torch.nn.Parameter): The trainable parameters of the model.
        gradients (list of torch.Tensor): The gradients to be assigned to the parameters.
    """
    for param, gradient in zip(train_params, gradients):
        device = param.device
        gradient = gradient.to(device)
        param.grad = gradient
@torch.no_grad
def vectorize(parameters):
    """
    Concatenates a list of parameter tensors into a single vector.
    Args:
        parameters (list of torch.nn.Parameter): An iterable of parameter tensors.
    Returns:
        torch.Tensor: A single vector containing all the elements of the input parameters with the
                      result lying on `utils.DEVICE`.
    """
    con = []
    for param in parameters:
        param = param.to(utils.DEVICE)
        con.append(param.view(-1))
    return torch.cat(con)
@torch.no_grad()
def get_ksi_unit(w_t0, w_tminus1, generator=None):
    """
    Calculates the heuristic unit vector (ksi) as a list of tensors
    matching the model's parameter structure, avoiding huge allocations.
    """
    ksi_tensors = []
    total_norm_sq = 0.0
    if w_tminus1 is None:
        for param in w_t0:
            rand_tensor = torch.randn(param.shape, dtype=param.dtype, device=utils.DEVICE, generator=generator)
            ksi_tensors.append(rand_tensor)
            total_norm_sq += torch.sum(rand_tensor ** 2).item()
    else:
        for p_t0, p_tminus1 in zip(w_t0, w_tminus1):
            diff = p_t0.to(utils.DEVICE) - p_tminus1.to(utils.DEVICE)
            ksi_tensors.append(diff)
            total_norm_sq += torch.sum(diff ** 2).item()
    norm = math.sqrt(total_norm_sq)
    if norm > 0.0:
        for ksi_tensor in ksi_tensors:
            ksi_tensor.div_(norm)
    return ksi_tensors
@torch.no_grad()
def linear_variance_approx(client_drifts, ksi_vector):
    """
    Computes variance approximation iteratively without vectorizing,
    saving significant VRAM and preventing memory fragmentation.
    """
    num_clients = len(client_drifts)
    sum_sq_norms = 0.0
    for drifts in client_drifts.values():
        norm_sq_drift = 0.0
        for drift_param in drifts:
            norm_sq_drift += torch.sum(drift_param.to(utils.DEVICE) ** 2).item()
        sum_sq_norms += norm_sq_drift
    S_1 = sum_sq_norms / num_clients
    S_2_sq = 0.0
    first_client_id = list(client_drifts.keys())[0]
    num_params = len(client_drifts[first_client_id])
    for param_idx in range(num_params):
        avg_param = None
        for client_id, drifts in client_drifts.items():
            if avg_param is None:
                avg_param = drifts[param_idx].clone().to(utils.DEVICE) / num_clients
            else:
                avg_param.add_(drifts[param_idx].to(utils.DEVICE) / num_clients)
        S_2_sq += torch.sum(avg_param ** 2).item()
    return S_1 - S_2_sq
def linear_variance_approx2(client_drifts, global_current, global_prev):
    """
    Memory-efficient 1:1 port of the FedL-Sync-FDA linear variance approximation.
    Args:
        client_drifts: List of state_dicts (or dicts of tensors) representing Delta_i for each client.
        global_current: state_dict of the current global model (w_t0).
        global_prev: state_dict of the previous global model (w_tminus1).
    Returns:
        float: The estimated variance.
    """
    num_clients = len(client_drifts)
    if num_clients == 0: return 0.0
    sum_sq_norms = 0.0
    for delta_i in client_drifts:
        client_sq_norm = 0.0
        for k in delta_i.keys():
            client_sq_norm += torch.sum(delta_i[k] ** 2).item()
        sum_sq_norms += client_sq_norm
    S_1 = sum_sq_norms / num_clients
    S_2_sq = 0.0
    keys = client_drifts[0].keys()
    for k in keys:
        avg_k = None
        for delta_i in client_drifts:
            if avg_k is None:
                avg_k = delta_i[k].clone() / num_clients
            else:
                avg_k.add_(delta_i[k] / num_clients)
        S_2_sq += torch.sum(avg_k ** 2).item()
    variance_approx = S_1 - S_2_sq
    return variance_approx
def get_ksi_unit2(global_current, global_prev):
    """
    Generates the projection vector ksi as the normalized difference
    between the current and previous global models, avoiding tensor flattening.
    """
    ksi = {}
    diff_norm_sq = 0.0
    for k in global_current.keys():
        diff = global_current[k] - global_prev[k]
        ksi[k] = diff
        diff_norm_sq += torch.sum(diff ** 2).item()
    diff_norm = math.sqrt(diff_norm_sq)
    if diff_norm == 0.0:
        rand_norm_sq = 0.0
        for k in ksi.keys():
            ksi[k] = torch.randn_like(ksi[k])
            rand_norm_sq += torch.sum(ksi[k] ** 2).item()
        rand_norm = math.sqrt(rand_norm_sq)
        for k in ksi.keys():
            ksi[k] = ksi[k] / rand_norm
    else:
        for k in ksi.keys():
            ksi[k] = ksi[k] / diff_norm
    return ksi
@torch.no_grad
def variance(client_drifts):
    """
    Computes the variance of the client models utilizing the drifts (see paper)
    Args:
        client_drifts (dict): A dictionary where keys are client IDs and values are lists of parameter tensors (drifts).
    Returns:
        float: The computed variance of the client drifts.
    """
    norm_sq_drifts = []
    for drifts in client_drifts.values():
        drift_vec = vectorize(drifts)
        norm_sq_drift = torch.dot(drift_vec, drift_vec)
        norm_sq_drifts.append(norm_sq_drift)
    avg_norm_sq_drifts = sum(norm_sq_drifts) / len(norm_sq_drifts)
    avg_drift = average_client_parameters(client_drifts)
    avg_drift_vec = vectorize(avg_drift)
    norm_sq_avg_drift = torch.dot(avg_drift_vec, avg_drift_vec)
    var = avg_norm_sq_drifts - norm_sq_avg_drift
    return var.item(), avg_norm_sq_drifts.item(), norm_sq_avg_drift.item()
def fda_sketch_estimation(client_drifts, ams_sketch):
    """
    Compute the linear estimation of ||avg(u_t)||^2 using SketchFDA. The way we compute it is by its equivalent form
    which is (1/1+e) * M_2(sk(avg(u_t))). Of course, in a real system implementation we would first compute each drift's
    sketch and then average the sketches on the server. They are equivalent, we simply do this because it is more
    efficient. (see paper)
    Args:
        client_drifts (dict): A dictionary where keys are client IDs and values are lists of parameter tensors (drifts).
        ams_sketch (AmsSketch): The AMS sketch used to compute the sketch estimation.
    Returns:
        float: The linear estimation of ||avg(u_t)||^2.
    """
    avg_drift = vectorize(average_client_parameters(client_drifts))
    sk = ams_sketch.sketch_for_vector(avg_drift)
    epsilon = ams_sketch.epsilon
    est = (1 / (1+epsilon)) * ams_sketch.estimate_euc_norm_squared(sk)
    return est
def fda_variance_approx(client_drifts, ams_sketch=None):
    norm_sq_drifts = []
    for drifts in client_drifts.values():
        drift_vec = vectorize(drifts)
        norm_sq_drift = torch.dot(drift_vec, drift_vec)
        norm_sq_drifts.append(norm_sq_drift)
    avg_norm_sq_drifts = sum(norm_sq_drifts) / len(norm_sq_drifts)
    norm_sq_avg_drift_approx = fda_sketch_estimation(client_drifts, ams_sketch)
    var_approx = avg_norm_sq_drifts - norm_sq_avg_drift_approx
    return var_approx.item()
@torch.no_grad
def get_updated_client_parameters(client_train_params_dict, sampled_clients, params):
    """
      Update the parameters of the sampled clients with the current global parameters.
      This function ensures there are no ID collisions or overwriting issues by creating a new dictionary
      for the updated client parameters.
      Args:
          client_train_params_dict (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
          sampled_clients (list): A list of newly sampled client IDs.
          params (list of torch.nn.Parameter): The current global parameters to be assigned to the sampled clients.
      Returns:
          dict: Dictionary where keys are client IDs and values are lists of parameter tensors.
      """
    for client_id in sampled_clients:
        copy_parameters(
            from_params=params,
            to_params=client_train_params_dict[client_id]
        )
    return client_train_params_dict
def stop_training_with_threshold(hyperparams, metrics):
    if hyperparams['checkpoint'] == 'roberta-base':
        if hyperparams['ds_name'] == 'mrpc':
            if metrics['accuracy'] >= 0.902 * 0.95:
                return True
        if hyperparams['ds_name'] == 'rte':
            if metrics['accuracy'] >= 0.789 * 0.95:
                return True
        if hyperparams['ds_name'] == 'cola':
            if metrics['matthews_correlation'] >= 0.636 * 0.95:
                return True
        if hyperparams['ds_name'] == 'sst2':
            if metrics['accuracy'] >= 0.948 * 0.99:
                return True
        if hyperparams['ds_name'] == 'qnli':
            if metrics['accuracy'] >= 0.928 * 0.95:
                return True
    if hyperparams['checkpoint'] == 'microsoft/deberta-v3-base':
        if hyperparams['ds_name'] == 'mnli':
            if metrics['accuracy_m'] >= 0.906 * 0.99 and metrics['accuracy_mm'] >= 0.907 * 0.99:
                return True
    return False
def get_target_metric_value(ds_name, metrics):
    """Extracts the primary evaluation metric based on the dataset."""
    if ds_name == 'cola':
        return metrics.get('matthews_correlation', 0.0)
    elif ds_name == 'mnli':
        return (metrics.get('accuracy_m', 0.0) + metrics.get('accuracy_mm', 0.0)) / 2.0
    return metrics.get('accuracy', 0.0)
def check_dynamic_early_stopping(hyperparams, metrics_history):
    patience = hyperparams.get('early_stopping_patience', 10)
    min_delta = hyperparams.get('early_stopping_min_delta', 0.005)
    window_size = hyperparams.get('early_stopping_window', 3)
    plateau_min_acc = hyperparams.get('plateau_min_accuracy', 0.90)
    stagnation_round = hyperparams.get('stagnation_round', 40)
    stagnation_tolerance = hyperparams.get('stagnation_tolerance', 0.05)
    if not metrics_history:
        return False
    values = [get_target_metric_value(hyperparams['ds_name'], m) for m in metrics_history]
    current_round = len(values)
    if current_round >= stagnation_round:
        initial_acc = sum(values[:3]) / 3.0 if current_round >= 3 else values[0]
        current_max = max(values[:stagnation_round])
        if current_max - initial_acc < stagnation_tolerance:
            return True
    if current_round < window_size:
        return False
    smoothed_values = [
        sum(values[i:i+window_size]) / window_size
        for i in range(current_round - window_size + 1)
    ]
    best_value = -float('inf')
    rounds_without_improvement = 0
    for val in smoothed_values:
        if val > best_value + min_delta:
            best_value = val
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
    if best_value >= plateau_min_acc and rounds_without_improvement >= patience:
        return True
    return False

```


## File: `fdaopt/training/optimizers.py`

```py
from fdaopt.utils import torch
def get_opt_hyperparams(where, hyperparams):
    """
    Extracts optimizer hyperparameters from the given hyperparameters dictionary.
    Args:
        where (str): A prefix indicating whether the optimizer is 'server' or 'client'.
        hyperparams (dict): A dictionary containing hyperparameters including optimizer hyperparameters. It is
            expected it contains `server_opt_name` and `client_opt_name`. The rest of the optimizer hyperparameters
            are expected to have keys `server_opt_HYPERPARAM` or `client_opt_HYPERPARAM` where the HYPERPARAM is
            a valid optimizer hyperparameter (for example, `lr`).
    Returns:
        dict: A dictionary containing the extracted optimizer hyperparameters.
    """
    opt_dict = {}
    where_opt = f"{where}_opt"
    for k, v in hyperparams.items():
        if where_opt in k:
            param = k.replace(f"{where_opt}_", '')
            if param != 'name':
                opt_dict[param] = v
    return opt_dict
def get_optimizer_class(optimizer_name):
    """
    Dynamically imports and returns the optimizer class from torch.optim based on the given optimizer name.
    Args:
        optimizer_name (str): The name of the optimizer class as a string.
    Returns:
        type: The optimizer class from torch.optim.
    """
    optimizer_class = getattr(torch.optim, optimizer_name)
    return optimizer_class
def server_client_optimizers(train_params, hyperparams, num_lora_params=None):
    """
    Creates and returns the server and client optimizers based on the provided hyperparameters.
    Args:
        train_params (list of torch.nn.Parameter): The parameters of the model to be optimized.
        hyperparams (dict): A dictionary containing hyperparameters including optimizer hyperparameters. It is
            expected it contains `server_opt_name` and `client_opt_name`. The rest of the optimizer hyperparameters
            are expected to have keys `server_opt_HYPERPARAM` or `client_opt_HYPERPARAM` where the HYPERPARAM is
            a valid optimizer hyperparameter (for example, `lr`).
    Returns:
        tuple: A tuple containing the server optimizer and the client optimizer.
    """
    server_opt_class = get_optimizer_class(hyperparams['server_opt_name'])
    client_opt_class = get_optimizer_class(hyperparams['client_opt_name'])
    server_kwargs = get_opt_hyperparams('server', hyperparams)
    client_kwargs = get_opt_hyperparams('client', hyperparams)
    server_opt = server_opt_class(train_params, **server_kwargs)
    if num_lora_params is not None and num_lora_params < len(train_params):
        client_lr = client_kwargs.pop('lr', 0.01)
        client_groups = [
            {'params': train_params[:num_lora_params], 'lr': client_lr},
            {'params': train_params[num_lora_params:], 'lr': client_lr * 0.1}
        ]
        client_opt = client_opt_class(client_groups, **client_kwargs)
    else:
        client_opt = client_opt_class(train_params, **client_kwargs)
    return server_opt, client_opt

```


## File: `fdaopt/training/fed_train.py`

```py
from fdaopt import utils
from fdaopt.models.ops import copy_parameters
import torch
def federated_training_step(model, train_params, client_train_params_dict, client_opt, fed_ds):
    """
    Performs a single federated training step.
    This function trains each client starting with its client-specific model parameters, updates them
    with the client optimizer and client-specific batch, and returns. At the end, client_train_params_dict
    have the updated client-specific parameters (after training on their specific batch).
    Args:
        model (torch.nn.Module): The model to be trained.
        train_params (list): A list of trainable parameters of the model (referencing `model` above).
        client_train_params_dict (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
        client_opt (torch.optim.Optimizer): The optimizer for updating the model parameters.
        fed_ds (FederatedDataset): An object that provides the batches for each client.
    Returns:
        float: The average training loss across all clients for this federated training step.
    """
    model.train()
    training_loss = 0.0
    num_clients = len(client_train_params_dict)
    for client_id in client_train_params_dict.keys():
        copy_parameters(
            from_params=client_train_params_dict[client_id],
            to_params=train_params
        )
        batch = fed_ds.next_client_batch(client_id)
        batch = {k: v.to(utils.DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        client_opt.step()
        copy_parameters(
            from_params=train_params,
            to_params=client_train_params_dict[client_id]
        )
        client_opt.zero_grad()
        training_loss += loss.item()
    training_loss = training_loss / num_clients
    return training_loss
def client_round_train(model, train_params, client_train_params_dict, client_opt, fed_ds, client_id, round_steps):
    """
    Trains a client-specific model for a specified number of steps (a round).
    At the end, the updated client-specific parameters are stored back in `client_train_params_dict`.
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_params (list): A list of trainable parameters of the model (referencing `model` above).
        client_train_params_dict (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
        client_opt (torch.optim.Optimizer): The optimizer for updating the model parameters.
        fed_ds (FederatedDataset): An object that provides the batches for each client.
        client_id (int): The ID of the client to be trained.
        round_steps (int): The number of training steps (mini-batches) for this client's round.
    Returns:
        float: The average training loss for the client over the specified number of steps.
    """
    model.train()
    copy_parameters(
        from_params=client_train_params_dict[client_id],
        to_params=train_params
    )
    training_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for step in range(round_steps):
        batch = fed_ds.next_client_batch(client_id)
        batch = {k: v.to(utils.DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == batch["labels"]).sum().item()
        total_predictions += batch["labels"].size(0)
        loss.backward()
        client_opt.step()
        client_opt.zero_grad()
        training_loss += loss.item()
    copy_parameters(
        from_params=train_params,
        to_params=client_train_params_dict[client_id]
    )
    training_loss = training_loss / round_steps
    train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return training_loss, train_accuracy

```


## File: `fdaopt/training/fed_opt.py`

```py
import os
import gc
import torch
import time
import math
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
from datetime import datetime
from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import average_client_parameters, compute_client_drifts, compute_pseudo_gradients, set_gradients, copy_parameters, \
    get_updated_client_parameters, variance, compute_metrics, stop_training_with_threshold, check_dynamic_early_stopping
from fdaopt.training.fed_train import client_round_train
from fdaopt.training.optimizers import get_opt_hyperparams, get_optimizer_class, server_client_optimizers
from fdaopt import utils
from transformers import AutoModelForSequenceClassification
def get_save_device():
    if utils.DEVICE_RAM_PROGRAM == 'performance' or utils.DEVICE_RAM_PROGRAM == 'moderate':
        return utils.DEVICE
    elif utils.DEVICE_RAM_PROGRAM == 'low':
        return 'cpu'
    return 'cpu'
def fed_opt(hyperparams):
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'],
        hyperparams['ds_name'],
        hyperparams['checkpoint'],
        hyperparams['num_clients'],
        hyperparams['alpha'],
        hyperparams['batch_size'],
        hyperparams.get('num_workers', 4),
        hyperparams.get('pin_memory', True)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'],
        num_labels=hyperparams['num_labels']
    )
    model = model.to(utils.DEVICE)
    train_params = [param for param in model.parameters() if param.requires_grad]
    round_start_train_params = [param.detach().clone() for param in train_params]
    server_opt, client_opt = server_client_optimizers(train_params, hyperparams)
    client_sampler = ClientSampler(
        list(range(hyperparams['num_clients'])),
        hyperparams['clients_per_round']
    )
    client_train_params_dict = {
        client_id: [param.detach().clone().to(get_save_device()) for param in round_start_train_params]
        for client_id in range(hyperparams['num_clients'])
    }
    metrics_handler = MetricsHandler(hyperparams)
    for r in range(hyperparams['total_rounds']):
        training_loss = 0.0
        sampled_clients = client_sampler.sample()
        copy_parameters(
            from_params=train_params,
            to_params=round_start_train_params
        )
        client_train_params_dict = get_updated_client_parameters(client_train_params_dict, sampled_clients, round_start_train_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        for client_id in sampled_clients:
            loss = client_round_train(model, train_params, client_train_params_dict, client_opt, fed_ds, client_id, round_steps)
            training_loss += loss
        copy_parameters(
            from_params=round_start_train_params,
            to_params=train_params
        )
        client_drifts = compute_client_drifts(round_start_train_params, client_train_params_dict)
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        set_gradients(train_params, pseudo_gradients)
        server_opt.step()
        server_opt.zero_grad()
        metrics = {"round": r + 1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics['round_steps'] = round_steps
        metrics_handler.append_round_metrics(metrics)
        gc.collect()
        if hyperparams['threshold']:
            if stop_training_with_threshold(hyperparams, metrics): break
    metrics_handler.save_metrics()
def toggle_rolora_matrices(model, active_matrix):
    """
    Dynamically freezes and unfreezes LoRA A and B matrices,
    ensuring all other parameters remain frozen per pure theory.
    """
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            param.requires_grad = (active_matrix == 'A')
        elif 'lora_B' in name:
            param.requires_grad = (active_matrix == 'B')
def fedavg_rolora(hyperparams):
    print(f"[{time.strftime('%H:%M:%S')}] [INIT] Starting dataset preparation for {hyperparams['ds_name']}...")
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    print(f"[{time.strftime('%H:%M:%S')}] [SUCCESS] Federated datasets and loaders successfully initialized.")
    if hyperparams.get('ds_name') == 'mnist':
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(
            rank=hyperparams.get('lora_r', 8),
            linear=hyperparams.get('linear', False)
        ).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
            lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
        )
        model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
        if not hyperparams.get('train_head', True):
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
    print(f"[{time.strftime('%H:%M:%S')}] [SUCCESS] Model loaded on {utils.DEVICE}.")
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    group_A = params_A + params_other
    group_B = params_B + params_other
    len_A, len_B = len(params_A), len(params_B)
    client_opt_class = get_optimizer_class(hyperparams['client_opt_name'])
    client_opt_kwargs = get_opt_hyperparams('client', hyperparams)
    base_lr = client_opt_kwargs.get('lr', 0.01)
    other_opt_kwargs = {k: v for k, v in client_opt_kwargs.items() if k != 'lr'}
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in group_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in group_B] for c_id in range(hyperparams['num_clients'])}
    print(f"[{time.strftime('%H:%M:%S')}] [START] Beginning Federated Training Loop for {hyperparams['total_rounds']} total rounds.")
    metrics_handler = MetricsHandler(hyperparams)
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        training_loss = 0.0
        training_accuracy = 0.0
        sampled_clients = client_sampler.sample()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        print(f"\n--- [ROUND {r + 1} START] Phase Matrix: {active_matrix} | Sampled Clients: {sampled_clients} ---")
        toggle_rolora_matrices(model, active_matrix)
        active_params = group_A if active_matrix == 'A' else group_B
        active_lora_params = params_A if active_matrix == 'A' else params_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        active_lora_len = len_A if active_matrix == 'A' else len_B
        round_start_active_params = [param.detach().clone() for param in active_params]
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        train_start = time.time()
        for client_id in sampled_clients:
            param_groups = [{'params': active_lora_params, 'lr': base_lr}]
            if params_other:
                param_groups.append({'params': params_other, 'lr': base_lr * 0.1})
            client_opt = client_opt_class(param_groups, **other_opt_kwargs)
            loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
            training_loss += loss
            training_accuracy += acc
            print(f"  -> Client {client_id}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
        train_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] [ROUND {r + 1}] Client training finished in {train_end - train_start:.2f}s. Running server aggregation...")
        agg_start = time.time()
        sampled_client_params = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
        client_drifts = compute_client_drifts(round_start_active_params, sampled_client_params)
        averaged_params = average_client_parameters(sampled_client_params)
        with torch.no_grad():
            update_l2_norm = math.sqrt(sum(torch.sum((p_new.to(utils.DEVICE) - p_old.to(utils.DEVICE))**2).item()
                                           for p_new, p_old in zip(averaged_params, round_start_active_params)))
        copy_parameters(from_params=averaged_params, to_params=active_params)
        lora_only_drifts = {c_id: drifts[:active_lora_len] for c_id, drifts in client_drifts.items()}
        agg_end = time.time()
        round_end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] [ROUND {r + 1}] Running model validation evaluation steps...")
        metrics = {
            "method": "FedAvg-RoLoRA",
            "round": r + 1,
            "active_matrix": active_matrix,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": hyperparams['local_epochs'],
            "round_steps": round_steps
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(lora_only_drifts)
        metrics_handler.append_round_metrics(metrics)
        print(f"=== [ROUND {r + 1} FINISHED] Avg Train Loss: {metrics['training_loss']:.4f} | Test Loss: {metrics.get('testing_loss', 0.0):.4f} ===")
        gc.collect()
        if hyperparams.get('early_stopping') and check_dynamic_early_stopping(hyperparams, metrics_handler.round_metrics):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break
    metrics_handler.save_metrics()
    print(f"[{time.strftime('%H:%M:%S')}] [COMPLETE] Evaluation metrics successfully flushed to metrics handler disk.")
def fedavg_lora(hyperparams):
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    if hyperparams.get('ds_name') == 'mnist':
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(
            rank=hyperparams.get('lora_r', 8),
            linear=hyperparams.get('linear', False)
        ).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
            lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
        )
        model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
        if not hyperparams.get('train_head', True):
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    num_lora = len(lora_params)
    train_params = lora_params + other_params
    server_opt, _ = server_client_optimizers(train_params, hyperparams, num_lora_params=num_lora)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict = {c_id: [param.detach().clone().to(get_save_device()) for param in train_params]
                   for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        training_loss = 0.0
        training_accuracy = 0.0
        sampled_clients = client_sampler.sample()
        round_start_params = [param.detach().clone() for param in train_params]
        client_dict = get_updated_client_parameters(client_dict, sampled_clients, round_start_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        train_start = time.time()
        for client_id in sampled_clients:
            _, client_opt = server_client_optimizers(train_params, hyperparams, num_lora_params=num_lora)
            loss, acc = client_round_train(model, train_params, client_dict, client_opt, fed_ds, client_id, round_steps)
            training_loss += loss
            training_accuracy += acc
        train_end = time.time()
        agg_start = time.time()
        sampled_client_dict = {c_id: client_dict[c_id] for c_id in sampled_clients}
        client_drifts = compute_client_drifts(round_start_params, sampled_client_dict)
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        with torch.no_grad():
            update_l2_norm = math.sqrt(sum(torch.sum(g**2).item() for g in pseudo_gradients))
        copy_parameters(from_params=round_start_params, to_params=train_params)
        set_gradients(train_params, pseudo_gradients)
        server_opt.step()
        server_opt.zero_grad()
        agg_end = time.time()
        round_end_time = time.time()
        metrics = {
            "method": "FedAvg-LoRA",
            "round": r + 1,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": hyperparams['local_epochs'],
            "round_steps": round_steps
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished.")
        gc.collect()
        if hyperparams.get('early_stopping') and check_dynamic_early_stopping(hyperparams, metrics_handler.round_metrics):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break
    metrics_handler.save_metrics()
def ffa_lora(hyperparams):
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    if hyperparams.get('ds_name') == 'mnist':
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(
            rank=hyperparams.get('lora_r', 8),
            linear=hyperparams.get('linear', False)
        ).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
            lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
        )
        model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
        if not hyperparams.get('train_head', True):
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            param.requires_grad = False
    lora_params = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    num_lora = len(lora_params)
    train_params = lora_params + other_params
    server_opt, _ = server_client_optimizers(train_params, hyperparams, num_lora_params=num_lora)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict = {c_id: [param.detach().clone().to(get_save_device()) for param in train_params]
                   for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        training_loss = 0.0
        training_accuracy = 0.0
        sampled_clients = client_sampler.sample()
        round_start_params = [param.detach().clone() for param in train_params]
        _, client_opt = server_client_optimizers(train_params, hyperparams, num_lora_params=num_lora)
        client_dict = get_updated_client_parameters(client_dict, sampled_clients, round_start_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        train_start = time.time()
        for client_id in sampled_clients:
            loss, acc = client_round_train(model, train_params, client_dict, client_opt, fed_ds, client_id, round_steps)
            training_loss += loss
            training_accuracy += acc
        train_end = time.time()
        agg_start = time.time()
        sampled_client_dict = {c_id: client_dict[c_id] for c_id in sampled_clients}
        client_drifts = compute_client_drifts(round_start_params, sampled_client_dict)
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        with torch.no_grad():
            update_l2_norm = math.sqrt(sum(torch.sum(g**2).item() for g in pseudo_gradients))
        copy_parameters(from_params=round_start_params, to_params=train_params)
        set_gradients(train_params, pseudo_gradients)
        server_opt.step()
        server_opt.zero_grad()
        agg_end = time.time()
        round_end_time = time.time()
        metrics = {
            "method": "FFA-LoRA",
            "round": r + 1,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": hyperparams['local_epochs'],
            "round_steps": round_steps
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished.")
        gc.collect()
        if hyperparams.get('early_stopping') and check_dynamic_early_stopping(hyperparams, metrics_handler.round_metrics):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break
    metrics_handler.save_metrics()

```


## File: `fdaopt/training/fda_opt.py`

```py
import os
import gc
import pandas as pd
import torch
import time
import math
from datetime import datetime
from peft import get_peft_model, LoraConfig, TaskType
from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (compute_client_drifts, compute_drifts, compute_pseudo_gradients, get_ksi_unit, linear_variance_approx, linear_variance_approx2, set_gradients, copy_parameters,
    get_updated_client_parameters, variance, compute_metrics, fda_variance_approx, stop_training_with_threshold, check_dynamic_early_stopping)
from fdaopt.training.fed_train import client_round_train
from fdaopt.training.sketch import AmsSketch
from fdaopt.training.optimizers import server_client_optimizers
from fdaopt import utils
from transformers import AutoModelForSequenceClassification
import numpy as np
import random
from fdaopt.training.optimizers import get_opt_hyperparams, get_optimizer_class, server_client_optimizers
def get_save_device():
    if utils.DEVICE_RAM_PROGRAM == 'performance' or utils.DEVICE_RAM_PROGRAM == 'moderate':
        return utils.DEVICE
    elif utils.DEVICE_RAM_PROGRAM == 'low':
        return 'cpu'
    return 'cpu'
def fda_opt(hyperparams):
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'],
        hyperparams['ds_name'],
        hyperparams['checkpoint'],
        hyperparams['num_clients'],
        hyperparams['alpha'],
        hyperparams['batch_size'],
        hyperparams.get('num_workers', 4),
        hyperparams.get('pin_memory', True)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'],
        num_labels=hyperparams['num_labels']
    )
    model = model.to(utils.DEVICE)
    train_params = [param for param in model.parameters() if param.requires_grad]
    round_start_train_params = [param.detach().clone() for param in train_params]
    server_opt, client_opt = server_client_optimizers(train_params, hyperparams)
    client_sampler = ClientSampler(
        list(range(hyperparams['num_clients'])),
        hyperparams['clients_per_round']
    )
    client_train_params_dict = {
        client_id: [param.detach().clone().to(get_save_device()) for param in round_start_train_params]
        for client_id in range(hyperparams['num_clients'])
    }
    ams_sketch = AmsSketch()
    metrics_handler = MetricsHandler(hyperparams)
    theta = hyperparams['theta']
    first_round = True
    for r in range(hyperparams['total_rounds']):
        training_loss = 0.0
        sampled_clients = client_sampler.sample()
        copy_parameters(
            from_params=train_params,
            to_params=round_start_train_params
        )
        client_train_params_dict = get_updated_client_parameters(client_train_params_dict, sampled_clients, round_start_train_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        var_approx = 0.0
        round_epochs = 0
        while var_approx <= theta and round_epochs < hyperparams['max_epochs']:
            round_epochs += 1
            for client_id in sampled_clients:
                training_loss += client_round_train(model, train_params, client_train_params_dict, client_opt, fed_ds,
                                                    client_id, round_steps)
            client_drifts = compute_client_drifts(round_start_train_params, client_train_params_dict)
            var_approx = fda_variance_approx(client_drifts, ams_sketch=ams_sketch)
            if first_round:
                first_round = False
                break
        training_loss = training_loss / round_epochs
        copy_parameters(
            from_params=round_start_train_params,
            to_params=train_params
        )
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        set_gradients(train_params, pseudo_gradients)
        server_opt.step()
        server_opt.zero_grad()
        metrics = {"round": r + 1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics['round_steps'] = round_epochs * round_steps
        metrics_handler.append_round_metrics(metrics)
        theta = ((hyperparams['max_epochs'] / 2) / round_epochs) * metrics['variance']
        gc.collect()
        if hyperparams['threshold']:
            if stop_training_with_threshold(hyperparams, metrics): break
    metrics_handler.save_metrics()
@torch.no_grad()
def compute_effective_variance_and_norm(model, client_drifts, active_matrix, lora_r):
    """
    Computes the exact variance and L2 norm of the global update in the effective
    weight space (W = AB) for LoRA modules, combined with the classification head.
    Args:
        model: The PEFT model containing both A and B matrices.
        client_drifts: Dictionary mapping client_id -> list of drift tensors.
        active_matrix: String, either 'A' or 'B'.
        lora_r: Int, LoRA rank.
    Returns:
        variance (float): The exact variance of the client updates in the expanded space.
        update_l2_norm (float): The L2 norm of the averaged global update.
        avg_norm_sq_drifts (float): Average of the squared norms of client updates.
        norm_sq_avg_drift (float): Squared norm of the averaged global update.
    """
    num_clients = len(client_drifts)
    sampled_clients = list(client_drifts.keys())
    client_sq_norms = {client_id: 0.0 for client_id in sampled_clients}
    norm_sq_avg_drift = 0.0
    active_names_lora = [n for n, p in model.named_parameters() if f'lora_{active_matrix}' in n and p.requires_grad]
    active_names_other = [n for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    ordered_active_names = active_names_lora + active_names_other
    model_named_params = dict(model.named_parameters())
    for active_param_idx, name in enumerate(ordered_active_names):
        param = model_named_params[name]
        is_lora = 'lora_' in name
        if is_lora:
            frozen_matrix_name = name.replace(f'lora_{active_matrix}', 'lora_B' if active_matrix == 'A' else 'lora_A')
            frozen_param = model_named_params[frozen_matrix_name].to(utils.DEVICE)
            is_transposed = param.shape[0] != lora_r if active_matrix == 'A' else frozen_param.shape[0] != lora_r
            layer_drift_sum = None
            for client_id in sampled_clients:
                delta_active = client_drifts[client_id][active_param_idx].to(utils.DEVICE)
                if is_transposed:
                    delta_W = (delta_active @ frozen_param) if active_matrix == 'A' else (frozen_param @ delta_active)
                else:
                    delta_W = (frozen_param @ delta_active) if active_matrix == 'A' else (delta_active @ frozen_param)
                client_sq_norms[client_id] += torch.sum(delta_W ** 2).item()
                if layer_drift_sum is None:
                    layer_drift_sum = delta_W.clone()
                else:
                    layer_drift_sum += delta_W
            layer_avg_drift = layer_drift_sum / num_clients
            norm_sq_avg_drift += torch.sum(layer_avg_drift ** 2).item()
        else:
            layer_drift_sum = None
            for client_id in sampled_clients:
                delta_other = client_drifts[client_id][active_param_idx].to(utils.DEVICE)
                client_sq_norms[client_id] += torch.sum(delta_other ** 2).item()
                if layer_drift_sum is None:
                    layer_drift_sum = delta_other.clone()
                else:
                    layer_drift_sum += delta_other
            layer_avg_drift = layer_drift_sum / num_clients
            norm_sq_avg_drift += torch.sum(layer_avg_drift ** 2).item()
    avg_norm_sq_drifts = sum(client_sq_norms.values()) / num_clients
    variance = max(0.0, avg_norm_sq_drifts - norm_sq_avg_drift)
    update_l2_norm = math.sqrt(norm_sq_avg_drift)
    return variance, update_l2_norm, avg_norm_sq_drifts, norm_sq_avg_drift
def fda_rolora_opt(hyperparams):
    """
    Executes FDA-RoLoRA federated training loop utilizing dynamically scaled theta thresholds
    measured directly on the effective weight space W = AB.
    """
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    gamma = hyperparams.get('gamma', 0.3)
    lora_r = hyperparams.get('lora_r', 8)
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False,
        r=lora_r, lora_alpha=hyperparams.get('lora_alpha', 16),
        lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
    )
    model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    params_A = params_A + params_other
    params_B = params_B + params_other
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    theta = hyperparams.get('theta', 0.0)
    first_round = True
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        _, client_opt = server_client_optimizers(active_params, hyperparams)
        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        training_loss = 0.0
        training_accuracy = 0.0
        var_exact = 0.0
        update_l2_norm = 0.0
        avg_norm_sq_drifts = 0.0
        norm_sq_avg_drift = 0.0
        round_epochs = 0
        train_start = time.time()
        while True:
            round_epochs += 1
            for client_id in sampled_clients:
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            if round_epochs >= e_min:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                var_exact, update_l2_norm, avg_norm_sq_drifts, norm_sq_avg_drift = compute_effective_variance_and_norm(
                    model, client_drifts, active_matrix, lora_r
                )
                if first_round:
                    first_round = False
                    break
                if var_exact > theta or round_epochs >= e_max:
                    break
            elif round_epochs >= e_max:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                var_exact, update_l2_norm, avg_norm_sq_drifts, norm_sq_avg_drift = compute_effective_variance_and_norm(
                    model, client_drifts, active_matrix, lora_r
                )
                break
        train_end = time.time()
        agg_start = time.time()
        training_loss = training_loss / round_epochs
        training_accuracy = training_accuracy / round_epochs
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        set_gradients(active_params, pseudo_gradients)
        active_server_opt.step()
        active_server_opt.zero_grad()
        agg_end = time.time()
        round_end_time = time.time()
        theta = ((e_max / 2) / round_epochs) * var_exact
        metrics = {
            "method": "FDA-RoLoRA",
            "round": r + 1,
            "active_matrix": active_matrix,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * round_steps,
            "variance": var_exact,
            "avg_norm_sq_drifts": avg_norm_sq_drifts,
            "norm_sq_avg_drift": norm_sq_avg_drift,
            "theta_phase": theta
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics_handler.append_round_metrics(metrics)
        gc.collect()
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished.")
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
    metrics_handler.save_metrics()
def toggle_rolora_matrices(model, active_matrix):
    """
    Dynamically freezes and unfreezes LoRA A and B matrices.
    active_matrix: 'A' or 'B'
    """
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            param.requires_grad = (active_matrix == 'A')
        elif 'lora_B' in name:
            param.requires_grad = (active_matrix == 'B')
def fda_rolora_opt2(hyperparams):
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    alpha_theta = hyperparams.get('alpha_theta', 0.01)
    gamma = hyperparams.get('gamma', 0.3)
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(hyperparams['checkpoint'], num_labels=hyperparams['num_labels'])
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False,
        r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
        lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
    )
    model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    params_A = params_A + params_other
    params_B = params_B + params_other
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        active_norm_sq = sum(torch.sum(p.to(utils.DEVICE) ** 2).item() for p in active_params)
        theta_phase = alpha_theta * active_norm_sq
        _, client_opt = server_client_optimizers(active_params, hyperparams)
        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        training_loss = 0.0
        training_accuracy = 0.0
        var_approx = 0.0
        round_epochs = 0
        train_start = time.time()
        while True:
            round_epochs += 1
            for client_id in sampled_clients:
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            if round_epochs >= e_min:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                shared_gen = torch.Generator(device=utils.DEVICE).manual_seed(42 + r * 1000 + round_epochs)
                ksi_vector = get_ksi_unit(round_start_active_params, w_tminus1=None, generator=shared_gen)
                var_approx = linear_variance_approx(client_drifts, ksi_vector)
                if var_approx > theta_phase or round_epochs >= e_max:
                    break
            elif round_epochs >= e_max:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
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
        metrics = {
            "method": "FDA-RoLoRA",
            "round": r + 1,
            "active_matrix": active_matrix,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * round_steps,
            "var_approx_linear": var_approx,
            "theta_phase": theta_phase
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished.")
        gc.collect()
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
    metrics_handler.save_metrics()
def fda_rolora_dual_track_opt(hyperparams):
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    gamma = hyperparams.get('gamma', 0.3)
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    if hyperparams.get('ds_name') == 'mnist':
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(
            rank=hyperparams.get('lora_r', 8),
            linear=hyperparams.get('linear', False)
        ).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
            lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
        )
        model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
        if not hyperparams.get('train_head', True):
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    num_lora = len(params_A)
    params_A = params_A + params_other
    params_B = params_B + params_other
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams, num_lora_params=num_lora)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams, num_lora_params=num_lora)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    theta_A = hyperparams.get('theta', 0.0)
    theta_B = hyperparams.get('theta', 0.0)
    first_round_A = True
    first_round_B = True
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        current_theta = theta_A if active_matrix == 'A' else theta_B
        is_first_round = first_round_A if active_matrix == 'A' else first_round_B
        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
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
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            if round_epochs >= e_min:
                sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                if is_first_round:
                    if active_matrix == 'A':
                        first_round_A = False
                    else:
                        first_round_B = False
                    break
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
        next_theta = ((e_max / 2) / round_epochs) * var_subspace
        if active_matrix == 'A':
            theta_A = next_theta
        else:
            theta_B = next_theta
        metrics = {
            "method": "FDA-RoLoRA-DualTrack",
            "round": r + 1,
            "active_matrix": active_matrix,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * round_steps,
            "variance": var_subspace,
            "avg_norm_sq_drifts": avg_norm_sq_drifts,
            "norm_sq_avg_drift": norm_sq_avg_drift,
            "theta_phase": current_theta
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished.")
        gc.collect()
        if hyperparams.get('early_stopping') and check_dynamic_early_stopping(hyperparams, metrics_handler.round_metrics):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break
    metrics_handler.save_metrics()
def fda_rolora_dual_track_ema_standard_opt(hyperparams):
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    rho = hyperparams.get('rho', 0.8)
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    if hyperparams.get('ds_name') == 'mnist':
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(
            rank=hyperparams.get('lora_r', 8),
            linear=hyperparams.get('linear', False)
        ).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
            lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
        )
        model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
        if not hyperparams.get('train_head', True):
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    num_lora = len(params_A)
    params_A = params_A + params_other
    params_B = params_B + params_other
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams, num_lora_params=num_lora)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams, num_lora_params=num_lora)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    theta_A = hyperparams.get('theta', 0.0)
    theta_B = hyperparams.get('theta', 0.0)
    first_round_A = True
    first_round_B = True
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        current_theta = theta_A if active_matrix == 'A' else theta_B
        is_first_round = first_round_A if active_matrix == 'A' else first_round_B
        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
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
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            if round_epochs >= e_min:
                sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                if is_first_round:
                    if active_matrix == 'A':
                        first_round_A = False
                    else:
                        first_round_B = False
                    break
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
        proposed_target = ((e_max / 2) / round_epochs) * var_subspace
        next_theta = rho * current_theta + (1 - rho) * proposed_target
        if active_matrix == 'A':
            theta_A = next_theta
        else:
            theta_B = next_theta
        metrics = {
            "method": "FDA-RoLoRA-DualTrack-EMA-Standard",
            "round": r + 1,
            "active_matrix": active_matrix,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * round_steps,
            "variance": var_subspace,
            "avg_norm_sq_drifts": avg_norm_sq_drifts,
            "norm_sq_avg_drift": norm_sq_avg_drift,
            "theta_phase": current_theta
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished.")
        gc.collect()
        if hyperparams.get('early_stopping') and check_dynamic_early_stopping(hyperparams, metrics_handler.round_metrics):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break
    metrics_handler.save_metrics()
def fda_rolora_dual_track_ema_pure_opt(hyperparams):
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    rho = hyperparams.get('rho', 0.8)
    alpha_ema = hyperparams.get('alpha_ema', 1.15)
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    if hyperparams.get('ds_name') == 'mnist':
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(
            rank=hyperparams.get('lora_r', 8),
            linear=hyperparams.get('linear', False)
        ).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
            lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
        )
        model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
        if not hyperparams.get('train_head', True):
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    num_lora = len(params_A)
    params_A = params_A + params_other
    params_B = params_B + params_other
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams, num_lora_params=num_lora)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams, num_lora_params=num_lora)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    theta_A = hyperparams.get('theta', 0.0)
    theta_B = hyperparams.get('theta', 0.0)
    first_round_A = True
    first_round_B = True
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        current_theta = theta_A if active_matrix == 'A' else theta_B
        is_first_round = first_round_A if active_matrix == 'A' else first_round_B
        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
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
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            if round_epochs >= e_min:
                sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                if is_first_round:
                    if active_matrix == 'A':
                        first_round_A = False
                    else:
                        first_round_B = False
                    break
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
        proposed_target = alpha_ema * var_subspace
        next_theta = rho * current_theta + (1 - rho) * proposed_target
        if active_matrix == 'A':
            theta_A = next_theta
        else:
            theta_B = next_theta
        metrics = {
            "method": "FDA-RoLoRA-DualTrack-EMA-Pure",
            "round": r + 1,
            "active_matrix": active_matrix,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * round_steps,
            "variance": var_subspace,
            "avg_norm_sq_drifts": avg_norm_sq_drifts,
            "norm_sq_avg_drift": norm_sq_avg_drift,
            "theta_phase": current_theta
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished.")
        gc.collect()
        if hyperparams.get('early_stopping') and check_dynamic_early_stopping(hyperparams, metrics_handler.round_metrics):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break
    metrics_handler.save_metrics()
def fda_rolora_dual_track_sma_opt(hyperparams):
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    gamma = hyperparams.get('gamma', 0.3)
    sma_window = hyperparams.get('sma_window', 3)
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    if hyperparams.get('ds_name') == 'mnist':
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(
            rank=hyperparams.get('lora_r', 8),
            linear=hyperparams.get('linear', False)
        ).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
            lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
        )
        model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
        if not hyperparams.get('train_head', True):
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    num_lora = len(params_A)
    params_A = params_A + params_other
    params_B = params_B + params_other
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams, num_lora_params=num_lora)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams, num_lora_params=num_lora)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    theta_A = hyperparams.get('theta', 0.0)
    theta_B = hyperparams.get('theta', 0.0)
    history_A = [theta_A] if theta_A > 0.0 else []
    history_B = [theta_B] if theta_B > 0.0 else []
    first_round_A = True
    first_round_B = True
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        current_theta = theta_A if active_matrix == 'A' else theta_B
        is_first_round = first_round_A if active_matrix == 'A' else first_round_B
        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
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
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            if round_epochs >= e_min:
                sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                if is_first_round:
                    if active_matrix == 'A':
                        first_round_A = False
                    else:
                        first_round_B = False
                    break
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
        proposed_target = ((e_max / 2) / round_epochs) * var_subspace
        if active_matrix == 'A':
            history_A.append(proposed_target)
            if len(history_A) > sma_window:
                history_A.pop(0)
            theta_A = sum(history_A) / len(history_A)
            next_theta = theta_A
        else:
            history_B.append(proposed_target)
            if len(history_B) > sma_window:
                history_B.pop(0)
            theta_B = sum(history_B) / len(history_B)
            next_theta = theta_B
        metrics = {
            "method": "FDA-RoLoRA-DualTrack-SMA",
            "round": r + 1,
            "active_matrix": active_matrix,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * round_steps,
            "variance": var_subspace,
            "avg_norm_sq_drifts": avg_norm_sq_drifts,
            "norm_sq_avg_drift": norm_sq_avg_drift,
            "theta_phase": current_theta
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished.")
        gc.collect()
        if hyperparams.get('early_stopping') and check_dynamic_early_stopping(hyperparams, metrics_handler.round_metrics):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break
    metrics_handler.save_metrics()
def fda_rolora_dual_track_calibrated_opt(hyperparams: dict):
    """
    Execute Dual-Track FDA-RoLoRA with configurable controller setpoint ratio
    and optional gated Matrix B variance discount.
    Configurable Hyperparameters:
      - `controller_ratio` (float, default 0.5):
          Multiplier for e_max in theta calculation:
          `target_epochs = controller_ratio * e_max`
          `next_theta = (target_epochs / round_epochs) * var_subspace`
          * 1.0: Targets E_max (e.g. 20 epochs when E_max=20).
          * 0.75: Targets 0.75 * E_max (e.g. 15 epochs when E_max=20).
          * 0.5: Default standard behavior (E_max / 2, e.g. 10 epochs).
      - `enable_b_discount` (bool, default False):
          Whether to discount Matrix B's measured variance when checking early break.
      - `b_discount_factor` (float, default 3.0):
          Discount divisor for Matrix B: `effective_variance = var_subspace / b_discount_factor`.
    """
    e_min = hyperparams.get('e_min', 3)
    e_max = hyperparams.get('max_epochs', 20)
    controller_ratio = float(hyperparams.get('controller_ratio', 0.5))
    enable_b_discount = bool(hyperparams.get('enable_b_discount', False))
    b_discount_factor = float(hyperparams.get('b_discount_factor', 3.0))
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    if hyperparams.get('ds_name') == 'mnist':
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(rank=hyperparams.get('lora_r', 8)).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
            lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
        )
        model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
        if not hyperparams.get('train_head', True):
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    num_lora = len(params_A)
    params_A = params_A + params_other
    params_B = params_B + params_other
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams, num_lora_params=num_lora)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams, num_lora_params=num_lora)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [p.detach().clone().to(get_save_device()) for p in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [p.detach().clone().to(get_save_device()) for p in params_B] for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    theta_A = hyperparams.get('theta', 0.0)
    theta_B = hyperparams.get('theta', 0.0)
    first_round_A = True
    first_round_B = True
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_matrix = 'A' if r % 2 == 0 else 'B'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        current_theta = theta_A if active_matrix == 'A' else theta_B
        is_first_round = first_round_A if active_matrix == 'A' else first_round_B
        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
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
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            if round_epochs >= e_min:
                sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                if is_first_round:
                    if active_matrix == 'A':
                        first_round_A = False
                    else:
                        first_round_B = False
                    break
                if enable_b_discount and active_matrix == 'B':
                    effective_variance = var_subspace / b_discount_factor
                else:
                    effective_variance = var_subspace
                if effective_variance > current_theta or round_epochs >= e_max:
                    break
            elif round_epochs >= e_max:
                sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                effective_variance = var_subspace / b_discount_factor if (enable_b_discount and active_matrix == 'B') else var_subspace
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
        if active_matrix == 'A':
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
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished (Matrix {active_matrix}, Epochs: {round_epochs}, Ratio: {controller_ratio}, B-Disc: {enable_b_discount}).")
        gc.collect()
        if hyperparams.get('early_stopping') and check_dynamic_early_stopping(hyperparams, metrics_handler.round_metrics):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break
    metrics_handler.save_metrics()
def fda_rolora_dual_track_pure_opt(hyperparams: dict):
    """
    Pure Closed-Loop FDA-RoLoRA Dual-Track Optimizer:
      1. Starts with Matrix B first (Round 1: B, Round 2: A, ...)
      2. No B-discount (NODISC): evaluations use true physical client drift variance.
      3. No e_min floor (e_min = 1).
      4. No first-round guard (natural cold-start bootstrap from initial Theta = 0.0).
      5. Configurable controller ratio setpoints: k = 1.0 (E_max) and k = 0.5 (E_max / 2).
    """
    e_min = int(hyperparams.get('e_min', 1))
    e_max = int(hyperparams.get('max_epochs', 20))
    controller_ratio = float(hyperparams.get('controller_ratio', 1.0))
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    if hyperparams.get('ds_name') == 'mnist':
        from fdaopt.models.toy_model import ToyModel
        model = ToyModel(rank=hyperparams.get('lora_r', 8)).to(utils.DEVICE)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
            lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
        )
        model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
        if not hyperparams.get('train_head', True):
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    num_lora = len(params_A)
    params_A = params_A + params_other
    params_B = params_B + params_other
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams, num_lora_params=num_lora)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams, num_lora_params=num_lora)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [p.detach().clone().to(get_save_device()) for p in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [p.detach().clone().to(get_save_device()) for p in params_B] for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    theta_A = 0.0
    theta_B = 0.0
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        current_theta = theta_A if active_matrix == 'A' else theta_B
        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
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
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
            client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
            var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
            if var_subspace > current_theta or round_epochs >= e_max:
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
        if active_matrix == 'A':
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
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished (Matrix {active_matrix}, Epochs: {round_epochs}, Ratio: {controller_ratio}, Pure NODISC).")
        gc.collect()
        if hyperparams.get('early_stopping') and check_dynamic_early_stopping(hyperparams, metrics_handler.round_metrics):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break

```


## File: `fdaopt/training/sketch.py`

```py
from fdaopt.utils import torch
from math import sqrt
import gc
DEVICE = 'cpu'
class AmsSketch:
    """
    AMS Sketch class for approximate second moment estimation in PyTorch.
    """
    def __init__(self, depth=3, width=500):
        self.depth = depth
        self.width = width
        self.epsilon = 1. / sqrt(width)
        self.F = torch.randint(0, (1 << 31) - 1, (6, depth), dtype=torch.int32)
        self.precomputed_dict = {}
    def precompute(self, d):
        pos_tensor = self.tensor_hash31(torch.arange(d), self.F[0], self.F[1]) % self.width
        four = self.tensor_fourwise(torch.arange(d)).float()
        self.precomputed_dict[('pos_tensor', d)] = pos_tensor.to(DEVICE)
        self.precomputed_dict[('four', d)] = four.to(DEVICE)
    @staticmethod
    def hash31(x, a, b):
        r = a * x + b
        fold = torch.bitwise_xor(r >> 31, r)
        return fold & 2147483647
    @staticmethod
    def tensor_hash31(x, a, b):
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., torch.arange(d)) """
        x_reshaped = x.unsqueeze(-1)
        r = a * x_reshaped + b
        fold = torch.bitwise_xor(r >> 31, r)
        return fold & 2147483647
    def tensor_fourwise(self, x):
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., torch.arange(d)) """
        in1 = self.tensor_hash31(x, self.F[2], self.F[3])
        in2 = self.tensor_hash31(x, in1, self.F[4])
        in3 = self.tensor_hash31(x, in2, self.F[5])
        in4 = in3 & 32768
        return 2 * (in4 >> 15) - 1
    def sketch_for_vector(self, v):
        """ Efficient computation of sketch using PyTorch tensors.
        Args:
        - v (torch.Tensor): Vector to sketch. Shape=(d,).
        Returns:
        - torch.Tensor: An AMS Sketch. Shape=(`depth`, `width`).
        """
        d = v.shape[0]
        if ('four', d) not in self.precomputed_dict:
            self.precompute(d)
        four, pos_tensor = self.precomputed_dict[('four', d)], self.precomputed_dict[('pos_tensor', d)]
        sketch = self._sketch_for_vector(v, four, pos_tensor)
        gc.collect()
        return sketch
    def _sketch_for_vector(self, v, four, pos_tensor):
        """
        PyTorch translation of the TensorFlow function using a simple for loop.
        Args:
        - v (torch.Tensor): Vector to sketch. Shape=(d,).
        - four (torch.Tensor): Precomputed fourwise tensor. Shape=(d, depth).
        - indices (torch.Tensor): Precomputed indices for scattering. Shape=(d, depth, 2).
        Returns:
        - sketch (torch.Tensor): The AMS sketch tensor. Shape=(depth, width).
        """
        v_expand = v.unsqueeze(-1).to(DEVICE)
        deltas_tensor = four * v_expand
        sketch = torch.zeros((self.depth, self.width), dtype=torch.float32).to(DEVICE)
        for i in range(self.depth):
            width_indices = pos_tensor[:, i]
            deltas = deltas_tensor[:, i]
            sketch[i].index_add_(0, width_indices, deltas)
        return sketch
    @staticmethod
    def estimate_euc_norm_squared(sketch):
        """ Estimate the Euclidean norm squared of a vector using its AMS sketch.
        Args:
        - sketch (torch.Tensor): AMS sketch of a vector. Shape=(`depth`, `width`).
        Returns:
        - float: Estimated squared Euclidean norm.
        """
        norm_sq_rows = torch.sum(sketch ** 2, dim=1)
        return torch.median(norm_sq_rows).item()

```


## File: `fdaopt/miscellaneous/parameter_tools.py`

```py
import json
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
HYPERPARAM_DIR = os.path.normpath(os.path.join(script_directory, '../../hyperparameters'))
def get_hyperparameters(hyperparam_filename):
    """
    Get the hyperparameters from a JSON file.
    Args:
        hyperparam_filename (str): The name of the JSON file (e.g. '0.json')
    Returns:
        dict: The hyperparameters as a dictionary.
    """
    with open(f'{HYPERPARAM_DIR}/{hyperparam_filename}', 'r') as f:
        hyperparams = json.load(f)
    return hyperparams

```


## File: `fdaopt/miscellaneous/create_hyperparameters.py`

```py
import json
import os
import argparse
import ast
script_directory = os.path.dirname(os.path.abspath(__file__))
HYPERPARAM_DIR = os.path.normpath(os.path.join(script_directory, '../../hyperparameters'))
def get_opt_args(opt_args, where):
    """ Get the optional arguments for the optimizer.
    Args:
        opt_args (list): The list of optional arguments.
        where (str): Either "server_opt" or "client_opt".
    Returns:
        dict: The optional arguments as a dictionary.
    """
    opt_params = {}
    if opt_args:
        for arg in opt_args:
            key, value = arg.split('=')
            opt_params[f"{where}_{key}"] = ast.literal_eval(value)
    return opt_params
def get_next_filename():
    """ Get the next filename for the hyperparameters.
    Returns:
        str: The next filename for the hyperparameters.
    """
    files = os.listdir(HYPERPARAM_DIR)
    return f"{len(files)}"
def create_hyperparameters(args):
    hyperparams = {
        'checkpoint': args.checkpoint,
        'ds_path': args.ds_path,
        'ds_name': args.ds_name,
        'num_labels': args.num_labels,
        'num_clients': args.num_clients,
        'clients_per_round': args.clients_per_round,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'local_epochs': args.local_epochs,
        'max_epochs': args.max_epochs,
        'total_rounds': args.total_rounds,
        'fda': args.fda,
        'threshold': args.threshold,
        'theta': args.theta,
        'train_head': not args.disable_train_head,
        'extras': args.extras
    }
    server_opt_args = get_opt_args(args.server_opt_args, "server_opt")
    client_opt_args = get_opt_args(args.client_opt_args, "client_opt")
    hyperparams = {
        **hyperparams,
        **{'server_opt_name': args.server_opt_name},
        **server_opt_args,
        **{'client_opt_name': args.client_opt_name},
        **client_opt_args
    }
    filename = get_next_filename()
    with open(f'{HYPERPARAM_DIR}/{filename}.json', 'w') as f:
        json.dump(hyperparams, f)
    print(f"OK! Created hyperparameters in {filename}.json.")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help="The checkpoint to use for the model.", default='roberta-base')
    parser.add_argument('--ds_path', type=str, help="The path to the dataset.", default='glue')
    parser.add_argument('--ds_name', type=str, help="The name of the dataset.", default='mrpc')
    parser.add_argument('--num_labels', type=int, help="The number of labels in the dataset.", default=2)
    parser.add_argument('--num_clients', type=int, help="The number of clients in the federated learning setup.", default=100)
    parser.add_argument('--clients_per_round', type=int, help="The number of clients to sample per round.", default=10)
    parser.add_argument('--alpha', type=float, help="The alpha parameter of the Dirichlet distribution.", default=1.)
    parser.add_argument('--batch_size', type=int, help="The batch size for the training.", default=8)
    parser.add_argument('--local_epochs', type=int, help="FedOpt: The number of local epochs until we synchronize the model.", default=1)
    parser.add_argument('--max_epochs', type=int, help="FDA-Opt: The number of local epochs until we forcibly synchronize the model.", default=10)
    parser.add_argument('--total_rounds', type=int, help="The total number of rounds for the training.", default=1000)
    parser.add_argument('--fda', action='store_true', help="If given, then we train with FDA.")
    parser.add_argument('--threshold', action='store_true', help="If given, then we stop training when the metric threshold is reached. See `stop_training_with_threshold` in models/ops.")
    parser.add_argument('--theta', type=float, help="The theta parameter for the FDA.", default=0.0)
    parser.add_argument('--server_opt_name', type=str, help="The name of the server optimizer.", required=True)
    parser.add_argument('--client_opt_name', type=str, help="The name of the client optimizer.", required=True)
    parser.add_argument('--server_opt_args', nargs='*', help="Hyperparameters for the server optimizer (e.g., --server_opt_args lr=0.001 betas='(0.9, 0.999)'")
    parser.add_argument('--client_opt_args', nargs='*', help="Hyperparameters for the client optimizer (e.g., --client_opt_args lr=0.001")
    parser.add_argument('--extras', type=str, help="Extra info.", default='')
    parser.add_argument('--disable_train_head', action='store_true', help="If given, then we disable training of the classification head.")
    create_hyperparameters(parser.parse_args())

```


## File: `fdaopt/datasets/fed_data_prep.py`

```py
from fdaopt.utils import np, random, DataLoader, AutoTokenizer, DataCollatorWithPadding, load_dataset
def calculate_prior_distribution(raw_train_dataset):
    """
    Calculate the prior distribution of labels in the training dataset.
    This function calculates the proportion of each label in the training dataset,
    which is used as the prior distribution for Dirichlet sampling.
    Args:
        raw_train_dataset (datasets.arrow_dataset.Dataset): The training dataset.
    Returns:
        tuple: A tuple containing:
            - prior_distribution (numpy.ndarray): An array of label proportions.
            - num_labels (int): The number of unique labels in the dataset.
    """
    label_count_dict = dict(raw_train_dataset.to_pandas()['label'].value_counts())
    n = len(raw_train_dataset)
    label_count_tuple = sorted(label_count_dict.items(), key=lambda x: x[0])
    prior_distribution = np.array([c / n for _, c in label_count_tuple])
    num_labels = len(label_count_dict)
    return prior_distribution, num_labels
def sample_to_client(client_sample_counts, sample_label):
    """
    Assign a sample to a client based on the sample's label and the remaining counts for each client.
    This function assigns a sample to the client that still needs more samples of the given label.
    It decrements the count for that label for the chosen client. If no client has a remaining count
    of at least 1 for the label, the function assigns the sample to the client with the largest leftover count
    for that label.
    Args:
        client_sample_counts (numpy.ndarray): A 2D array where each row corresponds to a client and each column corresponds
            to the count (float) of samples needed for each label. The element at (i, j) represents the decimal number of
            samples of label j that client i still needs.
        sample_label (int): The label of the sample to be assigned to a client.
    Returns:
        int: The index of the client to which the sample has been assigned.
    """
    num_clients = len(client_sample_counts)
    client_indices = np.random.permutation(num_clients)
    for client_idx in client_indices:
        client_sample_count = client_sample_counts[client_idx]
        if client_sample_count[sample_label] >= 1:
            client_sample_count[sample_label] -= 1
            return client_idx
    client_idx = np.argmax(client_sample_counts[:, sample_label])
    client_sample_counts[client_idx][sample_label] -= 1
    return client_idx
def federated_dirichlet_datasets(raw_train_dataset, prior_distribution, num_clients, alpha):
    """
    Create federated datasets using Dirichlet-distributed label partitions.
    This function partitions the training dataset into multiple subsets, each corresponding
    to a client. The label distribution for each client is drawn from a Dirichlet distribution
    parameterized by the given prior distribution and concentration parameter alpha.
    The function also aims to keep the datasets as equal in size as possible.
    Args:
        raw_train_dataset (datasets.arrow_dataset.Dataset): The training dataset.
        prior_distribution (numpy.ndarray): An array representing the prior distribution of labels.
        num_clients (int): The number of clients.
        alpha (float): The concentration parameter for the Dirichlet distribution.
    Returns:
        list: A list of `datasets.arrow_dataset.Dataset` objects, each representing a client's dataset.
    """
    n = len(raw_train_dataset)
    client_num_samples = n / num_clients
    client_distributions = np.random.dirichlet(alpha * prior_distribution, num_clients)
    client_sample_indices = [[] for i in range(num_clients)]
    client_sample_counts = np.array([
        client_distributions[client_idx, :] * client_num_samples
        for client_idx in range(num_clients)
    ])
    for sample_idx, sample in enumerate(raw_train_dataset):
        client_idx = sample_to_client(client_sample_counts, sample['label'])
        client_sample_indices[client_idx].append(sample_idx)
    client_datasets = []
    for client_indices in client_sample_indices:
        client_dataset = raw_train_dataset.select(client_indices)
        client_datasets.append(client_dataset)
    return client_datasets
def tokenize_client_datasets(client_datasets, tokenize_fn):
    """
    Tokenize and preprocess a list of client datasets.
    This function tokenizes and preprocesses each dataset in the provided list of client datasets.
    It applies the specified tokenization function, renames the "label" column to "labels",
    removes unnecessary columns, and sets the format to PyTorch tensors.
    Args:
        client_datasets (list of datasets.arrow_dataset.Dataset): A list of client datasets to be tokenized and preprocessed.
        tokenize_fn (function): A function that takes an example and returns its tokenized form.
    Returns:
        list of datasets.arrow_dataset.Dataset: A list of tokenized and preprocessed client datasets.
    """
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    tok_client_datasets = []
    for client_dataset in client_datasets:
        tok_client_dataset = client_dataset.map(tokenize_fn, batched=True)
        tok_client_dataset = tok_client_dataset.rename_column("label", "labels")
        columns_to_remove = [
            column for column in tok_client_dataset.column_names
            if column not in expected_columns
        ]
        tok_client_dataset = tok_client_dataset.remove_columns(columns_to_remove)
        tok_client_dataset.set_format("torch")
        tok_client_datasets.append(tok_client_dataset)
    return tok_client_datasets
def preprocess_test_dataset(raw_test_dataset, tokenize_fn, data_collator, batch_size):
    """
    Preprocess and tokenize the test dataset, then create a DataLoader for it.
    This function tokenizes and preprocesses the provided test dataset using the specified
    tokenization function. It renames the "label" column to "labels", removes unnecessary columns,
    sets the format to PyTorch tensors, and then creates a DataLoader for the test dataset.
    Args:
        raw_test_dataset (datasets.arrow_dataset.Dataset): The raw test dataset to be tokenized and preprocessed.
        tokenize_fn (function): A function that takes an example and returns its tokenized form.
        data_collator (transformers.DataCollator): A data collator to be used for padding and batching.
        batch_size (int): The batch size to be used by the DataLoader.
    Returns:
        DataLoader: A DataLoader for the tokenized and preprocessed test dataset.
    """
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    tok_test_dataset = raw_test_dataset.map(tokenize_fn, batched=True)
    tok_test_dataset = tok_test_dataset.rename_column("label", "labels")
    columns_to_remove = [
        column for column in tok_test_dataset.column_names
        if column not in expected_columns
    ]
    tok_test_dataset = tok_test_dataset.remove_columns(columns_to_remove)
    tok_test_dataset.set_format("torch")
    test_ds = DataLoader(
        tok_test_dataset, batch_size=batch_size, collate_fn=data_collator
    )
    return test_ds
def create_client_dataloaders(tok_client_datasets, batch_size, collate_fn, num_workers=0, pin_memory=False):
    """
    Create data loaders for a list of tokenized client datasets.
    This function takes a list of tokenized client datasets and creates a DataLoader for each dataset.
    The resulting data loaders are stored in a list and returned.
    Args:
        tok_client_datasets (list of datasets.arrow_dataset.Dataset): A list of tokenized client datasets.
        batch_size (int): The batch size to be used by the data loaders.
        collate_fn (function): A collate function to be used by the data loaders.
    Returns:
        list of DataLoader: A list of data loaders, each corresponding to a tokenized client dataset.
    """
    client_dataloaders = []
    for tok_client_dataset in tok_client_datasets:
        client_dataloader = DataLoader(
            tok_client_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
        client_dataloaders.append(client_dataloader)
    return client_dataloaders
def tokenize_function(ds_path, ds_name, tokenizer):
    """
    Return a tokenization function based on the dataset path and name.
    This function returns the appropriate tokenization function for the specified dataset.
    Currently, it supports the GLUE MRPC dataset.
    Args:
        ds_path (str): The path or identifier of the dataset.
        ds_name (str): The name of the dataset.
        tokenizer (AutoTokenizer): The tokenizer to be used.
    Returns:
        function: A tokenization function.
    """
    if ds_path == "glue" or ds_path == "nyu-mll/glue":
        if ds_name == "mrpc":
            return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
        if ds_name == "rte":
            return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
        if ds_name == "stsb":
            return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
        if ds_name == "cola":
            return lambda example: tokenizer(example["sentence"], truncation=True)
        if ds_name == "sst2":
            return lambda example: tokenizer(example["sentence"], truncation=True)
        if ds_name == "qnli":
            return lambda example: tokenizer(example["question"], example["sentence"], truncation=True)
        if ds_name == "mnli":
            return lambda example: tokenizer(example["premise"], example["hypothesis"], truncation=True)
    return None
def test_dataset_split(raw_datasets, ds_path, ds_name):
    """
    Split the raw dataset into the appropriate test set.
    This function extracts the test set from the provided raw dataset based on the dataset path.
    For the GLUE dataset, it returns the validation split as the test set following prior work.
    Args:
        raw_datasets (datasets.DatasetDict): The raw dataset containing multiple splits.
        ds_path (str): The path or identifier of the dataset.
    Returns:
        Dictionary with key the name of the test ds and value a datasets.Dataset from the extracted test dataset split.
    """
    if ds_path == 'nyu-mll/glue' or ds_path == 'glue':
        if ds_name == 'mnli':
            return {'mnli_matched': raw_datasets['validation_matched'],
                    'mnli_mismatched': raw_datasets['validation_mismatched']}
        else:
            return {ds_name: raw_datasets['validation']}
def prepare_federated_datasets(ds_path, ds_name, checkpoint, num_clients, alpha, batch_size, num_workers=0, pin_memory=False):
    raw_datasets = load_dataset(path=ds_path, name=ds_name)
    raw_train_dataset = raw_datasets['train']
    MAX_SAMPLES = 25000
    train_limit = min(len(raw_train_dataset), MAX_SAMPLES)
    raw_train_dataset = raw_train_dataset.select(range(train_limit))
    if ds_name == 'mnist':
        classes_per_client = 1 if num_clients == 10 else 2
        client_datasets = extreme_non_iid_datasets(raw_train_dataset, num_clients, classes_per_client)
        tok_client_datasets = []
        for client_ds in client_datasets:
            tok_ds = client_ds.rename_column("label", "labels")
            tok_ds.set_format("torch", columns=["image", "labels"])
            tok_client_datasets.append(tok_ds)
        test_ds_formatted = raw_datasets['test'].rename_column("label", "labels")
        test_ds_formatted.set_format("torch", columns=["image", "labels"])
        client_dataloaders = create_client_dataloaders(tok_client_datasets, batch_size, None, num_workers, pin_memory)
        fed_ds = FederatedDataset(client_dataloaders)
        test_ds = {'mnist': DataLoader(test_ds_formatted, batch_size=batch_size)}
        return fed_ds, test_ds
    raw_test_datasets = test_dataset_split(raw_datasets, ds_path, ds_name)
    for split_name in raw_test_datasets.keys():
        test_limit = min(len(raw_test_datasets[split_name]), MAX_SAMPLES)
        raw_test_datasets[split_name] = raw_test_datasets[split_name].select(range(test_limit))
    prior_distribution, num_labels = calculate_prior_distribution(raw_train_dataset)
    if checkpoint in ['microsoft/deberta-v3-base', 'prajjwal1/bert-tiny']:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenize_fn = tokenize_function(ds_path, ds_name, tokenizer)
    tok_raw_train_dataset = raw_train_dataset.map(tokenize_fn, batched=True)
    client_datasets = federated_dirichlet_datasets(tok_raw_train_dataset, prior_distribution, num_clients, alpha)
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    tok_client_datasets = []
    for client_ds in client_datasets:
        tok_ds = client_ds.rename_column("label", "labels")
        columns_to_remove = [col for col in tok_ds.column_names if col not in expected_columns]
        tok_ds = tok_ds.remove_columns(columns_to_remove)
        tok_ds.set_format("torch")
        tok_client_datasets.append(tok_ds)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    client_dataloaders = create_client_dataloaders(tok_client_datasets, batch_size, data_collator, num_workers=num_workers, pin_memory=pin_memory)
    fed_ds = FederatedDataset(client_dataloaders)
    test_ds = {k: preprocess_test_dataset(v, tokenize_fn, data_collator, batch_size) for k, v in raw_test_datasets.items()}
    return fed_ds, test_ds
def extreme_non_iid_datasets(raw_train_dataset, num_clients, classes_per_client):
    import numpy as np
    df = raw_train_dataset.to_pandas()
    num_classes = len(df['label'].unique())
    client_datasets = []
    if num_clients == 10 and classes_per_client == 1:
        classes_assigned = [[i] for i in range(10)]
    elif num_clients == 5 and classes_per_client == 2:
        classes_assigned = [[2*i, 2*i+1] for i in range(5)]
    else:
        raise ValueError("Unsupported non-IID configuration.")
    samples_per_client = len(raw_train_dataset) // num_clients
    samples_per_class_per_client = samples_per_client // classes_per_client
    label_indices = {label: df.index[df['label'] == label].tolist() for label in range(num_classes)}
    for client_classes in classes_assigned:
        client_indices = []
        for c in client_classes:
            idx_to_take = label_indices[c][:samples_per_class_per_client]
            label_indices[c] = label_indices[c][samples_per_class_per_client:]
            client_indices.extend(idx_to_take)
        client_datasets.append(raw_train_dataset.select(client_indices))
    return client_datasets
class FederatedDataset:
    """
    A class to handle federated datasets for training in a federated learning setup.
    This class encapsulates the logic for managing multiple client dataloaders and providing
    batched data for federated learning training loops.
    Args:
        client_dataloaders (list of DataLoader): A list of DataLoader objects, each corresponding to a client's dataset.
    """
    def __init__(self, client_dataloaders):
        self.client_dataloaders = client_dataloaders
        self.client_batch_generators = [
            self.dataloader_batch_generator(client_dl)
            for client_dl in client_dataloaders
        ]
    def epoch_steps(self, client_ids):
        """
        Determine the number of steps (batches) in an epoch for the given clients.
        Args:
            client_ids (list of int): A list of client IDs.
        Returns:
            int: The maximum number of steps (batches) for the given clients.
        """
        return max(
            len(self.client_dataloaders[client_id])
            for client_id in client_ids
        )
    def next_client_batch(self, client_id):
        """
        Retrieve the next batch of data for the specified client.
        Args:
            client_id (int): The ID of the client.
        Returns:
            dict: A batch of data from the client's DataLoader.
        """
        return next(
            self.client_batch_generators[client_id]
        )
    @staticmethod
    def dataloader_batch_generator(dataloader):
        """
        A generator that yields (unending) batches of data from a DataLoader.
        Args:
            dataloader (DataLoader): A DataLoader object.
        Yields:
            dict: A batch of data from the DataLoader.
        """
        while True:
            for batch in dataloader:
                yield batch
class ClientSampler:
    """
    A class that helps random sampling of clients in each round.
    Attributes:
        client_ids (list): A list of client IDs to be sampled.
        sample_size (int): The size of each group of sampled client IDs.
        num_clients (int): The total number of clients.
        client_sample_generator (generator): A generator that yields groups of client IDs.
    """
    def __init__(self, client_ids, sample_size):
        self.client_ids = client_ids
        self.sample_size = sample_size
        self.num_clients = len(client_ids)
        if self.num_clients % self.sample_size != 0:
            raise ValueError("The number of clients must be divisible with the sample size.")
        self.client_sample_generator = self.sample_generator()
    def sample(self):
        """
        Returns the next group of sampled client IDs.
        Returns:
            list: A list of sampled client IDs of length `sample_size`.
        """
        return next(self.client_sample_generator)
    def sample_generator(self):
        """
        A generator function to continuously generate groups of sampled client IDs.
        This function shuffles the client IDs and partitions them into groups of the specified sample size.
        It yields each group of sampled client IDs.
        Yields:
            list: A list of sampled client IDs of length `sample_size`.
        """
        while True:
            shuffled_client_ids = random.sample(self.client_ids, self.num_clients)
            groups = [shuffled_client_ids[i:i + self.sample_size] for i in range(0, self.num_clients, self.sample_size)]
            for sampled_client_ids in groups:
                yield sampled_client_ids

```


## File: `fdaopt/metrics/mentrics_handler.py`

```py
import pandas as pd
import os
import hashlib
import json
script_dir = os.path.dirname(os.path.realpath(__file__))
tmp_dir = '../../results'
ROUND_METRICS_PATH = os.path.normpath(os.path.join(script_dir, f'{tmp_dir}/round_metrics'))
class MetricsHandler:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        params_str = json.dumps(self.hyperparams, sort_keys=True).encode('utf-8')
        short_hash = hashlib.md5(params_str).hexdigest()[:10]
        prefix = self.hyperparams.get('extras', 'metrics')
        self.filename = f"{prefix}_{short_hash}.csv"
        self.round_metrics = []
    def append_round_metrics(self, metrics_dict):
        self.round_metrics.append(metrics_dict)
    def save_metrics(self, filepath=ROUND_METRICS_PATH):
        round_metrics_with_id = [
            {**self.hyperparams, **round_metrics}
            for round_metrics in self.round_metrics
        ]
        df = pd.DataFrame(
            round_metrics_with_id,
            columns=list(self.hyperparams.keys()) + list(self.round_metrics[0].keys())
        )
        df.to_csv(f"{filepath}/{self.filename}")

```


## File: `fillers/thesis_filler_experiments.py`

```py
"""Thesis Filler Experiments Orchestrator (Polytechnix Deployment).
Executes the remaining non-IID configurations to complete the thesis matrix:
  - Tier 1: C=50 Non-IID missing k=1.0 runs (MNLI and QNLI, Seeds 42 & 43) [4 runs]
  - Tier 2: C=20 Non-IID deep-compute replication on Seed 43 (Emax=40 & RoLoRA E20) [5 runs]
  - Tier 3: Additional Non-IID tasks (QQP & SST-2 at alpha=0.5 across C=20 & C=50) [24 runs]
Features:
  - Optimized, calibrated early stopping with realistic entry gates, tighter activation rounds (MR),
    and post-peak degradation guards.
  - Hardware & Runtime Acceleration (Self-Contained in fillers/):
      1. Setting 1 (TF32): TensorFloat-32 on Ampere Tensor Cores + cuDNN benchmark mode (~1.42x speedup).
      2. Setting 2 (Zero-Sync): In-memory monkey-patching of client_round_train to accumulate metrics
         directly on GPU registers, eliminating 1,000+ CPU-GPU .item() halts per round (~1.09x speedup).
      3. Setting 3 (Pinned Memory & DMA): pin_memory=True + non_blocking=True PCIe DMA streaming (~1.57x transfer).
      Combined end-to-end acceleration: ~1.75x faster execution (~43% total time saved) with ZERO core changes.
  - Multi-GPU scheduler supporting dynamic GPU lane assignment on cuda:0 and cuda:1.
  - Interactive operator controls: plan, start, status, stop <cid>, drop <cid>.
  - Full isolation of logs and control files in results/run_control_filler.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
try:
    import fcntl
except ImportError:
    fcntl = None
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "fillers" else Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import thesis_experiments as base
ROLORA_METHOD = "FedAvg-RoLoRA"
PURE_METHOD = "FDA-RoLoRA-DualTrack-Pure"
DEFAULT_SEEDS = [42, 43]
DEFAULT_CLIENT_COUNTS = [20, 50]
DEFAULT_E_MIN = 1
DEFAULT_E_MAX = 20
LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = ("query", "value")
FILLER_EARLY_STOPPING_MIN_DELTA = 0.005
FILLER_EARLY_STOPPING_DROP_TOLERANCE = 0.03
FILLER_GATES = {
    "qnli": {20: 0.80, 50: 0.70},
    "mnli": {20: 0.65, 50: 0.50},
    "qqp":  {20: 0.70, 50: 0.68},
    "sst2": {20: 0.85, 50: 0.82},
}
FILLER_MIN_ROUNDS = {
    "qnli": {20: 18, 50: 12},
    "mnli": {20: 20, 50: 12},
    "qqp":  {20: 8,  50: 12},
    "sst2": {20: 14, 50: 12},
}
FILLER_MAX_ROUNDS = {
    "qnli": {20: 35, 50: 22},
    "mnli": {20: 42, 50: 24},
    "qqp":  {20: 20, 50: 24},
    "sst2": {20: 25, 50: 20},
}
FILLER_PATIENCE = {
    "qnli": {20: 2, 50: 2},
    "mnli": {20: 3, 50: 3},
    "qqp":  {20: 2, 50: 3},
    "sst2": {20: 2, 50: 2},
}
FILLER_WINDOW = {
    "qnli": {20: 2, 50: 2},
    "mnli": {20: 3, 50: 3},
    "qqp":  {20: 2, 50: 3},
    "sst2": {20: 2, 50: 2},
}
DATASET_SETTINGS = {
    "qnli": {
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "learning_rate": 0.10,
    },
    "mnli": {
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "learning_rate": 0.10,
    },
    "qqp": {
        "batch_size": 32,
        "layers": list(range(7, 12)),
        "learning_rate": 0.10,
    },
    "sst2": {
        "batch_size": 64,
        "layers": list(range(9, 12)),
        "learning_rate": 0.10,
    },
}
RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "round_metrics"
LOG_DIR = RESULTS_DIR / "logs" / "thesis_filler"
RUN_CONTROL_DIR = RESULTS_DIR / "run_control_filler"
ACTIVE_RUNS_DIR = RUN_CONTROL_DIR / "active"
STOP_REQUESTS_DIR = RUN_CONTROL_DIR / "stop_requests"
DROPPED_RUNS_DIR = RUN_CONTROL_DIR / "dropped"
OPERATOR_EVENTS_FILE = RUN_CONTROL_DIR / "operator_events.jsonl"
SCHEDULER_LOCK_FILE = RUN_CONTROL_DIR / "scheduler.lock"
JOBS_PER_GPU = max(1, int(os.environ.get("THESIS_FILLER_JOBS_PER_GPU", "3")))
BATCH64_JOBS_PER_GPU = max(
    1,
    min(JOBS_PER_GPU, int(os.environ.get("THESIS_FILLER_BATCH64_JOBS_PER_GPU", "2"))),
)
def configure_base_runner() -> None:
    """Wire custom paths and method registrations into base runner."""
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
    base.MAIN_INTERRUPTED_SNAPSHOT_RUN_IDS = frozenset()
    base.MAIN_JOBS_PER_GPU = JOBS_PER_GPU
    base.MAIN_BATCH64_JOBS_PER_GPU = BATCH64_JOBS_PER_GPU
    base.REPO_ROOT = REPO_ROOT
    base.RESULTS_DIR = RESULTS_DIR
    base.METRICS_DIR = METRICS_DIR
    base.is_completed = is_filler_completed
    base.LOG_DIR = LOG_DIR
    base.RUN_CONTROL_DIR = RUN_CONTROL_DIR
    base.ACTIVE_RUNS_DIR = ACTIVE_RUNS_DIR
    base.STOP_REQUESTS_DIR = STOP_REQUESTS_DIR
    base.OPERATOR_EVENTS_FILE = OPERATOR_EVENTS_FILE
    base.main_config_priority = filler_config_priority
    base.wait_for_registered_run = wait_for_registered_filler_run
    base.run_config_on_device = run_filler_config_on_device
def filler_config_priority(config: dict) -> tuple:
    """Tiered scheduling priority."""
    tier = int(config.get("_tier_idx", 1))
    is_batch64 = int(config.get("batch_size", 32)) >= 64
    b64_tier = 1 if is_batch64 else 0
    ds_order = ["qnli", "mnli", "qqp", "sst2"]
    ds_idx = ds_order.index(config["ds_name"]) if config["ds_name"] in ds_order else 99
    c_order = int(config["num_clients"])
    ratio_val = float(config.get("controller_ratio", 1.0))
    ratio_order = 0 if ratio_val == 1.0 else 1
    seed_order = int(config["seed"])
    return (tier, b64_tier, ds_idx, c_order, ratio_order, seed_order)
def build_filler_configs(tier: str = "all") -> list[dict]:
    """Build filler benchmark configurations filtered by tier."""
    configs = []
    if tier in ("tier1", "tier1-2", "all"):
        for ds in ["qnli", "mnli"]:
            settings = DATASET_SETTINGS[ds]
            c = 50
            rounds = FILLER_MAX_ROUNDS[ds][c]
            gate = FILLER_GATES[ds][c]
            mr = FILLER_MIN_ROUNDS[ds][c]
            patience = FILLER_PATIENCE[ds][c]
            window = FILLER_WINDOW[ds][c]
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
                cfg["METHOD_NAME"] = PURE_METHOD
                cfg.update({
                    "server_opt_name": "SGD",
                    "server_opt_lr": 1.0,
                    "client_opt_name": "SGD",
                    "num_workers": 0,
                    "pin_memory": True,
                    "train_head": False,
                    "early_stopping": True,
                    "threshold": False,
                    "paper_parameter_budget": "P2",
                    "e_min": DEFAULT_E_MIN,
                    "max_epochs": DEFAULT_E_MAX,
                    "theta": 0.0,
                    "controller_ratio": 1.0,
                    "enable_b_discount": False,
                    "b_discount_factor": 1.0,
                    "start_matrix": "B",
                    "first_round_guard": False,
                    "early_stopping_start_accuracy": gate,
                    "plateau_min_accuracy": gate,
                    "early_stopping_min_rounds": mr,
                    "early_stopping_patience": patience,
                    "early_stopping_min_delta": FILLER_EARLY_STOPPING_MIN_DELTA,
                    "early_stopping_window": window,
                    "early_stopping_drop_tolerance": FILLER_EARLY_STOPPING_DROP_TOLERANCE,
                    "_tier_idx": 1,
                    "_tier_label": "TIER 1 (C=50 k=1.0)",
                })
                extra = (
                    f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                    f"-M-fda-pure-LR0.1-RK{LORA_RANK}-E{DEFAULT_E_MIN}TO{DEFAULT_E_MAX}-CRATIO1-NODISC-BSTART-NOGUARD-S{s}"
                    f"-ES{gate:g}-MR{mr}-P{patience}-D{FILLER_EARLY_STOPPING_MIN_DELTA:g}-W{window}"
                )
                cfg["extras"] = extra
                configs.append(cfg)
    if tier in ("tier2", "tier1-2", "all"):
        c = 20
        s = 43
        for ds in ["qnli", "mnli"]:
            settings = DATASET_SETTINGS[ds]
            rounds = FILLER_MAX_ROUNDS[ds][c]
            gate = FILLER_GATES[ds][c]
            mr = FILLER_MIN_ROUNDS[ds][c]
            patience = FILLER_PATIENCE[ds][c]
            window = FILLER_WINDOW[ds][c]
            cfg_ro = base.make_config(
                stage="non-iid",
                dataset=ds,
                method="FedAvg-RoLoRA",
                seed=s,
                num_clients=c,
                total_rounds=rounds,
                alpha=0.5,
                learning_rate=settings["learning_rate"],
                rank=LORA_RANK,
                layers=settings["layers"],
                batch_size=settings["batch_size"],
                clients_per_round=c,
                local_epochs=20,
                lora_alpha=LORA_ALPHA,
                target_modules=TARGET_MODULES,
            )
            cfg_ro["METHOD_NAME"] = ROLORA_METHOD
            cfg_ro.update({
                "server_opt_name": "SGD",
                "server_opt_lr": 1.0,
                "client_opt_name": "SGD",
                "num_workers": 0,
                "pin_memory": True,
                "train_head": False,
                "early_stopping": True,
                "threshold": False,
                "paper_parameter_budget": "P2",
                "start_matrix": "B",
                "first_round_guard": False,
                "early_stopping_start_accuracy": gate,
                "plateau_min_accuracy": gate,
                "early_stopping_min_rounds": mr,
                "early_stopping_patience": patience,
                "early_stopping_min_delta": FILLER_EARLY_STOPPING_MIN_DELTA,
                "early_stopping_window": window,
                "early_stopping_drop_tolerance": FILLER_EARLY_STOPPING_DROP_TOLERANCE,
                "_tier_idx": 2,
                "_tier_label": "TIER 2 (C=20 S43 Deep)",
            })
            extra_ro = (
                f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                f"-M-rolora-E20-LR0.1-RK{LORA_RANK}-BSTART-S{s}"
                f"-ES{gate:g}-MR{mr}-P{patience}-D{FILLER_EARLY_STOPPING_MIN_DELTA:g}-W{window}"
            )
            cfg_ro["extras"] = extra_ro
            configs.append(cfg_ro)
            cfg_fda40_k05 = base.make_config(
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
            cfg_fda40_k05["METHOD_NAME"] = PURE_METHOD
            cfg_fda40_k05.update({
                "server_opt_name": "SGD",
                "server_opt_lr": 1.0,
                "client_opt_name": "SGD",
                "num_workers": 0,
                "pin_memory": True,
                "train_head": False,
                "early_stopping": True,
                "threshold": False,
                "paper_parameter_budget": "P2",
                "e_min": DEFAULT_E_MIN,
                "max_epochs": 40,
                "theta": 0.0,
                "controller_ratio": 0.5,
                "enable_b_discount": False,
                "b_discount_factor": 1.0,
                "start_matrix": "B",
                "first_round_guard": False,
                "early_stopping_start_accuracy": gate,
                "plateau_min_accuracy": gate,
                "early_stopping_min_rounds": mr,
                "early_stopping_patience": patience,
                "early_stopping_min_delta": FILLER_EARLY_STOPPING_MIN_DELTA,
                "early_stopping_window": window,
                "early_stopping_drop_tolerance": FILLER_EARLY_STOPPING_DROP_TOLERANCE,
                "_tier_idx": 2,
                "_tier_label": "TIER 2 (C=20 S43 Deep)",
            })
            extra_fda40_k05 = (
                f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                f"-M-fda-pure-LR0.1-RK{LORA_RANK}-E{DEFAULT_E_MIN}TO40-CRATIO0.5-NODISC-BSTART-NOGUARD-S{s}"
                f"-ES{gate:g}-MR{mr}-P{patience}-D{FILLER_EARLY_STOPPING_MIN_DELTA:g}-W{window}"
            )
            cfg_fda40_k05["extras"] = extra_fda40_k05
            configs.append(cfg_fda40_k05)
            if ds == "qnli":
                cfg_fda40_k1 = base.make_config(
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
                cfg_fda40_k1["METHOD_NAME"] = PURE_METHOD
                cfg_fda40_k1.update({
                    "server_opt_name": "SGD",
                    "server_opt_lr": 1.0,
                    "client_opt_name": "SGD",
                    "num_workers": 0,
                    "pin_memory": True,
                    "train_head": False,
                    "early_stopping": True,
                    "threshold": False,
                    "paper_parameter_budget": "P2",
                    "e_min": DEFAULT_E_MIN,
                    "max_epochs": 40,
                    "theta": 0.0,
                    "controller_ratio": 1.0,
                    "enable_b_discount": False,
                    "b_discount_factor": 1.0,
                    "start_matrix": "B",
                    "first_round_guard": False,
                    "early_stopping_start_accuracy": gate,
                    "plateau_min_accuracy": gate,
                    "early_stopping_min_rounds": mr,
                    "early_stopping_patience": patience,
                    "early_stopping_min_delta": FILLER_EARLY_STOPPING_MIN_DELTA,
                    "early_stopping_window": window,
                    "early_stopping_drop_tolerance": FILLER_EARLY_STOPPING_DROP_TOLERANCE,
                    "_tier_idx": 2,
                    "_tier_label": "TIER 2 (C=20 S43 Deep)",
                })
                extra_fda40_k1 = (
                    f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                    f"-M-fda-pure-LR0.1-RK{LORA_RANK}-E{DEFAULT_E_MIN}TO40-CRATIO1-NODISC-BSTART-NOGUARD-S{s}"
                    f"-ES{gate:g}-MR{mr}-P{patience}-D{FILLER_EARLY_STOPPING_MIN_DELTA:g}-W{window}"
                )
                cfg_fda40_k1["extras"] = extra_fda40_k1
                configs.append(cfg_fda40_k1)
    if tier in ("tier3", "all"):
        for ds in ["qqp", "sst2"]:
            settings = DATASET_SETTINGS[ds]
            for c in DEFAULT_CLIENT_COUNTS:
                rounds = FILLER_MAX_ROUNDS[ds][c]
                gate = FILLER_GATES[ds][c]
                mr = FILLER_MIN_ROUNDS[ds][c]
                patience = FILLER_PATIENCE[ds][c]
                window = FILLER_WINDOW[ds][c]
                for s in DEFAULT_SEEDS:
                    cfg_ro = base.make_config(
                        stage="non-iid",
                        dataset=ds,
                        method="FedAvg-RoLoRA",
                        seed=s,
                        num_clients=c,
                        total_rounds=rounds,
                        alpha=0.5,
                        learning_rate=settings["learning_rate"],
                        rank=LORA_RANK,
                        layers=settings["layers"],
                        batch_size=settings["batch_size"],
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
                        "pin_memory": True,
                        "train_head": False,
                        "early_stopping": True,
                        "threshold": False,
                        "paper_parameter_budget": "P2",
                        "start_matrix": "B",
                        "first_round_guard": False,
                        "early_stopping_start_accuracy": gate,
                        "plateau_min_accuracy": gate,
                        "early_stopping_min_rounds": mr,
                        "early_stopping_patience": patience,
                        "early_stopping_min_delta": FILLER_EARLY_STOPPING_MIN_DELTA,
                        "early_stopping_window": window,
                        "early_stopping_drop_tolerance": FILLER_EARLY_STOPPING_DROP_TOLERANCE,
                        "_tier_idx": 3,
                        "_tier_label": f"TIER 3 ({ds.upper()})",
                    })
                    extra_ro = (
                        f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                        f"-M-rolora-E10-LR0.1-RK{LORA_RANK}-BSTART-S{s}"
                        f"-ES{gate:g}-MR{mr}-P{patience}-D{FILLER_EARLY_STOPPING_MIN_DELTA:g}-W{window}"
                    )
                    cfg_ro["extras"] = extra_ro
                    configs.append(cfg_ro)
                    for r in [0.5, 1.0]:
                        cfg_fda = base.make_config(
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
                        cfg_fda["METHOD_NAME"] = PURE_METHOD
                        cfg_fda.update({
                            "server_opt_name": "SGD",
                            "server_opt_lr": 1.0,
                            "client_opt_name": "SGD",
                            "num_workers": 0,
                            "pin_memory": True,
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
                            "early_stopping_patience": patience,
                            "early_stopping_min_delta": FILLER_EARLY_STOPPING_MIN_DELTA,
                            "early_stopping_window": window,
                            "early_stopping_drop_tolerance": FILLER_EARLY_STOPPING_DROP_TOLERANCE,
                            "_tier_idx": 3,
                            "_tier_label": f"TIER 3 ({ds.upper()})",
                        })
                        r_tag = f"CRATIO{r:g}"
                        extra_fda = (
                            f"THESIS-NONIID-A0.5-DS-{ds}-C{c}-R{rounds}"
                            f"-M-fda-pure-LR0.1-RK{LORA_RANK}-E{DEFAULT_E_MIN}TO{DEFAULT_E_MAX}-{r_tag}-NODISC-BSTART-NOGUARD-S{s}"
                            f"-ES{gate:g}-MR{mr}-P{patience}-D{FILLER_EARLY_STOPPING_MIN_DELTA:g}-W{window}"
                        )
                        cfg_fda["extras"] = extra_fda
                        configs.append(cfg_fda)
    return sorted(configs, key=filler_config_priority)
def is_filler_completed(config: dict) -> bool:
    """Check if run is finished on disk. Also checks equivalent existing NONIID runs."""
    pattern = str(METRICS_DIR / f"{config['extras']}_*.csv")
    for path in glob.glob(pattern):
        try:
            import pandas as pd
            frame = pd.read_csv(path)
            if not frame.empty and "round" in frame.columns and len(frame) >= 5:
                return True
        except Exception:
            continue
    ds = config["ds_name"]
    c = config["num_clients"]
    s = config["seed"]
    method_slug = "rolora" if config.get("METHOD_NAME") == ROLORA_METHOD else "fda-pure"
    if method_slug == "rolora":
        ep = config.get("local_epochs", 10)
        equiv_pat = str(METRICS_DIR / f"*NONIID*DS-{ds}-C{c}-*-M-rolora-E{ep}-*S{s}-*.csv")
    else:
        emax = config.get("max_epochs", 20)
        r = config.get("controller_ratio", 1.0)
        equiv_pat = str(METRICS_DIR / f"*NONIID*DS-{ds}-C{c}-*-M-fda-pure-*-E1TO{emax}-CRATIO{r:g}-*S{s}-*.csv")
    for path in glob.glob(equiv_pat):
        try:
            import pandas as pd
            frame = pd.read_csv(path)
            if not frame.empty and "round" in frame.columns and len(frame) >= 5:
                return True
        except Exception:
            continue
    return False
def is_dropped(config: dict) -> bool:
    cid = base.get_control_id(config["extras"])
    return (DROPPED_RUNS_DIR / f"{cid}.json").exists()
def run_filler_config_on_device(config: dict, device: str) -> bool:
    """Spawns this runner script itself with --run-config for child worker isolation."""
    run_id = config["extras"]
    registry_path = None
    request_path = None
    try:
        if is_filler_completed(config):
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
        environment["NVIDIA_TF32_OVERRIDE"] = "1"
        environment["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
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
def wait_for_registered_filler_run(
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
    if is_filler_completed(config):
        final_metric = base.read_latest_metric_line(record["stdout_path"])
        print(f"ADOPTED DONE [{control_id}] {run_id}; {final_metric}")
        return True
    print(f"ADOPTED ERROR [{control_id}] {run_id}: worker exited without a valid metrics CSV")
    return False
def resolve_filler_optimizers():
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
    return fedavg_opt, pure_opt
def check_filler_early_stopping(hyperparams: dict, metrics_history: list[dict]) -> bool:
    """Dataset-calibrated early stopping with plateau detection and post-peak drop guard."""
    if not metrics_history:
        return False
    base.log_latest_round_metrics(hyperparams, metrics_history)
    values = [
        base.get_primary_accuracy(hyperparams["ds_name"], metrics)
        for metrics in metrics_history
    ]
    min_rounds = max(1, int(hyperparams.get("early_stopping_min_rounds", 1)))
    if len(values) < min_rounds:
        return False
    start_accuracy = hyperparams.get(
        "early_stopping_start_accuracy",
        hyperparams.get("plateau_min_accuracy", 0.90),
    )
    patience = max(1, int(hyperparams.get("early_stopping_patience", 3)))
    min_delta = float(hyperparams.get("early_stopping_min_delta", 0.005))
    window_size = max(1, int(hyperparams.get("early_stopping_window", 2)))
    drop_tolerance = float(hyperparams.get("early_stopping_drop_tolerance", 0.03))
    gate_index = next(
        (index for index, value in enumerate(values) if value >= start_accuracy),
        None,
    )
    if gate_index is None:
        return False
    def trailing_mean(index: int) -> float:
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
            print(
                f"[EARLY STOPPING] Degradation guard triggered at round {index + 1}: "
                f"smoothed accuracy {current_value:.4f} dropped >= {drop_tolerance:.4f} below peak {best_value:.4f}",
                flush=True,
            )
            return True
    if rounds_without_improvement >= patience:
        print(
            f"[EARLY STOPPING] Plateau triggered at round {len(values)}: "
            f"{rounds_without_improvement} rounds without {min_delta:.4f} gain (peak: {best_value:.4f})",
            flush=True,
        )
        return True
    return False
def optimized_filler_client_round_train(
    model: Any,
    train_params: list,
    client_train_params_dict: dict,
    client_opt: Any,
    fed_ds: Any,
    client_id: int,
    round_steps: int,
) -> tuple[float, float]:
    """Optimized client_round_train eliminating inner-loop CPU-GPU sync stalls (.item())
    and using asynchronous non-blocking DMA copies from pinned host RAM over PCIe.
    Yields identical numerical gradients while reducing per-step runtime by ~40%+.
    """
    import torch
    from fdaopt import utils
    from fdaopt.models.ops import copy_parameters
    model.train()
    copy_parameters(
        from_params=client_train_params_dict[client_id],
        to_params=train_params,
    )
    dev = utils.DEVICE if hasattr(utils, "DEVICE") and utils.DEVICE else (
        train_params[0].device if train_params else "cuda:0"
    )
    running_loss = torch.tensor(0.0, device=dev)
    correct_predictions = torch.tensor(0, dtype=torch.long, device=dev)
    total_predictions = 0
    for _ in range(round_steps):
        batch = fed_ds.next_client_batch(client_id)
        batch = {
            k: (v.to(dev, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == batch["labels"]).sum()
        total_predictions += batch["labels"].size(0)
        loss.backward()
        client_opt.step()
        client_opt.zero_grad()
        running_loss += loss.detach()
    copy_parameters(
        from_params=train_params,
        to_params=client_train_params_dict[client_id],
    )
    avg_loss = (running_loss.item() / round_steps) if round_steps > 0 else 0.0
    train_acc = (correct_predictions.item() / total_predictions) if total_predictions > 0 else 0.0
    return avg_loss, train_acc
def execute_run(config_json: str) -> None:
    """Child worker execution target."""
    import torch
    from fdaopt import utils
    from fdaopt.training import fda_opt, fed_opt
    fedavg_opt, pure_opt = resolve_filler_optimizers()
    if pure_opt is not None:
        fda_opt.fda_rolora_dual_track_pure_opt = pure_opt
        fed_opt.fda_rolora_dual_track_pure_opt = pure_opt
    if fedavg_opt is not None:
        fed_opt.rolora_fedavg_opt = fedavg_opt
    config = json.loads(config_json)
    device = config.pop("TARGET_DEVICE", "cuda:0")
    base.install_runtime_patches(config)
    if config.get("early_stopping"):
        fed_opt.check_dynamic_early_stopping = check_filler_early_stopping
        fda_opt.check_dynamic_early_stopping = check_filler_early_stopping
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    try:
        from fdaopt.training import fed_train
        fed_train.client_round_train = optimized_filler_client_round_train
    except Exception as exc:
        print(f"[WARN] Failed patching fed_train.client_round_train: {exc}", flush=True)
    try:
        import pure_experiments.fda_rolora_pure as pure_mod
        pure_mod.client_round_train = optimized_filler_client_round_train
    except Exception as exc:
        print(f"[WARN] Failed patching fda_rolora_pure.client_round_train: {exc}", flush=True)
    try:
        import non_iid_experiments.fda_rolora_non_iid as non_iid_mod
        non_iid_mod.client_round_train = optimized_filler_client_round_train
    except Exception as exc:
        print(f"[WARN] Failed patching fda_rolora_non_iid.client_round_train: {exc}", flush=True)
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
    completed = [cfg for cfg in configs if is_filler_completed(cfg)]
    pending = [cfg for cfg in configs if not is_filler_completed(cfg)]
    displayed = configs if show_all else pending
    sep = "=" * 130
    print(f"\n{sep}")
    if show_all:
        print(f"THESIS FILLER EXPERIMENT AUDIT (Total: {len(configs)} | Done: {len(completed)} | Pending: {len(pending)})")
    else:
        print(f"THESIS FILLER EXECUTION PLAN (Pending: {len(pending)} to run | {len(completed)} already completed omitted)")
    print(f"Dataset-Tuned Early Stopping: Specific Gates (0.50-0.85), Adaptive MR (8-20), P=2-3, W=2-3, Delta=0.5%, Drop-Guard=3%")
    print(f"Scheduler: {JOBS_PER_GPU} dynamic lanes/GPU across {len(base.DEVICES)} GPUs; SST-2 capped at {BATCH64_JOBS_PER_GPU}/GPU.")
    print(sep)
    header = f"{'#':<3} | {'Control ID':<10} | {'Tier':<22} | {'DS':<5} | {'BS':<3} | {'C':<3} | {'R':<3} | {'Method / Ratio':<20} | {'Gate':<5} | {'MR':<3} | {'P':<2} | {'Seed':<4} | {'Status':<10}"
    print(header)
    print("-" * len(header))
    current_tier = None
    for idx, cfg in enumerate(displayed, 1):
        tier_label = cfg.get("_tier_label", "FILLER")
        if current_tier is not None and tier_label != current_tier:
            print("-" * len(header))
        current_tier = tier_label
        cid = base.get_control_id(cfg["extras"])
        method = cfg.get("METHOD_NAME", "")
        if method == ROLORA_METHOD:
            m_str = f"RoLoRA (E{cfg.get('local_epochs', 10)})"
        else:
            emax = cfg.get('max_epochs', 20)
            k = cfg.get('controller_ratio', 1.0)
            m_str = f"FDA Pure E{emax} (k={k:g})"
        status_str = "DONE" if is_filler_completed(cfg) else "PENDING"
        gate_val = cfg.get('early_stopping_start_accuracy', 0.8)
        mr_val = cfg.get('early_stopping_min_rounds', 12)
        p_val = cfg.get('early_stopping_patience', 3)
        line = (
            f"{idx:<3} | {cid:<10} | {tier_label:<22} | {cfg['ds_name'].upper():<5} | {cfg['batch_size']:<3} | {cfg['num_clients']:<3} | "
            f"{cfg['total_rounds']:<3} | {m_str:<20} | {gate_val:<5.0%} | {mr_val:<3} | {p_val:<2} | {cfg['seed']:<4} | {status_str:<10}"
        )
        print(line)
    print(f"{sep}\n")
def run_filler_phase(configs: list[dict], job_runner=None) -> None:
    job_runner = run_filler_config_on_device if job_runner is None else job_runner
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
        if config["extras"] not in active_run_ids and not is_filler_completed(config)
    ]
    pending_queue = sorted(pending, key=filler_config_priority)
    print(
        f"Scheduled {len(pending)} queued and "
        f"{len(adopted_configs)} adopted configurations"
    )
    print(
        f"Skipped {len(configs) - len(pending) - len(adopted_configs)} "
        "completed configurations"
    )
    print(
        f"Filler scheduler: {JOBS_PER_GPU} dynamic lanes/GPU across {len(base.DEVICES)} GPUs; "
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
            future = executor.submit(wait_for_registered_filler_run, config, record)
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
                    "Another Filler scheduler is already active. "
                    "Use the status command instead of starting a second one."
                ) from error
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        print_plan(configs)
        print(f"Launching filler scheduler across {JOBS_PER_GPU} lanes/GPU (max concurrent: {len(base.DEVICES) * JOBS_PER_GPU})...")
        run_filler_phase(runnable)
        print("\n=== ALL THESIS FILLER EXPERIMENTS COMPLETED ===")
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
        print("No active filler experiments currently running.")
        return
    print(f"\nActive Filler Experiments ({len(active_records)}):")
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
    parser = argparse.ArgumentParser(description="Thesis Filler Experiments Orchestrator")
    parser.add_argument(
        "--group",
        choices=["plan", "run", "start", "status", "stop", "drop", "dropped"],
        default="plan",
        help="Action to execute (default: plan)",
    )
    parser.add_argument(
        "--tier",
        choices=["tier1", "tier2", "tier1-2", "tier3", "all"],
        default="all",
        help="Experiment tier to target (default: all)",
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
    configs = build_filler_configs(tier=args.tier)
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

```


## File: `fillers/profile_tf32.py`

```py
"""Benchmark and Profiling Script: Standard FP32 vs. TensorFloat-32 (TF32).
Measures the exact speedup and numerical precision impact of enabling
Tensor Cores (TF32) on NVIDIA Ampere (A10) and Ada Lovelace GPUs for
RoBERTa-base + LoRA workloads.
Usage:
    python fillers/profile_tf32.py
"""
import sys
import time
import torch
import torch.nn as nn
def get_gpu_info():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires an NVIDIA GPU.")
        sys.exit(1)
    prop = torch.cuda.get_device_properties(0)
    print("=" * 80)
    print(f"GPU Device:               {prop.name}")
    print(f"Compute Capability:       {prop.major}.{prop.minor}")
    print(f"Total VRAM:               {prop.total_memory / (1024**3):.2f} GB")
    print(f"PyTorch Version:          {torch.__version__}")
    has_tf32 = prop.major >= 8
    print(f"Ampere/Ada TF32 Support:  {'YES (Supported)' if has_tf32 else 'NO (Requires SM >= 8.0)'}")
    print("=" * 80)
    return has_tf32
class MockRobertaLayerWithLora(nn.Module):
    """Simulates a RoBERTa-base transformer layer with rank-4 LoRA adapters."""
    def __init__(self, hidden_size=768, intermediate_size=3072, lora_rank=4):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.lora_q_a = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_q_b = nn.Linear(lora_rank, hidden_size, bias=False)
        self.lora_v_a = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_v_b = nn.Linear(lora_rank, hidden_size, bias=False)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, x):
        q = self.q_proj(x) + self.lora_q_b(self.lora_q_a(x)) * 2.0
        k = self.k_proj(x)
        v = self.v_proj(x) + self.lora_v_b(self.lora_v_a(x)) * 2.0
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (768 ** 0.5))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        x = x + self.out_proj(attn_out)
        x = self.norm(x)
        x = x + self.fc2(self.act(self.fc1(x)))
        return self.norm(x)
def benchmark_workload(fn, name, warmup=25, iters=100):
    """Accurately measure GPU execution time using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / iters
    return avg_time_ms
def run_benchmark():
    has_tf32 = get_gpu_info()
    if not has_tf32:
        print("Note: TF32 requires Ampere (Compute 8.0+) or newer. Running anyway...")
    batch_size = 32
    seq_len = 128
    hidden_size = 768
    intermediate_size = 3072
    print("\n--- TEST 1: RoBERTa-base + LoRA Forward & Backward Pass ---")
    print(f"Batch Size: {batch_size} | Seq Len: {seq_len} | Hidden Size: {hidden_size} | LoRA Rank: 4")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    model = MockRobertaLayerWithLora(hidden_size, intermediate_size, lora_rank=4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.randn(batch_size, seq_len, hidden_size, device=device)
    targets = torch.randn(batch_size, seq_len, hidden_size, device=device)
    loss_fn = nn.MSELoss()
    def step():
        optimizer.zero_grad(set_to_none=True)
        out = model(inputs)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
        return out
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    fp32_time = benchmark_workload(step, "FP32 Baseline")
    fp32_output = step().detach().clone()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    tf32_time = benchmark_workload(step, "TF32 Accelerated")
    tf32_output = step().detach().clone()
    speedup_1 = fp32_time / tf32_time
    pct_reduction_1 = (1.0 - (tf32_time / fp32_time)) * 100
    diff = (fp32_output - tf32_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.cosine_similarity(fp32_output.flatten(), tf32_output.flatten(), dim=0).item()
    print(f"  Standard FP32:      {fp32_time:.3f} ms / step")
    print(f"  TF32 Tensor Cores:  {tf32_time:.3f} ms / step")
    print(f"  >> Speedup:         {speedup_1:.2f}x faster ({pct_reduction_1:.1f}% time saved per step)")
    print(f"  >> Cosine Sim:      {cos_sim:.8f} (1.00000000 = identical direction)")
    print(f"  >> Mean Diff:       {mean_diff:.2e} (well below typical gradient variance)")
    print("\n--- TEST 2: Dense Matrix Multiplication (Large GEMM) ---")
    tokens = batch_size * seq_len
    print(f"Matrix Dimension: [{tokens}, {hidden_size}] x [{hidden_size}, {intermediate_size}]")
    a = torch.randn(tokens, hidden_size, device=device)
    b = torch.randn(hidden_size, intermediate_size, device=device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    gemm_fp32 = benchmark_workload(lambda: torch.matmul(a, b), "GEMM FP32", warmup=50, iters=200)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    gemm_tf32 = benchmark_workload(lambda: torch.matmul(a, b), "GEMM TF32", warmup=50, iters=200)
    speedup_2 = gemm_fp32 / gemm_tf32
    pct_reduction_2 = (1.0 - (gemm_tf32 / gemm_fp32)) * 100
    print(f"  Standard FP32:      {gemm_fp32:.3f} ms")
    print(f"  TF32 Tensor Cores:  {gemm_tf32:.3f} ms")
    print(f"  >> Speedup:         {speedup_2:.2f}x faster ({pct_reduction_2:.1f}% time saved)")
    print("\n" + "=" * 80)
    print("PROFILING VERDICT:")
    if speedup_1 >= 1.25 or speedup_2 >= 1.4:
        print(f"  [CONFIRMED] TF32 delivers a substantial speedup ({speedup_1:.2f}x - {speedup_2:.2f}x).")
        print("  Enabling this setting on NVIDIA Ampere A10 GPUs is definitely worth it.")
    else:
        print(f"  Speedup measured at {speedup_1:.2f}x.")
    print("=" * 80)
if __name__ == "__main__":
    run_benchmark()

```


## File: `fillers/profile_optimizations.py`

```py
"""Comprehensive Benchmark Script for the Top 3 Performance Optimizations.
Profiles:
  1. Setting 1: TF32 vs. Strict FP32 (Tensor Cores GEMM acceleration)
  2. Setting 2: Per-batch CPU-GPU sync stalls (.item() every batch vs. single sync at end)
  3. Setting 3: Host-to-Device transfer (Unpinned memory vs. Pinned memory with non_blocking=True)
  4. Combined: Cumulative speedup with all 3 optimizations active vs. vanilla baseline
Usage:
    python fillers/profile_optimizations.py
"""
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
def get_gpu_info():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires an NVIDIA GPU.")
        sys.exit(1)
    prop = torch.cuda.get_device_properties(0)
    print("=" * 85)
    print(f"GPU Device:               {prop.name}")
    print(f"Compute Capability:       {prop.major}.{prop.minor}")
    print(f"Total VRAM:               {prop.total_memory / (1024**3):.2f} GB")
    print(f"PyTorch Version:          {torch.__version__}")
    print("=" * 85)
class MockRobertaLayer(nn.Module):
    """Simulates a RoBERTa-base transformer layer with rank-4 LoRA adapters."""
    def __init__(self, hidden_size=768, intermediate_size=3072, lora_rank=4):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.lora_q_a = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_q_b = nn.Linear(lora_rank, hidden_size, bias=False)
        self.lora_v_a = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_v_b = nn.Linear(lora_rank, hidden_size, bias=False)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)
    def forward(self, input_ids):
        q = self.q_proj(input_ids) + self.lora_q_b(self.lora_q_a(input_ids)) * 2.0
        k = self.k_proj(input_ids)
        v = self.v_proj(input_ids) + self.lora_v_b(self.lora_v_a(input_ids)) * 2.0
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (768 ** 0.5))
        attn_out = torch.matmul(torch.softmax(scores, dim=-1), v)
        x = self.norm(input_ids + self.out_proj(attn_out))
        x = self.norm(x + self.fc2(self.act(self.fc1(x))))
        logits = self.classifier(x[:, 0, :])
        return logits
def benchmark_setting_1(device):
    print("\n" + "=" * 85)
    print("TEST 1: Setting 1 - TensorFloat-32 (TF32) Tensor Cores vs. Standard FP32")
    print("=" * 85)
    batch_size = 32
    seq_len = 128
    hidden_size = 768
    model = MockRobertaLayer(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    inputs = torch.randn(batch_size, seq_len, hidden_size, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for _ in range(15): step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(80): step()
    torch.cuda.synchronize()
    fp32_time = (time.perf_counter() - t0) / 80 * 1000
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    for _ in range(15): step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(80): step()
    torch.cuda.synchronize()
    tf32_time = (time.perf_counter() - t0) / 80 * 1000
    speedup = fp32_time / tf32_time
    pct = (1.0 - (tf32_time / fp32_time)) * 100
    print(f"  Standard FP32:           {fp32_time:.3f} ms / step")
    print(f"  TF32 Tensor Cores:       {tf32_time:.3f} ms / step")
    print(f"  >> Setting 1 Speedup:    {speedup:.2f}x faster ({pct:.1f}% time saved per step)")
    return speedup
def benchmark_setting_2(device):
    print("\n" + "=" * 85)
    print("TEST 2: Setting 2 - Inner-Loop CPU-GPU Synchronizations (.item() Stalls)")
    print("=" * 85)
    print("Simulating a local client training run (50 mini-batches of forward + backward + opt steps):")
    batch_size = 32
    seq_len = 128
    hidden_size = 768
    num_steps = 50
    model = MockRobertaLayer(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    batches = [
        (torch.randn(batch_size, seq_len, hidden_size, device=device),
         torch.randint(0, 2, (batch_size,), device=device))
        for _ in range(num_steps)
    ]
    def train_with_sync():
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / num_steps, correct / total
    def train_without_sync():
        running_loss = torch.zeros((), device=device)
        running_correct = torch.zeros((), device=device, dtype=torch.long)
        total = 0
        for inputs, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            running_correct += (preds == labels).sum()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach()
        avg_loss = (running_loss / num_steps).item()
        acc = running_correct.item() / total
        return avg_loss, acc
    train_with_sync()
    train_without_sync()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        train_with_sync()
    torch.cuda.synchronize()
    sync_time = (time.perf_counter() - t0) / 10 * 1000
    t0 = time.perf_counter()
    for _ in range(10):
        train_without_sync()
    torch.cuda.synchronize()
    no_sync_time = (time.perf_counter() - t0) / 10 * 1000
    speedup = sync_time / no_sync_time
    pct = (1.0 - (no_sync_time / sync_time)) * 100
    print(f"  Baseline (2 syncs/batch):  {sync_time:.2f} ms per 50-step client")
    print(f"  Optimized (1 sync at end): {no_sync_time:.2f} ms per 50-step client")
    print(f"  >> Setting 2 Speedup:      {speedup:.2f}x faster ({pct:.1f}% time saved in loop)")
    return speedup
def benchmark_setting_3(device):
    print("\n" + "=" * 85, flush=True)
    print("TEST 3: Setting 3 - Pinned Host Memory & Asynchronous PCIe Transfers", flush=True)
    print("=" * 85, flush=True)
    print("Testing data pipeline throughput across 50 batches from host RAM to GPU:", flush=True)
    batch_size = 32
    seq_len = 128
    hidden_size = 768
    num_batches = 50
    x_cpu = torch.randn(num_batches * batch_size, seq_len, hidden_size)
    y_cpu = torch.randint(0, 2, (num_batches * batch_size,))
    ds = TensorDataset(x_cpu, y_cpu)
    dl_unpinned = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=False)
    dl_pinned = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    model = nn.Linear(hidden_size, hidden_size).to(device)
    def run_unpinned():
        for x, y in dl_unpinned:
            x_gpu = x.to(device, non_blocking=False)
            y_gpu = y.to(device, non_blocking=False)
            out = model(x_gpu)
            _ = out.sum()
    def run_pinned():
        for x, y in dl_pinned:
            x_gpu = x.to(device, non_blocking=True)
            y_gpu = y.to(device, non_blocking=True)
            out = model(x_gpu)
            _ = out.sum()
    run_unpinned()
    run_pinned()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        run_unpinned()
    torch.cuda.synchronize()
    unpinned_time = (time.perf_counter() - t0) / 5 * 1000
    t0 = time.perf_counter()
    for _ in range(5):
        run_pinned()
    torch.cuda.synchronize()
    pinned_time = (time.perf_counter() - t0) / 5 * 1000
    speedup = unpinned_time / pinned_time
    pct = (1.0 - (pinned_time / unpinned_time)) * 100
    print(f"  Unpinned (pin_memory=False): {unpinned_time:.2f} ms per 50 batches", flush=True)
    print(f"  Pinned (pin_memory=True):    {pinned_time:.2f} ms per 50 batches", flush=True)
    print(f"  >> Setting 3 Speedup:        {speedup:.2f}x faster ({pct:.1f}% time saved in transfer)", flush=True)
    return speedup
def benchmark_combined(device):
    print("\n" + "=" * 85, flush=True)
    print("TEST 4: Combined Cumulative Speedup (All 3 Settings vs. Vanilla Baseline)", flush=True)
    print("=" * 85, flush=True)
    print("Simulating full client round (data loading + forward/backward + metrics):", flush=True)
    batch_size = 32
    seq_len = 128
    hidden_size = 768
    num_batches = 30
    x_cpu = torch.randn(num_batches * batch_size, seq_len, hidden_size)
    y_cpu = torch.randint(0, 2, (num_batches * batch_size,))
    ds = TensorDataset(x_cpu, y_cpu)
    dl_unpinned = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=False)
    dl_pinned = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    model = MockRobertaLayer(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    def vanilla_pipeline():
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in dl_unpinned:
            x_gpu = x.to(device, non_blocking=False)
            y_gpu = y.to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_gpu)
            loss = loss_fn(logits, y_gpu)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_gpu).sum().item()
            total += y_gpu.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / num_batches, correct / total
    def optimized_pipeline():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        running_loss = torch.zeros((), device=device)
        running_correct = torch.zeros((), device=device, dtype=torch.long)
        total = 0
        for x, y in dl_pinned:
            x_gpu = x.to(device, non_blocking=True)
            y_gpu = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_gpu)
            loss = loss_fn(logits, y_gpu)
            preds = torch.argmax(logits, dim=-1)
            running_correct += (preds == y_gpu).sum()
            total += y_gpu.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach()
        avg_loss = (running_loss / num_batches).item()
        acc = running_correct.item() / total
        return avg_loss, acc
    vanilla_pipeline()
    optimized_pipeline()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(8):
        vanilla_pipeline()
    torch.cuda.synchronize()
    vanilla_time = (time.perf_counter() - t0) / 8 * 1000
    t0 = time.perf_counter()
    for _ in range(8):
        optimized_pipeline()
    torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - t0) / 8 * 1000
    combined_speedup = vanilla_time / optimized_time
    combined_pct = (1.0 - (optimized_time / vanilla_time)) * 100
    print(f"  Vanilla Pipeline:         {vanilla_time:.2f} ms per client round")
    print(f"  Fully Optimized Pipeline: {optimized_time:.2f} ms per client round")
    print(f"  >> COMBINED SPEEDUP:      {combined_speedup:.2f}x FASTER ({combined_pct:.1f}% time saved overall)")
    return combined_speedup
def main():
    get_gpu_info()
    device = torch.device("cuda:0")
    s1 = benchmark_setting_1(device)
    s2 = benchmark_setting_2(device)
    s3 = benchmark_setting_3(device)
    s_comb = benchmark_combined(device)
    print("\n" + "=" * 85)
    print("FINAL BENCHMARK SUMMARY:")
    print(f"  Setting 1 (TF32 Tensor Cores):               {s1:.2f}x speedup")
    print(f"  Setting 2 (Remove Sync Stalls):              {s2:.2f}x speedup")
    print(f"  Setting 3 (Pinned Memory & DMA Transfers):   {s3:.2f}x speedup")
    print(f"  >>> TOTAL COMBINED ACCELERATION:             {s_comb:.2f}x FASTER (All 3 Combined)")
    print("=" * 85)
if __name__ == "__main__":
    main()

```

