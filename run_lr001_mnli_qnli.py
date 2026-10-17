#!/usr/bin/env python3
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
    python run_lr001_mnli_qnli.py                    # Run all 72 configurations (cheapest first)
    python run_lr001_mnli_qnli.py --clients 20       # Run C=20 only (36 runs)
    python run_lr001_mnli_qnli.py --dataset mnli     # Run MNLI only (36 runs)
    python run_lr001_mnli_qnli.py --dataset mnli --clients 20 # Run pilot (18 runs)
    python run_lr001_mnli_qnli.py --dry-run          # Print planned queue without executing
    python run_lr001_mnli_qnli.py --gpus 0,1         # Specify GPUs
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path whether executed from root or lr001_experiments/
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR if (CURRENT_DIR / "thesis_experiments.py").exists() else CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import thesis_experiments as te

# Configuration Constants
CLIENT_LEARNING_RATE = 0.01
SERVER_LEARNING_RATE = 1.0
DATASETS = ("mnli", "qnli")
CLIENT_SETTINGS = {
    20: 50,  # C=20 -> R=50
    50: 30,  # C=50 -> R=30
}
SEEDS = (42, 43, 44)
ALPHA = 1000.0  # IID partition

# Paired Epoch Tiers: Baseline E -> FDA (E_min, E_max)
EPOCH_TIERS = [
    {"baseline_epochs": 5,  "fda_e_min": 3, "fda_e_max": 10},
    {"baseline_epochs": 10, "fda_e_min": 3, "fda_e_max": 20},
    {"baseline_epochs": 20, "fda_e_min": 3, "fda_e_max": 40},
]

# Lighter Early Stopping Parameters (avoid premature kills at small LR 0.01)
EARLY_STOPPING_PARAMS = {
    "early_stopping": True,
    "early_stopping_min_rounds": 10,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.002,  # 0.2%
    "early_stopping_window": 3,
    "early_stopping_gates": {
        "mnli": 0.60,
        "qnli": 0.80,
    },
}

P2_LAYERS = list(range(7, 12))  # Budget P2: top layers 7-11 for RoBERTa-base


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
                    # 1. RoLoRA Baseline Config (Exact server schema)
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

                    # 2. FDA-RoLoRA Config (Old Controller, Midpoint Target, Exact server schema)
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
                            "controller_type": "standard",  # Old midpoint controller
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

    # Operator control shortcuts: status and stop
    if args.status:
        te.list_active_runs()
        return

    if args.stop:
        te.stop_active_run(args.stop, reason="operator stopped run via lr001 runner")
        return

    # Configure GPU devices
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        devices = [f"cuda:{i}" for i in gpu_ids]
    else:
        # Auto-detect available CUDA devices
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
