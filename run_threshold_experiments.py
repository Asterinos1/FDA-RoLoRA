#!/usr/bin/env python3
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
    Literature consensus eta = 0.01 (client SGD), server SGD (lr = 1.0)
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

# Resolve REPO_ROOT: supports running inside threshold_controller_experiments/ or repo root
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR if (CURRENT_DIR / "thesis_experiments.py").exists() else CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import thesis_experiments as te

# Configuration Defaults
CLIENT_LEARNING_RATE = 0.01
SERVER_LEARNING_RATE = 1.0
DATASETS = ("mnli", "qnli")
CLIENT_SETTINGS = {
    20: 50,  # C=20 -> R=50
    50: 30,  # C=50 -> R=30
}
DEFAULT_SEEDS = (42, 43, 44)
ALPHA = 1000.0  # IID partition

# Paired Epoch Tiers: Baseline E -> FDA (E_min, E_max)
EPOCH_TIERS = [
    {"baseline_epochs": 15, "fda_e_min": 3, "fda_e_max": 30},
    {"baseline_epochs": 20, "fda_e_min": 3, "fda_e_max": 40},
]

# Strict Early Stopping Parameters (restored standard thesis protocol)
STRICT_EARLY_STOPPING_PARAMS = {
    "early_stopping": True,
    "early_stopping_min_rounds": 6,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.005,  # 0.5%
    "early_stopping_window": 3,
    "early_stopping_gates": {
        "mnli": 0.60,
        "qnli": 0.80,
    },
}

P2_LAYERS = list(range(7, 12))  # Budget P2: top layers 7-11 for RoBERTa-base


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
                    # -------------------------------------------------------------
                    # STAGE 1: RoLoRA Baseline (Fixed epochs E=15, 20)
                    # -------------------------------------------------------------
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

                    # -------------------------------------------------------------
                    # STAGE 2: FDA-RoLoRA with kappa = 0.5 (E_max / 2)
                    # -------------------------------------------------------------
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

                    # -------------------------------------------------------------
                    # STAGE 3: FDA-RoLoRA with kappa = 1.0 (E_max / 1)
                    # -------------------------------------------------------------
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

    # Filter stages according to operator request
    if stage_filter == "rolora":
        return stage1_rolora
    elif stage_filter == "fda_k05":
        return stage2_fda_k05
    elif stage_filter == "fda_k1":
        return stage3_fda_k1
    elif stage_filter == "fda_all":
        return stage2_fda_k05 + stage3_fda_k1
    elif stage_filter == "all":
        # Strictly enforce: Stage 1 (RoLoRA) -> Stage 2 (FDA k=0.5) -> Stage 3 (FDA k=1 at very end)
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
        # Calibrated optimizer supports arbitrary controller_ratio (0.5 or 1.0)
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
        
        # Cross-platform temporary directory
        temp_base = "/tmp" if os.path.exists("/tmp") else tempfile.gettempdir()
        temporary_dir = tempfile.mkdtemp(prefix=f"{run_id}-", dir=temp_base)

        environment = os.environ.copy()
        environment["HF_METRICS_CACHE"] = os.path.join(temporary_dir, "metrics")
        environment["HF_DATASETS_CACHE"] = os.path.join(temporary_dir, "datasets")
        environment["TMPDIR"] = temporary_dir
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        environment["PYTHONPATH"] = f"{REPO_ROOT}:{environment.get('PYTHONPATH', '')}"

        # Target runner script is this file
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
        help=argparse.SUPPRESS,  # Internal worker dispatch argument
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Worker dispatch mode
    if args.run_config:
        execute_run(args.run_config)
        return

    # Operator control commands
    if args.status:
        te.list_active_runs()
        return

    if args.stop:
        te.stop_active_run(args.stop, reason="operator stopped run via threshold controller runner")
        return

    # GPU Device detection & setup
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
