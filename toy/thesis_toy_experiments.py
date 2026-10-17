#!/usr/bin/env python3
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

# Resolve REPO_ROOT
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

# Directory specifications
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

# ==============================================================================
# Worker Execution Engine (Invoked per-process on assigned CUDA device)
# ==============================================================================

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

    # 1. Dataset loading & Partition
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams['num_workers'], hyperparams['pin_memory']
    )

    # 2. In-VRAM pre-caching with pre-flattened float tensors for optimal throughput
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
        # Pre-normalize on GPU
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

    # 3. Model & Optimizers
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

            # Evaluate client drift variance on active factor
            sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
            client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
            var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)

            # Pure stopping condition: breaks if variance exceeds threshold or hits e_max
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
        if active_matrix == 'A':
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

# ==============================================================================
# Multi-GPU Staged Scheduler (Polytechnix Dispatcher)
# ==============================================================================

def get_available_devices() -> list[str]:
    if not torch.cuda.is_available():
        return ["cpu"]
    count = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(count)]

def run_scheduler(configs: list[dict]):
    devices = get_available_devices()
    print(f"=== Starting Toy Model Scheduler across devices: {devices} (max {JOBS_PER_GPU} jobs/GPU) ===", flush=True)

    # Filter pending configurations
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
        # 1. Prune finished workers
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

        # 2. Dispatch pending runs to free slots
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

# ==============================================================================
# CLI Entrypoint
# ==============================================================================

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
