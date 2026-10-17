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

    # 1. Strict round guard: never terminate before min_rounds
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

    # Moving average window
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
        # Matrix B on Round 1 (r=0), Matrix A on Round 2 (r=1)
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
        # Matrix B first on Round 1 (r=0), Matrix A on Round 2 (r=1)
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

            # Evaluate client drift variance on active low-rank factor
            sampled_active_client_dict = {
                c_id: active_client_dict[c_id] for c_id in sampled_clients
            }
            client_drifts = compute_client_drifts(
                round_start_active_params, sampled_active_client_dict
            )
            var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)

            # Pure stopping condition: breaks if variance exceeds threshold or hits e_max
            # (On round 1 for B and round 2 for A, current_theta is 0.0 -> breaks cleanly at epoch 1)
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

        # Update specific phase theta with proportional setpoint
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
