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
        model = ToyModel(
            rank=hyperparams.get('lora_r', 16),
            linear=hyperparams.get('linear', False)
        ).to(utils.DEVICE)
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
        # Matrix B first on Round 1 (r=0), Matrix A on Round 2 (r=1)
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

            # Evaluate client drift variance on active low-rank factor
            sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
            client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
            var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)

            # Pure stopping condition: breaks if variance exceeds threshold or hits e_max
            # (On first round for B and A, current_theta is 0.0, so var_subspace > 0.0 naturally breaks at epoch 1)
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
