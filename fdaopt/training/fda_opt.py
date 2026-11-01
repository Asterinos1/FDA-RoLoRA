"""
Fast Dynamic Aggregation (FDA) algorithms for low-rank federated optimization.

Implements variance-adaptive local synchronization algorithms:
  - `fda_opt`: Standard full-model FDA with AMS sketching.
  - `fda_rolora_opt`: Variance-adaptive alternating RoLoRA with exact subspace variance.
  - `fda_rolora_opt2`: FDA-RoLoRA with linear 1D random projection heuristic.
  - `fda_rolora_dual_track_opt`: Dual-track FDA maintaining independent theta thresholds for matrices A and B.
  - `fda_rolora_dual_track_ema_standard_opt`: Dual-track FDA with standard Exponential Moving Average smoothing.
  - `fda_rolora_dual_track_ema_pure_opt`: Dual-track FDA with pure instantaneous-scaled EMA smoothing.
  - `fda_rolora_dual_track_sma_opt`: Dual-track FDA with Simple Moving Average window smoothing.
"""

import os
import gc
import time
import math
import random
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification

from fdaopt import utils
from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (
    compute_client_drifts,
    compute_drifts,
    compute_pseudo_gradients,
    get_ksi_unit,
    linear_variance_approx,
    linear_variance_approx2,
    set_gradients,
    copy_parameters,
    get_updated_client_parameters,
    variance,
    compute_metrics,
    fda_variance_approx,
    stop_training_with_threshold,
    check_dynamic_early_stopping
)
from fdaopt.training.fed_train import client_round_train
from fdaopt.training.sketch import AmsSketch
from fdaopt.training.optimizers import (
    get_opt_hyperparams,
    get_optimizer_class,
    server_client_optimizers
)


def get_save_device() -> str:
    """
    Determine target storage device for client parameter caches based on RAM profile.

    Returns:
        str: PyTorch device ('cuda:0' or 'cpu').
    """
    if utils.DEVICE_RAM_PROGRAM in ('performance', 'moderate'):
        return utils.DEVICE
    return 'cpu'


def fda_opt(hyperparams: Dict[str, Any]):
    """
    Execute full-model Fast Dynamic Aggregation (FDA) training loop.

    Adapts the local epoch count per round dynamically using AMS sketching
    to approximate client drift variance and compare against threshold theta.

    Args:
        hyperparams (dict): Experiment configuration dictionary.
    """
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
    ).to(utils.DEVICE)

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
        client_train_params_dict = get_updated_client_parameters(
            client_train_params_dict, sampled_clients, round_start_train_params
        )
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)

        var_approx = 0.0
        round_epochs = 0

        # Dynamic local epoch loop based on sketched variance bound
        while var_approx <= theta and round_epochs < hyperparams['max_epochs']:
            round_epochs += 1
            for client_id in sampled_clients:
                training_loss += client_round_train(
                    model, train_params, client_train_params_dict, client_opt, fed_ds, client_id, round_steps
                )[0]

            client_drifts = compute_client_drifts(round_start_train_params, client_train_params_dict)
            var_approx = fda_variance_approx(client_drifts, ams_sketch=ams_sketch)

            # First round calibration: execute single local epoch to measure initial variance
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

        # Update dynamic variance threshold for next round
        theta = ((hyperparams['max_epochs'] / 2) / round_epochs) * metrics['variance']
        gc.collect()

        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break

    metrics_handler.save_metrics()


def toggle_rolora_matrices(model: torch.nn.Module, active_matrix: str):
    """
    Dynamically freeze/unfreeze LoRA A and B factor matrices.

    Args:
        model (torch.nn.Module): PEFT model.
        active_matrix (str): 'A' or 'B'.
    """
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            param.requires_grad = (active_matrix == 'A')
        elif 'lora_B' in name:
            param.requires_grad = (active_matrix == 'B')


def fda_rolora_opt(hyperparams: Dict[str, Any]):
    """
    Execute FDA-RoLoRA with dynamic norm-scaled theta thresholds and alternating phases.

    Args:
        hyperparams (dict): Experiment configuration dictionary.
    """
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    alpha_theta = hyperparams.get('alpha_theta', 0.01)

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
    group_A = params_A + params_other
    group_B = params_B + params_other

    server_opt_A, _ = server_client_optimizers(group_A, hyperparams)
    server_opt_B, _ = server_client_optimizers(group_B, hyperparams)

    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [p.detach().clone().to(get_save_device()) for p in group_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [p.detach().clone().to(get_save_device()) for p in group_B] for c_id in range(hyperparams['num_clients'])}

    metrics_handler = MetricsHandler(hyperparams)

    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)

        active_params = group_A if active_matrix == 'A' else group_B
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
                sampled_active_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_dict)
                var_approx, _, _ = variance(client_drifts)
                if var_approx > theta_phase or round_epochs >= e_max:
                    break
            elif round_epochs >= e_max:
                sampled_active_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
                client_drifts = compute_client_drifts(round_start_active_params, sampled_active_dict)
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
            "theta_phase": theta_phase
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


def fda_rolora_opt2(hyperparams: Dict[str, Any]):
    """
    Execute FDA-RoLoRA with 1D randomized projection linear variance approximation.

    Args:
        hyperparams (dict): Experiment configuration dictionary.
    """
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    alpha_theta = hyperparams.get('alpha_theta', 0.01)

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
    client_dict_A = {c_id: [p.detach().clone().to(get_save_device()) for p in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [p.detach().clone().to(get_save_device()) for p in params_B] for c_id in range(hyperparams['num_clients'])}

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


def fda_rolora_dual_track_opt(hyperparams: Dict[str, Any]):
    """
    Execute Dual-Track FDA-RoLoRA maintaining separate variance bounds for matrices A and B.

    Tracks `theta_A` and `theta_B` independently to accommodate differing geometric curvature
    between input-projection matrix A and output-projection matrix B.

    Args:
        hyperparams (dict): Experiment configuration dictionary.
    """
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)

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

        # Update specific phase theta
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


def fda_rolora_dual_track_ema_standard_opt(hyperparams: Dict[str, Any]):
    """
    Execute Dual-Track FDA-RoLoRA with standard Exponential Moving Average smoothing on theta.

    Smooths the target variance bound via: theta_{t+1} = rho * theta_t + (1 - rho) * target_t.

    Args:
        hyperparams (dict): Experiment configuration dictionary.
    """
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

        # EMA target calculation
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


def fda_rolora_dual_track_ema_pure_opt(hyperparams: Dict[str, Any]):
    """
    Execute Dual-Track FDA-RoLoRA with pure instantaneous-scaled EMA smoothing.

    Uses `proposed_target = alpha_ema * var_subspace` before applying exponential decay.

    Args:
        hyperparams (dict): Experiment configuration dictionary.
    """
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


def fda_rolora_dual_track_sma_opt(hyperparams: Dict[str, Any]):
    """
    Execute Dual-Track FDA-RoLoRA with Simple Moving Average (SMA) window smoothing on theta.

    Args:
        hyperparams (dict): Experiment configuration dictionary.
    """
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
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
    client_dict_A = {c_id: [p.detach().clone().to(get_save_device()) for p in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [p.detach().clone().to(get_save_device()) for p in params_B] for c_id in range(hyperparams['num_clients'])}

    metrics_handler = MetricsHandler(hyperparams)
    theta_A = hyperparams.get('theta', 0.0)
    theta_B = hyperparams.get('theta', 0.0)
    history_A = [theta_A] if theta_A > 0.0 else []
    history_B = [theta_B] if theta_B > 0.0 else []
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

        # SMA target calculation with sliding window
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



def fda_rolora_dual_track_pure_opt(hyperparams: Dict[str, Any]) -> None:
    """
    Execute Dual-Track FDA-RoLoRA (Pure / Calibrated Closed-Loop Optimizer).

    Unified implementation supporting both pure variance and calibrated setpoints:
      - Starts with Matrix B on Round 1 by default (`start_matrix='B'`).
      - Adaptive controller ratio setpoint: `target_epochs = controller_ratio * e_max`.
      - Natural cold-start bootstrap from Theta=0.0 (or optional first_round_guard).
      - Dynamic bounds [e_min, e_max] with optional Matrix B variance discounting.
      - Full metric logging and early stopping integration.

    Args:
        hyperparams (dict): Experiment configuration dictionary.
    """
    e_min = int(hyperparams.get('e_min', 1))
    e_max = int(hyperparams.get('max_epochs', 20))
    controller_ratio = float(hyperparams.get('controller_ratio', 1.0))
    enable_b_discount = bool(hyperparams.get('enable_b_discount', False))
    b_discount_factor = float(hyperparams.get('b_discount_factor', 3.0))
    start_matrix = hyperparams.get('start_matrix', 'B')
    first_round_guard = bool(hyperparams.get('first_round_guard', False))
    method_name = hyperparams.get('METHOD_NAME', hyperparams.get('method', 'FDA-RoLoRA-DualTrack-Pure'))

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
    theta_A = float(hyperparams.get('theta', 0.0))
    theta_B = float(hyperparams.get('theta', 0.0))
    first_round_A = True
    first_round_B = True

    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        if start_matrix == 'B':
            active_matrix = 'B' if r % 2 == 0 else 'A'
        else:
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

                if first_round_guard and is_first_round:
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
            "method": method_name,
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
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)

        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round']
        metrics_handler.append_round_metrics(metrics)

        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished (Matrix {active_matrix}, Epochs: {round_epochs}, Ratio: {controller_ratio}).")
        gc.collect()

        if hyperparams.get('early_stopping') and check_dynamic_early_stopping(hyperparams, metrics_handler.round_metrics):
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at round {r + 1}.")
            break

    metrics_handler.save_metrics()


# Canonical alias for calibrated experiments
fda_rolora_dual_track_calibrated_opt = fda_rolora_dual_track_pure_opt
