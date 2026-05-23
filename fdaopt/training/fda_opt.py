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
    get_updated_client_parameters, variance, compute_metrics, fda_variance_approx, stop_training_with_threshold)
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
    
    # Reconstruct the exact order of active_params: [params_A (or B)] + [params_other]
    active_names_lora = [n for n, p in model.named_parameters() if f'lora_{active_matrix}' in n and p.requires_grad]
    active_names_other = [n for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    ordered_active_names = active_names_lora + active_names_other
    
    model_named_params = dict(model.named_parameters())
    
    for active_param_idx, name in enumerate(ordered_active_names):
        param = model_named_params[name]
        is_lora = 'lora_' in name
        
        if is_lora:
            # Locate the corresponding frozen matrix to project the drift
            frozen_matrix_name = name.replace(f'lora_{active_matrix}', 'lora_B' if active_matrix == 'A' else 'lora_A')
            frozen_param = model_named_params[frozen_matrix_name].to(utils.DEVICE)
            
            is_transposed = param.shape[0] != lora_r if active_matrix == 'A' else frozen_param.shape[0] != lora_r
            
            layer_drift_sum = None
            
            for client_id in sampled_clients:
                delta_active = client_drifts[client_id][active_param_idx].to(utils.DEVICE)
                
                # Project drift into the effective weight space W = AB or W = BA
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
            # Standard calculation for classification head and non-LoRA parameters
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
    
    # Active parameter mapping ensures LoRA params are indexed before classification head
    params_A = params_A + params_other
    params_B = params_B + params_other
    
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams)
    
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])}
    
    metrics_handler = MetricsHandler(hyperparams)
    
    # Global dynamic threshold initialization
    theta = hyperparams.get('theta', 0.0)
    first_round = True
    
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        
        # Alternating Optimization Phase Switch
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        
        # Mitigate stale momentum bleed between alternating phases
        decay_optimizer_state(active_server_opt, gamma)
        
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
                
                # Check variance scaling inside the effective AB weight space
                var_exact, update_l2_norm, avg_norm_sq_drifts, norm_sq_avg_drift = compute_effective_variance_and_norm(
                    model, client_drifts, active_matrix, lora_r
                )
                
                if first_round:
                    first_round = False
                    break # Force full epoch cycle on round 1 to establish baseline theta
                    
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
        
        # Apply server update using raw parameter pseudo-gradients
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        set_gradients(active_params, pseudo_gradients)
        active_server_opt.step()
        active_server_opt.zero_grad()
        
        agg_end = time.time()
        round_end_time = time.time()
        
        # Dynamically scale the global variance threshold for the following round
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

@torch.no_grad()
def decay_optimizer_state(optimizer, gamma):
    """
    Episodic Optimizer State Management: Applies a decay scalar to the first and
    second moments to prevent the Stale Momentum Trap upon phase toggling.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'exp_avg' in state:
                state['exp_avg'].mul_(gamma)
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].mul_(gamma)
            if 'momentum_buffer' in state:         # <-- Add SGD Support
                state['momentum_buffer'].mul_(gamma)

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
    
    # BUG FIX: Include non-LoRA parameters (like the classification head) in the alternating groups
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
        decay_optimizer_state(active_server_opt, gamma)
        active_norm_sq = sum(torch.sum(p.to(utils.DEVICE) ** 2).item() for p in active_params)
        theta_phase = alpha_theta * active_norm_sq
        _, client_opt = server_client_optimizers(active_params, hyperparams)
        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        
        training_loss = 0.0
        training_accuracy = 0.0 # NEW: Initialize accumulator
        var_approx = 0.0
        round_epochs = 0
        train_start = time.time()
        
        while True:
            round_epochs += 1
            for client_id in sampled_clients:
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps) # NEW: Unpack both
                training_loss += loss
                training_accuracy += acc # NEW: Accumulate accuracy
                
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
        training_accuracy = training_accuracy / round_epochs # NEW: Average accuracy across dynamic epochs
        
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
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round'] # NEW: Log accuracy
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished.")
        gc.collect()
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
    metrics_handler.save_metrics()


def fda_rolora_opt3(hyperparams):
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    alpha_theta = hyperparams.get('alpha_theta', 0.01)
    gamma = hyperparams.get('gamma', 0.3)
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
    
    # BUG FIX: Include non-LoRA parameters (like the classification head) in the alternating groups
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    
    params_A = params_A + params_other
    params_B = params_B + params_other
    
    global_prev_A = {i: p.detach().clone() for i, p in enumerate(params_A)}
    global_prev_B = {i: p.detach().clone() for i, p in enumerate(params_B)}
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {
        c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])
    }
    client_dict_B = {
        c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])
    }
    metrics_handler = MetricsHandler(hyperparams)
    for r in range(hyperparams['total_rounds']):
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        decay_optimizer_state(active_server_opt, gamma)
        active_norm_sq = sum(torch.sum(p.to(utils.DEVICE) ** 2).item() for p in active_params)
        theta_phase = alpha_theta * active_norm_sq
        _, client_opt = server_client_optimizers(active_params, hyperparams)
        round_start_active_params = [param.detach().clone() for param in active_params]
        global_current_dict = {i: p for i, p in enumerate(round_start_active_params)}
        global_prev_dict = global_prev_A if active_matrix == 'A' else global_prev_B
        sampled_clients = client_sampler.sample()
        active_client_dict = get_updated_client_parameters(active_client_dict, sampled_clients, round_start_active_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        training_loss = 0.0
        var_approx = 0.0
        round_epochs = 0
        while True:
            round_epochs += 1
            for client_id in sampled_clients:
                training_loss += client_round_train(
                    model, active_params, active_client_dict,
                    client_opt, fed_ds, client_id, round_steps
                )
            if round_epochs >= e_min:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                client_drifts_dicts = [{i: tensor for i, tensor in enumerate(drift)} for drift in client_drifts.values()]
                var_approx = linear_variance_approx2(client_drifts_dicts, global_current_dict, global_prev_dict)
                if var_approx > theta_phase or round_epochs >= e_max:
                    break
            elif round_epochs >= e_max:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                break
        training_loss = training_loss / round_epochs
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        set_gradients(active_params, pseudo_gradients)
        active_server_opt.step()
        active_server_opt.zero_grad()
        if active_matrix == 'A':
            global_prev_A = global_current_dict
        else:
            global_prev_B = global_current_dict
        metrics = {"round": r + 1, "active_matrix": active_matrix} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics['round_steps'] = round_epochs * round_steps
        metrics['var_approx_linear'] = var_approx
        metrics['theta_phase'] = theta_phase
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r + 1}/{hyperparams['total_rounds']} finished.")
        gc.collect()
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
    metrics_handler.save_metrics()

def fda_rolora_dual_track_opt(hyperparams):
    """
    Executes FDA-RoLoRA with Dual-Track Thresholding.
    Calculates exact variance directly on the active low-rank subspace to maintain 
    computational efficiency, utilizing independent dynamic thresholds for matrices A and B.
    """
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    gamma = hyperparams.get('gamma', 0.3)

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

    # Separate parameters into independent active groups
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    params_A = params_A + params_other
    params_B = params_B + params_other

    # Instantiate separate server optimizers to avoid momentum cross-contamination
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams)

    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    
    # Independent local state tracking
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])}
    
    metrics_handler = MetricsHandler(hyperparams)

    # Dual-Track Threshold Initialization
    theta_A = hyperparams.get('theta', 0.0)
    theta_B = hyperparams.get('theta', 0.0)
    
    first_round = True

    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        
        # Determine current active matrix phase
        active_matrix = 'A' if r % 2 == 0 else 'B'
        toggle_rolora_matrices(model, active_matrix)
        
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        
        # Apply momentum decay to mitigate Stale Momentum Trap
        decay_optimizer_state(active_server_opt, gamma)
        
        # Retrieve the relevant threshold track for the active matrix
        current_theta = theta_A if active_matrix == 'A' else theta_B

        _, client_opt = server_client_optimizers(active_params, hyperparams)
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
        
        # Local training loop with subspace variance monitoring
        while True:
            round_epochs += 1
            for client_id in sampled_clients:
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
                
            if round_epochs >= e_min:
                # Compute drifts strictly within the active parameter space (O(r * d) complexity)
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                
                # Variance calculation utilizing the subspace drifts natively
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                
                if first_round:
                    first_round = False
                    break
                    
                if var_subspace > current_theta or round_epochs >= e_max:
                    break
                    
            elif round_epochs >= e_max:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                break
                
        train_end = time.time()
        agg_start = time.time()
        
        training_loss = training_loss / round_epochs
        training_accuracy = training_accuracy / round_epochs
        
        # Server aggregation
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        
        with torch.no_grad():
            update_l2_norm = math.sqrt(sum(torch.sum(g**2).item() for g in pseudo_gradients))
            
        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        set_gradients(active_params, pseudo_gradients)
        active_server_opt.step()
        active_server_opt.zero_grad()
        
        agg_end = time.time()
        round_end_time = time.time()
        
        # Dual-Track Threshold Evaluation: Assign computed threshold to the active matrix's specific tracker
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
        
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
            
    metrics_handler.save_metrics()

def fda_rolora_dual_track_ema_standard_opt(hyperparams):
    """
    Executes FDA-RoLoRA with Dual-Track Thresholding using Standard Exponential Moving Average (EMA).
    Applies a momentum parameter (rho) to smooth the target threshold generated by the inverse-epoch scaling.
    """
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    rho = hyperparams.get('rho', 0.8)
    
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
    
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])}
    
    metrics_handler = MetricsHandler(hyperparams)
    
    theta_A = hyperparams.get('theta', 0.0)
    theta_B = hyperparams.get('theta', 0.0)
    
    first_round = True
    
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        
        active_matrix = 'A' if r % 2 == 0 else 'B'
        toggle_rolora_matrices(model, active_matrix)
        
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        
        current_theta = theta_A if active_matrix == 'A' else theta_B
        
        _, client_opt = server_client_optimizers(active_params, hyperparams)
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
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            
            if round_epochs >= e_min:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                
                if first_round:
                    first_round = False
                    break
                
                if var_subspace > current_theta or round_epochs >= e_max:
                    break
            elif round_epochs >= e_max:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
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
        
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
            
    metrics_handler.save_metrics()

def fda_rolora_dual_track_ema_pure_opt(hyperparams):
    """
    Executes FDA-RoLoRA with Dual-Track Thresholding using Pure-EMA Geometric updating.
    Replaces epoch-scaling with a constant expansion coefficient (alpha_ema), 
    smoothed by EMA momentum (rho) to track geometric variance growth independently of epoch counts.
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
    
    client_dict_A = {c_id: [param.detach().clone().to(get_save_device()) for param in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [param.detach().clone().to(get_save_device()) for param in params_B] for c_id in range(hyperparams['num_clients'])}
    
    metrics_handler = MetricsHandler(hyperparams)
    
    theta_A = hyperparams.get('theta', 0.0)
    theta_B = hyperparams.get('theta', 0.0)
    
    first_round = True
    
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        
        active_matrix = 'A' if r % 2 == 0 else 'B'
        toggle_rolora_matrices(model, active_matrix)
        
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        
        current_theta = theta_A if active_matrix == 'A' else theta_B
        
        _, client_opt = server_client_optimizers(active_params, hyperparams)
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
                loss, acc = client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)
                training_loss += loss
                training_accuracy += acc
            
            if round_epochs >= e_min:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)
                
                if first_round:
                    first_round = False
                    break
                
                if var_subspace > current_theta or round_epochs >= e_max:
                    break
            elif round_epochs >= e_max:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
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
        
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
            
    metrics_handler.save_metrics()

def get_lora_pairs(model):
    """
    Extracts and pairs LoRA A and B matrices by their base module name.
    Returns a dictionary: { 'module_name': {'A': param_A, 'B': param_B} }
    """
    lora_pairs = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or 'lora_' not in name:
            continue
        base_name = name.replace('lora_A.', '').replace('lora_B.', '').replace('default.', '')
        if base_name not in lora_pairs:
            lora_pairs[base_name] = {}
        if 'lora_A' in name:
            lora_pairs[base_name]['A'] = param
        elif 'lora_B' in name:
            lora_pairs[base_name]['B'] = param
    return lora_pairs

def fda_lora_svd_opt(hyperparams):
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    alpha_theta = hyperparams.get('alpha_theta', 0.01)
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(hyperparams['checkpoint'], num_labels=hyperparams['num_labels'], dtype=torch.bfloat16)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False,
        r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
        lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
    )
    model = get_peft_model(base_model, peft_config).to(utils.DEVICE)
    train_params = [param for param in model.parameters() if param.requires_grad]
    lora_pairs = get_lora_pairs(model)
    dense_global_params = {}
    
    for name, matrices in lora_pairs.items():
        is_transposed = matrices['A'].shape[0] != hyperparams.get('lora_r', 8)
        if is_transposed:
            dense_w = torch.matmul(matrices['A'], matrices['B']).detach().clone().to(utils.DEVICE)
        else:
            dense_w = torch.matmul(matrices['B'], matrices['A']).detach().clone().to(utils.DEVICE)
        dense_w.requires_grad = True
        dense_global_params[name] = dense_w
        
    server_opt_class = get_optimizer_class(hyperparams['server_opt_name'])
    server_opt = server_opt_class(list(dense_global_params.values()), **get_opt_hyperparams('server', hyperparams))
    client_opt_class = get_optimizer_class(hyperparams['client_opt_name'])
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict = {c_id: [param.detach().clone().to(get_save_device()) for param in train_params] for c_id in range(hyperparams['num_clients'])}
    metrics_handler = MetricsHandler(hyperparams)
    
    for r in range(hyperparams['total_rounds']):
        round_start_time = time.time()
        active_norm_sq = sum(torch.sum(dense_w ** 2).item() for dense_w in dense_global_params.values())
        theta_phase = alpha_theta * active_norm_sq
        round_start_params = [param.detach().clone() for param in train_params]
        sampled_clients = client_sampler.sample()
        client_dict = get_updated_client_parameters(client_dict, sampled_clients, round_start_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        
        training_loss = 0.0
        training_accuracy = 0.0 # NEW: Initialize accumulator
        var_exact = 0.0
        round_epochs = 0
        client_opt = client_opt_class(train_params, **get_opt_hyperparams('client', hyperparams))
        train_start = time.time()
        
        while True:
            round_epochs += 1
            for client_id in sampled_clients:
                loss, acc = client_round_train(model, train_params, client_dict, client_opt, fed_ds, client_id, round_steps) # NEW: Unpack both
                training_loss += loss
                training_accuracy += acc # NEW: Accumulate accuracy
                
            if round_epochs >= e_min:
                mean_sq_norms = 0.0
                mean_drifts_sq_norm = 0.0
                with torch.no_grad():
                    for name, matrices in lora_pairs.items():
                        base_idx_A = next(i for i, p in enumerate(train_params) if p is matrices['A'])
                        base_idx_B = next(i for i, p in enumerate(train_params) if p is matrices['B'])
                        is_transposed = matrices['A'].shape[0] != hyperparams.get('lora_r', 8)
                        W_start = torch.matmul(round_start_params[base_idx_A].to(utils.DEVICE), round_start_params[base_idx_B].to(utils.DEVICE)) if is_transposed else torch.matmul(round_start_params[base_idx_B].to(utils.DEVICE), round_start_params[base_idx_A].to(utils.DEVICE))
                        layer_sq_norms_sum = 0.0
                        layer_drift_sum = torch.zeros_like(W_start)
                        for client_id in sampled_clients:
                            client_W = torch.matmul(client_dict[client_id][base_idx_A].to(utils.DEVICE), client_dict[client_id][base_idx_B].to(utils.DEVICE)) if is_transposed else torch.matmul(client_dict[client_id][base_idx_B].to(utils.DEVICE), client_dict[client_id][base_idx_A].to(utils.DEVICE))
                            drift = client_W - W_start
                            layer_sq_norms_sum += torch.sum(drift ** 2).item()
                            layer_drift_sum += drift
                        mean_sq_norms += layer_sq_norms_sum / len(sampled_clients)
                        mean_drifts_sq_norm += torch.sum((layer_drift_sum / len(sampled_clients)) ** 2).item()
                var_exact = mean_sq_norms - mean_drifts_sq_norm
                if var_exact > theta_phase or round_epochs >= e_max:
                    break
            elif round_epochs >= e_max:
                break
                
        train_end = time.time()
        agg_start = time.time()
        training_loss = training_loss / round_epochs
        training_accuracy = training_accuracy / round_epochs # NEW: Average accuracy across dynamic epochs
        
        server_opt.zero_grad()
        update_l2_norm = 0.0
        for name, matrices in lora_pairs.items():
            base_idx_A = next(i for i, p in enumerate(train_params) if p is matrices['A'])
            base_idx_B = next(i for i, p in enumerate(train_params) if p is matrices['B'])
            is_transposed = matrices['A'].shape[0] != hyperparams.get('lora_r', 8)
            W_start = torch.matmul(round_start_params[base_idx_A].to(utils.DEVICE), round_start_params[base_idx_B].to(utils.DEVICE)) if is_transposed else torch.matmul(round_start_params[base_idx_B].to(utils.DEVICE), round_start_params[base_idx_A].to(utils.DEVICE))
            
            # BUG FIX: Synchronize dense_global_params with truncated W_start before computing gradients
            dense_global_params[name].data.copy_(W_start)
            
            pseudo_grad_sum = torch.zeros_like(W_start)
            for client_id in sampled_clients:
                client_W = torch.matmul(client_dict[client_id][base_idx_A].to(utils.DEVICE), client_dict[client_id][base_idx_B].to(utils.DEVICE)) if is_transposed else torch.matmul(client_dict[client_id][base_idx_B].to(utils.DEVICE), client_dict[client_id][base_idx_A].to(utils.DEVICE))
                pseudo_grad_sum += (client_W - W_start)
            avg_drift = pseudo_grad_sum / len(sampled_clients)
            dense_global_params[name].grad = -avg_drift
            update_l2_norm += torch.sum(avg_drift**2).item()
            
        update_l2_norm = math.sqrt(update_l2_norm)
        server_opt.step()
        svd_effective_ranks = []
        
        with torch.no_grad():
            for name, matrices in lora_pairs.items():
                dense_w = dense_global_params[name].data
                U, S, Vh = torch.linalg.svd(dense_w.float(), full_matrices=False)
                
                is_transposed = matrices['A'].shape[0] != hyperparams.get('lora_r', 8)
                max_rank = matrices['A'].shape[1] if is_transposed else matrices['A'].shape[0]
                
                energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
                rank = torch.searchsorted(energy, 0.90).item() + 1
                rank = min(rank, max_rank)
                svd_effective_ranks.append(rank)
                
                U_r = U[:, :rank].to(dense_w.dtype)
                S_r = S[:rank].to(dense_w.dtype)
                Vh_r = Vh[:rank, :].to(dense_w.dtype)
                S_sqrt = torch.diag(torch.sqrt(S_r))
                
                left_matrix = U_r @ S_sqrt
                right_matrix = S_sqrt @ Vh_r
                
                if rank < max_rank:
                    pad_amount = max_rank - rank
                    # Pad left_matrix shape (M, rank) on the right dimension
                    left_matrix = torch.nn.functional.pad(left_matrix, (0, pad_amount, 0, 0))
                    # Pad right_matrix shape (rank, N) on the bottom dimension
                    right_matrix = torch.nn.functional.pad(right_matrix, (0, 0, 0, pad_amount))
                
                if is_transposed:
                    matrices['A'].copy_(left_matrix)
                    matrices['B'].copy_(right_matrix)
                else:
                    matrices['B'].copy_(left_matrix)
                    matrices['A'].copy_(right_matrix)
                    
        avg_svd_rank = sum(svd_effective_ranks) / len(svd_effective_ranks) if svd_effective_ranks else 0
        agg_end = time.time()
        round_end_time = time.time()
        metrics = {
            "method": "FDA-LoRA-SVD",
            "round": r + 1,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end_time - round_start_time,
            "update_l2_norm": update_l2_norm,
            "svd_effective_rank": avg_svd_rank,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * round_steps,
            "variance_exact": var_exact
        } | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['training_accuracy'] = training_accuracy / hyperparams['clients_per_round'] # NEW: Log accuracy
        metrics_handler.append_round_metrics(metrics)
        gc.collect()
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
    metrics_handler.save_metrics()