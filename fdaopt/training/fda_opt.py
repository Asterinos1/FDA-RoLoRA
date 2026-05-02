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
    second moments (m_t, v_t) to prevent the Stale Momentum Trap upon phase toggling.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'exp_avg' in state:
                state['exp_avg'].mul_(gamma)
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].mul_(gamma)

def fda_rolora_opt(hyperparams):
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
        gc.collect()
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
    metrics_handler.save_metrics()


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