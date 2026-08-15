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

