import gc
import torch
from tqdm import tqdm
# including LoRA imports
from peft import get_peft_model, LoraConfig, TaskType
from fdaopt.training.fed_train import sa_client_round_train

# from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.datasets.test_fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (compute_client_drifts, compute_drifts, compute_pseudo_gradients, set_gradients, copy_parameters,
    get_updated_client_parameters, variance, compute_metrics, fda_variance_approx, stop_training_with_threshold)
from fdaopt.training.fed_train import client_round_train
from fdaopt.training.sketch import AmsSketch
from fdaopt.training.optimizers import server_client_optimizers
from fdaopt.utils import DEVICE, AutoModelForSequenceClassification, np, random, DEVICE_RAM_PROGRAM

# Let RAM usage of one model be M, then,
SAVE_DEVICE = None
if DEVICE_RAM_PROGRAM == 'performance' or DEVICE_RAM_PROGRAM == 'moderate':
    # All client models are stored in DEVICE, which means DEVICE-RAM is O(num_clients * M)
    SAVE_DEVICE = DEVICE
elif DEVICE_RAM_PROGRAM == 'low':
    # All client models are stored in CPU, which means DEVICE-RAM is O(M)
    SAVE_DEVICE = 'cpu'


def fda_opt(hyperparams):

    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'],
        hyperparams['ds_name'],
        hyperparams['checkpoint'],
        hyperparams['num_clients'],
        hyperparams['alpha'],
        hyperparams['batch_size'],
        hyperparams.get('num_workers', 0),
        hyperparams.get('pin_memory', False)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'],
        num_labels=hyperparams['num_labels']
    )

    # Move the model to the `DEVICE`
    model = model.to(DEVICE)

    # Extract trainable parameters from the model, which reside on the device that the model resides in
    train_params = [param for param in model.parameters() if param.requires_grad]

    # Create a copy of the trainable parameters in `DEVICE`, detached from the computation graph
    round_start_train_params = [param.detach().clone() for param in train_params]

    server_opt, client_opt = server_client_optimizers(train_params, hyperparams)

    client_sampler = ClientSampler(
        list(range(hyperparams['num_clients'])),
        hyperparams['clients_per_round']
    )

    # A dictionary where keys are client IDs and values are lists of parameter tensors all lying on the SAVE_DEVICE
    client_train_params_dict = {
        client_id: [param.detach().clone().to(SAVE_DEVICE) for param in round_start_train_params]
        # for client_id in range(hyperparams['clients_per_round'])
        # small error here
        for client_id in range(hyperparams['num_clients'])
    }

    # Initialize the AMS sketch
    ams_sketch = AmsSketch()

    metrics_handler = MetricsHandler(hyperparams)

    theta = hyperparams['theta']
    first_round = True

    pbar_rounds = tqdm(range(hyperparams['total_rounds']), desc="FDA Federated Training")
    for r in pbar_rounds:

        training_loss = 0.0

        # Save the model parameters at the start of this round
        sampled_clients = client_sampler.sample()

        # Save the model parameters at the start of this round
        copy_parameters(
            from_params=train_params,
            to_params=round_start_train_params
        )

        client_train_params_dict = get_updated_client_parameters(client_train_params_dict, sampled_clients, round_start_train_params)

        # Calculate the total number of steps for this epoch/round
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        var_approx = 0.0
        round_epochs = 0

        #while var_approx <= hyperparams['theta'] and not random_synchronize(round_epochs):
        while var_approx <= theta and round_epochs < hyperparams['max_epochs']:

            round_epochs += 1
            
            pbar_clients = tqdm(sampled_clients, desc=f"  Round {r+1} | Epoch {round_epochs}", leave=False)
            for client_id in pbar_clients:
                training_loss += client_round_train(model, train_params, client_train_params_dict, client_opt, fed_ds,
                                                    client_id, round_steps)
                pbar_clients.set_postfix({"client_id": client_id})

            # Compute the drifts (differences) between the round start parameters and the client parameters
            client_drifts = compute_client_drifts(round_start_train_params, client_train_params_dict)
            # Calculate the current variance approximation with LinearFDA
            var_approx = fda_variance_approx(client_drifts, ams_sketch=ams_sketch)

            pbar_rounds.set_postfix({"var": f"{var_approx:.4e}", "theta": f"{theta:.4e}"})

            if first_round:
                first_round = False
                break

        # Calculate the accumulate training loss for the round for all clients (which might contain multiple epochs)
        training_loss = training_loss / round_epochs

        # Reset the model parameters to the parameters at the start of the round. This ensures that the
        # server-side optimizer updates are applied correctly (on the parameters at the start of the round)
        copy_parameters(
            from_params=round_start_train_params,
            to_params=train_params
        )

        # Compute pseudo-gradients based on the average drifts
        pseudo_gradients = compute_pseudo_gradients(client_drifts)

        # Set the computed pseudo-gradients to the trainable parameters
        set_gradients(train_params, pseudo_gradients)

        # Update model parameters (global model) using the server optimizer based on pseudo-gradients
        server_opt.step()

        # Zero the gradients before the next backward pass
        server_opt.zero_grad()

        # Calculate evaluation metrics on the test set
        metrics = {"round": r + 1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        # Calculate the average training loss for the round
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        # Calculate variance and helpful metrics
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        # Add epoch steps to metrics
        metrics['round_steps'] = round_epochs * round_steps
        # Pass round metrics to handler
        metrics_handler.append_round_metrics(metrics)

        theta = ((hyperparams['max_epochs'] / 2) / round_epochs) * metrics['variance']

        gc.collect()

        if hyperparams['threshold']:
            if stop_training_with_threshold(hyperparams, metrics): break

    metrics_handler.save_metrics()

# simple LoRA version of FDA-Opt
def fda_lora_opt(hyperparams):
    # --- Hardcoded logic for AmsSketch usage ---
    USE_AMSSKETCH = True 
    # -------------------------------------------

    # 1. Setup Federated Datasets
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'],
        hyperparams['ds_name'],
        hyperparams['checkpoint'],
        hyperparams['num_clients'],
        hyperparams['alpha'],
        hyperparams['batch_size'],
        hyperparams.get('num_workers', 0),
        hyperparams.get('pin_memory', False)
    )

    # 2. Model & LoRA Initialization
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'],
        num_labels=hyperparams['num_labels']
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=hyperparams.get('lora_r', 8),
        lora_alpha=hyperparams.get('lora_alpha', 16),
        lora_dropout=0.1,
        target_modules=hyperparams.get('target_modules', None)
    )

    model = get_peft_model(base_model, peft_config)
    model = model.to(DEVICE)

    # Extract only trainable LoRA parameters (A and B matrices)
    train_params = [param for param in model.parameters() if param.requires_grad]
    round_start_train_params = [param.detach().clone() for param in train_params]

    server_opt, client_opt = server_client_optimizers(train_params, hyperparams)

    client_sampler = ClientSampler(
        list(range(hyperparams['num_clients'])),
        hyperparams['clients_per_round']
    )

    # Dictionary for client-specific LoRA parameters
    client_train_params_dict = {
        client_id: [param.detach().clone().to(SAVE_DEVICE) for param in round_start_train_params]
        for client_id in range(hyperparams['num_clients'])
    }

    ams_sketch = AmsSketch() if USE_AMSSKETCH else None
    metrics_handler = MetricsHandler(hyperparams)

    theta = hyperparams['theta']
    first_round = True

    # 3. FDA Training Loop
    pbar_rounds = tqdm(range(hyperparams['total_rounds']), desc="Federated LoRA (FDA) Rounds")
    for r in pbar_rounds:
        training_loss = 0.0
        sampled_clients = client_sampler.sample()

        copy_parameters(from_params=train_params, to_params=round_start_train_params)
        client_train_params_dict = get_updated_client_parameters(
            client_train_params_dict, sampled_clients, round_start_train_params
        )

        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        var_approx = 0.0
        round_epochs = 0

        # Dynamic Synchronization Logic
        while var_approx <= theta and round_epochs < hyperparams['max_epochs']:
            round_epochs += 1
            
            pbar_clients = tqdm(sampled_clients, desc=f"  Round {r+1} | Epoch {round_epochs}", leave=False)
            for client_id in pbar_clients:
                training_loss += client_round_train(
                    model, train_params, client_train_params_dict, 
                    client_opt, fed_ds, client_id, round_steps
                )
                pbar_clients.set_postfix({"client_id": client_id})

            client_drifts = compute_client_drifts(round_start_train_params, client_train_params_dict)

            # Variance Approximation Selection
            if USE_AMSSKETCH:
                # Use sketching for efficient variance estimation
                var_approx = fda_variance_approx(client_drifts, ams_sketch=ams_sketch)
            else:
                # Use exact variance (computationally expensive for large LoRA ranks)
                var_approx, _, _ = variance(client_drifts)

            pbar_rounds.set_postfix({"var": f"{var_approx:.4e}", "theta": f"{theta:.4e}", "epochs": round_epochs})

            if first_round:
                first_round = False
                break

        # Server-side Update logic
        training_loss = training_loss / round_epochs
        copy_parameters(from_params=round_start_train_params, to_params=train_params)
        
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        set_gradients(train_params, pseudo_gradients)
        
        server_opt.step()
        server_opt.zero_grad()

        # Metrics Logging
        metrics = {"round": r + 1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics['round_steps'] = round_epochs * round_steps
        metrics_handler.append_round_metrics(metrics)

        # Dynamic adjustment of the synchronization threshold theta
        theta = ((hyperparams['max_epochs'] / 2) / round_epochs) * metrics['variance']

        gc.collect()

        if hyperparams.get('threshold'):
            if stop_training_with_threshold(hyperparams, metrics): 
                break

    metrics_handler.save_metrics()

def fda_sa_lora_opt(hyperparams):
    USE_AMSSKETCH = True 
    # toggle between including only the shared A parameters in the variance approximation or both A and B parameters.
    INCLUDE_B_IN_VARIANCE = hyperparams.get('include_b_in_variance', False)

    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], 
        hyperparams['ds_name'], 
        hyperparams['checkpoint'],
        hyperparams['num_clients'], 
        hyperparams['alpha'], 
        hyperparams['batch_size'],
        hyperparams.get('num_workers', 0), 
        hyperparams.get('pin_memory', False)
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'], 
        num_labels=hyperparams['num_labels']
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=hyperparams.get('lora_r', 8), 
        lora_alpha=hyperparams.get('lora_alpha', 16),
        lora_dropout=0.1, 
        target_modules=hyperparams.get('target_modules', None)
    )

    model = get_peft_model(base_model, peft_config)
    model = model.to(DEVICE)

    shared_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_B' not in n]
    local_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_B' in n]
    all_train_params = [p for p in model.parameters() if p.requires_grad]

    round_start_shared_params = [param.detach().clone() for param in shared_params]

    server_opt, _ = server_client_optimizers(shared_params, hyperparams)
    _, client_opt = server_client_optimizers(all_train_params, hyperparams)

    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])

    client_shared_params_dict = {
        client_id: [param.detach().clone().to(SAVE_DEVICE) for param in round_start_shared_params]
        for client_id in range(hyperparams['num_clients'])
    }
    
    client_local_params_dict = {
        client_id: [param.detach().clone().to(SAVE_DEVICE) for param in local_params]
        for client_id in range(hyperparams['num_clients'])
    }

    ams_sketch = AmsSketch() if USE_AMSSKETCH else None
    metrics_handler = MetricsHandler(hyperparams)

    theta = hyperparams['theta']
    first_round = True

    pbar_rounds = tqdm(range(hyperparams['total_rounds']), desc="FDA-SA-LoRA Rounds")
    for r in pbar_rounds:
        training_loss = 0.0
        sampled_clients = client_sampler.sample()

        copy_parameters(from_params=shared_params, to_params=round_start_shared_params)
        
        client_shared_params_dict = get_updated_client_parameters(
            client_shared_params_dict, sampled_clients, round_start_shared_params
        )

        if INCLUDE_B_IN_VARIANCE:
            # Snapshot the B parameters before the dynamic averaging epochs start
            round_start_local_params_dict = {
                client_id: [param.detach().clone().to(SAVE_DEVICE) for param in client_local_params_dict[client_id]]
                for client_id in sampled_clients
            }

        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        var_approx = 0.0
        round_epochs = 0

        while var_approx <= theta and round_epochs < hyperparams['max_epochs']:
            round_epochs += 1
            
            pbar_clients = tqdm(sampled_clients, desc=f"  Round {r+1} | Epoch {round_epochs}", leave=False)
            for client_id in pbar_clients:
                training_loss += sa_client_round_train(
                    model, shared_params, local_params, 
                    client_shared_params_dict, client_local_params_dict, 
                    client_opt, fed_ds, client_id, round_steps
                )
                
            # Compute the drift strictly for the SHARED A parameters
            client_drifts = compute_client_drifts(round_start_shared_params, client_shared_params_dict)

            # Route standard or combined drifts to the variance computation
            variance_drifts = client_drifts
            if INCLUDE_B_IN_VARIANCE:
                variance_drifts = {}
                for client_id in client_drifts.keys():
                    variance_drifts[client_id] = list(client_drifts[client_id])
                    
                    if client_id in sampled_clients:
                        drift_b = compute_drifts(
                            round_start_local_params_dict[client_id], 
                            client_local_params_dict[client_id]
                        )
                    else:
                        # Unsampled clients have 0 drift for B over the round
                        drift_b = [torch.zeros_like(p).to(SAVE_DEVICE) for p in client_local_params_dict[client_id]]
                    
                    variance_drifts[client_id].extend(drift_b)

            if USE_AMSSKETCH:
                var_approx = fda_variance_approx(variance_drifts, ams_sketch=ams_sketch)
            else:
                var_approx, _, _ = variance(variance_drifts)

            pbar_rounds.set_postfix({"var": f"{var_approx:.4e}", "theta": f"{theta:.4e}", "epochs": round_epochs})

            if first_round:
                first_round = False
                break

        training_loss = training_loss / round_epochs
        copy_parameters(from_params=round_start_shared_params, to_params=shared_params)
        
        # Pseudo-gradients apply ONLY to the shared A parameters 
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        set_gradients(shared_params, pseudo_gradients)
        
        server_opt.step()
        server_opt.zero_grad()

        metrics = {"round": r + 1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        
        if INCLUDE_B_IN_VARIANCE:
            metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(variance_drifts)
        else:
            metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
            
        metrics['round_steps'] = round_epochs * round_steps
        metrics_handler.append_round_metrics(metrics)

        theta = ((hyperparams['max_epochs'] / 2) / round_epochs) * metrics['variance']

        gc.collect()

        if hyperparams.get('threshold'):
            if stop_training_with_threshold(hyperparams, metrics): 
                break

    metrics_handler.save_metrics()

#rolora helper function to toggle A and B matrices
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

#rolora version of FDA-Opt
def fda_rolora_opt(hyperparams):
    USE_AMSSKETCH = True 
    MAX_CONSECUTIVE_ROUNDS = hyperparams.get('max_consecutive_rounds', 2)

    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 0), hyperparams.get('pin_memory', False)
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, 
        r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
        lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
    )

    model = get_peft_model(base_model, peft_config)
    model = model.to(DEVICE)

    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]

    group_A = params_A + params_other
    group_B = params_B + params_other
    
    len_A = len(params_A)
    len_B = len(params_B)

    server_opt_A, _ = server_client_optimizers(group_A, hyperparams)
    server_opt_B, _ = server_client_optimizers(group_B, hyperparams)

    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])

    client_dict_A = {
        client_id: [param.detach().clone().to(SAVE_DEVICE) for param in group_A]
        for client_id in range(hyperparams['num_clients'])
    }
    client_dict_B = {
        client_id: [param.detach().clone().to(SAVE_DEVICE) for param in group_B]
        for client_id in range(hyperparams['num_clients'])
    }

    ams_sketch = AmsSketch() if USE_AMSSKETCH else None
    metrics_handler = MetricsHandler(hyperparams)

    theta = hyperparams['theta']
    first_round = True
    
    var_history = {'A': float('inf'), 'B': float('inf')} 
    consecutive_counts = {'A': 0, 'B': 0}

    pbar_rounds = tqdm(range(hyperparams['total_rounds']), desc="FDA-RoLoRA Alt2 Rounds")
    for r in pbar_rounds:
        
        if (var_history['A'] >= var_history['B'] and consecutive_counts['A'] < MAX_CONSECUTIVE_ROUNDS) or consecutive_counts['B'] >= MAX_CONSECUTIVE_ROUNDS:
            active_matrix = 'A'
            consecutive_counts['A'] += 1
            consecutive_counts['B'] = 0
        else:
            active_matrix = 'B'
            consecutive_counts['B'] += 1
            consecutive_counts['A'] = 0

        toggle_rolora_matrices(model, active_matrix)

        active_params = group_A if active_matrix == 'A' else group_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        active_lora_len = len_A if active_matrix == 'A' else len_B

        _, client_opt = server_client_optimizers(active_params, hyperparams)
        round_start_active_params = [param.detach().clone() for param in active_params]

        training_loss = 0.0
        sampled_clients = client_sampler.sample()

        copy_parameters(from_params=active_params, to_params=round_start_active_params)
        
        active_client_dict = get_updated_client_parameters(
            active_client_dict, sampled_clients, round_start_active_params
        )

        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        var_approx = 0.0
        round_epochs = 0

        while var_approx <= theta and round_epochs < hyperparams['max_epochs']:
            round_epochs += 1
            
            pbar_clients = tqdm(sampled_clients, desc=f"  Round {r+1} | Epoch {round_epochs} | {active_matrix} Active", leave=False)
            for client_id in pbar_clients:
                training_loss += client_round_train(
                    model, active_params, active_client_dict, 
                    client_opt, fed_ds, client_id, round_steps
                )

            # Compute full drifts for aggregation
            client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
            
            # Filter out params_other to evaluate variance ONLY on LoRA pathways
            lora_only_drifts = {
                client_id: drifts[:active_lora_len] 
                for client_id, drifts in client_drifts.items()
            }

            if USE_AMSSKETCH:
                var_approx = fda_variance_approx(lora_only_drifts, ams_sketch=ams_sketch)
            else:
                var_approx, _, _ = variance(lora_only_drifts)

            if first_round:
                first_round = False
                break

        var_history[active_matrix] = var_approx

        training_loss = training_loss / round_epochs
        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        
        # Apply pseudo-gradients to ALL active parameters (LoRA + Head)
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        set_gradients(active_params, pseudo_gradients)
        
        active_server_opt.step()
        active_server_opt.zero_grad()

        metrics = {"round": r + 1, "active_matrix": active_matrix} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        
        # Log variance using the filtered drifts
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(lora_only_drifts)
        
        metrics['round_steps'] = round_epochs * round_steps
        metrics_handler.append_round_metrics(metrics)

        theta = ((hyperparams['max_epochs'] / 2) / round_epochs) * metrics['variance']
        gc.collect()

        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics): 
            break

    metrics_handler.save_metrics()