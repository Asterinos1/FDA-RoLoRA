import gc
import torch
# including LoRA imports
from peft import get_peft_model, LoraConfig, TaskType

from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
# from fdaopt.datasets.test_fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (compute_client_drifts, compute_drifts, compute_pseudo_gradients, get_ksi_unit, linear_variance_approx, linear_variance_approx2, set_gradients, copy_parameters,
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

    for r in range(hyperparams['total_rounds']):

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
            
            for client_id in sampled_clients:
                training_loss += client_round_train(model, train_params, client_train_params_dict, client_opt, fed_ds,
                                                    client_id, round_steps)

            # Compute the drifts (differences) between the round start parameters and the client parameters
            client_drifts = compute_client_drifts(round_start_train_params, client_train_params_dict)
            # Calculate the current variance approximation with LinearFDA
            var_approx = fda_variance_approx(client_drifts, ams_sketch=ams_sketch)

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
    # Retrieve hyperparameters with safe fallbacks for the new architecture
    e_min = hyperparams.get('e_min', 2)               # Lazy Checking grace period
    e_max = hyperparams.get('max_epochs', 10)         # Hard timeout to prevent starvation
    alpha_theta = hyperparams.get('alpha_theta', 0.01)# Relative threshold coefficient
    gamma = hyperparams.get('gamma', 0.3)             # Optimizer momentum decay scalar
    
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
    model = get_peft_model(base_model, peft_config).to(DEVICE)
    
    # 1. Strict Isolation: Isolate ONLY LoRA parameters. params_other are ignored.
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n]
    
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams)
    
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    
    client_dict_A = {
        c_id: [param.detach().clone().to(SAVE_DEVICE) for param in params_A] for c_id in range(hyperparams['num_clients'])
    }
    client_dict_B = {
        c_id: [param.detach().clone().to(SAVE_DEVICE) for param in params_B] for c_id in range(hyperparams['num_clients'])
    }
    
    metrics_handler = MetricsHandler(hyperparams)
    
    for r in range(hyperparams['total_rounds']):
        # Alternating Dynamic Phases (Typically start with B since B is init to 0 and needs gradients driven by A)
        active_matrix = 'B' if r % 2 == 0 else 'A'
        
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        
        # 2. Episodic Intervention: Decay stale momentum to adapt to the newly shifted landscape
        decay_optimizer_state(active_server_opt, gamma)
        
        # 3. Asymmetric Starvation Fix: Calculate active matrix Frobenius norm squared
        active_norm_sq = sum(torch.sum(p.to(DEVICE) ** 2).item() for p in active_params)
        theta_phase = alpha_theta * active_norm_sq
        
        _, client_opt = server_client_optimizers(active_params, hyperparams)
        round_start_active_params = [param.detach().clone() for param in active_params]
        
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
            
            # 4. Lazy Variance Checking
            if round_epochs >= e_min:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                
                # 5. Distributed Stochasticity & The Sketching Paradox Fix (Linear Approx)
                # Ensure all clients generate the identical random vector via a synced seed
                shared_gen = torch.Generator(device=DEVICE).manual_seed(42 + r * 1000 + round_epochs)
                ksi_vector = get_ksi_unit(round_start_active_params, w_tminus1=None, generator=shared_gen)
                
                var_approx = linear_variance_approx(client_drifts, ksi_vector)
                
                # Trigger sync if variance exceeds relative threshold or max epochs are reached
                if var_approx > theta_phase or round_epochs >= e_max:
                    break
            
            elif round_epochs >= e_max:
                # Fallback if e_min >= e_max
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                break

        training_loss = training_loss / round_epochs
        
        # Apply pseudo-gradients to the active parameters
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        set_gradients(active_params, pseudo_gradients)
        
        active_server_opt.step()
        active_server_opt.zero_grad()
        
        # Log exact variance for metrics tracking
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

def fda_rolora_opt2(hyperparams):
    e_min = hyperparams.get('e_min', 2)
    e_max = hyperparams.get('max_epochs', 10)
    alpha_theta = hyperparams.get('alpha_theta', 0.01)
    gamma = hyperparams.get('gamma', 0.3)
    
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
    model = get_peft_model(base_model, peft_config).to(DEVICE)
    
    # 1. Strict Isolation: Isolate ONLY LoRA parameters.
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n]
    
    # Track the global historical states separately for 'A' and 'B'
    global_prev_A = {i: p.detach().clone() for i, p in enumerate(params_A)}
    global_prev_B = {i: p.detach().clone() for i, p in enumerate(params_B)}
    
    server_opt_A, _ = server_client_optimizers(params_A, hyperparams)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams)
    
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    
    client_dict_A = {
        c_id: [param.detach().clone().to(SAVE_DEVICE) for param in params_A] for c_id in range(hyperparams['num_clients'])
    }
    client_dict_B = {
        c_id: [param.detach().clone().to(SAVE_DEVICE) for param in params_B] for c_id in range(hyperparams['num_clients'])
    }
    
    metrics_handler = MetricsHandler(hyperparams)
    
    for r in range(hyperparams['total_rounds']):
        active_matrix = 'B' if r % 2 == 0 else 'A'
        
        toggle_rolora_matrices(model, active_matrix)
        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        
        # 2. Episodic Intervention
        decay_optimizer_state(active_server_opt, gamma)
        
        # 3. Asymmetric Starvation Fix
        active_norm_sq = sum(torch.sum(p.to(DEVICE) ** 2).item() for p in active_params)
        theta_phase = alpha_theta * active_norm_sq
        
        _, client_opt = server_client_optimizers(active_params, hyperparams)
        round_start_active_params = [param.detach().clone() for param in active_params]
        
        # Map current and previous states for the linear variance approximation
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
            
            # 4. Lazy Variance Checking
            if round_epochs >= e_min:
                client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
                
                # Convert list of tensors to list of dicts to interface with new approximation
                # client_drifts_dicts = [{i: tensor for i, tensor in enumerate(drift)} for drift in client_drifts]
                client_drifts_dicts = [{i: tensor for i, tensor in enumerate(drift)} for drift in client_drifts.values()]

                # 5. Distributed Stochasticity & The Sketching Paradox Fix (Linear Approx)
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
        
        # Save the current state as the historical previous state for the next time this matrix activates
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