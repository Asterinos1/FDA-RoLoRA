import gc
from tqdm import tqdm
# including LoRA imports
from peft import get_peft_model, LoraConfig, TaskType
from fdaopt.training.fed_train import sa_client_round_train

# from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.datasets.test_fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import compute_client_drifts, compute_pseudo_gradients, set_gradients, copy_parameters, \
    get_updated_client_parameters, variance, compute_metrics, stop_training_with_threshold
from fdaopt.training.fed_train import client_round_train
from fdaopt.training.optimizers import server_client_optimizers
from fdaopt.utils import DEVICE, AutoModelForSequenceClassification, DEVICE_RAM_PROGRAM

# Let RAM usage of one model be M, then,
SAVE_DEVICE = None
if DEVICE_RAM_PROGRAM == 'performance' or DEVICE_RAM_PROGRAM == 'moderate':
    # All client models are stored in DEVICE, which means DEVICE-RAM is O(num_clients * M)
    SAVE_DEVICE = DEVICE
elif DEVICE_RAM_PROGRAM == 'low':
    # All client models are stored in CPU, which means DEVICE-RAM is O(M)
    SAVE_DEVICE = 'cpu'


def fed_opt(hyperparams):

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

    metrics_handler = MetricsHandler(hyperparams)

    pbar_rounds = tqdm(range(hyperparams['total_rounds']), desc="Standard Federated Training")
    for r in pbar_rounds:

        training_loss = 0.0

        sampled_clients = client_sampler.sample()

        # Save the model parameters at the start of this round
        copy_parameters(
            from_params=train_params,
            to_params=round_start_train_params
        )

        client_train_params_dict = get_updated_client_parameters(client_train_params_dict, sampled_clients, round_start_train_params)

        # Calculate the total number of steps for this epoch/round
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)

        pbar_clients = tqdm(sampled_clients, desc=f"  Round {r+1} Clients", leave=False)
        for client_id in pbar_clients:
            loss = client_round_train(model, train_params, client_train_params_dict, client_opt, fed_ds, client_id, round_steps)
            training_loss += loss
            pbar_clients.set_postfix({"client_id": client_id})

        # Reset the model parameters to the parameters at the start of the round.
        copy_parameters(
            from_params=round_start_train_params,
            to_params=train_params
        )

        # Compute the drifts (differences) between the round start parameters and the client parameters
        client_drifts = compute_client_drifts(round_start_train_params, client_train_params_dict)

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
        metrics['round_steps'] = round_steps
        # Pass round metrics to handler
        metrics_handler.append_round_metrics(metrics)

        pbar_rounds.set_postfix({"loss": f"{metrics['training_loss']:.4f}", "var": f"{metrics['variance']:.4e}"})

        gc.collect()

        if hyperparams['threshold']:
            if stop_training_with_threshold(hyperparams, metrics): break

    metrics_handler.save_metrics()

# simple LoRA version of FedOpt
def fed_lora_opt(hyperparams):
    # 1. Dataset Preparation
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

    # 2. Model Initialization with PEFT (LoRA)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'],
        num_labels=hyperparams['num_labels']
    )

    # Define LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=hyperparams.get('lora_r', 8), 
        lora_alpha=hyperparams.get('lora_alpha', 16),
        lora_dropout=0.1,
        target_modules=hyperparams.get('target_modules', None) 
    )

    # Wrap the model: This freezes base weights and creates trainable A/B matrices
    model = get_peft_model(base_model, peft_config)
    model = model.to(DEVICE)
    
    # Print trainable parameters for verification
    model.print_trainable_parameters()

    # 3. Parameter Tracking
    # Extract only trainable parameters (LoRA adapters)
    train_params = [param for param in model.parameters() if param.requires_grad]

    # Create a copy of the LoRA parameters for global reference
    round_start_train_params = [param.detach().clone() for param in train_params]

    # Initialize Optimizers
    server_opt, client_opt = server_client_optimizers(train_params, hyperparams)

    client_sampler = ClientSampler(
        list(range(hyperparams['num_clients'])),
        hyperparams['clients_per_round']
    )

    # Store client-specific LoRA parameters
    client_train_params_dict = {
        client_id: [param.detach().clone().to(SAVE_DEVICE) for param in round_start_train_params]
        for client_id in range(hyperparams['num_clients'])
    }

    metrics_handler = MetricsHandler(hyperparams)

    # 4. Federated Training Loop
    pbar_rounds = tqdm(range(hyperparams['total_rounds']), desc="Fed-LoRA Training")
    for r in pbar_rounds:
        training_loss = 0.0
        sampled_clients = client_sampler.sample()

        # Sync global LoRA weights to the starting point of the round
        copy_parameters(
            from_params=train_params,
            to_params=round_start_train_params
        )

        # Update the dictionary with current global parameters for the sampled clients
        client_train_params_dict = get_updated_client_parameters(
            client_train_params_dict, sampled_clients, round_start_train_params
        )

        # Steps calculation
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)

        # Local Training (updates only LoRA A/B matrices)
        pbar_clients = tqdm(sampled_clients, desc=f"  Round {r+1} Clients", leave=False)
        for client_id in pbar_clients:
            loss = client_round_train(
                model, train_params, client_train_params_dict, 
                client_opt, fed_ds, client_id, round_steps
            )
            training_loss += loss
            pbar_clients.set_postfix({"loss": f"{loss:.4f}"})

        # Global Aggregation
        copy_parameters(
            from_params=round_start_train_params,
            to_params=train_params
        )

        # Compute drifts specifically for the LoRA adapters
        client_drifts = compute_client_drifts(round_start_train_params, client_train_params_dict)

        # Server-side update using the pseudo-gradient (average drift)
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        set_gradients(train_params, pseudo_gradients)
        
        server_opt.step()
        server_opt.zero_grad()

        # 5. Metrics & Checkpointing
        metrics = {"round": r + 1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics['round_steps'] = round_steps
        
        metrics_handler.append_round_metrics(metrics)
        pbar_rounds.set_postfix({"loss": f"{metrics['training_loss']:.4f}", "var": f"{metrics['variance']:.4e}"})

        gc.collect()

        if hyperparams.get('threshold'):
            if stop_training_with_threshold(hyperparams, metrics): 
                break

    metrics_handler.save_metrics()

def fed_sa_lora_opt(hyperparams):
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
    
    # Separate parameters: 'lora_B' is local, everything else (lora_A, classifiers) is shared
    shared_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_B' not in n]
    local_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_B' in n]
    all_train_params = [p for p in model.parameters() if p.requires_grad]

    round_start_shared_params = [param.detach().clone() for param in shared_params]

    # Server optimizes only shared A. Client optimizes both A and B.
    server_opt, _ = server_client_optimizers(shared_params, hyperparams)
    _, client_opt = server_client_optimizers(all_train_params, hyperparams)

    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])

    # Dictionaries for stateful client parameters
    client_shared_params_dict = {
        client_id: [param.detach().clone().to(SAVE_DEVICE) for param in round_start_shared_params]
        for client_id in range(hyperparams['num_clients'])
    }
    
    # Local B parameters persist entirely on the client side and are never aggregated
    client_local_params_dict = {
        client_id: [param.detach().clone().to(SAVE_DEVICE) for param in local_params]
        for client_id in range(hyperparams['num_clients'])
    }

    metrics_handler = MetricsHandler(hyperparams)

    pbar_rounds = tqdm(range(hyperparams['total_rounds']), desc="FedSA-LoRA Training")
    for r in pbar_rounds:
        training_loss = 0.0
        sampled_clients = client_sampler.sample()

        copy_parameters(from_params=shared_params, to_params=round_start_shared_params)

        # Update only the shared parameters for the sampled clients
        client_shared_params_dict = get_updated_client_parameters(
            client_shared_params_dict, sampled_clients, round_start_shared_params
        )

        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)

        pbar_clients = tqdm(sampled_clients, desc=f"  Round {r+1} Clients", leave=False)
        for client_id in pbar_clients:
            loss = sa_client_round_train(
                model, shared_params, local_params, 
                client_shared_params_dict, client_local_params_dict, 
                client_opt, fed_ds, client_id, round_steps
            )
            training_loss += loss

        copy_parameters(from_params=round_start_shared_params, to_params=shared_params)

        # Drifts are computed strictly on the shared parameters
        client_drifts = compute_client_drifts(round_start_shared_params, client_shared_params_dict)

        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        set_gradients(shared_params, pseudo_gradients)
        
        server_opt.step()
        server_opt.zero_grad()

        metrics = {"round": r + 1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics['round_steps'] = round_steps
        
        metrics_handler.append_round_metrics(metrics)
        pbar_rounds.set_postfix({"loss": f"{metrics['training_loss']:.4f}", "var": f"{metrics['variance']:.4e}"})

        gc.collect()

        if hyperparams.get('threshold'):
            if stop_training_with_threshold(hyperparams, metrics): 
                break

    metrics_handler.save_metrics()