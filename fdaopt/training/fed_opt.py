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

#rolora version of FedOpt            
def fed_rolora_opt(hyperparams):
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

    # 1. Explicitly partition the parameters
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]

    group_A = params_A + params_other
    group_B = params_B + params_other
    
    # Store the lengths to easily slice out the LoRA drifts later
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

    metrics_handler = MetricsHandler(hyperparams)

    pbar_rounds = tqdm(range(hyperparams['total_rounds']), desc="Fed-RoLoRA Alt2 Training")
    for r in pbar_rounds:
        training_loss = 0.0
        sampled_clients = client_sampler.sample()
        
        active_matrix = 'A' if r % 2 == 0 else 'B'
        toggle_rolora_matrices(model, active_matrix)

        active_params = group_A if active_matrix == 'A' else group_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        active_lora_len = len_A if active_matrix == 'A' else len_B

        _, client_opt = server_client_optimizers(active_params, hyperparams)
        round_start_active_params = [param.detach().clone() for param in active_params]

        copy_parameters(from_params=active_params, to_params=round_start_active_params)
        
        active_client_dict = get_updated_client_parameters(
            active_client_dict, sampled_clients, round_start_active_params
        )

        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)

        pbar_clients = tqdm(sampled_clients, desc=f"  Round {r+1} ({active_matrix} Active)", leave=False)
        for client_id in pbar_clients:
            loss = client_round_train(
                model, active_params, active_client_dict, 
                client_opt, fed_ds, client_id, round_steps
            )
            training_loss += loss

        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        
        # Calculate full drifts for the server optimizer (LoRA + Classification Head)
        client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        
        set_gradients(active_params, pseudo_gradients)
        active_server_opt.step()
        active_server_opt.zero_grad()

        # Isolate ONLY the LoRA drifts for accurate variance tracking
        lora_only_drifts = {
            client_id: drifts[:active_lora_len] 
            for client_id, drifts in client_drifts.items()
        }

        metrics = {"round": r + 1, "active_matrix": active_matrix} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        
        # Compute variance metrics strictly on the LoRA parameters
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(lora_only_drifts)
        
        metrics['round_steps'] = round_steps
        metrics_handler.append_round_metrics(metrics)
        gc.collect()

        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics): 
            break

    metrics_handler.save_metrics()