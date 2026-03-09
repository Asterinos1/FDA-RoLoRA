import gc
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

#CHANGING DATA PREP FILE
# from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.datasets.test_fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (
    compute_client_drifts, compute_pseudo_gradients, set_gradients, 
    copy_parameters, get_updated_client_parameters, variance, 
    compute_metrics, stop_training_with_threshold
)
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