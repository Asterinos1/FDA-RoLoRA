import gc
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

#CHANGING DATA PREP FILE
# from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.datasets.test_fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (compute_client_drifts, compute_pseudo_gradients, set_gradients, copy_parameters,
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