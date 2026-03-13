import gc
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

from fdaopt.datasets.test_fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (
    compute_client_drifts, compute_pseudo_gradients, set_gradients, copy_parameters,
    get_updated_client_parameters, variance, compute_metrics, fda_variance_approx, 
    stop_training_with_threshold, compute_drifts
)
from fdaopt.training.sketch import AmsSketch
from fdaopt.training.optimizers import server_client_optimizers
from fdaopt.utils import torch, DEVICE, AutoModelForSequenceClassification, DEVICE_RAM_PROGRAM

from fdaopt.training.sa_fed_train import sa_client_round_train

SAVE_DEVICE = None
if DEVICE_RAM_PROGRAM == 'performance' or DEVICE_RAM_PROGRAM == 'moderate':
    SAVE_DEVICE = DEVICE
elif DEVICE_RAM_PROGRAM == 'low':
    SAVE_DEVICE = 'cpu'

def fda_sa_lora_opt(hyperparams):
    USE_AMSSKETCH = True 
    # toggle between including only the shared A parameters in the variance approximation or both A and B parameters.
    INCLUDE_B_IN_VARIANCE = hyperparams.get('include_b_in_variance', False)

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