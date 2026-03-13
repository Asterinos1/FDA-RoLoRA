import gc
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

from fdaopt.datasets.test_fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (
    compute_client_drifts, compute_pseudo_gradients, set_gradients, 
    copy_parameters, get_updated_client_parameters, variance, 
    compute_metrics, stop_training_with_threshold
)
from fdaopt.training.optimizers import server_client_optimizers
from fdaopt.utils import DEVICE, AutoModelForSequenceClassification, DEVICE_RAM_PROGRAM

# Import our new SA training loop
from fdaopt.training.sa_fed_train import sa_client_round_train

SAVE_DEVICE = None
if DEVICE_RAM_PROGRAM == 'performance' or DEVICE_RAM_PROGRAM == 'moderate':
    SAVE_DEVICE = DEVICE
elif DEVICE_RAM_PROGRAM == 'low':
    SAVE_DEVICE = 'cpu'

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