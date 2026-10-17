import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdaopt import utils
from fdaopt.models.toy_model import ToyModel
from fdaopt.datasets.fed_data_prep import prepare_federated_datasets
from fdaopt.models.ops import (
    compute_client_drifts,
    variance,
    compute_pseudo_gradients,
    copy_parameters,
    set_gradients,
    get_save_device,
    compute_metrics
)
from fdaopt.training.optimizers import server_client_optimizers
from fdaopt.metrics.mentrics_handler import MetricsHandler

def toggle_rolora_matrices(model, active_matrix: str):
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = f'lora_{active_matrix}' in name

def run_toy_pure_test(scenario='10-Clients-1-Class', linear=False, seed=42, num_rounds=5, controller_ratio=1.0):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.set_device(device)
    utils.set_seed(seed)
    
    num_clients = 10 if '10-Clients' in scenario else 5
    clients_per_round = num_clients
    arch = "LIN" if linear else "NONLIN"
    e_min = 1
    e_max = 15
    batch_size = 64
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    extras = f"TOY-PURE-TEST_{timestamp}_FDA-Pure_{scenario}_{arch}_CRATIO{controller_ratio}_S{seed}"
    
    hyperparams = {
        'checkpoint': 'toy-model',
        'ds_path': 'ylecun/mnist',
        'ds_name': 'mnist',
        'num_labels': 10,
        'num_clients': num_clients,
        'clients_per_round': clients_per_round,
        'total_rounds': num_rounds,
        'local_epochs': 1,
        'e_min': e_min,
        'max_epochs': e_max,
        'controller_ratio': controller_ratio,
        'batch_size': batch_size,
        'num_workers': 0,
        'pin_memory': True,
        'alpha': 0.0,
        'server_opt_name': 'SGD',
        'server_opt_lr': 1.0,
        'client_opt_name': 'SGD',
        'client_opt_lr': 0.01,
        'lora_r': 16,
        'linear': linear,
        'extras': extras,
        'method': 'FDA-RoLoRA-DualTrack-Pure',
        'METHOD_NAME': 'FDA-RoLoRA-DualTrack-Pure',
        'early_stopping': False,
        'seed': seed,
        'enable_b_discount': False,
        'b_discount_factor': 1.0,
        'start_matrix': 'B',
        'first_round_guard': False,
    }
    
    # Load dataset
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams['num_workers'], hyperparams['pin_memory']
    )
    
    # Pre-cache client data onto GPU tensors for safe acceleration
    client_gpu_data = {}
    steps_per_epoch = {}
    for c_id in range(num_clients):
        dl = fed_ds.client_dataloaders[c_id]
        all_imgs, all_lbls = [], []
        for b in dl:
            all_imgs.append(b['image'])
            all_lbls.append(b['labels'])
        client_gpu_data[c_id] = (
            torch.cat(all_imgs, dim=0).to(device),
            torch.cat(all_lbls, dim=0).to(device)
        )
        steps_per_epoch[c_id] = len(dl)

    # Pre-cache test dataset onto GPU tensors
    test_dl = test_ds['mnist']
    test_imgs, test_lbls = [], []
    for b in test_dl:
        test_imgs.append(b['image'])
        test_lbls.append(b['labels'])
    gpu_test_x = torch.cat(test_imgs, dim=0).to(device)
    gpu_test_y = torch.cat(test_lbls, dim=0).to(device)

    # Initialize model
    model = ToyModel(
        rank=hyperparams['lora_r'],
        linear=hyperparams['linear']
    ).to(device)
    
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    num_lora = len(params_A)
    params_A = params_A + params_other
    params_B = params_B + params_other

    server_opt_A, _ = server_client_optimizers(params_A, hyperparams, num_lora_params=num_lora)
    server_opt_B, _ = server_client_optimizers(params_B, hyperparams, num_lora_params=num_lora)

    client_dict_A = {c_id: [p.detach().clone().to(get_save_device()) for p in params_A] for c_id in range(num_clients)}
    client_dict_B = {c_id: [p.detach().clone().to(get_save_device()) for p in params_B] for c_id in range(num_clients)}

    metrics_handler = MetricsHandler(hyperparams)
    theta_A = 0.0
    theta_B = 0.0

    print(f"Starting {hyperparams['extras']}...")
    t_start_total = time.time()
    
    # Generator for reproducible RNG
    rng_generator = torch.Generator(device=device)
    rng_generator.manual_seed(seed)

    for r in range(num_rounds):
        r_start = time.time()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)

        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        current_theta = theta_A if active_matrix == 'A' else theta_B

        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = list(range(num_clients))
        for c_id in sampled_clients:
            active_client_dict[c_id] = [p.detach().clone().to(get_save_device()) for p in round_start_active_params]

        training_loss = 0.0
        training_accuracy = 0.0
        var_subspace = 0.0
        avg_norm_sq_drifts = 0.0
        norm_sq_avg_drift = 0.0
        round_epochs = 0

        train_start = time.time()
        while True:
            round_epochs += 1
            for client_id in sampled_clients:
                _, client_opt = server_client_optimizers(active_params, hyperparams, num_lora_params=num_lora)
                # Client local training for 1 epoch
                model.train()
                copy_parameters(from_params=active_client_dict[client_id], to_params=active_params)
                
                cx, cy = client_gpu_data[client_id]
                N = cx.size(0)
                perm = torch.randperm(N, generator=rng_generator, device=device)
                px, py = cx[perm], cy[perm]
                
                running_loss = torch.tensor(0.0, device=device)
                running_correct = torch.tensor(0, dtype=torch.long, device=device)
                n_batches = 0
                
                for i in range(0, N, batch_size):
                    bx = px[i : i + batch_size]
                    by = py[i : i + batch_size]
                    client_opt.zero_grad(set_to_none=True)
                    out = model(image=bx, labels=by)
                    loss = out.loss
                    preds = torch.argmax(out.logits, dim=-1)
                    running_correct += (preds == by).sum()
                    loss.backward()
                    client_opt.step()
                    running_loss += loss.detach()
                    n_batches += 1
                    
                copy_parameters(from_params=active_params, to_params=active_client_dict[client_id])
                training_loss += (running_loss.item() / n_batches)
                training_accuracy += (running_correct.item() / N)

            # Subspace drift variance evaluation
            sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
            client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
            var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)

            # Pure stopping condition: breaks if variance exceeds threshold or hits e_max
            if var_subspace > current_theta or round_epochs >= e_max:
                break

        train_end = time.time()
        agg_start = time.time()

        training_loss = training_loss / round_epochs
        training_accuracy = training_accuracy / round_epochs
        pseudo_gradients = compute_pseudo_gradients(client_drifts)

        with torch.no_grad():
            update_l2_norm = float(torch.sqrt(sum(torch.sum(g**2) for g in pseudo_gradients)).item())

        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        set_gradients(active_params, pseudo_gradients)
        active_server_opt.step()
        active_server_opt.zero_grad()

        agg_end = time.time()

        # Update specific phase theta with proportional setpoint
        target_epochs = controller_ratio * e_max
        next_theta = (target_epochs / round_epochs) * var_subspace
        if active_matrix == 'A':
            theta_A = next_theta
        else:
            theta_B = next_theta

        # Fast evaluation on GPU
        model.eval()
        with torch.no_grad():
            test_out = model(image=gpu_test_x, labels=gpu_test_y)
            test_loss = test_out.loss.item()
            test_preds = torch.argmax(test_out.logits, dim=-1)
            test_acc = (test_preds == gpu_test_y).sum().item() / gpu_test_y.size(0)

        round_end = time.time()

        metrics = {
            "method": "FDA-RoLoRA-DualTrack-Pure",
            "round": r + 1,
            "active_matrix": active_matrix,
            "start_matrix": "B",
            "first_round_guard": False,
            "e_min": e_min,
            "e_max": e_max,
            "time_client_training": train_end - train_start,
            "time_server_aggregation": agg_end - agg_start,
            "time_round_total": round_end - r_start,
            "update_l2_norm": update_l2_norm,
            "round_epochs": round_epochs,
            "round_steps": round_epochs * steps_per_epoch[0],
            "variance": var_subspace,
            "effective_variance": var_subspace,
            "avg_norm_sq_drifts": avg_norm_sq_drifts,
            "norm_sq_avg_drift": norm_sq_avg_drift,
            "theta_phase": current_theta,
            "controller_ratio": controller_ratio,
            "enable_b_discount": False,
            "b_discount_factor": 1.0,
            "accuracy": test_acc,
            "testing_loss": test_loss,
            "training_loss": training_loss / clients_per_round,
            "training_accuracy": training_accuracy / clients_per_round
        }
        metrics_handler.append_round_metrics(metrics)
        print(f"[{time.strftime('%H:%M:%S')}] Round {r+1}/{num_rounds} ({active_matrix}): epochs={round_epochs}, acc={test_acc:.4f}, loss={test_loss:.4f}, var={var_subspace:.3f}, time={round_end-r_start:.2f}s")

    metrics_handler.save_metrics()
    print(f"\nFinished in {time.time() - t_start_total:.2f}s! CSV saved to {metrics_handler.filename}")
    return metrics_handler.filename

if __name__ == '__main__':
    run_toy_pure_test(scenario='10-Clients-1-Class', linear=False, seed=42, num_rounds=5, controller_ratio=1.0)
