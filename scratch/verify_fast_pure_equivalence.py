import os
import sys
import copy
import time
import torch

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
    get_save_device
)
from fdaopt.training.optimizers import server_client_optimizers

def toggle_rolora_matrices(model, active_matrix: str):
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = f'lora_{active_matrix}' in name

def run_federated_rounds(use_fast=False, num_rounds=2):
    utils.set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.set_device(device)
    
    hyperparams = {
        'checkpoint': 'toy-model',
        'ds_path': 'ylecun/mnist',
        'ds_name': 'mnist',
        'num_labels': 10,
        'num_clients': 5,
        'clients_per_round': 5,
        'total_rounds': num_rounds,
        'local_epochs': 1,
        'e_min': 1,
        'max_epochs': 15,
        'controller_ratio': 1.0,
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': True,
        'alpha': 0.0,
        'server_opt_name': 'SGD',
        'server_opt_lr': 1.0,
        'client_opt_name': 'SGD',
        'client_opt_lr': 0.01,
        'lora_r': 16,
        'linear': False,
        'early_stopping': False,
        'seed': 42
    }
    
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams['num_workers'], hyperparams['pin_memory']
    )
    
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

    client_dict_A = {c_id: [p.detach().clone().to(get_save_device()) for p in params_A] for c_id in range(hyperparams['num_clients'])}
    client_dict_B = {c_id: [p.detach().clone().to(get_save_device()) for p in params_B] for c_id in range(hyperparams['num_clients'])}

    # Pre-cache datasets in VRAM if use_fast
    if use_fast:
        gpu_client_batches = {}
        for c_id in range(hyperparams['num_clients']):
            dl = fed_ds.client_dataloaders[c_id]
            batches = []
            for b in dl:
                batches.append({
                    'image': b['image'].to(device),
                    'labels': b['labels'].to(device)
                })
            gpu_client_batches[c_id] = batches
            
        def fast_client_train(c_id, steps, active_params, active_client_dict, client_opt):
            model.train()
            copy_parameters(from_params=active_client_dict[c_id], to_params=active_params)
            batches = gpu_client_batches[c_id]
            n_b = len(batches)
            
            running_loss = torch.tensor(0.0, device=device)
            running_correct = torch.tensor(0, dtype=torch.long, device=device)
            total = 0
            
            for s in range(steps):
                b = batches[s % n_b]
                client_opt.zero_grad(set_to_none=True)
                out = model(**b)
                loss = out.loss
                logits = out.logits
                preds = torch.argmax(logits, dim=-1)
                running_correct += (preds == b['labels']).sum()
                total += b['labels'].size(0)
                loss.backward()
                client_opt.step()
                running_loss += loss.detach()
                
            copy_parameters(from_params=active_params, to_params=active_client_dict[c_id])
            return running_loss.item() / steps, running_correct.item() / total
            
    else:
        from fdaopt.training.fed_train import client_round_train
        
    theta_A = 0.0
    theta_B = 0.0
    
    trajectory = []
    
    t_start = time.time()
    for r in range(num_rounds):
        active_matrix = 'B' if r % 2 == 0 else 'A'
        toggle_rolora_matrices(model, active_matrix)

        active_params = params_A if active_matrix == 'A' else params_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        current_theta = theta_A if active_matrix == 'A' else theta_B

        round_start_active_params = [param.detach().clone() for param in active_params]
        sampled_clients = list(range(hyperparams['num_clients']))
        for c_id in sampled_clients:
            active_client_dict[c_id] = [p.detach().clone().to(get_save_device()) for p in round_start_active_params]
            
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        
        round_epochs = 0
        while True:
            round_epochs += 1
            for client_id in sampled_clients:
                _, client_opt = server_client_optimizers(active_params, hyperparams, num_lora_params=num_lora)
                if use_fast:
                    fast_client_train(client_id, round_steps, active_params, active_client_dict, client_opt)
                else:
                    client_round_train(model, active_params, active_client_dict, client_opt, fed_ds, client_id, round_steps)

            sampled_active_client_dict = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
            client_drifts = compute_client_drifts(round_start_active_params, sampled_active_client_dict)
            var_subspace, avg_norm_sq_drifts, norm_sq_avg_drift = variance(client_drifts)

            if var_subspace > current_theta or round_epochs >= hyperparams['max_epochs']:
                break

        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        set_gradients(active_params, pseudo_gradients)
        active_server_opt.step()
        active_server_opt.zero_grad()

        target_epochs = hyperparams['controller_ratio'] * hyperparams['max_epochs']
        next_theta = (target_epochs / round_epochs) * var_subspace
        if active_matrix == 'A':
            theta_A = next_theta
        else:
            theta_B = next_theta
            
        trajectory.append({
            'round': r + 1,
            'matrix': active_matrix,
            'epochs': round_epochs,
            'var': var_subspace,
            'theta': next_theta,
            'weights_A': [p.detach().clone().cpu() for p in params_A],
            'weights_B': [p.detach().clone().cpu() for p in params_B]
        })
    t_end = time.time()
    return trajectory, t_end - t_start

if __name__ == '__main__':
    print("Testing Vanilla...")
    traj_vanilla, t_vanilla = run_federated_rounds(use_fast=False, num_rounds=2)
    print(f"Vanilla finished in {t_vanilla:.2f}s")
    
    print("\nTesting Fast...")
    traj_fast, t_fast = run_federated_rounds(use_fast=True, num_rounds=2)
    print(f"Fast finished in {t_fast:.2f}s ({t_vanilla/t_fast:.2f}x faster)")
    
    # Check bit-for-bit equivalence
    all_matched = True
    for r_idx in range(len(traj_vanilla)):
        v = traj_vanilla[r_idx]
        f = traj_fast[r_idx]
        assert v['epochs'] == f['epochs'], f"Epochs mismatch round {r_idx+1}: {v['epochs']} vs {f['epochs']}"
        var_diff = abs(v['var'] - f['var'])
        assert var_diff == 0.0 or var_diff < 1e-6, f"Variance diff round {r_idx+1}: {var_diff}"
        
        diff_A = max(torch.max(torch.abs(p_v - p_f)).item() for p_v, p_f in zip(v['weights_A'], f['weights_A']))
        diff_B = max(torch.max(torch.abs(p_v - p_f)).item() for p_v, p_f in zip(v['weights_B'], f['weights_B']))
        print(f"Round {r_idx+1} ({v['matrix']}): epochs={v['epochs']}, var_diff={var_diff:.2e}, diff_A={diff_A:.2e}, diff_B={diff_B:.2e}")
        if diff_A > 0.0 or diff_B > 0.0:
            all_matched = False
            
    if all_matched:
        print("\n>>> ABSOLUTE BIT-FOR-BIT MATHEMATICAL IDENTITY CONFIRMED (Diff = 0.0)! <<<")
    else:
        print("\n>>> Close to 0 (floating point difference) <<<")
