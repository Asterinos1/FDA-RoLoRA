import os
import sys
import torch
import gc

# This tells Python to look one folder up (in the FDA-LoRA root) for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fdaopt import utils
from fdaopt.training.fed_opt import fedavg_rolora

def run_fedavg_rolora_test():
    print("Initializing Hardware Settings...")

    # utils.DEVICE_RAM_PROGRAM = 'moderate'
    # this hasn't been tested yet, who knows what we will see?!
    utils.DEVICE_RAM_PROGRAM = 'performance'
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    utils.set_device(device)
    utils.set_seed(42)
    
    print(f"Target Device: {device}")
    if device == 'cuda:0':
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

    # hyperparams = {
    #     'checkpoint': "roberta-base",
    #     'use_fast_tokenizer': True,
    #     'ds_path': "glue",
    #     'ds_name': "sst2",
    #     'num_labels': 2,
    #     'num_clients': 50,
    #     'clients_per_round': 10,
    #     'total_rounds': 30,
    #     'local_epochs': 1,
    #     'batch_size': 16,
    #     'num_workers': 0,
    #     'pin_memory': False,
    #     'alpha': 1.0,
    #     'client_opt_name': 'SGD',
    #     'client_opt_lr': 1e-2,
    #     'threshold': False,
    #     'lora_r': 8,
    #     'lora_alpha': 16,
    #     'target_modules': ("query", "value"),
    #     'fda': False,
    #     'server_opt_name': 'None',
    #     'server_opt_lr': 0.0
    # }

    # --- Scaled Hyperparameters for Scientific Scenario ---
    # Designed for a ~1.5 hour run on RTX 4060
    hyperparams = {
        'checkpoint': "roberta-base",
        'ds_path': "glue",
        'ds_name': "sst2",
        'num_labels': 2,
        'num_clients': 80,          # Reduced from 200 to speed up partitioning
        'clients_per_round': 8,     # 10% participation (very standard)
        'total_rounds': 100,         # Deep enough to show convergence trends
        'local_epochs': 1,
        'batch_size': 32,            # Stable for 16GB RAM
        'num_workers': 0,            # Required for Win11
        'pin_memory': False,
        'alpha': 0.5,                # Keep the Non-IID complexity
        'client_opt_name': 'SGD',
        'client_opt_lr': 0.05,
        'lora_r': 8,
        'lora_alpha': 16,
        'target_modules': ("query", "value"),
        'fda': False,
        'server_opt_name': 'None',
        'server_opt_lr': 0.0,
        'threshold': False
    }

    print("\nHyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")

    print("\nStarting FedAvg-RoLoRA Training...")
    try:
        fedavg_rolora(hyperparams)
        print("\nTest finished successfully. Metrics are saved in 'results/round_metrics/'.")
    except Exception as e:
        print(f"\nTraining encountered an error: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_fedavg_rolora_test()