import os
import sys
import torch
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdaopt import utils
from fdaopt.training.fda_opt import fda_rolora_opt

def run_fda_rolora_test():
    print("Initializing Hardware Settings...")
    utils.DEVICE_RAM_PROGRAM = 'performance'
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    utils.set_device(device)
    utils.set_seed(42)
    
    print(f"Target Device: {device}")
    if device.startswith('cuda'):
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

    hyperparams = {
        'checkpoint': "hf-internal-testing/tiny-random-bert",
        'ds_path': "glue",
        'ds_name': "sst2",
        'num_labels': 2,
        'num_clients': 10,
        'clients_per_round': 2,
        'total_rounds': 4,
        'local_epochs': 1,
        'batch_size': 16,
        'num_workers': 0,
        'pin_memory': False,
        'alpha': 1.0,
        
        'e_min': 2,
        'max_epochs': 5,
        'alpha_theta': 0.01,
        'gamma': 0.3,
        
        'server_opt_name': 'Adam',
        'server_opt_lr': 1e-3,
        'client_opt_name': 'SGD',
        'client_opt_lr': 0.01,
        
        'lora_r': 4,
        'lora_alpha': 8,
        'target_modules': ("query", "value"),
        
        'threshold': False
    }

    print("\nHyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")

    print("\nStarting FDA-RoLoRA Training Test...")
    try:
        fda_rolora_opt(hyperparams)
        print("\nTest finished successfully. Metrics are saved in 'results/round_metrics/'.")
    except Exception as e:
        print(f"\nTraining encountered an error: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_fda_rolora_test()