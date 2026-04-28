import sys
import os
import time
import torch
import gc

# Add the root directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdaopt import utils
from fdaopt.training.fed_opt import fed_lora_opt, fedavg_rolora, fed_lora_svd_opt
from fdaopt.training.fda_opt import fda_rolora_opt, fda_lora_svd_opt

# Super-fast hyperparameter configuration for sanity checking
FAST_TEST_HYPERPARAMS = {
    'checkpoint': "hf-internal-testing/tiny-random-bert",  # Tiny model for instant passes
    'ds_path': "glue",
    'ds_name': "sst2",           # Smallest sequence classification task
    'num_labels': 2,
    'num_clients': 2,            # Only 2 clients
    'clients_per_round': 2,      # Sample both clients
    'total_rounds': 1,           # Exactly 1 round
    'local_epochs': 1,           # 1 base local epoch
    'max_epochs': 2,             # Cap FDA at 2 epochs
    'e_min': 1,                  # FDA minimum epochs
    'batch_size': 8,
    'num_workers': 0,
    'pin_memory': False,
    'alpha': 1.0,
    'server_opt_name': 'Adam',
    'server_opt_lr': 1e-3,
    'client_opt_name': 'SGD',
    'client_opt_lr': 1e-2,
    'lora_r': 4,
    'lora_alpha': 8,
    'target_modules': ("query", "value"),
    'alpha_theta': 0.01,
    'gamma': 0.3,                # For FDA-RoLoRA optimizer state decay
    'threshold': False           # Don't early stop based on metrics
}

def run_mini_experiment(name, func):
    print(f"\n{'='*50}")
    print(f"STARTING FAST TEST: {name}")
    print(f"{'='*50}")
    
    current_hyperparams = FAST_TEST_HYPERPARAMS.copy()
    # Add an extra identifier so the CSV file is clearly marked
    current_hyperparams['extras'] = f"SANITY_CHECK_{name}"
    
    start_time = time.time()
    try:
        func(current_hyperparams)
        duration = time.time() - start_time
        print(f"\nSUCCESS: {name} completed perfectly in {duration:.2f} seconds.")
    except Exception as e:
        print(f"\nFAILED: {name} encountered a critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up memory between tests to avoid OOM even on tiny models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Initializing Hardware Settings for Sanity Check...")
    utils.DEVICE_RAM_PROGRAM = 'performance'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    utils.set_device(device)
    utils.set_seed(42)
    
    print(f"Target Device: {device}")
    if device.startswith('cuda'):
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        
    print("\nExecuting the Grand Comparison Lineup...")
    
    # The First Track: RoLoRA Evaluation
    run_mini_experiment("Fed-LoRA (Baseline)", fed_lora_opt)
    run_mini_experiment("FedAvg-RoLoRA", fedavg_rolora)
    run_mini_experiment("FDA-RoLoRA", fda_rolora_opt)
    
    # The Second Track: SVD Evaluation
    run_mini_experiment("Fed-LoRA-SVD", fed_lora_svd_opt)
    run_mini_experiment("FDA-LoRA-SVD", fda_lora_svd_opt)
    
    print(f"\n{'='*50}")
    print("Sanity Check Complete! Check 'results/round_metrics/' to ensure the CSVs contain the new 'update_l2_norm' and 'time_*' columns.")