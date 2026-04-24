import os
import sys
import torch
import gc
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdaopt import utils
from fdaopt.training.fed_opt import fed_lora_opt
from fdaopt.training.fda_opt import fda_lora_opt

def run_comparison_test():
    print("Initializing Hardware Settings...")
    utils.DEVICE_RAM_PROGRAM = 'performance'
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    utils.set_device(device)
    utils.set_seed(42)
    
    print(f"Target Device: {device}")
    if device.startswith('cuda'):
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

    hyperparams = {
        # Model & Dataset (Realistic Scale)
        'checkpoint': "roberta-base",
        'ds_path': "glue",
        'ds_name': "sst2",
        'num_labels': 2,
        
        # Cross-Device Partitioning & Sampling
        'num_clients': 100,
        'clients_per_round': 10,
        'alpha': 1.0, 
        
        # Training Horizons
        'total_rounds': 200,
        'local_epochs': 1,
        'batch_size': 8,
        'num_workers': 4,
        'pin_memory': True,
        
        # FDA-Specific Boundaries
        'e_min': 2,
        'max_epochs': 10,
        'alpha_theta': 0.01,
        
        # Optimizers (Tuned for FedAdam on SST-2)
        'server_opt_name': 'AdamW',
        'server_opt_lr': 1e-5, 
        'client_opt_name': 'SGD',
        'client_opt_lr': 1e-4, 
        
        # LoRA Architecture
        'lora_r': 8,
        'lora_alpha': 16,
        'target_modules': ("query", "value"),
        
        # Early Stopping
        'threshold': True 
    }

    print("\n" + "="*50)
    print("Shared Hyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")

    print(">>> STARTING TEST 1: Standard Fed-LoRA (Fixed Epochs) <<<")
    start_time = time.time()
    try:
        hyperparams_fed = hyperparams.copy()
        hyperparams_fed['algorithm'] = 'FED_LORA_BASELINE'
        fed_lora_opt(hyperparams_fed)
        print(f"[SUCCESS] Fed-LoRA finished successfully in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"[ERROR] Fed-LoRA encountered an error: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "-"*50 + "\n")

    print(">>> STARTING TEST 2: FDA Fed-LoRA (Dynamic Epochs) <<<")
    start_time = time.time()
    try:
        hyperparams_fda = hyperparams.copy()
        hyperparams_fda['algorithm'] = 'FDA_LORA_DYNAMIC'
        fda_lora_opt(hyperparams_fda)
        print(f"[SUCCESS] FDA Fed-LoRA finished successfully in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"[ERROR] FDA Fed-LoRA encountered an error: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("\n" + "="*50)
    print("Comparison Test Complete. Check the 'results/round_metrics/' directory for the CSV files.")

if __name__ == "__main__":
    run_comparison_test()