import os
import sys
import torch
import gc
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdaopt import utils
from fdaopt.training.fed_opt import fed_lora_svd_opt
from fdaopt.training.fda_opt import fda_lora_svd_opt

def run_svd_sanity_comparison():
    print("Initializing Hardware Settings...")
    utils.DEVICE_RAM_PROGRAM = 'performance'
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    utils.set_device(device)
    utils.set_seed(42)
    
    print(f"Target Device: {device}")
    
    # Simpler Hyperparameters for Sanity Testing
    hyperparams = {
        'checkpoint': "hf-internal-testing/tiny-random-bert",
        'ds_path': "glue",
        'ds_name': "sst2",
        'num_labels': 2,
        
        'num_clients': 10,
        'clients_per_round': 2,
        'alpha': 1.0, 
        
        'total_rounds': 4,
        'local_epochs': 1, 
        'batch_size': 16,
        'num_workers': 0,
        'pin_memory': False,
        
        'e_min': 2,
        'max_epochs': 5,
        'alpha_theta': 0.01,
        
        'server_opt_name': 'AdamW',
        'server_opt_lr': 1e-3,
        'client_opt_name': 'SGD',
        'client_opt_lr': 0.01,
        
        'lora_r': 8,
        'lora_alpha': 16,
        'target_modules': ("query", "value"),
        
        'threshold': False
    }

    print("\n" + "="*50)
    print("Sanity Check Hyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")

    # --- TEST 1: FED-LORA-SVD ---
    print(">>> STARTING TEST 1: FED-LORA-SVD (Fixed Epochs) <<<")
    start_time = time.time()
    try:
        hyperparams_fed = hyperparams.copy()
        hyperparams_fed['algorithm'] = 'FED_LORA_SVD_SANITY'
        fed_lora_svd_opt(hyperparams_fed)
        print(f"\n[SUCCESS] FED-LORA-SVD finished in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"\n[ERROR] FED-LORA-SVD encountered an error: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "-"*50 + "\n")

    # --- TEST 2: FDA-LORA-SVD ---
    print(">>> STARTING TEST 2: FDA-LORA-SVD (Dynamic Epochs) <<<")
    start_time = time.time()
    try:
        hyperparams_fda = hyperparams.copy()
        hyperparams_fda['algorithm'] = 'FDA_LORA_SVD_SANITY'
        fda_lora_svd_opt(hyperparams_fda)
        print(f"\n[SUCCESS] FDA-LORA-SVD finished in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"\n[ERROR] FDA-LORA-SVD encountered an error: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("\n" + "="*50)
    print("Sanity Comparison Complete. Metrics saved to 'results/round_metrics/'.")

if __name__ == "__main__":
    run_svd_sanity_comparison()