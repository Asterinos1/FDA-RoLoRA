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


    # parameters more alinged with to the paper's main benchmarks
    hyperparams = {
        # 1. UPGRADE: Use the larger 355M parameter model from the paper
        'checkpoint': "roberta-base",  # approx 120m parameters
        # "roberta-large", 
        'ds_path': "glue",
        'ds_name': "sst2",
        'num_labels': 2,
        # 2. UPGRADE: Set num_clients and clients_per_round to equal (Full Participation)
        'num_clients': 50,      
        'clients_per_round': 10, 
        'total_rounds': 50,  # Keep rounds the same to observe convergence trends
        'local_epochs': 2,
        'batch_size': 32,
        'num_workers': 8,  
        'pin_memory': True, # Enable pin_memory for faster GPU data transfer
        'alpha': 0.8,
        'client_opt_name': 'SGD',
        # 3. UPGRADE: Decrease LR to within the optimal range (1e-3 to 2e-2)
        'client_opt_lr': 0.01,  
        # 4. UPGRADE: Set rank to 4 per the paper's main benchmarks
        'lora_r': 8,            
        'lora_alpha': 16,
        'target_modules': ("query", "value"),
        'fda': False,
        'server_opt_name': 'None',
        'server_opt_lr': 0.0,
        'threshold': False
    }

    # hyperparams = {
    #     # 1. Heaviest supported model in your pipeline
    #     'checkpoint': "roberta-large",  
    #     'ds_path': "glue",
    #     'ds_name': "sst2",
    #     'num_labels': 2,
    #     'num_clients': 100,      
    #     'clients_per_round': 10, 
    #     'total_rounds': 50,  
    #     'local_epochs': 2,    
        
    #     # 2. THE ACTIVATION HOG: Massive batch size
    #     # This forces the GPU to hold 128 sequences in memory during the forward/backward pass.
    #     'batch_size': 1024,        
        
    #     'num_workers': 4,  
    #     'pin_memory': True,   
    #     'alpha': 0.1,        
    #     'client_opt_name': 'SGD',
    #     'client_opt_lr': 0.01,  
        
    #     # 3. THE PARAMETER HOG: Max out the LoRA footprint
    #     # Rank 64 across ALL attention and feed-forward layers drastically increases trainable weights.
    #     'lora_r': 64,            
    #     'lora_alpha': 128,      
    #     'target_modules': ("query", "key", "value", "dense"), 
        
    #     'fda': False,
    #     'server_opt_name': 'None',
    #     'server_opt_lr': 0.0,
    #     'threshold': False
    # }

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