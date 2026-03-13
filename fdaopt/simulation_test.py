import sys
import os
import torch
import gc
from huggingface_hub import login

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.fda_opt import fda_opt
from training.fed_opt import fed_opt
from training.fed_lora_opt import fed_lora_opt
from training.fda_lora_opt import fda_lora_opt
from training.fed_sa_lora_opt import fed_sa_lora_opt
from training.fda_sa_lora_opt import fda_sa_lora_opt

try:
    from lol import HF_TOKEN
    login(token=HF_TOKEN)
except ImportError:
    print("HF_TOKEN not found in lol.py; proceeding without login.")

# currently for testing on my laptop
BASE_HYPERPARAMS = {
    'checkpoint': "distilbert-base-uncased", 
    'use_fast_tokenizer': True,
    'ds_path': "glue",                  
    'ds_name': "sst2",                  
    'num_labels': 2,
    'num_clients': 12,                  
    'clients_per_round': 3,              
    'total_rounds': 20,                  
    'local_epochs': 1,
    'max_epochs': 3,                     
    'batch_size': 32,                    
    'num_workers': 4,                    
    'pin_memory': True,    
    'alpha': 0.5,                        
    'server_opt_name': 'AdamW',
    'server_opt_lr': 5e-4,
    'client_opt_name': 'SGD',
    'client_opt_lr': 1e-2,
    'theta': 0.05,                       
    'threshold': False,
    'lora_r': 8,                         
    'lora_alpha': 16,
    'target_modules': ["q_lin", "v_lin"], # DistilBERT specific attention layers
    'include_b_in_variance': True
}

def run_performance_suite(selected_methods=None):
    """
    Runs a series of tests and cleans up VRAM between each to prevent OOM.
    """
    methods = {
        "Fed-Opt": fed_opt,
        "FedLoRA": fed_lora_opt,
        "FDA-LoRA": fda_lora_opt,
        "FedSA-LoRA": fed_sa_lora_opt,
        "FDA-SA-LoRA": fda_sa_lora_opt
    }

    if selected_methods:
        methods = {k: v for k, v in methods.items() if k in selected_methods}

    print(f"System Check: GPU - {torch.cuda.get_device_name(0)}")
    print(f"Total Rounds: {BASE_HYPERPARAMS['total_rounds']}")
    
    for name, func in methods.items():
        print(f"\n{'#'*30}")
        print(f"RUNNING EXPERIMENT: {name}")
        print(f"{'#'*30}")
        
        hp = BASE_HYPERPARAMS.copy()
        hp['method'] = name
        
        try:
            func(hp)
            print(f"COMPLETED: {name}")
        except Exception as e:
            print(f"CRASHED: {name} with error: {e}")
        finally:
            # Crucial for laptop with 8GB VRAM: Clear memory between runs
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # You can pass a list of specific methods to run just one or two
    # run_performance_suite(["FDA-LoRA", "FDA-SA-LoRA"])
    run_performance_suite()