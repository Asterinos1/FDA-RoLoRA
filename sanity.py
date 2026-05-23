import os
import sys
import time
import traceback
import torch

from fdaopt import utils
from fdaopt.training.fed_opt import fed_lora_opt, fedavg_rolora, fed_lora_svd_opt
from fdaopt.training.fda_opt import fda_opt, fda_rolora_opt, fda_lora_svd_opt, fda_rolora_dual_track_opt

BASE_HYPERPARAMS = {
    # Changed to bypass the 'use_fast=False' hardcode in fed_data_prep.py
    'checkpoint': "google/bert_uncased_L-2_H-128_A-2", 
    'ds_path': "glue",
    'ds_name': "mrpc",
    'num_labels': 2,
    'total_rounds': 3,
    'num_clients': 4,
    'clients_per_round': 2,
    'local_epochs': 1,
    'batch_size': 2,
    'alpha': 10.0,
    'lora_r': 4,
    'lora_alpha': 8,
    'target_modules': ("query", "value"),
    'num_workers': 0,
    'pin_memory': False,
    'threshold': False,
    'server_opt_name': 'SGD',
    'server_opt_lr': 1.0,
    'client_opt_name': 'SGD',
    'client_opt_lr': 0.01,
    'e_min': 1,
    'max_epochs': 2,
    'alpha_theta': 0.01,
    'gamma': 0.3,
    'theta': 0.01
}

METHODS = {
    "Fed-LoRA": fed_lora_opt,
    "FedAvg-RoLoRA": fedavg_rolora,
    "FDA-RoLoRA": fda_rolora_opt,
    "FDA-RoLoRA2": fda_rolora_opt2,  # Running the same method twice to check for consistency
    "FDA-RoLoRA-Dual-Track": fda_rolora_dual_track_opt,
    # "Fed-LoRA-SVD": fed_lora_svd_opt,
    # "FDA-LoRA-SVD": fda_lora_svd_opt
}

def run_sanity_check():
    target_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(target_device)
    utils.set_seed(42)

    os.makedirs(os.path.join("results", "round_metrics"), exist_ok=True)

    print(f"Starting Sanity Check on {target_device}...")
    print("-" * 50)

    success_count = 0
    total_methods = len(METHODS)

    for method_name, method_func in METHODS.items():
        print(f"[{time.strftime('%H:%M:%S')}] Testing {method_name}...")
        
        hyperparams = BASE_HYPERPARAMS.copy()
        hyperparams['METHOD_NAME'] = method_name
        
        start_time = time.time()
        try:
            method_func(hyperparams)
            duration = time.time() - start_time
            print(f"SUCCESS: {method_name} completed in {duration:.2f} seconds.")
            success_count += 1
        except Exception as e:
            print(f"FAILED: {method_name} encountered an error.")
            traceback.print_exc()
        
        print("-" * 50)

    print("Sanity Check Summary")
    print(f"Passed: {success_count}/{total_methods}")
    
    if success_count == total_methods:
        print("All methods are working correctly.")
        sys.exit(0)
    else:
        print("Some methods failed. Please review the tracebacks above.")
        sys.exit(1)

if __name__ == "__main__":
    run_sanity_check()