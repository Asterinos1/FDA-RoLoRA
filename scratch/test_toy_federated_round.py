import os
import sys
import torch
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdaopt import utils
from fdaopt.training import fed_opt, fda_opt

def run_smoke_test():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    utils.set_device(device)
    utils.set_seed(42)
    
    base_hyperparams = {
        'checkpoint': 'toy-model',
        'ds_path': 'ylecun/mnist',
        'ds_name': 'mnist',
        'num_labels': 10,
        'num_clients': 5,
        'clients_per_round': 5,
        'total_rounds': 1,
        'local_epochs': 1,
        'e_min': 1,
        'max_epochs': 2,
        'batch_size': 128,
        'num_workers': 0,
        'pin_memory': False,
        'alpha': 0.0,
        'server_opt_name': 'SGD',
        'server_opt_lr': 1.0,
        'client_opt_name': 'SGD',
        'client_opt_lr': 0.01,
        'lora_r': 16,
        'linear': False,
        'extras': 'SMOKE_TEST',
        'method': 'test',
        'threshold': False,
        'seed': 42,
        'alpha_theta': 0.01,
        'gamma': 0.3
    }
    
    test_cases = [
        ("FedAvg-LoRA", fed_opt.fedavg_lora),
        ("FedAvg-RoLoRA", fed_opt.fedavg_rolora),
        ("FFA-LoRA", fed_opt.ffa_lora),
        ("FDA-RoLoRA-DualTrack", fda_opt.fda_rolora_dual_track_opt),
        ("FDA-EMA-Standard", fda_opt.fda_rolora_dual_track_ema_standard_opt),
        ("FDA-SMA", fda_opt.fda_rolora_dual_track_sma_opt),
    ]
    
    results = {}
    for name, func in test_cases:
        print(f"\n--- Running smoke round for {name} ---")
        hp = base_hyperparams.copy()
        hp['extras'] = f"SMOKE_{name}"
        hp['method'] = name
        hp['METHOD_NAME'] = name
        try:
            func(hp)
            print(f"[SUCCESS] {name} completed 1 round successfully.")
            results[name] = "PASSED"
        except Exception as e:
            print(f"[FAILED] {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"FAILED: {e}"
            
    print("\n=== SMOKE TEST SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    run_smoke_test()
