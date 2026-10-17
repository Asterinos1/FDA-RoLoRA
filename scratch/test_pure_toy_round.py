import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdaopt import utils
from pure_experiments.fda_rolora_pure import fda_rolora_dual_track_pure_opt

def test():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    utils.set_device(device)
    utils.set_seed(42)
    
    hyperparams = {
        'checkpoint': 'toy-model',
        'ds_path': 'ylecun/mnist',
        'ds_name': 'mnist',
        'num_labels': 10,
        'num_clients': 10,
        'clients_per_round': 10,
        'total_rounds': 2,
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
        'extras': 'SMOKE_PURE_TOY',
        'method': 'FDA-RoLoRA-DualTrack-Pure',
        'early_stopping': False,
        'seed': 42
    }
    
    t0 = time.time()
    fda_rolora_dual_track_pure_opt(hyperparams)
    t1 = time.time()
    print(f"\nCompleted 2 rounds in {t1 - t0:.2f} seconds.")

if __name__ == '__main__':
    test()
