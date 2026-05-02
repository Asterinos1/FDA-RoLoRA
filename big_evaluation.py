import os
import sys
import time
import subprocess
import argparse
import json
import itertools

TARGET_DEVICE = 'cuda:1'

# 1. TOPOLOGY SCENARIOS
SCENARIOS = {
    "Cross-Silo": {
        'num_clients': 10,
        'clients_per_round': 10,
        'local_epochs': 5,
        'batch_size': 16
    },
    "Cross-Device": {
        'num_clients': 100,
        'clients_per_round': 10,
        'local_epochs': 1,
        'batch_size': 8
    }
}

# 2. DATA DISTRIBUTIONS (IID vs Non-IID)
DISTRIBUTIONS = {
    "Good-IID": {'alpha': 10.0},
    "Bad-IID": {'alpha': 0.1}
}

# 3. OPTIMIZER SETTINGS
# 'Without-Server-Opt' simulates standard FedAvg (Server learning rate 1.0, SGD)
OPTIMIZERS = {
    "Without-Server-Opt": {
        'server_opt_name': 'SGD', 
        'server_opt_lr': 1.0, 
        'client_opt_name': 'SGD', 
        'client_opt_lr': 0.01
    },
    "With-Server-Opt": {
        'server_opt_name': 'AdamW', 
        'server_opt_lr': 1e-3, 
        'client_opt_name': 'SGD', 
        'client_opt_lr': 0.01
    }
}

# 4. METHODS & SPECIFIC HYPERPARAMETERS
# We define lists of dictionaries for each method to test their unique traits.
METHODS_CONFIG = {
    "Fed-LoRA": [
        {} # Baseline, no special params
    ],
    "FDA-LoRA": [
        {'alpha_theta': 0.01, 'max_epochs': 10}, # Standard FDA
        {'alpha_theta': 0.05, 'max_epochs': 10}  # Loose threshold
    ],
    "FedAvg-RoLoRA": [
        {} 
    ],
    "FDA-RoLoRA": [
        {'alpha_theta': 0.01, 'gamma': 0.3}, # Aggressive momentum decay
        {'alpha_theta': 0.01, 'gamma': 0.8}, # Light momentum decay
        {'alpha_theta': 0.05, 'gamma': 0.3}  # Loose threshold + aggressive decay
    ],
    "Fed-LoRA-SVD": [
        {} 
    ],
    "FDA-LoRA-SVD": [
        {'alpha_theta': 0.01}
    ]
}

# 5. CONSTANT BASE PARAMETERS
BASE_HYPERPARAMS = {
    'checkpoint': "roberta-base",
    'ds_path': "glue",
    'ds_name': "mrpc",
    'num_labels': 2,
    'total_rounds': 80,
    'e_min': 2,
    'num_workers': 4,
    'pin_memory': True,
    'lora_r': 8,
    'lora_alpha': 16,
    'target_modules': ("query", "value"),
    'threshold': False
}

def execute_run(config_json):
    import torch
    from fdaopt import utils
    from fdaopt.training.fed_opt import fed_lora_opt, fedavg_rolora, fed_lora_svd_opt
    from fdaopt.training.fda_opt import fda_rolora_opt, fda_lora_svd_opt, fda_lora_opt

    hyperparams = json.loads(config_json)
    method_name = hyperparams.pop('METHOD_NAME')
    run_name = hyperparams['extras']
    
    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(TARGET_DEVICE)
    utils.set_seed(42)

    print(f"[{time.strftime('%H:%M:%S')}] Launching {run_name} on {TARGET_DEVICE}")

    if method_name == "Fed-LoRA":
        fed_lora_opt(hyperparams)
    elif method_name == "FedAvg-RoLoRA":
        fedavg_rolora(hyperparams)
    elif method_name == "FDA-RoLoRA":
        fda_rolora_opt(hyperparams)
    elif method_name == "FDA-LoRA":
        fda_lora_opt(hyperparams)
    elif method_name == "Fed-LoRA-SVD":
        fed_lora_svd_opt(hyperparams)
    elif method_name == "FDA-LoRA-SVD":
        fda_lora_svd_opt(hyperparams)
    else:
        raise ValueError(f"Unknown method requested: {method_name}")

def orchestrate():
    print(f"Starting Massive Cross-Scenario Orchestrator. Target: {TARGET_DEVICE}")
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Generate all combinations
    all_runs = []
    for scenario_name, scenario_cfg in SCENARIOS.items():
        for dist_name, dist_cfg in DISTRIBUTIONS.items():
            for opt_name, opt_cfg in OPTIMIZERS.items():
                for method_name, method_variants in METHODS_CONFIG.items():
                    for variant_idx, variant_cfg in enumerate(method_variants):
                        
                        run_config = BASE_HYPERPARAMS.copy()
                        run_config.update(scenario_cfg)
                        run_config.update(dist_cfg)
                        run_config.update(opt_cfg)
                        run_config.update(variant_cfg)
                        
                        run_config['METHOD_NAME'] = method_name
                        run_config['extras'] = f"{scenario_name}_{dist_name}_{opt_name}_{method_name}_v{variant_idx}"
                        
                        all_runs.append(run_config)

    total_runs = len(all_runs)
    print(f"Total configurations scheduled: {total_runs}\n")

    for current_run, run_config in enumerate(all_runs, 1):
        run_identifier = run_config['extras']
        print(f"--- Progress: {current_run}/{total_runs} | {run_identifier} ---")
        
        out_file = os.path.join(log_dir, f"{run_identifier}.out")
        err_file = os.path.join(log_dir, f"{run_identifier}.err")
        
        config_str = json.dumps(run_config)
        
        cmd = [
            sys.executable, "-u", os.path.abspath(__file__),
            "--run_config", config_str
        ]
        
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        start_time = time.time()
        try:
            with open(out_file, 'w') as stdout_f, open(err_file, 'w') as stderr_f:
                subprocess.run(cmd, env=env, stdout=stdout_f, stderr=stderr_f, check=True)
            duration = (time.time() - start_time) / 60
            print(f"SUCCESS: Finished in {duration:.2f} minutes.")
        except subprocess.CalledProcessError:
            print(f"CRITICAL ERROR: Process crashed. Check {err_file} for stack trace.")
        except Exception as e:
            print(f"UNEXPECTED ERROR: {e}")
        finally:
            time.sleep(5) 

    print("\nAll scheduled experiments have been processed. Metrics are in results/round_metrics/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, default=None, help="Internal use: JSON string of hyperparams.")
    args = parser.parse_args()

    if args.run_config:
        execute_run(args.run_config)
    else:
        orchestrate()