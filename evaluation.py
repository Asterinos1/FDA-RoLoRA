import os
import sys
import time
import subprocess
import argparse
import json
import concurrent.futures
import queue
import threading

# =============================================================================
# PARALLELIZATION SETTINGS
# =============================================================================
# Define the GPUs you want to use. The script will run exactly one test per GPU listed.
AVAILABLE_DEVICES = ['cuda:0', 'cuda:1'] 
JOBS_PER_DEVICE = 10
MAX_CONCURRENT_JOBS = len(AVAILABLE_DEVICES) * JOBS_PER_DEVICE

# =============================================================================
# 1. THE GLOBAL BASELINE (Immutable rules for all methods)
# =============================================================================
BASE_HYPERPARAMS = {
    'checkpoint': "roberta-base",
    'ds_path': "glue",
    'ds_name': "mrpc",
    'num_labels': 2,
    'total_rounds': 80,
    'batch_size': 16,
    'lora_r': 8,
    'lora_alpha': 16,
    'target_modules': ("query", "value"),
    'num_workers': 4,
    'pin_memory': True,
    'threshold': False
}

# =============================================================================
# 2. THE ENVIRONMENTS (Cross-Silo vs Cross-Device)
# =============================================================================
SCENARIOS = {
    "Cross-Silo": {
        'num_clients': 10,
        'clients_per_round': 10, # 100% Full Participation
        'local_epochs': 4,       # Powerful clients can do more local work
    },
    "Cross-Device": {
        'num_clients': 80,
        'clients_per_round': 10, # 12.5% Partial Participation
        'local_epochs': 1,       # Constrained edge devices
    }
}

# =============================================================================
# 3. DATA DISTRIBUTIONS
# =============================================================================
DISTRIBUTIONS = {
    "IID": {'alpha': 10.0},
    "Semi-IID": {'alpha': 1.0},
    "Non-IID": {'alpha': 0.1}
}

# =============================================================================
# 4. OPTIMIZATION STRATEGIES
# =============================================================================
OPTIMIZERS = {
    "Non-Optimizers": {
        # Strict FedAvg style: Simple averaging on server (SGD with LR=1.0)
        'server_opt_name': 'SGD',
        'server_opt_lr': 1.0,
        'client_opt_name': 'SGD',
        'client_opt_lr': 0.01
    },
    "Advanced-Optimizers": {
        # FedOpt style: Server uses momentum to escape local minima traps
        'server_opt_name': 'AdamW',
        'server_opt_lr': 1e-3,
        'client_opt_name': 'SGD', # Clients usually stay SGD to save memory
        'client_opt_lr': 0.01
    }
}

# =============================================================================
# 5. METHOD SPECIFIC MECHANICS & HYPERPARAMETER SWEEPS
# =============================================================================
METHODS_MECHANICS = {
    "FedAvg-RoLoRA": [
        {
            'server_opt_name': 'SGD',
            'server_opt_lr': 1.0,
        }
    ],
    
    "FDA-RoLoRA": [
        # Sweeping alpha_theta (variance threshold) and gamma (momentum decay)
        {'e_min': 2, 'max_epochs': 10, 'alpha_theta': 0.01, 'gamma': 0.3},
        {'e_min': 2, 'max_epochs': 10, 'alpha_theta': 0.05, 'gamma': 0.3},
        {'e_min': 2, 'max_epochs': 10, 'alpha_theta': 0.01, 'gamma': 0.8},
        {'e_min': 2, 'max_epochs': 10, 'alpha_theta': 0.05, 'gamma': 0.8},
    ],
    
    "Fed-LoRA": [
        {} 
    ],
    
    "Fed-LoRA-SVD": [
        {} 
    ],
    
    "FDA-LoRA-SVD": [
        # Sweeping alpha_theta for the SVD variant
        {'e_min': 2, 'max_epochs': 10, 'alpha_theta': 0.005},
        {'e_min': 2, 'max_epochs': 10, 'alpha_theta': 0.01},
        {'e_min': 2, 'max_epochs': 10, 'alpha_theta': 0.05},
        {'e_min': 2, 'max_epochs': 10, 'alpha_theta': 0.1},
        {'e_min': 2, 'max_epochs': 10, 'alpha_theta': 0.5},
    ]
}

# =============================================================================
# EXECUTION LOGIC
# =============================================================================
def execute_run(config_json):
    import torch
    from fdaopt import utils
    from fdaopt.training.fed_opt import fed_lora_opt, fedavg_rolora, fed_lora_svd_opt
    from fdaopt.training.fda_opt import fda_rolora_opt, fda_lora_svd_opt
    
    hyperparams = json.loads(config_json)
    method_name = hyperparams.pop('METHOD_NAME')
    run_name = hyperparams['extras']
    
    # Extract the dynamically assigned device
    target_device = hyperparams.pop('TARGET_DEVICE', 'cuda:0') 
    
    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(target_device)
    utils.set_seed(42)
    
    print(f"[{time.strftime('%H:%M:%S')}] Launching {run_name} on {target_device}")
    
    if method_name == "Fed-LoRA": fed_lora_opt(hyperparams)
    elif method_name == "FedAvg-RoLoRA": fedavg_rolora(hyperparams)
    elif method_name == "FDA-RoLoRA": fda_rolora_opt(hyperparams)
    elif method_name == "Fed-LoRA-SVD": fed_lora_svd_opt(hyperparams)
    elif method_name == "FDA-LoRA-SVD": fda_lora_svd_opt(hyperparams)
    else: raise ValueError(f"Unknown method requested: {method_name}")

def _worker_task(run_config, device_queue, log_dir):
    """
    Worker function that claims an available GPU, runs the subprocess, and returns the GPU.
    """
    device = device_queue.get()
    
    try:
        run_config['TARGET_DEVICE'] = device
        run_identifier = run_config['extras']
        
        out_file = os.path.join(log_dir, f"{run_identifier}.out")
        err_file = os.path.join(log_dir, f"{run_identifier}.err")
        
        config_str = json.dumps(run_config)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--run_config", config_str]
        
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        start_time = time.time()
        with open(out_file, 'w') as stdout_f, open(err_file, 'w') as stderr_f:
            subprocess.run(cmd, env=env, stdout=stdout_f, stderr=stderr_f, check=True)
            
        duration = (time.time() - start_time) / 60
        print(f"✅ SUCCESS: {run_identifier} finished on {device} in {duration:.2f} minutes.")
        
    except subprocess.CalledProcessError:
        print(f"❌ CRITICAL ERROR: {run_identifier} crashed on {device}. Check {err_file}.")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {run_identifier} | {e}")
    finally:
        time.sleep(3) 
        device_queue.put(device)

def orchestrate():
    print(f"Starting Parallel Evaluation Matrix. Max Concurrency: {MAX_CONCURRENT_JOBS}")
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    all_runs = []
    
    # Build the Matrix
    for scen_name, scen_cfg in SCENARIOS.items():
        for dist_name, dist_cfg in DISTRIBUTIONS.items():
            for opt_name, opt_cfg in OPTIMIZERS.items():
                for method_name, method_variants in METHODS_MECHANICS.items():
                    
                    # SMART FILTERING
                    if method_name == "FedAvg-RoLoRA" and opt_name == "Advanced-Optimizers":
                        continue
                    
                    # Loop through the specific hyperparameter variants for this method
                    for variant_idx, method_cfg in enumerate(method_variants):
                        
                        run_config = BASE_HYPERPARAMS.copy()
                        run_config.update(scen_cfg)
                        run_config.update(dist_cfg)
                        run_config.update(opt_cfg)
                        run_config.update(method_cfg) 
                        
                        run_config['METHOD_NAME'] = method_name
                        
                        # Append the variant index so outputs don't overwrite each other
                        if len(method_variants) > 1:
                            run_config['extras'] = f"{scen_name}_{dist_name}_{opt_name}_{method_name}_v{variant_idx}"
                        else:
                            run_config['extras'] = f"{scen_name}_{dist_name}_{opt_name}_{method_name}"
                        
                        all_runs.append(run_config)
                    
    # --- ADD THIS LINE TO PRIORITIZE NON-SVD METHODS ---
    all_runs.sort(key=lambda config: "SVD" in config['METHOD_NAME'])
    
    total_runs = len(all_runs)
    print(f"Total configurations scheduled: {total_runs}")
    print(f"Available Devices: {AVAILABLE_DEVICES}\n")
    
    # Initialize the thread-safe queue with available GPUs
    device_queue = queue.Queue()
    for dev in AVAILABLE_DEVICES:
        for _ in range(JOBS_PER_DEVICE):
            device_queue.put(dev)

    # Launch the parallel executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = [executor.submit(_worker_task, run_config, device_queue, log_dir) for run_config in all_runs]
        concurrent.futures.wait(futures)
            
    print("\n🎉 All scheduled experiments have been processed. Metrics are in results/round_metrics/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, default=None, help="Internal use: JSON string of hyperparams.")
    args = parser.parse_args()
    
    if args.run_config:
        execute_run(args.run_config)
    else:
        orchestrate()