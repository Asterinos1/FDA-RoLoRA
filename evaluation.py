import os
import sys
import time
import subprocess
import argparse

# ==========================================
# GRAND COMPARISON HYPERPARAMETERS
# ==========================================
BASE_HYPERPARAMS = {
    'checkpoint': "roberta-base", 
    'ds_path': "glue",
    'ds_name': "mrpc",
    'num_labels': 2,
    'num_clients': 50,
    'clients_per_round': 10,
    'total_rounds': 80,
    'local_epochs': 1,
    'max_epochs': 10,
    'e_min': 2,
    'batch_size': 16,
    'num_workers': 4,
    'pin_memory': True,
    'server_opt_name': 'AdamW',
    'server_opt_lr': 1e-3,
    'client_opt_name': 'SGD',
    'client_opt_lr': 0.01,
    'lora_r': 8,
    'lora_alpha': 16,
    'target_modules': ("query", "value"),
    'alpha_theta': 0.01,
    'gamma': 0.3,
    'threshold': False  # Set to True if you want early stopping based on ops.py limits
}

ALPHAS = {
    "Ideal": 10.0,
    "Semi-Ideal": 1.0,
    "Non-Ideal": 0.1
}

METHODS = [
    "Fed-LoRA", 
    "FedAvg-RoLoRA", 
    "FDA-RoLoRA", 
    "Fed-LoRA-SVD", 
    "FDA-LoRA-SVD"
]

TARGET_DEVICE = 'cuda:1'

# ==========================================
# CHILD PROCESS EXECUTION LOGIC
# ==========================================
def execute_run(method_name, alpha):
    """
    This runs inside the isolated child process. It imports PyTorch, 
    allocates memory on cuda:1, runs the specific method, and then dies, 
    completely returning all VRAM to the OS.
    """
    # Import locally to prevent the parent orchestrator from binding to the GPU
    import torch
    from fdaopt import utils
    from fdaopt.training.fed_opt import fed_lora_opt, fedavg_rolora, fed_lora_svd_opt
    from fdaopt.training.fda_opt import fda_rolora_opt, fda_lora_svd_opt

    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(TARGET_DEVICE)
    utils.set_seed(42)

    hyperparams = BASE_HYPERPARAMS.copy()
    hyperparams['alpha'] = alpha
    hyperparams['extras'] = f"THESIS_{method_name}_Alpha_{alpha}"

    print(f"[{time.strftime('%H:%M:%S')}] Launching {method_name} with Alpha={alpha} on {TARGET_DEVICE}")

    if method_name == "Fed-LoRA":
        fed_lora_opt(hyperparams)
    elif method_name == "FedAvg-RoLoRA":
        fedavg_rolora(hyperparams)
    elif method_name == "FDA-RoLoRA":
        fda_rolora_opt(hyperparams)
    elif method_name == "Fed-LoRA-SVD":
        fed_lora_svd_opt(hyperparams)
    elif method_name == "FDA-LoRA-SVD":
        fda_lora_svd_opt(hyperparams)
    else:
        raise ValueError(f"Unknown method requested: {method_name}")

# ==========================================
# PARENT ORCHESTRATOR LOGIC
# ==========================================
def orchestrate():
    """
    The main loop. Spawns isolated processes for every scenario.
    If a process OOMs (e.g., SVD heavy lifting on cuda:1), the parent catches it,
    logs the failure, and proceeds to the next experiment seamlessly.
    """
    print(f"Starting Grand Comparison Orchestrator. Target: {TARGET_DEVICE}")
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    total_runs = len(ALPHAS) * len(METHODS)
    current_run = 1

    for scenario_name, alpha in ALPHAS.items():
        for method in METHODS:
            print(f"\n--- Progress: {current_run}/{total_runs} | {method} ({scenario_name}, Alpha={alpha}) ---")
            
            out_file = os.path.join(log_dir, f"{method}_Alpha{alpha}.out")
            err_file = os.path.join(log_dir, f"{method}_Alpha{alpha}.err")
            
            # Formulate the subprocess command to call this exact script again, but as a worker
            cmd = [
                sys.executable, "-u", os.path.abspath(__file__), 
                "--run_method", method, 
                "--alpha", str(alpha)
            ]
            
            # Enforce PyTorch memory fragmentation mitigation for the 20.5GB available on cuda:1
            env = os.environ.copy()
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            start_time = time.time()
            try:
                with open(out_file, 'w') as stdout_f, open(err_file, 'w') as stderr_f:
                    subprocess.run(cmd, env=env, stdout=stdout_f, stderr=stderr_f, check=True)
                duration = (time.time() - start_time) / 60
                print(f"SUCCESS: Finished in {duration:.2f} minutes.")
            except subprocess.CalledProcessError:
                print(f"CRITICAL ERROR: Process crashed (likely OOM). Check {err_file} for stack trace.")
            except Exception as e:
                print(f"UNEXPECTED ERROR: {e}")
            finally:
                # Wait for OS to completely reclaim VRAM before starting the next heavy iteration
                time.sleep(5)
                current_run += 1

    print("\n🎉 All scheduled experiments have been processed. You can find the metrics in results/round_metrics/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_method', type=str, default=None, help="Internal use: specifies the method to run.")
    parser.add_argument('--alpha', type=float, default=None, help="Internal use: specifies the alpha value.")
    args = parser.parse_args()

    # Routing logic: if arguments are passed, act as the worker. If not, act as the orchestrator.
    if args.run_method and args.alpha is not None:
        execute_run(args.run_method, args.alpha)
    else:
        orchestrate()