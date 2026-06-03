import os
import sys
import time
import subprocess
import argparse
import json
import concurrent.futures
import queue

AVAILABLE_DEVICES = ['cuda:0', 'cuda:1']
JOBS_PER_DEVICE = 2
MAX_CONCURRENT_JOBS = len(AVAILABLE_DEVICES) * JOBS_PER_DEVICE

BASE_HYPERPARAMS = {
    'ds_path': "glue",
    'total_rounds': 80,
    'batch_size': 16,
    'target_modules': ("query", "value"),
    'num_workers': 4,
    'pin_memory': True,
    'threshold': True
}

CHECKPOINTS = [
    "roberta-large"
]

DATASETS = [
    # {'ds_name': 'rte', 'num_labels': 2, 'local_epochs': 1},
    {'ds_name': 'mrpc', 'num_labels': 2, 'local_epochs': 2},
    {'ds_name': 'sst2', 'num_labels': 2, 'local_epochs': 5}
]

LORA_CONFIGS = [
    {'lora_r': 8, 'lora_alpha': 16},
    {'lora_r': 16, 'lora_alpha': 32}
]

SCENARIOS = {
    "Partial-Participation": {
        'num_clients': 100,
        'clients_per_round': 10,
    },
    "Full-Participation": {
        'num_clients': 10,
        'clients_per_round': 10,
    }
}

DISTRIBUTIONS = {
    "IID": {'alpha': 10.0},
    "Non-IID": {'alpha': 0.5}
}

METHODS_MECHANICS = {
    "FedAvg-RoLoRA": [
        {
            'server_opt_name': 'SGD',
            'server_opt_lr': 1.0,
            'client_opt_name': 'SGD',
            'client_opt_lr': 0.01
        }
    ],
    "FDA-RoLoRA-DualTrack": [
        {
            'e_min': 2, 'max_epochs': 10,
            'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3,
            'client_opt_name': 'SGD', 'client_opt_lr': 0.01
        }
    ],
    "FDA-RoLoRA-DualTrack-EMA-Standard": [
        {
            'e_min': 2, 'max_epochs': 10, 'rho': 0.8,
            'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3,
            'client_opt_name': 'SGD', 'client_opt_lr': 0.01
        }
    ],
    "FDA-RoLoRA-DualTrack-EMA-Pure": [
        {
            'e_min': 2, 'max_epochs': 10, 'rho': 0.8, 'alpha_ema': 1.15,
            'server_opt_name': 'AdamW', 'server_opt_lr': 1e-3,
            'client_opt_name': 'SGD', 'client_opt_lr': 0.01
        }
    ]
}

def execute_run(config_json):
    import torch
    from fdaopt import utils
    from fdaopt.training.fed_opt import fedavg_rolora
    from fdaopt.training.fda_opt import (
        fda_rolora_dual_track_opt,
        fda_rolora_dual_track_ema_standard_opt,
        fda_rolora_dual_track_ema_pure_opt
    )
    
    hyperparams = json.loads(config_json)
    method_name = hyperparams.pop('METHOD_NAME')
    run_name = hyperparams['extras']
    target_device = hyperparams.pop('TARGET_DEVICE', 'cuda:0')
    
    utils.DEVICE_RAM_PROGRAM = 'performance'
    utils.set_device(target_device)
    utils.set_seed(42)
    
    print(f"[{time.strftime('%H:%M:%S')}] Launching {run_name} on {target_device}")
    
    if method_name == "FedAvg-RoLoRA":
        fedavg_rolora(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack":
        fda_rolora_dual_track_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-EMA-Standard":
        fda_rolora_dual_track_ema_standard_opt(hyperparams)
    elif method_name == "FDA-RoLoRA-DualTrack-EMA-Pure":
        fda_rolora_dual_track_ema_pure_opt(hyperparams)
    else:
        raise ValueError(f"Unknown method requested: {method_name}")

def _worker_task(run_config, device_queue, log_dir):
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
        print(f"SUCCESS: {run_identifier} finished on {device} in {duration:.2f} minutes.")
        
    except subprocess.CalledProcessError:
        print(f"CRITICAL ERROR: {run_identifier} crashed on {device}. Check {err_file}.")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {run_identifier} | {e}")
    finally:
        time.sleep(3)
        device_queue.put(device)

def orchestrate():
    print(f"Starting Parallel Evaluation Matrix. Max Concurrency: {MAX_CONCURRENT_JOBS}")
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    all_runs = []
    
    for chkpt in CHECKPOINTS:
        for ds in DATASETS:
            for lora_cfg in LORA_CONFIGS:
                for scen_name, scen_cfg in SCENARIOS.items():
                    for dist_name, dist_cfg in DISTRIBUTIONS.items():
                        for method_name, method_variants in METHODS_MECHANICS.items():
                            for variant_idx, method_cfg in enumerate(method_variants):
                                run_config = BASE_HYPERPARAMS.copy()
                                run_config['checkpoint'] = chkpt
                                run_config.update(ds)
                                run_config.update(lora_cfg)
                                run_config.update(scen_cfg)
                                run_config.update(dist_cfg)
                                run_config.update(method_cfg)
                                run_config['METHOD_NAME'] = method_name
                                
                                ds_name = ds['ds_name']
                                r_val = lora_cfg['lora_r']
                                chkpt_short = chkpt.split('/')[-1]
                                
                                if len(method_variants) > 1:
                                    run_config['extras'] = f"NEW1_{chkpt_short}_{ds_name}_r{r_val}_{scen_name}_{dist_name}_{method_name}_v{variant_idx}"
                                else:
                                    run_config['extras'] = f"NEW1_{chkpt_short}_{ds_name}_r{r_val}_{scen_name}_{dist_name}_{method_name}"
                                    
                                all_runs.append(run_config)
                                
    total_runs = len(all_runs)
    print(f"Total configurations scheduled: {total_runs}")
    print(f"Available Devices: {AVAILABLE_DEVICES}\n")
    
    device_queue = queue.Queue()
    for dev in AVAILABLE_DEVICES:
        for _ in range(JOBS_PER_DEVICE):
            device_queue.put(dev)
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = [executor.submit(_worker_task, run_config, device_queue, log_dir) for run_config in all_runs]
        concurrent.futures.wait(futures)
        
    print("\nAll scheduled experiments have been processed. Metrics are in results/round_metrics/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, default=None, help="Internal use: JSON string of hyperparams.")
    args = parser.parse_args()
    if args.run_config:
        execute_run(args.run_config)
    else:
        orchestrate()