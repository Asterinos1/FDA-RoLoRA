import sys
import os

from lol import HF_TOKEN
# Add the parent directory (FDA-LoRA) to sys.path so 'fdaopt' is found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import login

try:
    login(token=HF_TOKEN)
    print("Authentication successful!")
except Exception as e:
    print(f"Auth failed: {e}. Downloads might be throttled.")

import torch
from training.fda_opt import fda_opt, fda_lora_opt, fda_sa_lora_opt, fda_rolora_opt
from training.fed_opt import fed_opt, fed_lora_opt, fed_sa_lora_opt, fed_rolora_opt

# Configuration for a fast "Tiny" run
# TEST_HYPERPARAMS = {
#     'checkpoint': "distilbert-base-uncased", # Extremely small model for testing
#     # 'checkpoint': "prajjwal1/bert-tiny", # Extremely small model for testing
#     # 'use_fast_tokenizer': False,
#     'use_fast_tokenizer': True, # Fast Tokinizer works for this model!!!
#     'ds_path': "glue",                  # Using standard GLUE dataset
#     'ds_name': "sst2",                  # Sentiment Analysis (binary)
#     'num_labels': 2,
#     'num_clients': 2,                # Just 2 clients for the test
#     'clients_per_round': 2,
#     'total_rounds': 1,                  # Just 1 round for the test
#     'local_epochs': 1,
#     'max_epochs': 1,                    # Max epochs for FDA loop set to 1 for quick test
#     'batch_size': 64,                   # Larger batch size for faster processing
#     'alpha': 1.0,                       # Data distribution parameter
#     'server_opt_name': 'Adam',
#     'server_opt_lr': 1e-3,
#     'client_opt_name': 'SGD',
#     'client_opt_lr': 1e-2,
#     'theta': 0.1,                       # FDA threshold
#     'threshold': False,
#     'lora_r': 4,                        # Tiny LoRA rank
#     'lora_alpha': 8,
#     'target_modules': ["query", "value"] # Standard for BERT
# }

TEST_HYPERPARAMS = {
    'checkpoint': "hf-internal-testing/tiny-random-bert", # Microscopic testing model
    'use_fast_tokenizer': True,
    'ds_path': "glue",                  
    'ds_name': "sst2",                  
    'num_labels': 2,
    'num_clients': 2,                    
    'clients_per_round': 2,
    'total_rounds': 1,                   
    'local_epochs': 1,
    'max_epochs': 1,                     
    'batch_size': 64,                   
    'num_workers': 0,     
    'pin_memory': False,    
    'alpha': 1.0,                       
    'server_opt_name': 'Adam',
    'server_opt_lr': 1e-3,
    'client_opt_name': 'SGD',
    'client_opt_lr': 1e-2,
    'theta': 0.1,                       
    'threshold': False,
    'lora_r': 4,                        
    'lora_alpha': 8,
    # 'target_modules': ["query", "value"] 
    'target_modules': ("query", "value") ,
    # Toggle this to True or False to test the respective logic branches
    'include_b_in_variance': True,
    'max_consecutive_rounds': 2
}

def run_test(name, func):
    print(f"\n{'='*20}")
    print(f"STARTING TEST: {name}")
    print(f"{'='*20}")
    
    # 1. Create a shallow copy of the global hyperparameters
    current_hyperparams = TEST_HYPERPARAMS.copy()
    
    # 2. Inject the method name so MetricsHandler generates a unique CSV per test
    current_hyperparams['method'] = name
    
    try:
        # 3. Pass the modified copy to the training function
        func(current_hyperparams)
        print(f"\nSUCCESS: {name} completed successfully.")
    except Exception as e:
        print(f"\nFAILED: {name} encountered an error: {e}")

if __name__ == "__main__":
    print(f"System Check: Using {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Run tests sequentially
    #basic concept tests
    run_test("Fed-Opt", fed_opt)
    run_test("FedLoRA-Opt", fed_lora_opt)
    run_test("FDA-Opt", fda_opt)
    run_test("FDA-LoRA-Opt", fda_lora_opt)
    #fedsalora tests
    run_test("FedSA-LoRA-Opt", fed_sa_lora_opt)
    run_test(f"FDA-SA-LoRA-Opt, B included from variance", lambda hp: fda_sa_lora_opt({**hp, 'include_b_in_variance': False}))
    
    # B should be excluded from variance calculation but I will leave this inside just for seeing
    # how it affects the logs and metrics. 
    run_test(f"FDA-SA-LoRA-Opt, B excluded from variance", lambda hp: fda_sa_lora_opt({**hp, 'include_b_in_variance': True}))

    #rolora tests
    run_test("Fed-RoLoRA-Opt", fed_rolora_opt)
    run_test("FDA-RoLoRA-Opt", fda_rolora_opt)

    print("\nAll tests completed. Check your console for round metrics and variance logs.")