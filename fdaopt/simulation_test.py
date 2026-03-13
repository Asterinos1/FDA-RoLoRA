import time
import logging

from fdaopt.training.fed_opt import fed_opt
from fdaopt.training.fda_opt import fda_opt
from fdaopt.training.fed_lora_opt import fed_lora_opt
from fdaopt.training.fda_lora_opt import fda_lora_opt
from fdaopt.training.fed_sa_lora_opt import fed_sa_lora_opt
from fdaopt.training.fda_sa_lora_opt import fda_sa_lora_opt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def run_simulation(method_name, opt_function, config):
    """
    Executes a federated learning simulation for a given method and configuration.
    """
    logger.info(f"--- Starting Simulation: {method_name} ---")
    start_time = time.time()
    
    try:
        opt_function(config)
    except Exception as e:
        logger.error(f"Simulation failed for {method_name} with error: {str(e)}")
        raise
        
    elapsed_time = time.time() - start_time
    logger.info(f"--- Completed {method_name} in {elapsed_time:.2f} seconds ---\n")

def main():
    """
    Main execution block defining hyperparameters and triggering the simulations.
    """
    
    # Common parameters across all experiments
    common_params = {
        'checkpoint': 'roberta-base',
        'ds_path': 'glue',
        'ds_name': 'mrpc',
        'num_labels': 2,
        'num_clients': 10,
        'clients_per_round': 4,
        'total_rounds': 20,
        'local_epochs': 3,
        'batch_size': 16,
        'alpha': 1.0,           
        'lr_server': 1.0,       
        'weight_decay': 0.01,
        'num_workers': 2,
        'pin_memory': True,
        'threshold': True       
    }

    # Configuration for Full Fine-Tuning methods
    full_ft_config = common_params.copy()
    full_ft_config.update({
        'lr_client': 2e-5, # Lower learning rate required for full parameter tuning
        'theta': 0.05,
        'max_epochs': 10
    })

    # Configuration for LoRA-based methods
    lora_config = common_params.copy()
    lora_config.update({
        'lora_r': 8,
        'lora_alpha': 16,
        'target_modules': ['query', 'value'], 
        'lr_client': 2e-4, # Higher learning rate typical for PEFT
        'theta': 0.05,
        'max_epochs': 10
    })

    # FDA-SA-LoRA specific configurations
    fda_sa_lora_a_only = lora_config.copy()
    fda_sa_lora_a_only['include_b_in_variance'] = False

    fda_sa_lora_a_and_b = lora_config.copy()
    fda_sa_lora_a_and_b['include_b_in_variance'] = True

    logger.info("Initializing Federated Simulations Pipeline...")

    # 1. Full Fine-Tuning Baselines
    run_simulation("FedAvg (Full Fine-Tuning)", fed_opt, full_ft_config)
    run_simulation("FDA (Full Fine-Tuning)", fda_opt, full_ft_config)

    # 2. Standard LoRA Baselines
    run_simulation("Fed-LoRA", fed_lora_opt, lora_config)
    run_simulation("FDA-LoRA", fda_lora_opt, lora_config)

    # 3. Shared-A LoRA Methods
    run_simulation("Fed-SA-LoRA", fed_sa_lora_opt, lora_config)
    run_simulation("FDA-SA-LoRA (A-Only Variance)", fda_sa_lora_opt, fda_sa_lora_a_only)
    run_simulation("FDA-SA-LoRA (A+B Variance)", fda_sa_lora_opt, fda_sa_lora_a_and_b)

if __name__ == "__main__":
    main()