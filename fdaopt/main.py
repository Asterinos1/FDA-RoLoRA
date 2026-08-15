"""
Main simulation execution entry point for FDA-LoRA experiments.

Parses command-line execution parameters, initializes global device and seed state,
loads the targeted hyperparameter JSON specification, and dispatches execution to
either standard FedOpt or Fast Dynamic Aggregation (FDA) optimization loops.
"""

import pprint
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FDA-LoRA Federated Simulation Runner")
    parser.add_argument('--filename', type=str, required=True, help="Hyperparameter JSON filename (e.g. '0.json').")
    parser.add_argument('--device', type=str, default='cuda:0', help="PyTorch compute device identifier (e.g. 'cuda:0', 'cpu').")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility across clients and model init.")
    args = parser.parse_args()

    from fdaopt import utils
    utils.set_seed(args.seed)
    utils.set_device(args.device)

    from fdaopt.training.fed_opt import fed_opt
    from fdaopt.training.fda_opt import fda_opt
    from fdaopt.miscellaneous.parameter_tools import get_hyperparameters

    # Load experiment configuration
    hyperparams = get_hyperparameters(args.filename)
    print(f"Using device: {utils.DEVICE}")
    pprint.pprint(hyperparams, sort_dicts=False)
    print()

    # Route execution to appropriate optimization strategy
    if hyperparams.get('fda', False):
        fda_opt(hyperparams)
    else:
        fed_opt(hyperparams)

    print("Finished Testing!")
