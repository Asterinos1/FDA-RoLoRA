"""
Command-line generator for federated learning hyperparameter configuration files.

Generates serialized JSON configurations specifying model architectures,
optimizer settings, client Dirichlet partition parameters, and FDA synchronization bounds.
"""

import json
import os
import argparse
import ast
from typing import List, Dict, Any, Optional

script_directory = os.path.dirname(os.path.abspath(__file__))
HYPERPARAM_DIR = os.path.normpath(os.path.join(script_directory, '../../hyperparameters'))


def get_opt_args(opt_args: Optional[List[str]], where: str) -> Dict[str, Any]:
    """
    Parse optional key=value command-line strings into typed optimizer arguments.

    Args:
        opt_args (Optional[List[str]]): List of raw strings formatted as 'key=value'.
        where (str): Scope prefix, either 'server_opt' or 'client_opt'.

    Returns:
        dict: Parsed dictionary mapping '{where}_{key}' to evaluated Python literals.
    """
    opt_params = {}
    if opt_args:
        for arg in opt_args:
            key, value = arg.split('=', 1)
            opt_params[f"{where}_{key}"] = ast.literal_eval(value)
    return opt_params


def get_next_filename() -> str:
    """
    Determine the next sequential integer filename for saving a new hyperparameter config.

    Returns:
        str: Base filename string without extension (e.g. '0', '1', ...).
    """
    os.makedirs(HYPERPARAM_DIR, exist_ok=True)
    files = [f for f in os.listdir(HYPERPARAM_DIR) if f.endswith('.json')]
    return f"{len(files)}"


def create_hyperparameters(args: argparse.Namespace) -> str:
    """
    Assemble and serialize a complete hyperparameter dictionary from parsed CLI arguments.

    Args:
        args (argparse.Namespace): Parsed command-line options.

    Returns:
        str: Path to the generated JSON configuration file.
    """
    hyperparams = {
        'checkpoint': args.checkpoint,
        'ds_path': args.ds_path,
        'ds_name': args.ds_name,
        'num_labels': args.num_labels,
        'num_clients': args.num_clients,
        'clients_per_round': args.clients_per_round,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'local_epochs': args.local_epochs,
        'max_epochs': args.max_epochs,
        'total_rounds': args.total_rounds,
        'fda': args.fda,
        'threshold': args.threshold,
        'theta': args.theta,
        'train_head': not args.disable_train_head,
        'extras': args.extras
    }

    server_opt_args = get_opt_args(args.server_opt_args, "server_opt")
    client_opt_args = get_opt_args(args.client_opt_args, "client_opt")

    hyperparams = {
        **hyperparams,
        'server_opt_name': args.server_opt_name,
        **server_opt_args,
        'client_opt_name': args.client_opt_name,
        **client_opt_args
    }

    filename = get_next_filename()
    out_path = os.path.join(HYPERPARAM_DIR, f'{filename}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=2)

    print(f"OK! Created hyperparameters in {filename}.json.")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create FDA / Federated Learning hyperparameter configuration JSON.")
    parser.add_argument('--checkpoint', type=str, help="Model backbone checkpoint (e.g. 'roberta-base').", default='roberta-base')
    parser.add_argument('--ds_path', type=str, help="Dataset identifier or repository path.", default='glue')
    parser.add_argument('--ds_name', type=str, help="Dataset task subset name (e.g. 'mrpc', 'sst2').", default='mrpc')
    parser.add_argument('--num_labels', type=int, help="Number of classification target labels.", default=2)
    parser.add_argument('--num_clients', type=int, help="Total number of partitioned federated clients.", default=100)
    parser.add_argument('--clients_per_round', type=int, help="Number of clients randomly sampled per communication round.", default=10)
    parser.add_argument('--alpha', type=float, help="Concentration parameter for Dirichlet non-IID data partitioning.", default=1.0)
    parser.add_argument('--batch_size', type=int, help="Mini-batch size for local client training.", default=8)
    parser.add_argument('--local_epochs', type=int, help="Fixed local epochs per round in FedOpt / FedAvg.", default=1)
    parser.add_argument('--max_epochs', type=int, help="Forced maximum local epoch bound in FDA.", default=10)
    parser.add_argument('--total_rounds', type=int, help="Maximum communication rounds for the simulation.", default=1000)
    parser.add_argument('--fda', action='store_true', help="Enable Fast Dynamic Aggregation (FDA) variance-adaptive training.")
    parser.add_argument('--threshold', action='store_true', help="Enable metric threshold early stopping.")
    parser.add_argument('--theta', type=float, help="Initial variance threshold for FDA synchronization.", default=0.0)
    parser.add_argument('--server_opt_name', type=str, help="PyTorch optimizer class for server aggregation.", required=True)
    parser.add_argument('--client_opt_name', type=str, help="PyTorch optimizer class for client local training.", required=True)
    parser.add_argument('--server_opt_args', nargs='*', help="Extra server optimizer kwargs (e.g. lr=0.001 betas='(0.9, 0.999)').")
    parser.add_argument('--client_opt_args', nargs='*', help="Extra client optimizer kwargs (e.g. lr=0.001).")
    parser.add_argument('--extras', type=str, help="Experiment label prefix or metadata.", default='')
    parser.add_argument('--disable_train_head', action='store_true', help="Freeze classification head weights during training.")
    create_hyperparameters(parser.parse_args())
