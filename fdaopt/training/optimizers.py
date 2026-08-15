"""
Optimizer factory and parameter grouping utilities for federated training.

Provides routines to dynamically instantiate PyTorch server and client optimizers
with support for differential learning rates between low-rank adapter matrices (LoRA)
and dense classification head layers.
"""

from typing import List, Dict, Any, Tuple, Optional, Type
import torch
from fdaopt.utils import torch as utils_torch


def get_opt_hyperparams(where: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract optimizer keyword arguments from the experiment hyperparameter dictionary.

    Args:
        where (str): Scope prefix, either 'server' or 'client'.
        hyperparams (dict): Experiment configuration dictionary containing keys like
            'server_opt_lr', 'client_opt_momentum', etc.

    Returns:
        dict: Clean dictionary of kwargs stripped of scope prefixes (e.g. {'lr': 0.01}).
    """
    opt_dict = {}
    where_opt = f"{where}_opt"
    for k, v in hyperparams.items():
        if where_opt in k:
            param = k.replace(f"{where_opt}_", '')
            if param != 'name':
                opt_dict[param] = v
    return opt_dict


def get_optimizer_class(optimizer_name: str) -> Type[torch.optim.Optimizer]:
    """
    Dynamically resolve and return a PyTorch optimizer class by string name.

    Args:
        optimizer_name (str): Name of the optimizer class (e.g. 'AdamW', 'SGD', 'Adam').

    Returns:
        Type[torch.optim.Optimizer]: The corresponding class from `torch.optim`.
    """
    optimizer_class = getattr(torch.optim, optimizer_name)
    return optimizer_class


def server_client_optimizers(
    train_params: List[torch.nn.Parameter],
    hyperparams: Dict[str, Any],
    num_lora_params: Optional[int] = None
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Construct separate server and client optimizer instances based on experiment configuration.

    If `num_lora_params` is provided and smaller than `len(train_params)`, differential
    learning rates are applied to client optimization: LoRA parameters receive the base
    client learning rate, while non-LoRA parameters (e.g. classification head) receive
    a 10x smaller learning rate (`0.1 * base_lr`) to maintain head stability.

    Args:
        train_params (list of torch.nn.Parameter): Trainable model parameters.
        hyperparams (dict): Hyperparameter dictionary containing optimizer specifications.
        num_lora_params (Optional[int]): Number of initial parameters belonging to LoRA matrices.

    Returns:
        tuple: (server_optimizer, client_optimizer)
    """
    server_opt_class = get_optimizer_class(hyperparams['server_opt_name'])
    client_opt_class = get_optimizer_class(hyperparams['client_opt_name'])
    server_kwargs = get_opt_hyperparams('server', hyperparams)
    client_kwargs = get_opt_hyperparams('client', hyperparams)

    # Server optimizer updates global aggregated parameters
    server_opt = server_opt_class(train_params, **server_kwargs)

    # Client optimizer with optional differential LR for LoRA vs. Dense head
    if num_lora_params is not None and num_lora_params < len(train_params):
        client_lr = client_kwargs.pop('lr', 0.01)
        client_groups = [
            {'params': train_params[:num_lora_params], 'lr': client_lr},
            {'params': train_params[num_lora_params:], 'lr': client_lr * 0.1}
        ]
        client_opt = client_opt_class(client_groups, **client_kwargs)
    else:
        client_opt = client_opt_class(train_params, **client_kwargs)

    return server_opt, client_opt
