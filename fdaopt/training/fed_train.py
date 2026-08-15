"""
Client-side training iteration and local epoch optimization routines.

Executes local gradient descent steps for sampled federated clients,
managing client-specific parameter state restoration, batch iteration,
loss computation, and prediction accuracy tracking.
"""

from typing import List, Dict, Tuple
import torch
from fdaopt import utils
from fdaopt.models.ops import copy_parameters


def federated_training_step(
    model: torch.nn.Module,
    train_params: List[torch.nn.Parameter],
    client_train_params_dict: Dict[int, List[torch.Tensor]],
    client_opt: torch.optim.Optimizer,
    fed_ds
) -> float:
    """
    Execute a single mini-batch training step across all clients in the parameter dictionary.

    Args:
        model (torch.nn.Module): Shared neural network model backbone.
        train_params (list of torch.nn.Parameter): Trainable model parameter references.
        client_train_params_dict (dict): Map of client IDs to their persistent parameter tensors.
        client_opt (torch.optim.Optimizer): Client-side optimizer instance.
        fed_ds (FederatedDataset): Batch generator provider for each client.

    Returns:
        float: Average training loss across the updated clients.
    """
    model.train()
    training_loss = 0.0
    num_clients = len(client_train_params_dict)

    for client_id in client_train_params_dict.keys():
        # Restore client-specific state into the model parameters
        copy_parameters(
            from_params=client_train_params_dict[client_id],
            to_params=train_params
        )

        batch = fed_ds.next_client_batch(client_id)
        batch = {k: v.to(utils.DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        client_opt.step()
        client_opt.zero_grad()

        # Save updated client state back into the dictionary
        copy_parameters(
            from_params=train_params,
            to_params=client_train_params_dict[client_id]
        )
        training_loss += loss.item()

    return training_loss / num_clients


def client_round_train(
    model: torch.nn.Module,
    train_params: List[torch.nn.Parameter],
    client_train_params_dict: Dict[int, List[torch.Tensor]],
    client_opt: torch.optim.Optimizer,
    fed_ds,
    client_id: int,
    round_steps: int
) -> Tuple[float, float]:
    """
    Train a single client's model locally for a specified number of mini-batch steps.

    Args:
        model (torch.nn.Module): Shared model backbone.
        train_params (list of torch.nn.Parameter): Trainable parameters of the model.
        client_train_params_dict (dict): Map of client IDs to persistent parameter tensors.
        client_opt (torch.optim.Optimizer): Client optimizer for local updates.
        fed_ds (FederatedDataset): Batch generator provider for client datasets.
        client_id (int): ID of the client being trained.
        round_steps (int): Total number of mini-batch gradient steps to perform.

    Returns:
        tuple: (average_training_loss, training_accuracy) over the local round.
    """
    model.train()

    # Load client's starting parameter state into the active model
    copy_parameters(
        from_params=client_train_params_dict[client_id],
        to_params=train_params
    )

    training_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for _ in range(round_steps):
        batch = fed_ds.next_client_batch(client_id)
        batch = {k: v.to(utils.DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == batch["labels"]).sum().item()
        total_predictions += batch["labels"].size(0)

        loss.backward()
        client_opt.step()
        client_opt.zero_grad()

        training_loss += loss.item()

    # Save updated parameters back to the client's dictionary entry
    copy_parameters(
        from_params=train_params,
        to_params=client_train_params_dict[client_id]
    )

    avg_loss = training_loss / round_steps if round_steps > 0 else 0.0
    train_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return avg_loss, train_acc
