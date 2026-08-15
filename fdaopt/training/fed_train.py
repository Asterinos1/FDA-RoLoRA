from fdaopt import utils
from fdaopt.models.ops import copy_parameters
import torch
def federated_training_step(model, train_params, client_train_params_dict, client_opt, fed_ds):
    """
    Performs a single federated training step.
    This function trains each client starting with its client-specific model parameters, updates them
    with the client optimizer and client-specific batch, and returns. At the end, client_train_params_dict
    have the updated client-specific parameters (after training on their specific batch).
    Args:
        model (torch.nn.Module): The model to be trained.
        train_params (list): A list of trainable parameters of the model (referencing `model` above).
        client_train_params_dict (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
        client_opt (torch.optim.Optimizer): The optimizer for updating the model parameters.
        fed_ds (FederatedDataset): An object that provides the batches for each client.
    Returns:
        float: The average training loss across all clients for this federated training step.
    """
    model.train()
    training_loss = 0.0
    num_clients = len(client_train_params_dict)
    for client_id in client_train_params_dict.keys():
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
        copy_parameters(
            from_params=train_params,
            to_params=client_train_params_dict[client_id]
        )
        client_opt.zero_grad()
        training_loss += loss.item()
    training_loss = training_loss / num_clients
    return training_loss
def client_round_train(model, train_params, client_train_params_dict, client_opt, fed_ds, client_id, round_steps):
    """
    Trains a client-specific model for a specified number of steps (a round).
    At the end, the updated client-specific parameters are stored back in `client_train_params_dict`.
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_params (list): A list of trainable parameters of the model (referencing `model` above).
        client_train_params_dict (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
        client_opt (torch.optim.Optimizer): The optimizer for updating the model parameters.
        fed_ds (FederatedDataset): An object that provides the batches for each client.
        client_id (int): The ID of the client to be trained.
        round_steps (int): The number of training steps (mini-batches) for this client's round.
    Returns:
        float: The average training loss for the client over the specified number of steps.
    """
    model.train()
    copy_parameters(
        from_params=client_train_params_dict[client_id],
        to_params=train_params
    )
    training_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for step in range(round_steps):
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
    copy_parameters(
        from_params=train_params,
        to_params=client_train_params_dict[client_id]
    )
    training_loss = training_loss / round_steps
    train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return training_loss, train_accuracy

