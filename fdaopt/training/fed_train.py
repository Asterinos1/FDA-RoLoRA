from fdaopt.utils import DEVICE
from fdaopt.models.ops import copy_parameters
from tqdm import tqdm

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

    # Set the model to training mode
    model.train()

    training_loss = 0.0
    num_clients = len(client_train_params_dict)

    # Iterate over each client with tqdm
    pbar_step = tqdm(client_train_params_dict.keys(), desc="Federated Step (Clients)", leave=False)
    for client_id in pbar_step:
        # Copy client-specific parameters to the model's parameters
        copy_parameters(
            from_params=client_train_params_dict[client_id],
            to_params=train_params
        )

        # Get the next batch of data for the current client
        batch = fed_ds.next_client_batch(client_id)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Perform a forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        client_opt.step()

        # Copy the updated model parameters back to the client's parameter set
        copy_parameters(
            from_params=train_params,
            to_params=client_train_params_dict[client_id]
        )

        # Zero the gradients before the next backward pass
        client_opt.zero_grad()

        # Accumulate the loss
        training_loss += loss.item()
        pbar_step.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate the average loss
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

    # Set the model to training mode
    model.train()

    # Copy client-specific parameters to the model's parameters
    copy_parameters(
        from_params=client_train_params_dict[client_id],
        to_params=train_params
    )

    training_loss = 0.0

    # Added tqdm for local client steps
    pbar_local = tqdm(range(round_steps), desc=f"  Client {client_id} Steps", leave=False)
    for step in pbar_local:
        # Get the next batch of data for the current client
        batch = fed_ds.next_client_batch(client_id)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Perform a forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        client_opt.step()

        # Zero the gradients before the next backward pass
        client_opt.zero_grad()

        # Accumulate the loss
        training_loss += loss.item()
        if (step + 1) % 10 == 0: # Update postfix every 10 steps to reduce overhead
            pbar_local.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    # Copy the updated model parameters back to the client's parameter set
    copy_parameters(
        from_params=train_params,
        to_params=client_train_params_dict[client_id]
    )

    # Calculate the average loss
    training_loss = training_loss / round_steps

    return training_loss