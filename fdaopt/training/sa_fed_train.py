from fdaopt.utils import DEVICE
from fdaopt.models.ops import copy_parameters
from tqdm import tqdm

def sa_client_round_train(
    model, 
    shared_params, 
    local_params, 
    client_shared_params_dict, 
    client_local_params_dict, 
    client_opt, 
    fed_ds, 
    client_id, 
    round_steps
):
    """Custom client round training for shared A matrice and B private variation."""
    model.train()

    # Load both shared (A) and local (B) parameters for this specific client
    copy_parameters(
        from_params=client_shared_params_dict[client_id],
        to_params=shared_params
    )
    copy_parameters(
        from_params=client_local_params_dict[client_id],
        to_params=local_params
    )

    training_loss = 0.0

    pbar_local = tqdm(range(round_steps), desc=f"  Client {client_id} Steps", leave=False)
    for step in pbar_local:
        batch = fed_ds.next_client_batch(client_id)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        client_opt.step()
        client_opt.zero_grad()

        training_loss += loss.item()
        if (step + 1) % 10 == 0:
            pbar_local.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    # Save the updated parameters back to their respective client states
    copy_parameters(
        from_params=shared_params,
        to_params=client_shared_params_dict[client_id]
    )
    copy_parameters(
        from_params=local_params,
        to_params=client_local_params_dict[client_id]
    )

    return training_loss / round_steps