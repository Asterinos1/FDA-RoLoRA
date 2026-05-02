from fdaopt import utils
import torch
import evaluate
import math

def get_save_device():
    if utils.DEVICE_RAM_PROGRAM == 'performance':
        return utils.DEVICE
    elif utils.DEVICE_RAM_PROGRAM == 'moderate' or utils.DEVICE_RAM_PROGRAM == 'low':
        return 'cpu'
    return 'cpu'

@torch.no_grad
def copy_parameters(from_params, to_params):
    """
    Copies the values from one set of parameters to another.
    This function operates in-place and modifies the `to_parameters` directly.
    The @torch.no_grad() decorator ensures that this operation is not tracked
    by autograd, preventing unnecessary computation and memory usage.
    Args:
        from_params (list of torch.nn.Parameter): An iterable of source parameters to copy from.
        to_params (list of torch.nn.Parameter): An iterable of destination parameters to copy to.
    """
    for from_param, to_param in zip(from_params, to_params):
        device = to_param.device
        to_param.copy_(from_param.to(device))

@torch.no_grad
def average_client_parameters(client_train_params_dict):
    """
    Efficiently averages parameters from multiple clients on GPU.
    This function uses an incremental mean calculation to avoid creating large
    intermediate tensors, performing in-place additions to keep memory usage low.
    Args:
        client_train_params_dict (dict): Dictionary where keys are client IDs and values are lists of parameter tensors.
    Returns:
        list: A list of averaged parameters, stored on `utils.DEVICE`.
    """
    num_clients = len(client_train_params_dict)
    average_params = None
    for train_params in client_train_params_dict.values():
        if not average_params:
            average_params = [param.clone().to(utils.DEVICE) / num_clients for param in train_params]
        else:
            for param, avg_param in zip(train_params, average_params):
                avg_param.add_(param.to(utils.DEVICE) / num_clients)
    return average_params

@torch.no_grad
def average_client_parameters2(client_train_params_dict):
    """
    Averages the parameters from multiple clients.
    This function computes the mean of the parameters from all clients. It stacks
    the parameters for each layer across clients, computes the mean, and returns
    the averaged parameters. The @torch.no_grad decorator ensures that this operation
    is not tracked by autograd.
    Args:
        client_train_params_dict (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
    Returns:
        list: A list of averaged parameters which lies on `utils.DEVICE`.
    """
    average_params = [
        torch.mean(torch.stack(param_list), dim=0)
        for param_list in zip(*client_train_params_dict.values())
    ]
    return average_params

@torch.no_grad
def compute_metrics(model, ds_path, ds_name, test_ds_dict):
    """
    Computes evaluation metrics for the given model on the test dataset.
    This function evaluates the model on the provided test dataset and computes
    metrics using the `evaluate` library. The `@torch.no_grad` decorator ensures
    that the evaluation is performed without tracking gradients, which saves memory
    and computation.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        ds_path (str): The path or identifier of the dataset. This is used to load the appropriate evaluation metric.
        ds_name (str): The name of the specific dataset configuration. This helps in loading the correct evaluation metric.
        test_ds_dict (Dict): A dictionary with key the name of the test_ds and value the DataLoader for this test dataset.
    Returns:
        dict: A dictionary containing the computed evaluation metrics.
    """
    def _compute_metrics(_model, _ds_path, _config_name, _test_ds):
        metric = evaluate.load(path=_ds_path, config_name=_config_name)
        testing_loss = 0.0
        num_batches = len(_test_ds)
        _model.eval()
        for batch in _test_ds:
            batch = {k: v.to(utils.DEVICE) for k, v in batch.items()}
            outputs = _model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            testing_loss += loss.item()
        average_test_loss = testing_loss / num_batches
        metrics = metric.compute()
        metrics['testing_loss'] = average_test_loss
        return metrics
    if ds_path == 'glue':
        if ds_name == 'mnli':
            metrics_m = _compute_metrics(model, ds_path, 'mnli_matched', test_ds_dict['mnli_matched'])
            metrics_m = {k + "_m": v for k, v in metrics_m.items()}
            metrics_mm = _compute_metrics(model, ds_path, 'mnli_mismatched', test_ds_dict['mnli_mismatched'])
            metrics_mm = {k + "_mm": v for k, v in metrics_mm.items()}
            return {**metrics_m, **metrics_mm}
        else:
            return _compute_metrics(model, ds_path, ds_name, test_ds_dict[ds_name])

@torch.no_grad
def compute_drifts(old_params, new_params):
    """
    Calculate the drift (difference) between old and new parameters.
    Args:
        old_params (list of torch.nn.Parameter): The original parameters.
        new_params (list of torch.nn.Parameter): The updated parameters.
    Returns:
        list of torch.Tensor: The computed drifts for each parameter stored in SAVE_DEVICE.
    """
    drifts = []
    for old_param, new_param in zip(old_params, new_params):
        old_param = old_param.to(utils.DEVICE)
        new_param = new_param.to(utils.DEVICE)
        drifts.append(
            (new_param - old_param).to(get_save_device())
        )
    return drifts

@torch.no_grad
def compute_client_drifts(old_params, client_train_params_dict):
    """
    Compute the drifts for all clients based on the original parameters.
    Args:
        old_params (list of torch.nn.Parameter): The original parameters.
        client_train_params_dict (dict): Dictionary of client IDs and their corresponding parameters.
    Returns:
        dict: A dictionary where keys are client IDs and values are lists of drifts for each parameter stored in DEVICE_SAVE.
    """
    return {
        client_id: compute_drifts(old_params, client_params)
        for client_id, client_params in client_train_params_dict.items()
    }

@torch.no_grad
def compute_pseudo_gradients(client_drifts):
    """
    Compute the pseudo-gradient based on the drifts between old and client parameters.
    Args:
        client_drifts (dict): Dictionary of client IDs and their corresponding parameter drifts.
    Returns:
        list of torch.Tensor: The computed pseudo-gradient with the result lying on utils.DEVICE.
    """
    average_drifts = average_client_parameters(client_drifts)
    pseudo_gradient = [-drift for drift in average_drifts]
    return pseudo_gradient

@torch.no_grad
def set_gradients(train_params, gradients):
    """
    Set gradients for trainable parameters.
    This function assigns the provided gradients to the .grad attribute of the corresponding trainable parameters.
    Args:
        train_params (list of torch.nn.Parameter): The trainable parameters of the model.
        gradients (list of torch.Tensor): The gradients to be assigned to the parameters.
    """
    for param, gradient in zip(train_params, gradients):
        device = param.device
        gradient = gradient.to(device)
        param.grad = gradient

@torch.no_grad
def vectorize(parameters):
    """
    Concatenates a list of parameter tensors into a single vector.
    Args:
        parameters (list of torch.nn.Parameter): An iterable of parameter tensors.
    Returns:
        torch.Tensor: A single vector containing all the elements of the input parameters with the
                      result lying on `utils.DEVICE`.
    """
    con = []
    for param in parameters:
        param = param.to(utils.DEVICE)
        con.append(param.view(-1))
    return torch.cat(con)

@torch.no_grad()
def get_ksi_unit(w_t0, w_tminus1, generator=None):
    """
    Calculates the heuristic unit vector (ksi) as a list of tensors
    matching the model's parameter structure, avoiding huge allocations.
    """
    ksi_tensors = []
    total_norm_sq = 0.0
    if w_tminus1 is None:
        for param in w_t0:
            # Use torch.randn and pass shape/dtype explicitly since randn_like doesn't take generator
            rand_tensor = torch.randn(param.shape, dtype=param.dtype, device=utils.DEVICE, generator=generator)
            ksi_tensors.append(rand_tensor)
            total_norm_sq += torch.sum(rand_tensor ** 2).item()
    else:
        for p_t0, p_tminus1 in zip(w_t0, w_tminus1):
            diff = p_t0.to(utils.DEVICE) - p_tminus1.to(utils.DEVICE)
            ksi_tensors.append(diff)
            total_norm_sq += torch.sum(diff ** 2).item()
    norm = math.sqrt(total_norm_sq)
    if norm > 0.0:
        for ksi_tensor in ksi_tensors:
            ksi_tensor.div_(norm)
    return ksi_tensors

@torch.no_grad()
def linear_variance_approx(client_drifts, ksi_vector):
    """
    Computes variance approximation iteratively without vectorizing,
    saving significant VRAM and preventing memory fragmentation.
    """
    # BUG FIX: Compute the full L2 norm squared of the average drift vector instead of a 1D projection.
    # This prevents the mathematical overestimation of variance that forces catastrophic overtraining.
    num_clients = len(client_drifts)
    sum_sq_norms = 0.0
    
    # S_1: Expected squared norm
    for drifts in client_drifts.values():
        norm_sq_drift = 0.0
        for drift_param in drifts:
            norm_sq_drift += torch.sum(drift_param.to(utils.DEVICE) ** 2).item()
        sum_sq_norms += norm_sq_drift
        
    S_1 = sum_sq_norms / num_clients
    
    # S_2: Squared norm of expected drift
    S_2_sq = 0.0
    first_client_id = list(client_drifts.keys())[0]
    num_params = len(client_drifts[first_client_id])
    
    for param_idx in range(num_params):
        avg_param = None
        for client_id, drifts in client_drifts.items():
            if avg_param is None:
                avg_param = drifts[param_idx].clone().to(utils.DEVICE) / num_clients
            else:
                avg_param.add_(drifts[param_idx].to(utils.DEVICE) / num_clients)
        S_2_sq += torch.sum(avg_param ** 2).item()
        
    return S_1 - S_2_sq


def linear_variance_approx2(client_drifts, global_current, global_prev):
    """
    Memory-efficient 1:1 port of the FedL-Sync-FDA linear variance approximation.
    Args:
        client_drifts: List of state_dicts (or dicts of tensors) representing Delta_i for each client.
        global_current: state_dict of the current global model (w_t0).
        global_prev: state_dict of the previous global model (w_tminus1).
    Returns:
        float: The estimated variance.
    """
    # BUG FIX: Compute the full L2 norm squared of the average drift vector instead of a 1D projection.
    # This prevents the mathematical overestimation of variance that forces catastrophic overtraining.
    num_clients = len(client_drifts)
    if num_clients == 0: return 0.0
    sum_sq_norms = 0.0
    
    # S_1: Expected squared norm
    for delta_i in client_drifts:
        client_sq_norm = 0.0
        for k in delta_i.keys():
            client_sq_norm += torch.sum(delta_i[k] ** 2).item()
        sum_sq_norms += client_sq_norm
        
    S_1 = sum_sq_norms / num_clients
    
    # S_2: Squared norm of expected drift
    S_2_sq = 0.0
    keys = client_drifts[0].keys()
    
    for k in keys:
        avg_k = None
        for delta_i in client_drifts:
            if avg_k is None:
                avg_k = delta_i[k].clone() / num_clients
            else:
                avg_k.add_(delta_i[k] / num_clients)
        S_2_sq += torch.sum(avg_k ** 2).item()
    
    variance_approx = S_1 - S_2_sq
    return variance_approx

def get_ksi_unit2(global_current, global_prev):
    """
    Generates the projection vector ksi as the normalized difference
    between the current and previous global models, avoiding tensor flattening.
    """
    ksi = {}
    diff_norm_sq = 0.0
    for k in global_current.keys():
        diff = global_current[k] - global_prev[k]
        ksi[k] = diff
        diff_norm_sq += torch.sum(diff ** 2).item()
    diff_norm = math.sqrt(diff_norm_sq)
    if diff_norm == 0.0:
        rand_norm_sq = 0.0
        for k in ksi.keys():
            ksi[k] = torch.randn_like(ksi[k])
            rand_norm_sq += torch.sum(ksi[k] ** 2).item()
        rand_norm = math.sqrt(rand_norm_sq)
        for k in ksi.keys():
            ksi[k] = ksi[k] / rand_norm
    else:
        for k in ksi.keys():
            ksi[k] = ksi[k] / diff_norm
    return ksi

@torch.no_grad
def variance(client_drifts):
    """
    Computes the variance of the client models utilizing the drifts (see paper)
    Args:
        client_drifts (dict): A dictionary where keys are client IDs and values are lists of parameter tensors (drifts).
    Returns:
        float: The computed variance of the client drifts.
    """
    norm_sq_drifts = []
    for drifts in client_drifts.values():
        drift_vec = vectorize(drifts)
        norm_sq_drift = torch.dot(drift_vec, drift_vec)
        norm_sq_drifts.append(norm_sq_drift)
    avg_norm_sq_drifts = sum(norm_sq_drifts) / len(norm_sq_drifts)
    avg_drift = average_client_parameters(client_drifts)
    avg_drift_vec = vectorize(avg_drift)
    norm_sq_avg_drift = torch.dot(avg_drift_vec, avg_drift_vec)
    var = avg_norm_sq_drifts - norm_sq_avg_drift
    return var.item(), avg_norm_sq_drifts.item(), norm_sq_avg_drift.item()

def fda_sketch_estimation(client_drifts, ams_sketch):
    """
    Compute the linear estimation of ||avg(u_t)||^2 using SketchFDA. The way we compute it is by its equivalent form
    which is (1/1+e) * M_2(sk(avg(u_t))). Of course, in a real system implementation we would first compute each drift's
    sketch and then average the sketches on the server. They are equivalent, we simply do this because it is more
    efficient. (see paper)
    Args:
        client_drifts (dict): A dictionary where keys are client IDs and values are lists of parameter tensors (drifts).
        ams_sketch (AmsSketch): The AMS sketch used to compute the sketch estimation.
    Returns:
        float: The linear estimation of ||avg(u_t)||^2.
    """
    avg_drift = vectorize(average_client_parameters(client_drifts))
    sk = ams_sketch.sketch_for_vector(avg_drift)
    epsilon = ams_sketch.epsilon
    est = (1 / (1+epsilon)) * ams_sketch.estimate_euc_norm_squared(sk)
    return est

def fda_variance_approx(client_drifts, ams_sketch=None):
    norm_sq_drifts = []
    for drifts in client_drifts.values():
        drift_vec = vectorize(drifts)
        norm_sq_drift = torch.dot(drift_vec, drift_vec)
        norm_sq_drifts.append(norm_sq_drift)
    avg_norm_sq_drifts = sum(norm_sq_drifts) / len(norm_sq_drifts)
    norm_sq_avg_drift_approx = fda_sketch_estimation(client_drifts, ams_sketch)
    var_approx = avg_norm_sq_drifts - norm_sq_avg_drift_approx
    return var_approx.item()

@torch.no_grad
def get_updated_client_parameters(client_train_params_dict, sampled_clients, params):
    """
    Update the parameters of the sampled clients with the current global parameters.
    This function ensures there are no ID collisions or overwriting issues by creating a new dictionary
    for the updated client parameters.
    Args:
        client_train_params_dict (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
        sampled_clients (list): A list of newly sampled client IDs.
        params (list of torch.nn.Parameter): The current global parameters to be assigned to the sampled clients.
    Returns:
        dict: Dictionary where keys are client IDs and values are lists of parameter tensors.
    """
    new_client_train_params_dict = dict()
    for new_client_id, old_params in zip(sampled_clients, client_train_params_dict.values()):
        copy_parameters(
            from_params=params,
            to_params=old_params
        )
        new_client_train_params_dict[new_client_id] = old_params
    return new_client_train_params_dict

def stop_training_with_threshold(hyperparams, metrics):
    if hyperparams['checkpoint'] == 'roberta-base':
        if hyperparams['ds_name'] == 'mrpc':
            if metrics['accuracy'] >= 0.902 * 0.95:
                return True
        if hyperparams['ds_name'] == 'rte':
            if metrics['accuracy'] >= 0.789 * 0.95:
                return True
        if hyperparams['ds_name'] == 'cola':
            if metrics['matthews_correlation'] >= 0.636 * 0.95:
                return True
        if hyperparams['ds_name'] == 'sst2':
            if metrics['accuracy'] >= 0.948 * 0.99:
                return True
        if hyperparams['ds_name'] == 'qnli':
            if metrics['accuracy'] >= 0.928 * 0.95:
                return True
    if hyperparams['checkpoint'] == 'microsoft/deberta-v3-base':
        if hyperparams['ds_name'] == 'mnli':
            if metrics['accuracy_m'] >= 0.906 * 0.99 and metrics['accuracy_mm'] >= 0.907 * 0.99:
                return True
    return False