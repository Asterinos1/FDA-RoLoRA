"""
Mathematical tensor operations, variance approximations, and evaluation metrics for FDA.

Provides core tensor manipulations:
  - In-place parameter copying and memory-efficient averaging across clients.
  - Client drift vector calculation and pseudo-gradient construction.
  - Variance estimation routines (exact L2, linear 1D projection, and AMS sketching).
  - Validation metrics computation with in-memory evaluator caching.
  - Dynamic early stopping and stagnation detection.
"""

import math
from typing import List, Dict, Tuple, Any, Optional
import torch
import evaluate
from fdaopt import utils

# Global evaluation metric cache to prevent repeated disk I/O
_METRICS_CACHE: Dict[str, Any] = {}


def get_save_device() -> str:
    """
    Select tensor storage device for offloaded client parameter dictionaries.

    Returns:
        str: PyTorch device ('cuda:0' or 'cpu').
    """
    if utils.DEVICE_RAM_PROGRAM == 'performance':
        return utils.DEVICE
    return 'cpu'


@torch.no_grad()
def copy_parameters(from_params: List[torch.Tensor], to_params: List[torch.Tensor]):
    """
    Copy tensor values element-wise from source to destination in-place.

    Args:
        from_params (list of torch.Tensor): Source parameters.
        to_params (list of torch.Tensor): Destination parameters modified in-place.
    """
    for from_param, to_param in zip(from_params, to_params):
        to_param.copy_(from_param.to(to_param.device))


@torch.no_grad()
def average_client_parameters(client_train_params_dict: Dict[int, List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Compute incremental mean of client parameter tensors on GPU to minimize VRAM usage.

    Args:
        client_train_params_dict (dict): Map of client IDs to lists of parameter tensors.

    Returns:
        list of torch.Tensor: Mean parameter tensors on `utils.DEVICE`.
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


@torch.no_grad()
def average_client_parameters2(client_train_params_dict: Dict[int, List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Compute layer-stacked mean of client parameter tensors.

    Args:
        client_train_params_dict (dict): Map of client IDs to lists of parameter tensors.

    Returns:
        list of torch.Tensor: Stacked mean parameter tensors on `utils.DEVICE`.
    """
    average_params = [
        torch.mean(torch.stack([p.to(utils.DEVICE) for p in param_list]), dim=0)
        for param_list in zip(*client_train_params_dict.values())
    ]
    return average_params


@torch.no_grad()
def compute_metrics(
    model: torch.nn.Module,
    ds_path: str,
    ds_name: str,
    test_ds_dict: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate the global model on task validation/test sets with cached evaluators.

    Args:
        model (torch.nn.Module): Model to evaluate.
        ds_path (str): Dataset repository path (e.g. 'glue', 'mnist').
        ds_name (str): Task name (e.g. 'mrpc', 'sst2', 'mnist').
        test_ds_dict (dict): Map of split names to DataLoader instances.

    Returns:
        dict: Evaluation metric key-value pairs (e.g. accuracy, F1, testing_loss).
    """
    def _compute_metrics(_model, _ds_path, _config_name, _test_ds):
        is_mnist = 'mnist' in _ds_path or 'mnist' in _config_name
        if not is_mnist:
            cache_key = f"{_ds_path}_{_config_name}"
            if cache_key not in _METRICS_CACHE:
                actual_metric_path = "glue" if "glue" in _ds_path else _ds_path
                _METRICS_CACHE[cache_key] = evaluate.load(
                    path=actual_metric_path,
                    config_name=_config_name,
                    keep_in_memory=True
                )
            metric = _METRICS_CACHE[cache_key]

        testing_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_batches = len(_test_ds)
        _model.eval()

        for batch in _test_ds:
            batch = {key: value.to(utils.DEVICE) for key, value in batch.items()}
            outputs = _model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            testing_loss += outputs.loss.item()

            if is_mnist:
                correct_predictions += (predictions == batch["labels"]).sum().item()
                total_predictions += batch["labels"].size(0)
            else:
                metric.add_batch(predictions=predictions, references=batch["labels"])

        average_test_loss = testing_loss / num_batches if num_batches > 0 else 0.0
        if is_mnist:
            return {
                "accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0.0,
                "testing_loss": average_test_loss
            }

        metrics = metric.compute()
        metrics["testing_loss"] = average_test_loss
        return metrics

    if ds_path in ('glue', 'nyu-mll/glue'):
        if ds_name == 'mnli':
            metrics_m = _compute_metrics(model, ds_path, 'mnli_matched', test_ds_dict['mnli_matched'])
            metrics_m = {k + "_m": v for k, v in metrics_m.items()}
            metrics_mm = _compute_metrics(model, ds_path, 'mnli_mismatched', test_ds_dict['mnli_mismatched'])
            metrics_mm = {k + "_mm": v for k, v in metrics_mm.items()}
            return {**metrics_m, **metrics_mm}
        return _compute_metrics(model, ds_path, ds_name, test_ds_dict[ds_name])

    target_name = ds_name if ds_name in test_ds_dict else list(test_ds_dict.keys())[0]
    return _compute_metrics(model, ds_path, target_name, test_ds_dict[target_name])


@torch.no_grad()
def compute_drifts(old_params: List[torch.Tensor], new_params: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Calculate drift tensor differences between initial and updated parameters (Delta = w_new - w_old).

    Args:
        old_params (list of torch.Tensor): Starting parameters at beginning of round.
        new_params (list of torch.Tensor): Updated parameters after local training.

    Returns:
        list of torch.Tensor: Parameter drifts stored on target save device.
    """
    drifts = []
    for old_param, new_param in zip(old_params, new_params):
        old_p = old_param.to(utils.DEVICE)
        new_p = new_param.to(utils.DEVICE)
        drifts.append((new_p - old_p).to(get_save_device()))
    return drifts


@torch.no_grad()
def compute_client_drifts(
    old_params: List[torch.Tensor],
    client_train_params_dict: Dict[int, List[torch.Tensor]]
) -> Dict[int, List[torch.Tensor]]:
    """
    Compute parameter drift vectors for all clients relative to the global starting point.

    Args:
        old_params (list of torch.Tensor): Starting global parameters.
        client_train_params_dict (dict): Map of client IDs to local parameter tensors.

    Returns:
        dict: Map of client IDs to lists of drift tensors.
    """
    return {
        client_id: compute_drifts(old_params, client_params)
        for client_id, client_params in client_train_params_dict.items()
    }


@torch.no_grad()
def compute_pseudo_gradients(client_drifts: Dict[int, List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Compute global pseudo-gradients as the negative average of client drifts: G = -avg(Delta_i).

    Args:
        client_drifts (dict): Map of client IDs to drift tensors.

    Returns:
        list of torch.Tensor: Pseudo-gradient tensors on `utils.DEVICE`.
    """
    average_drifts = average_client_parameters(client_drifts)
    pseudo_gradient = [-drift for drift in average_drifts]
    return pseudo_gradient


@torch.no_grad()
def set_gradients(train_params: List[torch.nn.Parameter], gradients: List[torch.Tensor]):
    """
    Assign pseudo-gradient tensors directly to parameter `.grad` attributes for optimizer step.

    Args:
        train_params (list of torch.nn.Parameter): Target model parameters.
        gradients (list of torch.Tensor): Gradients to assign.
    """
    for param, gradient in zip(train_params, gradients):
        param.grad = gradient.to(param.device)


@torch.no_grad()
def vectorize(parameters: List[torch.Tensor]) -> torch.Tensor:
    """
    Flatten and concatenate a list of parameter tensors into a single 1D vector on `utils.DEVICE`.

    Args:
        parameters (list of torch.Tensor): Iterable of tensor parameters.

    Returns:
        torch.Tensor: Contiguous 1D flattened vector.
    """
    con = [param.to(utils.DEVICE).view(-1) for param in parameters]
    return torch.cat(con)


@torch.no_grad()
def get_ksi_unit(
    w_t0: List[torch.Tensor],
    w_tminus1: Optional[List[torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None
) -> List[torch.Tensor]:
    """
    Generate normalized 1D projection unit vector matching the model parameter structure.

    Args:
        w_t0 (list of torch.Tensor): Current round parameters.
        w_tminus1 (Optional[list of torch.Tensor]): Previous round parameters (if computing momentum direction).
        generator (Optional[torch.Generator]): Seeded PyTorch random generator.

    Returns:
        list of torch.Tensor: Normalized projection unit tensors.
    """
    ksi_tensors = []
    total_norm_sq = 0.0

    if w_tminus1 is None:
        for param in w_t0:
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
def linear_variance_approx(
    client_drifts: Dict[int, List[torch.Tensor]],
    ksi_vector: List[torch.Tensor]
) -> float:
    """
    Compute memory-efficient variance approximation without full vector flattening:
        Var = (1/N) * sum(||Delta_i||^2) - ||avg(Delta_i)||^2

    Args:
        client_drifts (dict): Map of client IDs to drift tensors.
        ksi_vector (list of torch.Tensor): Normalized projection unit tensors.

    Returns:
        float: Estimated drift variance.
    """
    num_clients = len(client_drifts)
    sum_sq_norms = 0.0

    for drifts in client_drifts.values():
        norm_sq_drift = 0.0
        for drift_param in drifts:
            norm_sq_drift += torch.sum(drift_param.to(utils.DEVICE) ** 2).item()
        sum_sq_norms += norm_sq_drift

    s_1 = sum_sq_norms / num_clients
    s_2_sq = 0.0
    first_client_id = list(client_drifts.keys())[0]
    num_params = len(client_drifts[first_client_id])

    for param_idx in range(num_params):
        avg_param = None
        for client_id, drifts in client_drifts.items():
            if avg_param is None:
                avg_param = drifts[param_idx].clone().to(utils.DEVICE) / num_clients
            else:
                avg_param.add_(drifts[param_idx].to(utils.DEVICE) / num_clients)
        s_2_sq += torch.sum(avg_param ** 2).item()

    return s_1 - s_2_sq


def linear_variance_approx2(client_drifts: List[Dict[str, torch.Tensor]], global_current, global_prev) -> float:
    """
    Memory-efficient 1:1 port of the FedL-Sync-FDA state_dict linear variance approximation.

    Args:
        client_drifts (list of dict): List of drift state_dicts.
        global_current: Current global model state dict.
        global_prev: Previous global model state dict.

    Returns:
        float: Estimated drift variance.
    """
    num_clients = len(client_drifts)
    if num_clients == 0:
        return 0.0

    sum_sq_norms = 0.0
    for delta_i in client_drifts:
        client_sq_norm = sum(torch.sum(delta_i[k] ** 2).item() for k in delta_i.keys())
        sum_sq_norms += client_sq_norm

    s_1 = sum_sq_norms / num_clients
    s_2_sq = 0.0
    keys = client_drifts[0].keys()

    for k in keys:
        avg_k = None
        for delta_i in client_drifts:
            if avg_k is None:
                avg_k = delta_i[k].clone() / num_clients
            else:
                avg_k.add_(delta_i[k] / num_clients)
        s_2_sq += torch.sum(avg_k ** 2).item()

    return s_1 - s_2_sq


def get_ksi_unit2(global_current: Dict[str, torch.Tensor], global_prev: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Generate normalized projection vector ksi across state_dict entries without flattening.

    Args:
        global_current (dict): Current model state_dict.
        global_prev (dict): Previous model state_dict.

    Returns:
        dict: Normalized projection state_dict.
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


@torch.no_grad()
def variance(client_drifts: Dict[int, List[torch.Tensor]]) -> Tuple[float, float, float]:
    """
    Calculate exact client drift variance:
        Var = avg(||Delta_i||^2) - ||avg(Delta_i)||^2

    Args:
        client_drifts (dict): Map of client IDs to lists of drift tensors.

    Returns:
        tuple: (variance, avg_norm_sq_drifts, norm_sq_avg_drift)
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


def fda_sketch_estimation(client_drifts: Dict[int, List[torch.Tensor]], ams_sketch) -> float:
    """
    Compute linear sketch estimation of ||avg(u_t)||^2 using AMS Sketch:
        est = (1 / (1 + epsilon)) * M_2(sk(avg(u_t)))

    Args:
        client_drifts (dict): Map of client IDs to drift tensors.
        ams_sketch (AmsSketch): Initialized AMS sketch operator.

    Returns:
        float: Linear sketch estimation of squared norm of average drift.
    """
    avg_drift = vectorize(average_client_parameters(client_drifts))
    sk = ams_sketch.sketch_for_vector(avg_drift)
    epsilon = ams_sketch.epsilon
    est = (1.0 / (1.0 + epsilon)) * ams_sketch.estimate_euc_norm_squared(sk)
    return est


def fda_variance_approx(client_drifts: Dict[int, List[torch.Tensor]], ams_sketch=None) -> float:
    """
    Compute FDA variance approximation combining exact client norms with sketched average norm.

    Args:
        client_drifts (dict): Map of client IDs to drift tensors.
        ams_sketch (AmsSketch): AMS sketch operator.

    Returns:
        float: Approximate drift variance.
    """
    norm_sq_drifts = []
    for drifts in client_drifts.values():
        drift_vec = vectorize(drifts)
        norm_sq_drift = torch.dot(drift_vec, drift_vec)
        norm_sq_drifts.append(norm_sq_drift)

    avg_norm_sq_drifts = sum(norm_sq_drifts) / len(norm_sq_drifts)
    norm_sq_avg_drift_approx = fda_sketch_estimation(client_drifts, ams_sketch)
    var_approx = avg_norm_sq_drifts - norm_sq_avg_drift_approx
    return var_approx.item()


@torch.no_grad()
def get_updated_client_parameters(
    client_train_params_dict: Dict[int, List[torch.Tensor]],
    sampled_clients: List[int],
    params: List[torch.Tensor]
) -> Dict[int, List[torch.Tensor]]:
    """
    Synchronize sampled client parameter buffers with the latest global round start parameters.

    Args:
        client_train_params_dict (dict): Full dictionary of client parameter buffers.
        sampled_clients (list of int): Client IDs sampled for current communication round.
        params (list of torch.Tensor): Global round start parameters.

    Returns:
        dict: Updated client parameter dictionary.
    """
    for client_id in sampled_clients:
        copy_parameters(
            from_params=params,
            to_params=client_train_params_dict[client_id]
        )
    return client_train_params_dict


def stop_training_with_threshold(hyperparams: Dict[str, Any], metrics: Dict[str, float]) -> bool:
    """
    Check static convergence threshold criteria based on target task accuracy bounds.

    Args:
        hyperparams (dict): Experiment configuration dictionary.
        metrics (dict): Latest computed round metrics.

    Returns:
        bool: True if model has satisfied convergence target accuracy threshold.
    """
    if hyperparams.get('checkpoint') == 'roberta-base':
        ds_name = hyperparams.get('ds_name')
        if ds_name == 'mrpc' and metrics.get('accuracy', 0.0) >= 0.902 * 0.95:
            return True
        if ds_name == 'rte' and metrics.get('accuracy', 0.0) >= 0.789 * 0.95:
            return True
        if ds_name == 'cola' and metrics.get('matthews_correlation', 0.0) >= 0.636 * 0.95:
            return True
        if ds_name == 'sst2' and metrics.get('accuracy', 0.0) >= 0.948 * 0.99:
            return True
        if ds_name == 'qnli' and metrics.get('accuracy', 0.0) >= 0.928 * 0.95:
            return True

    if hyperparams.get('checkpoint') == 'microsoft/deberta-v3-base':
        if hyperparams.get('ds_name') == 'mnli':
            if metrics.get('accuracy_m', 0.0) >= 0.906 * 0.99 and metrics.get('accuracy_mm', 0.0) >= 0.907 * 0.99:
                return True

    return False


def get_target_metric_value(ds_name: str, metrics: Dict[str, Any]) -> float:
    """
    Extract the primary performance metric scalar for a given benchmark dataset.

    Args:
        ds_name (str): Dataset name (e.g. 'cola', 'mnli', 'mrpc', 'sst2').
        metrics (dict): Metric dictionary from evaluation.

    Returns:
        float: Primary target metric value.
    """
    if ds_name == 'cola':
        return metrics.get('matthews_correlation', 0.0)
    elif ds_name == 'mnli':
        return (metrics.get('accuracy_m', 0.0) + metrics.get('accuracy_mm', 0.0)) / 2.0
    return metrics.get('accuracy', 0.0)


def check_dynamic_early_stopping(hyperparams: Dict[str, Any], metrics_history: List[Dict[str, Any]]) -> bool:
    """
    Evaluate dynamic early stopping conditions based on rolling average improvement and stagnation.

    Stops training if:
      1. Model stagnates without meaningful accuracy gain within `stagnation_round` rounds.
      2. Performance reaches plateau threshold (`plateau_min_accuracy`) with no improvement for `patience` rounds.

    Args:
        hyperparams (dict): Experiment configuration containing early stopping hyperparams.
        metrics_history (list of dict): Chronological list of collected round metric dicts.

    Returns:
        bool: True if training should terminate early.
    """
    patience = hyperparams.get('early_stopping_patience', 10)
    min_delta = hyperparams.get('early_stopping_min_delta', 0.005)
    window_size = hyperparams.get('early_stopping_window', 3)
    plateau_min_acc = hyperparams.get('plateau_min_accuracy', 0.90)
    stagnation_round = hyperparams.get('stagnation_round', 40)
    stagnation_tolerance = hyperparams.get('stagnation_tolerance', 0.05)

    if not metrics_history:
        return False

    values = [get_target_metric_value(hyperparams['ds_name'], m) for m in metrics_history]
    current_round = len(values)

    # Check for early stagnation (failure to learn)
    if current_round >= stagnation_round:
        initial_acc = sum(values[:3]) / 3.0 if current_round >= 3 else values[0]
        current_max = max(values[:stagnation_round])
        if current_max - initial_acc < stagnation_tolerance:
            return True

    if current_round < window_size:
        return False

    # Smooth values using a moving average window
    smoothed_values = [
        sum(values[i:i + window_size]) / window_size
        for i in range(current_round - window_size + 1)
    ]

    best_value = -float('inf')
    rounds_without_improvement = 0

    for val in smoothed_values:
        if val > best_value + min_delta:
            best_value = val
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

    if best_value >= plateau_min_acc and rounds_without_improvement >= patience:
        return True

    return False
