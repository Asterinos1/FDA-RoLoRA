import gc
from peft import get_peft_model, LoraConfig, TaskType
from datetime import datetime
from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import average_client_parameters, compute_client_drifts, compute_pseudo_gradients, set_gradients, copy_parameters, \
    get_updated_client_parameters, variance, compute_metrics, stop_training_with_threshold
from fdaopt.training.fed_train import client_round_train
from fdaopt.training.optimizers import get_opt_hyperparams, get_optimizer_class, server_client_optimizers
from fdaopt import utils
from transformers import AutoModelForSequenceClassification

def get_save_device():
    if utils.DEVICE_RAM_PROGRAM == 'performance' or utils.DEVICE_RAM_PROGRAM == 'moderate':
        return utils.DEVICE
    elif utils.DEVICE_RAM_PROGRAM == 'low':
        return 'cpu'
    return 'cpu'

def fed_opt(hyperparams):
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'],
        hyperparams['ds_name'],
        hyperparams['checkpoint'],
        hyperparams['num_clients'],
        hyperparams['alpha'],
        hyperparams['batch_size'],
        hyperparams.get('num_workers', 4),
        hyperparams.get('pin_memory', True)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'],
        num_labels=hyperparams['num_labels']
    )
    model = model.to(utils.DEVICE)
    train_params = [param for param in model.parameters() if param.requires_grad]
    round_start_train_params = [param.detach().clone() for param in train_params]
    server_opt, client_opt = server_client_optimizers(train_params, hyperparams)
    client_sampler = ClientSampler(
        list(range(hyperparams['num_clients'])),
        hyperparams['clients_per_round']
    )
    client_train_params_dict = {
        client_id: [param.detach().clone().to(get_save_device()) for param in round_start_train_params]
        for client_id in range(hyperparams['num_clients'])
    }
    metrics_handler = MetricsHandler(hyperparams)
    for r in range(hyperparams['total_rounds']):
        training_loss = 0.0
        sampled_clients = client_sampler.sample()
        copy_parameters(
            from_params=train_params,
            to_params=round_start_train_params
        )
        client_train_params_dict = get_updated_client_parameters(client_train_params_dict, sampled_clients, round_start_train_params)
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        for client_id in sampled_clients:
            loss = client_round_train(model, train_params, client_train_params_dict, client_opt, fed_ds, client_id, round_steps)
            training_loss += loss
        copy_parameters(
            from_params=round_start_train_params,
            to_params=train_params
        )
        client_drifts = compute_client_drifts(round_start_train_params, client_train_params_dict)
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        set_gradients(train_params, pseudo_gradients)
        server_opt.step()
        server_opt.zero_grad()
        metrics = {"round": r + 1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        metrics['round_steps'] = round_steps
        metrics_handler.append_round_metrics(metrics)
        gc.collect()
        if hyperparams['threshold']:
            if stop_training_with_threshold(hyperparams, metrics): break
    metrics_handler.save_metrics()

def toggle_rolora_matrices(model, active_matrix):
    """
    Dynamically freezes and unfreezes LoRA A and B matrices,
    ensuring all other parameters remain frozen per pure theory.
    """
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            param.requires_grad = (active_matrix == 'A')
        elif 'lora_B' in name:
            param.requires_grad = (active_matrix == 'B')
        else:
            param.requires_grad = False

def fedavg_rolora(hyperparams):
    """
    Executes the FedAvg-RoLoRA federated training loop strictly adhering to the original paper's theory.
    This ensures alternating optimization of LoRA matrices while keeping all non-LoRA parameters fixed.
    """
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4),
        hyperparams.get('pin_memory', True)
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'],
        num_labels=hyperparams['num_labels']
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False,
        r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
        lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
    )
    model = get_peft_model(base_model, peft_config)
    model = model.to(utils.DEVICE)
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n]
    group_A = params_A
    group_B = params_B
    len_A = len(params_A)
    len_B = len(params_B)
    client_opt_class = get_optimizer_class(hyperparams['client_opt_name'])
    client_opt_kwargs = get_opt_hyperparams('client', hyperparams)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {
        client_id: [param.detach().clone().to(get_save_device()) for param in group_A]
        for client_id in range(hyperparams['num_clients'])
    }
    client_dict_B = {
        client_id: [param.detach().clone().to(get_save_device()) for param in group_B]
        for client_id in range(hyperparams['num_clients'])
    }
    metrics_handler = MetricsHandler(hyperparams)
    time_log = []
    for r in range(hyperparams['total_rounds']):
        training_loss = 0.0
        sampled_clients = client_sampler.sample()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        start_time = datetime.now()
        print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Round {r+1} | Active Matrix: {active_matrix}")
        toggle_rolora_matrices(model, active_matrix)
        active_params = group_A if active_matrix == 'A' else group_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        active_lora_len = len_A if active_matrix == 'A' else len_B
        client_opt = client_opt_class(active_params, **client_opt_kwargs)
        round_start_active_params = [param.detach().clone() for param in active_params]
        active_client_dict = get_updated_client_parameters(
            active_client_dict, sampled_clients, round_start_active_params
        )
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        total_sampled = len(sampled_clients)
        for i, client_id in enumerate(sampled_clients, 1):
            loss = client_round_train(
                model, active_params, active_client_dict,
                client_opt, fed_ds, client_id, round_steps
            )
            training_loss += loss
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"#{i} client complete (out of {total_sampled}) at {current_time}")
        client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
        sampled_client_params = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
        averaged_params = average_client_parameters(sampled_client_params)
        copy_parameters(from_params=averaged_params, to_params=active_params)
        lora_only_drifts = {
            client_id: drifts[:active_lora_len]
            for client_id, drifts in client_drifts.items()
        }
        metrics = {"round": r + 1, "active_matrix": active_matrix} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(lora_only_drifts)
        metrics['round_steps'] = round_steps
        metrics_handler.append_round_metrics(metrics)
        end_time = datetime.now()
        print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Finished Round {r+1} | Active Matrix: {active_matrix}\n")
        time_log.append({
            "round": r + 1,
            "active_matrix": active_matrix,
            "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "duration_seconds": (end_time - start_time).total_seconds()
        })
        gc.collect()
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
    metrics_handler.save_metrics()
    if time_log:
        time_df = pd.DataFrame(time_log)
        time_filename = f"TIME_{metrics_handler.filename}"
        time_filepath = os.path.join(ROUND_METRICS_PATH, time_filename)
        time_df.to_csv(time_filepath, index=False)
        print(f"Time metrics successfully saved to: {time_filepath}")

def fedavg_rolora2(hyperparams):
    """
    Executes the FedAvg-RoLoRA federated training loop strictly adhering to the original paper's theory.
    This ensures alternating optimization of LoRA matrices while keeping all non-LoRA parameters fixed.
    """
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4),
        hyperparams.get('pin_memory', True)
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False,
        r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
        lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
    )
    model = get_peft_model(base_model, peft_config)
    model = model.to(utils.DEVICE)
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n]
    group_A = params_A
    group_B = params_B
    len_A = len(params_A)
    len_B = len(params_B)
    client_opt_class = get_optimizer_class(hyperparams['client_opt_name'])
    client_opt_kwargs = get_opt_hyperparams('client', hyperparams)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {
        client_id: [param.detach().clone().to(get_save_device()) for param in group_A]
        for client_id in range(hyperparams['num_clients'])
    }
    client_dict_B = {
        client_id: [param.detach().clone().to(get_save_device()) for param in group_B]
        for client_id in range(hyperparams['num_clients'])
    }
    metrics_handler = MetricsHandler(hyperparams)
    for r in range(hyperparams['total_rounds']):
        training_loss = 0.0
        sampled_clients = client_sampler.sample()
        active_matrix = 'B' if r % 2 == 0 else 'A'
        print(f"Starting Round {r+1} | Active Matrix: {active_matrix}")
        toggle_rolora_matrices(model, active_matrix)
        active_params = group_A if active_matrix == 'A' else group_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        active_lora_len = len_A if active_matrix == 'A' else len_B
        client_opt = client_opt_class(active_params, **client_opt_kwargs)
        round_start_active_params = [param.detach().clone() for param in active_params]
        active_client_dict = get_updated_client_parameters(
            active_client_dict, sampled_clients, round_start_active_params
        )
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        for client_id in sampled_clients:
            loss = client_round_train(
                model, active_params, active_client_dict,
                client_opt, fed_ds, client_id, round_steps
            )
            training_loss += loss
        client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
        sampled_client_params = {c_id: active_client_dict[c_id] for c_id in sampled_clients}
        averaged_params = average_client_parameters(sampled_client_params)
        copy_parameters(from_params=averaged_params, to_params=active_params)
        lora_only_drifts = {
            client_id: drifts[:active_lora_len]
            for client_id, drifts in client_drifts.items()
        }
        metrics = {"round": r + 1, "active_matrix": active_matrix} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(lora_only_drifts)
        metrics['round_steps'] = round_steps
        metrics_handler.append_round_metrics(metrics)
        gc.collect()
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
    metrics_handler.save_metrics()

def fed_rolora_opt(hyperparams):
    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'],
        hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'],
        hyperparams.get('num_workers', 4), hyperparams.get('pin_memory', True)
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'], num_labels=hyperparams['num_labels']
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False,
        r=hyperparams.get('lora_r', 8), lora_alpha=hyperparams.get('lora_alpha', 16),
        lora_dropout=0.1, target_modules=hyperparams.get('target_modules', None)
    )
    model = get_peft_model(base_model, peft_config)
    model = model.to(utils.DEVICE)
    params_A = [p for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    group_A = params_A + params_other
    group_B = params_B + params_other
    len_A = len(params_A)
    len_B = len(params_B)
    server_opt_A, _ = server_client_optimizers(group_A, hyperparams)
    server_opt_B, _ = server_client_optimizers(group_B, hyperparams)
    client_sampler = ClientSampler(list(range(hyperparams['num_clients'])), hyperparams['clients_per_round'])
    client_dict_A = {
        client_id: [param.detach().clone().to(get_save_device()) for param in group_A]
        for client_id in range(hyperparams['num_clients'])
    }
    client_dict_B = {
        client_id: [param.detach().clone().to(get_save_device()) for param in group_B]
        for client_id in range(hyperparams['num_clients'])
    }
    metrics_handler = MetricsHandler(hyperparams)
    for r in range(hyperparams['total_rounds']):
        training_loss = 0.0
        sampled_clients = client_sampler.sample()
        active_matrix = 'A' if r % 2 == 0 else 'B'
        toggle_rolora_matrices(model, active_matrix)
        active_params = group_A if active_matrix == 'A' else group_B
        active_server_opt = server_opt_A if active_matrix == 'A' else server_opt_B
        active_client_dict = client_dict_A if active_matrix == 'A' else client_dict_B
        active_lora_len = len_A if active_matrix == 'A' else len_B
        _, client_opt = server_client_optimizers(active_params, hyperparams)
        round_start_active_params = [param.detach().clone() for param in active_params]
        copy_parameters(from_params=active_params, to_params=round_start_active_params)
        active_client_dict = get_updated_client_parameters(
            active_client_dict, sampled_clients, round_start_active_params
        )
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        for client_id in sampled_clients:
            loss = client_round_train(
                model, active_params, active_client_dict,
                client_opt, fed_ds, client_id, round_steps
            )
            training_loss += loss
        copy_parameters(from_params=round_start_active_params, to_params=active_params)
        client_drifts = compute_client_drifts(round_start_active_params, active_client_dict)
        pseudo_gradients = compute_pseudo_gradients(client_drifts)
        set_gradients(active_params, pseudo_gradients)
        active_server_opt.step()
        active_server_opt.zero_grad()
        lora_only_drifts = {
            client_id: drifts[:active_lora_len]
            for client_id, drifts in client_drifts.items()
        }
        metrics = {"round": r + 1, "active_matrix": active_matrix} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(lora_only_drifts)
        metrics['round_steps'] = round_steps
        metrics_handler.append_round_metrics(metrics)
        gc.collect()
        if hyperparams.get('threshold') and stop_training_with_threshold(hyperparams, metrics):
            break
    metrics_handler.save_metrics()