import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdaopt import utils
from fdaopt.models.toy_model import ToyModel
from fdaopt.datasets.fed_data_prep import prepare_federated_datasets
from fdaopt.models.ops import copy_parameters

def profile_toy_optimizations():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = torch.device("cuda:0")
    utils.set_device("cuda:0")
    utils.set_seed(42)

    print("=" * 80)
    print("PROFILING OPTIMIZATIONS SPECIFICALLY FOR TOY MODEL + MNIST")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # 1. Prepare federated MNIST dataset
    fed_ds, test_ds = prepare_federated_datasets(
        ds_path="ylecun/mnist",
        ds_name="mnist",
        checkpoint="toy-model",
        num_clients=5,
        alpha=0.0,
        batch_size=64,
        num_workers=0,
        pin_memory=False
    )
    client_id = 0
    num_steps = 94  # ~1 epoch for 6000 samples @ batch 64

    # Baseline Model & Optimizer
    model_vanilla = ToyModel(input_dim=784, rank=16, output_dim=10, linear=False).to(device)
    train_params = [p for p in model_vanilla.parameters() if p.requires_grad]
    client_dict = {client_id: [p.detach().clone() for p in train_params]}
    opt = torch.optim.SGD(train_params, lr=0.01)

    # -------------------------------------------------------------------------
    # Baseline: Vanilla client_round_train (as in fdaopt/training/fed_train.py)
    # -------------------------------------------------------------------------
    def run_vanilla(steps=num_steps):
        # Strict FP32, synchronous, .item() every batch
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model_vanilla.train()
        copy_parameters(from_params=client_dict[client_id], to_params=train_params)

        training_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for _ in range(steps):
            batch = fed_ds.next_client_batch(client_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_vanilla(**batch)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            # 2 syncs per batch
            correct_predictions += (preds == batch["labels"]).sum().item()
            total_predictions += batch["labels"].size(0)

            loss.backward()
            opt.step()
            opt.zero_grad()
            training_loss += loss.item()

        copy_parameters(from_params=train_params, to_params=client_dict[client_id])
        return training_loss / steps, correct_predictions / total_predictions

    # -------------------------------------------------------------------------
    # Opt 1: Setting 1 from fillers (TF32 Tensor Cores)
    # -------------------------------------------------------------------------
    def run_opt1_tf32(steps=num_steps):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return run_vanilla(steps)

    # -------------------------------------------------------------------------
    # Opt 2: Setting 2 from fillers (Zero-Sync GPU Metrics Accumulation)
    # -------------------------------------------------------------------------
    def run_opt2_zero_sync(steps=num_steps):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        model_vanilla.train()
        copy_parameters(from_params=client_dict[client_id], to_params=train_params)

        running_loss = torch.tensor(0.0, device=device)
        running_correct = torch.tensor(0, dtype=torch.long, device=device)
        total_predictions = 0

        for _ in range(steps):
            batch = fed_ds.next_client_batch(client_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_vanilla(**batch)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            running_correct += (preds == batch["labels"]).sum()
            total_predictions += batch["labels"].size(0)

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            running_loss += loss.detach()

        copy_parameters(from_params=train_params, to_params=client_dict[client_id])
        return (running_loss.item() / steps), (running_correct.item() / total_predictions)

    # -------------------------------------------------------------------------
    # Opt 3: Setting 1 + 2 + 3 Combined (TF32 + Zero-Sync + Non-blocking DMA)
    # -------------------------------------------------------------------------
    def run_opt3_combined(steps=num_steps):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        model_vanilla.train()
        copy_parameters(from_params=client_dict[client_id], to_params=train_params)

        running_loss = torch.tensor(0.0, device=device)
        running_correct = torch.tensor(0, dtype=torch.long, device=device)
        total_predictions = 0

        for _ in range(steps):
            batch = fed_ds.next_client_batch(client_id)
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            outputs = model_vanilla(**batch)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            running_correct += (preds == batch["labels"]).sum()
            total_predictions += batch["labels"].size(0)

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            running_loss += loss.detach()

        copy_parameters(from_params=train_params, to_params=client_dict[client_id])
        return (running_loss.item() / steps), (running_correct.item() / total_predictions)

    # -------------------------------------------------------------------------
    # Opt 4: In-VRAM Tensor Batching (Toy Model Specific Super-Optimization)
    # Because MNIST is only ~47MB, clients can hold pre-loaded GPU tensors!
    # -------------------------------------------------------------------------
    # Collect client 0 dataset into a single GPU tensor
    raw_loader = fed_ds.client_dataloaders[client_id]
    all_imgs = []
    all_lbls = []
    for b in raw_loader:
        all_imgs.append(b['image'])
        all_lbls.append(b['labels'])
    gpu_x = torch.cat(all_imgs, dim=0).to(device)
    gpu_y = torch.cat(all_lbls, dim=0).to(device)

    def run_opt4_vram_cached(steps=num_steps):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        model_vanilla.train()
        copy_parameters(from_params=client_dict[client_id], to_params=train_params)

        running_loss = torch.tensor(0.0, device=device)
        running_correct = torch.tensor(0, dtype=torch.long, device=device)
        total_predictions = 0
        N = gpu_x.size(0)
        bs = 64

        for step in range(steps):
            idx = (step * bs) % (N - bs)
            bx = gpu_x[idx : idx + bs]
            by = gpu_y[idx : idx + bs]

            outputs = model_vanilla(image=bx, labels=by)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=-1)
            running_correct += (preds == by).sum()
            total_predictions += bs

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            running_loss += loss.detach()

        copy_parameters(from_params=train_params, to_params=client_dict[client_id])
        return (running_loss.item() / steps), (running_correct.item() / total_predictions)

    # Warmup
    run_vanilla(10)
    run_opt1_tf32(10)
    run_opt2_zero_sync(10)
    run_opt3_combined(10)
    run_opt4_vram_cached(10)
    torch.cuda.synchronize()

    # Benchmark iterations
    N_REPS = 5

    # 1. Vanilla
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        run_vanilla()
    torch.cuda.synchronize()
    vanilla_time = (time.perf_counter() - t0) / N_REPS * 1000

    # 2. Opt 1 (TF32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        run_opt1_tf32()
    torch.cuda.synchronize()
    opt1_time = (time.perf_counter() - t0) / N_REPS * 1000

    # 3. Opt 2 (Zero Sync)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        run_opt2_zero_sync()
    torch.cuda.synchronize()
    opt2_time = (time.perf_counter() - t0) / N_REPS * 1000

    # 4. Opt 3 (Fillers Combined: TF32 + Zero Sync + DMA)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        run_opt3_combined()
    torch.cuda.synchronize()
    opt3_time = (time.perf_counter() - t0) / N_REPS * 1000

    # 5. Opt 4 (Toy-Specific In-VRAM Cached)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        run_opt4_vram_cached()
    torch.cuda.synchronize()
    opt4_time = (time.perf_counter() - t0) / N_REPS * 1000

    print(f"\nRESULTS (per local client round, {num_steps} mini-batches):")
    print(f"  Vanilla Pipeline:                     {vanilla_time:8.2f} ms")
    print(f"  Opt 1 (TF32 Precision):               {opt1_time:8.2f} ms  ({vanilla_time/opt1_time:5.2f}x speedup)")
    print(f"  Opt 2 (Zero-Sync CPU-GPU Registers):  {opt2_time:8.2f} ms  ({vanilla_time/opt2_time:5.2f}x speedup)")
    print(f"  Opt 3 (Fillers Combined: 1+2+3):      {opt3_time:8.2f} ms  ({vanilla_time/opt3_time:5.2f}x speedup)")
    print(f"  Opt 4 (Toy-Specific In-VRAM Caching): {opt4_time:8.2f} ms  ({vanilla_time/opt4_time:5.2f}x speedup)")

if __name__ == '__main__':
    profile_toy_optimizations()
