import os
import sys
import torch
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdaopt import utils
from fdaopt.models.toy_model import ToyModel
from fdaopt.datasets.fed_data_prep import prepare_federated_datasets
from fdaopt.models.ops import copy_parameters

def check_math_equivalence():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set up fixed seed and initial model
    torch.manual_seed(42)
    initial_model = ToyModel(input_dim=784, rank=16, output_dim=10, linear=False).to(device)
    
    # 1. Base data
    fed_ds, _ = prepare_federated_datasets(
        ds_path="ylecun/mnist", ds_name="mnist", checkpoint="toy-model",
        num_clients=5, alpha=0.0, batch_size=64, num_workers=0, pin_memory=False
    )
    
    # Fetch 5 identical batches
    batches_cpu = [fed_ds.next_client_batch(0) for _ in range(5)]

    # -------------------------------------------------------------
    # Case A: Vanilla training (Strict FP32, .item() syncs, zero_grad())
    # -------------------------------------------------------------
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    
    model_A = copy.deepcopy(initial_model)
    opt_A = torch.optim.SGD([p for p in model_A.parameters() if p.requires_grad], lr=0.01)
    
    losses_A = []
    for b in batches_cpu:
        opt_A.zero_grad()
        batch = {k: v.to(device) for k, v in b.items()}
        out = model_A(**batch)
        loss = out.loss
        losses_A.append(loss.item())
        loss.backward()
        opt_A.step()

    # -------------------------------------------------------------
    # Case B: Zero-Sync + set_to_none=True + Pinned/Non-blocking
    # -------------------------------------------------------------
    model_B = copy.deepcopy(initial_model)
    opt_B = torch.optim.SGD([p for p in model_B.parameters() if p.requires_grad], lr=0.01)
    
    running_loss_B = torch.tensor(0.0, device=device)
    for b in batches_cpu:
        opt_B.zero_grad(set_to_none=True)
        batch = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        out = model_B(**batch)
        loss = out.loss
        running_loss_B += loss.detach()
        loss.backward()
        opt_B.step()
    avg_loss_B = running_loss_B.item() / 5

    # Check difference between A and B
    max_diff_A_B_lora_A = torch.max(torch.abs(model_A.lora_A - model_B.lora_A)).item()
    max_diff_A_B_lora_B = torch.max(torch.abs(model_A.lora_B - model_B.lora_B)).item()
    loss_diff_A_B = abs(sum(losses_A)/5 - avg_loss_B)

    print("\n--- TEST 1: Zero-Sync & set_to_none vs Vanilla ---")
    print(f"Max parameter difference in lora_A: {max_diff_A_B_lora_A:.10e}")
    print(f"Max parameter difference in lora_B: {max_diff_A_B_lora_B:.10e}")
    print(f"Loss difference:                   {loss_diff_A_B:.10e}")
    if max_diff_A_B_lora_A == 0.0 and max_diff_A_B_lora_B == 0.0:
        print(">> EXACT BIT-FOR-BIT MATHEMATICAL IDENTITY CONFIRMED (Diff = 0.0)!")
    else:
        print(">> Close to zero.")

    # -------------------------------------------------------------
    # Case C: In-VRAM Pre-cached Tensors vs Vanilla
    # -------------------------------------------------------------
    # Pre-load batches onto GPU
    gpu_batches = [
        {'image': b['image'].to(device), 'labels': b['labels'].to(device)}
        for b in batches_cpu
    ]
    model_C = copy.deepcopy(initial_model)
    opt_C = torch.optim.SGD([p for p in model_C.parameters() if p.requires_grad], lr=0.01)
    
    for b in gpu_batches:
        opt_C.zero_grad(set_to_none=True)
        out = model_C(**b)
        out.loss.backward()
        opt_C.step()

    max_diff_A_C_lora_A = torch.max(torch.abs(model_A.lora_A - model_C.lora_A)).item()
    max_diff_A_C_lora_B = torch.max(torch.abs(model_A.lora_B - model_C.lora_B)).item()

    print("\n--- TEST 2: In-VRAM Pre-caching vs Vanilla ---")
    print(f"Max parameter difference in lora_A: {max_diff_A_C_lora_A:.10e}")
    print(f"Max parameter difference in lora_B: {max_diff_A_C_lora_B:.10e}")
    if max_diff_A_C_lora_A == 0.0 and max_diff_A_C_lora_B == 0.0:
        print(">> EXACT BIT-FOR-BIT MATHEMATICAL IDENTITY CONFIRMED (Diff = 0.0)!")
    else:
        print(">> Close to zero.")

    # -------------------------------------------------------------
    # Case D: TF32 Precision vs Strict FP32
    # -------------------------------------------------------------
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    
    model_D = copy.deepcopy(initial_model)
    opt_D = torch.optim.SGD([p for p in model_D.parameters() if p.requires_grad], lr=0.01)
    
    for b in batches_cpu:
        opt_D.zero_grad(set_to_none=True)
        batch = {k: v.to(device) for k, v in b.items()}
        out = model_D(**batch)
        out.loss.backward()
        opt_D.step()

    max_diff_A_D_lora_A = torch.max(torch.abs(model_A.lora_A - model_D.lora_A)).item()
    max_diff_A_D_lora_B = torch.max(torch.abs(model_A.lora_B - model_D.lora_B)).item()

    print("\n--- TEST 3: TF32 vs Strict FP32 ---")
    print(f"Max parameter difference in lora_A: {max_diff_A_D_lora_A:.10e}")
    print(f"Max parameter difference in lora_B: {max_diff_A_D_lora_B:.10e}")
    print(f"Relative difference: ~{max_diff_A_D_lora_A / model_A.lora_A.abs().mean().item() * 100:.4f}%")

if __name__ == '__main__':
    check_math_equivalence()
