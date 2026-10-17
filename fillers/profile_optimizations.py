"""Comprehensive Benchmark Script for the Top 3 Performance Optimizations.

Profiles:
  1. Setting 1: TF32 vs. Strict FP32 (Tensor Cores GEMM acceleration)
  2. Setting 2: Per-batch CPU-GPU sync stalls (.item() every batch vs. single sync at end)
  3. Setting 3: Host-to-Device transfer (Unpinned memory vs. Pinned memory with non_blocking=True)
  4. Combined: Cumulative speedup with all 3 optimizations active vs. vanilla baseline

Usage:
    python fillers/profile_optimizations.py
"""
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def get_gpu_info():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires an NVIDIA GPU.")
        sys.exit(1)
    prop = torch.cuda.get_device_properties(0)
    print("=" * 85)
    print(f"GPU Device:               {prop.name}")
    print(f"Compute Capability:       {prop.major}.{prop.minor}")
    print(f"Total VRAM:               {prop.total_memory / (1024**3):.2f} GB")
    print(f"PyTorch Version:          {torch.__version__}")
    print("=" * 85)


class MockRobertaLayer(nn.Module):
    """Simulates a RoBERTa-base transformer layer with rank-4 LoRA adapters."""
    def __init__(self, hidden_size=768, intermediate_size=3072, lora_rank=4):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.lora_q_a = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_q_b = nn.Linear(lora_rank, hidden_size, bias=False)
        self.lora_v_a = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_v_b = nn.Linear(lora_rank, hidden_size, bias=False)
        
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids):
        # Attention with LoRA
        q = self.q_proj(input_ids) + self.lora_q_b(self.lora_q_a(input_ids)) * 2.0
        k = self.k_proj(input_ids)
        v = self.v_proj(input_ids) + self.lora_v_b(self.lora_v_a(input_ids)) * 2.0
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (768 ** 0.5))
        attn_out = torch.matmul(torch.softmax(scores, dim=-1), v)
        x = self.norm(input_ids + self.out_proj(attn_out))
        x = self.norm(x + self.fc2(self.act(self.fc1(x))))
        # Pool to sequence representation and classify
        logits = self.classifier(x[:, 0, :])
        return logits


# =============================================================================
# BENCHMARK 1: Setting 1 (TF32 vs Strict FP32)
# =============================================================================
def benchmark_setting_1(device):
    print("\n" + "=" * 85)
    print("TEST 1: Setting 1 - TensorFloat-32 (TF32) Tensor Cores vs. Standard FP32")
    print("=" * 85)
    
    batch_size = 32
    seq_len = 128
    hidden_size = 768
    model = MockRobertaLayer(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    inputs = torch.randn(batch_size, seq_len, hidden_size, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    # 1. Standard FP32 (highest precision, no TF32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    for _ in range(15): step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(80): step()
    torch.cuda.synchronize()
    fp32_time = (time.perf_counter() - t0) / 80 * 1000

    # 2. TF32 (high precision, Tensor Cores enabled)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    for _ in range(15): step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(80): step()
    torch.cuda.synchronize()
    tf32_time = (time.perf_counter() - t0) / 80 * 1000

    speedup = fp32_time / tf32_time
    pct = (1.0 - (tf32_time / fp32_time)) * 100
    print(f"  Standard FP32:           {fp32_time:.3f} ms / step")
    print(f"  TF32 Tensor Cores:       {tf32_time:.3f} ms / step")
    print(f"  >> Setting 1 Speedup:    {speedup:.2f}x faster ({pct:.1f}% time saved per step)")
    return speedup


# =============================================================================
# BENCHMARK 2: Setting 2 (Per-Batch CPU-GPU Sync Stalls)
# =============================================================================
def benchmark_setting_2(device):
    print("\n" + "=" * 85)
    print("TEST 2: Setting 2 - Inner-Loop CPU-GPU Synchronizations (.item() Stalls)")
    print("=" * 85)
    print("Simulating a local client training run (50 mini-batches of forward + backward + opt steps):")

    batch_size = 32
    seq_len = 128
    hidden_size = 768
    num_steps = 50
    
    model = MockRobertaLayer(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    batches = [
        (torch.randn(batch_size, seq_len, hidden_size, device=device),
         torch.randint(0, 2, (batch_size,), device=device))
        for _ in range(num_steps)
    ]

    # Baseline: Synchronizing on EVERY batch (.item() twice per step)
    def train_with_sync():
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            # Sync 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            # Sync 2
            total_loss += loss.item()
        return total_loss / num_steps, correct / total

    # Optimized: Accumulate on GPU; sync ONLY ONCE at the end
    def train_without_sync():
        running_loss = torch.zeros((), device=device)
        running_correct = torch.zeros((), device=device, dtype=torch.long)
        total = 0
        for inputs, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            running_correct += (preds == labels).sum()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach()
        # Single sync at end
        avg_loss = (running_loss / num_steps).item()
        acc = running_correct.item() / total
        return avg_loss, acc

    # Warmup
    train_with_sync()
    train_without_sync()
    torch.cuda.synchronize()

    # Time Baseline
    t0 = time.perf_counter()
    for _ in range(10):
        train_with_sync()
    torch.cuda.synchronize()
    sync_time = (time.perf_counter() - t0) / 10 * 1000

    # Time Optimized
    t0 = time.perf_counter()
    for _ in range(10):
        train_without_sync()
    torch.cuda.synchronize()
    no_sync_time = (time.perf_counter() - t0) / 10 * 1000

    speedup = sync_time / no_sync_time
    pct = (1.0 - (no_sync_time / sync_time)) * 100
    print(f"  Baseline (2 syncs/batch):  {sync_time:.2f} ms per 50-step client")
    print(f"  Optimized (1 sync at end): {no_sync_time:.2f} ms per 50-step client")
    print(f"  >> Setting 2 Speedup:      {speedup:.2f}x faster ({pct:.1f}% time saved in loop)")
    return speedup


# =============================================================================
# BENCHMARK 3: Setting 3 (Pinned Memory & Asynchronous DMA Transfers)
# =============================================================================
def benchmark_setting_3(device):
    print("\n" + "=" * 85, flush=True)
    print("TEST 3: Setting 3 - Pinned Host Memory & Asynchronous PCIe Transfers", flush=True)
    print("=" * 85, flush=True)
    print("Testing data pipeline throughput across 50 batches from host RAM to GPU:", flush=True)

    batch_size = 32
    seq_len = 128
    hidden_size = 768
    num_batches = 50

    # Synthetic host dataset in CPU RAM
    x_cpu = torch.randn(num_batches * batch_size, seq_len, hidden_size)
    y_cpu = torch.randint(0, 2, (num_batches * batch_size,))
    ds = TensorDataset(x_cpu, y_cpu)

    # 1. Unpinned DataLoader (pin_memory=False)
    dl_unpinned = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=False)

    # 2. Pinned DataLoader (pin_memory=True)
    dl_pinned = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Compute task to overlap with
    model = nn.Linear(hidden_size, hidden_size).to(device)

    # Baseline: Unpinned + synchronous copy
    def run_unpinned():
        for x, y in dl_unpinned:
            x_gpu = x.to(device, non_blocking=False)
            y_gpu = y.to(device, non_blocking=False)
            out = model(x_gpu)
            _ = out.sum()

    # Optimized: Pinned + asynchronous non_blocking copy
    def run_pinned():
        for x, y in dl_pinned:
            x_gpu = x.to(device, non_blocking=True)
            y_gpu = y.to(device, non_blocking=True)
            out = model(x_gpu)
            _ = out.sum()

    # Warmup
    run_unpinned()
    run_pinned()
    torch.cuda.synchronize()

    # Time Unpinned
    t0 = time.perf_counter()
    for _ in range(5):
        run_unpinned()
    torch.cuda.synchronize()
    unpinned_time = (time.perf_counter() - t0) / 5 * 1000

    # Time Pinned
    t0 = time.perf_counter()
    for _ in range(5):
        run_pinned()
    torch.cuda.synchronize()
    pinned_time = (time.perf_counter() - t0) / 5 * 1000

    speedup = unpinned_time / pinned_time
    pct = (1.0 - (pinned_time / unpinned_time)) * 100
    print(f"  Unpinned (pin_memory=False): {unpinned_time:.2f} ms per 50 batches", flush=True)
    print(f"  Pinned (pin_memory=True):    {pinned_time:.2f} ms per 50 batches", flush=True)
    print(f"  >> Setting 3 Speedup:        {speedup:.2f}x faster ({pct:.1f}% time saved in transfer)", flush=True)
    return speedup


# =============================================================================
# BENCHMARK 4: Combined Cumulative Speedup (All 3 Active vs. Vanilla Baseline)
# =============================================================================
def benchmark_combined(device):
    print("\n" + "=" * 85, flush=True)
    print("TEST 4: Combined Cumulative Speedup (All 3 Settings vs. Vanilla Baseline)", flush=True)
    print("=" * 85, flush=True)
    print("Simulating full client round (data loading + forward/backward + metrics):", flush=True)

    batch_size = 32
    seq_len = 128
    hidden_size = 768
    num_batches = 30

    x_cpu = torch.randn(num_batches * batch_size, seq_len, hidden_size)
    y_cpu = torch.randint(0, 2, (num_batches * batch_size,))
    ds = TensorDataset(x_cpu, y_cpu)

    dl_unpinned = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=False)
    dl_pinned = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = MockRobertaLayer(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    # --- Mode A: Vanilla Baseline ---
    # Strict FP32 + Sync every batch + Unpinned DataLoader
    def vanilla_pipeline():
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in dl_unpinned:
            x_gpu = x.to(device, non_blocking=False)
            y_gpu = y.to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_gpu)
            loss = loss_fn(logits, y_gpu)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_gpu).sum().item()
            total += y_gpu.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / num_batches, correct / total

    # --- Mode B: Fully Optimized Pipeline ---
    # TF32 Tensor Cores + Sync once at end + Pinned DataLoader with non_blocking=True
    def optimized_pipeline():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        running_loss = torch.zeros((), device=device)
        running_correct = torch.zeros((), device=device, dtype=torch.long)
        total = 0
        for x, y in dl_pinned:
            x_gpu = x.to(device, non_blocking=True)
            y_gpu = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_gpu)
            loss = loss_fn(logits, y_gpu)
            preds = torch.argmax(logits, dim=-1)
            running_correct += (preds == y_gpu).sum()
            total += y_gpu.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach()
        avg_loss = (running_loss / num_batches).item()
        acc = running_correct.item() / total
        return avg_loss, acc

    # Warmup
    vanilla_pipeline()
    optimized_pipeline()
    torch.cuda.synchronize()

    # Time Vanilla
    t0 = time.perf_counter()
    for _ in range(8):
        vanilla_pipeline()
    torch.cuda.synchronize()
    vanilla_time = (time.perf_counter() - t0) / 8 * 1000

    # Time Optimized
    t0 = time.perf_counter()
    for _ in range(8):
        optimized_pipeline()
    torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - t0) / 8 * 1000

    combined_speedup = vanilla_time / optimized_time
    combined_pct = (1.0 - (optimized_time / vanilla_time)) * 100

    print(f"  Vanilla Pipeline:         {vanilla_time:.2f} ms per client round")
    print(f"  Fully Optimized Pipeline: {optimized_time:.2f} ms per client round")
    print(f"  >> COMBINED SPEEDUP:      {combined_speedup:.2f}x FASTER ({combined_pct:.1f}% time saved overall)")
    return combined_speedup


def main():
    get_gpu_info()
    device = torch.device("cuda:0")
    
    s1 = benchmark_setting_1(device)
    s2 = benchmark_setting_2(device)
    s3 = benchmark_setting_3(device)
    s_comb = benchmark_combined(device)

    print("\n" + "=" * 85)
    print("FINAL BENCHMARK SUMMARY:")
    print(f"  Setting 1 (TF32 Tensor Cores):               {s1:.2f}x speedup")
    print(f"  Setting 2 (Remove Sync Stalls):              {s2:.2f}x speedup")
    print(f"  Setting 3 (Pinned Memory & DMA Transfers):   {s3:.2f}x speedup")
    print(f"  >>> TOTAL COMBINED ACCELERATION:             {s_comb:.2f}x FASTER (All 3 Combined)")
    print("=" * 85)


if __name__ == "__main__":
    main()
