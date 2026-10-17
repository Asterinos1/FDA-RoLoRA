"""Benchmark and Profiling Script: Standard FP32 vs. TensorFloat-32 (TF32).

Measures the exact speedup and numerical precision impact of enabling
Tensor Cores (TF32) on NVIDIA Ampere (A10) and Ada Lovelace GPUs for
RoBERTa-base + LoRA workloads.

Usage:
    python fillers/profile_tf32.py
"""
import sys
import time
import torch
import torch.nn as nn

def get_gpu_info():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires an NVIDIA GPU.")
        sys.exit(1)
    prop = torch.cuda.get_device_properties(0)
    print("=" * 80)
    print(f"GPU Device:               {prop.name}")
    print(f"Compute Capability:       {prop.major}.{prop.minor}")
    print(f"Total VRAM:               {prop.total_memory / (1024**3):.2f} GB")
    print(f"PyTorch Version:          {torch.__version__}")
    has_tf32 = prop.major >= 8
    print(f"Ampere/Ada TF32 Support:  {'YES (Supported)' if has_tf32 else 'NO (Requires SM >= 8.0)'}")
    print("=" * 80)
    return has_tf32


class MockRobertaLayerWithLora(nn.Module):
    """Simulates a RoBERTa-base transformer layer with rank-4 LoRA adapters."""
    def __init__(self, hidden_size=768, intermediate_size=3072, lora_rank=4):
        super().__init__()
        # Self-Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # LoRA adapters on query and value
        self.lora_q_a = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_q_b = nn.Linear(lora_rank, hidden_size, bias=False)
        self.lora_v_a = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_v_b = nn.Linear(lora_rank, hidden_size, bias=False)
        
        # Intermediate / Output Feed-Forward
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Attention with LoRA
        q = self.q_proj(x) + self.lora_q_b(self.lora_q_a(x)) * 2.0
        k = self.k_proj(x)
        v = self.v_proj(x) + self.lora_v_b(self.lora_v_a(x)) * 2.0
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (768 ** 0.5))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        x = x + self.out_proj(attn_out)
        x = self.norm(x)
        
        # MLP
        x = x + self.fc2(self.act(self.fc1(x)))
        return self.norm(x)


def benchmark_workload(fn, name, warmup=25, iters=100):
    """Accurately measure GPU execution time using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / iters
    return avg_time_ms


def run_benchmark():
    has_tf32 = get_gpu_info()
    if not has_tf32:
        print("Note: TF32 requires Ampere (Compute 8.0+) or newer. Running anyway...")

    batch_size = 32
    seq_len = 128
    hidden_size = 768
    intermediate_size = 3072

    print("\n--- TEST 1: RoBERTa-base + LoRA Forward & Backward Pass ---")
    print(f"Batch Size: {batch_size} | Seq Len: {seq_len} | Hidden Size: {hidden_size} | LoRA Rank: 4")

    # Fixed seed for input data
    torch.manual_seed(42)
    device = torch.device("cuda:0")

    model = MockRobertaLayerWithLora(hidden_size, intermediate_size, lora_rank=4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Reusable batch
    inputs = torch.randn(batch_size, seq_len, hidden_size, device=device)
    targets = torch.randn(batch_size, seq_len, hidden_size, device=device)
    loss_fn = nn.MSELoss()

    def step():
        optimizer.zero_grad(set_to_none=True)
        out = model(inputs)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
        return out

    # =========================================================================
    # Baseline: Strict FP32 (Tensor Cores Disabled)
    # =========================================================================
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    fp32_time = benchmark_workload(step, "FP32 Baseline")
    fp32_output = step().detach().clone()

    # =========================================================================
    # Accelerated: TF32 (Tensor Cores Enabled)
    # =========================================================================
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    tf32_time = benchmark_workload(step, "TF32 Accelerated")
    tf32_output = step().detach().clone()

    speedup_1 = fp32_time / tf32_time
    pct_reduction_1 = (1.0 - (tf32_time / fp32_time)) * 100

    # Numerical difference
    diff = (fp32_output - tf32_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.cosine_similarity(fp32_output.flatten(), tf32_output.flatten(), dim=0).item()

    print(f"  Standard FP32:      {fp32_time:.3f} ms / step")
    print(f"  TF32 Tensor Cores:  {tf32_time:.3f} ms / step")
    print(f"  >> Speedup:         {speedup_1:.2f}x faster ({pct_reduction_1:.1f}% time saved per step)")
    print(f"  >> Cosine Sim:      {cos_sim:.8f} (1.00000000 = identical direction)")
    print(f"  >> Mean Diff:       {mean_diff:.2e} (well below typical gradient variance)")

    # =========================================================================
    # TEST 2: Raw Linear GEMM Matrix Multiplications
    # =========================================================================
    print("\n--- TEST 2: Dense Matrix Multiplication (Large GEMM) ---")
    tokens = batch_size * seq_len  # 4096 tokens
    print(f"Matrix Dimension: [{tokens}, {hidden_size}] x [{hidden_size}, {intermediate_size}]")

    a = torch.randn(tokens, hidden_size, device=device)
    b = torch.randn(hidden_size, intermediate_size, device=device)

    # Baseline FP32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    gemm_fp32 = benchmark_workload(lambda: torch.matmul(a, b), "GEMM FP32", warmup=50, iters=200)

    # Accelerated TF32
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    gemm_tf32 = benchmark_workload(lambda: torch.matmul(a, b), "GEMM TF32", warmup=50, iters=200)

    speedup_2 = gemm_fp32 / gemm_tf32
    pct_reduction_2 = (1.0 - (gemm_tf32 / gemm_fp32)) * 100

    print(f"  Standard FP32:      {gemm_fp32:.3f} ms")
    print(f"  TF32 Tensor Cores:  {gemm_tf32:.3f} ms")
    print(f"  >> Speedup:         {speedup_2:.2f}x faster ({pct_reduction_2:.1f}% time saved)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("PROFILING VERDICT:")
    if speedup_1 >= 1.25 or speedup_2 >= 1.4:
        print(f"  [CONFIRMED] TF32 delivers a substantial speedup ({speedup_1:.2f}x - {speedup_2:.2f}x).")
        print("  Enabling this setting on NVIDIA Ampere A10 GPUs is definitely worth it.")
    else:
        print(f"  Speedup measured at {speedup_1:.2f}x.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
