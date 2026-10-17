# Simulate scheduling of the 39 pending experiments across lanes

runs = [
    # STAGE 1: BATCH-32 IID (16 runs)
    # QQP C20 (4 runs, ~3.0h each)
    {"id": "QQP-C20-k1.0-s42", "ds": "qqp", "c": 20, "stage": 1, "hours": 3.0, "bs": 32},
    {"id": "QQP-C20-k1.0-s43", "ds": "qqp", "c": 20, "stage": 1, "hours": 3.0, "bs": 32},
    {"id": "QQP-C20-k0.5-s42", "ds": "qqp", "c": 20, "stage": 1, "hours": 3.0, "bs": 32},
    {"id": "QQP-C20-k0.5-s43", "ds": "qqp", "c": 20, "stage": 1, "hours": 3.0, "bs": 32},
    # QQP C50 (4 runs, ~10.0h each)
    {"id": "QQP-C50-k1.0-s42", "ds": "qqp", "c": 50, "stage": 1, "hours": 10.0, "bs": 32},
    {"id": "QQP-C50-k1.0-s43", "ds": "qqp", "c": 50, "stage": 1, "hours": 10.0, "bs": 32},
    {"id": "QQP-C50-k0.5-s42", "ds": "qqp", "c": 50, "stage": 1, "hours": 10.0, "bs": 32},
    {"id": "QQP-C50-k0.5-s43", "ds": "qqp", "c": 50, "stage": 1, "hours": 10.0, "bs": 32},
    # QNLI C20 (2 runs, ~8.5h each)
    {"id": "QNLI-C20-k1.0-s43", "ds": "qnli", "c": 20, "stage": 1, "hours": 8.5, "bs": 32},
    {"id": "QNLI-C20-k0.5-s43", "ds": "qnli", "c": 20, "stage": 1, "hours": 8.5, "bs": 32},
    # QNLI C50 (2 runs, ~16.0h each)
    {"id": "QNLI-C50-k1.0-s43", "ds": "qnli", "c": 50, "stage": 1, "hours": 16.0, "bs": 32},
    {"id": "QNLI-C50-k0.5-s43", "ds": "qnli", "c": 50, "stage": 1, "hours": 16.0, "bs": 32},
    # MNLI C20 (2 runs, ~20.0h each)
    {"id": "MNLI-C20-k1.0-s43", "ds": "mnli", "c": 20, "stage": 1, "hours": 20.0, "bs": 32},
    {"id": "MNLI-C20-k0.5-s43", "ds": "mnli", "c": 20, "stage": 1, "hours": 20.0, "bs": 32},
    # MNLI C50 (2 runs, ~13.8h each)
    {"id": "MNLI-C50-k1.0-s43", "ds": "mnli", "c": 50, "stage": 1, "hours": 13.8, "bs": 32},
    {"id": "MNLI-C50-k0.5-s43", "ds": "mnli", "c": 50, "stage": 1, "hours": 13.8, "bs": 32},

    # STAGE 2: BATCH-32 NON-IID (15 runs)
    # QNLI C20 Non-IID (3 runs, ~27.0h each)
    {"id": "QNLI-C20-A0.5-RoLoRA-s43", "ds": "qnli", "c": 20, "stage": 2, "hours": 27.0, "bs": 32},
    {"id": "QNLI-C20-A0.5-k1.0-s43", "ds": "qnli", "c": 20, "stage": 2, "hours": 27.0, "bs": 32},
    {"id": "QNLI-C20-A0.5-k0.5-s43", "ds": "qnli", "c": 20, "stage": 2, "hours": 27.0, "bs": 32},
    # QNLI C50 Non-IID (4 runs, ~18.0h each)
    {"id": "QNLI-C50-A0.5-RoLoRA-s42", "ds": "qnli", "c": 50, "stage": 2, "hours": 18.0, "bs": 32},
    {"id": "QNLI-C50-A0.5-RoLoRA-s43", "ds": "qnli", "c": 50, "stage": 2, "hours": 18.0, "bs": 32},
    {"id": "QNLI-C50-A0.5-k0.5-s42", "ds": "qnli", "c": 50, "stage": 2, "hours": 18.0, "bs": 32},
    {"id": "QNLI-C50-A0.5-k0.5-s43", "ds": "qnli", "c": 50, "stage": 2, "hours": 18.0, "bs": 32},
    # MNLI C20 Non-IID (4 runs, ~24.5h each)
    {"id": "MNLI-C20-A0.5-k1.0-s42", "ds": "mnli", "c": 20, "stage": 2, "hours": 24.5, "bs": 32},
    {"id": "MNLI-C20-A0.5-RoLoRA-s43", "ds": "mnli", "c": 20, "stage": 2, "hours": 24.5, "bs": 32},
    {"id": "MNLI-C20-A0.5-k1.0-s43", "ds": "mnli", "c": 20, "stage": 2, "hours": 24.5, "bs": 32},
    {"id": "MNLI-C20-A0.5-k0.5-s43", "ds": "mnli", "c": 20, "stage": 2, "hours": 24.5, "bs": 32},
    # MNLI C50 Non-IID (4 runs, ~16.0h each)
    {"id": "MNLI-C50-A0.5-RoLoRA-s42", "ds": "mnli", "c": 50, "stage": 2, "hours": 16.0, "bs": 32},
    {"id": "MNLI-C50-A0.5-RoLoRA-s43", "ds": "mnli", "c": 50, "stage": 2, "hours": 16.0, "bs": 32},
    {"id": "MNLI-C50-A0.5-k0.5-s42", "ds": "mnli", "c": 50, "stage": 2, "hours": 16.0, "bs": 32},
    {"id": "MNLI-C50-A0.5-k0.5-s43", "ds": "mnli", "c": 50, "stage": 2, "hours": 16.0, "bs": 32},

    # STAGE 3: BATCH-64 SST-2 (8 runs)
    # SST-2 C20 (4 runs, ~5.5h each)
    {"id": "SST2-C20-k1.0-s42", "ds": "sst2", "c": 20, "stage": 3, "hours": 5.5, "bs": 64},
    {"id": "SST2-C20-k1.0-s43", "ds": "sst2", "c": 20, "stage": 3, "hours": 5.5, "bs": 64},
    {"id": "SST2-C20-k0.5-s42", "ds": "sst2", "c": 20, "stage": 3, "hours": 5.5, "bs": 64},
    {"id": "SST2-C20-k0.5-s43", "ds": "sst2", "c": 20, "stage": 3, "hours": 5.5, "bs": 64},
    # SST-2 C50 (4 runs, ~9.0h each)
    {"id": "SST2-C50-k1.0-s42", "ds": "sst2", "c": 50, "stage": 3, "hours": 9.0, "bs": 64},
    {"id": "SST2-C50-k1.0-s43", "ds": "sst2", "c": 50, "stage": 3, "hours": 9.0, "bs": 64},
    {"id": "SST2-C50-k0.5-s42", "ds": "sst2", "c": 50, "stage": 3, "hours": 9.0, "bs": 64},
    {"id": "SST2-C50-k0.5-s43", "ds": "sst2", "c": 50, "stage": 3, "hours": 9.0, "bs": 64},
]

import heapq

def simulate(num_b32_lanes, num_b64_lanes, order_key=None):
    # order_key can sort the runs
    st1 = [r for r in runs if r['stage'] == 1]
    st2 = [r for r in runs if r['stage'] == 2]
    st3 = [r for r in runs if r['stage'] == 3]

    if order_key:
        st1.sort(key=order_key)
        st2.sort(key=order_key)
        st3.sort(key=order_key)

    # Simulate Stage 1 & 2 continuously through b32 lanes
    # Heap of lane free times
    lanes = [0.0] * num_b32_lanes
    heapq.heapify(lanes)

    # Run stage 1
    s1_lanes = [0.0] * num_b32_lanes
    heapq.heapify(s1_lanes)
    for r in st1:
        t = heapq.heappop(s1_lanes)
        heapq.heappush(s1_lanes, t + r['hours'])
    s1_end = max(s1_lanes)

    # If running stage 1 then stage 2 strictly or pipelined
    # In our scheduler, stage 2 tasks backfill as stage 1 tasks finish!
    pipeline_lanes = [0.0] * num_b32_lanes
    heapq.heapify(pipeline_lanes)
    for r in st1 + st2:
        t = heapq.heappop(pipeline_lanes)
        heapq.heappush(pipeline_lanes, t + r['hours'])
    b32_end = max(pipeline_lanes)

    # Now stage 3 (SST-2) drains through num_b64_lanes
    b64_lanes = [b32_end] * num_b64_lanes
    heapq.heapify(b64_lanes)
    for r in st3:
        t = heapq.heappop(b64_lanes)
        heapq.heappush(b64_lanes, t + r['hours'])
    s3_end = max(b64_lanes)

    return {
        "s1_isolated_wall_clock": s1_end,
        "b32_total_wall_clock": b32_end,
        "s3_duration": s3_end - b32_end,
        "total_wall_clock": s3_end
    }

print("=== CONFIGURATION 1: 12 LANES TOTAL (6 JOBS PER GPU) ===")
res12 = simulate(12, 4)
print(f"Stage 1 (Batch-32 IID): ~{res12['s1_isolated_wall_clock']:.1f} hours")
print(f"Stage 1 + Stage 2 (All Batch-32): ~{res12['b32_total_wall_clock']:.1f} hours")
print(f"Stage 3 (SST-2, 4 lanes): ~{res12['s3_duration']:.1f} hours")
print(f"Total Wall-Clock Time: ~{res12['total_wall_clock']:.1f} hours")

print("\n=== CONFIGURATION 2: 6 LANES TOTAL (3 JOBS PER GPU) ===")
res6 = simulate(6, 2)
print(f"Stage 1 (Batch-32 IID): ~{res6['s1_isolated_wall_clock']:.1f} hours")
print(f"Stage 1 + Stage 2 (All Batch-32): ~{res6['b32_total_wall_clock']:.1f} hours")
print(f"Stage 3 (SST-2, 2 lanes): ~{res6['s3_duration']:.1f} hours")
print(f"Total Wall-Clock Time: ~{res6['total_wall_clock']:.1f} hours")

# Let's test orderings:
# Does Shortest-Job-First (SJF) or Longest-Job-First (LJF) change the total makespan?
res12_ljf = simulate(12, 4, order_key=lambda r: -r['hours'])
print(f"\n12 lanes with Longest-Job-First: ~{res12_ljf['total_wall_clock']:.1f} hours")
res12_sjf = simulate(12, 4, order_key=lambda r: r['hours'])
print(f"12 lanes with Shortest-Job-First: ~{res12_sjf['total_wall_clock']:.1f} hours")
