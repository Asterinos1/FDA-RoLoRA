# Pure FDA-RoLoRA Experiment Suite (Polytechnix Deployment)

This directory contains the self-contained, isolated experiment suite for **Pure Closed-Loop FDA-RoLoRA Dual-Track**, featuring:
1. **Starting with Matrix B First (`BSTART`)**: Parity with standard RoLoRA initialization ($B=0$).
2. **Pure Drift Metric (`NODISC`)**: Genuine physical inter-client drift variance ($\gamma = 1.0$, zero heuristic divisors).
3. **No Minimum Epoch Clamp ($e_{\min} = 1$)**: Allows adaptive truncation from epoch 1 onwards.
4. **Zero First-Round Guard (`NOGUARD`)**: Natural cold-start bootstrap from initial $\Theta_A = 0.0, \Theta_B = 0.0$.
5. **Full 4-Dataset Benchmark Grid**: `QQP`, `QNLI`, `MNLI`, and `SST-2` across both $C \in \{20, 50\}$ for $k \in \{1.0, 0.5\}$ and Seeds $S \in \{42, 43\}$ (32 total configurations, 8 already completed, 24 to run).
6. **Staged Memory-Safe Scheduling**:
   - **Stage 1 (Batch-32 Sprint)**: `QQP`, `QNLI`, and `MNLI` are scheduled first across **6 parallel jobs/GPU (12 parallel slots total)** on Polytechnix A10s, maximizing GPU saturation.
   - **Stage 2 (Batch-64 Draining)**: `SST-2` queues at the end and dynamically backfills into freed lanes, strictly abiding by a safety cap of **2 jobs/GPU** to eliminate any risk of CUDA Out-Of-Memory (OOM).

---

## Expected Wall-Clock Runtimes

* **Stage 1 (Batch-32 Heavyweights: QQP, QNLI, MNLI - 16 runs)**: $\approx 16\text{--}18\text{ hours}$ across 12 concurrent lanes.
* **Stage 2 (Batch-64 Backfill: SST-2 - 8 runs)**: $\approx 12\text{--}14\text{ hours}$ across 4 lanes (overlaps with Stage 1 draining).
* **Total Expected Completion Time**: **$\approx 28\text{--}32\text{ wall-clock hours}$ ($\approx 1.2\text{ calendar days}$)**.

---

## Files in this Suite

| File | Description |
| :--- | :--- |
| `pure_experiments/fda_rolora_pure.py` | Standalone implementation of `fda_rolora_dual_track_pure_opt`. |
| `pure_experiments/thesis_pure_experiments.py` | Master orchestrator supporting planning, staged GPU scheduling, active run registry, and process control. |
| `pure_experiments/run_pure_experiments.sh` | Bash launcher for the Ubuntu GPU cluster with concurrency defaults (6 jobs/GPU batch-32, 2 jobs/GPU batch-64). |
| `run_pure_experiments.sh` | Convenient root launcher dispatching directly to `pure_experiments/run_pure_experiments.sh`. |

---

## Instructions for Polytechnix Deployment

### Step 1: Make Bash Scripts Executable
```bash
chmod +x run_pure_experiments.sh pure_experiments/run_pure_experiments.sh
```

### Step 2: View the Execution Plan (Shows 24 Pending Runs)
```bash
./run_pure_experiments.sh plan
```

### Step 3: Launch in Background
```bash
nohup ./run_pure_experiments.sh start > pure_suite.log 2>&1 &
```

### Step 4: Monitor Live Progress
```bash
# Check running jobs, GPU assignment, elapsed time, and latest round
./run_pure_experiments.sh status
```

### Step 5: Process Control (Optional)
```bash
# Gracefully stop a run by its Control ID (from plan/status)
./run_pure_experiments.sh stop <CONTROL_ID>

# Drop a run permanently so the scheduler skips it
./run_pure_experiments.sh drop <CONTROL_ID>
```

---

## Distinct Run Identifier (`EXTRA`) Specification

Every pure run generates an isolated, unambiguous identifier:
```text
THESIS-RBASE-PURE-DS-{dataset}-C{clients}-R{rounds}-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO{ratio}-NODISC-BSTART-NOGUARD-S{seed}-ES{gate}-MR6-P3-D0.005-W3
```
* **`PURE`**: Pure dual-track implementation with zero artificial bounds.
* **`E1TO20`**: Dynamic epoch search bounds $e_{\min}=1$ to $e_{\max}=20$ (no minimum epoch clamping).
* **`CRATIO{ratio}`**: Proportional controller setpoint ratio ($k=0.5$ or $k=1.0$).
* **`NODISC`**: Pure un-discounted subspace drift variance ($\gamma = 1.0$, no heuristic scaling).
* **`BSTART`**: Alternation starts with Matrix $\mathbf{B}$ first at Round 1 (Round index 0).
* **`NOGUARD`**: Zero first-round forced epoch guard; natural cold-start bootstrap from $\Theta_{\text{init}}=0.0$.
* **Metrics CSV:** Saved to `results/round_metrics/{EXTRA}_YYYYMMDD_HHMMSS.csv`
* **Worker Logs:** Saved to `results/logs/thesis_pure/{EXTRA}.stdout.log`
* **Run Control:** Managed inside `results/run_control_pure/`

Zero existing baseline files or previous calibration metrics are touched or overwritten.
