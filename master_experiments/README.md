# Master Thesis Benchmark Suite (Polytechnix Deployment)

This directory contains the **Master Unified Orchestrator** designed to execute all required thesis benchmark runs over the weekend (Thursday Oct 8 to Monday Oct 12) without manual intervention.

---

## 1. Unified Benchmark Scope (52 Total Configurations)

1. **Stage 1: Batch-32 Pure IID Sprint** (16 pending runs)
   - Tasks: `QQP`, `QNLI`, `MNLI`
   - Settings: $C \in \{20, 50\}$, $k \in \{1.0, 0.5\}$, Seeds $S \in \{42, 43\}$
   - Protocol: Matrix B first, NODISC ($\gamma = 1.0$), $e_{\min}=1$, zero round-1 guard.
   - Concurrency: **3 parallel jobs/GPU (6 parallel slots total)** on Polytechnix A10s.

2. **Stage 2: Batch-32 Lean Non-IID Follow-Through** (15 pending runs)
   - Tasks: `QNLI`, `MNLI` under Dirichlet skew $\alpha = 0.5$
   - Settings: $C \in \{20, 50\}$, Seeds $S \in \{42, 43\}$
   - Methods: `FDA Pure E1TO20` ($k \in \{0.5, 1.0\}$) and `FedAvg-RoLoRA` (Fixed $E=10$)
   - **Rescues the crashed MNLI $C=20$ $k=1.0$ run!**

3. **Stage 3: Batch-64 IID Draining & Backfill** (8 pending runs)
   - Tasks: `SST-2` ($C \in \{20, 50\}$, $k \in \{1.0, 0.5\}$, Seeds $S \in \{42, 43\}$)
   - Concurrency Safety: Strictly capped at **2 jobs/GPU** to eliminate any risk of CUDA Out-Of-Memory (OOM).

---

## 2. Interactive Operator Control & Live Plateau Killing

Each round's metrics are saved to disk immediately upon completion (`metrics_handler.save_metrics()`). If you tail the logs or check `./run_master_experiments.sh status` and notice a model is plateauing (e.g. hovering around 80%–82% accuracy):

1. **Check Live Progress**:
   ```bash
   ./run_master_experiments.sh status
   ```
   *Displays active PIDs, assigned GPUs, elapsed time, current round, and latest validation accuracy.*

2. **Gracefully Stop a Plateaued Run**:
   ```bash
   ./run_master_experiments.sh stop <CONTROL_ID>
   ```
   *What happens:*
   - Sends `SIGTERM` to the worker process group.
   - All completed rounds are safely preserved in `results/round_metrics/*.csv`.
   - The scheduler **immediately dispatches the next queued configuration** into that freed GPU lane!
   - Zero GPU downtime, zero manual restart needed.

3. **Drop a Run Permanently** (Optional):
   ```bash
   ./run_master_experiments.sh drop <CONTROL_ID>
   ```

---

## 3. Quick-Start Commands

### Step 1: Make Scripts Executable
```bash
chmod +x run_master_experiments.sh master_experiments/run_master_experiments.sh
```

### Step 2: Preview the 52-Run Execution Manifest
```bash
./run_master_experiments.sh plan
```

### Step 3: Launch in Background
```bash
nohup ./run_master_experiments.sh start > master_suite.log 2>&1 &
```

### Step 4: Monitor Live Workers
```bash
./run_master_experiments.sh status
```
