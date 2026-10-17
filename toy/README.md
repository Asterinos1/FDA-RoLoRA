# Pure FDA-RoLoRA Toy Model Experiments (Polytechnix Deployment)

This directory contains the self-contained, production-ready experiment suite for **Pure Closed-Loop FDA-RoLoRA Dual-Track** on the Toy Model benchmark ($d=784, r=16, c=10$ on MNIST).

---

## 1. Mathematical Protocol & Design Parity

Abiding strictly by the paper formulation (`rolora_verr4.pdf` Section 4.2) and the final FDA Pure design:

1. **Matrix Alternation (`BSTART`)**:
   - Round 1 (Round index 0): Matrix $\mathbf{B}$ is trained while $\mathbf{A}$ is frozen.
   - Round 2 (Round index 1): Matrix $\mathbf{A}$ is trained while $\mathbf{B}$ is frozen.
   - Alternates strictly every communication round (50 rounds active $B$, 50 rounds active $A$).
2. **Dead-Neuron Prevention Initialization**:
   - $\mathbf{W}_{\text{out}} \in \mathbb{R}^{d \times c}$ is drawn $\sim \mathcal{N}(0, 1)$ and permanently frozen.
   - $\mathbf{A} \in \mathbb{R}^{d \times r} \sim \mathcal{N}(0, 10^{-4})$.
   - $\mathbf{B} \in \mathbb{R}^{r \times d} \sim \mathcal{N}(0, 10^{-4})$. Small random initialization prevents dead ReLU neurons ($\text{ReLU}'(0) = 0$) while maintaining low initial drift.
3. **Pure Drift Metric (`NODISC`)**:
   - Genuine physical inter-client parameter drift variance:
     $$\sigma^2 = \frac{1}{|S|}\sum_{i \in S} \|\Delta_i - \bar{\Delta}\|^2$$
   - Pure scaling ($\gamma = 1.0$), with zero heuristic B-discount or artificial divisors.
4. **Natural Cold-Start Bootstrap (`NOGUARD`)**:
   - Discards forced first-round epoch guards; boots naturally from initial $\Theta_A = 0.0, \Theta_B = 0.0$.
5. **Closed-Loop Proportional Controller**:
   - Target Setpoint update rule:
     $$\Theta_{t+1} \leftarrow \frac{e^*_{\mathrm{target}}}{e_t} \cdot \sigma^2_t = \frac{\kappa \cdot e_{\max}}{e_t} \cdot \sigma^2_t$$
   - **Evaluated Setpoint Regimes**:
     - **Baseline-Matched Parity ($e_{\max}=10, \kappa=0.5$)**: Target setpoint $e^*_{\mathrm{target}} = 0.5 \times 10 = 5.0$ epochs. The controller regulates dynamically around **$\bar{e} \approx 5.0$ epochs/round**, matching the exact total compute budget ($500.0$ local epochs across 100 rounds) of the fixed $E=5$ baselines (**FedAvg-LoRA** and **RoLoRA**).
     - **Balanced Search ($e_{\max}=15, \kappa=0.5$)**: Target setpoint $e^*_{\mathrm{target}} = 0.5 \times 15 = 7.5$ epochs. Empirically regulates at $\bar{e} \approx 8.2$ epochs/round, achieving peak accuracy on nonlinear spaces ($81.96\%$).
     - **Ceiling Search ($e_{\max}=15, \kappa=1.0$)**: Target setpoint $e^*_{\mathrm{target}} = 1.0 \times 15 = 15.0$ epochs. Pushes local training toward $e_{\max}=15$, averaging $\bar{e} \approx 12.3 - 13.1$ epochs/round.
6. **4 Non-IID Experimental Scenarios**:
   - `10-Clients-1-Class_NONLIN` (10 clients, 1 class/client, ReLU MLP)
   - `10-Clients-1-Class_LIN` (10 clients, 1 class/client, Linear)
   - `5-Clients-2-Classes_NONLIN` (5 clients, 2 classes/client, ReLU MLP)
   - `5-Clients-2-Classes_LIN` (5 clients, 2 classes/client, Linear)
   - Random seeds: $\{42, 43, 44\}$.
   - Total rounds: $T = 100$.

---

## 2. Directory Structure & Launchers

```text
toy/
├── README.md                  # This deployment & execution guide
├── run_toy_emax10.sh          # Dedicated runner for Emax=10, k=0.5 (Target=5.0 epochs)
├── run_toy_experiments.sh     # Master multi-suite runner with emax10 presets
├── thesis_toy_experiments.py  # Multi-GPU orchestrator, scheduler & worker engine
└── analyze_toy.py             # Automated analysis, table generation & trajectory plotting
```

Root symlink/forwarding launchers:
```text
./run_toy_emax10.sh      -> toy/run_toy_emax10.sh
./run_toy.sh             -> toy/run_toy_experiments.sh
./run_toy_experiments.sh -> toy/run_toy_experiments.sh
```

---

## 3. Deployment Instructions for Polytechnix

### Step 1: Drag and Drop & Grant Permissions
Copy or rsync the `toy/` folder to your workspace on Polytechnix, then grant execute permissions:
```bash
chmod +x run_toy_emax10.sh run_toy_experiments.sh run_toy.sh toy/run_toy_emax10.sh toy/run_toy_experiments.sh
```

### Step 2: Running the $E_{\max}=10$ Suite ($\kappa \in \{1.0, 0.5\}$)

#### Option A: Using the Dedicated Runner (Recommended)
By default, `./run_toy_emax10.sh` schedules **all 24 runs** ($4\text{ scenarios} \times 2\text{ ratios } (\kappa=1.0, \kappa=0.5) \times 3\text{ seeds}$):
```bash
# 1. View execution plan (24 pending runs: Target=10.0 for k=1.0, Target=5.0 for k=0.5)
./run_toy_emax10.sh plan

# 2. Launch scheduler in the background on Polytechnix GPUs
nohup ./run_toy_emax10.sh start > toy_emax10_scheduler.log 2>&1 &

# 3. Monitor live status
./run_toy_emax10.sh status
```

To run only the baseline-matching $\kappa = 0.5$ setpoint (12 runs):
```bash
./run_toy_emax10.sh plan --ratios 0.5
nohup ./run_toy_emax10.sh start --ratios 0.5 > toy_emax10_scheduler.log 2>&1 &
```

#### Option B: Using General Runner Presets
```bash
# Plan all 24 Emax=10 runs
./run_toy_experiments.sh emax10

# Start in background
nohup ./run_toy_experiments.sh start-emax10 > toy_emax10_scheduler.log 2>&1 &

# Status
./run_toy_experiments.sh status-emax10
```

#### Option C: Custom Execution Flags
```bash
./run_toy_experiments.sh plan --emax 10 --ratios 1.0 0.5 --seeds 42 43 44
nohup ./run_toy_experiments.sh start --emax 10 --ratios 1.0 0.5 > toy_scheduler.log 2>&1 &
```

---

## 4. Live Monitoring & Process Management

### Live Status Watcher
```bash
watch -n 5 ./run_toy_emax10.sh status
```

### Inspect Worker Logs in Real Time
```bash
tail -f results/logs/thesis_toy/<CONTROL_ID>.stdout.log
```

### Stopping or Dropping a Run
To gracefully stop a run by its 10-character `CONTROL_ID`:
```bash
./run_toy_emax10.sh stop <CONTROL_ID>
```

To permanently remove a run from the scheduler:
```bash
./run_toy_emax10.sh drop <CONTROL_ID>
```

---

## 5. Post-Experiment Analysis & Graph Generation

Once runs complete, execute the automated analysis script:
```bash
python toy/analyze_toy.py
```

This will automatically:
1. Parse all baseline runs (`FedAvg-LoRA`, `RoLoRA`), legacy variants, and all FDA Pure runs ($e_{\max}=15$ and $e_{\max}=10$).
2. Compute Mean $\pm$ Std of final accuracy, peak accuracy, mean epochs per round ($\bar{e}$), mean epochs track A ($\bar{e}_A$), and mean epochs track B ($\bar{e}_B$).
3. Generate publication-quality 300 DPI trajectory curves with $\pm 1\sigma$ ribbons in `analysis/toy_pure_analysis/figures/`.
4. Export the complete scorecard to `analysis/toy_pure_analysis/README.md`.

---

## 6. Hardware & Concurrency Specifications

On Polytechnix (2x NVIDIA A10 GPUs):
- In-VRAM caching with pre-flattened float tensors minimizes GPU memory bandwidth overhead, achieving **~1.5 seconds per communication round**.
- 1 full 100-round run completes in **~2.5 minutes**.
- Multi-GPU concurrency default: `TOY_JOBS_PER_GPU=4` (8 parallel workers across 2x A10 GPUs).
- **Execution Time for 12 runs ($e_{\max}=10, \kappa=0.5$)**:
  $$\frac{12 \text{ runs}}{8 \text{ concurrent workers}} \times 2.5 \text{ min/run} \approx 4\text{--}5 \text{ wall-clock minutes}$$
- Output files are written to:
  - Metric CSVs: `results/round_metrics/TOY-PURE-FDA_PURE-*_EMAX10_CRATIO0.5_*.csv`
  - Execution Logs: `results/logs/thesis_toy/<CONTROL_ID>.stdout.log`
  - Active Registry: `results/run_control_toy/active/<CONTROL_ID>.json`
