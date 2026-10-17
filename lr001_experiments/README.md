# LR=0.01 RoLoRA vs. FDA-RoLoRA Experiments (Budget P2)

This directory contains standalone execution scripts to benchmark **RoLoRA vs. FDA-RoLoRA** using a stable learning rate $\eta = 0.01$ (with server SGD $\eta_g = 1.0$), addressing parameter drift jitter and variance observed at $\eta = 0.1$.

---

## Key Experimental Details

- **Methods**: RoLoRA vs. FDA-RoLoRA (DualTrack, Old Controller with $\kappa = 0.5$ and midpoint target $E_{\max}/2$).
- **Budget**: Budget $P2$ only (layers $7\text{--}11$ for `roberta-base`).
- **Datasets**: MNLI and QNLI (batch size = 32).
- **Data Partition**: Dirichlet IID ($\alpha = 1000.0$).
- **Epoch Tiers**:
  - $E=5 \longleftrightarrow [3, 10]$
  - $E=10 \longleftrightarrow [3, 20]$
  - $E=20 \longleftrightarrow [3, 40]$
- **Client Scales**:
  - $C=20$ clients ($R=50$ rounds)
  - $C=50$ clients ($R=30$ rounds)
- **Seeds**: 42, 43, 44 (3 seeds per configuration).
- **Lighter Early Stopping**:
  - Min Rounds: 10
  - Patience: 5 rounds
  - Min Delta: $0.2\%$ ($\Delta = 0.002$)
  - Window: 3 rounds
  - Accuracy Gates: MNLI $\ge 60\%$, QNLI $\ge 80\%$.

---

## Scheduling & Concurrency Features

1. **Cheapest Runs First (Default)**:
   - The queue sorts runs by compute cost ($\text{Rounds} \times \text{Epochs}$).
   - Fast $R=30$ ($C=50$) runs with $E=5$ run first, followed by $R=50$ with $E=5$, progressively advancing to heavier runs.
   - Initial convergence and accuracy feedback are available within the first 1–2 hours.
   - Use `--no-cheapest-first` if you want sequential dataset/client loop order.

2. **Continuous Dynamic Run Allocation**:
   - The scheduler operates an asynchronous worker pool with **3 concurrent lanes per GPU** (e.g., 6 simultaneous runs on 2 GPUs).
   - As soon as **any** individual job completes or terminates early via early stopping, that lane immediately picks up the next pending job from the queue.
   - GPUs never sit idle waiting for an entire batch of jobs to finish.

---

## Quick Start on GPU Server

```bash
# Make executable
chmod +x lr001_experiments/run_lr001_mnli_qnli.sh

# 1. Preview planned queue (Dry Run showing cheapest runs first)
./lr001_experiments/run_lr001_mnli_qnli.sh --dry-run

# 2. Pilot Wave (18 runs: MNLI, C=20, 3 seeds)
./lr001_experiments/run_lr001_mnli_qnli.sh --dataset mnli --clients 20

# 3. Run Full 72-Run Suite
./lr001_experiments/run_lr001_mnli_qnli.sh

# 4. Check Live Status of Running Workers (PIDs, GPUs, Live Metrics)
./lr001_experiments/run_lr001_mnli_qnli.sh status
# Or in Python:
# python lr001_experiments/run_lr001_mnli_qnli.py --status

# 5. Stop/Kill a Specific Run Gracefully
./lr001_experiments/run_lr001_mnli_qnli.sh stop <CONTROL_ID or PID>
# Or in Python:
# python lr001_experiments/run_lr001_mnli_qnli.py --stop <CONTROL_ID or PID>
```
