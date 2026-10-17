# Non-IID Dirichlet (alpha=0.5) Experiment Suite (Polytechnix Deployment)

This directory contains the self-contained experiment suite for evaluating **FDA-RoLoRA Dual-Track** against **FedAvg-RoLoRA** under real-world **Dirichlet data heterogeneity ($\alpha=0.5$)**.

---

## 1. Experimental Protocol

1. **Non-IID Partitioning:** Dirichlet distribution $\alpha = 0.5$ across $C=20$ clients ($K=20$ full participation).
2. **Methods Evaluated:**
   - **`FedAvg-RoLoRA`**: Fixed local epochs $E=10$ and $E=20$ (literature baselines).
   - **`FDA-RoLoRA-DualTrack-Pure`**: Closed-loop adaptive local epochs with convexity-compensated setpoint ratio $k=1.0$:
     - **$E_{\max} = 20, k = 1.0$**: Matches baseline $E=10$ compute ($\bar{e} \approx 11.4$).
     - **$E_{\max} = 40, k = 1.0$**: Matches baseline $E=20$ deep compute ($\bar{e} \approx 22.8$).
3. **Alternation Protocol:** Matrix $\mathbf{B}$ starts first at Round 1 ($r=0$), Matrix $\mathbf{A}$ at Round 2 ($r=1$).
4. **Drift Telemetry:** True physical inter-client drift variance ($\gamma = 1.0$, `NODISC`, no heuristic divisors) is recorded every round for both methods.
5. **Robust Non-IID Early Stopping:**
   - **Strict 60% Round Delay ($R_{\min}=30$ for $R=50$):** Early stopping cannot trigger during Rounds 1–30.
   - **5-Round Moving Average Window:** Eliminates spurious client sampling noise from Dirichlet skew.
   - **Patience:** 3 consecutive window evaluations without $\ge 0.5\%$ gain.
   - **Baseline Gates:** $80\%$ for QNLI, $60\%$ for MNLI.
6. **Per-Round Metric Persistence:** Metrics are incrementally saved to CSV after every single round.

---

## 2. Continuous 6-Lane GPU Saturation

On Polytechnix (2 $\times$ 24GB A10 GPUs, 3 jobs/GPU = **6 concurrent lanes**):
- **Initial Dispatch ($t=0$):**
  - Lanes 1–4 immediately launch all **4 QNLI runs** (#1–4).
  - Lanes 5–6 immediately launch the first **2 MNLI runs** (#5–6).
  - **All 6 GPU lanes are 100% saturated from second zero.**
- **Dynamic Backfill:**
  - As soon as *any* lane completes (e.g. QNLI RoLoRA E10 finishes), it immediately pulls the next pending MNLI run (#7, #8).
- **Zero Inter-Stage Idle Time:**
  - The scheduler maintains **6 active jobs at all times** across datasets and tiers until the entire suite completes.

---

## 3. The 8-Run Experimental Grid ($C=20, R=50, \alpha=0.5$, Seed 42)

| # | Dataset | Method | Setting | Realized Duty | Scientific Purpose / Comparison |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **QNLI** | `FedAvg-RoLoRA` | Fixed $E=10$ | $E=10$ | **Tier 1 Baseline**: Standard literature baseline |
| **2** | **QNLI** | `FDA-RoLoRA-DualTrack-Pure` | $k=1.0$ ($e \in [1, 20]$) | $\bar{e} \approx 11.4$ | **Tier 1 Parity**: Convexity compensated, matches $E=10$ |
| **3** | **QNLI** | `FedAvg-RoLoRA` | Fixed $E=20$ | $E=20$ | **Tier 2 Baseline**: Deep compute ceiling, tests drift degradation |
| **4** | **QNLI** | `FDA-RoLoRA-DualTrack-Pure` | $k=1.0$ ($e \in [1, 40]$) | $\bar{e} \approx 22.8$ | **Tier 2 Parity**: Convexity compensated, matches $E=20$ |
| **5** | **MNLI** | `FedAvg-RoLoRA` | Fixed $E=10$ | $E=10$ | **Tier 1 Baseline**: Standard literature baseline on 3-class skew |
| **6** | **MNLI** | `FDA-RoLoRA-DualTrack-Pure` | $k=1.0$ ($e \in [1, 20]$) | $\bar{e} \approx 11.4$ | **Tier 1 Parity**: Convexity compensated, matches $E=10$ on MNLI |
| **7** | **MNLI** | `FedAvg-RoLoRA` | Fixed $E=20$ | $E=20$ | **Tier 2 Baseline**: Deep compute baseline, divergence check |
| **8** | **MNLI** | `FDA-RoLoRA-DualTrack-Pure` | $k=1.0$ ($e \in [1, 40]$) | $\bar{e} \approx 22.8$ | **Tier 2 Parity**: Convexity compensated, matches $E=20$ on MNLI |

---

## 4. Instructions for Polytechnix Deployment

### Step 1: Drag & Drop
Copy the `non_iid_experiments/` folder and `run_non_iid_experiments.sh` directly into the repository root on Polytechnix.

### Step 2: Make Scripts Executable
```bash
chmod +x run_non_iid_experiments.sh non_iid_experiments/run_non_iid_experiments.sh
```

### Step 3: View the Execution Plan
```bash
./run_non_iid_experiments.sh plan
```

### Step 4: Launch in Background
```bash
nohup ./run_non_iid_experiments.sh start > non_iid_suite.log 2>&1 &
```

### Step 5: Monitor Live Telemetry
```bash
# Check live PIDs, GPU device allocation, elapsed time, and latest [METRIC]
./run_non_iid_experiments.sh status
```

### Step 6: Operator Process Control (Optional)
```bash
# Gracefully stop a run by its Control ID (from plan or status)
./run_non_iid_experiments.sh stop <CONTROL_ID>

# Drop a run permanently so scheduler skips it
./run_non_iid_experiments.sh drop <CONTROL_ID>
```

---

## 5. Artifact & Log Locations

- **Metrics CSV:** `results/round_metrics/THESIS-NONIID-A0.5-...csv` (saved incrementally every round)
- **Worker Logs:** `results/logs/thesis_non_iid/{EXTRA}.stdout.log`
- **Error Logs:** `results/logs/thesis_non_iid/{EXTRA}.stderr.log`
- **Scheduler Control:** `results/run_control_non_iid/`
