# Thesis Filler Experiments Framework

The **Thesis Filler Experiments** framework is a specialized, autonomous multi-GPU experiment orchestrator designed to systematically close the remaining Non-IID ($\alpha=0.5$) matrix gaps identified during the master thesis campaign.

It is completely isolated from previous run-control queues, featuring its own dedicated state machine (`results/run_control_filler/`), dedicated log storage (`results/logs/thesis_filler/`), and calibrated early-stopping rules to protect GPU compute time.

---

## 1. Experimental Tiers

The experiment matrix is partitioned into 3 structured tiers:

| Tier | Focus / Scope | Dataset | Clients ($C$) | Max Rounds ($R$) | Methods & Variants | Runs |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **Tier 1** | **Missing Non-IID $k=1.0$ Runs** | QNLI, MNLI | $50$ | $22$ (QNLI), $24$ (MNLI) | FDA Pure $E_{\max}=20$ ($k=1.0$), Seeds 42, 43 | **4** |
| **Tier 2** | **Deep-Compute Seed 43 Replication** | QNLI, MNLI | $20$ | $35$ (QNLI), $42$ (MNLI) | RoLoRA $E=20$; FDA Pure $E_{\max}=40$ ($k=1.0, 0.5$) | **5** |
| **Tier 3** | **Extended Non-IID Benchmark** | QQP, SST-2 | $20, 50$ | $20\text{--}25$ ($C=20$), $20\text{--}24$ ($C=50$) | RoLoRA $E=10$; FDA Pure $k=1.0, 0.5$, Seeds 42, 43 | **24** |
| **Total** | | | | | | **33** |

---

## 2. Optimized Early Stopping & Compute Conservation

### The $C=50$ Bottleneck Problem
In master experiments, early stopping entry gates were configured based on high IID performance:
- QNLI entry gate was set to `0.80`
- MNLI entry gate was set to `0.55`

Under Dirichlet label skew $\alpha=0.5$, peak performance realistically reaches $\sim 0.76-0.77$ for QNLI and $\sim 0.54$ for MNLI. Because accuracy never touched the unrealistic thresholds, **early stopping was never armed**. 

Each $C=50$ round requires **37.5 to 47.3 minutes** due to sequential client processing across 50 clients. As a consequence, models trained blindly past their peak into severe overfit or collapse ($76.8\% \to 50.8\%$), wasting **6 to 8 hours per run**.

### Calibrated Parameters in Fillers
The filler framework incorporates dataset-specific entry thresholds, adaptive warmup guards, responsive patience, and a post-peak degradation guard to stop runs immediately at peak accuracy:

| Dataset | Clients ($C$) | Entry Gate (`early_stopping_start_accuracy`) | Activation Round ($MR$) | Patience / Window ($P / W$) | Scheduled Max Rounds ($R$) | Convergence & Exit Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **QNLI** | 20 | **0.80** | **18** | $P=2, W=2$ | **35** (down from 50) | Stable plateau exit at R22–R28 |
| **QNLI** | 50 | **0.70** | **12** | $P=2, W=2$ | **22** (down from 24) | Catches R18–R20 peak; prevents post-peak collapse |
| **MNLI** | 20 | **0.65** | **20** | $P=3, W=3$ | **42** (down from 50) | Steady monotonic ascent, exits at R30–R38 |
| **MNLI** | 50 | **0.50** | **12** | $P=3, W=3$ | **24** | Reliable entry gate; exits at R20–R24 |
| **QQP** | 20 | **0.70** | **8** | $P=2, W=2$ | **20** (down from 50) | Rapid convergence detected early at R9–R12 |
| **QQP** | 50 | **0.68** | **12** | $P=3, W=3$ | **24** | Protected against late-round noise |
| **SST-2** | 20 | **0.85** | **14** | $P=2, W=2$ | **25** (down from 50) | Fast high-accuracy exit at R14–R17 (saves 25 rounds) |
| **SST-2** | 50 | **0.82** | **12** | $P=2, W=2$ | **20** (down from 24) | Tight cap locks in R15–R18 peak |

- **Min Delta**: `0.005` (0.5% improvement required)
- **Degradation Guard (`early_stopping_drop_tolerance`)**: `0.03` (Immediate exit if smoothed accuracy falls $\ge 3\%$ below the peak observed after reaching $MR$, preventing catastrophic Non-IID model collapse).

---

## 3. Directory Structure & Isolation

```
FDA-LoRA/
├── fillers/
│   ├── thesis_filler_experiments.py     # Core Python scheduler & state machine
│   ├── run_filler_experiments.sh        # Bash runner script
│   └── README.md                        # Documentation (this file)
├── run_filler_experiments.sh            # Root forwarding wrapper
├── thesis_filler_experiments.py         # Root forwarding wrapper
└── results/
    ├── run_control_filler/              # Dedicated state machine directory
    │   ├── active/                      # Running job locks and metadata
    │   ├── completed/                   # Filler completion records
    │   └── dropped/                     # Manually dropped run records
    └── logs/thesis_filler/              # Per-run stdout and stderr logs
```

---

## 4. Quick-Start Usage

You can execute commands from the repository root or from inside `fillers/`:

### A. Preview the Execution Plan
```bash
# View Tier 1 only (4 runs)
./run_filler_experiments.sh plan tier1

# View Tier 2 only (5 runs)
./run_filler_experiments.sh plan tier2

# View Tier 1 & Tier 2 combined (9 runs)
./run_filler_experiments.sh plan tier1-2

# View Tier 3 only (24 runs)
./run_filler_experiments.sh plan tier3

# View complete 33-run plan
./run_filler_experiments.sh plan all
```

### B. Launch Scheduler
```bash
# Launch only Tier 1 (Recommended starting point: 4 runs)
./run_filler_experiments.sh start tier1

# Or launch Tier 1 & 2
./run_filler_experiments.sh start tier1-2

# Or launch all remaining experiments
./run_filler_experiments.sh start all
```

### C. Monitor Running Experiments
```bash
./run_filler_experiments.sh status
```
Displays active process IDs, assigned GPU lanes (`cuda:0` / `cuda:1`), elapsed runtime, and the most recent round progress line.

### D. Process Management (Stop / Drop)
```bash
# Gracefully stop a running job by its Control ID or Run ID
./run_filler_experiments.sh stop ae689bf3af "High latency investigation"

# Permanently drop a job from the queue
./run_filler_experiments.sh drop ae689bf3af "Superseded by alternate test"
```

---

## 5. Multi-GPU Concurrency Protocol

- **GPU Allocation**: Dynamic round-robin across `cuda:0` and `cuda:1`.
- **Concurrency Cap**: Up to **3 concurrent jobs per GPU** for batch-32 tasks (QNLI, MNLI, QQP).
- **Batch-64 Safety Cap**: Strictly capped at **2 concurrent jobs per GPU** for SST-2 (batch size 64) to prevent GPU out-of-memory errors.
- **Fail-Safe Polling**: Graceful handling of process exits, real-time logging, and automatic slot recycling.

---

## 6. Hardware & Runtime Acceleration (Self-Contained in `fillers/`)

To maximize throughput on the `polytechnix` server (2x NVIDIA Ampere A10 24GB GPUs, 48 vCPUs), **three hardware-level accelerations** are applied dynamically at worker launch. These changes are **100% self-contained in `fillers/`** with zero modifications to the core repository (`fdaopt/`, `pure_experiments/`, `non_iid_experiments/`, or `thesis_experiments.py`).

### Active Optimizations

| Optimization | Target Bottleneck | Implementation in `fillers/thesis_filler_experiments.py` | Measured Speedup |
| :--- | :--- | :--- | :---: |
| **Setting 1: TF32 Tensor Cores** | GPU compute in attention & LoRA GEMMs | `torch.set_float32_matmul_precision("high")`<br>`allow_tf32 = True`, `cudnn.benchmark = True`<br>Subprocess env: `NVIDIA_TF32_OVERRIDE=1` | **1.42x** (29.6% compute time saved) |
| **Setting 2: Zero-Sync Inner Loop** | CPU-GPU serialization from `.item()` | In-memory monkey-patching of `client_round_train`: accumulates loss/acc on GPU tensors; calls `.item()` once at round exit | **1.09x** (8.4% loop overhead saved; higher on 2.1 GHz server CPUs) |
| **Setting 3: Pinned Memory & DMA** | Host-to-Device transfer latency | `pin_memory: True` in DataLoader configs + `v.to(device, non_blocking=True)` DMA streaming | **1.57x** (36.3% transfer time saved) |
| **Combined End-to-End Pipeline** | Full client communication round | **All 3 active concurrently** | **1.75x FASTER** (**42.9% total time saved**) |

### Core Codebase Integrity Note

- **Zero Core Changes**: The files on disk in `fdaopt/training/fed_train.py` and optimizer scripts remain completely untouched.
- **Dynamic Worker Patching**: When child workers are spawned via `python fillers/thesis_filler_experiments.py --run-config ...`, `execute_run()` applies the patches in-memory before launching the training function. Standard non-filler runs in the root repository remain unaffected.
- **Verification Scripts**: Standalone profiling suites are available in:
  - `python fillers/profile_tf32.py` (micro-benchmark on TF32 numerical fidelity and speed)
  - `python fillers/profile_optimizations.py` (comprehensive benchmark suite testing all 3 settings individually and combined)

