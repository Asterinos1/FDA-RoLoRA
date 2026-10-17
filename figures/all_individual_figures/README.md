# All Individual Standalone Telemetry Figures

This folder contains **all individual standalone figures** corresponding to the telemetry and dynamics analyses presented in [`FDA_vs_Baselines_v2.md`](../../FDA_vs_Baselines_v2.md).

In `FDA_vs_Baselines_v2.md`, the telemetry is visualized primarily as 6-in-1 composite stacked grids (`*_all6_telemetry_stacked.png`) accompanied by the Train vs. Validation Accuracy Gap plot (`*_7_train_val_comparison.png`).

In this folder, **every single sub-plot is extracted, rendered independently, and saved separately and alone** at high resolution (180 DPI) with dedicated titles, readable axis labels, full legends, and peak performance milestone stars (`*`).

---

## Folder Statistics

- **Total Standalone Figures**: `224` PNG files
- **Configurations**: `16` GLUE benchmark parameter groups
- **Metrics per Configuration**: `7` individual telemetry curves
- **Comparative Suites**: `2` evaluation scopes (Master All Methods vs. FDA vs. RoLoRA)

---

## File Naming Convention

Each file follows the standardized format:

```text
{suite}_{cfg_index}_{dataset}_C{clients}_R{rounds}_LR{lr}_{metric_id}_{metric_name}.png
```

Example:
- `master_all_methods_01_MNLI_C20_R50_LR0.1_1_accuracy.png`
- `fda_vs_rolora_01_MNLI_C20_R50_LR0.1_1_accuracy.png`

---

## Comparative Suites

1. **`master_all_methods_*`** (112 figures):
   Comprehensive evaluation comparing all methods:
   - **FDA-RoLoRA** (`E3TO20`, `E3TO30`, and historical runs where applicable)
   - **RoLoRA** (`E20`, `E20 RK8`, `E10`, `E5`)
   - **LoRA** (`E20`, `E20 RK8`, `E10`, `E5`)
   - **FFA-LoRA** (`E10`, `E5`)

2. **`fda_vs_rolora_*`** (112 figures):
   Architectural ablation isolating the orthonormal random projection constraint:
   - **FDA-RoLoRA** (`E3TO20`, `E3TO30`, etc.)
   - **RoLoRA Baselines** (`E20`, `E20 RK8`, `E10`, `E5`)

---

## Standalone Metrics (1 through 7)

For each configuration and comparative suite, the 7 standalone figures correspond to:

| Metric ID | File Suffix | Metric Description |
| :---: | :--- | :--- |
| **1** | `1_accuracy.png` | **Validation Accuracy vs. Communication Round**: Tracks global validation accuracy across communication rounds with milestone peak stars (`*`). |
| **2** | `2_allocated_epochs.png` | **Allocated Local Epochs ($E_t$) vs. Round**: Shows dynamic local training budget alternating between Matrix A and Matrix B (shaded bars) vs. baseline fixed epochs. |
| **3** | `3_client_variance.png` | **Inter-Client Drift Variance ($\sigma^2_t$) vs. Round**: Decomposes client drift variance into Matrix A and Matrix B updates across rounds. |
| **4** | `4_accuracy_vs_cost.png` | **Validation Accuracy vs. Cumulative Communication Cost (MB)**: Tracks validation performance as a function of Float32 payload transferred over the network. |
| **5** | `5_fda_threshold.png` | **FDA Closed-Loop Dynamic Thresholds ($\theta_A, \theta_B$) vs. Round**: Telemetry of the adaptive exploration controller thresholds. |
| **6** | `6_variance_vs_epoch.png` | **Inter-Client Drift Variance vs. Cumulative Local Epochs**: Quantifies variance accumulation per unit of client compute budget ($\sum E_t$). |
| **7** | `7_train_val_comparison.png` | **Train vs. Validation Accuracy Gap (Dual-Curve Telemetry)**: Compares online client training accuracy (dashed) with global post-aggregation validation accuracy (solid). |

---

## Benchmark Configurations (1 through 16)

| Index | Benchmark / Task | Tier | Clients ($C$) | Rounds ($R$) | Learning Rate ($\text{lr}$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **01** | `MNLI` | Tier 1 | 20 | 50 | 0.1 |
| **02** | `MNLI` | Tier 1 | 20 | 50 | 0.01 |
| **03** | `MNLI` | Tier 1 | 50 | 30 | 0.1 |
| **04** | `MNLI` | Tier 1 | 50 | 30 | 0.01 |
| **05** | `QNLI` | Tier 1 | 20 | 50 | 0.1 |
| **06** | `QNLI` | Tier 1 | 20 | 50 | 0.01 |
| **07** | `QNLI` | Tier 1 | 50 | 30 | 0.1 |
| **08** | `QNLI` | Tier 1 | 50 | 30 | 0.01 |
| **09** | `QQP` | Tier 2 | 20 | 50 | 0.1 |
| **10** | `QQP` | Tier 2 | 50 | 30 | 0.1 |
| **11** | `SST2` | Tier 2 | 20 | 50 | 0.1 |
| **12** | `SST2` | Tier 2 | 50 | 30 | 0.1 |
| **13** | `MRPC` | Tier 3 | 20 | 30 | 0.05 |
| **14** | `MRPC` | Tier 3 | 50 | 12 | 0.05 |
| **15** | `RTE` | Tier 3 | 20 | 30 | 0.05 |
| **16** | `RTE` | Tier 3 | 50 | 12 | 0.05 |
