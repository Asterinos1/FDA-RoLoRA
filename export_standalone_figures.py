"""
Script to collect and save every individual standalone telemetry figure used in FDA_vs_Baselines_v2.md
into a single dedicated folder: figures/all_individual_figures/

Each composite 6-panel figure used in the Markdown report is broken down into its constituent
standalone sub-figures (Metrics 1 through 6), alongside the companion Metric 7 (Train vs Validation Gap).

Total figures collected: 224
  - 16 Parameter Configurations across GLUE benchmark (MNLI, QNLI, QQP, SST-2, MRPC, RTE)
  - 7 Standalone Metrics per configuration:
      1. Validation Accuracy vs Communication Round
      2. Allocated Local Epochs (E_t) vs Communication Round
      3. Inter-Client Drift Variance vs Communication Round
      4. Validation Accuracy vs Cumulative Communication Cost (MB)
      5. FDA Closed-Loop Dynamic Thresholds (theta_A, theta_B) vs Communication Round
      6. Client Drift Variance vs Cumulative Local Epochs (Compute Budget)
      7. Train vs Validation Accuracy Gap (Dual-Curve Telemetry)
  - 2 Comparative Suites:
      - master_all_methods (FDA-RoLoRA, RoLoRA, LoRA, FFA-LoRA) [112 figures]
      - fda_vs_rolora (FDA-RoLoRA vs RoLoRA baselines) [112 figures]
"""

import os
import shutil

CONFIGS = [
    ("01_MNLI_C20_R50_LR0.1", "MNLI (C=20, R=50, lr=0.1)"),
    ("02_MNLI_C20_R50_LR0.01", "MNLI (C=20, R=50, lr=0.01)"),
    ("03_MNLI_C50_R30_LR0.1", "MNLI (C=50, R=30, lr=0.1)"),
    ("04_MNLI_C50_R30_LR0.01", "MNLI (C=50, R=30, lr=0.01)"),
    ("05_QNLI_C20_R50_LR0.1", "QNLI (C=20, R=50, lr=0.1)"),
    ("06_QNLI_C20_R50_LR0.01", "QNLI (C=20, R=50, lr=0.01)"),
    ("07_QNLI_C50_R30_LR0.1", "QNLI (C=50, R=30, lr=0.1)"),
    ("08_QNLI_C50_R30_LR0.01", "QNLI (C=50, R=30, lr=0.01)"),
    ("09_QQP_C20_R50_LR0.1", "QQP (C=20, R=50, lr=0.1)"),
    ("10_QQP_C50_R30_LR0.1", "QQP (C=50, R=30, lr=0.1)"),
    ("11_SST2_C20_R50_LR0.1", "SST-2 (C=20, R=50, lr=0.1)"),
    ("12_SST2_C50_R30_LR0.1", "SST-2 (C=50, R=30, lr=0.1)"),
    ("13_MRPC_C20_R30_LR0.05", "MRPC (C=20, R=30, lr=0.05)"),
    ("14_MRPC_C50_R12_LR0.05", "MRPC (C=50, R=12, lr=0.05)"),
    ("15_RTE_C20_R30_LR0.05", "RTE (C=20, R=30, lr=0.05)"),
    ("16_RTE_C50_R12_LR0.05", "RTE (C=50, R=12, lr=0.05)"),
]

METRICS = [
    ("1_accuracy.png", "Validation Accuracy vs. Communication Round"),
    ("2_allocated_epochs.png", "Allocated Local Epochs (E_t) vs. Communication Round"),
    ("3_client_variance.png", "Inter-Client Drift Variance vs. Communication Round"),
    ("4_accuracy_vs_cost.png", "Validation Accuracy vs. Cumulative Communication Cost (MB)"),
    ("5_fda_threshold.png", "FDA Closed-Loop Dynamic Thresholds vs. Communication Round"),
    ("6_variance_vs_epoch.png", "Client Drift Variance vs. Cumulative Local Epochs"),
    ("7_train_val_comparison.png", "Train vs. Validation Accuracy Gap (Dual-Curve Telemetry)"),
    ("8_log_client_variance.png", "Inter-Client Drift Variance [Log-Scale] vs. Communication Round"),
]

SUITES = [
    ("master_all_methods", "figures/individual_plots/master_all_methods", "Section 1: Master All Methods (FDA, RoLoRA, LoRA, FFA-LoRA)"),
    ("fda_vs_rolora", "figures/individual_plots/fda_vs_rolora", "Section 2: FDA-RoLoRA vs. RoLoRA Baselines"),
]


def export_all(dest_dir="figures/all_individual_figures"):
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    missing = []

    for suite_key, src_dir, suite_title in SUITES:
        for cfg_prefix, cfg_title in CONFIGS:
            for metric_suffix, metric_title in METRICS:
                src_file = os.path.join(src_dir, f"{cfg_prefix}_{metric_suffix}")
                dest_file = os.path.join(dest_dir, f"{suite_key}_{cfg_prefix}_{metric_suffix}")
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dest_file)
                    count += 1
                else:
                    missing.append(src_file)

    print(f"Exported {count} standalone figures to '{dest_dir}'.")
    if missing:
        print(f"WARNING: {len(missing)} files were not found: {missing}")
    return count


if __name__ == "__main__":
    export_all()
