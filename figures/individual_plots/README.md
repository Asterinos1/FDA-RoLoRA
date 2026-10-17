# Individual Plots: FDA vs. Baselines Comprehensive Telemetry

This directory contains standalone, high-resolution individual and stacked plots for all 12 configurations evaluated in `FDA_vs_Baselines_v2.md`.

## Folder Organization

- **`fda_vs_rolora/`**: Direct architectural head-to-head comparison between **FDA-RoLoRA** and **RoLoRA** baselines (E10 & E5), directly corresponding to Section 2 of `FDA_vs_Baselines_v2.md`.
- **`master_all_methods/`**: Head-to-head telemetry across all tested methods (**FDA-RoLoRA**, **RoLoRA E10/E5**, and standard **LoRA E10/E5**), directly corresponding to Section 1 of `FDA_vs_Baselines_v2.md`.

---

## File Naming Convention

Each configuration is prefixed with its standard order `[01-12]_[DATASET]_C[Clients]_R[Rounds]`:

1. `..._1_accuracy.png`: Communication rounds vs. validation accuracy trajectory, highlighting peak accuracy milestone markers (`*`).
2. `..._2_allocated_epochs.png`: Round-by-round allocated local epochs ($E_t$). Discrete alternating bins for FDA (Dark shade = Matrix A / odd rounds; Light shade = Matrix B / even rounds), alongside reference baseline lines.
3. `..._3_client_variance.png`: Communication rounds vs. inter-client drift variance ($\sigma^2_t$).
4. `..._4_accuracy_vs_cost.png`: Cumulative bi-directional communication cost (MB) vs. validation accuracy, demonstrating bandwidth convergence efficiency.
5. `..._5_fda_threshold.png`: Communication rounds vs. FDA adaptive threshold ($	heta_A$ for Matrix A, $	heta_B$ for Matrix B) demonstrating closed-loop dual-track asymmetry regulation.
6. `..._6_variance_vs_epoch.png`: Cumulative local epochs vs. inter-client drift variance ($\sigma^2$), illustrating variance regulation as computational budget accumulates.
7. `..._all3_telemetry_stacked.png`: Vertical 3-panel stack combining Accuracy, Allocated Epochs, and Client Drift Variance vs. Round.
8. `..._all6_telemetry_stacked.png`: Comprehensive 6-panel 2x3 grid combining all 6 telemetry plots for that configuration.

---

## Configurations Index

| Index | Dataset | Clients ($C$) | Max Communication Rounds ($R$) | Tier |
|:---:|:---:|:---:|:---:|:---:|
| `01` | MNLI | 20 | 50 | Tier 1 (Large Scale) |
| `02` | MNLI | 50 | 30 | Tier 1 (Large Scale) |
| `03` | QNLI | 20 | 50 | Tier 1 (Large Scale) |
| `04` | QNLI | 50 | 30 | Tier 1 (Large Scale) |
| `05` | QQP | 20 | 50 | Tier 2 (Medium Scale) |
| `06` | QQP | 50 | 30 | Tier 2 (Medium Scale) |
| `07` | SST-2 | 20 | 50 | Tier 2 (Medium Scale) |
| `08` | SST-2 | 50 | 30 | Tier 2 (Medium Scale) |
| `09` | MRPC | 20 | 30 | Tier 3 (Small Scale) |
| `10` | MRPC | 50 | 12 | Tier 3 (Small Scale) |
| `11` | RTE | 20 | 30 | Tier 3 (Small Scale) |
| `12` | RTE | 50 | 12 | Tier 3 (Small Scale) |
