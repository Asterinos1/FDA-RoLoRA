# Figures & Telemetry Visualizations Catalog

This directory serves as the centralized repository for all publication figures, comparative telemetry visualizations, and per-run performance plots for the **FDA-RoLoRA** thesis benchmark evaluation.

---

## Directory Organization

```text
figures/
├── FDA_Pure_vs_RoLoRA/              # Comparative telemetry for Pure FDA vs RoLoRA baselines (20 plots)
│                                    # Referenced by FDA_Pure_vs_RoLoRA.md
├── FDA_Pure_vs_Baselines/           # Comparative telemetry for Pure FDA vs all PEFT baselines (20 plots)
│                                    # Referenced by FDA_Pure_vs_Baselines.md
├── individual_plots/                # Categorized high-resolution individual metric trajectories
│   ├── fda_vs_rolora/               # Head-to-head plots for FDA-RoLoRA vs RoLoRA (E5, E10)
│   ├── master_all_methods/          # Complete multi-method comparisons (FDA, RoLoRA, LoRA)
│   └── README.md                    # Detailed naming convention and configuration index
├── all_individual_figures/          # Flat archive of standalone single-metric plots
└── [Root Plots]                     # Cross-benchmark summaries, convergence panels & paper figures
```

---

## Subdirectory Overview

### 1. `FDA_Pure_vs_RoLoRA/`
Contains 20 high-resolution stacked 6-panel telemetry grids (`*_all6_telemetry_stacked.png`) and inter-client drift log-variance plots (`*_8_log_client_variance.png`) across all 10 evaluated task configurations comparing **Pure FDA-RoLoRA** against **RoLoRA** baselines. These assets directly embed into [`FDA_Pure_vs_RoLoRA.md`](../FDA_Pure_vs_RoLoRA.md).

### 2. `FDA_Pure_vs_Baselines/`
Contains 20 high-resolution stacked telemetry grids and log-variance plots comparing **Pure FDA-RoLoRA** across all evaluated parameter-efficient fine-tuning methods (**RoLoRA**, vanilla **LoRA**, and **FFA-LoRA**). These assets directly embed into [`FDA_Pure_vs_Baselines.md`](../FDA_Pure_vs_Baselines.md).

### 3. `individual_plots/`
Organized into two subdirectories for fine-grained analysis:
* **`fda_vs_rolora/`**: Accuracy, allocated epochs, drift variance, accuracy-vs-cost, and dual-track threshold curves isolated between FDA-RoLoRA and RoLoRA.
* **`master_all_methods/`**: Identical metric breakdowns extended to all baselines including LoRA and FFA-LoRA.
* See [`figures/individual_plots/README.md`](individual_plots/README.md) for the metric-by-metric suffix schema.

### 4. `all_individual_figures/`
A flat export archive of all standalone component plots for automated ingestion, slide presentations, and thesis inclusion.

### 5. Root Publication & Summary Panels
High-level overview and composite figures embedded in [`README.md`](../README.md) and [`FDA_vs_Baselines_Evaluation_Assessment.md`](../FDA_vs_Baselines_Evaluation_Assessment.md):
* **`convergence_c50_scaled.png`**: Cross-task convergence trajectories under $C=50$ client scaling.
* **`fda_dynamic_controller_telemetry.png`**: Closed-loop proportional threshold controller action ($\Theta_A, \Theta_B$) and dynamic epoch allocation dynamics.
* **`non_iid_benchmark_telemetry.png`** & **`non_iid_vs_iid_head_to_head.png`**: Dirichlet non-IID label skew ($\alpha=0.5$) robustness comparisons.
* **`pure_fda_vs_calib_and_baselines.png`**: Head-to-head comparison between uncalibrated Pure FDA, calibrated FDA with Matrix B discounting, and baselines.
