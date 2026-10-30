import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from update_benchmark_suite import pure_configs, get_methods, prep, average_runs

THESIS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "thesis-stuff"))
THESIS_FIG_DIR = os.path.join(THESIS_DIR, "thesis", "figures")
TELEMETRY_DIR = os.path.join(THESIS_FIG_DIR, "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)

# Correct comm rate for SST-2: 3 layers = 144 KB per client per round
# All others: 5 layers = 240 KB per client per round
def get_comm_rate(dataset, cohort_C):
    if dataset.lower() == "sst2":
        return 144.0 * cohort_C / 1024.0 # MB per round
    else:
        return 240.0 * cohort_C / 1024.0 # MB per round

# Styling defaults matching thesis
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.0,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0
})

COLOR_MAP = {
    "FDA-RoLoRA (k=1.0)": {"color": "#1f77b4", "ls": "-", "marker": "o", "label": "FDA-RoLoRA ($k=1.0$)"},
    "FDA-RoLoRA (k=0.5)": {"color": "#7b1fa2", "ls": "-", "marker": "s", "label": "FDA-RoLoRA ($k=0.5$)"},
    "RoLoRA (E=20)":      {"color": "#c2410c", "ls": "--", "marker": "^", "label": "RoLoRA ($E=20$)"},
    "RoLoRA (E=10)":      {"color": "#ea580c", "ls": "--", "marker": "v", "label": "RoLoRA ($E=10$)"},
}

def clean_method_name(raw_name):
    if "fda pure" in raw_name.lower() or "fda-pure" in raw_name.lower():
        if "k=1.0" in raw_name or "cratio1" in raw_name.lower():
            return "FDA-RoLoRA (k=1.0)"
        elif "k=0.5" in raw_name or "cratio0.5" in raw_name.lower():
            return "FDA-RoLoRA (k=0.5)"
        else:
            return "FDA-RoLoRA"
    elif "rolora" in raw_name.lower():
        if "e20" in raw_name.lower():
            return "RoLoRA (E=20)"
        elif "e10" in raw_name.lower():
            return "RoLoRA (E=10)"
        elif "e15" in raw_name.lower():
            return "RoLoRA (E=15)"
        elif "e5" in raw_name.lower():
            return "RoLoRA (E=5)"
        return "RoLoRA"
    return raw_name

# Select target configs to render: 01, 03, 05, 07, 09, 10, 11, 12 (IID) and 17, 18 (Non-IID)
TARGET_CONFIG_PREFIXES = [
    "01_MNLI_C20_R50_LR0.1",
    "03_MNLI_C50_R30_LR0.1",
    "05_QNLI_C20_R50_LR0.1",
    "07_QNLI_C50_R30_LR0.1",
    "09_QQP_C20_R50_LR0.1",
    "10_QQP_C50_R30_LR0.1",
    "11_SST2_C20_R50_LR0.1",
    "12_SST2_C50_R30_LR0.1",
    "17_MNLI_C20_R50_LR0.1_A0.5",
    "18_QNLI_C20_R50_LR0.1_A0.5",
]

print("=== Re-rendering Pure Telemetry Plots for Thesis ===")

all_loaded_runs = {}

for cfg in pure_configs:
    cfg_prefix, title, ds, c, r_cfg, lr, _, _ = cfg
    if cfg_prefix not in TARGET_CONFIG_PREFIXES:
        continue
    
    print(f"Processing configuration: {cfg_prefix} ({title})...")
    methods = get_methods(ds, c, r_cfg, lr, include_lora=False, cfg_prefix=cfg_prefix, pure_only=True)
    rate_mb = get_comm_rate(ds, c)
    
    loaded_methods = {}
    for mname, pat, is_lora in methods:
        norm_name = clean_method_name(mname)
        if norm_name not in COLOR_MAP:
            continue
        matches = pat if isinstance(pat, list) else sorted(glob.glob(pat))
        valid_dfs = []
        for f in matches:
            if "5095b3daf3" in f:
                continue
            df = prep(pd.read_csv(f))
            if len(df) > 0:
                valid_dfs.append(df)
        if valid_dfs:
            avg_df = average_runs(valid_dfs)
            avg_df["exact_mb"] = avg_df["round"] * rate_mb
            # Cumulative local epochs
            ep_col = avg_df["round_epochs"] if "round_epochs" in avg_df.columns else avg_df.get("epochs", pd.Series(0, index=avg_df.index))
            avg_df["cum_epochs"] = ep_col.cumsum()
            loaded_methods[norm_name] = avg_df

    all_loaded_runs[cfg_prefix] = (ds, c, r_cfg, loaded_methods)

    # -------------------------------------------------------------
    # Render Individual Subpanels (1 to 6)
    # -------------------------------------------------------------
    # 1. Validation Accuracy vs. Round
    plt.figure(figsize=(7.5, 4.8), dpi=180)
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            plt.plot(df["round"], df["accuracy"], color=meta["color"], linestyle=meta["ls"],
                     marker=meta["marker"], markersize=4.5, markevery=max(1, len(df)//15), label=meta["label"])
    plt.xlabel("Communication Round")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{title}: Validation Accuracy vs. Communication Rounds")
    plt.legend(loc="lower right", framealpha=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(TELEMETRY_DIR, f"{cfg_prefix}_1_accuracy.png"), dpi=180)
    plt.close()

    # 2. Allocated Epochs vs. Round
    plt.figure(figsize=(7.5, 4.8), dpi=180)
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            ep_col = df["round_epochs"] if "round_epochs" in df.columns else df.get("epochs", None)
            if ep_col is not None:
                plt.plot(df["round"], ep_col, color=meta["color"], linestyle=meta["ls"],
                         marker=meta["marker"], markersize=4.5, markevery=max(1, len(df)//15), label=meta["label"])
    plt.xlabel("Communication Round")
    plt.ylabel("Allocated Local Epochs")
    plt.title(f"{title}: Realized Local Epochs ($e^*$) per Round")
    plt.legend(loc="upper right", framealpha=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(TELEMETRY_DIR, f"{cfg_prefix}_2_allocated_epochs.png"), dpi=180)
    plt.close()

    # 3. Client Drift Variance vs. Round
    plt.figure(figsize=(7.5, 4.8), dpi=180)
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            var_col = df["variance"] if "variance" in df.columns else df.get("client_variance", None)
            if var_col is not None and not var_col.isna().all():
                plt.plot(df["round"], var_col, color=meta["color"], linestyle=meta["ls"],
                         marker=meta["marker"], markersize=4.5, markevery=max(1, len(df)//15), label=meta["label"])
    plt.xlabel("Communication Round")
    plt.ylabel("Inter-Client Drift Variance")
    plt.title(title + r": Inter-Client Subspace Drift Variance $\mathrm{Var}(\Delta)$")
    plt.legend(loc="upper right", framealpha=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(TELEMETRY_DIR, f"{cfg_prefix}_3_client_variance.png"), dpi=180)
    plt.close()

    # 4. Accuracy vs. Communication Cost (MB)
    plt.figure(figsize=(7.5, 4.8), dpi=180)
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            plt.plot(df["exact_mb"], df["accuracy"], color=meta["color"], linestyle=meta["ls"],
                     marker=meta["marker"], markersize=4.5, markevery=max(1, len(df)//15), label=meta["label"])
    plt.xlabel("Communication Payload (MB)")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{title}: Validation Accuracy vs. Communication Payload")
    plt.legend(loc="lower right", framealpha=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(TELEMETRY_DIR, f"{cfg_prefix}_4_accuracy_vs_cost.png"), dpi=180)
    plt.close()

    # 5. FDA Threshold per Matrix
    plt.figure(figsize=(7.5, 4.8), dpi=180)
    has_theta = False
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            if "theta_phase" in df.columns:
                plt.plot(df["round"], df["theta_phase"], color=meta["color"], linestyle=meta["ls"],
                         marker=meta["marker"], markersize=4.5, markevery=max(1, len(df)//15), label=f"{meta['label']} $\\theta$")
                has_theta = True
    if not has_theta:
        plt.text(0.5, 0.5, "Dynamic FDA Threshold Active", ha="center", va="center", transform=plt.gca().transAxes)
    plt.xlabel("Communication Round")
    plt.ylabel("Adaptive Threshold $\\theta$")
    plt.title(f"{title}: Adaptive Variance Threshold Trajectory")
    plt.legend(loc="lower right", framealpha=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(TELEMETRY_DIR, f"{cfg_prefix}_5_fda_threshold.png"), dpi=180)
    plt.close()

    # 6. Variance vs. Cumulative Local Epochs
    plt.figure(figsize=(7.5, 4.8), dpi=180)
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            var_col = df["variance"] if "variance" in df.columns else df.get("client_variance", None)
            if var_col is not None and not var_col.isna().all():
                plt.plot(df["cum_epochs"], var_col, color=meta["color"], linestyle=meta["ls"],
                         marker=meta["marker"], markersize=4.5, markevery=max(1, len(df)//15), label=meta["label"])
    plt.xlabel("Cumulative Local Epochs")
    plt.ylabel("Inter-Client Drift Variance")
    plt.title(f"{title}: Subspace Variance vs. Cumulative Local Epochs")
    plt.legend(loc="upper right", framealpha=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(TELEMETRY_DIR, f"{cfg_prefix}_6_variance_vs_epoch.png"), dpi=180)
    plt.close()

    # -------------------------------------------------------------
    # Comprehensive 6-Panel (3x2) Stacked Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15.0, 13.0), dpi=180)
    
    # [0, 0] Accuracy vs Round
    ax = axes[0, 0]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            ax.plot(df["round"], df["accuracy"], color=meta["color"], linestyle=meta["ls"],
                    marker=meta["marker"], markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy vs. Communication Rounds")
    ax.legend(loc="lower right", framealpha=0.92)

    # [0, 1] Accuracy vs Cost
    ax = axes[0, 1]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            ax.plot(df["exact_mb"], df["accuracy"], color=meta["color"], linestyle=meta["ls"],
                    marker=meta["marker"], markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
    ax.set_xlabel("Communication Payload (MB)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy vs. Communication Payload (MB)")
    ax.legend(loc="lower right", framealpha=0.92)

    # [1, 0] Epochs vs Round
    ax = axes[1, 0]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            ep_col = df["round_epochs"] if "round_epochs" in df.columns else df.get("epochs", None)
            if ep_col is not None:
                ax.plot(df["round"], ep_col, color=meta["color"], linestyle=meta["ls"],
                        marker=meta["marker"], markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Allocated Local Epochs")
    ax.set_title("Realized Local Epochs ($e^*$) per Round")
    ax.legend(loc="upper right", framealpha=0.92)

    # [1, 1] FDA Threshold
    ax = axes[1, 1]
    has_theta = False
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            if "theta_phase" in df.columns:
                ax.plot(df["round"], df["theta_phase"], color=meta["color"], linestyle=meta["ls"],
                        marker=meta["marker"], markersize=3.5, markevery=max(1, len(df)//15), label=f"{meta['label']} $\\theta$")
                has_theta = True
    if not has_theta:
        ax.text(0.5, 0.5, "Dynamic Variance Threshold Active", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Adaptive Threshold $\\theta$")
    ax.set_title("Adaptive Variance Threshold Trajectory")
    ax.legend(loc="lower right", framealpha=0.92)

    # [2, 0] Drift Variance vs Round
    ax = axes[2, 0]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            var_col = df["variance"] if "variance" in df.columns else df.get("client_variance", None)
            if var_col is not None and not var_col.isna().all():
                ax.plot(df["round"], var_col, color=meta["color"], linestyle=meta["ls"],
                        marker=meta["marker"], markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Inter-Client Drift Variance")
    ax.set_title(r"Inter-Client Subspace Drift Variance $\mathrm{Var}(\Delta)$")
    ax.legend(loc="upper right", framealpha=0.92)

    # [2, 1] Variance vs Epochs
    ax = axes[2, 1]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in loaded_methods:
            df = loaded_methods[mname]
            meta = COLOR_MAP[mname]
            var_col = df["variance"] if "variance" in df.columns else df.get("client_variance", None)
            if var_col is not None and not var_col.isna().all():
                ax.plot(df["cum_epochs"], var_col, color=meta["color"], linestyle=meta["ls"],
                        marker=meta["marker"], markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
    ax.set_xlabel("Cumulative Local Epochs")
    ax.set_ylabel("Inter-Client Drift Variance")
    ax.set_title("Subspace Variance vs. Cumulative Local Epochs")
    ax.legend(loc="upper right", framealpha=0.92)

    plt.suptitle(f"{title}: Closed-Loop Adaptive FL Telemetry", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(TELEMETRY_DIR, f"{cfg_prefix}_all6_telemetry_stacked.png"), dpi=180)
    plt.savefig(os.path.join(THESIS_FIG_DIR, f"{cfg_prefix}_all6_telemetry_stacked.png"), dpi=180)
    plt.close()

# -------------------------------------------------------------
# Re-render Figures 5.11 & 5.12 (Cross-Task Diagnostics for MNLI & QNLI)
# -------------------------------------------------------------
print("Re-rendering Figure 5.11 (non_iid_benchmark_telemetry.png)...")
fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), dpi=180)

# (0,0): MNLI Accuracy C=20 Non-IID
if "17_MNLI_C20_R50_LR0.1_A0.5" in all_loaded_runs:
    _, _, _, mdict = all_loaded_runs["17_MNLI_C20_R50_LR0.1_A0.5"]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in mdict:
            df = mdict[mname]
            meta = COLOR_MAP[mname]
            axes[0, 0].plot(df["round"], df["accuracy"], color=meta["color"], ls=meta["ls"], marker=meta["marker"],
                            markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
axes[0, 0].set_title("MNLI ($C=20, \\alpha=0.5$): Validation Accuracy")
axes[0, 0].set_xlabel("Communication Round")
axes[0, 0].set_ylabel("Validation Accuracy")
axes[0, 0].legend(loc="lower right")

# (0,1): QNLI Accuracy C=20 Non-IID
if "18_QNLI_C20_R50_LR0.1_A0.5" in all_loaded_runs:
    _, _, _, mdict = all_loaded_runs["18_QNLI_C20_R50_LR0.1_A0.5"]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in mdict:
            df = mdict[mname]
            meta = COLOR_MAP[mname]
            axes[0, 1].plot(df["round"], df["accuracy"], color=meta["color"], ls=meta["ls"], marker=meta["marker"],
                            markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
axes[0, 1].set_title("QNLI ($C=20, \\alpha=0.5$): Validation Accuracy")
axes[0, 1].set_xlabel("Communication Round")
axes[0, 1].set_ylabel("Validation Accuracy")
axes[0, 1].legend(loc="lower right")

# (1,0): MNLI Drift Variance C=20 Non-IID
if "17_MNLI_C20_R50_LR0.1_A0.5" in all_loaded_runs:
    _, _, _, mdict = all_loaded_runs["17_MNLI_C20_R50_LR0.1_A0.5"]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in mdict:
            df = mdict[mname]
            meta = COLOR_MAP[mname]
            var_col = df["variance"] if "variance" in df.columns else df.get("client_variance", None)
            if var_col is not None:
                axes[1, 0].plot(df["round"], var_col, color=meta["color"], ls=meta["ls"], marker=meta["marker"],
                                markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
axes[1, 0].set_title(r"MNLI ($C=20, \alpha=0.5$): Inter-Client Drift Variance")
axes[1, 0].set_xlabel("Communication Round")
axes[1, 0].set_ylabel(r"Subspace Variance $\mathrm{Var}(\Delta)$")
axes[1, 0].legend(loc="upper right")

# (1,1): QNLI Drift Variance C=20 Non-IID
if "18_QNLI_C20_R50_LR0.1_A0.5" in all_loaded_runs:
    _, _, _, mdict = all_loaded_runs["18_QNLI_C20_R50_LR0.1_A0.5"]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)", "RoLoRA (E=20)", "RoLoRA (E=10)"]:
        if mname in mdict:
            df = mdict[mname]
            meta = COLOR_MAP[mname]
            var_col = df["variance"] if "variance" in df.columns else df.get("client_variance", None)
            if var_col is not None:
                axes[1, 1].plot(df["round"], var_col, color=meta["color"], ls=meta["ls"], marker=meta["marker"],
                                markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
axes[1, 1].set_title(r"QNLI ($C=20, \alpha=0.5$): Inter-Client Drift Variance")
axes[1, 1].set_xlabel("Communication Round")
axes[1, 1].set_ylabel(r"Subspace Variance $\mathrm{Var}(\Delta)$")
axes[1, 1].legend(loc="upper right")

plt.suptitle("Non-IID Empirical Convergence & Client Drift Dynamics (MNLI & QNLI, $\\alpha=0.5$)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(THESIS_FIG_DIR, "non_iid_benchmark_telemetry.png"), dpi=180)
plt.close()

# -------------------------------------------------------------
# Re-render Figure 5.12 (non_iid_epoch_dynamics.png)
# -------------------------------------------------------------
print("Re-rendering Figure 5.12 (non_iid_epoch_dynamics.png)...")
fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), dpi=180)

if "17_MNLI_C20_R50_LR0.1_A0.5" in all_loaded_runs:
    _, _, _, mdict = all_loaded_runs["17_MNLI_C20_R50_LR0.1_A0.5"]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)"]:
        if mname in mdict:
            df = mdict[mname]
            meta = COLOR_MAP[mname]
            ep_col = df["round_epochs"] if "round_epochs" in df.columns else df.get("epochs", None)
            if ep_col is not None:
                axes[0].plot(df["round"], ep_col, color=meta["color"], ls=meta["ls"], marker=meta["marker"],
                             markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
    axes[0].set_title("MNLI ($C=20, \\alpha=0.5$): Adaptive Local Epochs ($e^*$)")
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Allocated Local Epochs")
    axes[0].axhline(y=20, color="gray", linestyle=":", label="Upper Bound $e_{\\max}=20$")
    axes[0].legend(loc="upper right")

if "18_QNLI_C20_R50_LR0.1_A0.5" in all_loaded_runs:
    _, _, _, mdict = all_loaded_runs["18_QNLI_C20_R50_LR0.1_A0.5"]
    for mname in ["FDA-RoLoRA (k=1.0)", "FDA-RoLoRA (k=0.5)"]:
        if mname in mdict:
            df = mdict[mname]
            meta = COLOR_MAP[mname]
            ep_col = df["round_epochs"] if "round_epochs" in df.columns else df.get("epochs", None)
            if ep_col is not None:
                axes[1].plot(df["round"], ep_col, color=meta["color"], ls=meta["ls"], marker=meta["marker"],
                             markersize=3.5, markevery=max(1, len(df)//15), label=meta["label"])
    axes[1].set_title("QNLI ($C=20, \\alpha=0.5$): Adaptive Local Epochs ($e^*$)")
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Allocated Local Epochs")
    axes[1].axhline(y=20, color="gray", linestyle=":", label="Upper Bound $e_{\\max}=20$")
    axes[1].legend(loc="upper right")

plt.suptitle(r"Closed-Loop Local Epoch Adaptation Under Non-IID Heterogeneity ($\alpha=0.5$)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(THESIS_FIG_DIR, "non_iid_epoch_dynamics.png"), dpi=180)
plt.close()

print("=== Telemetry plots re-rendered successfully! ===")
