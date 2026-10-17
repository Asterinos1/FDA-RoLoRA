#!/usr/bin/env python3
"""
Comprehensive Analysis & Comparison Suite for Toy Model Experiments:
Compares FDA Pure against authoritative baselines (LoRA, FFA-LoRA, RoLoRA)
and legacy FDA variants (Standard, EMA-Standard, SMA).

Outputs:
  - Markdown summary table (final accuracy, max accuracy, rounds to target, total epochs).
  - Round vs Accuracy trajectory figures (with std error bands).
  - Cumulative Epochs vs Accuracy efficiency figures.
  - Round epoch distribution plots.
"""

import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.autolayout': True
})

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIRS = [
    os.path.join(REPO_ROOT, "results", "toy_experiments"),
    os.path.join(REPO_ROOT, "results", "round_metrics")
]
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "toy_pure_analysis")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

ALLOWED_SEEDS = {42, 43, 44}

def parse_run_file(filepath):
    filename = os.path.basename(filepath)
    if not filename.endswith('.csv'):
        return None

    # Architecture
    arch = "LIN" if "_LIN_" in filename else "NONLIN"

    # Scenario
    if "5-Clients-2-Classes" in filename:
        scenario = "5c2"
    elif "10-Clients-1-Class" in filename:
        scenario = "10c1"
    else:
        return None

    # Seed
    seed = None
    for part in filename.split('_'):
        if part.startswith('S') and part[1:].isdigit():
            seed = int(part[1:])
            break
    if seed not in ALLOWED_SEEDS:
        return None

    # Method identification
    method = None
    if filename.startswith("TOY-PURE-FDA_PURE"):
        if "CRATIO0.5" in filename:
            method = "FDA Pure (k=0.5)"
        else:
            method = "FDA Pure (k=1.0)"
    elif filename.startswith("TOY-BASELINES"):
        if "FedAvg-LoRA" in filename:
            method = "LoRA"
        elif "FFA-LoRA" in filename:
            method = "FFA-LoRA"
        elif "FedAvg-RoLoRA" in filename:
            method = "RoLoRA"
    elif filename.startswith("TOY-NEW2FDA-FDA_STANDARD"):
        method = "FDA Standard (Legacy)"
    elif filename.startswith("TOY-NEW2FDA-FDA_EMA_STANDARD"):
        method = "FDA EMA (Legacy)"
    elif filename.startswith("TOY-NEW3FDA-FDA_SMA"):
        method = "FDA SMA (Legacy)"

    if method is None:
        return None

    run_hash = filename.split('_')[-1].replace('.csv', '')
    return {
        "filepath": filepath,
        "filename": filename,
        "arch": arch,
        "scenario": scenario,
        "seed": seed,
        "method": method,
        "run_hash": run_hash
    }

def load_all_runs():
    all_files = []
    for d in RESULTS_DIRS:
        if os.path.exists(d):
            all_files.extend(glob.glob(os.path.join(d, "*.csv")))

    parsed = []
    seen_hashes = set()
    for f in all_files:
        meta = parse_run_file(f)
        if meta and meta["run_hash"] not in seen_hashes:
            seen_hashes.add(meta["run_hash"])
            try:
                df = pd.read_csv(meta["filepath"])
                if len(df) >= 100:  # Only complete runs
                    meta["df"] = df
                    parsed.append(meta)
            except Exception as e:
                print(f"Error reading {f}: {e}")

    return parsed

def generate_analysis():
    runs = load_all_runs()
    print(f"Loaded {len(runs)} complete 100-round runs across search directories.")

    # Group runs by (scenario, arch, method)
    groups = {}
    for r in runs:
        key = (r["scenario"], r["arch"], r["method"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    # Summary table collection
    summary_rows = []
    method_order = [
        "LoRA", "FFA-LoRA", "RoLoRA",
        "FDA Pure (k=1.0)", "FDA Pure (k=0.5)",
        "FDA Standard (Legacy)", "FDA EMA (Legacy)", "FDA SMA (Legacy)"
    ]

    scenarios = ["10c1", "5c2"]
    archs = ["NONLIN", "LIN"]

    for sc in scenarios:
        for ar in archs:
            for m in method_order:
                key = (sc, ar, m)
                if key not in groups:
                    continue
                group_runs = groups[key]
                seeds_found = sorted([r["seed"] for r in group_runs])
                
                final_accs = [r["df"]["accuracy"].iloc[-1] * 100 for r in group_runs]
                max_accs = [r["df"]["accuracy"].max() * 100 for r in group_runs]
                total_epochs = [r["df"]["round_epochs"].sum() for r in group_runs]

                summary_rows.append({
                    "Scenario": sc,
                    "Arch": ar,
                    "Method": m,
                    "Seeds": f"{len(seeds_found)}/3 ({','.join(map(str, seeds_found))})",
                    "Final Acc Mean (%)": np.mean(final_accs),
                    "Final Acc Std (%)": np.std(final_accs) if len(final_accs) > 1 else 0.0,
                    "Max Acc Mean (%)": np.mean(max_accs),
                    "Max Acc Std (%)": np.std(max_accs) if len(max_accs) > 1 else 0.0,
                    "Total Epochs Mean": np.mean(total_epochs),
                    "Total Epochs Std": np.std(total_epochs) if len(total_epochs) > 1 else 0.0,
                })

    summary_df = pd.DataFrame(summary_rows)
    csv_out = os.path.join(OUT_DIR, "toy_pure_summary_metrics.csv")
    summary_df.to_csv(csv_out, index=False)
    print(f"Summary metrics saved to: {csv_out}")

    # Generate Markdown Table
    md_lines = [
        "# Toy Model Experiments: FDA Pure vs. Baselines Summary Report\n",
        "Authoritative benchmark comparison evaluating **FDA Pure** (`FDA-RoLoRA-DualTrack-Pure`) "
        "against the static federated baselines (**LoRA**, **FFA-LoRA**, **RoLoRA** with $E=5$) "
        "and legacy FDA variants across 4 non-IID scenarios on MNIST ($d=784, r=16, c=10$).\n",
        "| Scenario | Arch | Method | Seeds | Final Acc (Mean ± Std %) | Max Acc (Mean ± Std %) | Total Epochs |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|"
    ]

    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| `{row['Scenario']}` | `{row['Arch']}` | **{row['Method']}** | {row['Seeds']} | "
            f"{row['Final Acc Mean (%)']:.2f} ± {row['Final Acc Std (%)']:.2f}% | "
            f"{row['Max Acc Mean (%)']:.2f} ± {row['Max Acc Std (%)']:.2f}% | "
            f"{row['Total Epochs Mean']:.1f} ± {row['Total Epochs Std']:.1f} |"
        )

    md_out = os.path.join(OUT_DIR, "README.md")
    with open(md_out, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown report generated: {md_out}")

    # Generate Comparative Trajectory Plots for each (Scenario, Arch)
    colors = {
        "LoRA": "#1f77b4",
        "FFA-LoRA": "#2ca02c",
        "RoLoRA": "#ff7f0e",
        "FDA Pure (k=1.0)": "#d62728",
        "FDA Pure (k=0.5)": "#9467bd",
        "FDA Standard (Legacy)": "#8c564b",
        "FDA EMA (Legacy)": "#e377c2",
        "FDA SMA (Legacy)": "#7f7f7f"
    }

    for sc in scenarios:
        for ar in archs:
            plt.figure(figsize=(9, 6))
            plotted = False
            for m in method_order:
                key = (sc, ar, m)
                if key not in groups:
                    continue
                group_runs = groups[key]
                all_accs = np.array([r["df"]["accuracy"].values[:100] * 100 for r in group_runs])
                mean_acc = np.mean(all_accs, axis=0)
                std_acc = np.std(all_accs, axis=0) if len(group_runs) > 1 else np.zeros_like(mean_acc)
                rounds = np.arange(1, len(mean_acc) + 1)

                c = colors.get(m, "#333333")
                lw = 2.5 if "Pure" in m else 1.8
                ls = '-' if "Pure" in m or m in ["LoRA", "RoLoRA", "FFA-LoRA"] else '--'
                plt.plot(rounds, mean_acc, label=f"{m} (N={len(group_runs)})", color=c, linewidth=lw, linestyle=ls)
                plt.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, color=c, alpha=0.15)
                plotted = True

            if plotted:
                plt.title(f"Toy Model Accuracy vs Communication Rounds ({sc.upper()} - {ar})", fontsize=13, fontweight='bold')
                plt.xlabel("Communication Round", fontsize=11)
                plt.ylabel("Test Accuracy (%)", fontsize=11)
                plt.ylim(0, 100)
                plt.legend(loc="lower right", frameon=True)
                fig_path = os.path.join(OUT_DIR, "figures", f"trajectory_{sc}_{ar}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved figure: {fig_path}")

    print("\nAnalysis completed successfully!")

if __name__ == '__main__':
    generate_analysis()
