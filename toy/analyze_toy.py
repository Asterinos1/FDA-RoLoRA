#!/usr/bin/env python3
"""
Automated Telemetry Analysis & Comparative Benchmarking for Pure Toy Model Experiments.

Parses:
  - 36 Baseline runs from results/toy_experiments/ (LoRA, FFA-LoRA, RoLoRA, E=5)
  - Pure FDA-RoLoRA Dual-Track runs from results/round_metrics/
  - Legacy FDA variants for historical comparison

Generates:
  - Markdown summary table with Mean ± Std of Final & Peak Accuracy, Total Epochs
  - High-resolution trajectory plots (Accuracy vs. Communication Rounds with error bands)
  - Accuracy vs. Cumulative Epochs efficiency curves
  - Detailed CSV metrics export
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Styling
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.autolayout': True
})

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR if (CURRENT_DIR / "thesis_experiments.py").exists() else CURRENT_DIR.parent
RESULTS_DIRS = [
    REPO_ROOT / "results" / "toy_experiments",
    REPO_ROOT / "results" / "round_metrics"
]
OUT_DIR = REPO_ROOT / "analysis" / "toy_pure_analysis"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SEEDS = {42, 43, 44}

def parse_run_file(filepath: Path) -> dict | None:
    filename = filepath.name
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

    # Method normalization
    method = None
    if filename.startswith("TOY-PURE-FDA_PURE"):
        if "EMAX10" in filename:
            ratio_str = "k=0.5" if "CRATIO0.5" in filename else "k=1.0"
            method = f"FDA Pure ({ratio_str}, Emax=10)"
        elif "CRATIO0.5" in filename:
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

def main():
    print("=" * 80)
    print("ANALYZING TOY MODEL EXPERIMENTS: FDA PURE VS BASELINES")
    print("=" * 80)

    all_files = []
    for d in RESULTS_DIRS:
        if d.exists():
            all_files.extend(d.glob("*.csv"))

    parsed_runs = []
    seen_hashes = set()
    for f in all_files:
        meta = parse_run_file(f)
        if meta and meta["run_hash"] not in seen_hashes:
            seen_hashes.add(meta["run_hash"])
            try:
                df = pd.read_csv(meta["filepath"])
                if len(df) >= 100:
                    meta["df"] = df
                    parsed_runs.append(meta)
            except Exception as e:
                print(f"Error reading {f.name}: {e}")

    print(f"Loaded {len(parsed_runs)} verified complete 100-round runs.")

    # Group runs
    groups = {}
    for r in parsed_runs:
        key = (r["scenario"], r["arch"], r["method"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    method_order = [
        "LoRA", "RoLoRA",
        "FDA Pure (k=0.5, Emax=10)", "FDA Pure (k=1.0, Emax=10)",
        "FDA Pure (k=0.5)", "FDA Pure (k=1.0)",
        "FFA-LoRA", "FDA Standard (Legacy)", "FDA EMA (Legacy)", "FDA SMA (Legacy)"
    ]

    scenarios = ["10c1", "5c2"]
    archs = ["NONLIN", "LIN"]

    summary_rows = []
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
                mean_epochs = [r["df"]["round_epochs"].mean() for r in group_runs]
                mean_a_epochs = [
                    r["df"][r["df"]["active_matrix"] == 'A']["round_epochs"].mean()
                    if 'active_matrix' in r["df"].columns and (r["df"]["active_matrix"] == 'A').any()
                    else r["df"]["round_epochs"].mean()
                    for r in group_runs
                ]
                mean_b_epochs = [
                    r["df"][r["df"]["active_matrix"] == 'B']["round_epochs"].mean()
                    if 'active_matrix' in r["df"].columns and (r["df"]["active_matrix"] == 'B').any()
                    else r["df"]["round_epochs"].mean()
                    for r in group_runs
                ]

                summary_rows.append({
                    "Scenario": sc,
                    "Arch": ar,
                    "Method": m,
                    "Seeds": f"{len(seeds_found)}/3 ({','.join(map(str, seeds_found))})",
                    "Final Acc Mean (%)": np.mean(final_accs),
                    "Final Acc Std (%)": np.std(final_accs) if len(final_accs) > 1 else 0.0,
                    "Max Acc Mean (%)": np.mean(max_accs),
                    "Max Acc Std (%)": np.std(max_accs) if len(max_accs) > 1 else 0.0,
                    "Mean Ep/Rd": np.mean(mean_epochs),
                    "Mean Ep (A)": np.mean(mean_a_epochs),
                    "Mean Ep (B)": np.mean(mean_b_epochs),
                    "Total Epochs Mean": np.mean(total_epochs),
                    "Total Epochs Std": np.std(total_epochs) if len(total_epochs) > 1 else 0.0,
                })

    summary_df = pd.DataFrame(summary_rows)
    csv_out = OUT_DIR / "toy_pure_summary_metrics.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"Saved summary CSV: {csv_out}")

    # Generate Markdown Table
    md_lines = [
        "# Toy Model Experiments: FDA Pure vs. Baselines Summary Report\n",
        "Authoritative benchmark comparison evaluating **FDA Pure** (`FDA-RoLoRA-DualTrack-Pure`) "
        "against the static federated baselines (**LoRA**, **FFA-LoRA**, **RoLoRA** with $E=5$) "
        "and legacy FDA variants across 4 extreme non-IID scenarios on MNIST ($d=784, r=16, c=10$).\n",
        "| Scenario | Arch | Method | Seeds | Final Acc (Mean ± Std %) | Max Acc (Mean ± Std %) | Mean Ep/Rd | Mean Ep (A) | Mean Ep (B) | Total Epochs |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]

    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| `{row['Scenario']}` | `{row['Arch']}` | **{row['Method']}** | {row['Seeds']} | "
            f"{row['Final Acc Mean (%)']:.2f} ± {row['Final Acc Std (%)']:.2f}% | "
            f"{row['Max Acc Mean (%)']:.2f} ± {row['Max Acc Std (%)']:.2f}% | "
            f"{row['Mean Ep/Rd']:.2f} | {row['Mean Ep (A)']:.2f} | {row['Mean Ep (B)']:.2f} | "
            f"{row['Total Epochs Mean']:.1f} ± {row['Total Epochs Std']:.1f} |"
        )

    md_out = OUT_DIR / "README.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Generated Markdown report: {md_out}")

    # Plots
    colors = {
        "LoRA": "#1f77b4",
        "FFA-LoRA": "#8c564b",
        "RoLoRA": "#ff7f0e",
        "FDA Pure (k=1.0)": "#d62728",
        "FDA Pure (k=0.5)": "#9467bd",
        "FDA Pure (k=0.5, Emax=10)": "#2ca02c",
        "FDA Pure (k=1.0, Emax=10)": "#17becf",
        "FDA Standard (Legacy)": "#bcbd22",
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
                fig_path = FIG_DIR / f"trajectory_{sc}_{ar}.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved figure: {fig_path.name}")

    print("\nAnalysis suite executed successfully!")

if __name__ == '__main__':
    main()
