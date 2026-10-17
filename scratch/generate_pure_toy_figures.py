#!/usr/bin/env python3
"""
Comprehensive Graph Generation and Statistical Analysis for Pure FDA-RoLoRA vs. Baselines on Toy MNIST.
Generates:
1. Master 4-Panel Accuracy vs Communication Rounds
2. Master 4-Panel Accuracy vs Cumulative Local Epochs (Compute Efficiency)
3. Master 4-Panel Test Loss vs Communication Rounds (Log Scale)
4. Master 4-Panel Inter-Client Drift Variance vs Communication Rounds (Log Scale)
5. FDA Pure Telemetry: Realized Local Epochs (e*) per Round
6. Individual High-Resolution Scenario Figures:
   - Trajectory (Accuracy vs Rounds)
   - Compute Efficiency (Accuracy vs Cumulative Epochs)
   - Inter-Client Drift Variance (Log Scale)
   - Side-by-Side Dual-Panel Figure (Accuracy vs Rounds + Accuracy vs Epochs)
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Styling configuration
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0
})

REPO_ROOT = Path(r"C:\Users\PC\Documents\GitHub\FDA-LoRA")
PURE_DIR = REPO_ROOT / "results" / "round_metrics"
BASELINE_DIR = REPO_ROOT / "results" / "toy_experiments"
FIG_DIR = REPO_ROOT / "results" / "figures" / "toy_pure"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SEEDS = {42, 43, 44}

METHODS_MAP = {
    "FedAvg-LoRA": {"label": "FedAvg-LoRA (E=5)", "color": "#1f77b4", "ls": "--", "lw": 1.8},
    "RoLoRA": {"label": "RoLoRA (E=5)", "color": "#ff7f0e", "ls": "-.", "lw": 2.0},
    "FDA Pure (k=1.0)": {"label": "FDA Pure (k=1.0)", "color": "#d62728", "ls": "-", "lw": 2.6},
    "FDA Pure (k=0.5)": {"label": "FDA Pure (k=0.5)", "color": "#9467bd", "ls": "-", "lw": 2.2},
}

METHOD_ORDER = ["FedAvg-LoRA", "RoLoRA", "FDA Pure (k=1.0)", "FDA Pure (k=0.5)"]

SCENARIOS = [
    ("10c1", "NONLIN", "10 Clients, 1 Class/Client (Nonlinear / ReLU)"),
    ("10c1", "LIN", "10 Clients, 1 Class/Client (Linear)"),
    ("5c2", "NONLIN", "5 Clients, 2 Classes/Client (Nonlinear / ReLU)"),
    ("5c2", "LIN", "5 Clients, 2 Classes/Client (Linear)"),
]

def load_data():
    runs = []
    
    # Load pure runs
    for f in PURE_DIR.glob("TOY-PURE-FDA_PURE*.csv"):
        fname = f.name
        arch = "LIN" if "_LIN_" in fname else "NONLIN"
        scen = "5c2" if "5-Clients-2-Classes" in fname else "10c1"
        seed = [int(p[1:]) for p in fname.split('_') if p.startswith('S') and p[1:].isdigit()][0]
        if seed not in ALLOWED_SEEDS:
            continue
        ratio = 0.5 if "CRATIO0.5" in fname else 1.0
        method = f"FDA Pure (k={ratio})"
        df = pd.read_csv(f)
        runs.append({
            "method": method, "arch": arch, "scen": scen, "seed": seed,
            "df": df, "file": f
        })

    # Load baseline runs
    for f in BASELINE_DIR.glob("TOY-BASELINES*.csv"):
        fname = f.name
        arch = "LIN" if "_LIN_" in fname else "NONLIN"
        scen = "5c2" if "5-Clients-2-Classes" in fname else "10c1"
        seed = [int(p[1:]) for p in fname.split('_') if p.startswith('S') and p[1:].isdigit()][0]
        if seed not in ALLOWED_SEEDS:
            continue
        if "FedAvg-LoRA" in fname:
            method = "FedAvg-LoRA"
        elif "FedAvg-RoLoRA" in fname:
            method = "RoLoRA"
        else:
            continue
        df = pd.read_csv(f)
        if "round_epochs" not in df.columns:
            df["round_epochs"] = 5
        runs.append({
            "method": method, "arch": arch, "scen": scen, "seed": seed,
            "df": df, "file": f
        })

    return runs

def compute_statistics(runs):
    grouped = {}
    for r in runs:
        key = (r["scen"], r["arch"], r["method"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    rows = []
    for sc, ar, title in SCENARIOS:
        for m in METHOD_ORDER:
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            n_seeds = len(group)
            
            acc_matrix = np.array([r["df"]["accuracy"].values[:100] * 100 for r in group])
            r25 = acc_matrix[:, 24] if acc_matrix.shape[1] >= 25 else np.zeros(n_seeds)
            r50 = acc_matrix[:, 49] if acc_matrix.shape[1] >= 50 else np.zeros(n_seeds)
            r75 = acc_matrix[:, 74] if acc_matrix.shape[1] >= 75 else np.zeros(n_seeds)
            r100 = acc_matrix[:, 99] if acc_matrix.shape[1] >= 100 else acc_matrix[:, -1]
            peak_acc = np.max(acc_matrix, axis=1)
            peak_round = np.argmax(acc_matrix, axis=1) + 1
            total_epochs = [r["df"]["round_epochs"].sum() for r in group]

            milestones = {}
            for target in [50, 60, 70, 75, 80]:
                target_rounds = []
                for s_idx in range(n_seeds):
                    hits = np.where(acc_matrix[s_idx] >= target)[0]
                    if len(hits) > 0:
                        target_rounds.append(hits[0] + 1)
                milestones[target] = target_rounds

            rows.append({
                "scen": sc,
                "arch": ar,
                "method": m,
                "n_seeds": n_seeds,
                "r25_mean": float(np.mean(r25)), "r25_std": float(np.std(r25)),
                "r50_mean": float(np.mean(r50)), "r50_std": float(np.std(r50)),
                "r75_mean": float(np.mean(r75)), "r75_std": float(np.std(r75)),
                "r100_mean": float(np.mean(r100)), "r100_std": float(np.std(r100)),
                "peak_mean": float(np.mean(peak_acc)), "peak_std": float(np.std(peak_acc)),
                "peak_round_mean": float(np.mean(peak_round)), "peak_round_std": float(np.std(peak_round)),
                "epochs_mean": float(np.mean(total_epochs)), "epochs_std": float(np.std(total_epochs)),
                "milestones": milestones
            })

    return grouped, rows

def generate_plots(grouped):
    # --------------------------------------------------------------------------
    # 1. 4-Panel Master Figure: Accuracy vs Communication Rounds
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    fig.suptitle("Toy Model Benchmark: FDA Pure vs. Baselines across Communication Rounds (MNIST)", fontsize=16, fontweight='bold', y=0.98)
    
    for idx, (sc, ar, title) in enumerate(SCENARIOS):
        ax = axes[idx // 2, idx % 2]
        for m in METHOD_ORDER:
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            acc_matrix = np.array([r["df"]["accuracy"].values[:100] * 100 for r in group])
            mean_acc = np.mean(acc_matrix, axis=0)
            std_acc = np.std(acc_matrix, axis=0) if len(group) > 1 else np.zeros_like(mean_acc)
            rounds = np.arange(1, len(mean_acc) + 1)
            
            cfg = METHODS_MAP[m]
            ax.plot(rounds, mean_acc, label=cfg["label"], color=cfg["color"],
                    linestyle=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                            color=cfg["color"], alpha=0.12)
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 90)
        ax.legend(loc="lower right", framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = FIG_DIR / "toy_pure_4panel_accuracy_vs_rounds.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print("Saved:", p1)

    # --------------------------------------------------------------------------
    # 2. 4-Panel Efficiency Curves: Accuracy vs Cumulative Local Epochs
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    fig.suptitle("Computation Trade-Off: Accuracy vs. Cumulative Local Epochs (MNIST)", fontsize=16, fontweight='bold', y=0.98)
    
    for idx, (sc, ar, title) in enumerate(SCENARIOS):
        ax = axes[idx // 2, idx % 2]
        for m in METHOD_ORDER:
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            
            max_ep = max(r["df"]["round_epochs"].sum() for r in group)
            common_grid = np.linspace(5, max_ep, 200)
            interp_accs = []
            for r in group:
                cum_epochs = r["df"]["round_epochs"].cumsum().values[:100]
                accs = r["df"]["accuracy"].values[:100] * 100
                interp_accs.append(np.interp(common_grid, cum_epochs, accs))
            
            interp_accs = np.array(interp_accs)
            mean_interp = np.mean(interp_accs, axis=0)
            std_interp = np.std(interp_accs, axis=0) if len(group) > 1 else np.zeros_like(mean_interp)
            
            cfg = METHODS_MAP[m]
            ax.plot(common_grid, mean_interp, label=cfg["label"], color=cfg["color"],
                    linestyle=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(common_grid, mean_interp - std_interp, mean_interp + std_interp,
                            color=cfg["color"], alpha=0.12)
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel("Cumulative Local Epochs Trained Across Clients")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_ylim(0, 90)
        ax.legend(loc="lower right", framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = FIG_DIR / "toy_pure_4panel_accuracy_vs_epochs.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print("Saved:", p2)

    # --------------------------------------------------------------------------
    # 3. 4-Panel Loss Trajectories: Test Loss vs Communication Rounds
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Convergence Loss: Test Loss vs. Communication Rounds (MNIST)", fontsize=16, fontweight='bold', y=0.98)
    
    for idx, (sc, ar, title) in enumerate(SCENARIOS):
        ax = axes[idx // 2, idx % 2]
        for m in METHOD_ORDER:
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            loss_col = "testing_loss" if "testing_loss" in group[0]["df"].columns else "loss"
            loss_matrix = np.array([r["df"][loss_col].values[:100] for r in group])
            mean_loss = np.mean(loss_matrix, axis=0)
            std_loss = np.std(loss_matrix, axis=0) if len(group) > 1 else np.zeros_like(mean_loss)
            rounds = np.arange(1, len(mean_loss) + 1)
            
            cfg = METHODS_MAP[m]
            ax.plot(rounds, mean_loss, label=cfg["label"], color=cfg["color"],
                    linestyle=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(rounds, mean_loss - std_loss, mean_loss + std_loss,
                            color=cfg["color"], alpha=0.10)
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Cross-Entropy Test Loss")
        ax.set_xlim(1, 100)
        ax.set_yscale('log')
        ax.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p3 = FIG_DIR / "toy_pure_4panel_loss_vs_rounds.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print("Saved:", p3)

    # --------------------------------------------------------------------------
    # 4. 4-Panel Inter-Client Drift Variance: Subspace Variance vs Rounds (Log Scale)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Parameter Drift Dynamics: Inter-Client Variance vs. Rounds (MNIST, Log Scale)", fontsize=16, fontweight='bold', y=0.98)
    
    for idx, (sc, ar, title) in enumerate(SCENARIOS):
        ax = axes[idx // 2, idx % 2]
        for m in METHOD_ORDER:
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            var_col = "variance" if "variance" in group[0]["df"].columns else "effective_variance"
            var_matrix = np.array([r["df"][var_col].values[:100] for r in group])
            # replace 0 or negative values with small epsilon for log plotting
            var_matrix = np.clip(var_matrix, 1e-4, None)
            mean_var = np.mean(var_matrix, axis=0)
            std_var = np.std(var_matrix, axis=0) if len(group) > 1 else np.zeros_like(mean_var)
            rounds = np.arange(1, len(mean_var) + 1)
            
            cfg = METHODS_MAP[m]
            ax.plot(rounds, mean_var, label=cfg["label"], color=cfg["color"],
                    linestyle=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(rounds, np.maximum(mean_var - std_var, 1e-4), mean_var + std_var,
                            color=cfg["color"], alpha=0.10)
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Inter-Client Subspace Variance (Log Scale)")
        ax.set_xlim(1, 100)
        ax.set_yscale('log')
        ax.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p_var = FIG_DIR / "toy_pure_4panel_variance_vs_rounds.png"
    plt.savefig(p_var, dpi=300)
    plt.close()
    print("Saved:", p_var)

    # --------------------------------------------------------------------------
    # 5. FDA Pure Internal Telemetry: Realized Local Epochs e* per Round
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    fig.suptitle("FDA Pure Adaptive Dynamics: Realized Local Epochs ($e^*$) per Round", fontsize=16, fontweight='bold', y=0.98)
    
    for idx, (sc, ar, title) in enumerate(SCENARIOS):
        ax = axes[idx // 2, idx % 2]
        for ratio, color, ls in [(1.0, "#d62728", "-"), (0.5, "#9467bd", "--")]:
            m = f"FDA Pure (k={ratio})"
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            ep_matrix = np.array([r["df"]["round_epochs"].values[:100] for r in group])
            mean_ep = np.mean(ep_matrix, axis=0)
            rounds = np.arange(1, len(mean_ep) + 1)
            
            ax.plot(rounds, mean_ep, label=f"FDA Pure (k={ratio})", color=color, linestyle=ls, linewidth=2.0)
        
        ax.axhline(5, color="#ff7f0e", linestyle=":", label="RoLoRA Baseline (E=5)", alpha=0.8, linewidth=1.5)
        ax.axhline(15, color="#7f7f7f", linestyle="--", label="e_max Ceiling (15)", alpha=0.6, linewidth=1.2)
        ax.axhline(1, color="#7f7f7f", linestyle=":", label="e_min Floor (1)", alpha=0.6, linewidth=1.2)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Realized Epochs ($e^*$)")
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 16)
        ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p4 = FIG_DIR / "toy_pure_epoch_dynamics_breakdown.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print("Saved:", p4)

    # --------------------------------------------------------------------------
    # 6. Individual Scenario Plots (Trajectory, Compute, Variance, and Combo)
    # --------------------------------------------------------------------------
    for sc, ar, title in SCENARIOS:
        # A. Trajectory (Accuracy vs Round)
        fig, ax = plt.subplots(figsize=(9, 6))
        for m in METHOD_ORDER:
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            acc_matrix = np.array([r["df"]["accuracy"].values[:100] * 100 for r in group])
            mean_acc = np.mean(acc_matrix, axis=0)
            std_acc = np.std(acc_matrix, axis=0) if len(group) > 1 else np.zeros_like(mean_acc)
            rounds = np.arange(1, len(mean_acc) + 1)
            
            cfg = METHODS_MAP[m]
            ax.plot(rounds, mean_acc, label=f"{cfg['label']} (N={len(group)})",
                    color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                            color=cfg["color"], alpha=0.15)
        
        ax.set_title(f"Accuracy Trajectory: {title}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 90)
        ax.legend(loc="lower right", framealpha=0.92)
        
        p = FIG_DIR / f"toy_pure_trajectory_{sc}_{ar.lower()}.png"
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved:", p)

        # B. Compute Efficiency (Accuracy vs Cumulative Local Epochs)
        fig, ax = plt.subplots(figsize=(9, 6))
        for m in METHOD_ORDER:
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            max_ep = max(r["df"]["round_epochs"].sum() for r in group)
            common_grid = np.linspace(5, max_ep, 200)
            interp_accs = []
            for r in group:
                cum_epochs = r["df"]["round_epochs"].cumsum().values[:100]
                accs = r["df"]["accuracy"].values[:100] * 100
                interp_accs.append(np.interp(common_grid, cum_epochs, accs))
            interp_accs = np.array(interp_accs)
            mean_interp = np.mean(interp_accs, axis=0)
            std_interp = np.std(interp_accs, axis=0) if len(group) > 1 else np.zeros_like(mean_interp)

            cfg = METHODS_MAP[m]
            ax.plot(common_grid, mean_interp, label=f"{cfg['label']} (N={len(group)})",
                    color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(common_grid, mean_interp - std_interp, mean_interp + std_interp,
                            color=cfg["color"], alpha=0.15)

        ax.set_title(f"Compute Efficiency: {title}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Cumulative Local Epochs Trained Across Clients", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_ylim(0, 90)
        ax.legend(loc="lower right", framealpha=0.92)
        
        p_comp = FIG_DIR / f"toy_pure_compute_{sc}_{ar.lower()}.png"
        plt.savefig(p_comp, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved:", p_comp)

        # C. Inter-Client Drift Variance (Log Scale)
        fig, ax = plt.subplots(figsize=(9, 6))
        for m in METHOD_ORDER:
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            var_col = "variance" if "variance" in group[0]["df"].columns else "effective_variance"
            var_matrix = np.array([r["df"][var_col].values[:100] for r in group])
            var_matrix = np.clip(var_matrix, 1e-4, None)
            mean_var = np.mean(var_matrix, axis=0)
            std_var = np.std(var_matrix, axis=0) if len(group) > 1 else np.zeros_like(mean_var)
            rounds = np.arange(1, len(mean_var) + 1)

            cfg = METHODS_MAP[m]
            ax.plot(rounds, mean_var, label=f"{cfg['label']}",
                    color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(rounds, np.maximum(mean_var - std_var, 1e-4), mean_var + std_var,
                            color=cfg["color"], alpha=0.12)

        ax.set_title(f"Inter-Client Subspace Drift Variance: {title}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel("Drift Variance (Log Scale)", fontsize=12)
        ax.set_xlim(1, 100)
        ax.set_yscale('log')
        ax.legend(loc="upper right", framealpha=0.92)

        p_var_single = FIG_DIR / f"toy_pure_variance_{sc}_{ar.lower()}.png"
        plt.savefig(p_var_single, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved:", p_var_single)

        # D. High-Res Dual-Panel Figure: [Accuracy vs Rounds | Accuracy vs Cumulative Epochs]
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"{title} - Comprehensive Benchmark Telemetry", fontsize=15, fontweight='bold', y=0.98)

        # Left panel: Rounds
        for m in METHOD_ORDER:
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            acc_matrix = np.array([r["df"]["accuracy"].values[:100] * 100 for r in group])
            mean_acc = np.mean(acc_matrix, axis=0)
            std_acc = np.std(acc_matrix, axis=0) if len(group) > 1 else np.zeros_like(mean_acc)
            rounds = np.arange(1, len(mean_acc) + 1)
            cfg = METHODS_MAP[m]
            ax_left.plot(rounds, mean_acc, label=cfg["label"], color=cfg["color"],
                         linestyle=cfg["ls"], linewidth=cfg["lw"])
            ax_left.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                                 color=cfg["color"], alpha=0.14)
        ax_left.set_title("Test Accuracy vs. Communication Rounds", fontsize=12, fontweight='bold')
        ax_left.set_xlabel("Communication Round", fontsize=11)
        ax_left.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax_left.set_xlim(1, 100)
        ax_left.set_ylim(0, 90)
        ax_left.legend(loc="lower right", framealpha=0.92)

        # Right panel: Compute
        for m in METHOD_ORDER:
            key = (sc, ar, m)
            if key not in grouped:
                continue
            group = grouped[key]
            max_ep = max(r["df"]["round_epochs"].sum() for r in group)
            common_grid = np.linspace(5, max_ep, 200)
            interp_accs = []
            for r in group:
                cum_epochs = r["df"]["round_epochs"].cumsum().values[:100]
                accs = r["df"]["accuracy"].values[:100] * 100
                interp_accs.append(np.interp(common_grid, cum_epochs, accs))
            interp_accs = np.array(interp_accs)
            mean_interp = np.mean(interp_accs, axis=0)
            std_interp = np.std(interp_accs, axis=0) if len(group) > 1 else np.zeros_like(mean_interp)
            cfg = METHODS_MAP[m]
            ax_right.plot(common_grid, mean_interp, label=cfg["label"], color=cfg["color"],
                          linestyle=cfg["ls"], linewidth=cfg["lw"])
            ax_right.fill_between(common_grid, mean_interp - std_interp, mean_interp + std_interp,
                                  color=cfg["color"], alpha=0.14)
        ax_right.set_title("Test Accuracy vs. Cumulative Local Epochs", fontsize=12, fontweight='bold')
        ax_right.set_xlabel("Cumulative Local Epochs Trained Across Clients", fontsize=11)
        ax_right.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax_right.set_ylim(0, 90)
        ax_right.legend(loc="lower right", framealpha=0.92)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p_combo = FIG_DIR / f"toy_pure_combo_{sc}_{ar.lower()}.png"
        plt.savefig(p_combo, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved:", p_combo)

if __name__ == '__main__':
    runs = load_data()
    print(f"Loaded total runs: {len(runs)}")
    grouped, stats = compute_statistics(runs)
    generate_plots(grouped)
    
    stats_json_path = FIG_DIR / "stats_summary.json"
    clean_stats = []
    for s in stats:
        s_copy = s.copy()
        s_copy["milestones"] = {str(k): list(map(int, v)) for k, v in s["milestones"].items()}
        clean_stats.append(s_copy)
    stats_json_path.write_text(json.dumps(clean_stats, indent=2), encoding="utf-8")
    print("Saved stats JSON:", stats_json_path)
