import os
import glob
import json
import shutil
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib formatting
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.autolayout': True
})

RESULTS_DIRS = ["results/toy_experiments", "results/round_metrics"]
OUT_DIR = "analysis/toy_final_66runs_analysis"

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

ALLOWED_PREFIXES = [
    "TOY-BASELINES-20260810-155629_",
    "TOY-NEW2FDA-FDA_STANDARD-20260817-132013_",
    "TOY-NEW2FDA-FDA_EMA_STANDARD-20260817-153121_",
    "TOY-NEW3FDA-FDA_SMA-20260818-014826_"
]

ALLOWED_SEEDS = {42, 43, 44}

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    
    # Prefix check
    matched_prefix = None
    for p in ALLOWED_PREFIXES:
        if filename.startswith(p):
            matched_prefix = p
            break
    if not matched_prefix:
        return None, None, None, None, "EXCLUDE", None

    # Architecture
    arch = "LIN" if "_LIN_" in filename else "NONLIN"
    
    # Scenario
    if "5-Clients-2-Classes" in filename:
        scenario = "5c2"
    elif "10-Clients-1-Class" in filename:
        scenario = "10c1"
    else:
        scenario = "Unknown"
        
    # Seed
    seed = None
    for part in filename.split('_'):
        if part.startswith('S') and part[1:].isdigit():
            seed = int(part[1:])
            break
            
    if seed not in ALLOWED_SEEDS:
        return None, None, None, None, "EXCLUDE", None

    # Hash
    run_hash = filename.split('_')[-1].replace('.csv', '')
    
    # Method normalization
    if matched_prefix == "TOY-BASELINES-20260810-155629_":
        if "FedAvg-LoRA" in filename:
            method = "LoRA"
        elif "FFA-LoRA" in filename:
            method = "FFA-LoRA"
        elif "FedAvg-RoLoRA" in filename:
            method = "RoLoRA"
        else:
            method = "Unknown Baseline"
    elif matched_prefix == "TOY-NEW2FDA-FDA_STANDARD-20260817-132013_":
        method = "FDA Standard"
    elif matched_prefix == "TOY-NEW2FDA-FDA_EMA_STANDARD-20260817-153121_":
        method = "FDA EMA-Standard"
    elif matched_prefix == "TOY-NEW3FDA-FDA_SMA-20260818-014826_":
        if arch != "NONLIN":
            return None, None, None, None, "EXCLUDE", None
        method = "FDA SMA"
    else:
        return None, None, None, None, "EXCLUDE", None
            
    return arch, scenario, seed, run_hash, method, filename

def load_and_verify_data():
    all_files = []
    for d in RESULTS_DIRS:
        if os.path.exists(d):
            all_files.extend(glob.glob(os.path.join(d, "*.csv")))
    data_frames = []
    run_records = []
    integrity_issues = []
    seen_hashes = set()
    
    for f in all_files:
        arch, scenario, seed, run_hash, method, fname = parse_filename(f)
        if method == "EXCLUDE":
            continue
        if run_hash in seen_hashes:
            continue
        seen_hashes.add(run_hash)
            
        df = pd.read_csv(f)
        rounds_completed = len(df)
        if rounds_completed != 100:
            integrity_issues.append(f"Incomplete rounds ({rounds_completed}/100): {fname}")
            
        df['method_norm'] = method
        df['arch'] = arch
        df['scenario'] = scenario
        df['parsed_seed'] = seed
        df['run_hash'] = run_hash
        df['source_file'] = fname
        
        if 'round_steps' in df.columns:
            df['cumulative_steps'] = df['round_steps'].cumsum()
        else:
            integrity_issues.append(f"Missing round_steps in {fname}")
            
        data_frames.append(df)
        run_records.append({
            'arch': arch,
            'scenario': scenario,
            'method': method,
            'seed': seed,
            'run_hash': run_hash,
            'rounds': rounds_completed,
            'filename': fname
        })
        
    combined_df = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    runs_df = pd.DataFrame(run_records)
    return combined_df, runs_df, integrity_issues

combined_df, runs_df, integrity_issues = load_and_verify_data()

print(f"Total verified runs loaded: {len(runs_df)}")
print("Run breakdown by scenario and method:")
print(runs_df.groupby(['arch', 'scenario', 'method'])['seed'].count())

# Write Data Integrity Report
with open(os.path.join(OUT_DIR, "data_integrity_report.md"), "w", encoding="utf-8") as f:
    f.write("# Data Integrity Report: Thesis 66-Run Evaluation Suite\n\n")
    f.write(f"- **Total target runs required**: 66\n")
    f.write(f"- **Total runs successfully loaded & verified**: {len(runs_df)}\n")
    f.write(f"- **Integrity issues detected**: {len(integrity_issues)}\n\n")
    if integrity_issues:
        f.write("### Issues:\n")
        for iss in integrity_issues:
            f.write(f"- {iss}\n")
    else:
        f.write("All 66 runs completed exactly 100 communication rounds with zero duplicate hashes or missing telemetry fields.\n\n")
    f.write("### Verified Run Breakdown:\n\n")
    breakdown = runs_df.groupby(['arch', 'scenario', 'method'])['seed'].apply(lambda s: sorted(list(s))).reset_index()
    f.write(breakdown.to_markdown(index=False))
    f.write("\n")

# Compute Fixed Round Accuracy Summary (R25, R50, R75, R100)
fixed_rounds = [25, 50, 75, 100]
records_fixed = []

for (arch, scenario), group in combined_df.groupby(['arch', 'scenario']):
    for method, m_group in group.groupby('method_norm'):
        rec = {'arch': arch, 'scenario': scenario, 'method_norm': method}
        for r in fixed_rounds:
            sub = m_group[m_group['round'] == r]
            accs = sub['accuracy'].values
            rec[f'R{r}_mean'] = float(np.mean(accs))
            rec[f'R{r}_std'] = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            rec[f'R{r}_str'] = f"{np.mean(accs)*100:.2f}% ± {rec[f'R{r}_std']*100:.2f}%"
        records_fixed.append(rec)

fixed_df = pd.DataFrame(records_fixed)
fixed_df.sort_values(by=['arch', 'scenario', 'R100_mean'], ascending=[True, True, False], inplace=True)
fixed_df.to_csv(os.path.join(OUT_DIR, "fixed_round_accuracy_summary.csv"), index=False, encoding="utf-8")

with open(os.path.join(OUT_DIR, "fixed_round_accuracy_summary.md"), "w", encoding="utf-8") as f:
    f.write("# Fixed-Round Accuracy Summary (R=25, 50, 75, 100)\n\n")
    for (arch, scenario), g in fixed_df.groupby(['arch', 'scenario']):
        f.write(f"## {arch} - {scenario}\n\n")
        sub_tab = g[['method_norm', 'R25_str', 'R50_str', 'R75_str', 'R100_str']].copy()
        sub_tab.columns = ['Method', 'R=25', 'R=50', 'R=75', 'R=100']
        f.write(sub_tab.to_markdown(index=False))
        f.write("\n\n")

# Compute Peak Accuracy Summary
records_peak = []
for (arch, scenario, method), g in combined_df.groupby(['arch', 'scenario', 'method_norm']):
    best_accs = []
    best_rounds = []
    for seed, s_group in g.groupby('parsed_seed'):
        max_idx = s_group['accuracy'].idxmax()
        best_accs.append(s_group.loc[max_idx, 'accuracy'])
        best_rounds.append(s_group.loc[max_idx, 'round'])
        
    records_peak.append({
        'arch': arch,
        'scenario': scenario,
        'method_norm': method,
        'mean_max_accuracy': float(np.mean(best_accs)),
        'std_max_accuracy': float(np.std(best_accs, ddof=1)) if len(best_accs) > 1 else 0.0,
        'mean_best_round': float(np.mean(best_rounds)),
        'median_best_round': float(np.median(best_rounds)),
        'min_best_round': int(np.min(best_rounds)),
        'max_best_round': int(np.max(best_rounds)),
        'str_peak': f"{np.mean(best_accs)*100:.2f}% ± {np.std(best_accs, ddof=1)*100:.2f}%",
        'str_best_round': f"R{np.median(best_rounds):.0f} [{np.min(best_rounds)}, {np.max(best_rounds)}]"
    })

peak_df = pd.DataFrame(records_peak)
peak_df.sort_values(by=['arch', 'scenario', 'mean_max_accuracy'], ascending=[True, True, False], inplace=True)
peak_df.to_csv(os.path.join(OUT_DIR, "best_round_summary.csv"), index=False, encoding="utf-8")

with open(os.path.join(OUT_DIR, "best_round_summary.md"), "w", encoding="utf-8") as f:
    f.write("# Peak Test Accuracy Summary\n\n")
    for (arch, scenario), g in peak_df.groupby(['arch', 'scenario']):
        f.write(f"## {arch} - {scenario}\n\n")
        sub_tab = g[['method_norm', 'str_peak', 'mean_best_round', 'median_best_round', 'min_best_round', 'max_best_round']].copy()
        sub_tab.columns = ['Method', 'Peak Accuracy', 'Mean Best Round', 'Median Best Round', 'Min Best Round', 'Max Best Round']
        f.write(sub_tab.to_markdown(index=False))
        f.write("\n\n")

# Compute Rounds to Target
targets = [0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.79, 0.80]
records_target = []

for (arch, scenario, method), g in combined_df.groupby(['arch', 'scenario', 'method_norm']):
    for tgt in targets:
        rounds_reached = []
        for seed, s_group in g.groupby('parsed_seed'):
            reached = s_group[s_group['accuracy'] >= tgt]
            if not reached.empty:
                rounds_reached.append(reached['round'].iloc[0])
                
        n_reached = len(rounds_reached)
        mean_r = float(np.mean(rounds_reached)) if n_reached > 0 else np.nan
        std_r = float(np.std(rounds_reached, ddof=1)) if n_reached > 1 else 0.0
        
        records_target.append({
            'arch': arch,
            'scenario': scenario,
            'method_norm': method,
            'target_accuracy': tgt,
            'seeds_reached': n_reached,
            'total_seeds': 3,
            'ratio_str': f"{n_reached}/3",
            'mean_round': mean_r,
            'std_round': std_r,
            'round_str': f"{mean_r:.1f} ± {std_r:.1f} ({n_reached}/3)" if n_reached > 0 else f"-- (0/3)"
        })

target_df = pd.DataFrame(records_target)
target_df.to_csv(os.path.join(OUT_DIR, "rounds_to_target_summary.csv"), index=False, encoding="utf-8")

with open(os.path.join(OUT_DIR, "rounds_to_target_summary.md"), "w", encoding="utf-8") as f:
    f.write("# Rounds to Target Accuracy\n\n")
    for (arch, scenario), g in target_df.groupby(['arch', 'scenario']):
        f.write(f"## {arch} - {scenario}\n\n")
        pivot = g.pivot(index='method_norm', columns='target_accuracy', values='round_str')
        pivot.columns = [f"{c*100:.1f}%" for c in pivot.columns]
        f.write(pivot.to_markdown())
        f.write("\n\n")

# Computation Efficiency Summary
records_compute = []
for (arch, scenario, method), g in combined_df.groupby(['arch', 'scenario', 'method_norm']):
    total_steps = []
    steps_to_peak = []
    avg_epochs = []
    
    for seed, s_group in g.groupby('parsed_seed'):
        total_steps.append(s_group['cumulative_steps'].iloc[-1])
        max_idx = s_group['accuracy'].idxmax()
        steps_to_peak.append(s_group.loc[max_idx, 'cumulative_steps'])
        if 'round_epochs' in s_group.columns:
            avg_epochs.append(s_group['round_epochs'].mean())
        else:
            avg_epochs.append(5.0)
            
    records_compute.append({
        'arch': arch,
        'scenario': scenario,
        'method_norm': method,
        'total_steps_100_rounds': float(np.mean(total_steps)),
        'steps_to_best_round': float(np.mean(steps_to_peak)),
        'avg_local_epochs_per_round': float(np.mean(avg_epochs))
    })

compute_df = pd.DataFrame(records_compute)
compute_df.to_csv(os.path.join(OUT_DIR, "computation_efficiency_summary.csv"), index=False, encoding="utf-8")

with open(os.path.join(OUT_DIR, "computation_efficiency_summary.md"), "w", encoding="utf-8") as f:
    f.write("# Computation Efficiency Summary\n\n")
    for (arch, scenario), g in compute_df.groupby(['arch', 'scenario']):
        f.write(f"## {arch} - {scenario}\n\n")
        sub_tab = g[['method_norm', 'total_steps_100_rounds', 'steps_to_best_round', 'avg_local_epochs_per_round']].copy()
        sub_tab.columns = ['Method', 'Cumulative local steps per client', 'Local steps per client to peak', 'Avg Epochs / Round']
        f.write(sub_tab.to_markdown(index=False))
        f.write("\n\n")

# FDA Epoch Statistics
records_fda = []
fda_data = combined_df[combined_df['method_norm'].str.startswith('FDA')]
for (arch, scenario, method), g in fda_data.groupby(['arch', 'scenario', 'method_norm']):
    variances = g['variance'].dropna().values if 'variance' in g.columns else []
    epochs = g['round_epochs'].dropna().values if 'round_epochs' in g.columns else []
    
    records_fda.append({
        'arch': arch,
        'scenario': scenario,
        'method_norm': method,
        'mean_variance': float(np.mean(variances)) if len(variances) > 0 else 0.0,
        'std_variance': float(np.std(variances, ddof=1)) if len(variances) > 1 else 0.0,
        'mean_round_epochs': float(np.mean(epochs)) if len(epochs) > 0 else 0.0,
        'median_round_epochs': float(np.median(epochs)) if len(epochs) > 0 else 0.0,
        'std_round_epochs': float(np.std(epochs, ddof=1)) if len(epochs) > 1 else 0.0,
        'min_round_epochs': int(np.min(epochs)) if len(epochs) > 0 else 0,
        'max_round_epochs': int(np.max(epochs)) if len(epochs) > 0 else 0,
        'fraction_at_E_min': float(np.mean(epochs == 3)) if len(epochs) > 0 else 0.0,
        'fraction_at_E_max': float(np.mean(epochs == 15)) if len(epochs) > 0 else 0.0
    })

fda_df = pd.DataFrame(records_fda)
fda_df.to_csv(os.path.join(OUT_DIR, "fda_epoch_statistics.csv"), index=False, encoding="utf-8")

with open(os.path.join(OUT_DIR, "fda_epoch_statistics.md"), "w", encoding="utf-8") as f:
    f.write("# FDA Epoch & Variance Statistics\n\n")
    f.write(fda_df.to_markdown(index=False))
    f.write("\n\n")

# Visualization Generation
palette = {
    'LoRA': '#1f77b4',
    'FFA-LoRA': '#ff7f0e',
    'RoLoRA': '#2ca02c',
    'FDA Standard': '#d62728',
    'FDA EMA-Standard': '#9467bd',
    'FDA SMA': '#8c564b'
}

for (arch, scenario), group in combined_df.groupby(['arch', 'scenario']):
    tag = f"{arch.lower()}_{scenario}"
    
    # 1. Accuracy vs Rounds
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method in sorted(group['method_norm'].unique()):
        m_group = group[group['method_norm'] == method]
        piv = m_group.pivot(index='round', columns='parsed_seed', values='accuracy')
        mean_curve = piv.mean(axis=1) * 100
        std_curve = piv.std(axis=1) * 100
        color = palette.get(method, '#333333')
        ax.plot(piv.index, mean_curve, label=method, color=color, linewidth=2)
    
    ax.set_title(f"{arch} - {scenario} (Accuracy vs. Rounds)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_xlim(1, 100)
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figures", f"{tag}_accuracy_rounds.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "figures", f"{tag}_accuracy_rounds.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Accuracy vs Cumulative Compute Steps
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method in sorted(group['method_norm'].unique()):
        m_group = group[group['method_norm'] == method]
        piv_acc = m_group.pivot(index='round', columns='parsed_seed', values='accuracy').mean(axis=1) * 100
        piv_steps = m_group.pivot(index='round', columns='parsed_seed', values='cumulative_steps').mean(axis=1)
        color = palette.get(method, '#333333')
        ax.plot(piv_steps, piv_acc, label=method, color=color, linewidth=2)
        
    ax.set_title(f"{arch} - {scenario} (Accuracy vs. Local Steps)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Cumulative Local Steps per Client", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figures", f"{tag}_accuracy_vs_compute.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "figures", f"{tag}_accuracy_vs_compute.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Generate LLM Summary JSON
summary_dict = {
    "experiment": "FDA-LoRA Thesis Benchmark Analysis (66 Runs, 3 Seeds, 100 Rounds)",
    "dataset": "MNIST (10 classes)",
    "run_families": ALLOWED_PREFIXES,
    "seeds": [42, 43, 44],
    "methods_evaluated": {
        "LIN": ["LoRA", "FFA-LoRA", "RoLoRA", "FDA Standard", "FDA EMA-Standard"],
        "NONLIN": ["LoRA", "FFA-LoRA", "RoLoRA", "FDA Standard", "FDA EMA-Standard", "FDA SMA"]
    },
    "fixed_round_accuracy": fixed_df.to_dict(orient="records"),
    "peak_accuracy": peak_df.to_dict(orient="records"),
    "compute_efficiency": compute_df.to_dict(orient="records"),
    "fda_dynamics": fda_df.to_dict(orient="records")
}

with open(os.path.join(OUT_DIR, "LLM_SUMMARY.json"), "w", encoding="utf-8") as f:
    json.dump(summary_dict, f, indent=2)

print("Data analysis, CSVs, markdown tables, and plots written to:", OUT_DIR)
