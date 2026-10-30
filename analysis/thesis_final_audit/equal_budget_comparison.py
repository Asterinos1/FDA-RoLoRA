import os
import sys
import glob
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from update_benchmark_suite import pure_configs, get_methods, prep, comm_rates

OUT_DIR = os.path.join(BASE_DIR, "analysis", "thesis_final_audit")
os.makedirs(OUT_DIR, exist_ok=True)

# Correct comm rate for SST-2: 3 layers = 144 KB per client per round
# All others: 5 layers = 240 KB per client per round
def get_comm_rate(dataset, cohort_C):
    if dataset.lower() == "sst2":
        return 144.0 * cohort_C / 1024.0 # MB per round
    else:
        return 240.0 * cohort_C / 1024.0 # MB per round

# Milestones chosen a priori, above majority-class baselines:
# SST-2: 80%, 85%, 90%
# QNLI:  75%, 80%, 85%
# MNLI:  65%, 70%, 75%
# QQP:   70%, 75%, 80%
TASK_MILESTONES = {
    "sst2": [0.80, 0.85, 0.90],
    "qnli": [0.75, 0.80, 0.85],
    "mnli": [0.65, 0.70, 0.75],
    "qqp":  [0.70, 0.75, 0.80],
}

print("=== Running Equal-Budget & Milestone Comparison ===")

# Load all valid runs for each config
milestone_rows = []
matched_budget_rows = []

for cfg in pure_configs:
    cfg_prefix, title, ds, c, r_cfg, lr, _, _ = cfg
    methods = get_methods(ds, c, r_cfg, lr, include_lora=False, cfg_prefix=cfg_prefix, pure_only=True)
    rate_mb = get_comm_rate(ds, c)
    alpha = 0.5 if "A0.5" in cfg_prefix else 1000.0
    
    # Store processed method dfs: method_name -> list of seed DataFrames
    method_dfs = {}
    for mname, pat, is_lora in methods:
        matches = pat if isinstance(pat, list) else sorted(glob.glob(pat))
        valid_dfs = []
        for f in matches:
            # Exclude contaminated aborted run
            if "5095b3daf3" in f:
                continue
            df = prep(pd.read_csv(f))
            if len(df) > 0:
                # Add exact MB column
                df["exact_mb"] = df["round"] * rate_mb
                valid_dfs.append(df)
        if valid_dfs:
            method_dfs[mname] = valid_dfs

    # 1. Milestone Analysis: Calculate round and MB to reach milestone for each seed, then average
    milestones = TASK_MILESTONES.get(ds.lower(), [0.70, 0.75, 0.80])
    for target_m in milestones:
        for mname, dfs in method_dfs.items():
            seed_rounds = []
            seed_mbs = []
            reached_count = 0
            for df in dfs:
                # Find first round where accuracy >= target_m
                hit = df[df["accuracy"] >= target_m]
                if len(hit) > 0:
                    r_hit = hit.iloc[0]["round"]
                    mb_hit = hit.iloc[0]["exact_mb"]
                    seed_rounds.append(r_hit)
                    seed_mbs.append(mb_hit)
                    reached_count += 1
                else:
                    seed_rounds.append(np.nan)
                    seed_mbs.append(np.nan)
            
            # Report
            if reached_count == len(dfs):
                r_mean = np.mean(seed_rounds)
                r_std = np.std(seed_rounds, ddof=1) if len(seed_rounds) > 1 else 0.0
                mb_mean = np.mean(seed_mbs)
                mb_std = np.std(seed_mbs, ddof=1) if len(seed_mbs) > 1 else 0.0
                status = "REACHED"
            elif reached_count > 0:
                r_mean = np.nanmean(seed_rounds)
                r_std = np.nan
                mb_mean = np.nanmean(seed_mbs)
                mb_std = np.nan
                status = f"PARTIAL ({reached_count}/{len(dfs)})"
            else:
                r_mean = np.nan
                r_std = np.nan
                mb_mean = np.nan
                mb_std = np.nan
                status = "UNATTAINED"

            milestone_rows.append({
                "cfg_prefix": cfg_prefix,
                "dataset": ds,
                "cohort_C": c,
                "alpha": alpha,
                "milestone": target_m,
                "milestone_pct": f"{int(target_m*100)}%",
                "method": mname,
                "n_seeds": len(dfs),
                "reached_count": reached_count,
                "status": status,
                "round_mean": r_mean,
                "round_std": r_std,
                "mb_mean": mb_mean,
                "mb_std": mb_std,
            })

    # 2. Matched Budget Evaluation:
    # Determine the maximum common communication budget available across primary methods:
    # Primary methods: FDA Pure k=1.0, FDA Pure k=0.5, RoLoRA E20, RoLoRA E10
    key_methods = [m for m in ["FDA Pure E1TO20 [k=1.0]", "FDA Pure E1TO20 [k=0.5]", "RoLoRA E20", "RoLoRA E10"] if m in method_dfs]
    if len(key_methods) >= 2:
        # Find maximum common MB: min over key methods of max MB reached by all seeds
        max_mb_per_method = {}
        for km in key_methods:
            # For each seed, what is max MB?
            seed_max_mbs = [df["exact_mb"].max() for df in method_dfs[km]]
            max_mb_per_method[km] = min(seed_max_mbs) # budget available in all seeds of this method
        
        common_budget_mb = min(max_mb_per_method.values())
        
        # Also check common round horizon
        min_rounds_per_method = {}
        for km in key_methods:
            min_rounds_per_method[km] = min([len(df) for df in method_dfs[km]])
        common_round_horizon = min(min_rounds_per_method.values())

        # Evaluate accuracy at common_budget_mb and at common_round_horizon
        for km in key_methods:
            dfs = method_dfs[km]
            accs_at_budget = []
            accs_at_round = []
            peak_up_to_budget = []
            for df in dfs:
                # df up to common_budget_mb
                sub_b = df[df["exact_mb"] <= common_budget_mb + 1e-4]
                if len(sub_b) > 0:
                    accs_at_budget.append(sub_b.iloc[-1]["accuracy"])
                    peak_up_to_budget.append(sub_b["accuracy"].max())
                else:
                    accs_at_budget.append(np.nan)
                    peak_up_to_budget.append(np.nan)
                
                # df at common_round_horizon
                sub_r = df[df["round"] <= common_round_horizon]
                if len(sub_r) > 0:
                    accs_at_round.append(sub_r.iloc[-1]["accuracy"])
                else:
                    accs_at_round.append(np.nan)

            matched_budget_rows.append({
                "cfg_prefix": cfg_prefix,
                "dataset": ds,
                "cohort_C": c,
                "alpha": alpha,
                "method": km,
                "common_budget_mb": common_budget_mb,
                "acc_at_budget_mean": np.nanmean(accs_at_budget),
                "acc_at_budget_std": np.nanstd(accs_at_budget, ddof=1) if len(accs_at_budget) > 1 else 0.0,
                "peak_at_budget_mean": np.nanmean(peak_up_to_budget),
                "peak_at_budget_std": np.nanstd(peak_up_to_budget, ddof=1) if len(peak_up_to_budget) > 1 else 0.0,
                "common_round_horizon": common_round_horizon,
                "acc_at_round_mean": np.nanmean(accs_at_round),
                "acc_at_round_std": np.nanstd(accs_at_round, ddof=1) if len(accs_at_round) > 1 else 0.0,
            })

df_milestones = pd.DataFrame(milestones_rows if 'milestones_rows' in locals() else milestone_rows)
df_milestones.to_csv(os.path.join(OUT_DIR, "milestone_comparison.csv"), index=False)

df_matched = pd.DataFrame(matched_budget_rows)
df_matched.to_csv(os.path.join(OUT_DIR, "matched_budget_comparison.csv"), index=False)

print(f"Generated milestone_comparison.csv ({len(df_milestones)} rows)")
print(f"Generated matched_budget_comparison.csv ({len(df_matched)} rows)")
