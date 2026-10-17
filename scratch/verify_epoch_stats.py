import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

PURE_DIR = Path("results/round_metrics")
ALLOWED_SEEDS = {42, 43, 44}

runs = []
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
        "mean_e": df['round_epochs'].mean(),
        "mean_a": df[df['active_matrix'] == 'A']['round_epochs'].mean(),
        "mean_b": df[df['active_matrix'] == 'B']['round_epochs'].mean(),
        "tot_e": df['round_epochs'].sum()
    })

for scen in ["10c1", "5c2"]:
    for arch in ["NONLIN", "LIN"]:
        for method in ["FedAvg-LoRA", "RoLoRA"]:
            for s in ALLOWED_SEEDS:
                runs.append({
                    "method": method, "arch": arch, "scen": scen, "seed": s,
                    "mean_e": 5.0, "mean_a": 5.0, "mean_b": 5.0, "tot_e": 500.0
                })

df_all = pd.DataFrame(runs)
summary = df_all.groupby(["scen", "arch", "method"]).agg(
    mean_e_m=("mean_e", "mean"), mean_e_s=("mean_e", "std"),
    mean_a_m=("mean_a", "mean"), mean_a_s=("mean_a", "std"),
    mean_b_m=("mean_b", "mean"), mean_b_s=("mean_b", "std"),
    tot_m=("tot_e", "mean"), tot_s=("tot_e", "std")
).reset_index()

for idx, r in summary.iterrows():
    print(f"{r['scen']} {r['arch']} | {r['method']}: mean_e={r['mean_e_m']:.2f} +/- {r['mean_e_s']:.2f}, mean_a={r['mean_a_m']:.2f} +/- {r['mean_a_s']:.2f}, mean_b={r['mean_b_m']:.2f} +/- {r['mean_b_s']:.2f}, tot={r['tot_m']:.1f} +/- {r['tot_s']:.1f}")
