import sys
sys.path.append(".")
import os
import glob
import numpy as np
import pandas as pd
from update_benchmark_suite import (
    pure_configs, get_methods, prep, average_runs, comm_rates
)

def test_scorecard():
    rows = []
    for cfg_tuple in pure_configs:
        cfg_prefix, title, ds, c, r, lr, milestones, m_headers = cfg_tuple
        methods = get_methods(ds, c, r, lr, include_lora=False, cfg_prefix=cfg_prefix, pure_only=True)
        
        fda_candidates = []
        rolora_candidates = []
        
        for mname, pat, is_lora in methods:
            matches = pat if isinstance(pat, list) else sorted(glob.glob(pat))
            if not matches:
                continue
            dfs = [prep(pd.read_csv(f)) for f in matches]
            rate = comm_rates[(c, is_lora)]
            df_avg = average_runs(dfs)
            max_acc = float(df_avg["accuracy"].max())
            tot_r = len(df_avg)
            mb = tot_r * rate
            ep = float(np.mean([df["round_epochs"].mean() for df in dfs]))
            
            entry = {
                "mname": mname,
                "max_acc": max_acc,
                "tot_r": tot_r,
                "mb": mb,
                "ep": ep,
            }
            if "FDA Pure" in mname:
                fda_candidates.append(entry)
            elif "RoLoRA" in mname:
                rolora_candidates.append(entry)
                
        if not fda_candidates or not rolora_candidates:
            continue
            
        best_fda = max(fda_candidates, key=lambda x: x["max_acc"])
        best_rolora = max(rolora_candidates, key=lambda x: x["max_acc"])
        
        delta_acc = (best_fda["max_acc"] - best_rolora["max_acc"]) * 100.0
        
        # Clean title for display
        clean_title = title.replace("$", "").replace("\\text{lr}=", "lr=").replace("\\alpha=", "alpha=")
        
        rows.append({
            "scenario": clean_title,
            "fda_mname": best_fda["mname"],
            "fda_acc": best_fda["max_acc"],
            "fda_ep": best_fda["ep"],
            "fda_mb": best_fda["mb"],
            "ro_mname": best_rolora["mname"],
            "ro_acc": best_rolora["max_acc"],
            "ro_ep": best_rolora["ep"],
            "ro_mb": best_rolora["mb"],
            "delta_acc": delta_acc,
        })
        
    for r in rows:
        sign = "+" if r["delta_acc"] >= 0 else ""
        print(f"{r['scenario']:<45} | FDA: {r['fda_mname']:<25} ({r['fda_acc']*100:.2f}%, ep={r['fda_ep']:.1f}) | RoLoRA: {r['ro_mname']:<12} ({r['ro_acc']*100:.2f}%) | Delta: {sign}{r['delta_acc']:.2f}%")

if __name__ == "__main__":
    test_scorecard()
