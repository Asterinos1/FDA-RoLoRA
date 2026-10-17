import glob
import os
import re
import pandas as pd
from collections import defaultdict

# Find all CSV files in results/round_metrics
csv_files = sorted(glob.glob("results/round_metrics/*.csv"))

records = []
for path in csv_files:
    fname = os.path.basename(path)
    # Filter out toy runs and LR calibration runs
    if fname.startswith("TOY-") or "LR4-E1" in fname:
        continue
        
    try:
        # Read header + first row to get parameters
        df = pd.read_csv(path, nrows=5)
        if df.empty:
            continue
        row0 = df.iloc[0]
        
        # Read total rows
        # Let's count lines quickly
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            line_count = sum(1 for _ in f) - 1 # minus header
            
        ds = str(row0.get("ds_name", "")).lower()
        if not ds or ds == "nan":
            for d in ["sst2", "qnli", "mnli", "qqp", "rte", "mrpc", "cola"]:
                if f"_{d}_" in fname.lower() or f"-{d}-" in fname.lower() or f"_{d}." in fname.lower() or f"-{d}." in fname.lower() or f"_ds-{d}_" in fname.lower() or f"-ds-{d}-" in fname.lower():
                    ds = d
                    break
                    
        method = str(row0.get("METHOD_NAME", row0.get("method", "")))
        if not method or method == "nan":
            # parse from filename
            if "M-lora" in fname: method = "LoRA"
            elif "M-rolora" in fname: method = "RoLoRA"
            elif "M-ffa-lora" in fname: method = "FFA-LoRA"
            elif "M-fda" in fname or "FDA" in fname: method = "FDA"
            else: method = "Other"
            
        num_clients = row0.get("num_clients", None)
        if pd.isna(num_clients):
            m_c = re.search(r'[-_]C(\d+)', fname)
            num_clients = int(m_c.group(1)) if m_c else None
        else:
            num_clients = int(num_clients)
            
        seed = row0.get("seed", None)
        if pd.isna(seed):
            m_s = re.search(r'[-_]S(?:EED)?[-_]?(\d+)', fname, re.I)
            seed = int(m_s.group(1)) if m_s else None
        else:
            seed = int(seed)
            
        alpha = row0.get("alpha", row0.get("dirichlet_alpha", None))
        if pd.isna(alpha):
            m_a = re.search(r'[-_]A(?:LPHA)?[-_]?([0-9.]+)', fname, re.I)
            alpha = float(m_a.group(1)) if m_a else 1000.0
        else:
            alpha = float(alpha)
            
        local_epochs = row0.get("local_epochs", None)
        if pd.isna(local_epochs):
            m_e = re.search(r'[-_]FEPOCH[-_]?(\d+)', fname, re.I)
            if not m_e:
                m_e = re.search(r'[-_]E(\d+)(?:[-_]|$)', fname, re.I)
            local_epochs = int(m_e.group(1)) if m_e else None
        else:
            local_epochs = int(local_epochs)
            
        # Is it pure FDA?
        is_pure = "pure" in fname.lower() or "pure" in method.lower()
        # Is it lr=0.1 or 0.01?
        lr = row0.get("client_opt_lr", row0.get("lr", None))
        if pd.isna(lr):
            if "LR1e1" in fname or "LR0.1" in fname or "lr0.1" in fname: lr = 0.1
            elif "LR1e2" in fname or "LR0.01" in fname or "lr0.01" in fname: lr = 0.01
            else: lr = 0.1
        else:
            lr = float(lr)
            
        records.append({
            "file": fname,
            "dataset": ds,
            "method": method,
            "local_epochs": local_epochs,
            "clients": num_clients,
            "alpha": alpha,
            "lr": lr,
            "seed": seed,
            "rounds": line_count,
            "is_pure": is_pure
        })
    except Exception as e:
        print(f"Error reading {fname}: {e}")

df = pd.DataFrame(records)
print(f"Total parsed thesis runs: {len(df)}")
df.to_csv("scratch/parsed_thesis_runs.csv", index=False)
