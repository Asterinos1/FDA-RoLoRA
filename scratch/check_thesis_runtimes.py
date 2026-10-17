import glob
import os
import pandas as pd

files = sorted(glob.glob('results/round_metrics/*.csv'))
records = []
for f in files:
    fname = os.path.basename(f)
    if not fname.startswith("THESIS-"):
        continue
    try:
        df = pd.read_csv(f)
        if len(df) < 5:
            continue
        total_time_s = df['time_round_total'].sum() if 'time_round_total' in df.columns else 0
        rounds = len(df)
        last_acc = df['accuracy'].iloc[-1] if 'accuracy' in df.columns else 0
        best_acc = df['accuracy'].max() if 'accuracy' in df.columns else 0
        
        # ds
        ds = 'unknown'
        for d in ['mnli', 'qnli', 'qqp', 'sst2', 'mrpc', 'rte']:
            if f"-{d}-" in fname.lower() or f"_{d}_" in fname.lower() or f"-ds-{d}-" in fname.lower():
                ds = d; break
        
        c = 20 if '-C20-' in fname or '_C20_' in fname else (50 if '-C50-' in fname or '_C50_' in fname else 0)
        alpha = 0.5 if 'A0.5' in fname else 1000.0
        is_pure = 'pure' in fname.lower()
        method = 'RoLoRA' if 'rolora' in fname.lower() and not is_pure else ('LoRA' if 'lora' in fname.lower() and not is_pure else ('Pure FDA' if is_pure else 'FDA Old'))
        
        records.append({
            'file': fname,
            'ds': ds,
            'c': c,
            'alpha': alpha,
            'method': method,
            'rounds': rounds,
            'hours': total_time_s / 3600.0,
            'sec_per_round': total_time_s / rounds if rounds > 0 else 0,
            'best_acc': best_acc,
            'last_acc': last_acc
        })
    except Exception as e:
        pass

df = pd.DataFrame(records)
print(f"Total valid THESIS runs: {len(df)}")
# Group by (ds, alpha, c, method)
grouped = df.groupby(['ds', 'alpha', 'c', 'method']).agg(
    n=('hours', 'count'),
    mean_hours=('hours', 'mean'),
    min_hours=('hours', 'min'),
    max_hours=('hours', 'max'),
    mean_rounds=('rounds', 'mean'),
    sec_per_round=('sec_per_round', 'mean')
).reset_index()

print("\n=== EMPIRICAL RUNTIME STATS PER TASK CONFIGURATION ===")
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
print(grouped.to_string())
