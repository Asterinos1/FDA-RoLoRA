import glob
import os
import pandas as pd

files = sorted(glob.glob('results/round_metrics/*.csv'))
pure_runs = []
for f in files:
    fname = os.path.basename(f)
    if 'pure' in fname.lower() or 'PURE' in fname:
        df = pd.read_csv(f)
        if len(df) < 5:
            continue
        total_time_s = df['time_round_total'].sum() if 'time_round_total' in df.columns else 0
        rounds = len(df)
        last_acc = df['accuracy'].iloc[-1] if 'accuracy' in df.columns else 0
        
        # Determine dataset, c, alpha, k
        ds = 'unknown'
        for d in ['mnli', 'qnli', 'qqp', 'sst2']:
            if d in fname.lower():
                ds = d; break
        c = 20 if 'C20' in fname else (50 if 'C50' in fname else 0)
        alpha = 0.5 if 'A0.5' in fname else 1000.0
        k = 1.0 if 'CRATIO1' in fname or 'CR1' in fname else 0.5
        
        pure_runs.append({
            'file': fname,
            'ds': ds,
            'c': c,
            'alpha': alpha,
            'k': k,
            'rounds': rounds,
            'hours': total_time_s / 3600.0,
            'acc': last_acc
        })

df_pure = pd.DataFrame(pure_runs)
print(f"Total completed Pure FDA runs: {len(df_pure)}")
for _, r in df_pure.iterrows():
    print(f"{r['ds']:4s} | C={r['c']:2d} | a={r['alpha']:<6.1f} | k={r['k']:<3.1f} | rds={r['rounds']:2d} | {r['hours']:5.2f}h | acc={r['acc']:.4f} | {r['file'][:50]}")
