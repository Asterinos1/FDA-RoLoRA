from pathlib import Path
import pandas as pd
import glob

metrics_dir = Path('results/round_metrics')
patterns = [
    '*qnli*C50*fda-pure*CRATIO0.5*',
    '*mnli*C20*fda-pure*CRATIO1*',
    '*mnli*C20*rolora*',
    '*mnli*C20*fda-pure*CRATIO0.5*',
    '*mnli*C50*rolora*'
]

for pat in patterns:
    files = list(metrics_dir.glob(pat + '.csv'))
    print(f'Pattern: {pat} -> {len(files)} files')
    for f in files[:2]:
        try:
            df = pd.read_csv(f)
            if 'time_round_total' in df.columns:
                avg_time = df['time_round_total'].mean()
                median_time = df['time_round_total'].median()
                last_time = df['time_round_total'].iloc[-1]
                print(f'   {f.name[:50]}: avg={avg_time/60:.1f}m, median={median_time/60:.1f}m, last={last_time/60:.1f}m, total_rounds={len(df)}')
            else:
                time_cols = [c for c in df.columns if "time" in c]
                print(f'   {f.name[:50]}: time_cols={time_cols}')
        except Exception as e:
            print(f'   Error reading {f.name}: {e}')
