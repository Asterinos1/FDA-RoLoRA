import pandas as pd
import sys
sys.path.append('scratch')
from check_thesis_runtimes import records

df_recs = pd.DataFrame(records)
target = df_recs[df_recs['ds'].isin(['qqp', 'qnli', 'mnli', 'sst2'])]
grouped = target.groupby(['ds', 'alpha', 'c', 'method']).agg(
    n=('hours', 'count'),
    mean_hours=('hours', 'mean'),
    median_hours=('hours', 'median'),
    min_hours=('hours', 'min'),
    max_hours=('hours', 'max'),
    mean_rounds=('rounds', 'mean'),
    sec_per_round=('sec_per_round', 'mean')
).reset_index()

print("=" * 105)
print(f"{'Dataset':<7} | {'Type':<8} | {'Clients':<7} | {'Method':<10} | {'Mean Time':<10} | {'Rounds':<8} | {'Min/Round':<10} | {'Sample Count'}")
print("-" * 105)

for _, r in grouped.iterrows():
    m_type = "Non-IID" if r['alpha'] < 10 else "IID"
    min_per_round = r['sec_per_round'] / 60.0
    print(f"{r['ds'].upper():<7} | {m_type:<8} | C={r['c']:<5d} | {r['method']:<10} | {r['mean_hours']:4.1f} hours | {r['mean_rounds']:4.1f} rds | {min_per_round:4.1f} min/rd | n={r['n']}")
print("=" * 105)
