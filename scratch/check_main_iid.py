import pandas as pd

df = pd.read_csv("scratch/parsed_thesis_runs.csv")
valid_df = df[df['rounds'] >= 5].copy()

import sys
sys.path.append("scratch")
from analyze_seeds import clean_method
valid_df['clean_method'] = valid_df.apply(clean_method, axis=1)

# Let's filter to IID (alpha=1000) and lr=0.1 on the 4 main datasets
main_iid = valid_df[(valid_df['alpha'] == 1000.0) & (valid_df['lr'] == 0.1) & (valid_df['dataset'].isin(['mnli', 'qnli', 'qqp', 'sst2']))]

pivot = main_iid.groupby(['dataset', 'clients', 'clean_method'])['seed'].unique().reset_index()
pivot['seed_count'] = pivot['seed'].apply(len)
pivot['seeds'] = pivot['seed'].apply(lambda s: sorted([int(x) for x in s if pd.notna(x)]))

print("=== MAIN IID (lr=0.1, alpha=1000) BASELINES & FDA RUNS ===")
for _, r in pivot.iterrows():
    print(f"{r['dataset']:5s} | C={r['clients']:2d} | {r['clean_method']:30s} | count={r['seed_count']} | seeds={r['seeds']}")

