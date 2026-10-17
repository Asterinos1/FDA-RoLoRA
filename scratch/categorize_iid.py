import pandas as pd

df = pd.read_csv("scratch/parsed_thesis_runs.csv")
valid_df = df[df['rounds'] >= 5].copy()

import sys
sys.path.append("scratch")
from analyze_seeds import clean_method
valid_df['clean_method'] = valid_df.apply(clean_method, axis=1)

iid = valid_df[(valid_df['alpha'] == 1000.0) & (valid_df['lr'] == 0.1)]
grouped = iid.groupby(['dataset', 'clients', 'clean_method'])['seed'].unique().reset_index()
grouped['seed_count'] = grouped['seed'].apply(len)
grouped['seeds'] = grouped['seed'].apply(lambda s: sorted([int(x) for x in s if pd.notna(x)]))

single_seed = grouped[grouped['seed_count'] == 1]
print("=== IID RUNS WITH ONLY 1 SEED ===")
for _, r in single_seed.iterrows():
    print(f"{r['dataset']:5s} | C={r['clients']:2d} | {r['clean_method']:30s} | seeds={r['seeds']}")

print("\n=== IID RUNS WITH 2 SEEDS ===")
two_seeds = grouped[grouped['seed_count'] == 2]
for _, r in two_seeds.iterrows():
    print(f"{r['dataset']:5s} | C={r['clients']:2d} | {r['clean_method']:30s} | seeds={r['seeds']}")

print("\n=== IID RUNS WITH 3 SEEDS ===")
three_seeds = grouped[grouped['seed_count'] == 3]
for _, r in three_seeds.iterrows():
    print(f"{r['dataset']:5s} | C={r['clients']:2d} | {r['clean_method']:30s} | seeds={r['seeds']}")
