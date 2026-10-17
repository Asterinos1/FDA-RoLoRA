import glob
import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_thesis_66_analysis import parse_filename

def main():
    files = glob.glob('results/toy_experiments/*.csv')
    dfs = []
    incomplete = []
    
    for f in files:
        arch, scenario, seed, run_hash, method, fname = parse_filename(f)
        if method != 'EXCLUDE':
            df = pd.read_csv(f)
            if len(df) != 100:
                incomplete.append((fname, len(df)))
            df['arch'] = arch
            df['scenario'] = scenario
            df['method_norm'] = method
            df['seed'] = seed
            dfs.append(df)

    print(f"Total valid runs identified: {len(dfs)}")
    if incomplete:
        print(f"Incomplete runs: {incomplete}")
    else:
        print("All 66 runs completed all 100 rounds!")

    all_df = pd.concat(dfs, ignore_index=True)

    for (arch, sc), g in all_df.groupby(['arch', 'scenario']):
        print(f"\n=== {arch} - {sc} Fixed-Round Accuracies (R25, R50, R75, R100) ===")
        for m, mg in g.groupby('method_norm'):
            acc_str = []
            for r in [25, 50, 75, 100]:
                sub = mg[mg['round'] == r]['accuracy']
                acc_str.append(f"R{r}: {sub.mean()*100:.2f} +/- {sub.std()*100:.2f}%")
            print(f"  {m:18s} | " + " | ".join(acc_str))

if __name__ == '__main__':
    main()
