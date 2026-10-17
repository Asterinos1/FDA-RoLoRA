import sys, os
sys.path.insert(0, os.path.abspath("."))
import glob
import pandas as pd
import thesis_master_experiments as master
import thesis_experiments as base

configs = master.build_master_configs()
print(f"Total configs: {len(configs)}")

for idx, c in enumerate(configs, 1):
    pat = f"results/round_metrics/{c['extras']}_*.csv"
    matches = glob.glob(pat)
    master_done = master.is_master_completed(c)
    base_done = base.is_completed(c)
    if matches or master_done or base_done:
        print(f"#{idx:2d} | master={master_done} | base={base_done} | matches={len(matches)} | {c['extras'][:60]}")
