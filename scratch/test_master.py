import os, sys
sys.path.insert(0, os.path.abspath("."))
import run_thesis_suite as master

configs = master.build_master_configs()
completed = [c for c in configs if master.is_completed(c)]
pending = [c for c in configs if not master.is_completed(c)]
print(f"Total configs: {len(configs)}")
print(f"Master completed: {len(completed)}")
print(f"Pending queue length: {len(pending)}")
print("First 8 pending in queue:")
for idx, c in enumerate(pending[:8], 1):
    print(f"  #{idx}: {c['extras'][:75]}")
