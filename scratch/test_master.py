import os, sys
os.chdir("master_experiments")
sys.path.insert(0, os.path.abspath(".."))
import thesis_master_experiments as master
master.configure_base_runner()
configs = master.build_master_configs()
completed = [c for c in configs if master.is_master_completed(c)]
pending = [c for c in configs if not master.is_master_completed(c)]
print(f"Total configs: {len(configs)}")
print(f"Master completed: {len(completed)}")
print(f"Pending queue length: {len(pending)}")
print("First 8 pending in queue:")
for idx, c in enumerate(pending[:8], 1):
    print(f"  #{idx}: {c['extras'][:75]}")
