import json
import subprocess
import sys
from pathlib import Path

# Build config for 1 round smoke test
config = {
    "scenario": "10-Clients-1-Class",
    "arch": "NONLIN",
    "linear": False,
    "num_clients": 10,
    "controller_ratio": 0.5,
    "seed": 42,
    "run_id": "TEST_SMOKE_EMAX10",
    "control_id": "testsmoke1",
    "total_rounds": 1,
    "e_min": 1,
    "e_max": 10,
    "batch_size": 64,
    "lora_r": 16
}

cmd = [
    sys.executable, "-u", "toy/thesis_toy_experiments.py",
    "--worker",
    "--config-json", json.dumps(config),
    "--device", "cpu"
]

print("Launching 1-round smoke test on CPU...")
proc = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:\n", proc.stdout)
if proc.stderr:
    print("STDERR:\n", proc.stderr)
print("Return code:", proc.returncode)
